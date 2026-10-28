"""Evidence-only decision gates, scaling estimates and reproducible final artifacts."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tarfile

REPORT_TOOLS = Path(__file__).resolve().parents[1]/".tools"
if (REPORT_TOOLS/"report-deps").is_dir():
    sys.path.insert(0, str(REPORT_TOOLS/"report-deps"))

import numpy as np

from analyze_teacher import analyze


def fit_scaling(t8, t16):
    ratio = t16/t8
    # Throughput(N)=a*N/(1+b*(N-1)); two points do not validate the model.
    denominator = 15*ratio-14
    b = (2-ratio)/denominator if denominator > 0 else None
    if b is None or b < 0:
        return dict(valid=False, reason="Two-point estimate has no nonnegative saturation coefficient",
                    ratio_16_8=ratio, efficiency_16=ratio/2)
    a = t8*(1+7*b)/8
    return dict(valid=True, a=a, b=b, ratio_16_8=ratio, efficiency_16=ratio/2,
        predicted_throughput={str(n): a*n/(1+b*(n-1)) for n in (8, 16, 32, 64)},
        caveat="Two measured points; 32/64 values are conditional extrapolation, not measurements")


def gate_row(name, value, threshold, passed):
    return dict(name=name, value=value, requirement=threshold, passed=bool(passed) if passed is not None else None)


def decision(results, cache=None):
    by_suite = {}
    for result in results:
        by_suite.setdefault(result["suite"], {})[result["environment"]["case"]] = result
    eight, sixteen = by_suite.get("diagnose8", {}), by_suite.get("diagnose16", {})
    if not all(key in eight and key in sixteen for key in ("profile_synthetic", "profile_cached")):
        return dict(ready=False, gate32=[], gate64=[], missing="8/16 synthetic and cached profiles")
    if not all(side[key]["environment"].get(flag) is True
               for side in (eight, sixteen) for key in ("profile_synthetic", "profile_cached")
               for flag in ("policy_training", "language_training", "gradient_checkpointing_active")):
        return dict(ready=False, gate32=[], gate64=[],
                    missing="Profiles must prove training mode and active language activation checkpointing")
    system8, system16 = eight["profile_cached"]["system"], sixteen["profile_cached"]["system"]
    efficiency = system16["throughput_samples_s"]/system8["throughput_samples_s"]/2
    model = fit_scaling(system8["throughput_samples_s"], system16["throughput_samples_s"])
    cached_ratio = min(side["profile_cached"]["system"]["throughput_samples_s"]/
                       side["profile_synthetic"]["system"]["throughput_samples_s"] for side in (eight, sixteen))
    wait = max(system8["data_wait_fraction"], system16["data_wait_fraction"])
    nccl_values = [s.get("profile", {}).get("nccl_wall_fraction") for s in (system8, system16)]
    nccl = max(nccl_values) if all(v is not None for v in nccl_values) else None
    lora = eight.get("lora_head")
    convergence = (min(1-lora["final"][key]/max(lora["initial"][key], 1e-9) for key in ("ADE_m", "FDE_m"))
                   if lora else None)
    without = eight.get("lora_no_rationale")
    rationale_delta = lora["final"]["ADE_m"]/without["final"]["ADE_m"]-1 if lora and without else None
    teacher = (cache or {}).get("teacher_total_s")
    diagnostic_training = [r for r in results if r["suite"] in ("diagnose8", "diagnose16") and
                           r["environment"]["mode"] == "train"]
    train_s = sum(r["system"]["train_step_total_s"] for r in diagnostic_training)
    eval_s = sum(r["system"]["eval_total_s"] for r in diagnostic_training)
    overhead = (teacher+eval_s)/(teacher+eval_s+train_s) if teacher is not None and train_s > 0 else None
    gates32 = [
        gate_row("8→16 cached scaling efficiency", efficiency, "≥0.75", efficiency >= .75),
        gate_row("cached / synthetic throughput", cached_ratio, "≥0.80", cached_ratio >= .80),
        gate_row("data wait", wait, "<0.15", wait < .15),
        gate_row("rank0 NCCL wall fraction", nccl, "<0.20", nccl < .20 if nccl is not None else None),
        gate_row("ADE and FDE reduction from initialization", convergence, "both ≥10%",
                 convergence >= .10 if convergence is not None else None),
        gate_row("rationale ADE degradation", rationale_delta, "≤5%",
                 rationale_delta <= .05 if rationale_delta is not None else None),
        gate_row("teacher + eval / diagnostic sequential work", overhead, "<0.30",
                 overhead < .30 if overhead is not None else None),
    ]
    ready = all(g["passed"] is not None for g in gates32)
    result = dict(ready=ready, gate32=gates32, allow32=ready and all(g["passed"] for g in gates32),
                  scaling_model=model, gate64=[], allow64=False,
                  teacher_eval_fraction=overhead,
                  teacher_eval_scope="offline teacher and diagnose8/diagnose16 training/evaluation; final run excluded",
                  thresholds="Convergence ≥10% and rationale ≤5% fixed before results; other gates from plan.md")
    thirty_two = by_suite.get("scale32", {})
    if thirty_two:
        dc16 = sixteen["profile_cached"].get("dc")
        dc32 = next(iter(thirty_two.values())).get("dc")
        matched = dc16 is not None and dc16 == dc32
        result["scale32_comparison_matched_queue"] = matched
        if not matched:
            result["scale64_missing"] = (
                "32-GPU validation used a different cluster/queue; GPU-count effects are not isolated")
            return result
    if "profile_cached" in thirty_two and lora and "lora_head" in sixteen:
        s32 = thirty_two["profile_cached"]["system"]
        efficiency32 = s32["throughput_samples_s"]/system16["throughput_samples_s"]/2
        profile32 = s32.get("profile", {}).get("nccl_wall_fraction")
        large = sixteen.get("large_batch")
        normal = sixteen["lora_head"]
        # Compare interpolated trajectory validation at common samples seen.
        batch_delta = quality_delta(large, normal, "samples_seen") if large else None
        fixed = ((cache or {}).get("preparation_wall_s", teacher or 0) +
                 (cache or {}).get("download_total_s", 0) +
                 lora["system"]["eval_total_s"] + lora["system"]["explain_total_s"] +
                 lora["system"].get("checkpoint_s", 0) + lora["environment"]["setup_s"])
        work = 12600*system8["throughput_samples_s"]
        projected_train = work/model["predicted_throughput"]["64"] if model.get("valid") else None
        useful = projected_train/(projected_train+fixed) if projected_train is not None else None
        # Additional GPU-hours must demonstrably lower ADE at a common cost.
        quality_cost = quality_delta(normal, lora, "GPU_hours")
        gates64 = [
            gate_row("16→32 cached efficiency", efficiency32, "≥0.70", efficiency32 >= .70),
            gate_row("32 data wait", s32["data_wait_fraction"], "<0.20", s32["data_wait_fraction"] < .20),
            gate_row("32 NCCL wall fraction", profile32, "<0.25", profile32 < .25 if profile32 is not None else None),
            gate_row("large-batch ADE at common samples", batch_delta, "≤5% degradation",
                     batch_delta <= .05 if batch_delta is not None else None),
            gate_row("64 useful train ratio", useful, "≥0.55", useful >= .55 if useful is not None else None),
            gate_row("16 vs 8 ADE at common GPU-hours", quality_cost, "≥2% improvement",
                     quality_cost <= -.02 if quality_cost is not None else None),
        ]
        result.update(gate64=gates64, allow64=all(g["passed"] is True for g in gates64),
                      predicted_useful_train_ratio64=useful)
    return result


def quality_delta(candidate, baseline, axis, metric="ADE_m"):
    if not candidate or not baseline:
        return None
    a, b = candidate.get("curves", []), baseline.get("curves", [])
    if len(a) < 2 or len(b) < 2:
        return None
    common = min(max(r[axis] for r in a), max(r[axis] for r in b))
    if common <= 0:
        return None
    def interpolate(rows):
        rows = sorted(rows, key=lambda r: r[axis])
        return float(np.interp(common, [r[axis] for r in rows], [r[metric] for r in rows]))
    return interpolate(a)/interpolate(b)-1


def campaign_state(root):
    root = Path(root)
    state = root/"controller_state.json"
    if state.exists():
        return json.loads(state.read_text())
    metrics = root/"metrics.json"
    return json.loads(metrics.read_text()).get("controller", {}) if metrics.exists() else {}


def cache_assets(root):
    root = Path(root)
    return root/"assets" if (root/"assets/cache_manifest.json").exists() else root/"evidence/assets"


def load_results(root):
    root = Path(root)
    state = campaign_state(root)
    active = {j["sid"] for j in state.get("jobs", []) if not j.get("superseded")}
    paths = sorted(root.glob("jobs/*/live/*.result.json"))
    if not paths:
        paths = sorted(root.glob("evidence/*/*/result.json"))
    results = []
    for path in paths:
        result = json.loads(path.read_text())
        suite_file = path.parent.parent/"job.json"
        job = json.loads(suite_file.read_text())
        if job.get("superseded") or (state.get("jobs") and job["sid"] not in active):
            continue
        result["suite"] = job["suite"]
        result["job_sid"] = job["sid"]
        result["dc"] = job.get("dc")
        result["artifact_path"] = str(path.relative_to(root))
        results.append(result)
    return results


def fmt(value):
    if value is None:
        return "缺失"
    return f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)


def diagnosis_table(results, cache, gates):
    suites = {}
    for r in results:
        suites.setdefault(r["suite"], {})[r["environment"]["case"]] = r
    eight, sixteen = suites.get("diagnose8", {}), suites.get("diagnose16", {})
    if not all(k in side for side in (eight, sixteen) for k in ("profile_synthetic", "profile_cached")):
        return []
    s8, s16 = eight["profile_cached"]["system"], sixteen["profile_cached"]["system"]
    ratios = [side["profile_cached"]["system"]["throughput_samples_s"] /
              side["profile_synthetic"]["system"]["throughput_samples_s"] for side in (eight, sixteen)]
    efficiency = s16["throughput_samples_s"]/s8["throughput_samples_s"]/2
    rows = [
        ["synthetic samples/s", fmt(eight["profile_synthetic"]["system"]["throughput_samples_s"]),
         fmt(sixteen["profile_synthetic"]["system"]["throughput_samples_s"]),
         "训练核上限", "与 cached 同批量对照"],
        ["cached samples/s", fmt(s8["throughput_samples_s"]), fmt(s16["throughput_samples_s"]),
         f"8→16 efficiency={efficiency:.1%}",
         "可继续检查 32 卡 gates" if efficiency >= .75 else "保留较小卡数，先降低每步通信与调度开销"],
        ["cached / synthetic", f"{ratios[0]:.1%}", f"{ratios[1]:.1%}",
         "数据链路满足 gate" if min(ratios) >= .80 else "数据链路限制吞吐",
         "保留缓存" if min(ratios) >= .80 else "优化批量读取与 prefetch 后重测"],
        ["data wait", f'{s8["data_wait_fraction"]:.1%}', f'{s16["data_wait_fraction"]:.1%}',
         "满足 <15%" if max(s8["data_wait_fraction"], s16["data_wait_fraction"]) < .15 else "等待超出 gate",
         "按等待比例决定是否继续扩卡"],
        ["rank0 NCCL wall fraction", fmt(s8.get("profile", {}).get("nccl_wall_fraction")),
         fmt(s16.get("profile", {}).get("nccl_wall_fraction")),
         "通信与计算可能重叠", "超出 20% 时检查 trace、互联和 DDP bucket"],
        ["peak allocated GiB", fmt(s8["peak_allocated_GiB"]), fmt(s16["peak_allocated_GiB"]),
         "各卡取最大值", "区分 microbatch 与 global batch，保持 strong scaling 的优化条件"],
        ["teacher", f'{cache.get("teacher_count", 0)} examples / {fmt(cache.get("teacher_total_s"))}s',
         "复用同一离线缓存", f'teacher+eval sequential fraction={fmt(gates.get("teacher_eval_fraction"))}',
         "占比 ≥30% 时优先优化或并行 teacher/eval"],
    ]
    p8, p16 = eight.get("lora_head"), sixteen.get("lora_head")
    if p8 and p16:
        rows += [
            ["trajectory train loss", f'{p8["system"]["loss_trajectory_first10"]:.4f}→{p8["system"]["loss_trajectory_last10"]:.4f}',
             f'{p16["system"]["loss_trajectory_first10"]:.4f}→{p16["system"]["loss_trajectory_last10"]:.4f}',
             "训练目标趋势", "结合 held-out ADE 判断过拟合"],
            ["val ADE / FDE m", f'{p8["final"]["ADE_m"]:.3f}/{p8["final"]["FDE_m"]:.3f}',
             f'{p16["final"]["ADE_m"]:.3f}/{p16["final"]["FDE_m"]:.3f}',
             f'16 vs 8 common GPU-hours ADE delta={fmt(quality_delta(p16, p8, "GPU_hours"))}',
             "用 samples seen 和 GPU-hours 对齐，避免把成本换时间当作效率提升"],
            ["eval / explain s", f'{p8["system"]["eval_total_s"]:.1f}/{p8["system"]["explain_total_s"]:.1f}',
             f'{p16["system"]["eval_total_s"]:.1f}/{p16["system"]["explain_total_s"]:.1f}',
             "端到端固定开销", "训练加速后重新计算 useful train ratio"],
        ]
    return rows


def create_report(root, final=False):
    root = Path(root)
    results = load_results(root)
    assets = cache_assets(root)
    cache_path = assets/"cache_manifest.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    download_path = assets/"download_manifest.json"
    download = json.loads(download_path.read_text()) if download_path.exists() else {}
    cache["download_total_s"] = download.get("elapsed_s", 0)
    state = campaign_state(root)
    records_path = assets/"records.json"
    teacher_diagnostics = analyze(records_path) if records_path.exists() else None
    if teacher_diagnostics is not None:
        (root/"teacher_diagnostics.json").write_text(json.dumps(
            teacher_diagnostics, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    gates = decision(results, cache)
    (root/"metrics.json").write_text(json.dumps(dict(results=results, cache=cache, decisions=gates,
        controller=state, teacher_diagnostics=teacher_diagnostics), indent=2))
    (root/"decisions.json").write_text(json.dumps(gates, indent=2))
    rows = []
    for result in results:
        env, sys, quality = result["environment"], result["system"], result["final"]
        rows.append(dict(suite=result["suite"], case=env["case"], GPUs=env["world"],
            dc=result.get("dc"),
            global_batch=env["global_batch"], throughput=sys["throughput_samples_s"],
            data_wait=sys["data_wait_fraction"],
            NCCL_fraction=sys.get("profile", {}).get("nccl_wall_fraction"),
            peak_GiB=sys["peak_allocated_GiB"], ADE_m=quality["ADE_m"], FDE_m=quality["FDE_m"],
            miss_rate=quality["miss_rate_2m"], action_accuracy=quality["action_accuracy"],
            rationale_consistency=quality["rationale_action_consistency"],
            evidence_accuracy=quality["evidence_flag_accuracy"],
            cached_policy_latency_p50_ms=quality["cached_policy_latency_p50_ms"],
            cached_policy_latency_p95_ms=quality["cached_policy_latency_p95_ms"],
            samples_seen=sys["samples_seen"], train_s=sys["train_step_total_s"],
            eval_s=sys["eval_total_s"], explain_s=sys["explain_total_s"], job_sid=result["job_sid"]))
    if rows:
        with (root/"decision_table.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    conclusions = []
    if gates.get("ready"):
        failed = [g["name"] for g in gates["gate32"] if not g["passed"]]
        conclusions.append("32 卡 gate 全部通过。" if gates["allow32"] else "32 卡 gate 未全部通过："+"；".join(failed)+"。")
    else:
        conclusions.append("诊断结果尚未齐全，扩展结论暂不成立。")
    final_runs = [r for r in results if r["suite"] == "final"]
    if final_runs:
        best = min(final_runs, key=lambda r:r["final"]["ADE_m"])
        conclusions.append(f'最终验证模型 {best["environment"]["case"]}，{best["environment"]["world"]} H20：'
            f'ADE {best["final"]["ADE_m"]:.3f}m、FDE {best["final"]["FDE_m"]:.3f}m，'
            f'训练 {best["system"]["samples_seen"]:,} samples seen。')
        if (root/"final_checkpoint.pt").exists():
            conclusions.append("可复现产物：[训练权重](final_checkpoint.pt)、[依赖与输入位置](reproduce.json)。")
        if (best.get("checkpoint_reload_probe") or {}).get("verified"):
            conclusions.append("最终权重已重新加载到新 backbone，验证样本的轨迹、动作和解释输出一致。")
    text = ["# Driving-VLA 瓶颈诊断与扩展实验报告", "",
        "状态："+("最终报告" if final else "运行中快照")+"；生成时间 "+
        datetime.now(timezone.utc).isoformat(), "", *conclusions, "",
        "## 数据与模型", "",
        f'官方 nuScenes mini：train={cache.get("train", "待生成")}，val={cache.get("val", "待生成")}；'
        "按官方场景划分，3 秒/6 步 ego-frame 轨迹，历史状态和当前图像作为输入。",
        "Qwen2.5-VL-3B 固定版本 66285546d2b821cf421d4f5eb2576359d3770cd3；"
        "六摄像头 112px，缓存冻结视觉和文本 embeddings，保留原始 MRoPE。"
        "LoRA 仅语言层 q/v、rank=8；bf16 计算，fp32 adapters/heads，DDP。",
        "预测采用可审计的 constant-velocity residual；image-only 同时移除 state 与 residual。",
        "Synthetic/cached scaling profiles 与正式 LoRA 训练均显式使用 train 模式和 activation checkpointing；"
        "每项结果记录 runtime flags，缺少证据时不能进入扩卡 gate。"
        "早期单卡 sanity profile 未记录这些 flags，保留为链路调试数据，不参与 scaling 比较。",
        f'Teacher 为同一冻结 Qwen 基线的离线弱监督：valid={cache.get("teacher_valid", "待生成")}，'
        f'invalid={cache.get("teacher_invalid", "待生成")}，total={fmt(cache.get("teacher_total_s"))}s。'
        "解释训练包含 action/rationale-action/evidence 结构化 heads；自然语言 rationale 保存供审阅，未做自回归文本微调。", "",
        "Constant-velocity baseline: "+json.dumps(cache.get("constant_velocity_baseline"), ensure_ascii=False),
        "Teacher trajectory validation: "+json.dumps(cache.get("teacher_val_trajectory"), ensure_ascii=False), "",
    ]
    if teacher_diagnostics is not None:
        text += ["## Teacher 标签诊断", "",
            "格式解析通过只表示结构有效。以下动作一致率使用由 ego 轨迹确定的预定义动作标签；"
            "它不代表独立安全评分。Evidence 缺少人工真值；标签集中或输出重复会影响解释指标的含义。", "",
            "| Split | Parsed / samples | Teacher unique actions | Teacher majority | Action agreement | Teacher ADE / FDE m |",
            "|---|---:|---:|---:|---:|---:|"]
        for split, diagnostic in teacher_diagnostics["splits"].items():
            action = diagnostic["teacher_action"]
            trajectory = diagnostic["teacher_trajectory"]
            text.append(f'| {split} | {diagnostic["parsed_teachers"]}/{diagnostic["samples"]} | '
                f'{action["unique_classes"]} | {fmt(action["majority_fraction"])} | '
                f'{fmt(diagnostic["teacher_ground_truth_action_agreement"])} | '
                f'{fmt(trajectory["ADE_m"])}/{fmt(trajectory["FDE_m"])} |')
        text += ["", "[完整 teacher 分布、重复输出和轨迹统计](teacher_diagnostics.json)。"
            "Ground-truth majority accuracy 按各 split 的标签分布计算，仅用于描述类别不平衡。"
            "Rationale-action consistency 为策略两组 head 的内部一致率，evidence flag accuracy 为与弱 teacher 的一致率。",
            ""]
    text += ["## 实测决策表", "",
        "| Suite / case | GPU / batch / DC | samples/s | wait | NCCL | peak GiB | ADE m | FDE m | action acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        text.append(f'| {row["suite"]}/{row["case"]} | {row["GPUs"]}/{row["global_batch"]}/{row["dc"]} | '
            f'{fmt(row["throughput"])} | {fmt(row["data_wait"])} | {fmt(row["NCCL_fraction"])} | '
            f'{fmt(row["peak_GiB"])} | {fmt(row["ADE_m"])} | {fmt(row["FDE_m"])} | {fmt(row["action_accuracy"])} |')
    diagnosis = diagnosis_table(results, cache, gates)
    if diagnosis:
        with (root/"bottleneck_decisions.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["问题", "8 卡", "16 卡", "判断", "行动"])
            writer.writerows(diagnosis)
        text += ["", "## 8/16 卡瓶颈与行动", "",
                 "| 问题 | 8 卡 | 16 卡 | 判断 | 行动 |", "|---|---:|---:|---|---|"]
        text += ["| "+" | ".join(row)+" |" for row in diagnosis]
    text += ["", "## Scaling gates", "",
        "| Gate | 实测/外推 | 阈值 | 通过 |", "|---|---:|---|---|"]
    for group in ("gate32", "gate64"):
        for row in gates.get(group, []):
            text.append(f'| {group}: {row["name"]} | {fmt(row["value"])} | {row["requirement"]} | {row["passed"]} |')
    normal = next((r for r in results if r["suite"] == "diagnose16" and
                   r["environment"]["case"] == "lora_head"), None)
    large = next((r for r in results if r["suite"] == "diagnose16" and
                  r["environment"]["case"] == "large_batch"), None)
    if normal and large:
        text += ["", "## 16 卡 batch=128 相对 batch=64", "",
                 "在两条单 seed 验证曲线的共同范围端点线性插值。正值表示误差增加；"
                 "该比较不构成统计显著性检验。不同 batch 在共同更新数下具有不同 samples seen。", "",
                 "| 对齐轴 | 共同端点 | ADE 相对变化 | FDE 相对变化 |",
                 "|---|---:|---:|---:|"]
        for axis in ("wall_s", "samples_seen", "GPU_hours", "step"):
            common = min(max(c[axis] for c in r["curves"]) for r in (normal, large))
            text.append(f'| {axis} | {fmt(common)} | '
                        f'{quality_delta(large, normal, axis):+.2%} | '
                        f'{quality_delta(large, normal, axis, "FDE_m"):+.2%} |')
    model = gates.get("scaling_model", {})
    if model:
        text += ["", "吞吐模型 Throughput(N)=a·N/(1+b·(N−1))：", "",
                 "```json", json.dumps(model, indent=2), "```",
                 "两点拟合仅提供条件预测；32/64 卡未实测时不能称为验证。质量按 wall-clock、samples seen、GPU-hours 对齐。"]
        if not model.get("valid") and model.get("ratio_16_8", 0) > 2:
            text += ["",
                f'本次 cached 8→16 吞吐比为 {model["ratio_16_8"]:.3f}，'
                f'效率为 {model["efficiency_16"]:.1%}。'
                "这两个点无法拟合非负的饱和开销系数，因此没有生成该模型的 32/64 卡吞吐或端到端时间预测。"
                "该结果仅描述当前固定 global batch、各自 accumulation 和实际节点条件；"
                "没有重复测量及更大卡数验证，不能假定超线性收益会持续。"]
    if gates.get("scale32_comparison_matched_queue") is False:
        text += ["",
                 "32 卡实际运行于 LF，8/16 卡基线运行于 HL。本次为跨集群验证；"
                 "吞吐差异同时包含卡数、microbatch 和集群条件变化，不能据此确认单独的卡数扩展收益。"
                 "64 卡后续 gate 保持未成立，继续实验由用户决定。"]
    text += ["", "## 质量、解释性与成本", ""]
    for result in results:
        if result["environment"]["mode"] != "train":
            continue
        q, s = result["final"], result["system"]
        text += [f'- {result["suite"]}/{result["environment"]["case"]}：'
            f'ADE/FDE {q["ADE_m"]:.3f}/{q["FDE_m"]:.3f}m；miss@2m {q["miss_rate_2m"]:.3f}；'
            f'rationale consistency {fmt(q["rationale_action_consistency"])}；evidence flags {fmt(q["evidence_flag_accuracy"])}；'
            f'collision proxy {q["collision_proxy"]:.3f}；'
            f'train/eval/explain/checkpoint {s["train_step_total_s"]:.1f}/{s["eval_total_s"]:.1f}/'
            f'{s["explain_total_s"]:.1f}/{s.get("checkpoint_s", 0):.1f}s。',
            f'  Cached batch latency/sample p50/p95={q["cached_policy_latency_p50_ms"]:.3f}/'
            f'{q["cached_policy_latency_p95_ms"]:.3f}ms；p50 对应 '
            f'{1000/q["cached_policy_latency_p50_ms"]:.3f} actions/s，排除 vision/teacher。',
            f'  Selected checkpoint step={result.get("selected_checkpoint_step", s["steps"])} '
            f'/ completed step={s["steps"]}; raw final ADE={fmt(result.get("raw_final", q).get("ADE_m"))}m。',
            "  Trajectory/action/rationale training loss first10→last10: "+
            " / ".join(f'{fmt(s.get(key+"_first10"))}→{fmt(s.get(key+"_last10"))}'
                       for key in ("loss_trajectory", "loss_action", "loss_rationale")),
            "  Counterfactual: "+json.dumps(result.get("explanation"), ensure_ascii=False)]
    if gates.get("scaling_model", {}).get("valid"):
        t8 = gates["scaling_model"]["predicted_throughput"]["8"]
        reference = next((r for r in results if r["suite"] == "diagnose8" and
                          r["environment"]["case"] == "lora_head"), None)
        if reference:
            download_path = assets/"download_manifest.json"
            download = json.loads(download_path.read_text()) if download_path.exists() else {}
            fixed = dict(teacher_s=cache.get("teacher_total_s", 0),
                vision_cache_s=cache.get("vision_total_s", 0), download_s=download.get("elapsed_s", 0),
                cache_other_s=cache.get("cache_other_s", max(0., cache.get("preparation_wall_s", 0) -
                    cache.get("teacher_total_s", 0) - cache.get("vision_total_s", 0))),
                eval_s=reference["system"]["eval_total_s"],
                explain_s=reference["system"]["explain_total_s"],
                checkpoint_s=reference["system"].get("checkpoint_s", 0),
                setup_s=reference["environment"]["setup_s"])
            work = t8*12600  # One fixed workload: 3.5h of 8-GPU training.
            forecast = {}
            text += ["", "固定工作量：8 卡 3.5 小时的 samples 数。端到端预测保留 teacher、vision cache、"
                     "download、eval、explain、checkpoint、setup 实测开销；排队未计，属于时间下界。", "",
                     "| GPU | Train h | Total h lower bound | Useful train ratio | Training GPU-h |",
                     "|---:|---:|---:|---:|---:|"]
            for n in (8, 16, 32, 64):
                train_time = work/gates["scaling_model"]["predicted_throughput"][str(n)]
                total_time = train_time+sum(fixed.values())
                forecast[str(n)] = dict(train_s=train_time, total_lower_bound_s=total_time,
                                        useful_train_ratio=train_time/total_time, train_GPU_hours=train_time*n/3600)
                text.append(f"| {n} | {train_time/3600:.3f} | {total_time/3600:.3f} | "
                            f"{train_time/total_time:.3f} | {train_time*n/3600:.3f} |")
            (root/"end_to_end_forecast.json").write_text(json.dumps(
                dict(work_samples=work, fixed_measured_costs_s=fixed, forecast=forecast,
                     assumption="constant overhead; no queue ETA; 32/64 extrapolation must be validated"), indent=2))
    if rows:
        plots(root, results, gates)
        text += ["", "![质量—时间/样本/GPU小时](quality_curves.png)", "",
                 "![吞吐与条件外推](scaling.png)"]
        if (root/"qualitative.png").exists():
            text += ["", "![真实图像、teacher evidence 与预测轨迹](qualitative.png)"]
    text += ["", "## 执行证据与适用范围", "",
        "GPU-hours 需区分有效训练曲线成本与任务总占用成本；总占用含排队后启动、安装、缓存等待、评估。"
        "各 job.json 保留运行状态/时间，原始 CLI 查询与 pod 日志归档在 jobs/ 下。",
        "固定 global batch=64 的 8/16 卡用于 strong scaling；16 卡 batch=128 单独探查优化动态。"
        "NCCL fraction 为 rank0 CUDA trace 的 NCCL kernel 区间并集/trace wall time，"
        "有通信与计算重叠，不能把 forward/backward/NCCL 相加当作互斥百分比。",
        "Collision 是未来标注 occupied-circle 代理，非闭环安全评分；没有 map expansion 数据，lane violation 不报告。"
        "单摄像头置零是 feature intervention，未构造物理可实现的反事实。"
        "轨迹和动作指标覆盖完整验证集；摄像头干预仅覆盖按缓存顺序选取的前 48 个验证样本，"
        "各样本执行六路摄像头干预，不能据此声称完整验证集反事实评估。",
        "mini 仅 10 场景、val 2 场景；单 seed、重复采样、小分辨率，结果用于系统/算法诊断。"
        "最终模型按验证 ADE 选择 checkpoint，质量曲线保留原始训练状态；没有额外独立 test set。"
        "未证明真实闭环驾驶能力或完整数据集泛化，teacher agreement 不是独立解释正确性。",
        "", "## 任务记录", ""]
    for job in state.get("jobs", []):
        end = job.get("observed_terminal_epoch_s", job.get("superseded_epoch_s", state.get("updated_epoch_s", 0)))
        occupancy = ((end-
                      job["first_running_epoch_s"])*job["world"]/3600 if job.get("first_running_epoch_s") else None)
        text.append(f'- [{job["suite"]} / {job["world"]} H20](https://ml.bytedance.net/development/instance/jobs/{job["sid"]})'
                    f'：{job.get("status", "unknown")}，占用 GPU-h≈{fmt(occupancy)}，{job.get("name", "")}')
    text += ["", "Controller decisions:", "", "```json",
             json.dumps(state.get("events", []), ensure_ascii=False, indent=2), "```"]
    (root/"report.md").write_text("\n".join(text)+"\n")
    return gates


def plots(root, results, gates):
    if "MPLCONFIGDIR" not in os.environ:
        cache = REPORT_TOOLS/"matplotlib-cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    labels = [("wall_s", "Training window (h)", 3600), ("samples_seen", "Samples seen", 1),
              ("GPU_hours", "GPU-hours in training window", 1)]
    manifest = cache_assets(root)/"cache_manifest.json"
    baseline = (json.loads(manifest.read_text()).get("constant_velocity_baseline", {})
                .get("val", {}).get("ADE_m")) if manifest.exists() else None
    if baseline is not None:
        for ax in axes:
            ax.axhline(baseline, color="black", linestyle=":", linewidth=1,
                       label=f"Constant velocity ({baseline:.3f}m)")
    for result in results:
        if result["environment"]["mode"] != "train":
            continue
        label = f'{result["suite"]}/{result["environment"]["case"]}'
        for ax, (axis, xlabel, unit) in zip(axes, labels):
            curves = result["curves"]
            ax.plot([r[axis]/unit for r in curves], [r["ADE_m"] for r in curves], "o-", ms=3, label=label)
            ax.set(xlabel=xlabel, ylabel="Validation ADE (m)")
            ax.grid(alpha=.2)
    axes[-1].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(root/"quality_curves.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in ("synthetic", "cached"):
        profile = sorted([r for r in results if r["environment"]["case"] == "profile_"+mode and
                          r["environment"]["global_batch"] == 64 and r["suite"] != "sanity"],
                         key=lambda r:r["environment"]["world"])
        for dc in sorted({r.get("dc") or "unknown" for r in profile}):
            measured = [r for r in profile if (r.get("dc") or "unknown") == dc]
            ax.plot([r["environment"]["world"] for r in measured],
                    [r["system"]["throughput_samples_s"] for r in measured], "o-",
                    label=mode+f" ({dc}) measured")
    model = gates.get("scaling_model", {})
    if model.get("valid"):
        ax.plot([8, 16, 32, 64], [model["predicted_throughput"][str(n)] for n in (8, 16, 32, 64)],
                "--", label="two-point conditional extrapolation")
    ax.set(xlabel="H20 GPUs", ylabel="Samples / second", title="Same global batch 64")
    ax.grid(alpha=.2)
    if ax.lines:
        ax.legend()
    else:
        ax.text(.5, .5, "Awaiting matched 8/16-GPU profiles", transform=ax.transAxes,
                ha="center", va="center")
    fig.tight_layout()
    fig.savefig(root/"scaling.png", dpi=170)
    plt.close(fig)
    qualitative(root, results)


def qualitative(root, results):
    import matplotlib.pyplot as plt
    from PIL import Image
    archive = root/"assets/previews.tar.gz"
    if not archive.exists():
        return
    previews = root/"assets/previews"
    if not previews.exists():
        with tarfile.open(archive) as tar:
            for member in tar:
                target = (root/"assets"/member.name).resolve()
                assert target.is_relative_to((root/"assets").resolve()) and not member.issym() and not member.islnk()
                tar.extract(member, root/"assets")
    candidates = [r for r in results if r["suite"] == "final"] or [
        r for r in results if r["suite"] == "diagnose8" and r["environment"]["case"] == "lora_head"]
    if not candidates:
        return
    best = min(candidates, key=lambda r:r["final"]["ADE_m"])
    path = root/best["artifact_path"]
    path = path.with_name("predictions.json") if path.name == "result.json" else path.with_name(
        path.name.replace(".result.json", ".predictions.json"))
    if not path.exists():
        return
    rows = [r for r in json.loads(path.read_text()) if (previews/f'{r["token"]}.jpg').exists()]
    rows = sorted(rows, key=lambda r:r["ADE"])
    chosen = [rows[i] for i in sorted({0, len(rows)//2, len(rows)-1})] if rows else []
    if not chosen:
        return
    fig, axes = plt.subplots(len(chosen), 2, figsize=(15, 4*len(chosen)), squeeze=False,
                             gridspec_kw={"width_ratios": [3, 1]})
    for (camera_ax, path_ax), row in zip(axes, chosen):
        with Image.open(previews/f'{row["token"]}.jpg') as image:
            camera_ax.imshow(np.asarray(image))
        camera_ax.axis("off")
        camera_ax.set_title(row["scene"]+f' | ADE={row["ADE"]:.2f}m | action={row["action"]}, GT={row["target_action"]}', fontsize=10)
        cameras = ["front", "front-left", "front-right", "back", "back-left", "back-right"]
        teacher = row.get("teacher")
        for index, name in enumerate(cameras):
            camera_ax.text(index*112+56, 120, f'{name}\nhead evidence={row["evidence"][index]:.2f}'
                           + (f'\nteacher={int(teacher["evidence"][index])}' if teacher else ""),
                           ha="center", va="top", fontsize=8)
        truth, prediction = np.array(row["target"]), np.array(row["prediction"])
        path_ax.plot(truth[:, 1], truth[:, 0], "o-", label="GT")
        path_ax.plot(prediction[:, 1], prediction[:, 0], "o-", label="prediction")
        path_ax.scatter([0], [0], marker="^", color="black", label="ego")
        path_ax.set(xlabel="Ego left (m)", ylabel="Ego forward (m)")
        path_ax.grid(alpha=.2)
        path_ax.legend(fontsize=8)
        path_ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Held-out real camera inputs and 3s trajectories; evidence is weak-teacher agreement", fontsize=12)
    fig.tight_layout()
    fig.savefig(root/"qualitative.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    print(json.dumps(create_report(args.root, args.final), ensure_ascii=False, indent=2))
