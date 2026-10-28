"""Recompute one-H20 probes from archived steps and CUDA traces on CPU."""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile

PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "results"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".tools/matplotlib-cache"))
sys.path.insert(0, str(PROJECT / ".tools/report-deps"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from verify_results import kernel_evidence


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_summary(trace, trace_steps):
    events = trace["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    selected = defaultdict(lambda: dict(count=0, total_s=0.))
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in ("user_annotation", "cuda_runtime"):
            selected[e["name"]]["count"] += 1
            selected[e["name"]]["total_s"] += e["dur"] / 1e6
    assert selected["policy_forward"]["count"] == trace_steps
    assert selected["policy_backward"]["count"] == trace_steps
    span = (max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)) / 1e6
    result = kernel_evidence(trace)
    result.update(
        steps=trace_steps, kernel_span_s=span,
        kernel_active_fraction_of_span=result["gpu_kernel_union_s"] / span,
        kernels_per_step=result["kernel_count"] / trace_steps,
        cpu={k: selected[k] for k in ("policy_forward", "policy_backward", "cudaLaunchKernel",
                                      "cudaStreamSynchronize", "cudaDeviceSynchronize")},
        cuda_runtime_version=trace.get("cuda_runtime_version"),
        cuda_driver_version=trace.get("cuda_driver_version"),
    )
    return result


def main():
    job = read(ROOT / "single_gpu_submission.json")
    assert job["status"] in {"success", "done", "succeeded", "completed"}
    directory = ROOT / "jobs" / job["name"]
    archives = list(directory.glob("*.tar.gz"))
    assert len(archives) == 1
    archive = archives[0]
    artifacts, traces, hardware, exits = {}, {}, [], []
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.isfile():
                continue
            if member.name.endswith("exit_code.txt"):
                exits.append(tar.extractfile(member).read().strip().decode())
            elif member.name.endswith("gpus.csv"):
                hardware = list(csv.DictReader(io.StringIO(
                    tar.extractfile(member).read().decode()), skipinitialspace=True))
            elif member.name.startswith("results/single_gpu/"):
                name = Path(member.name).name
                if name.endswith(".trace.json"):
                    traces[name.removesuffix(".trace.json")] = trace_summary(
                        json.load(tar.extractfile(member)), 4)
                elif name.endswith(".json"):
                    artifacts[name] = json.load(tar.extractfile(member))
    assert exits == ["0"] and len(hardware) == 1
    result = artifacts["result.json"]
    assert result == read(directory / "live/single_gpu.result.json")
    env = result["environment"]
    assert env["world"] == 1 and env["distributed"] is False
    assert env["source_sha256"] == job["source_sha256"]
    old = read(ROOT / "jobs/driving-diagnose16-16h20-a1-20261028/live/lora_head.result.json")
    assert env["cache_manifest_sha256"] == old["environment"]["cache_manifest_sha256"]
    assert env["config"] == old["environment"]["config"]
    assert env["model_revision"] == "66285546d2b821cf421d4f5eb2576359d3770cd3"
    assert result["frozen_weight_sample_unchanged"]
    assert env["trainable_parameters"] == 2382640
    windows = artifacts["windows.json"]
    steps = artifacts["steps.json"]
    assert windows == result["windows"] and len(windows) == 18
    assert len(steps) == 432 and all(not r["instrumented"] for r in steps)
    assert len(traces) == 6 and all(t["nccl_kernel_count"] == 0 for t in traces.values())
    by_key = defaultdict(list)
    for w in windows:
        rows = [r for r in steps if r["key"] == w["key"] and r["repeat"] == w["repeat"]]
        assert [r["step"] for r in rows] == list(range(24))
        assert w["samples"] == w["microbatch"] * 24
        assert np.isclose(w["samples_per_s"], w["samples"] / w["window_wall_s"], rtol=1e-12)
        assert np.isclose(w["step_wall_s"]["p50"],
                          np.median([r["full_step_wall_s"] for r in rows]), rtol=1e-12)
        assert sum(r["full_step_wall_s"] for r in rows) <= w["window_wall_s"]
        for row in rows:
            assert row["full_step_wall_s"] >= row["local_compute_wall_s"] > 0
            assert np.isclose(row["full_step_wall_s"],
                              row["local_compute_wall_s"] + row["tail_host_wall_s"], rtol=1e-12)
            assert np.isclose(row["gpu_step_elapsed_s"],
                              sum(row[k] for k in ("gpu_forward_elapsed_s",
                                                  "gpu_backward_elapsed_s",
                                                  "gpu_optimizer_elapsed_s")), atol=5e-7)
            assert np.isfinite([row["loss"], row["gradient_norm"]]).all()
        by_key[w["key"]].append(w)
    summaries = {}
    for key, group in by_key.items():
        assert sorted(w["repeat"] for w in group) == [0, 1, 2]
        rows = [r for r in steps if r["key"] == key]
        speed = [w["samples_per_s"] for w in group]
        total_s = sum(w["window_wall_s"] for w in group)
        total_samples = sum(w["samples"] for w in group)
        summary = dict(
            microbatch=group[0]["microbatch"], checkpointing=group[0]["checkpointing"],
            samples_per_s=total_samples / total_s,
            window_samples_per_s=speed,
            window_samples_per_s_min=min(speed), window_samples_per_s_max=max(speed),
            window_speed_cv=float(np.std(speed, ddof=1) / np.mean(speed)),
            window_wall_s=total_s, samples=total_samples, timed_steps=len(rows),
            full_step_p50_ms=float(np.median([r["full_step_wall_s"] for r in rows]) * 1000),
            full_step_p95_ms=float(np.quantile([r["full_step_wall_s"] for r in rows], .95) * 1000),
            wall_ms_per_sample=total_s / total_samples * 1000,
            cpu_forward_backward_mean_ms=float(np.mean(
                [r["cpu_forward_s"] + r["cpu_backward_s"] for r in rows]) * 1000),
            local_compute_mean_ms=float(np.mean([r["local_compute_wall_s"] for r in rows]) * 1000),
            tail_host_mean_ms=float(np.mean([r["tail_host_wall_s"] for r in rows]) * 1000),
            gpu_step_elapsed_mean_ms=float(np.mean([r["gpu_step_elapsed_s"] for r in rows]) * 1000),
            peak_allocated_gib=max(w["peak_allocated_bytes"] for w in group) / 2**30,
            peak_reserved_gib=max(w["peak_reserved_bytes"] for w in group) / 2**30,
            trace=traces[key],
        )
        summaries[key] = summary
    sanity = artifacts["numerical_sanity.json"]
    assert sanity == result["numerical_sanity"]
    for micro in (2, 4, 8):
        v = sanity[f"micro{micro}_ckptoff"]
        assert v["gradient_relative_l2_vs_checkpoint_on"] < 1e-5
        assert v["trajectory_max_abs_m_vs_checkpoint_on"] < 1e-5
    comparisons = {}
    for ckpt in ("on", "off"):
        for m in (4, 8):
            base = summaries[f"micro2_ckpt{ckpt}"]
            value = summaries[f"micro{m}_ckpt{ckpt}"]
            comparisons[f"micro{m}_vs2_ckpt{ckpt}"] = dict(
                speed_ratio=value["samples_per_s"] / base["samples_per_s"],
                step_p50_ratio=value["full_step_p50_ms"] / base["full_step_p50_ms"],
                kernel_count_ratio=value["trace"]["kernel_count"] / base["trace"]["kernel_count"],
            )
    for m in (2, 4, 8):
        a, b = summaries[f"micro{m}_ckpton"], summaries[f"micro{m}_ckptoff"]
        comparisons[f"checkpoint_off_vs_on_micro{m}"] = dict(
            speed_ratio=b["samples_per_s"] / a["samples_per_s"],
            allocated_gib_delta=b["peak_allocated_gib"] - a["peak_allocated_gib"],
            kernel_count_ratio=b["trace"]["kernel_count"] / a["trace"]["kernel_count"])
    terminal_path = ROOT / "single_gpu_terminal.json"
    assert terminal_path.exists(), "Collect and verify terminal evidence before reporting"
    terminal = read(terminal_path)
    evidence = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        job=job, environment=env, hardware=hardware, archive=dict(
            path=str(archive.relative_to(ROOT)), bytes=archive.stat().st_size, sha256=sha(archive)),
        terminal=terminal, summaries=summaries, comparisons=comparisons,
        numerical_sanity=sanity, python_wall_s=result["python_wall_s"],
        uninstrumented_window_wall_s=sum(w["window_wall_s"] for w in windows),
        evidence_checks=dict(
            successful_one_gpu_archive=exits == ["0"] and len(hardware) == 1,
            live_archive_equal=True, matched_source_model_cache=True,
            frozen_sample_unchanged=True, all_432_steps_recomputed=True,
            six_traces_and_no_nccl=True, checkpoint_semantics_match=True,
            terminal_one_pod_exit_zero=bool(terminal and len(terminal["pods"]) == 1
                                           and terminal["pods"][0]["exit_status"] == 0),
            bound_satisfied=bool(terminal and terminal["creation_to_exit_s"] <= 1200),
        ),
        limitations=[
            "One host, eight resident train specimens, three short windows per configuration",
            "No DDP, cross-node tail collectives, one-hour thermal/load evolution or ADE/FDE measured",
            "GPU event elapsed and kernel interval union are not SM utilization",
            "Optimizer batch varies across microbatch sizes; cannot infer convergence at global batch 64",
        ])
    assert all(evidence["evidence_checks"].values()), "Evidence gates must pass before report delivery"
    (ROOT / "single_gpu_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False))
    with (ROOT / "single_gpu_windows.csv").open("w", newline="") as file:
        columns = ["key", "microbatch", "checkpointing", "repeat", "steps", "samples",
                   "window_wall_s", "samples_per_s", "peak_allocated_bytes", "peak_reserved_bytes"]
        writer = csv.DictWriter(file, columns)
        writer.writeheader()
        writer.writerows({k: row[k] for k in columns} for row in windows)
    plot(summaries)
    report(evidence)
    integrate(evidence)
    print(json.dumps(dict(summaries={k: {f: v[f] for f in (
        "samples_per_s", "full_step_p50_ms", "wall_ms_per_sample",
        "peak_allocated_gib", "window_speed_cv")} for k, v in summaries.items()},
                          comparisons=comparisons, checks=evidence["evidence_checks"]),
                     ensure_ascii=False, indent=2))


def plot(summaries):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = {"on": "#3265a8", "off": "#db8d36"}
    for mode in ("on", "off"):
        rows = [summaries[f"micro{m}_ckpt{mode}"] for m in (2, 4, 8)]
        x = np.arange(3) + (-.18 if mode == "on" else .18)
        for ax, field, label in (
            (axes[0, 0], "samples_per_s", "Throughput (samples/s)"),
            (axes[0, 1], "wall_ms_per_sample", "Wall time per sample (ms)"),
            (axes[1, 0], "peak_allocated_gib", "Peak allocated memory (GiB)"),
            (axes[1, 1], "full_step_p50_ms", "Full optimizer micro-step p50 (ms)"),
        ):
            ax.bar(x, [r[field] for r in rows], .35, label=f"Checkpoint {mode}",
                   color=colors[mode])
            if field == "samples_per_s":
                for xi, r in zip(x, rows):
                    ax.scatter([xi] * 3, r["window_samples_per_s"], s=18,
                               edgecolors="white", linewidths=.4, color="#283444", zorder=3)
            ax.set(xticks=np.arange(3), xticklabels=["2", "4", "8"],
                   xlabel="Microbatch / optimizer batch on one GPU", ylabel=label)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Same H20, real cached LoRA workload; 3 x 24 timed steps after 12 warmup steps")
    fig.tight_layout()
    fig.savefig(ROOT / "single_gpu_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def report(evidence):
    env, rows, comparison = evidence["environment"], evidence["summaries"], evidence["comparisons"]
    table = []
    for m in (2, 4, 8):
        for mode in ("on", "off"):
            r = rows[f"micro{m}_ckpt{mode}"]
            table.append(
                f"| {m} | {'开' if mode == 'on' else '关'} | {r['samples_per_s']:.2f} "
                f"({r['window_samples_per_s_min']:.2f}～{r['window_samples_per_s_max']:.2f}) "
                f"| {r['full_step_p50_ms']:.2f} | {r['wall_ms_per_sample']:.2f} "
                f"| {r['peak_allocated_gib']:.3f} |")
    lines = [
        "# Driving-VLA：单张 H20 的 microbatch 与 checkpointing 诊断",
        "", "2026-10-28｜同一主机、同一模型和真实缓存｜性能诊断",
        "", f"checkpointing 开启时，microbatch 4／8 相对 2 的吞吐为 "
        f"**{comparison['micro4_vs2_ckpton']['speed_ratio']:.2f}／"
        f"{comparison['micro8_vs2_ckpton']['speed_ratio']:.2f} 倍**。固定 microbatch，"
        f"关闭 checkpointing 后吞吐分别为开启时的 "
        f"**{comparison['checkpoint_off_vs_on_micro2']['speed_ratio']:.2f}／"
        f"{comparison['checkpoint_off_vs_on_micro4']['speed_ratio']:.2f}／"
        f"{comparison['checkpoint_off_vs_on_micro8']['speed_ratio']:.2f} 倍**。"
        "这是单卡执行路径收益；分布式扩展效率和训练质量仍需分别验证。[S1](#s1)[S2](#s2)",
        "", "## 1. 测量与成本",
        "", f"任务 `{evidence['job']['sid']}` 在 HL 的 `search_douyin_life_nlp` 使用 **1 张 H20**。"
        f"入口硬超时 1180 秒、kill grace 10 秒，独立监视器另从提交时限制 1200 秒。"
        f"Python 从解包开始的实测耗时为 **{evidence['python_wall_s']:.1f} 秒**；"
        f"模型／cache 解包 {env['extract_s']:.1f} 秒，模型建立 {env['model_setup_s']:.1f} 秒。"
        "依赖安装、平台启动和归档不包含在 Python 耗时中。[S1](#s1)[S3](#s3)",
    ]
    terminal = evidence["terminal"]
    if terminal:
        lines += [
            "", f"平台 pod 启动到退出 **{terminal['pod_wall_s']:.1f} 秒 "
            f"（{terminal['pod_wall_s'] / 60:.2f} 分钟，约 {terminal['pod_wall_s'] / 3600:.4f} GPU·h）**；"
            f"创建到退出 **{terminal['creation_to_exit_s']:.1f} 秒**，退出码 0。"
            "成本包含准备、warmup、数值检查、所有短窗口、trace 和归档；未新增多卡任务。[S3](#s3)",
        ]
    lines += [
        "", f"采用原有 Qwen2.5-VL-3B pinned revision、LoRA q/v rank 8、"
        f"2,382,640 个可训练参数、bf16 backbone／fp32 LoRA 和 head、SDPA。"
        f"运行时 Torch {env['torch']}、Transformers {env['transformers']}；"
        f"驱动 {evidence['hardware'][0]['driver_version']}。模型、原 `train.py` 和 cache manifest "
        "与 16／32 卡实验一致。[S1](#s1)[S2](#s2)",
        "", "本次单卡的 NVIDIA 驱动与原 16／32 卡节点不同。六组同机对照固定了驱动，"
        "可比较本次执行路径的相对收益；绝对延迟不能直接回填原 HL／LF 实验或据此"
        "消除原集群的硬件混杂。",
        "", f"均匀选取训练缓存中的 8 个真实样本，实际 embedding shape 为 `{env['embeddings_shape']}`，"
        f"attention 有效长度均为 {env['valid_lengths'][0]}。输入提前驻留 GPU，"
        "排除 dataloader 和 HDFS IO。每组在相同可训练参数初始化下 warmup 12 step，"
        "计时 24 step，重复 3 次；第二轮逆序，减轻单调主机负载与温度变化的影响。"
        "每个 step 是当前 microbatch 的一次 optimizer 更新，未按 global batch 64 累积。[S1](#s1)",
        "", "## 2. 六组性能实测",
        "", "| Microbatch | Checkpointing | 吞吐及三窗口范围，samples/s | Full step p50，ms | 每样本 wall，ms | 峰值 allocated，GiB |",
        "|---:|---|---:|---:|---:|---:|",
        *table,
        "", "吞吐采用三窗口总样本数／总 wall 时间；括号为独立窗口吞吐最小～最大值，"
        "不作为置信区间。Full step 覆盖 zero_grad、forward、backward、clip、optimizer、"
        "CUDA completion 和主机指标回传；原始记录另保留 CPU forward/backward、CUDA event "
        "elapsed 和尾部 host 时间。同一 rank 的字段可以核对边界，CPU 与 CUDA elapsed "
        "有重叠，不能加成训练耗时分解。[S1](#s1)[S2](#s2)",
        "", "![同一 H20 上六组实测：吞吐、每样本耗时、显存和整步延迟](single_gpu_comparison.png)",
        "", "## 3. 执行路径与数值核验",
        "", "Profiler 与吞吐计时分开录制，每组单独 4 个 step。下面的 kernel 并集／跨度"
        "仅描述此短 trace 内的活动时间，不是 SM 或 Tensor Core 利用率。[S2](#s2)",
        "", "| Microbatch | Checkpointing | Kernel／step | Kernel 并集／跨度 | CPU forward+backward／trace step，ms |",
        "|---:|---|---:|---:|---:|",
    ]
    for m in (2, 4, 8):
        for mode in ("on", "off"):
            trace = rows[f"micro{m}_ckpt{mode}"]["trace"]
            cpu = trace["cpu"]
            lines.append(
                f"| {m} | {'开' if mode == 'on' else '关'} | {trace['kernels_per_step']:.0f} "
                f"| {100 * trace['kernel_active_fraction_of_span']:.2f}% "
                f"| {1000 * (cpu['policy_forward']['total_s'] + cpu['policy_backward']['total_s']) / 4:.2f} |")
    checkpoint_max = max(evidence["numerical_sanity"][f"micro{m}_ckptoff"][
        "gradient_relative_l2_vs_checkpoint_on"] for m in (2, 4, 8))
    micro_max = max(v["gradient_relative_l2_vs_micro8"] for v in evidence["numerical_sanity"].values())
    lines += [
        "", f"同一 microbatch 下，checkpointing 开／关在相同 8 个样本与初始参数上的"
        f"聚合梯度相对 L2 最大差异为 **{checkpoint_max:.3g}**，trajectory 最大绝对差异"
        f"为 **{max(evidence['numerical_sanity'][f'micro{m}_ckptoff']['trajectory_max_abs_m_vs_checkpoint_on'] for m in (2, 4, 8)):.3g} m**。"
        "全部计时 step 的 loss 与梯度 norm 有限，冻结 base 的 8192 元素抽样不变。"
        "这些检查支持开关未改变此执行路径的数值语义。[S1](#s1)[S2](#s2)",
        "", f"把同样 8 个样本按 microbatch 2／4／8 分块累积，梯度相对 microbatch 8 的"
        f"L2 最大差异为 **{micro_max:.3g}**；bf16 计算的 shape 差异可能改变舍入。"
        "这是一次短数值检查，不证明不同 batch 的长期收敛、精度或权重完全一致。",
        "", "## 4. 对 16／32 卡差异的意义",
        "", "增加 microbatch 若使每样本耗时下降而 kernel 数近似不变，"
        "说明固定执行成本被更多样本分摊；固定 batch 关闭 checkpointing 的收益则直接"
        "检验重算路径的成本。此次对照固定了主机、驱动、模型与缓存，"
        "为两项执行路径因素提供了独立于卡数的测量依据。",
        "", "原 16／32 卡训练使用 global batch 64、每卡 microbatch 4／2。"
        "若减少重算或改善 microbatch，应在保留全局 batch、优化器及数据顺序的条件下"
        "验证最终收益；单卡 samples/s 不能乘以 16 或 32 来预测真实集群吞吐。"
        "本测量没有 DDP、统计 collective、跨节点慢 rank，也没有解释长训后段额外下降的全部原因。"
        "在现有每样本成本下，继续暂停 32／64 卡长训；优先使用少量卡验证 checkpointing 优化"
        "和完整 step 计时，再决定是否值得分布式复测。[16／32 卡差异报告](scaling_difference_report.md)",
        "", "## 5. 证据文件",
        "", '<a id="s1"></a>',
        "**S1：实际源码和归档结果。** [benchmark_single_gpu.py](../experiments/benchmark_single_gpu.py)、"
        f"[完整结果](jobs/{evidence['job']['name']}/live/single_gpu.result.json)。"
        "源码、cache、模型版本及归档 SHA256 见 S2；归档中的 `steps.json` 与 `windows.json`"
        "保留逐步数据，未把插桩 trace 吞吐当作稳态吞吐。",
        "", '<a id="s2"></a>',
        "**S2：独立重算证据。** [single_gpu_evidence.json](single_gpu_evidence.json)、"
        "[三次窗口数据](single_gpu_windows.csv)、"
        "[CPU 分析脚本](../experiments/analyze_single_gpu.py)。"
        "记录六份 trace 的 kernel 区间并集、CPU 标注区间、数值检查与完整性门禁。",
        "", '<a id="s3"></a>',
        f"**S3：平台终态。** [任务](https://ml.bytedance.net/development/instance/jobs/{evidence['job']['sid']})、"
        "[终态与时间投影](single_gpu_terminal.json)、"
        "[提交配置摘要](single_gpu_submission.json)。"
        "签名 URL、原始启动脚本和认证信息不进入报告。",
    ]
    (ROOT / "single_gpu_report.md").write_text("\n".join(lines) + "\n")


def integrate(evidence):
    """Update the existing reviewable report, preserving the 16/32 observations."""
    rows, comparison = evidence["summaries"], evidence["comparisons"]
    terminal = evidence["terminal"]
    m2, m4, m8 = [rows[f"micro{m}_ckpton"] for m in (2, 4, 8)]
    cost = terminal["pod_wall_s"] / 3600
    cost_note = (
        f"16／32 卡差异复盘复用了原始记录，离线分析新增 GPU 成本为 0。"
        f"用户随后同意追加低成本诊断，现已完成 **1 张 H20、"
        f"{terminal['pod_wall_s'] / 60:.2f} 分钟、约 {cost:.4f} GPU·h** 的单卡测试，"
        f"创建到退出 {terminal['creation_to_exit_s'] / 60:.2f} 分钟，满足整任务 20 分钟上限。"
        "单卡实测见第 7 节；没有追加多卡任务。"
    )
    path = ROOT / "scaling_difference_report.md"
    text = path.read_text()
    text = text.replace(
        "# Driving-VLA：16／32 卡差异分析与低成本诊断建议",
        "# Driving-VLA：16／32 卡差异分析与单卡诊断实测")
    old_cost = (
        "本轮分析复用了原始记录，**新增 GPU 任务 0 个、新增 GPU 成本 0 GPU·h**。"
        "所有图表使用已有观测；验证集缺失的步数点没有插值，根因判断保留证据边界。")
    if old_cost in text:
        text = text.replace(old_cost, cost_note)
    else:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("16／32 卡差异复盘复用了原始记录"):
                lines[i] = cost_note
        text = "\n".join(lines) + "\n"
    text = text.replace(
        "这些路径是具体优化入口，但尚无消融实测，不能给出各自导致多少减速的比例。",
        "第 7 节补充了 microbatch 和 checkpointing 的单卡消融实测；"
        "单机收益不能直接解释原跨集群减速的全部比例。")
    text = text.replace("下一次小规模测量应增加三个时间：", "下一次分布式测量应记录三个时间：")
    text = text.replace(
        "有直接 trace 支持；具体算子贡献待消融",
        "原 trace 与新增单卡 batch 消融均支持；具体小算子贡献未单独测量")
    text = text.replace(
        "| 单卡 shape 与执行路径诊断 | **1 张 H20，整任务最多 20 分钟** |",
        "| 单卡 shape 与执行路径诊断，已完成 | **1 张 H20，整任务最多 20 分钟** |")
    text = text.replace(
        "两项补测即使都达到上限，新增 GPU 成本也最多约 **0.67 GPU·h**，"
        "不包含尚未决定的后续训练；本轮没有提交它们。",
        f"单卡补测实际成本约 **{cost:.4f} GPU·h**，两卡同步诊断尚未提交。"
        "若两项均用满各自上限，总新增成本约 0.67 GPU·h；后续任务范围由用户决定。")
    text = text.replace(
        "模型峰值 allocated 显存远低于 H20 容量，值得在单卡上测试更大的 microbatch 和减少重算；"
        "能否关闭 checkpointing 仍需测峰值及正确性。",
        "新增单卡测试已确认 microbatch 2／4／8 关闭 checkpointing 的显存峰值均在 H20 容量内，"
        "同一 microbatch 的开／关聚合梯度与输出一致。是否在分布式训练中保留收益仍需验证。")
    text = text.replace(
        "本次证据支持的资源决策是：**暂不新增 32／64 卡长训；先用单卡解释 per-sample "
        "计算效率，再按证据决定是否需要两卡同步诊断。**",
        "本次证据支持的资源决策是：**暂不新增 32／64 卡长训；单卡执行路径诊断已完成，"
        "优先验证减少重算与完整 step 计时，再决定是否需要两卡同步诊断。**")
    extra = [
        "## 7. 追加单卡实测：microbatch 与重算成本",
        "", f"同一张 HL H20、相同 pinned 模型与真实 cache，分别测 microbatch 2／4／8 "
        "与 checkpointing 开／关；每组 warmup 12 step 后计时 24 step，重复 3 次，"
        "第二轮逆序。所有输入驻留 GPU，隔离 dataloader／HDFS IO。"
        "吞吐来自未插桩窗口；每组另录 4 step trace。[E7](#e7)",
        "", f"本次单卡驱动为 **{evidence['hardware'][0]['driver_version']}**，"
        "与原 16／32 卡节点不同。六组实验都在同一主机／驱动下进行，"
        "相对收益有效；绝对延迟不直接回填原 HL／LF 结果。"
        "各组三窗口吞吐的变异系数为 "
        f"{100 * min(r['window_speed_cv'] for r in rows.values()):.2f}%～"
        f"{100 * max(r['window_speed_cv'] for r in rows.values()):.2f}%，"
        "改善幅度超过本次观察到的窗口波动，但没有跨主机独立重复。[E7](#e7)",
        "", "| Microbatch | Checkpointing 开：samples/s | Checkpointing 关：samples/s | 关闭后的吞吐倍数 | 开→关峰值 allocated，GiB |",
        "|---:|---:|---:|---:|---:|",
    ]
    for m in (2, 4, 8):
        a, b = rows[f"micro{m}_ckpton"], rows[f"micro{m}_ckptoff"]
        extra.append(
            f"| {m} | {a['samples_per_s']:.2f} | {b['samples_per_s']:.2f} "
            f"| {comparison[f'checkpoint_off_vs_on_micro{m}']['speed_ratio']:.2f}× "
            f"| {a['peak_allocated_gib']:.3f} → {b['peak_allocated_gib']:.3f} |")
    extra += [
        "", f"Checkpointing 开启时，microbatch 2／4 的 full step p50 为 "
        f"**{m2['full_step_p50_ms']:.2f}／{m4['full_step_p50_ms']:.2f} ms**，"
        f"每样本耗时为 **{m2['wall_ms_per_sample']:.2f}／{m4['wall_ms_per_sample']:.2f} ms**。"
        f"Batch 增加一倍后每步时间变化 "
        f"{100 * (comparison['micro4_vs2_ckpton']['step_p50_ratio'] - 1):+.2f}%，"
        f"每样本成本明显下降。Microbatch 8 的每样本耗时 "
        f"{m8['wall_ms_per_sample']:.2f} ms，吞吐为 microbatch 2 的 "
        f"{comparison['micro8_vs2_ckpton']['speed_ratio']:.2f} 倍，仍非四倍线性增长。"
        "这与原 trace 所示的固定执行成本不能随 batch 缩小而同比下降一致。",
        "", "在原 global batch 64 下，从 16 卡／每卡 4 改成 32 卡／每卡 2，"
        "如果 local step 时间近似不变，仅增加卡数就难以缩短一次全局更新。"
        "本次 batch 对照支持原 cached profile 几乎没有扩卡收益的固定成本假设；"
        "跨节点同步、慢 rank 与长训额外下降仍未被此单机测量分离。",
        "", f"固定 microbatch 后，关闭 checkpointing 的吞吐提升与 trace kernel 数变化"
        "提供了重算成本的直接对照。"
        f"三种 batch 的开／关聚合梯度相对 L2 差异最大值为 "
        f"{max(evidence['numerical_sanity'][f'micro{m}_ckptoff']['gradient_relative_l2_vs_checkpoint_on'] for m in (2, 4, 8)):.3g}；"
        "loss／梯度 norm 全部有限，冻结权重抽样未改变。"
        "应优先验证关闭 checkpointing 后的分布式吞吐和显存，"
        "保持 global batch、LR、数据顺序及训练时长一致。",
        "", "![单张 H20 的 microbatch 与 checkpointing 六组对照](single_gpu_comparison.png)",
        "", f"完整 pod 生命周期 **{terminal['pod_wall_s']:.1f} 秒／{cost:.4f} GPU·h**，"
        f"创建到退出 **{terminal['creation_to_exit_s']:.1f} 秒**。"
        f"其中模型／cache 解包 {evidence['environment']['extract_s']:.1f} 秒，"
        f"模型建立 {evidence['environment']['model_setup_s']:.1f} 秒；"
        f"18 个稳态计时窗口合计 {evidence['uninstrumented_window_wall_s']:.1f} 秒。"
        "其余时间包含启动、依赖、数值检查、warmup、trace、清理与归档，"
        "不能只用稳态窗口时长报告此次新增资源成本。",
        "", "**适用范围：** 1 台主机、8 个真实 train 样本、每组 3 个短窗口。"
        "单卡 optimizer batch 随 microbatch 变化，本次不评估 global batch 64 的收敛。"
        "没有 DDP、跨节点步尾 collective 或一小时验证指标；"
        "不能把单卡吞吐乘以卡数当作实测，也不能把 kernel 活动比例当 SM 利用率。"
        "两卡同步诊断和更多卡训练均未启动。",
        "",
    ]
    begin = text.find("## 7. 追加单卡实测：")
    if begin != -1:
        end = text.index("## 8. 证据与交付文件", begin)
        text = text[:begin] + "\n".join(extra) + "\n" + text[end:]
    else:
        text = text.replace("## 7. 证据与交付文件", "\n".join(extra) + "\n## 8. 证据与交付文件")
    e7 = (
        '\n<a id="e7"></a>\n'
        "**E7：追加单卡实测与终态核验。** [单卡完整报告](single_gpu_report.md)、"
        "[重算证据与归档 SHA256](single_gpu_evidence.json)、"
        "[三次窗口原始数据](single_gpu_windows.csv)、"
        "[平台退出与时间记录](single_gpu_terminal.json)。"
        "共 432 个未插桩 step、18 个计时窗口、6 份 4-step trace，"
        "源码、模型和 cache 与既有实验核对一致；平台和归档退出码均为 0。"
        "这些属于同一任务内多种记录的交叉核验，不是多台主机的独立重复实验。\n")
    if '<a id="e7"></a>' in text:
        text = text[:text.index('\n<a id="e7"></a>')] + e7
    else:
        text += e7
    path.write_text(text)


if __name__ == "__main__":
    main()
