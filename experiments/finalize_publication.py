"""Finalize measured documentation and publish only a fully audited campaign."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile

from package_results import digest, inspect_text, package
from report import create_report, load_results, quality_delta
from verify_results import audit

REPOSITORY = "Cheongroach/bottleneck-guided-scaling-driving-vla"
REPOSITORY_URL = "https://github.com/"+REPOSITORY


def atomic_text(path, text):
    temporary = path.with_suffix(path.suffix+".upload")
    temporary.write_text(text)
    temporary.replace(path)


def results_section(root, state, results, cache):
    sides = {}
    for result in results:
        sides.setdefault(result["suite"], {})[result["environment"]["case"]] = result
    final_job = next(j for j in reversed(state["jobs"]) if j["suite"] == "final")
    final = sides["final"][final_job["case"]]
    q, system, env = final["final"], final["system"], final["environment"]
    baseline = cache["constant_velocity_baseline"]["val"]
    completed = datetime.fromtimestamp(state["completed_epoch_s"], timezone.utc).isoformat()
    gates = json.loads((root/"decisions.json").read_text())
    lines = ["## 结果与讨论", "",
        f'Campaign 于 {completed} 完成本地实验，'
        f'总墙钟 {(state["completed_epoch_s"]-state["start_epoch"])/3600:.3f} 小时。'
        f'输入共 {cache["train"]+cache["val"]} 样本，'
        f'{cache["train"]} train / {cache["val"]} val；teacher 生成 {cache["teacher_total_s"]:.3f} 秒。',
        "", "| 8 卡 / 30 分钟消融 | ADE m | FDE m | Action accuracy | Samples seen |",
        "|---|---:|---:|---:|---:|"]
    for case in ("trajectory_only", "trajectory_action", "frozen_head", "state_only",
                 "image_only", "lora_head", "lora_no_rationale"):
        r = sides["diagnose8"][case]
        lines.append(f'| {case} | {r["final"]["ADE_m"]:.3f} | {r["final"]["FDE_m"]:.3f} | '
                     f'{r["final"]["action_accuracy"]:.2%} | {r["system"]["samples_seen"]:,} |')
    lines += ["", "| 16 卡 / 一小时 LoRA | ADE m | FDE m | Samples seen |",
              "|---|---:|---:|---:|"]
    for case in ("lora_head", "large_batch"):
        r = sides["diagnose16"][case]
        lines.append(f'| batch={r["environment"]["global_batch"]} | '
                     f'{r["final"]["ADE_m"]:.3f} | {r["final"]["FDE_m"]:.3f} | '
                     f'{r["system"]["samples_seen"]:,} |')
    normal, large = sides["diagnose16"]["lora_head"], sides["diagnose16"]["large_batch"]
    lines += ["",
        f'在共同 samples seen 端点插值，batch=128 相对 batch=64 的 ADE 变化为 '
        f'{quality_delta(large, normal, "samples_seen"):+.2%}；'
        f'共同 GPU-hours 下为 {quality_delta(large, normal, "GPU_hours"):+.2%}。'
        "比较基于单 seed 曲线，不能替代显著性检验。",
        "", "| H20 | Synthetic samples/s | Cached samples/s | Cached data wait | Rank0 NCCL wall fraction |",
        "|---:|---:|---:|---:|---:|"]
    for suite in ("diagnose8", "diagnose16"):
        synthetic, cached = sides[suite]["profile_synthetic"], sides[suite]["profile_cached"]
        s = cached["system"]
        lines.append(f'| {cached["environment"]["world"]} | '
                     f'{synthetic["system"]["throughput_samples_s"]:.3f} | '
                     f'{s["throughput_samples_s"]:.3f} | {s["data_wait_fraction"]:.3%} | '
                     f'{s["profile"]["nccl_wall_fraction"]:.3%} |')
    if "scale32" in sides:
        validation = next(r for r in sides["scale32"].values() if r["environment"]["mode"] == "train")
        vq, vs = validation["final"], validation["system"]
        lines += ["", "### 32 卡验证", "",
            f'集群={validation.get("dc", "unknown")}，case={validation["environment"]["case"]}，'
            f'实际训练窗口={vs["train_window_s"]/3600:.3f} 小时；'
            f'ADE/FDE={vq["ADE_m"]:.3f}/{vq["FDE_m"]:.3f} m，'
            f'训练 throughput={vs["throughput_samples_s"]:.3f} samples/s。']
        if validation.get("dc") != normal.get("dc"):
            lines.append("32 卡运行于 LF，8/16 卡基线运行于 HL；这是跨集群验证，"
                         "不能把吞吐差异全部归因于卡数，后续实验按用户决定执行。")
    model = gates["scaling_model"]
    lines += ["",
        f'Cached 8→16 speedup={model["ratio_16_8"]:.3f}，'
        f'efficiency={model["efficiency_16"]:.2%}。'
        f'Teacher+eval 在诊断顺序工作量中的占比为 {gates["teacher_eval_fraction"]:.2%}，'
        "最终训练不参与这一 gate 统计。"]
    if not model["valid"]:
        lines.append("两点不能拟合非负饱和开销系数；本轮不提供该模型的 32/64 数值预测。"
                     "当前超线性结果不证明更大卡数仍有相同收益。")
    for suite in ("scale32", "scale64"):
        if suite not in sides:
            event = next(e for e in reversed(state["events"]) if e["event"] == suite.upper()+"_SKIPPED")
            lines.append(f'{suite} 于 {event["timestamp"]} 明确跳过，原因：{event["reason"]}。')
    attempts = [e for e in state["events"] if e["event"] == "CAPACITY" and
                e.get("requested") == 32 and e.get("dc") == normal.get("dc")]
    if attempts:
        lines.append(f'最初的 32 卡 gate 通过后，同一诊断队列的查询余量为 '
                     f'{attempts[-1]["available_H20"]:g} 张 H20；该次同队列启动未获资源。')
    lines += ["", f'### 最终配置：{env["case"]} / {env["world"]} H20', "",
        "配置按全部等时长 8 卡消融的最低验证 ADE 选择。"
        f'本次 LoRA={env["config"]["lora"]}，'
        f'λ_action={env["config"]["action"]}，λ_rationale={env["config"]["rationale"]}。',
        f'最终保存权重的 ADE/FDE={q["ADE_m"]:.3f}/{q["FDE_m"]:.3f} m，'
        f'miss@2m={q["miss_rate_2m"]:.2%}，action accuracy={q["action_accuracy"]:.2%}，'
        f'collision proxy={q["collision_proxy"]:.2%}。',
        f'缓存策略 batch latency/sample p50/p95='
        f'{q["cached_policy_latency_p50_ms"]:.3f}/{q["cached_policy_latency_p95_ms"]:.3f} ms，'
        f'p50 对应 {1000/q["cached_policy_latency_p50_ms"]:.3f} actions/s；排除 vision/teacher。',
        f'恒速基线 ADE/FDE={baseline["ADE_m"]:.3f}/{baseline["FDE_m"]:.3f} m；'
        f'最终相对变化分别为 {q["ADE_m"]/baseline["ADE_m"]-1:+.2%} / '
        f'{q["FDE_m"]/baseline["FDE_m"]-1:+.2%}。正值表示误差增加。',
        f'训练窗口 {system["train_window_s"]/3600:.3f} 小时，'
        f'总更新 {system["steps"]:,}，总 samples seen={system["samples_seen"]:,}；'
        f'选中 checkpoint step={final["selected_checkpoint_step"]:,}。'
        f'原始训练末端 ADE={final["raw_final"]["ADE_m"]:.3f} m，质量曲线保留原始状态。',
        f'最终四组输出已在新 backbone 上重载核验；SHA-256：`{final["checkpoint_sha256"]}`。',
        f'轨迹及动作验证覆盖 {q["samples"]} 个样本，摄像头干预覆盖 '
        f'{final["explanation"]["examples"]} 个样本 / {final["explanation"]["interventions"]} 次置零。']
    if env["config"]["rationale"] == 0:
        lines.append("最终配置舍弃弱 rationale 监督；rationale/evidence heads 未接受该项监督，"
                     "这些输出不能作为已学会的解释能力。其 camera intervention 指标仅反映输入敏感性。")
    lines += ["", "### 取舍与下一步", "",
        "当前数据等待和 rank0 通信占比低于预定阈值；LoRA 的语言反向与 activation checkpointing "
        "使等墙钟预算下的样本吞吐低于 frozen-head 消融。先评估学习收益，再承担更高训练成本。",
        "本轮 teacher 动作全部集中在 straight_brake，val 动作真值一致率仅 18.75%，"
        "teacher trajectory ADE/FDE 为 11.058/17.315 m。格式有效与监督有效分别报告；"
        "轨迹训练继续使用真实 ego pose。",
        "后续优先扩大独立场景验证和 test set、改善 teacher 标签、保留恒速基线；"
        "32/64 需要同队列资源和短跑实测，重新拟合吞吐模型后再评估端到端收益。",
        "mini 仅 10 场景，val 2 场景；单 seed、重复采样和低分辨率，"
        "验证集同时用于配置和 checkpoint 选择。Collision 为 occupied-circle 代理，"
        "没有 lane/map 或闭环安全验证。",
        "早期继承 eval 模式的 profile 已作废，正式决策只使用修正版；"
        "作废任务及其占用成本保留在报告。", "",
        "完整实测、逐步曲线、质量—成本比较和任务证据见 "
        "[实验报告](../results/report.md)、[机读结果](../results/metrics.json)。",
        "最终权重为 trainable LoRA/head 参数范围；冻结 base 的固定版本及输入位置见 "
        "[reproduce.json](../results/reproduce.json)。"]
    return "\n".join(lines)+"\n"


def finalize(project):
    root = project/"results"
    state = json.loads((root/"controller_state.json").read_text())
    if state.get("phase") != "complete":
        raise RuntimeError("Campaign not complete; finalization and publication were not started")
    before = audit(project)
    remaining = [c["requirement"] for c in before["checks"]
                 if not c["passed"] and c["requirement"] != "Manuscript includes completed results"]
    if remaining:
        raise RuntimeError("Artifacts failed the independent audit: "+"; ".join(remaining))
    results = load_results(root)
    cache = json.loads((root/"assets/cache_manifest.json").read_text())
    manuscript = project/"docs/manuscript.md"
    methods = manuscript.read_text().split("## 结果与讨论", 1)[0]
    methods = methods.replace("**方法手稿；实验进行中。**", "**方法与实测结果手稿；本地实验已完成。**")
    methods = methods.replace("本手稿记录结果产生前确定的实验协议与实现取舍",
                              "本手稿记录预定实验协议、实现取舍和最终实测结果")
    atomic_text(manuscript, methods+results_section(root, state, results, cache))
    archive_readme = project/"docs/README.md"
    old_status = "实验目前进行中。GitHub 的最终更新在实验结束、权重和报告核验后执行；报告中的“运行中快照”不能视为最终结果。"
    atomic_text(archive_readme, archive_readme.read_text().replace(old_status,
        "本地实验、最终权重重载及独立归档核验已完成。最终实测以实验报告和完成核验为准；"
        "GitHub 发布使用已核验的文件清单，并另行核对远端 commit 与文件内容。"))
    conversation = project/"docs/conversation.md"
    atomic_text(conversation, conversation.read_text().split("## 后续记录", 1)[0]+
        "## 完成记录\n\n"+results_section(root, state, results, cache).replace(
            "## 结果与讨论", "### 实验与权重完成摘要", 1)+
        "\n代码、报告、手稿、可恢复对话及选定证据经完整核验后归档至 "+
        REPOSITORY_URL+"；源归档 hashes 和公开文件清单见 results/publication_manifest.json。\n")
    create_report(root, final=True)
    verification = audit(project)
    if not verification["complete"]:
        raise RuntimeError("Final documentation or artifacts failed the completion audit")
    exported = package(project)
    return exported


def publish(project):
    root = project/"results"
    if not audit(project)["complete"]:
        raise RuntimeError("Publication requires a fresh successful completion audit")
    manifest = json.loads((root/"publication_manifest.json").read_text())
    manifest_digest = digest(root/"publication_manifest.json")
    selected = sorted([*manifest["files"], "results/publication_manifest.json"])
    for relative, item in manifest["files"].items():
        path = project/relative
        if not path.resolve().is_relative_to(project) or digest(path) != item["sha256"]:
            raise RuntimeError("Publication manifest content mismatch: "+relative)
    for relative in selected:
        path = project/relative
        if path.stat().st_size >= 100*1024**2:
            raise RuntimeError("Publication file exceeds the GitHub file limit: "+relative)
        if path.suffix in (".py", ".md", ".txt", ".json", ".csv") or path.name == ".gitignore":
            inspect_text(path.read_bytes(), relative)

    def command(arguments, echo=False, environment=None):
        result = subprocess.run(arguments, cwd=project, capture_output=True, text=True,
                                env=environment, timeout=300)
        if echo:
            print(result.stdout, flush=True)
        if result.returncode:
            print(result.stderr, flush=True)
            raise RuntimeError("Publication command failed: "+arguments[0])
        return result.stdout.strip()

    if command(["git", "branch", "--show-current"]) != "main":
        raise RuntimeError("Publication requires the reviewed main branch")
    if command(["git", "remote", "get-url", "origin"]) != REPOSITORY_URL+".git":
        raise RuntimeError("Publication origin differs from the authorized repository")
    staged = set(command(["git", "diff", "--cached", "--name-only"]).splitlines())
    if staged-set(selected):
        raise RuntimeError("Previously staged files fall outside the audited publication manifest")
    command(["git", "status", "--short"], echo=True)
    command(["git", "diff", "HEAD", "--", *selected], echo=True)
    command(["git", "log", "-5", "--oneline"], echo=True)
    command(["git", "add", "--", *selected])
    if set(command(["git", "diff", "--cached", "--name-only"]).splitlines())-set(selected):
        raise RuntimeError("Staged publication contains unreviewed files")
    command(["git", "diff", "--cached", "--stat"], echo=True)
    command(["git", "diff", "--cached", "--", *selected], echo=True)
    created_commit = bool(command(["git", "diff", "--cached", "--name-only"]))
    if created_commit:
        commit_script = """git commit -m "$(cat <<'EOF'
feat: publish reproducible H20 Driving-VLA diagnosis
EOF
)"
"""
        command(["bash", "-c", commit_script], echo=True)
    head = command(["git", "rev-parse", "HEAD"])
    if created_commit and set(command([
            "git", "diff-tree", "--no-commit-id", "--name-only", "-r", head]).splitlines())-set(selected):
        raise RuntimeError("Commit contains files outside the audited publication manifest")
    if digest(root/"publication_manifest.json") != manifest_digest:
        raise RuntimeError("Publication manifest changed during commit")
    for relative, item in manifest["files"].items():
        if digest(project/relative) != item["sha256"]:
            raise RuntimeError("Audited content changed during commit: "+relative)
    for relative in selected:
        committed = command(["git", "rev-parse", head+":"+relative])
        actual = command(["git", "hash-object", "--", relative])
        if committed != actual:
            raise RuntimeError("Committed content differs from audited file: "+relative)
    gh = str(project/".tools/gh")
    # Use the existing official gh login without changing git configuration.
    # The token goes only through the askpass pipe to git, never into a log or file.
    with tempfile.TemporaryDirectory(prefix="github-push-", dir="/tmp/driving-session") as directory:
        askpass = Path(directory)/"askpass.py"
        askpass.write_text("""#!/usr/bin/env python3
import os, subprocess, sys
prompt = sys.argv[1].lower()
if "username" in prompt:
    print("x-access-token")
elif "password" in prompt:
    result = subprocess.run([os.environ["DRIVING_GH_CLI"], "auth", "token",
                             "--hostname", "github.com"], capture_output=True, text=True,
                            timeout=30)
    if result.returncode:
        raise SystemExit(result.returncode)
    sys.stdout.write(result.stdout.strip())
else:
    raise SystemExit(1)
""")
        askpass.chmod(0o700)
        environment = dict(os.environ, GIT_ASKPASS=str(askpass),
                           GIT_TERMINAL_PROMPT="0", DRIVING_GH_CLI=gh)
        for key in ("GIT_TRACE", "GIT_TRACE_CURL", "GIT_CURL_VERBOSE", "GIT_TRACE_PACKET",
                    "GIT_TRACE2", "GIT_TRACE2_EVENT", "GIT_TRACE2_PERF"):
            environment.pop(key, None)
        command(["git", "push", "-u", "origin", "main"], echo=True, environment=environment)
    remote_head = json.loads(command([gh, "api", "repos/"+REPOSITORY+"/commits/main"]))["sha"]
    if remote_head != head:
        raise RuntimeError("GitHub main does not match the published commit")
    response = json.loads(command([gh, "api", "repos/"+REPOSITORY+"/git/trees/"+head+"?recursive=1"]))
    if response.get("truncated"):
        raise RuntimeError("GitHub returned an incomplete file tree")
    blobs = {item["path"]: item["sha"] for item in response["tree"] if item["type"] == "blob"}
    for relative in selected:
        if blobs.get(relative) != command(["git", "hash-object", "--", relative]):
            raise RuntimeError("GitHub file content mismatch: "+relative)
    command(["git", "status", "--short"], echo=True)
    receipt = dict(publication_verified=True, repository=REPOSITORY_URL, commit=head,
                   commit_url=REPOSITORY_URL+"/commit/"+head,
                   verified_files=len(selected), verified_epoch_s=datetime.now(timezone.utc).timestamp())
    atomic_text(root/"publication_receipt.json", json.dumps(receipt, indent=2)+"\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    exported = finalize(project)
    print(json.dumps(dict(milestone="campaign_artifacts_finalized", **exported)), flush=True)
    if args.publish:
        print(json.dumps(dict(milestone="github_publication_verified", **publish(project))), flush=True)
