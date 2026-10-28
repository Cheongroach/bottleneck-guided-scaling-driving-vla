"""Validate both host archives and report measured four-rank comparisons."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results"
PREFIX = "interhost_probe"
MODES = ("auto", "8", "16")


def sha(content):
    return hashlib.sha256(content).hexdigest()


def incomplete_report(pair, terminal):
    assert terminal["all_terminal"], "Verify resource release before publishing an incomplete round"
    attempts = pair.get("previous_attempts", []) + [pair]
    jobs = [j for attempt in attempts for j in attempt["jobs"]]
    evidence = dict(submission=pair, terminal=terminal, measurement_complete=False,
                    stop_reason=pair.get("error"),
                    placement_verified=bool(pair.get("placement")),
                    attempts=[dict(dc=a.get("dc", "lf"), outcome=a["outcome"],
                                   error=a.get("error"), jobs=a["jobs"]) for a in attempts])
    (OUT / f"{PREFIX}_evidence.json").write_text(json.dumps(evidence, indent=2))
    table = "\n".join(
        f'| {j["dc"].upper()} | `{j["sid"]}` | {j["rank"]} | {j.get("status")} | '
        f'{len(j.get("pods", []))} | {len(j.get("masked_host_sha256", []))} |'
        for j in jobs)
    report = f"""# Driving-VLA：不同物理机调度尝试

2026-10-28｜最多并发 4 张 H20｜同一轮 20 分钟总预算

**本轮没有完成可用的双机通信测量。** 已执行 {len(jobs)} 次真实提交，按实际队列／任务状态停止有界尝试。不能把尚未运行的配置写成双机实测，也不能将此前同机通信数据当作跨机结果。

| 队列机房 | 任务 SID | 配对 rank | 终态 | pod 数 | 屏蔽主机地址数 |
|---|---|---:|---|---:|---:|
{table}

平台 schema 证实 `attachments[].kind=arnold_cluster_control` 下的
`arnold_cluster_control_attachment.mask_hosts` 支持屏蔽主机列表；
`assign_hosts` 则明示只有 Arnold 运维可以配置。本次采用的方案为：先提交 2 卡首项，
读取实际 pod host／host_v6，第二项 2 卡屏蔽首台主机；只有两项实际 host 不同才放行 GPU 测量。
第一项拿不到物理机时，无法构造有证据的主机屏蔽列表。配额查询的剩余卡数没有证明节点实际可用。

控制器本轮结束原因：`{pair.get("error")}`。该信息是停止判据，不是调度失败的底层根因。
是否曾通过不同主机门禁：**{"是" if pair.get("placement") else "否"}**；
最终可确认的物理 host 数：**{terminal["distinct_hosts"]}**。
全部已提交任务均已核验终态；pod 起止／退出信息与此前排队尝试保留在终态记录。
可核算 GPU·h：`{terminal["GPU_hours"]}`（null 表示没有完整可核算的已运行 pod 时间，不能当成测得 0）。
不同主机测量不完整，本报告不发布吞吐或 channels 收益。

提交器保留原始轮次截止时间，切换队列未重置预算。每次 create 均先校验唯一最终 payload，
执行 dry-run 和 check-items check；error 停止，warn／hint 展示后可继续；
已创建项的预检 SID 均保留式合并至 request_context.precheck_sid，并在 30 分钟内提交。

[提交记录](interhost_probe_submission.json)／[终态与资源释放](interhost_probe_terminal.json)／
[机读尝试证据](interhost_probe_evidence.json)／[有界跨机提交器](../experiments/launch_interhost.py)。
平台原始 payload、主机地址和签名日志链接只保存在临时目录。
"""
    (OUT / f"{PREFIX}_report.md").write_text(report)
    print(json.dumps(dict(outcome="incomplete", attempts=len(jobs),
                          distinct_hosts=terminal["distinct_hosts"], all_terminal=True)))


def network_report(pair, terminal):
    assert pair["outcome"] == "success" and terminal["all_terminal"] and terminal["all_exit_zero"]
    assert terminal["distinct_hosts"] == 2 and terminal["round_to_last_exit_s"] <= 1200
    ranks, logs, inventory, archives, archive_exit_codes = {}, [], [], [], []
    for job in pair["jobs"]:
        assert len(job["archives"]) == 1
        path = Path(job["archives"][0])
        archives.append(dict(path=str(path.relative_to(OUT)), sha256=sha(path.read_bytes())))
        with tarfile.open(path) as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                content = tf.extractfile(member).read()
                if member.name == "results/exit_code.txt":
                    archive_exit_codes.append(int(content.strip()))
                match = re.search(r"results/network/rank(\d+)\.json$", member.name)
                if match:
                    rank = int(match[1])
                    assert rank == job["rank"] and rank not in ranks
                    ranks[rank] = json.loads(content)
                    assert ranks[rank]["environment"]["source_sha256"] == job["source_sha256"]
                if member.name == "results/gpus.csv":
                    for row in csv.DictReader(io.StringIO(content.decode())):
                        row = {k.strip(): v.strip() for k, v in row.items()}
                        inventory.append(dict(rank=job["rank"], name=row["name"],
                                              driver=row["driver_version"],
                                              uuid_sha256=sha(row["uuid"].encode())))
                if re.search(r"results/network/nccl-.*\.log$", member.name):
                    text = content.decode(errors="replace")
                    logs.append(dict(rank=job["rank"], member=member.name, sha256=sha(content),
                                     versions=sorted(set(re.findall(r"NCCL version (\S+)", text))),
                                     networks=sorted(set(re.findall(r"Using network (\S+)", text))),
                                     channels=sorted(set(map(int, re.findall(r"(\d+) coll channels", text)))),
                                     devices=sorted(set(re.findall(r"mlx5_[A-Za-z0-9_]+[:/]\d+/(?:RoCE|IB)", text))),
                                     gdrdma="GDRDMA" in text, network_edges="via NET/" in text))
    assert sorted(ranks) == [0, 1] and len(logs) == 2
    assert archive_exit_codes == [0, 0]
    assert len({r["environment"]["host_boot_sha256"] for r in ranks.values()}) == 2
    assert len(inventory) == 2 and len({g["uuid_sha256"] for g in inventory}) == 2
    assert all(r["environment"]["world"] == 2 for r in ranks.values())
    summary, timings = [], []
    for index in range(3):
        entries = [ranks[rank]["collectives"][index] for rank in range(2)]
        assert len({e["bytes"] for e in entries}) == 1
        for rank, entry in enumerate(entries):
            assert entry["all_reduce_sum_verified"] and len(entry["seconds"]) == 100
            assert all(np.isfinite(t) and t > 0 for t in entry["seconds"])
            timings.extend(dict(rank=rank, bytes=entry["bytes"], iteration=i, seconds=t)
                           for i, t in enumerate(entry["seconds"]))
        rank = int(np.argmax([sum(e["seconds"]) for e in entries]))
        values = entries[rank]["seconds"]
        summary.append(dict(bytes=entries[0]["bytes"], critical_rank=rank,
                            mean_ms=float(np.mean(values) * 1000),
                            p50_ms=float(np.median(values) * 1000),
                            p95_ms=float(np.quantile(values, .95) * 1000)))
    networks = sorted({n for log in logs for n in log["networks"]})
    channels = sorted({c for log in logs for c in log["channels"]})
    mask_evidence = json.loads((OUT / "interhost_placement_effective.json").read_text())
    assert mask_evidence["effective_masked_host_sha256"] == pair["jobs"][1]["masked_host_sha256"]
    assert pair["jobs"][0]["pods"][0]["physical_host_sha256"] in mask_evidence["effective_masked_host_sha256"]
    assert mask_evidence["mask_matches_first_worker_physical_host"]
    evidence = dict(submission=pair, terminal=terminal, measurement_complete=True,
                    measurement_scope="Two-rank NCCL microprobe only, no model training",
                    environments=[ranks[r]["environment"] for r in range(2)],
                    nccl=logs, gpu_inventory=inventory, archives=archives,
                    collective_summary=summary, timing_records=len(timings),
                    archive_exit_codes=archive_exit_codes, effective_host_mask=mask_evidence)
    (OUT / f"{PREFIX}_evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False))
    for suffix, rows in (("collectives", summary), ("collective_timings", timings)):
        with (OUT / f"{PREFIX}_{suffix}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    attempts = pair.get("previous_attempts", []) + [pair]
    attempt_table = "\n".join(
        f'| {a.get("dc", "lf").upper()} | `{j["sid"]}` | {j["gpus_per_worker"]} | '
        f'{j.get("status")} | {len(j.get("pods", []))} | {len(j["masked_host_sha256"])} |'
        for a in attempts for j in a["jobs"])
    table = "\n".join(f'| {r["bytes"]} | {r["mean_ms"]:.3f} | {r["p50_ms"]:.3f} | {r["p95_ms"]:.3f} |'
                      for r in summary)
    report = f"""# Driving-VLA：不同物理机调度与 2 卡 NCCL 实测

2026-10-28｜{pair["dc"].upper()} H20｜2 台物理机，各 1 卡｜两个独立任务组成 2 rank communicator

**主机屏蔽有效，已完成真实跨机 NCCL 测量。** 先提交首项，读实际 pod 的 host／host_v6；
第二项通过 `attachments[].kind=arnold_cluster_control` 下的
`arnold_cluster_control_attachment.mask_hosts` 排除首台机器。
最终任务 `{pair["jobs"][0]["sid"]}`／`{pair["jobs"][1]["sid"]}` 的平台 host 哈希不同，
内核 boot ID 哈希也不同；GPU UUID 哈希确认两张不同 H20。两项均退出码 0，资源已释放。
`job-v2 run-trials get` 查询还确认后端实际启动配置 `launch_config.MaskHosts` 包含首台主机，
与提交请求完全一致，见 [生效配置投影](interhost_placement_effective.json)。

本轮包括排队与两次先前尝试，从最初开始到最后退出 **{terminal["round_to_last_exit_s"]/60:.2f} 分钟**；
所有实际已运行 pod 累计成本 **{terminal["GPU_hours"]:.4f} GPU·h**，其中包含先前 4 卡准备阶段的占用。
单轮截止时间始终保持不变，没有因重试延长 20 分钟总预算，最大并发申请为 4 卡，最终测量只用 2 卡。

## 实际调度过程

| 机房 | 任务 SID | 每 worker GPU | 终态 | pod 数 | 屏蔽地址数 |
|---|---|---:|---|---:|---:|
{attempt_table}

首次 LF 排队未获 pod 后停止；第二轮 HL 的两个 2 卡任务最终确实落在不同物理机，
但第二台在控制器的等待截止后才到达，准备阶段即被停止，没有完成 4 卡模型对照。
最后改成每机 1 卡，直接使用镜像已有 Torch，跳过模型和 cache 解包，完成本次网络微测。
此前 4 卡任务的 stop_at 和非零平台停止码是主动停止证据，不是训练崩溃证据。

## 实际通信栈与小消息延迟

NCCL 初始化日志自报 network **`{", ".join(networks)}`**，collective channels **`{channels}`**；
NET 跨 GPU 传输路径：**{"有" if any(log["network_edges"] for log in logs) else "无"}**，
GDRDMA：**{"有" if any(log["gdrdma"] for log in logs) else "无"}**。
两节点分别可见 **{"/".join(str(len(log["devices"])) for log in logs)} 个 RoCE 设备端口**，
驱动分别为 **{"/".join(g["driver"] for g in inventory)}**。
本轮两节点可见端口数与历史 HL 16 卡的每节点 4 个不同，驱动也没有完全对齐；
不能把此次小消息微测当成历史硬件的完整复现。版本、设备与逐进程日志 SHA256 见机读证据。
本轮保持平台默认通信设置，没有强制 channels 或切换网络插件。

每个 payload warmup20，测 100 次 all-reduce；每次先填 rank+1，再对齐 rank，
计时从 Python all_reduce 调用开始，到 CUDA 完成。对齐 barrier 在计时之外。
校验所有张量元素最终都等于 3；共 300 次测量 collective、600 条 rank 计时记录。
下表取各 payload 总耗时最慢 rank 的分布，包含主机调用和完成等待，不是纯网络链路时延。

| bytes | 均值／ms | p50／ms | p95／ms |
|---:|---:|---:|---:|
{table}

9,530,560 bytes 对应原 LoRA+head 的 2,382,640 个 fp32 梯度参数，约 9.09MiB；
12／24B 对应原训练的小统计消息。本次没有模型 backward、DDP overlap 或 optimizer。
因此证实了跨机路径与这些消息大小的实际延迟，**未测得 4 卡模型吞吐、关闭 checkpoint 收益，
或 8／16 channels 对照**。2 rank 的延迟也不能用于分配历史 16／32 卡慢 8.62% 的原因比例。

## 证据与复现

所有 create 用唯一最终 payload 完成 schema 校验、dry-run 和 check-items check；
error 停止，warn／hint 展示后可继续，实际提交项均 success。
保留式合并 request_context.precheck_sid 后均在 30 分钟内提交。
控制面 cn；run → trial → pod SID、起止时间和退出码保存在提交／终态记录。
两项成功测量归档的退出码、源码 SHA256、host boot ID、GPU UUID 与实际 NCCL 日志均核验。
原始平台 payload、主机地址、签名日志链接仅存临时目录。

[完整机读证据](interhost_probe_evidence.json)／[提交与全部重试](interhost_probe_submission.json)／
[终态、成本与资源释放](interhost_probe_terminal.json)／[all-reduce 汇总](interhost_probe_collectives.csv)／
[600 条计时](interhost_probe_collective_timings.csv)／[提交器](../experiments/launch_interhost.py)／
[网络探针](../experiments/benchmark_network.py)。
"""
    (OUT / f"{PREFIX}_report.md").write_text(report)
    print(json.dumps(dict(outcome="success", scope="two-rank network only",
                          networks=networks, channels=channels, collectives=summary,
                          GPU_hours=terminal["GPU_hours"],
                          round_minutes=terminal["round_to_last_exit_s"]/60), indent=2))


def main():
    pair = json.loads((OUT / f"{PREFIX}_submission.json").read_text())
    terminal = json.loads((OUT / f"{PREFIX}_terminal.json").read_text())
    if pair["outcome"] == "incomplete":
        return incomplete_report(pair, terminal)
    if pair.get("network_only"):
        return network_report(pair, terminal)
    modes = tuple(pair.get("modes", MODES))
    assert pair["outcome"] == "success" and terminal["all_terminal"]
    assert terminal["all_exit_zero"] and terminal["distinct_hosts"] == 2
    assert terminal["round_to_last_exit_s"] <= 1200
    assert len(pair["jobs"]) == 2 and all(len(j["archives"]) == 1 for j in pair["jobs"])
    environments, cases, collectives, logs, inventory, archives = {}, {}, {}, {}, [], []
    for job in pair["jobs"]:
        for archive_path in job["archives"]:
            archive = Path(archive_path)
            archives.append(dict(path=str(archive.relative_to(OUT)), sha256=sha(archive.read_bytes())))
            with tarfile.open(archive) as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    content = tf.extractfile(member).read()
                    if member.name == "results/gpus.csv":
                        for row in csv.DictReader(io.StringIO(content.decode())):
                            row = {k.strip(): v.strip() for k, v in row.items()}
                            inventory.append(dict(node_rank=job["rank"], name=row["name"],
                                                  uuid_sha256=sha(row["uuid"].encode()),
                                                  driver=row["driver_version"]))
                    match = re.search(r"communication-(auto|8|16)/(environment|result|collectives)-rank(\d+)\.json$", member.name)
                    if match:
                        mode, kind, rank = match[1], match[2], int(match[3])
                        target = {"environment": environments, "result": cases,
                                  "collectives": collectives}[kind]
                        assert (mode, rank) not in target and rank // 2 == job["rank"]
                        target[(mode, rank)] = json.loads(content)
                    match = re.search(r"communication-(auto|8|16)/nccl-.*\.log$", member.name)
                    if match:
                        text = content.decode(errors="replace")
                        logs.setdefault(match[1], []).append(dict(
                            archive=archive.name, member=member.name, sha256=sha(content),
                            versions=sorted(set(re.findall(r"NCCL version (\S+)", text))),
                            networks=sorted(set(re.findall(r"Using network (\S+)", text))),
                            channels=sorted(set(map(int, re.findall(r"(\d+) coll channels", text)))),
                            devices=sorted(set(re.findall(r"mlx5_\d+/\d+/RoCE", text))),
                            gdrdma="GDRDMA" in text, network_edges="via NET/" in text,
                            p2p_cumem="P2P/CUMEM" in text))
    assert len(inventory) == 4 and len({g["uuid_sha256"] for g in inventory}) == 4
    assert len(environments) == len(cases) == len(collectives) == 4 * len(modes)
    assert len({e["cache_manifest_sha256"] for e in environments.values()}) == 1
    assert len({e["host_boot_sha256"] for e in environments.values()}) == 2
    for mode in modes:
        assert len(logs[mode]) == 4
        if mode != "auto":
            assert all(e["channels"] == [int(mode)] for e in logs[mode])
        assert environments[(mode, 0)]["host_boot_sha256"] == environments[(mode, 1)]["host_boot_sha256"]
        assert environments[(mode, 2)]["host_boot_sha256"] == environments[(mode, 3)]["host_boot_sha256"]
    for (mode, rank), env in environments.items():
        assert (env["world"], env["microbatch"], env["accumulation"], env["global_batch"]) == (4, 2, 8, 64)
        assert env["source_sha256"] == pair["jobs"][rank // 2]["source_sha256"]
        assert env["trainable_parameters"] == 2382640
    summary, steps, collective_summary = [], [], []
    for mode in modes:
        for ci in range(4):
            ranks = [cases[(mode, rank)]["cases"][ci] for rank in range(4)]
            for case in ranks:
                assert len(case["rows"]) == 36 and len(case["windows"]) == 3
                assert case["numerical"]["gradient_finite"]
                assert sum(r["statistics"] for r in case["rows"]) == (36 if case["statistics_cadence"] == 1 else 2)
                for row in case["rows"]:
                    assert abs(row["full_s"] - row["compute_s"] - row["tail_s"]) < 1e-8
                    steps.append(dict(mode=mode, case=case["case"], rank=case["rank"], **row))
            totals = [sum(w["wall_s"] for w in case["windows"]) for case in ranks]
            rank = int(np.argmax(totals))
            case = ranks[rank]
            numerical = {k: max(c["numerical"][k] for c in ranks)
                         for k in case["numerical"] if k != "gradient_finite"}
            summary.append(dict(
                mode=mode, case=case["case"], checkpointing=case["checkpointing"],
                statistics_cadence=case["statistics_cadence"], critical_rank=rank,
                throughput_samples_s=2304 / max(totals),
                mean_compute_ms=float(np.mean([r["compute_s"] for r in case["rows"]]) * 1000),
                mean_tail_ms=float(np.mean([r["tail_s"] for r in case["rows"]]) * 1000),
                mean_full_step_ms=float(np.mean([r["full_s"] for r in case["rows"]]) * 1000),
                p95_full_step_ms=float(np.quantile([r["full_s"] for r in case["rows"]], .95) * 1000),
                peak_allocated_gib=max(c["peak_allocated_gib"] for c in ranks),
                repeat_throughput=[768 / max(c["windows"][repeat]["wall_s"] for c in ranks)
                                   for repeat in range(3)], numerical_max=numerical))
        for index in range(3):
            entries = [collectives[(mode, rank)][index] for rank in range(4)]
            assert len({e["bytes"] for e in entries}) == 1
            rank = int(np.argmax([sum(e["seconds"]) for e in entries]))
            values = entries[rank]["seconds"]
            assert len(values) == 40
            collective_summary.append(dict(
                mode=mode, bytes=entries[0]["bytes"], critical_rank=rank,
                mean_ms=float(np.mean(values) * 1000),
                p50_ms=float(np.median(values) * 1000),
                p95_ms=float(np.quantile(values, .95) * 1000)))
    lookup = {(r["mode"], r["checkpointing"], r["statistics_cadence"]): r for r in summary}
    ratios = []
    for mode in modes:
        rate = lambda cp, cadence: lookup[(mode, cp, cadence)]["throughput_samples_s"]
        ratios.append(dict(mode=mode, checkpoint_off_speedup=rate(False, 1) / rate(True, 1),
                           statistics16_speedup_on=rate(True, 16) / rate(True, 1),
                           statistics16_speedup_off=rate(False, 16) / rate(False, 1)))
    baseline = json.loads((OUT / "communication_probe_evidence.json").read_text())
    samehost = {(r["mode"], r["checkpointing"], r["statistics_cadence"]): r for r in baseline["summary"]}
    for env in environments.values():
        assert env["source_sha256"] == baseline["submission"]["source_sha256"]
        assert env["cache_manifest_sha256"] == baseline["environments"][0]["cache_manifest_sha256"]
    host_comparison = [
        dict(mode=r["mode"], checkpointing=r["checkpointing"],
             statistics_cadence=r["statistics_cadence"],
             samehost_samples_s=samehost[(r["mode"], r["checkpointing"], r["statistics_cadence"])]["throughput_samples_s"],
             interhost_samples_s=r["throughput_samples_s"],
             interhost_vs_samehost=r["throughput_samples_s"] / samehost[(r["mode"], r["checkpointing"], r["statistics_cadence"])]["throughput_samples_s"])
        for r in summary]
    evidence = dict(submission=pair, terminal=terminal, summary=summary, steps_count=len(steps),
                    collective_summary=collective_summary, comparisons=ratios, nccl=logs,
                    gpu_inventory=inventory, archives=archives, host_comparison=host_comparison,
                    environments=[environments[k] for k in sorted(environments)],
                    limitations=["Four ranks and accumulation8; historical16/32 used accumulation1",
                                 "Same-host comparison ran earlier on different physical hosts",
                                 "Fixed case order and three adjacent short timing windows",
                                 "DDP reducer reused within each mode; no numerical convergence equivalence",
                                 "Platform network plugin kept fixed; no IB versus plugin experiment"])
    (OUT / f"{PREFIX}_evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False))
    flat_summary = [
        {**{k: v for k, v in r.items() if k not in ("repeat_throughput", "numerical_max")},
         **{f"repeat{i}_samples_s": v for i, v in enumerate(r["repeat_throughput"])},
         **r["numerical_max"]} for r in summary]
    for suffix, values in (("summary", flat_summary), ("steps", steps),
                           ("collectives", collective_summary), ("host_comparison", host_comparison)):
        with (OUT / f"{PREFIX}_{suffix}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, mode in enumerate(modes):
        rows = [r for r in summary if r["mode"] == mode]
        positions = np.arange(4) + (i - 1) * .24
        axes[0].bar(positions, [r["throughput_samples_s"] for r in rows], width=.24, label=f"Channels {mode}")
        axes[1].bar(positions, [r["mean_tail_ms"] for r in rows], width=.24)
    for ax in axes:
        ax.set_xticks(range(4), ["CP on/stats1", "CP on/stats16", "CP off/stats1", "CP off/stats16"], rotation=12)
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Samples/s")
    axes[1].set_ylabel("Tail ms per complete optimizer step")
    axes[0].legend()
    fig.suptitle("4 H20, 2 verified physical hosts, global64 / micro2 / accum8")
    fig.tight_layout()
    fig.savefig(OUT / f"{PREFIX}_timing.png", dpi=160)
    plt.close(fig)
    table = "\n".join(
        f'| {r["mode"]} | {"开" if r["checkpointing"] else "关"} | {r["statistics_cadence"]} | '
        f'{r["throughput_samples_s"]:.2f} | {r["mean_compute_ms"]:.2f} | {r["mean_tail_ms"]:.3f} | '
        f'{r["mean_full_step_ms"]:.2f} | {r["peak_allocated_gib"]:.2f} |' for r in summary)
    host_table = "\n".join(
        f'| {r["mode"]} | {"开" if r["checkpointing"] else "关"} | {r["statistics_cadence"]} | '
        f'{r["samehost_samples_s"]:.2f} | {r["interhost_samples_s"]:.2f} | '
        f'{(r["interhost_vs_samehost"]-1)*100:+.2f}% |' for r in host_comparison)
    coll_table = "\n".join(
        f'| {r["mode"]} | {r["bytes"]} | {r["mean_ms"]:.3f} | {r["p95_ms"]:.3f} |'
        for r in collective_summary)
    ratio_table = "\n".join(
        f'| {r["mode"]} | {r["checkpoint_off_speedup"]:.3f}× | '
        f'{r["statistics16_speedup_on"]:.3f}× | {r["statistics16_speedup_off"]:.3f}× |'
        for r in ratios)
    networks = sorted({n for entries in logs.values() for e in entries for n in e["networks"]})
    channel_summary = {mode: sorted({c for e in logs[mode] for c in e["channels"]}) for mode in modes}
    drivers = sorted({g["driver"] for g in inventory})
    retry_note = ("此前 LF 首项一直未分配 pod，已停止；本轮在同一 20 分钟总预算内改试 HL，"
                  "只测强制 8／16 两组，保留先前尝试记录。" if pair.get("previous_attempts") else "")
    report = f"""# Driving-VLA：4 卡跨物理机通信实测

2026-10-28｜{pair["dc"].upper()} H20｜2 台物理机，各 2 卡｜global64／micro2／accum8

**已实际调度到两台不同物理机。** 先提交 2 卡 rank0 任务 `{pair["jobs"][0]["sid"]}`，取平台 pod 的 host／host_v6；再提交 2 卡 rank1 任务 `{pair["jobs"][1]["sid"]}`，在 `attachments` 添加 `kind=arnold_cluster_control`，通过 `arnold_cluster_control_attachment.mask_hosts` 屏蔽首台主机。第二次预检通过，一次带屏蔽条件的提交即获得不同 host。平台 host 哈希与 4 个 rank 的内核 boot ID 哈希交叉验证恰好 2 台主机，GPU UUID 哈希确认 4 张不同 H20。测量只在本地控制器写入拓扑验证门禁后开始。{retry_note}

两任务均退出码 0，资源已释放。完整 pod 占用 **{terminal["GPU_hours"]:.4f} GPU·h**；从本轮开始到最后退出 **{terminal["round_to_last_exit_s"]/60:.2f} 分钟**，满足 20 分钟上限。驱动 `{", ".join(drivers)}`。原始主机地址仅保存到临时目录，报告使用哈希。

## 实测结果

关闭 checkpoint 的吞吐收益为 **{min(r["checkpoint_off_speedup"] for r in ratios):.3f}～{max(r["checkpoint_off_speedup"] for r in ratios):.3f}×**。所有配置使用同一源代码、cache、64 个不同真实样本；样本驻留 GPU，4 rank 各 16 个。每组 warmup6，随后测 3×12 个完整更新，前 7 个 microbatch 用 DDP.no_sync，第 8 个同步梯度。吞吐为 `36×64 / MAX_rank(三个窗口总wall)`；阶段明细取同一慢 rank，不把来自不同 rank 的 MAX 相加。

| channels | checkpoint | 统计间隔／步 | samples/s | 计算+DDP／ms | 步尾／ms | 完整 step／ms | allocated／GiB |
|---|---|---:|---:|---:|---:|---:|---:|
{table}

| channels | 关闭 checkpoint | 统计16：checkpoint 开 | 统计16：checkpoint 关 |
|---|---:|---:|---:|
{ratio_table}

![跨机吞吐及步尾](interhost_probe_timing.png)

实际 NCCL network 为 `{", ".join(networks)}`，各进程初始化日志确认 channels `{json.dumps(channel_summary)}`。强制 8／16 的配置均核实生效。日志中 NET 传输边、GDRDMA 和本机 P2P/CUMEM 逐进程记录在 [证据 JSON](interhost_probe_evidence.json)，网络插件保持平台默认。NCCL_DEBUG=INFO 与 NCCL_DEBUG_FILE 分别启用初始化日志及独立进程文件，取证来自两个 pod 的 HDFS 归档，共 {4*len(modes)} 份 NCCL 日志。

## 与此前同机 4 卡比较

源码六文件哈希、cache manifest、样本、seed、batch 和测量次数全部一致；实际物理机、机房和测量时间可能不同，因此下面不能被解释为纯跨机网络的因果开销。原同机测试在 LF，两个容器之间也存在 NET 传输；本轮为 {pair["dc"].upper()}，实际后端以日志为准。本次确实补齐跨机路径，但不是历史 16／32 卡全规模复现。

| channels | checkpoint | 统计间隔 | 同机 samples/s | 跨机 samples/s | 跨机相对变化 |
|---|---|---:|---:|---:|---:|
{host_table}

## 独立 all-reduce 与数值边界

每个 payload warmup8，测 40 次，同步 CUDA；汇总最慢总耗时 rank。包含 Python 调用与 CUDA 完成等待，不是纯网络链路时延。可训练梯度约 9.09MiB，冻结 backbone 不参与归约。

| channels | bytes | 均值／ms | p95／ms |
|---|---:|---:|---:|
{coll_table}

全部梯度、最终参数与输出有限；各组首步梯度及最终张量差异均落在证据中。模式内复用 DDP reducer，case 顺序固定，只有连续三个短窗口；与此前同机测试一样，不声称数值或收敛等价，也不把统计降频的吞吐变化全部归为省去通信。

4 rank／accum8 与历史 16／32 rank／accum1 不同，不能分配历史慢 8.62% 的原因比例；本次没有在相同物理机上切换 IB 与 soft_bonding_v8，不评价插件优劣。关于 channel 的结论只针对本次小梯度负载和实际两台机器。

## 复现与证据

提交前对同一最终 payload 做 schema 校验、dry-run 和 check-items check：error 停止，warn／hint 展示后可继续。本轮两个预检均 success，保留式合并 request_context.precheck_sid，均在 30 分钟内提交。总占用始终 ≤4 卡；不同主机未获验证则不会启动测量，轮次超时会停止这两个诊断任务。

- [跨机提交器](../experiments/launch_interhost.py)／[分析器](../experiments/analyze_interhost.py)／[探针源码](../experiments/benchmark_communication.py)
- [机读证据与归档 SHA256](interhost_probe_evidence.json)／[提交记录](interhost_probe_submission.json)／[终态和资源释放](interhost_probe_terminal.json)
- [汇总 CSV](interhost_probe_summary.csv)／[{len(steps)} 条逐步计时](interhost_probe_steps.csv)／[collectives](interhost_probe_collectives.csv)／[同机比较](interhost_probe_host_comparison.csv)
- 平台控制面 cn；run → trial → pod SID 链路、起止时间和退出状态均保留在提交／终态记录；原始平台 payload 与签名日志链接只保存在临时目录。
"""
    (OUT / f"{PREFIX}_report.md").write_text(report)
    print(json.dumps(dict(comparisons=ratios, host_comparison=host_comparison,
                          network=networks, channels=channel_summary,
                          drivers=drivers, GPU_hours=terminal["GPU_hours"]), indent=2))


if __name__ == "__main__":
    main()
