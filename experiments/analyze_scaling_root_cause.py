"""Reconcile the original slowdown with step distributions and historical metrics.

This performs no GPU work. Phase maxima are kept separate, and the distribution
decomposition is explicitly descriptive rather than a causal effect estimate.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
TMP = Path("/tmp/driving-session")
sys.path.insert(0, str(ROOT / ".tools/report-deps"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {
    16: "driving-diagnose16-16h20-a1-20261028",
    32: "driving-scale32-32h20-a1-20261028",
}


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distributions(values, threshold):
    fast = values[values < threshold]
    slow = values[values >= threshold]
    assert len(fast) and len(slow)
    return dict(fast_count=len(fast), slow_count=len(slow),
                fast_fraction=float(len(fast) / len(values)),
                fast_mean_ms=float(fast.mean()), slow_mean_ms=float(slow.mean()))


def decomposition(a, b, threshold):
    """Symmetric exact decomposition: changed mixture vs changed within-group mean."""
    x, y = distributions(a, threshold), distributions(b, threshold)
    p = (x["fast_fraction"] + y["fast_fraction"]) / 2
    frequency = (y["fast_fraction"] - x["fast_fraction"]) * (
        (x["fast_mean_ms"] + y["fast_mean_ms"]) / 2 -
        (x["slow_mean_ms"] + y["slow_mean_ms"]) / 2)
    within = p * (y["fast_mean_ms"] - x["fast_mean_ms"]) + (
        1 - p) * (y["slow_mean_ms"] - x["slow_mean_ms"])
    delta = float(b.mean() - a.mean())
    assert np.isclose(frequency + within, delta, atol=1e-10)
    return dict(threshold_ms=threshold, runs={"16": x, "32": y},
                frequency_component_ms=float(frequency),
                within_group_component_ms=float(within), observed_delta_ms=delta,
                interpretation="Descriptive identity; not a causal allocation to hardware or NCCL")


def archived_case(world):
    directory = OUT / "jobs" / RUNS[world]
    for archive in directory.glob("*.tar.gz"):
        with tarfile.open(archive) as tar:
            members = {m.name: m for m in tar if m.isfile()}
            target = "results/training/lora_head/steps.json"
            if target not in members:
                continue
            rows = json.load(tar.extractfile(members[target]))
            result = json.load(tar.extractfile(members["results/training/lora_head/result.json"]))
            times = {
                "profile_cached_result_mtime": members["results/training/profile_cached/result.json"].mtime,
                "lora_result_mtime": members["results/training/lora_head/result.json"].mtime,
            }
            logs = []
            for path in directory.glob("*.stdout.log"):
                for line in path.read_text(errors="replace").splitlines():
                    if not line.startswith("{"):
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("case") == "lora_head" and event.get("event") == "TRAIN_STEP":
                        logs.append(event)
            logs.sort(key=lambda e: e["step"])
            assert len({e["step"] for e in logs}) == len(logs)
            return rows, result, times, logs, archive
    raise FileNotFoundError(f"No archived lora_head steps for world={world}")


def metrics(world, times):
    # Both endpoints lie comfortably inside the one-hour LoRA case.
    # mtime is not an exact step timestamp; the 60-second guard avoids that claim.
    lower = times["profile_cached_result_mtime"] + 60
    upper = times["lora_result_mtime"] - 60
    records, sources = [], []
    for panel in (40, 90, 132, 124, 7, 112, 62):
        path = TMP / f"goal-grafana{world}-p{panel}.json"
        response = load(path)
        kept = []
        for entry in response.get("total", []):
            if not isinstance(entry.get("data"), list):
                continue
            query = json.loads(entry["payload"]) if entry.get("payload") else {}
            for series in entry["data"]:
                if not isinstance(series, dict) or not series.get("dps"):
                    continue
                tags = series.get("tags", {})
                pod = tags.get("pod_name", tags.get("_pod_name"))
                if pod and f"trial-{load(directory_job(world))['trial']}-" not in pod:
                    continue
                points = {k: v for k, v in series["dps"].items()
                          if lower <= float(k) <= upper}
                if not points:
                    continue
                values = np.array(list(points.values()), dtype=float)
                item = dict(panel=panel, metric=series["metric"], tags=tags,
                            query_fields=[q.get("multiFieldExpr", q.get("fields"))
                                          for q in query.get("queries", [])],
                            points=len(points), min=float(values.min()),
                            max=float(values.max()), mean=float(values.mean()),
                            nonzero_points=int(np.count_nonzero(values)),
                            raw_examples=dict(list(points.items())[:2] + list(points.items())[-2:]),
                            dps=points)
                kept.append(item)
                records.append(item)
        sources.append(dict(panel=panel, response_sha256=digest(path),
                            references=response.get("references", []),
                            retained_series=len(kept)))
    return dict(guarded_case_interval_epoch_s=[lower, upper],
                interval_method="profile_cached result mtime +60s through lora_head result mtime -60s",
                interval_utc=[datetime.fromtimestamp(t, timezone.utc).isoformat()
                              for t in (lower, upper)],
                sources=sources, series=records)


def directory_job(world):
    return OUT / "jobs" / RUNS[world] / "job.json"


def single_rank_control():
    source = load(OUT / "single_gpu_evidence.json")
    archive = Path(source["job"]["archives"][0])
    with tarfile.open(archive) as tar:
        rows = json.load(tar.extractfile("results/single_gpu/steps.json"))
    controls = {}
    for key in ("micro2_ckpton", "micro2_ckptoff", "micro4_ckpton", "micro4_ckptoff"):
        selected = [r for r in rows if r["key"] == key]
        controls[key] = dict(
            **{k: source["summaries"][key][k] for k in
               ("full_step_p50_ms", "full_step_p95_ms", "samples_per_s", "peak_allocated_gib")},
            kernels_per_step=source["summaries"][key]["trace"]["kernels_per_step"],
            nccl_kernels=source["summaries"][key]["trace"]["nccl_kernel_count"],
            timed_steps=len(selected), slow_steps_ge250ms=sum(r["full_step_wall_s"] >= .25 for r in selected),
            forward_spikes=[dict(repeat=r["repeat"], step=r["step"], forward_ms=r["cpu_forward_s"] * 1000)
                            for r in selected if r["cpu_forward_s"] > .12])
    return dict(source_sha256=digest(OUT / "single_gpu_evidence.json"),
                archive_sha256=digest(archive), distributed=False,
                driver=source["hardware"][0]["driver_version"], controls=controls,
                caveat="Periodic forward tails exist without NCCL; old short windows do not identify their physical cause")


def main():
    data, arrays, block_rows = {}, {}, []
    for world in RUNS:
        rows, result, times, logs, archive = archived_case(world)
        assert [r["step"] for r in rows] == list(range(1, len(rows) + 1))
        system, environment = result["system"], result["environment"]
        assert environment["global_batch"] == 64 and environment["accumulation"] == 1
        skip = system["throughput_excluded_steps"]
        warm = rows[skip:]
        step = np.array([r["step_s"] * 1000 for r in warm])
        arrays[world] = step
        mean_ms = float(step.mean())
        assert np.isclose(64_000 / mean_ms, system["throughput_samples_s"], rtol=1e-10)
        phases = {}
        for key in ("forward_s", "backward_s", "data_wait_s", "optimizer_s"):
            values = np.array([r[key] * 1000 for r in warm])
            phases[key] = dict(mean_ms=float(values.mean()),
                               quantiles_ms=dict(zip(("p10", "p50", "p90", "p95"),
                                                     map(float, np.percentile(values, [10, 50, 90, 95])))),
                               correlation_with_step=float(np.corrcoef(values, step)[0, 1]))
        forward = np.array([r["forward_s"] * 1000 for r in warm])
        forward_slow = forward > 120
        phases["forward_gt120ms"] = dict(
            fraction=float(forward_slow.mean()),
            step_ge250ms_given_forward_gt120ms=float((step[forward_slow] >= 250).mean()),
            step_mean_with_ms=float(step[forward_slow].mean()),
            step_mean_without_ms=float(step[~forward_slow].mean()))
        residual = system["train_window_s"] - sum(r["step_s"] for r in rows)
        wall_check = []
        for previous, current in zip(logs, logs[1:]):
            segment = rows[previous["step"]:current["step"]]
            elapsed = current["wall_s"] - previous["wall_s"]
            measured = sum(r["step_s"] for r in segment)
            wall_check.append(dict(step_from=previous["step"] + 1, step_to=current["step"],
                                   wall_s=elapsed, max_compute_sum_s=measured,
                                   residual_s=elapsed - measured))
        summary = dict(
            world=world, name=RUNS[world], sid=load(directory_job(world))["sid"],
            microbatch=environment["microbatch"], warm_steps=len(warm), all_steps=len(rows),
            mean_step_ms=mean_ms, throughput_samples_s=system["throughput_samples_s"],
            quantiles_ms=dict(zip(("p05", "p10", "p50", "p90", "p95", "p99"),
                                  map(float, np.percentile(step, [5, 10, 50, 90, 95, 99])))),
            clipped_under400_mean_ms=float(step[step < 400].mean()),
            step_gt400_fraction=float((step >= 400).mean()),
            train_window_s=system["train_window_s"],
            recorded_compute_sum_s=sum(r["step_s"] for r in rows),
            wall_mean_per_update_ms=system["train_window_s"] / len(rows) * 1000,
            wall_residual_s=residual, wall_residual_fraction=residual / system["train_window_s"],
            wall_residual_per_update_ms=residual / len(rows) * 1000,
            caveat="Residual is an accounting diagnostic, not a strict tail-collective upper bound",
            phases=phases,
            log_interval_reconciliation=dict(intervals=len(wall_check),
                total_wall_s=sum(x["wall_s"] for x in wall_check),
                total_max_compute_s=sum(x["max_compute_sum_s"] for x in wall_check),
                total_residual_s=sum(x["residual_s"] for x in wall_check)),
            source=dict(archive=str(archive.relative_to(OUT)), sha256=digest(archive), mtimes=times),
            metrics=metrics(world, times))
        curves = [c for c in result["curves"] if c["step"] > 0]
        windows = []
        previous_step, previous_wall = 0, 0
        for curve in curves:
            if curve["step"] == previous_step:
                continue
            segment = rows[previous_step:curve["step"]]
            latencies = np.array([r["step_s"] * 1000 for r in segment])
            windows.append(dict(wall_end_s=curve["wall_s"], steps=len(segment),
                                mean_step_ms=float(latencies.mean()),
                                slow_fraction_ge250ms=float((latencies >= 250).mean()),
                                true_wall_throughput=64 * len(segment) / (curve["wall_s"] - previous_wall)))
            previous_step, previous_wall = curve["step"], curve["wall_s"]
        summary["five_minute_windows"] = windows
        for row in windows:
            block_rows.append(dict(world=world, **row))
        data[str(world)] = summary
    decomp = [decomposition(arrays[16], arrays[32], cut) for cut in (230, 240, 250, 260, 270, 280)]
    allocation = dict(new_jobs=0, max_concurrent_gpus=0, gpu_hours=0)
    if (OUT / "gc_probe_submission.json").exists():
        allocation.update(new_jobs=1, max_concurrent_gpus=1, gpu_hours=None)
    if (OUT / "gc_probe_terminal.json").exists():
        allocation["gpu_hours"] = load(OUT / "gc_probe_terminal.json")["gpu_hours"]
    result = dict(
        generated_at=datetime.now(timezone.utc).isoformat(),
        gpu_usage_this_goal=allocation, single_rank_control=single_rank_control(),
        runs=data, distribution_decomposition=decomp,
        timing_delta_ms=data["32"]["mean_step_ms"] - data["16"]["mean_step_ms"],
        wall_update_delta_ms=data["32"]["wall_mean_per_update_ms"] - data["16"]["wall_mean_per_update_ms"],
        phase_scope="Each phase is independently MAX-reduced across ranks; phases are not additive",
        limitations=["No per-rank hour-long trace or critical-rank identity",
                     "Different physical clusters, microbatch, plugin, channels, and one driver",
                     "Grafana GPU clock and duty cycle are pod aggregates",
                     "CPU panel90 averages ungrouped containers; limit=80 is not a verified worker cgroup quota",
                     "Step groups and consecutive samples are not independent trials",
                     "A mixture decomposition identifies where observed delay appears, not its physical origin"])
    (OUT / "scaling_root_cause_evidence.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    with (OUT / "scaling_root_cause_windows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(block_rows[0]))
        writer.writeheader()
        writer.writerows(block_rows)
    metric_rows = []
    for world, run in data.items():
        for series in run["metrics"]["series"]:
            for stamp, value in series["dps"].items():
                metric_rows.append(dict(world=world, panel=series["panel"], metric=series["metric"],
                    tags=json.dumps(series["tags"], sort_keys=True),
                    epoch_s=stamp, value=value))
    with (OUT / "scaling_root_cause_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = {16: "#2563eb", 32: "#e56b2f"}
    for world in RUNS:
        values = arrays[world]
        # Density is normalized over displayed values; outliers are reported separately.
        axes[0, 0].hist(values[values < 430], bins=np.arange(190, 435, 5), density=True,
                        histtype="step", linewidth=1.7, color=colors[world], label=f"{world} GPUs")
        forward = np.array([r["forward_s"] * 1000 for r in
                            archived_case(world)[0][10:]])
        x = np.sort(forward)
        axes[0, 1].plot(x, np.arange(1, len(x) + 1) / len(x), color=colors[world], label=f"{world} GPUs")
        windows = data[str(world)]["five_minute_windows"]
        axes[1, 0].plot([w["wall_end_s"] / 60 for w in windows],
                        [w["slow_fraction_ge250ms"] * 100 for w in windows],
                        marker="o", color=colors[world], label=f"{world} GPUs")
    axes[0, 0].axvline(250, color="#64748b", linestyle="--", linewidth=1)
    axes[0, 0].set(xlabel="Recorded global step (ms)", ylabel="Density",
                   title="Fast steps improve; slow steps become common", xlim=(190, 430))
    axes[0, 1].set(xlabel="MAX-reduced forward event duration (ms)", ylabel="Empirical CDF",
                   title="Delay shift is already visible in forward", xlim=(50, 300))
    axes[1, 0].set(xlabel="End of five-minute training window (min)", ylabel="Steps >=250 ms (%)",
                   title="Slow-step frequency grows in both long runs")
    chosen = next(d for d in decomp if d["threshold_ms"] == 250)
    bars = [chosen["frequency_component_ms"], chosen["within_group_component_ms"],
            chosen["observed_delta_ms"]]
    axes[1, 1].bar(["Changed frequency", "Within-group timing", "Observed delta"], bars,
                   color=["#e56b2f", "#2563eb", "#334155"])
    axes[1, 1].axhline(0, color="#64748b", linewidth=.7)
    for i, value in enumerate(bars):
        axes[1, 1].text(i, value + (1 if value >= 0 else -2), f"{value:+.2f}", ha="center")
    axes[1, 1].set(ylabel="Mean step change: 32 minus 16 (ms)",
                   title="Exact descriptive decomposition (cut=250 ms)")
    for ax in axes.flat:
        ax.grid(alpha=.15, axis="y")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.legend(frameon=False)
    fig.savefig(OUT / "scaling_root_cause.png", dpi=170)
    plt.close(fig)
    concise = {w: {k: r[k] for k in ("mean_step_ms", "wall_mean_per_update_ms",
        "wall_residual_s", "wall_residual_per_update_ms", "quantiles_ms", "phases")}
        for w, r in data.items()}
    print(json.dumps(dict(runs=concise, decomposition=chosen,
        metric_points=len(metric_rows)), indent=2))


if __name__ == "__main__":
    main()
