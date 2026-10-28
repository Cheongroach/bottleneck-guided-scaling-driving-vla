"""Audit GC-event overlap, paired windows, and full one-GPU allocation cost."""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

from collect_single_gpu_terminal import call, epoch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
sys.path.insert(0, str(ROOT / ".tools/report-deps"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summary(values):
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), p50=float(np.median(a)), p95=float(np.percentile(a, 95)),
                min=float(a.min()), max=float(a.max()))


def main():
    job = json.loads((OUT / "gc_probe_submission.json").read_text())
    response, status_sha = call(["job-v2", "runs", "get"], {"sid": job["sid"]}, "goal-gc-terminal-status")
    current = response["data"]
    assert current["status"] in ("done", "success")
    trial = current["runtime_context"]["trial_id"]
    response, pods_sha = call(["job-v2", "run-trial-pods", "list"],
                             {"filter": {"trial_sid_any": [trial]}}, "goal-gc-terminal-pods")
    pods = [{k: p.get(k) for k in ("sid", "pod_name", "start_at", "stop_at", "status", "exit_status")}
            for p in response["data"]]
    assert len(pods) == 1 and pods[0]["exit_status"] == 0 and pods[0]["stop_at"]
    created, start, stop = (epoch(current["created_at"]), epoch(pods[0]["start_at"]),
                            epoch(pods[0]["stop_at"]))
    assert 0 < stop - created <= 1200
    terminal = dict(sid=job["sid"], trial_id=trial, status=current["status"], pods=pods,
        created_at=current["created_at"], creation_to_exit_s=stop - created,
        pod_wall_s=stop - start, allocated_gpus=1, gpu_hours=(stop - start) / 3600,
        resource_release_evidence="Terminal success; single pod exit_status=0 and stop_at present",
        source_response_sha256=dict(status=status_sha, pods=pods_sha))
    (OUT / "gc_probe_terminal.json").write_text(json.dumps(terminal, indent=2))
    assert len(job["archives"]) == 1
    archive = Path(job["archives"][0])
    content, exits, hardware = {}, [], []
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.isfile():
                continue
            if member.name.endswith("exit_code.txt"):
                exits.append(tar.extractfile(member).read().decode().strip())
            elif member.name.endswith("gpus.csv"):
                hardware.extend(csv.DictReader(io.StringIO(tar.extractfile(member).read().decode()),
                                               skipinitialspace=True))
            elif member.name.startswith("results/gc_probe/") and member.name.endswith(".json"):
                content[Path(member.name).name] = json.load(tar.extractfile(member))
    assert exits == ["0"] and len(hardware) == 1
    environment = content["environment.json"]
    assert not environment["distributed"] and environment["world"] == 1
    assert environment["source_sha256"]["benchmark_gc.py"] == job["source_sha256"]["benchmark_gc.py"]
    assert content["result.json"] == json.loads(
        (archive.parent / "live/gc_probe.result.json").read_text())
    rows, events, windows = content["steps.json"], content["gc_events.json"], content["windows.json"]
    assert len(rows) == 864 and len(windows) == 9
    keys = ("ckpton_gcauto", "ckpton_gcdisabled", "ckptoff_gcauto")
    associations = []
    for row in rows:
        related = [e for e in events if e["timed"] and e["window"] == f'{row["repeat"]}_{row["key"]}'
                   and e["step"] == row["step"] and
                   e["start_s"] >= row["start_s"] and e["end_s"] <= row["end_s"]]
        forward_gc = [e for e in related if e["phase"] == "forward"]
        associations.append(dict(key=row["key"], repeat=row["repeat"], step=row["step"],
            full_step_ms=row["wall_s"] * 1000, forward_ms=row["cpu_forward_wall_s"] * 1000,
            forward_thread_cpu_ms=row["forward_thread_cpu_s"] * 1000,
            forward_gc_ms=sum(e["wall_s"] for e in forward_gc) * 1000,
            forward_gc_thread_cpu_ms=sum(e["thread_cpu_s"] for e in forward_gc) * 1000,
            gc_ms=sum(e["wall_s"] for e in related) * 1000,
            forward_gc_generations=",".join(str(e["generation"]) for e in forward_gc)))
    comparisons = {}
    for key in keys:
        selected = [r for r in associations if r["key"] == key]
        original = [r for r in rows if r["key"] == key]
        group_windows = [w for w in windows if w["key"] == key]
        forward_long = [r for r in selected if r["forward_ms"] > 120]
        long_gc = [r for r in selected if r["forward_gc_ms"] > 30]
        timed_events = [e for e in events if e["timed"] and e["window"].endswith("_" + key)]
        manual = [e for e in events if not e["timed"] and e["window"].endswith("_" + key)
                  and e["phase"] == "manual_cleanup"]
        comparisons[key] = dict(
            timed_steps=len(selected), samples_per_s=len(selected) * 2 / sum(w["window_wall_s"] for w in group_windows),
            window_samples_per_s=[w["samples_per_s"] for w in group_windows],
            full_step_ms=summary([r["full_step_ms"] for r in selected]),
            forward_ms=summary([r["forward_ms"] for r in selected]),
            forward_minus_gc_ms=summary([r["forward_ms"] - r["forward_gc_ms"] for r in selected]),
            sync_wait_ms=summary([r["sync_wait_s"] * 1000 for r in original]),
            forward_gt120ms_steps=len(forward_long), step_ge250ms=sum(r["full_step_ms"] >= 250 for r in selected),
            forward_long_with_gc_gt30ms=sum(r["forward_gc_ms"] > 30 for r in forward_long),
            timed_gc_by_generation={str(g): dict(
                count=len(selected_events), wall_ms=summary([e["wall_s"] * 1000 for e in selected_events]),
                thread_cpu_ms=summary([e["thread_cpu_s"] * 1000 for e in selected_events]))
                for g in (0, 1, 2) if (selected_events := [e for e in timed_events if e["generation"] == g])},
            long_forward_gc_ms=summary([r["forward_gc_ms"] for r in long_gc]) if long_gc else None,
            manual_cleanup_gc_total_ms=sum(e["wall_s"] for e in manual) * 1000,
            peak_allocated_gib=max(w["peak_allocated_bytes"] for w in group_windows) / 1024**3)
    auto, disabled = comparisons[keys[0]], comparisons[keys[1]]
    lookup = {(r["key"], r["repeat"], r["step"]): r for r in rows}
    pairs = [(r, lookup[(keys[1], r["repeat"], r["step"])]) for r in rows if r["key"] == keys[0]]
    evidence = dict(
        generated_at=datetime.now(timezone.utc).isoformat(), job=job, terminal=terminal,
        environment=environment,
        hardware=[{k: v for k, v in h.items() if k != "uuid"} for h in hardware],
        archive=dict(path=str(archive.relative_to(OUT)), sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                     bytes=archive.stat().st_size, exit_codes=exits),
        summaries=comparisons,
        gc_disabled_throughput_ratio=disabled["samples_per_s"] / auto["samples_per_s"],
        paired_loss_max_abs=float(max(abs(a["loss"] - b["loss"]) for a, b in pairs)),
        paired_grad_norm_max_abs=float(max(abs(a["norm"] - b["norm"]) for a, b in pairs)),
        windows=windows, gc_events=events,
        limitations=["One newly allocated HL host; no direct per-rank GC timestamps from original 16/32 jobs",
                     "GC callbacks and retained timing dictionaries alter object allocation cadence",
                     "GC-disabled timing excludes manual cleanup; not a recommendation to disable GC indefinitely",
                     "Resident micro2 probe is not global64 distributed training",
                     "Paired loss/norm and a head sum do not establish full convergence or all-weight equality"])
    (OUT / "gc_probe_evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False))
    for filename, values in (("gc_probe_steps.csv", associations), ("gc_probe_events.csv", events)):
        with (OUT / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    labels = ("Checkpoint on / GC auto", "Checkpoint on / GC disabled", "Checkpoint off / GC auto")
    colors = ("#e56b2f", "#2563eb", "#16a085")
    for key, label, color in zip(keys, labels, colors):
        a = np.sort([r["full_step_ms"] for r in associations if r["key"] == key])
        axes[0].plot(a, np.arange(1, len(a) + 1) / len(a), label=label, color=color)
    axes[0].set(xlabel="Full step (ms)", ylabel="Empirical CDF", title="One GPU: removing GC tests forward tails")
    axes[0].legend(frameon=False, fontsize=8)
    selected = [r for r in associations if r["key"] == keys[0]]
    axes[1].scatter([r["forward_gc_ms"] for r in selected], [r["forward_ms"] for r in selected],
                    color=colors[0], alpha=.5, s=16)
    base = np.median([r["forward_ms"] - r["forward_gc_ms"] for r in selected])
    x = np.array([0, max(r["forward_gc_ms"] for r in selected)])
    axes[1].plot(x, base + x, linestyle="--", color="#334155", label="Normal forward + GC duration")
    axes[1].set(xlabel="GC inside forward (ms)", ylabel="Forward wall (ms)", title="Temporal attribution")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].bar(["On / auto", "On / disabled", "Off / auto"],
                [comparisons[key]["samples_per_s"] for key in keys], color=colors)
    axes[2].set(ylabel="Samples / second", title="Timed windows; cleanup reported separately")
    for ax in axes:
        ax.grid(alpha=.15, axis="y")
    fig.savefig(OUT / "gc_probe.png", dpi=170)
    plt.close(fig)
    print(json.dumps(dict(summaries=comparisons, terminal=terminal,
                         gc_disabled_throughput_ratio=evidence["gc_disabled_throughput_ratio"],
                         paired_loss_max_abs=evidence["paired_loss_max_abs"],
                         paired_grad_norm_max_abs=evidence["paired_grad_norm_max_abs"]), indent=2))


if __name__ == "__main__":
    main()
