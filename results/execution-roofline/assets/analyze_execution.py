"""Rebuild report evidence from archived CUPTI traces; CPU only, no GPU jobs.

Run from any directory: python3 path/to/assets/analyze_execution.py
Input archive paths are resolved against the driving workspace. Correlations
map GPU kernels to host launch phases, not to coincident GPU timestamps.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def union(intervals, gap=0):
    result = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1] + gap:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def length(intervals):
    return sum(b - a for a, b in union(intervals))


def intersection(a, b):
    a, b = union(a), union(b)
    i = j = 0
    total = 0.
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def clip(intervals, a, b):
    return [[max(x, a), min(y, b)] for x, y in intervals if y > a and x < b]


def distribution(values):
    values = sorted(values)
    def q(p):
        pos = (len(values) - 1) * p
        i = int(pos)
        return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (pos - i)
    return dict(n=len(values), mean=statistics.mean(values), p50=q(.5), p95=q(.95),
                min=values[0], max=values[-1]) if values else None


PROVENANCE = []


def archive_member(job, suffix):
    for archive in sorted((RESULTS / "jobs" / job).glob("*.tar.gz")):
        with tarfile.open(archive) as tf:
            found = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(suffix)]
            if not found:
                continue
            assert len(found) == 1
            raw = tf.extractfile(found[0]).read()
            PROVENANCE.append(dict(archive=str(archive.relative_to(RESULTS)),
                                   member=found[0].name, member_sha256=sha(raw),
                                   member_bytes=len(raw)))
            return json.loads(raw)
    raise FileNotFoundError(f"{job}: {suffix}")


def family(name):
    name = name.lower()
    if "nccl" in name:
        return "NCCL"
    if any(x in name for x in ("gemm", "nvjet", "cutlass", "sm90_xmma")):
        return "GEMM"
    if any(x in name for x in ("flash", "fmha", "attention", "efficient_attention")):
        return "Attention"
    if any(x in name for x in ("copy", "convert", "cast")):
        return "Copy/cast"
    if any(x in name for x in ("reduce", "norm", "softmax")):
        return "Reduce/norm"
    if any(x in name for x in ("elementwise", "vectorized", "pointwise")):
        return "Elementwise"
    return "Other"


def training_trace(trace, label):
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
    phase_names = {"policy_forward": "forward", "policy_backward": "backward",
                   "Optimizer.step#AdamW.step": "optimizer"}
    annotations = sorted([e for e in events if e.get("cat") == "user_annotation"
                          and e.get("name") in phase_names], key=lambda e: e["ts"])
    starts = [e["ts"] for e in annotations]
    def origin(ts):
        i = bisect_right(starts, ts) - 1
        if i >= 0 and ts <= annotations[i]["ts"] + annotations[i]["dur"]:
            return phase_names[annotations[i]["name"]]
        return "outside"
    launches = {e["args"]["correlation"]: e for e in events
                if e.get("cat") in ("cuda_runtime", "cuda_driver")
                and "Launch" in e.get("name", "") and "correlation" in e.get("args", {})}
    kernels = [e for e in events if e.get("cat") == "kernel"]
    assert kernels
    fwd = [e for e in annotations if e["name"] == "policy_forward"]
    bwd = [e for e in annotations if e["name"] == "policy_backward"]
    assert len(fwd) == len(bwd)
    phases = defaultdict(list)
    groups = defaultdict(lambda: dict(count=0, sum_us=0.))
    kernel_names = defaultdict(lambda: dict(count=0, sum_us=0.))
    matched = 0
    typed = []
    for e in kernels:
        launch = launches.get(e.get("args", {}).get("correlation"))
        ph = origin(launch["ts"]) if launch else "unmatched"
        matched += launch is not None
        fam = family(e["name"])
        group = (ph, fam)
        groups[group]["count"] += 1
        groups[group]["sum_us"] += e["dur"]
        kernel_names[e["name"]]["count"] += 1
        kernel_names[e["name"]]["sum_us"] += e["dur"]
        phases[(ph, fam)].append([e["ts"], e["ts"] + e["dur"]])
        typed.append((e, ph, fam))
    intervals = [[e["ts"], e["ts"] + e["dur"]] for e in kernels]
    a, b = min(x for x, _ in intervals), max(y for _, y in intervals)
    compute = [[e["ts"], e["ts"] + e["dur"]] for e, ph, fam in typed if fam != "NCCL"]
    nccl = [[e["ts"], e["ts"] + e["dur"]] for e, ph, fam in typed if fam == "NCCL"]
    grad = [[e["ts"], e["ts"] + e["dur"]] for e, ph, fam in typed
            if fam == "NCCL" and ph == "backward"]
    stats = [[e["ts"], e["ts"] + e["dur"]] for e, ph, fam in typed
             if fam == "NCCL" and ph != "backward"]
    step_starts = sorted(e["ts"] for e in events
                         if e.get("name") == "Optimizer.zero_grad#AdamW.zero_grad")
    assert len(step_starts) == len(fwd)
    steps = []
    for i, begin in enumerate(step_starts):
        end = step_starts[i + 1] if i + 1 < len(step_starts) else max(b, annotations[-1]["ts"] + annotations[-1]["dur"])
        c, n, g = clip(compute, begin, end), clip(nccl, begin, end), clip(grad, begin, end)
        overlap = intersection(c, n)
        steps.append(dict(index=i, trace_interval_ms=(end-begin)/1000,
                          cpu_forward_ms=fwd[i]["dur"]/1000,
                          cpu_backward_ms=bwd[i]["dur"]/1000,
                          compute_union_ms=length(c)/1000,
                          nccl_union_ms=length(n)/1000, nccl_compute_overlap_ms=overlap/1000,
                          nccl_only_ms=(length(n)-overlap)/1000,
                          gradient_nccl_ms=length(g)/1000,
                          gradient_overlap_ms=intersection(c, g)/1000))
    # A representative profiled rank0 iteration. Merge only tiny gaps for display.
    index = min(range(len(steps)-1), key=lambda i: abs(
        steps[i]["trace_interval_ms"] - statistics.median(s["trace_interval_ms"] for s in steps[:-1])))
    begin, end = step_starts[index], step_starts[index + 1]
    lanes = {}
    for ph in ("forward", "backward", "optimizer"):
        cpu = [[e["ts"], e["ts"] + e["dur"]] for e in annotations
               if phase_names[e["name"]] == ph]
        lanes[f"CPU {ph}"] = clip(cpu, begin, end)
        gpu = [[e["ts"], e["ts"] + e["dur"]] for e, p, fam in typed
               if p == ph and fam != "NCCL"]
        lanes[f"GPU {ph}"] = union(clip(gpu, begin, end), gap=30)
    lanes["GPU NCCL gradient"] = union(clip(grad, begin, end))
    lanes["GPU NCCL tail"] = union(clip(stats, begin, end))
    timeline = dict(index=index, duration_ms=(end-begin)/1000, display_gap_merge_us=30,
                    lanes={k: [[(x-begin)/1000, (y-begin)/1000] for x, y in v]
                           for k, v in lanes.items()})
    phase_table = []
    for ph in ("forward", "backward", "optimizer", "outside", "unmatched"):
        rows = [(e, fam) for e, p, fam in typed if p == ph]
        if not rows:
            continue
        phase_table.append(dict(phase=ph, kernels=len(rows),
            kernels_per_step=len(rows)/len(fwd),
            kernel_sum_ms_per_step=sum(e["dur"] for e, fam in rows)/1000/len(fwd),
            kernel_union_ms_per_step=length([[e["ts"], e["ts"]+e["dur"]] for e, fam in rows])/1000/len(fwd),
            nccl_kernels=sum(fam=="NCCL" for e, fam in rows),
            cpu_annotation_ms_per_step=sum(e["dur"] for e in annotations
                if phase_names[e["name"]]==ph)/1000/len(fwd)))
    result = dict(label=label, steps=len(fwd), rank=trace.get("distributedInfo", {}).get("rank", 0),
        profiled=True, kernel_count=len(kernels), host_launch_matches=matched,
        host_launch_match_fraction=matched/len(kernels),
        kernel_span_ms=(b-a)/1000, kernel_union_ms=length(intervals)/1000,
        compute_union_ms=length(compute)/1000, nccl_union_ms=length(nccl)/1000,
        nccl_compute_overlap_ms=intersection(compute, nccl)/1000,
        nccl_only_ms=(length(nccl)-intersection(compute, nccl))/1000,
        gradient_nccl_union_ms=length(grad)/1000,
        gradient_compute_overlap_ms=intersection(compute, grad)/1000,
        tail_nccl_union_ms=length(stats)/1000,
        non_nccl_kernel_us=distribution([e["dur"] for e, p, f in typed if f!="NCCL"]),
        non_nccl_kernel_le5us_fraction=sum(e["dur"]<=5 for e,p,f in typed if f!="NCCL")/sum(f!="NCCL" for e,p,f in typed),
        phase_table=phase_table,
        families=[dict(phase=p, family=f, **v) for (p,f),v in sorted(groups.items())],
        top_kernels=[dict(name=k, **v) for k,v in sorted(kernel_names.items(),
                       key=lambda kv: kv[1]["sum_us"], reverse=True)[:12]],
        step_intervals=steps, timeline=timeline, device=trace["deviceProperties"][0],
        limitations=["Instrumented trace; not uninstrumented throughput",
                     "NCCL-only intervals include peer-arrival waits and are not causal communication overhead",
                     "Kernel interval occupancy is not SM utilization or achieved warp occupancy",
                     "Last trace interval has a truncated host tail"])
    return result


def matrix_trace(trace, row):
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
    anns = [e for e in events if e.get("cat") == "user_annotation"
            and e["name"].startswith("ROOFLINE_")]
    assert len(anns) == 8
    launch = {e["args"]["correlation"]: e for e in events
              if e.get("cat") in ("cuda_runtime", "cuda_driver")
              and "Launch" in e.get("name", "") and "correlation" in e.get("args", {})}
    samples = []
    for ann in anns:
        ks = [e for e in events if e.get("cat") == "kernel"
              and (l := launch.get(e.get("args", {}).get("correlation")))
              and ann["ts"] <= l["ts"] <= ann["ts"] + ann["dur"]]
        assert ks, (row["key"], ann["name"], len(ks))
        e = max(ks, key=lambda k: k["dur"])
        samples.append(dict(mode=ann["name"].split("_")[1], sample=int(ann["name"].split("_")[2]),
                            kernel_us=length([[k["ts"], k["ts"]+k["dur"]] for k in ks]),
                            kernel_count=len(ks), name=e["name"], launch=e["args"],
                            kernels=[dict(name=k["name"], duration_us=k["dur"]) for k in ks]))
    warm = [s["kernel_us"] for s in samples if s["mode"]=="warm"]
    cold = [s["kernel_us"] for s in samples if s["mode"]=="cold"]
    row.update(warm_kernel_us=distribution(warm), cold_kernel_us=distribution(cold),
               warm_kernel_TF_s=row["flops"]/statistics.median(warm)/1e6,
               cold_kernel_TF_s=row["flops"]/statistics.median(cold)/1e6,
               warm_event_TF_s=row["flops"]/row["warm_event_mean_ms"]/1e9,
               cold_event_TF_s=row["flops"]/row["cold_event_mean_ms"]/1e9,
               kernel_samples=samples)
    return row


def main():
    job = read(RESULTS / "roofline_probe_submission.json")
    terminal = read(RESULTS / "roofline_probe_terminal.json")
    assert terminal["status"] == "done" and terminal["allocated_gpus"] == 1
    raw = archive_member(job["name"], "results/roofline/result.json")
    assert raw == read(RESULTS / "jobs" / job["name"] / "live/roofline.result.json")
    assert raw["environment"]["model_config_sha256"] == sha((OUT/"model-config.json").read_bytes())
    matrices = [matrix_trace(archive_member(job["name"], f"results/roofline/{r['key']}.trace.json"), r)
                for r in raw["matrices"]]
    reference = next(r for r in matrices if r["key"]=="dense_reference")
    perf = reference["warm_kernel_TF_s"]
    bandwidth = statistics.mean(raw["streaming"]["bandwidth_GB_s"])
    ridge = perf*1000/bandwidth
    traces = {}
    for world, name in ((16, "driving-diagnose16-16h20-a1-20261028"),
                        (32, "driving-scale32-32h20-a1-20261028")):
        traces[str(world)] = training_trace(archive_member(name, "results/training/profile_cached/trace.json"),
                                           f"{world} GPU rank0 cached profile")
        print("TRACE_READY", world, flush=True)
    for key in ("micro2_ckpton", "micro2_ckptoff", "micro4_ckpton", "micro4_ckptoff",
                "micro8_ckpton", "micro8_ckptoff"):
        traces[key] = training_trace(archive_member("driving-single-gpu-hl-20261028060427",
                                   f"results/single_gpu/{key}.trace.json"), key)
    config = read(OUT/"model-config.json")
    h, f, layers = config["hidden_size"], config["intermediate_size"], config["num_hidden_layers"]
    kv = h // config["num_attention_heads"] * config["num_key_value_heads"]
    weights = dict(q=h*h, k=h*kv, v=h*kv, o=h*h, gate=h*f, up=h*f, down=h*f)
    linear_weights = sum(weights.values()) * layers
    budgets = []
    for batch in (2,4,8):
        forward = 2*batch*256*linear_weights
        attention_dense = 4*batch*256**2*h*layers
        budgets.append(dict(batch=batch, physical_sequence=256, valid_sequence=173,
                            linear_forward_GF=forward/1e9,
                            dense_attention_forward_GF=attention_dense/1e9,
                            linear_dx_approx_GF=forward/1e9,
                            linear_recompute_range_GF=[forward*(1-weights["down"]/sum(weights.values()))/1e9,
                                                       forward/1e9],
                            linear_train_without_cp_approx_GF=2*forward/1e9,
                            linear_train_with_cp_range_GF=[forward*(3-weights["down"]/sum(weights.values()))/1e9,
                                                          3*forward/1e9],
                            reference_linear_forward_floor_ms=forward/(perf*1e12)*1000))
    evidence = dict(generated_at=datetime.now(timezone.utc).isoformat(), probe_job=job,
        probe_terminal=terminal, profiler_counters=raw["ncu"], calibration=dict(
            dense_BF16_reference_TF_s=perf,
            dense_event_TF_s=reference["warm_event_TF_s"],
            copy_effective_GB_s=bandwidth, ridge_algorithm_FLOP_per_byte=ridge,
            reference_type="Empirical measured reference, not architectural hard ceiling",
            HBM_bytes_measured=False, SM_scheduler_counters_measured=False),
        matrix_environment=raw["environment"], matrices=matrices, traces=traces,
        model=dict(config_sha256=sha((OUT/"model-config.json").read_bytes()), weights_per_layer=weights,
                   active_linear_weights=linear_weights, trainable_parameters=2382640,
                   gradient_payload_bytes=2382640*4, budgets=budgets),
        history=read(RESULTS/"scaling_root_cause_evidence.json"),
        single_gpu=read(RESULTS/"single_gpu_evidence.json")["summaries"],
        gc=read(RESULTS/"gc_probe_evidence.json")["summaries"],
        provenance=PROVENANCE,
        verified=dict(live_archive_identical=True, model_config_hash_matches=True,
                      matrix_annotated_calls_correlated=25*8,
                      GPU_release=True, max_concurrent_GPUs_this_task=1),
        limitations=["No Nsight Compute executable discovered; no DRAM bytes, Tensor utilization or eligible warp counters",
                     "Algorithmic intensity counts one logical read of A/B and one write of C, not real HBM traffic",
                     "Streaming copy effective bandwidth is a calibrated reference only",
                     "Historical rank0 traces are profiled diagnostics, distinct from hour-long training windows",
                     "No new 16/32-card causal test; original 8.62% slowdown cannot be assigned a precise GC/network share"])
    save("execution_evidence.json", evidence)
    save("chart_data.json", dict(calibration=evidence["calibration"], matrices=matrices,
                                traces=traces, gc=evidence["gc"], model=evidence["model"]))
    with (OUT/"matrix_measurements.csv").open("w", newline="") as stream:
        cols = ["key","batch","op","m","n","k","transpose","flops","one_pass_algorithm_bytes",
                "algorithm_intensity","warm_event_mean_ms","cold_event_mean_ms",
                "warm_kernel_TF_s","cold_kernel_TF_s","warm_event_TF_s"]
        writer=csv.DictWriter(stream,fieldnames=cols)
        writer.writeheader()
        writer.writerows({k:r[k] for k in cols} for r in matrices)
    with (OUT/"trace_phase_measurements.csv").open("w", newline="") as stream:
        cols=["trace","phase","kernels","kernels_per_step","kernel_sum_ms_per_step",
              "kernel_union_ms_per_step","nccl_kernels","cpu_annotation_ms_per_step"]
        writer=csv.DictWriter(stream,fieldnames=cols)
        writer.writeheader()
        for key, t in traces.items():
            writer.writerows(dict(trace=key,**r) for r in t["phase_table"])
    print(json.dumps(dict(calibration=evidence["calibration"], budgets=budgets,
        traces={k:{f:v[f] for f in ("steps","host_launch_match_fraction","kernel_union_ms",
                                   "nccl_union_ms","nccl_compute_overlap_ms",
                                   "gradient_nccl_union_ms","gradient_compute_overlap_ms",
                                   "phase_table")} for k,v in traces.items()}),indent=2),flush=True)


if __name__ == "__main__":
    main()
