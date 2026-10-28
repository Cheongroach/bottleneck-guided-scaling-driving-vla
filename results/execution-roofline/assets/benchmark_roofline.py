"""Bounded one-H20 matrix probe with kernel traces and optional NCU counters."""
import argparse
import ctypes
import gc
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

import torch


def atomic(path, data):
    path = Path(path)
    temporary = path.with_suffix(".upload")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def cases(config):
    h, f = config["hidden_size"], config["intermediate_size"]
    kv = h // config["num_attention_heads"] * config["num_key_value_heads"]
    result = {}
    for batch in (2, 4, 8):
        m = batch * 256
        for name, n, k, transpose in (
            ("q_forward", h, h, True), ("q_dx", h, h, False),
            ("kv_forward", kv, h, True),
            ("mlp_up", f, h, True), ("mlp_up_dx", h, f, False),
            ("mlp_down", h, f, True),
            ("lora_a", 8, h, True), ("lora_b", h, 8, True)):
            result[f"b{batch}_{name}"] = dict(m=m, n=n, k=k, transpose=transpose, batch=batch, op=name)
    result["dense_reference"] = dict(m=8192, n=8192, k=8192, transpose=True,
                                      batch=None, op="dense_reference")
    return result


def matrices(case):
    m, n, k = (case[x] for x in ("m", "n", "k"))
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k) if case["transpose"] else (k, n),
                         device="cuda", dtype=torch.bfloat16)
    b = weight.T if case["transpose"] else weight
    c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    return a, b, c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu-case")
    args = parser.parse_args()
    local = Path(os.environ["DRIVING_LOCAL"])
    out = local / "results/roofline"
    out.mkdir(exist_ok=True)
    config = json.loads((Path(__file__).parent / "roofline-model-config.json").read_text())
    all_cases = cases(config)
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    gc.disable()
    torch.manual_seed(20261028)
    if args.ncu_case:
        a, b, c = matrices(all_cases[args.ncu_case])
        for _ in range(8):
            torch.mm(a, b, out=c)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        torch.mm(a, b, out=c)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        print("NCU_TARGET_DONE", args.ncu_case, flush=True)
        return

    deadline = float(os.environ["DRIVING_BENCH_DEADLINE"])
    def budget(reserve=150):
        if time.time() > deadline - reserve:
            raise TimeoutError("One-GPU matrix probe budget reached")

    props = torch.cuda.get_device_properties(0)
    environment = dict(
        gpu=props.name, sm_count=props.multi_processor_count, total_memory=props.total_memory,
        torch=torch.__version__, world=1, distributed=False,
        model_config_sha256=hashlib.sha256((Path(__file__).parent / "roofline-model-config.json").read_bytes()).hexdigest(),
        config={k: config[k] for k in ("hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads")},
        dtype="dense BF16 GEMM, BF16 output; no sparse path; no TF32",
        generated_inputs=True, purpose="Real project shapes, synthetic input values; performance only",
        timing="CUDA event windows include inter-launch gaps; traces separately expose kernel durations",
        cold_cache="192MiB streaming read/write before every isolated call, excluded from that timer",
        scope="Single-host microbenchmarks, not complete training or cluster communication")
    atomic(out / "environment.json", environment)
    flush = torch.zeros(192 * 1024**2 // 4, device="cuda", dtype=torch.float32)
    results = []
    for key, case in all_cases.items():
        budget()
        a, b, c = matrices(case)
        for _ in range(12):
            torch.mm(a, b, out=c)
        torch.cuda.synchronize()
        warm_windows, cold_calls = [], []
        for repeat in range(3):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(24):
                torch.mm(a, b, out=c)
            end.record()
            end.synchronize()
            warm_windows.append(start.elapsed_time(end) / 24)
            for _ in range(12):
                flush.add_(1)
                torch.cuda.synchronize()
                start.record()
                torch.mm(a, b, out=c)
                end.record()
                end.synchronize()
                cold_calls.append(start.elapsed_time(end))
        flops = 2 * case["m"] * case["n"] * case["k"]
        algorithm_bytes = 2 * (case["m"] * case["k"] + case["k"] * case["n"] + case["m"] * case["n"])
        row = dict(key=key, **case, flops=flops, one_pass_algorithm_bytes=algorithm_bytes,
                   algorithm_intensity=flops / algorithm_bytes, warm_event_ms=warm_windows,
                   cold_event_ms=cold_calls, warm_event_mean_ms=statistics.mean(warm_windows),
                   cold_event_mean_ms=statistics.mean(cold_calls))
        results.append(row)
        # All operations in the trace carry an explicit host origin annotation.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=False) as profile:
            for mode in ("warm", "cold"):
                for index in range(4):
                    if mode == "cold":
                        flush.add_(1)
                        torch.cuda.synchronize()
                    with torch.profiler.record_function(f"ROOFLINE_{mode}_{index}"):
                        torch.mm(a, b, out=c)
                    torch.cuda.synchronize()
        profile.export_chrome_trace(str(out / f"{key}.trace.json"))
        print(json.dumps(dict(event="MATRIX_DONE", key=key,
                              warm_ms=row["warm_event_mean_ms"], cold_ms=row["cold_event_mean_ms"])), flush=True)
        del a, b, c, profile
    # Large streaming copy: algorithm counts a source read and destination write.
    source = torch.randn(256 * 1024**2 // 2, device="cuda", dtype=torch.bfloat16)
    target = torch.empty_like(source)
    copy_windows = []
    for _ in range(12):
        target.copy_(source)
    for _ in range(3):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(24):
            target.copy_(source)
        end.record()
        end.synchronize()
        copy_windows.append(start.elapsed_time(end) / 24)
    streaming = dict(bytes_per_call=2 * source.numel() * source.element_size(),
                     event_ms=copy_windows,
                     bandwidth_GB_s=[2 * source.numel() * source.element_size() / (ms / 1000) / 1e9
                                     for ms in copy_windows],
                     definition="Effective source+destination copy bandwidth; not direct DRAM counter bandwidth")
    atomic(out / "matrix_results.json", results)
    atomic(out / "streaming_copy.json", streaming)
    del source, target, flush

    paths = [shutil.which("ncu")] + glob.glob("/usr/local/cuda*/bin/ncu") + glob.glob("/opt/nvidia/nsight-compute/*/ncu")
    ncu = next((p for p in paths if p and Path(p).is_file()), None)
    profiler_status = dict(ncu=ncu, status="unavailable", profiles=[])
    if ncu:
        version = subprocess.run([ncu, "--version"], capture_output=True, text=True, timeout=15)
        profiler_status["version"] = version.stdout
        query = subprocess.run([ncu, "--query-metrics", "--query-metrics-mode", "suffix"],
                               capture_output=True, text=True, timeout=30)
        (out / "ncu-query-metrics.txt").write_text(query.stdout + query.stderr)
        candidates = (
            "gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "smsp__warps_eligible.avg.per_cycle_active",
            "smsp__warps_active.avg.per_cycle_active",
            "smsp__issue_active.avg.pct_of_peak_sustained_active",
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct")
        metrics = [m for m in candidates if m in query.stdout]
        profiler_status["metrics"] = metrics
        profiler_status["status"] = "queried"
        if not metrics:
            profiler_status["status"] = "metric_query_failed"
        for key in ("b2_q_forward", "b2_mlp_up", "b4_mlp_up", "b8_mlp_up",
                    "b2_lora_a", "b2_lora_b", "dense_reference") if metrics else ():
            if time.time() > deadline - 180:
                profiler_status["status"] = "budget_limited"
                break
            command = [ncu, "--profile-from-start", "off", "--cache-control", "all",
                       "--clock-control", "none", "--metrics", ",".join(metrics),
                       "--csv", "--page", "raw", "--target-processes", "all",
                       "python3", str(Path(__file__).resolve()), "--ncu-case", key]
            try:
                response = subprocess.run(command, capture_output=True, text=True, timeout=65)
                (out / f"{key}.ncu.csv").write_text(response.stdout)
                (out / f"{key}.ncu.stderr.txt").write_text(response.stderr)
                profiler_status["profiles"].append(dict(key=key, returncode=response.returncode))
                if "ERR_NVGPUCTRPERM" in response.stdout + response.stderr:
                    profiler_status["status"] = "counter_permission_denied"
                    break
                profiler_status["status"] = "collected" if response.returncode == 0 else "collection_failed"
                if response.returncode:
                    break
            except subprocess.TimeoutExpired:
                profiler_status["status"] = "collection_timeout"
                break
    atomic(out / "ncu_status.json", profiler_status)
    result = dict(environment=environment, matrices=results, streaming=streaming,
                  ncu=profiler_status, completed_epoch_s=time.time())
    atomic(out / "result.json", result)
    live = Path(os.environ["DRIVING_STORE"]) / "runs" / os.environ["DRIVING_RUN"] / "live"
    live.mkdir(exist_ok=True)
    atomic(live / "roofline.result.json", result)
    print(json.dumps(dict(event="ROOFLINE_COMPLETE", matrices=len(results),
                          ncu_status=profiler_status["status"])), flush=True)


if __name__ == "__main__":
    main()
