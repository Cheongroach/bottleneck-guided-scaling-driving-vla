"""One-GPU causal probe of periodic forward stalls and Python cyclic GC."""
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from cache_inputs import extract
from train import CASES, CachedData, Policy, atomic_json, emit, move, objective

STEPS = 96
WARMUP = 12
REPEATS = 3
SEED = 20261028


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    local = Path(os.environ["DRIVING_LOCAL"])
    out = local / "results/gc_probe"
    live = Path(os.environ["DRIVING_STORE"]) / "runs" / os.environ["DRIVING_RUN"] / "live"
    out.mkdir(exist_ok=True)
    live.mkdir(exist_ok=True)
    deadline = float(os.environ["DRIVING_BENCH_DEADLINE"])
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    device = torch.device("cuda", 0)
    for name in ("model.tar", "cache.tar"):
        extract(Path(os.environ["DRIVING_STORE"]) / "assets" / name, local)
    data = CachedData(local / "cache", "train")
    indices = np.linspace(0, len(data) - 1, 8, dtype=int).tolist()
    examples = [data[i] for i in indices]
    batches = [move(default_collate(examples[i:i + 2]), device) for i in range(0, 8, 2)]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    policy = Policy(local / "model", CASES["lora_head"]).to(device).train()
    named = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    initial = {n: p.detach().cpu().clone() for n, p in named.items()}
    environment = dict(
        world=1, distributed=False, gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, python=__import__("sys").version,
        transformers=__import__("transformers").__version__,
        cpu_threads=torch.get_num_threads(), gc_threshold=gc.get_threshold(),
        microbatch=2, steps_per_window=STEPS, warmup=WARMUP, repeats=REPEATS,
        cache_sha256=sha(local / "cache/manifest.json"), specimen_indices=indices,
        specimen_files=[dict(file=data.records[i]["cached_file"],
                             sha256=sha(local / "cache" / data.records[i]["cached_file"])) for i in indices],
        source_sha256={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        data_mode="8 resident real specimens; same batches and reset parameters in every window",
        scope="GC on/off causal performance probe; no DDP, no training-quality measurement")
    atomic_json(out / "environment.json", environment)
    emit("GC_PROBE_READY", **environment)
    state = dict(window="setup", step=-1, phase="setup", timed=False)
    active, collections, rows, windows = {}, [], [], []

    def callback(phase, info):
        generation = info["generation"]
        if phase == "start":
            active[generation] = (time.monotonic(), time.thread_time(), dict(state))
        else:
            tick, cpu_tick, context = active.pop(generation)
            collections.append(dict(
                **context, generation=generation, start_s=tick, end_s=time.monotonic(),
                wall_s=time.monotonic() - tick, thread_cpu_s=time.thread_time() - cpu_tick,
                collected=info["collected"], uncollectable=info["uncollectable"]))

    gc.callbacks.append(callback)

    def step(batch, optimizer):
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        state["phase"] = "forward"
        f0, cpu0 = time.monotonic(), time.thread_time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = policy(batch)
            loss, _ = objective(prediction, batch, CASES["lora_head"])
        f1, cpu1 = time.monotonic(), time.thread_time()
        state["phase"] = "backward"
        loss.backward()
        b1 = time.monotonic()
        state["phase"] = "optimizer"
        norm = torch.nn.utils.clip_grad_norm_(list(named.values()), 1.)
        optimizer.step()
        submitted = time.monotonic()
        state["phase"] = "sync"
        torch.cuda.synchronize()
        synced = time.monotonic()
        state["phase"] = "tail"
        values = dict(loss=float(loss.detach()), norm=float(norm.detach()))
        assert np.isfinite(list(values.values())).all()
        return dict(**values, start_s=tick, end_s=time.monotonic(),
                    wall_s=time.monotonic() - tick, cpu_forward_wall_s=f1 - f0,
                    forward_thread_cpu_s=cpu1 - cpu0, cpu_backward_wall_s=b1 - f1,
                    optimizer_wall_s=submitted - b1, sync_wait_s=synced - submitted)

    configs = (("ckpton_gcauto", True, True), ("ckpton_gcdisabled", True, False),
               ("ckptoff_gcauto", False, True))
    try:
        for repeat in range(REPEATS):
            order = configs if repeat % 2 == 0 else tuple(reversed(configs))
            for key, checkpointing, auto_gc in order:
                if time.time() > deadline - 180:
                    raise TimeoutError("Reserving time for evidence and archive")
                state.update(window=f"{repeat}_{key}", step=-1, phase="reset", timed=False)
                gc.enable()
                policy.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    for name, parameter in named.items():
                        parameter.copy_(initial[name])
                if checkpointing:
                    policy.language.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False})
                else:
                    policy.language.gradient_checkpointing_disable()
                policy.train()
                torch.manual_seed(SEED)
                torch.cuda.manual_seed_all(SEED)
                optimizer = torch.optim.AdamW([
                    dict(params=[p for n, p in named.items() if n.startswith("head.")], lr=1e-3),
                    dict(params=[p for n, p in named.items() if n.startswith("language.")], lr=3e-5),
                ], weight_decay=.01)
                gc.enable() if auto_gc else gc.disable()
                for i in range(WARMUP):
                    state.update(step=i, phase="warmup")
                    step(batches[i % len(batches)], optimizer)
                torch.cuda.reset_peak_memory_stats()
                state["timed"] = True
                offset = len(rows)
                tick = time.monotonic()
                for i in range(STEPS):
                    state["step"] = i
                    row = step(batches[i % len(batches)], optimizer)
                    rows.append(dict(key=key, repeat=repeat, step=i, **row))
                    if torch.cuda.memory_allocated() > 60 * 1024**3:
                        raise RuntimeError("GC-disabled window exceeds bounded GPU memory")
                window_s = time.monotonic() - tick
                state.update(timed=False, phase="manual_cleanup", step=-1)
                finished = rows[offset:]
                windows.append(dict(
                    key=key, repeat=repeat, steps=STEPS, window_wall_s=window_s,
                    samples_per_s=STEPS * 2 / window_s,
                    p50_ms=float(np.median([r["wall_s"] * 1000 for r in finished])),
                    p95_ms=float(np.percentile([r["wall_s"] * 1000 for r in finished], 95)),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    head_sum=float(next(p for n, p in named.items() if n == "head.3.weight").sum()),
                    gc_automatic=auto_gc, checkpointing=checkpointing))
                emit("GC_PROBE_WINDOW", **windows[-1])
                del optimizer, finished, row
                policy.zero_grad(set_to_none=True)
                gc.enable()
                gc.collect()
                atomic_json(out / "steps.json", rows)
                atomic_json(out / "gc_events.json", collections)
                atomic_json(out / "windows.json", windows)
        head_sum_deltas = []
        for repeat in range(REPEATS):
            selected = [w for w in windows if w["repeat"] == repeat]
            head_sum_deltas.append(max(w["head_sum"] for w in selected) -
                                   min(w["head_sum"] for w in selected))
        result = dict(environment=environment, windows=windows,
                      python_wall_s=time.monotonic() - started, completed_epoch_s=time.time(),
                      head_sum_range_per_repeat=head_sum_deltas,
                      numerical_check="Finite loss/gradients; head.3.weight sum recorded; not full gradient equality")
        atomic_json(out / "result.json", result)
        atomic_json(live / "gc_probe.result.json", result)
        emit("GC_PROBE_COMPLETE", python_wall_s=result["python_wall_s"])
    finally:
        gc.enable()
        gc.callbacks.remove(callback)


if __name__ == "__main__":
    main()
