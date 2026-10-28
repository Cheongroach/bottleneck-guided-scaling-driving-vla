"""Four-rank bounded H20 communication probes on resident real cached specimens."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import default_collate

from train import CASES, CachedData, Policy, atomic_json, move, objective

SEED = 20261028
GLOBAL_BATCH = 64
MICROBATCH = 2
WARMUP = 6
REPEATS = 3
STEPS = 12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget():
    if time.time() > float(os.environ["DRIVING_BENCH_DEADLINE"]) - 70:
        raise TimeoutError("Probe deadline; reserve time to archive")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel-mode", choices=("auto", "8", "16"), required=True)
    args = parser.parse_args()
    world, rank, local_rank = (int(os.environ[k]) for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    assert world == 4 and torch.cuda.device_count() == 2
    accumulation = GLOBAL_BATCH // (world * MICROBATCH)
    assert GLOBAL_BATCH == world * MICROBATCH * accumulation
    torch.set_num_threads(2)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    local = Path(os.environ["DRIVING_LOCAL"])
    out = local / "results" / f"communication-{args.channel_mode}"
    out.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    data = CachedData(local / "cache", "train")
    indices = np.linspace(0, len(data) - 1, GLOBAL_BATCH, dtype=int).tolist()
    assert len(set(indices)) == GLOBAL_BATCH
    local_indices = indices[rank * 16:(rank + 1) * 16]
    batches = [move(default_collate([data[i] for i in local_indices[j:j + 2]]), device)
               for j in range(0, 16, 2)]
    policy = Policy(local / "model", CASES["lora_head"]).to(device).train()
    named = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    initial = {n: p.detach().clone() for n, p in named.items()}
    ddp = DDP(policy, device_ids=[local_rank], broadcast_buffers=False, bucket_cap_mb=25)
    host_boot = sha("/proc/sys/kernel/random/boot_id")
    env_names = ("NCCL_NET", "NCCL_NET_PLUGIN", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX",
                 "NCCL_IB_QPS_PER_CONNECTION", "NCCL_CROSS_NIC", "NCCL_MIN_NCHANNELS",
                 "NCCL_MAX_NCHANNELS", "NCCL_MIN_CTAS", "NCCL_MAX_CTAS",
                 "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE")
    environment = dict(
        rank=rank, world=world, local_rank=local_rank, host_boot_sha256=host_boot,
        gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        nccl_version=list(torch.cuda.nccl.version()) if isinstance(torch.cuda.nccl.version(), tuple)
        else torch.cuda.nccl.version(), transformers=__import__("transformers").__version__,
        global_batch=GLOBAL_BATCH, microbatch=MICROBATCH, accumulation=accumulation,
        seed=SEED, trainable_parameters=sum(p.numel() for p in named.values()),
        trainable_gradient_bytes=sum(p.numel() * p.element_size() for p in named.values()),
        specimen_indices=local_indices, cache_manifest_sha256=sha(local / "cache/manifest.json"),
        embeddings_shape=list(batches[0]["embeddings"].shape),
        valid_lengths=batches[0]["attention"].sum(-1).tolist(),
        network_env={k: os.environ[k] for k in env_names if k in os.environ},
        source_sha256={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        channel_mode=args.channel_mode, warmup=WARMUP, repeats=REPEATS, steps=STEPS,
        data_mode="64 distinct real train specimens, 16 per rank, resident on GPU; no timed data IO",
        timing_scope="Compute includes DDP backward and optimizer; tail includes statistics and control broadcast; full includes CUDA completion",
    )
    atomic_json(out / f"environment-rank{rank}.json", environment)
    # Synchronised, stand-alone collectives isolate latency without model overlap.
    collectives = []
    for count in (3, 6, environment["trainable_gradient_bytes"] // 4):
        tensor = torch.zeros(count, device=device, dtype=torch.float32)
        for _ in range(8):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        dist.barrier()
        timings = []
        for _ in range(40):
            tick = time.monotonic()
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            timings.append(time.monotonic() - tick)
        collectives.append(dict(bytes=count * 4, seconds=timings))
    atomic_json(out / f"collectives-rank{rank}.json", collectives)
    reference = None
    results = []
    control = torch.ones(1, device=device, dtype=torch.int32)
    trainable = list(named.values())
    for checkpointing, cadence in ((True, 1), (True, 16), (False, 1), (False, 16)):
        budget()
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        policy.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, parameter in named.items():
                parameter.copy_(initial[name])
        if checkpointing:
            policy.language.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            policy.language.gradient_checkpointing_disable()
        optimizer = torch.optim.AdamW([
            dict(params=[p for n, p in named.items() if n.startswith("head.")], lr=1e-3),
            dict(params=[p for n, p in named.items() if n.startswith("language.")], lr=3e-5),
        ], weight_decay=.01)
        case = f"checkpoint_{int(checkpointing)}_stats_{cadence}"
        rows, windows = [], []
        first_gradient = None
        torch.cuda.reset_peak_memory_stats()

        def step(index, warmup=False):
            nonlocal first_gradient
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            metrics = torch.zeros(3, device=device)
            for micro, batch in enumerate(batches):
                with ddp.no_sync() if micro < accumulation - 1 else nullcontext():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        predictions = ddp(batch)
                        loss, losses = objective(predictions, batch, CASES["lora_head"])
                    (loss / accumulation).backward()
                metrics += losses / accumulation
            if warmup and index == 0:
                first_gradient = torch.cat([p.grad.detach().flatten() for p in trainable]).cpu()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
            if not torch.isfinite(norm):
                raise RuntimeError(f"nonfinite gradient {case}/{index}")
            optimizer.step()
            torch.cuda.synchronize()
            compute_done = time.monotonic()
            stats = (index + 1) % cadence == 0
            if stats:
                durations = torch.tensor([compute_done - tick, 0., 0., 0., 0., 0.], device=device)
                dist.all_reduce(durations, op=dist.ReduceOp.MAX)
                dist.all_reduce(metrics)
                # Match original host copies after both small collectives.
                durations.tolist()
                metrics.tolist()
            dist.broadcast(control, src=0)
            torch.cuda.synchronize()
            end = time.monotonic()
            return dict(index=index, compute_s=compute_done - tick,
                        tail_s=end - compute_done, full_s=end - tick, statistics=stats)

        for index in range(WARMUP):
            step(index, warmup=True)
        torch.cuda.synchronize()
        dist.barrier()
        for repeat in range(REPEATS):
            budget()
            start = time.monotonic()
            for offset in range(STEPS):
                row = step(repeat * STEPS + offset)
                row["repeat"] = repeat
                rows.append(row)
            windows.append(dict(repeat=repeat, wall_s=time.monotonic() - start))
        final = torch.cat([p.detach().flatten() for p in trainable]).cpu()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = torch.cat([p.float().flatten() for p in policy(batches[0])]).cpu()
        assert torch.isfinite(final).all() and torch.isfinite(output).all()
        if reference is None:
            reference = dict(gradient=first_gradient, parameters=final, output=output)
        numerical = dict(
            gradient_max_abs_vs_checkpoint_stats1=float((first_gradient - reference["gradient"]).abs().max()),
            parameter_max_abs_vs_checkpoint_stats1=float((final - reference["parameters"]).abs().max()),
            output_max_abs_vs_checkpoint_stats1=float((output - reference["output"]).abs().max()),
            gradient_finite=bool(torch.isfinite(first_gradient).all()),
        )
        result = dict(case=case, checkpointing=checkpointing, statistics_cadence=cadence,
                      rows=rows, windows=windows, numerical=numerical, rank=rank,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        results.append(result)
        atomic_json(out / f"result-rank{rank}.json", dict(environment=environment, cases=results))
        dist.barrier()
        if rank == 0:
            print(json.dumps(dict(event="PROBE_CASE_DONE", mode=args.channel_mode,
                                  case=case, windows=windows, numerical=numerical)), flush=True)
        del optimizer
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
