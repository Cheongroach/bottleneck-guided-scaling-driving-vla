"""Bounded same-host H20 execution probes; performance, not convergence."""
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from train import CASES, CachedData, Policy, atomic_json, emit, move, objective
from cache_inputs import extract

SEED = 20261028
WARMUP = 12
WINDOW_STEPS = 24
REPEATS = 3
TRACE_STEPS = 4


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    return dict(mean=float(np.mean(values)), p50=float(np.median(values)),
                p95=float(np.quantile(values, .95)), minimum=float(np.min(values)),
                maximum=float(np.max(values)))


def main():
    started = time.monotonic()
    local = Path(os.environ["DRIVING_LOCAL"])
    out = local / "results" / "single_gpu"
    live = Path(os.environ["DRIVING_STORE"]) / "runs" / os.environ["DRIVING_RUN"] / "live"
    out.mkdir(exist_ok=True)
    live.mkdir(exist_ok=True)
    hard_deadline = float(os.environ["DRIVING_BENCH_DEADLINE"])

    def check_budget():
        if time.time() > hard_deadline - 90:
            raise TimeoutError("Leaving 90 seconds for evidence and archive")

    assert torch.cuda.device_count() == 1, "This probe is authorized for one GPU only"
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    device = torch.device("cuda", 0)
    extract_start = time.monotonic()
    extract(Path(os.environ["DRIVING_STORE"]) / "assets/model.tar", local)
    extract(Path(os.environ["DRIVING_STORE"]) / "assets/cache.tar", local)
    extract_s = time.monotonic() - extract_start
    check_budget()
    data = CachedData(local / "cache", "train")
    indices = np.linspace(0, len(data) - 1, 8, dtype=int).tolist()
    examples = [data[i] for i in indices]
    batches = {
        micro: [move(default_collate(examples[offset:offset + micro]), device)
                for offset in range(0, 8, micro)] for micro in (2, 4, 8)
    }
    manifest = json.loads((local / "cache/manifest.json").read_text())
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    setup_start = time.monotonic()
    policy = Policy(local / "model", CASES["lora_head"]).to(device).train()
    named = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    initial = {n: p.detach().cpu().clone() for n, p in named.items()}
    frozen = next(p for n, p in policy.named_parameters()
                  if not p.requires_grad and "q_proj" in n and n.endswith("weight"))
    frozen_before = frozen.detach().flatten()[:8192].cpu().clone()
    setup_s = time.monotonic() - setup_start
    environment = dict(
        world=1, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        transformers=__import__("transformers").__version__,
        seed=SEED, model_revision=(local / "model/MODEL_REVISION").read_text().strip(),
        cache_manifest_sha256=digest(local / "cache/manifest.json"),
        source_sha256={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        config=CASES["lora_head"], trainable_parameters=sum(p.numel() for p in named.values()),
        specimen_indices=indices,
        specimen_files=[dict(file=data.records[i]["cached_file"],
                             sha256=digest(local / "cache" / data.records[i]["cached_file"]))
                        for i in indices],
        specimen_tokens=[data.records[i]["token"] for i in indices],
        embeddings_shape=list(batches[8][0]["embeddings"].shape),
        valid_lengths=batches[8][0]["attention"].sum(-1).tolist(),
        cache_sequence_lengths=manifest["sequence_lengths"],
        data_mode="Eight real cached train specimens, resident on GPU; no dataloader IO in timing",
        dtype="bf16 frozen backbone/autocast, fp32 LoRA and head", attention="sdpa",
        distributed=False, extract_s=extract_s, model_setup_s=setup_s,
        warmup_steps=WARMUP, window_steps=WINDOW_STEPS, repeats=REPEATS,
        trace_steps=TRACE_STEPS, cpu_threads=torch.get_num_threads(),
        timing_scope="Single-rank optimizer micro-step; full wall includes CUDA completion and host metric copies; no DDP/collectives",
    )
    emit("SINGLE_GPU_READY", **environment)
    atomic_json(out / "environment.json", environment)

    def restore(checkpointing):
        policy.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n, p in named.items():
                p.copy_(initial[n])
        if checkpointing:
            policy.language.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            policy.language.gradient_checkpointing_disable()
        policy.train()
        assert policy.training and policy.language.training
        assert bool(policy.language.gradient_checkpointing) == checkpointing
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.synchronize()

    # Aggregate gradients for the same eight specimens and parameters. These
    # values check execution semantics, not one-hour optimization or ADE/FDE.
    sanity = {}
    sanity_gradients = {}
    for micro in (2, 4, 8):
        for checkpointing in (True, False):
            check_budget()
            restore(checkpointing)
            predictions, losses = [], []
            for batch in batches[micro]:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = policy(batch)
                    loss, _ = objective(prediction, batch, CASES["lora_head"])
                predictions.append(prediction[0].detach().float().cpu())
                losses.append(float(loss.detach()) * micro / 8)
                (loss * micro / 8).backward()
            grads = torch.cat([named[n].grad.detach().flatten().cpu() for n in sorted(named)])
            assert torch.isfinite(grads).all()
            key = f"micro{micro}_ckpt{'on' if checkpointing else 'off'}"
            sanity_gradients[key] = grads
            sanity[key] = dict(loss=sum(losses), gradient_norm=float(grads.norm()),
                               trajectory=torch.cat(predictions).tolist())
    for micro in (2, 4, 8):
        left, right = (f"micro{micro}_ckpt{mode}" for mode in ("on", "off"))
        a, b = sanity_gradients[left], sanity_gradients[right]
        rel = float((a - b).norm() / a.norm().clamp_min(1e-12))
        prediction_delta = float(np.max(np.abs(
            np.array(sanity[left]["trajectory"]) - np.array(sanity[right]["trajectory"]))))
        sanity[right].update(gradient_relative_l2_vs_checkpoint_on=rel,
                             trajectory_max_abs_m_vs_checkpoint_on=prediction_delta,
                             loss_abs_delta_vs_checkpoint_on=abs(
                                 sanity[right]["loss"] - sanity[left]["loss"]))
        assert rel < 1e-5 and prediction_delta < 1e-5, "Checkpoint toggle changed semantics"
    for key, grads in sanity_gradients.items():
        ref = sanity_gradients["micro8_ckpton"]
        sanity[key]["gradient_relative_l2_vs_micro8"] = float(
            (grads - ref).norm() / ref.norm().clamp_min(1e-12))
    del sanity_gradients
    atomic_json(out / "numerical_sanity.json", sanity)
    emit("NUMERICAL_SANITY_COMPLETE",
         checkpoint_gradient_delta={k: v.get("gradient_relative_l2_vs_checkpoint_on")
                                    for k, v in sanity.items() if k.endswith("off")})

    def optimizer():
        return torch.optim.AdamW([
            dict(params=[p for n, p in named.items() if n.startswith("head.")], lr=1e-3),
            dict(params=[p for n, p in named.items() if n.startswith("language.")], lr=3e-5),
        ], weight_decay=.01)

    def step(batch, optim, instrumented=False):
        t0 = time.monotonic()
        optim.zero_grad(set_to_none=True)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
        f0 = time.monotonic()
        with torch.profiler.record_function("policy_forward"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = policy(batch)
                loss, _ = objective(prediction, batch, CASES["lora_head"])
        f1 = time.monotonic()
        events[1].record()
        with torch.profiler.record_function("policy_backward"):
            loss.backward()
        b1 = time.monotonic()
        events[2].record()
        norm = torch.nn.utils.clip_grad_norm_(list(named.values()), 1.)
        optim.step()
        events[3].record()
        submitted = time.monotonic()
        torch.cuda.synchronize()
        synced = time.monotonic()
        row = dict(
            local_compute_wall_s=synced - t0,
            cpu_forward_s=f1 - f0, cpu_backward_s=b1 - f1,
            cpu_optimizer_s=submitted - b1, cpu_sync_wait_s=synced - submitted,
            gpu_forward_elapsed_s=events[0].elapsed_time(events[1]) / 1000,
            gpu_backward_elapsed_s=events[1].elapsed_time(events[2]) / 1000,
            gpu_optimizer_elapsed_s=events[2].elapsed_time(events[3]) / 1000,
            gpu_step_elapsed_s=events[0].elapsed_time(events[3]) / 1000,
            loss=float(loss.detach()), gradient_norm=float(norm.detach()),
            instrumented=instrumented,
        )
        if not np.isfinite([row["loss"], row["gradient_norm"]]).all():
            raise RuntimeError("Nonfinite optimizer probe")
        row["full_step_wall_s"] = time.monotonic() - t0
        row["tail_host_wall_s"] = row["full_step_wall_s"] - row["local_compute_wall_s"]
        return row

    results, all_steps = [], []
    # Pair checkpoint states within each microbatch, then interleave/reverse
    # config order across repeats to reduce monotonic temperature/host bias.
    configs = [(m, c) for m in (2, 4, 8) for c in (True, False)]
    for repeat in range(REPEATS):
        order = configs if repeat % 2 == 0 else list(reversed(configs))
        for micro, checkpointing in order:
            check_budget()
            restore(checkpointing)
            optim = optimizer()
            for i in range(WARMUP):
                step(batches[micro][i % len(batches[micro])], optim)
            torch.cuda.reset_peak_memory_stats()
            rows = []
            wall0 = time.monotonic()
            for i in range(WINDOW_STEPS):
                rows.append(step(batches[micro][i % len(batches[micro])], optim))
            window_s = time.monotonic() - wall0
            key = f"micro{micro}_ckpt{'on' if checkpointing else 'off'}"
            result = dict(
                key=key, microbatch=micro, checkpointing=checkpointing, repeat=repeat,
                steps=WINDOW_STEPS, samples=WINDOW_STEPS * micro,
                window_wall_s=window_s, samples_per_s=WINDOW_STEPS * micro / window_s,
                step_wall_s=summary([r["full_step_wall_s"] for r in rows]),
                gpu_step_elapsed_s=summary([r["gpu_step_elapsed_s"] for r in rows]),
                cpu_forward_s=summary([r["cpu_forward_s"] for r in rows]),
                cpu_backward_s=summary([r["cpu_backward_s"] for r in rows]),
                tail_host_wall_s=summary([r["tail_host_wall_s"] for r in rows]),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
            results.append(result)
            all_steps.extend([dict(key=key, repeat=repeat, step=i, **row)
                              for i, row in enumerate(rows)])
            atomic_json(out / "windows.json", results)
            atomic_json(out / "steps.json", all_steps)
            emit("SINGLE_GPU_WINDOW", **result)
            del optim
            policy.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

    # Traces are deliberately separate from the uninstrumented throughput.
    for micro, checkpointing in configs:
        check_budget()
        restore(checkpointing)
        optim = optimizer()
        for i in range(WARMUP):
            step(batches[micro][i % len(batches[micro])], optim)
        key = f"micro{micro}_ckpt{'on' if checkpointing else 'off'}"
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False,
        ) as profile:
            for i in range(TRACE_STEPS):
                step(batches[micro][i % len(batches[micro])], optim, instrumented=True)
                profile.step()
        profile.export_chrome_trace(str(out / f"{key}.trace.json"))
        emit("SINGLE_GPU_TRACE_COMPLETE", key=key)
        del optim, profile
        policy.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    frozen_unchanged = torch.equal(frozen.detach().flatten()[:8192].cpu(), frozen_before)
    assert frozen_unchanged
    complete = dict(environment=environment, windows=results, numerical_sanity=sanity,
                    frozen_weight_sample_unchanged=frozen_unchanged,
                    completed_epoch_s=time.time(), python_wall_s=time.monotonic() - started,
                    experiment_scope="Performance only; microbatch is optimizer batch, not global batch 64; no distributed communication measured")
    atomic_json(out / "result.json", complete)
    atomic_json(live / "single_gpu.result.json", complete)
    emit("SINGLE_GPU_COMPLETE", python_wall_s=complete["python_wall_s"], windows=len(results))


if __name__ == "__main__":
    main()
