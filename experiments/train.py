"""Matched-batch DDP profiles and scene-isolated driving policy probes on H20."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler

from contracts import CAMERAS, HORIZON, action_class
from cache_inputs import extract

CASES = {
    "trajectory_only": dict(action=0., rationale=0., image=True, state=True, lora=False),
    "trajectory_action": dict(action=.2, rationale=0., image=True, state=True, lora=False),
    "frozen_head": dict(action=.2, rationale=.1, image=True, state=True, lora=False),
    "state_only": dict(action=.2, rationale=.1, image=False, state=True, lora=False),
    "image_only": dict(action=.2, rationale=.1, image=True, state=False, lora=False),
    "lora_head": dict(action=.2, rationale=.1, image=True, state=True, lora=True),
    "lora_no_rationale": dict(action=.2, rationale=0., image=True, state=True, lora=True),
}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".upload")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def emit(event, **values):
    if int(os.environ.get("RANK", 0)) == 0:
        print(json.dumps(dict(event=event, **values)), flush=True)


class LoRALinear(nn.Module):
    """Actual low-rank q/v adapters; all original VLM weights stay frozen."""
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = 2.

    def forward(self, values):
        return self.base(values) + (
            (values.to(self.a.dtype) @ self.a.T) @ self.b.T).to(values.dtype) * self.scale


class Policy(nn.Module):
    def __init__(self, model_path, config):
        super().__init__()
        from transformers import Qwen2_5_VLForConditionalGeneration
        backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        self.language = backbone.model
        hidden = backbone.config.hidden_size
        del backbone
        self.language.requires_grad_(False)
        self.language.config.use_cache = False
        if config["lora"]:
            for layer in self.language.layers:
                for name in ("q_proj", "v_proj"):
                    base = getattr(layer.self_attn, name)
                    setattr(layer.self_attn, name, LoRALinear(base))
            self.language.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        self.head = nn.Sequential(nn.LayerNorm(hidden+6), nn.Linear(hidden+6, 256),
                                  nn.GELU(), nn.Linear(256, 12+9+9+6))
        self.config = config

    def forward(self, batch, remove_camera=None):
        inputs = batch["embeddings"].to(torch.bfloat16)
        mask = batch["camera_masks"].any(1) if not self.config["image"] else None
        if remove_camera is not None:
            intervention = batch["camera_masks"][:, remove_camera]
            mask = intervention if mask is None else mask | intervention
        if mask is not None:
            inputs = inputs.masked_fill(mask[..., None], 0)
        with torch.set_grad_enabled(self.config["lora"] and self.training):
            hidden = self.language(inputs_embeds=inputs,
                attention_mask=batch["attention"], position_ids=batch["positions"].permute(1, 0, 2),
                use_cache=False, return_dict=True).last_hidden_state
            last = batch["attention"].sum(-1)-1
            feature = hidden[torch.arange(len(hidden), device=hidden.device), last].float()
        state = batch["state"].float() if self.config["state"] else torch.zeros_like(batch["state"])
        normalized = state / state.new_tensor([15, 5, 5, 5, 1, 15])
        prediction = self.head(torch.cat([feature, normalized], -1))
        trajectory = prediction[:, :12].reshape(-1, 6, 2) * 10
        # A transparent constant-velocity residual baseline, removed in image-only.
        trajectory = trajectory + state[:, None, :2] * state.new_tensor(HORIZON)[None, :, None]
        return trajectory, prediction[:, 12:21], prediction[:, 21:30], prediction[:, 30:36]


def restore_policy(model_path, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    policy = Policy(model_path, checkpoint["config"]).to(device).eval()
    parameters = dict(policy.named_parameters())
    expected = {n for n, p in parameters.items() if p.requires_grad}
    assert expected == set(checkpoint["trainable_state"]), "Checkpoint architecture does not match"
    with torch.no_grad():
        for name, value in checkpoint["trainable_state"].items():
            parameters[name].copy_(value.to(device=device, dtype=parameters[name].dtype))
    return policy, checkpoint


class CachedData(Dataset):
    def __init__(self, root, split, synthetic=False):
        self.root = Path(root)
        self.records = [r for r in json.loads((self.root/"records.json").read_text()) if r["split"] == split]
        self.synthetic = synthetic
        self.random_cache = None
        if synthetic:
            example = torch.load(self.root/self.records[0]["cached_file"], weights_only=True)
            generator = torch.Generator().manual_seed(4001)
            self.random_cache = {**example,
                "embeddings": torch.randn(example["embeddings"].shape, generator=generator).half()}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        values = (self.random_cache if self.synthetic else
                  torch.load(self.root/row["cached_file"], weights_only=True))
        teacher = row["teacher"]
        if self.synthetic:
            rng = np.random.default_rng(index+991)
            trajectory = rng.normal(0, 10, (6, 2)).astype(np.float32)
            action = int(rng.integers(9))
        else:
            trajectory, action = row["trajectory"], row["action"]
        return dict(**values, state=torch.tensor(row["state"], dtype=torch.float32),
            trajectory=torch.tensor(trajectory, dtype=torch.float32),
            action=torch.tensor(action), teacher_valid=torch.tensor(teacher is not None),
            teacher_action=torch.tensor(teacher["action"] if teacher else 0),
            teacher_evidence=torch.tensor(teacher["evidence"] if teacher else [0.]*6),
            record_index=torch.tensor(index))


class InfiniteSampler(Sampler):
    def __init__(self, count, rank, world, microbatch, seed):
        self.count, self.rank, self.world, self.microbatch, self.seed = count, rank, world, microbatch, seed

    def __iter__(self):
        chunk = 0
        permutations = {}
        while True:
            start = chunk*self.world*self.microbatch+self.rank*self.microbatch
            for offset in range(self.microbatch):
                epoch, index = divmod(start+offset, self.count)
                if epoch not in permutations:
                    permutations = {epoch: np.random.default_rng(self.seed+epoch).permutation(self.count)}
                yield int(permutations[epoch][index])
            chunk += 1


def move(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def objective(prediction, batch, config):
    trajectory, action, rationale, evidence = prediction
    trajectory_loss = nn.functional.smooth_l1_loss(trajectory.float()/10, batch["trajectory"]/10)
    action_loss = nn.functional.cross_entropy(action.float(), batch["action"])
    valid = batch["teacher_valid"].float()
    explanation = (nn.functional.cross_entropy(rationale.float(), batch["teacher_action"], reduction="none") +
        nn.functional.binary_cross_entropy_with_logits(evidence.float(), batch["teacher_evidence"],
                                                       reduction="none").mean(-1))
    rationale_loss = (explanation*valid).mean()
    # Every head participates in the graph; zero weights denote an explicit ablation.
    loss = trajectory_loss + config["action"]*action_loss + config["rationale"]*rationale_loss
    return loss, torch.stack([trajectory_loss.detach(), action_loss.detach(), rationale_loss.detach()])


@torch.inference_mode()
def evaluate(policy, dataset, device, rank, world, microbatch, explain=False):
    policy.eval()
    totals = torch.zeros(9, device=device, dtype=torch.float64)
    details, latencies = [], []
    loader = DataLoader(dataset, batch_size=microbatch,
        sampler=list(range(rank, len(dataset), world)), num_workers=0)
    started = time.monotonic()
    for batch in loader:
        batch = move(batch, device)
        torch.cuda.synchronize()
        tick = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            trajectory, action, rationale, evidence = policy(batch)
        torch.cuda.synchronize()
        latencies.append((time.monotonic()-tick)/len(trajectory))
        distances = (trajectory.float()-batch["trajectory"]).norm(dim=-1)
        valid = batch["teacher_valid"]
        totals[0] += distances.mean(-1).sum()
        totals[1] += distances[:, -1].sum()
        totals[2] += (distances[:, -1] > 2).sum()
        totals[3] += (action.argmax(-1) == batch["action"]).sum()
        totals[4] += ((rationale.argmax(-1) == action.argmax(-1)) & valid).sum()
        totals[5] += ((evidence.sigmoid() > .5) == batch["teacher_evidence"].bool()).float().mean(-1)[valid].sum()
        totals[6] += len(trajectory)
        totals[7] += valid.sum()
        for local_index, record_index in enumerate(batch["record_index"].tolist()):
            row = dataset.records[record_index]
            predicted = trajectory[local_index].float().cpu().numpy()
            collision = any(any(np.linalg.norm(predicted[t]-obj[:2]) < obj[2] for obj in occupied)
                            for t, occupied in enumerate(row["future_occupied_circles"]))
            totals[8] += collision
            details.append(dict(token=row["token"], scene=row["scene"],
                prediction=predicted.tolist(), target=row["trajectory"],
                ADE=float(distances[local_index].mean()), FDE=float(distances[local_index, -1]),
                action=int(action[local_index].argmax()), target_action=row["action"],
                rationale_action=int(rationale[local_index].argmax()),
                evidence=evidence[local_index].sigmoid().float().cpu().tolist(),
                teacher=row["teacher"], teacher_text=row.get("teacher_text"),
                collision_proxy=collision))
    dist.all_reduce(totals)
    gathered = [None]*world
    dist.all_gather_object(gathered, details)
    latency_gathered = [None]*world
    dist.all_gather_object(latency_gathered, latencies)
    elapsed = torch.tensor(time.monotonic()-started, device=device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    n, valid_count = float(totals[6]), float(totals[7])
    latency = [v for values in latency_gathered for v in values]
    metrics = dict(ADE_m=float(totals[0]/n), FDE_m=float(totals[1]/n),
        miss_rate_2m=float(totals[2]/n), action_accuracy=float(totals[3]/n),
        rationale_action_consistency=float(totals[4]/valid_count) if valid_count else None,
        evidence_flag_accuracy=float(totals[5]/valid_count) if valid_count else None,
        teacher_valid=int(valid_count), samples=int(n), collision_proxy=float(totals[8]/n),
        lane_violation_proxy=None, eval_s=float(elapsed),
        cached_policy_latency_p50_ms=float(np.median(latency)*1000),
        cached_policy_latency_p95_ms=float(np.percentile(latency, 95)*1000),
        latency_scope="cached embeddings; batch latency / samples; excludes vision/teacher")
    explanation = None
    if explain:
        tick = time.monotonic()
        changes, hits, count, teacher_count = 0, 0, 0, 0
        # One rank-strided subset, all camera interventions, and original dimensions.
        selected = list(range(rank, min(len(dataset), 48), world))
        for record_index in selected:
            batch = move({k: v.unsqueeze(0) for k, v in dataset[record_index].items()}, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                original = policy(batch)
                differences = []
                for camera in range(6):
                    changed = policy(batch, remove_camera=camera)
                    changes += int(changed[1].argmax() != original[1].argmax())
                    differences.append(float((changed[0]-original[0]).float().norm(dim=-1).mean()))
            teacher = dataset.records[record_index]["teacher"]
            if teacher:
                teacher_count += 1
                hits += int(teacher["evidence"][int(np.argmax(differences))] > 0)
            count += 1
            for row in details:
                if row["token"] == dataset.records[record_index]["token"]:
                    row["camera_ablation_trajectory_change_m"] = differences
        torch.cuda.synchronize()
        summary = torch.tensor([changes, hits, count, teacher_count, time.monotonic()-tick], device=device, dtype=torch.float64)
        dist.all_reduce(summary[:4])
        dist.all_reduce(summary[4:], op=dist.ReduceOp.MAX)
        explanation = dict(counterfactual_action_flip_rate=float(summary[0]/(summary[2]*6)),
            camera_importance_teacher_hit_rate=float(summary[1]/summary[3]) if summary[3] else None,
            examples=int(summary[2]), teacher_valid=int(summary[3]),
            interventions=int(summary[2]*6), explain_s=float(summary[4]),
            intervention="zero one camera's cached visual tokens; not a physically realistic counterfactual")
        dist.all_gather_object(gathered, details)
    policy.train()
    return metrics, [row for values in gathered for row in values], explanation


def interval_union(intervals):
    total, end = 0., float("-inf")
    for start, stop in sorted(intervals):
        if stop > end:
            total += stop-max(start, end)
            end = stop
    return total


def trace_summary(path, wall_seconds):
    events = json.loads(Path(path).read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    all_intervals = [(e["ts"], e["ts"]+e["dur"]) for e in kernels]
    communication = [e for e in kernels if "nccl" in e.get("name", "").lower()]
    comm_intervals = [(e["ts"], e["ts"]+e["dur"]) for e in communication]
    return dict(kernel_count=len(kernels), nccl_kernel_count=len(communication),
        gpu_kernel_union_s=interval_union(all_intervals)/1e6,
        nccl_union_s=interval_union(comm_intervals)/1e6,
        nccl_wall_fraction=interval_union(comm_intervals)/1e6/max(wall_seconds, 1e-9),
        nccl_kernel_fraction=interval_union(comm_intervals)/max(interval_union(all_intervals), 1e-9),
        fraction_definition="union of rank0 NCCL CUDA kernel intervals / measured profile wall time; overlap counted once")


def run_case(args, name, config, train, val, device, rank, world, seconds, mode="train", global_batch=None):
    global_batch = global_batch or args.global_batch
    assert global_batch % (world*args.microbatch) == 0
    accumulation = global_batch//(world*args.microbatch)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    setup_start = time.monotonic()
    policy = Policy(args.model, config).to(device).train()
    checkpointing = bool(getattr(policy.language, "gradient_checkpointing", False))
    assert policy.training and policy.language.training, "Profile and training modes must match"
    assert checkpointing == config["lora"], "Unexpected language activation-checkpointing configuration"
    trainable = [p for p in policy.parameters() if p.requires_grad]
    lora = [p for n, p in policy.named_parameters() if p.requires_grad and n.startswith("language.")]
    head = [p for n, p in policy.named_parameters() if p.requires_grad and n.startswith("head.")]
    before = {n: p.detach().cpu().clone() for n, p in policy.named_parameters() if p.requires_grad}
    frozen_name, frozen_parameter = next((n, p) for n, p in policy.named_parameters()
                                         if not p.requires_grad and "q_proj" in n and n.endswith("weight"))
    frozen_before = frozen_parameter.detach().flatten()[:8192].float().cpu().clone()
    ddp = DDP(policy, device_ids=[args.local_rank], broadcast_buffers=False, bucket_cap_mb=25)
    optimizer = torch.optim.AdamW([dict(params=head, lr=1e-3, base_lr=1e-3),
                                  dict(params=lora, lr=3e-5, base_lr=3e-5)], weight_decay=.01)
    environment = dict(case=name, world=world, hosts=args.hosts, global_batch=global_batch,
        microbatch=args.microbatch, accumulation=accumulation, seed=args.seed, config=config,
        trainable_parameters=sum(p.numel() for p in trainable), lora_parameters=sum(p.numel() for p in lora),
        total_parameters=sum(p.numel() for p in policy.parameters()),
        cache_manifest_sha256=args.data_hash,
        torch=torch.__version__, device=torch.cuda.get_device_name(),
        dtype="bf16 frozen backbone and autocast, fp32 trainable adapters/heads",
        policy_training=policy.training, language_training=policy.language.training,
        gradient_checkpointing_enabled=checkpointing,
        gradient_checkpointing_active=checkpointing and policy.language.training,
        attention="sdpa", setup_s=time.monotonic()-setup_start,
        train_samples=len(train), val_samples=len(val), mode=mode)
    emit("CASE_START", **environment)
    out = args.output/name
    if rank == 0:
        out.mkdir(exist_ok=True)
    dist.barrier()
    initial, _, _ = evaluate(policy, val, device, rank, world, args.microbatch) if mode == "train" else ({}, [], None)
    best_ADE, best_step = initial.get("ADE_m", float("inf")), 0
    best_state = {n: p.detach().cpu().clone() for n, p in policy.named_parameters() if p.requires_grad}
    loader = DataLoader(train, batch_size=args.microbatch,
        sampler=InfiniteSampler(len(train), rank, world, args.microbatch, args.seed),
        num_workers=2, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    iterator = iter(loader)
    trace = None
    if mode != "train" and rank == 0:
        trace = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=False)
    rows, curves = [], [dict(step=0, wall_s=0., samples_seen=0, GPU_hours=0., **initial)] if initial else []
    total_eval, total_ckpt = initial.get("eval_s", 0), 0.
    torch.cuda.reset_peak_memory_stats()
    dist.barrier()
    start = time.monotonic()
    next_eval = min(300., seconds/4)
    profile_wall = None
    if trace:
        trace.start()
    step = 0
    running = True
    while running:
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        metrics = torch.zeros(3, device=device)
        wait_seconds = 0.
        events = []
        transfer_events = []
        factor = min(1., (step+1)/20)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"]*factor
        for micro in range(accumulation):
            wait_start = time.monotonic()
            cpu_batch = next(iterator)
            h0, h1 = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            h0.record()
            batch = move(cpu_batch, device)
            h1.record()
            transfer_events.append((h0, h1))
            wait_seconds += time.monotonic()-wait_start
            f0, f1, b0, b1 = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            f0.record()
            context = ddp.no_sync() if micro < accumulation-1 else nullcontext()
            with context:
                with torch.autocast("cuda", dtype=torch.bfloat16), torch.profiler.record_function("policy_forward"):
                    prediction = ddp(batch)
                    loss, losses = objective(prediction, batch, config)
                f1.record()
                b0.record()
                with torch.profiler.record_function("policy_backward"):
                    (loss/accumulation).backward()
                b1.record()
            events.append((f0, f1, b0, b1))
            metrics += losses/accumulation
        o0, o1 = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        o0.record()
        gradnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(gradnorm):
            raise RuntimeError(f"nonfinite gradient at step {step}")
        optimizer.step()
        o1.record()
        torch.cuda.synchronize()
        durations = torch.tensor([time.monotonic()-tick, wait_seconds,
            sum(f0.elapsed_time(f1) for f0, f1, _, _ in events)/1000,
            sum(b0.elapsed_time(b1) for _, _, b0, b1 in events)/1000,
            o0.elapsed_time(o1)/1000,
            sum(h0.elapsed_time(h1) for h0, h1 in transfer_events)/1000], device=device)
        dist.all_reduce(durations, op=dist.ReduceOp.MAX)
        dist.all_reduce(metrics)
        step += 1
        row = dict(step=step, step_s=float(durations[0]), data_wait_s=float(durations[1]),
            forward_s=float(durations[2]), backward_s=float(durations[3]), optimizer_s=float(durations[4]),
            H2D_s=float(durations[5]),
            loss_trajectory=float(metrics[0]/world), loss_action=float(metrics[1]/world),
            loss_rationale=float(metrics[2]/world), grad_norm=float(gradnorm))
        rows.append(row)
        elapsed = time.monotonic()-start
        if rank == 0 and (step == 1 or step % 20 == 0):
            emit("TRAIN_STEP", case=name, mode=mode, wall_s=elapsed, **row)
        if trace and step == args.profile_steps:
            trace.stop()
            profile_wall = elapsed
            trace.export_chrome_trace(str(out/"trace.json"))
            trace = None
        # A rank0 wall-clock gate broadcast keeps collective counts identical.
        signal = torch.tensor([int(elapsed >= seconds or time.time() >= args.deadline_epoch-240),
                               int(mode == "train" and elapsed >= next_eval)], device=device)
        dist.broadcast(signal, src=0)
        if signal[1]:
            evaluated, _, _ = evaluate(policy, val, device, rank, world, args.microbatch)
            total_eval += evaluated["eval_s"]
            curve = dict(step=step, wall_s=time.monotonic()-start, samples_seen=step*global_batch,
                         GPU_hours=(time.monotonic()-start)*world/3600, **evaluated)
            curves.append(curve)
            if evaluated["ADE_m"] < best_ADE:
                best_ADE, best_step = evaluated["ADE_m"], step
                best_state = {n: p.detach().cpu().clone() for n, p in policy.named_parameters() if p.requires_grad}
            emit("VALIDATION", case=name, **curve)
            next_eval += min(300., seconds/4)
            if rank == 0:
                atomic_json(out/"curves.json", curves)
                atomic_json(args.live/f"{name}.progress.json",
                            dict(environment=environment, curves=curves, last_step=row))
        running = not bool(signal[0])
        if mode != "train" and step >= max(args.profile_steps+20, 60):
            running = False
    if trace:
        trace.stop()
        profile_wall = time.monotonic()-start
        trace.export_chrome_trace(str(out/"trace.json"))
    total_wall = time.monotonic()-start
    final, details, explanation = evaluate(policy, val, device, rank, world, args.microbatch,
                                           explain=mode == "train")
    total_eval += final["eval_s"]
    total_explain = explanation["explain_s"] if explanation else 0.
    raw_final = dict(final)
    raw_curve = dict(step=step, wall_s=total_wall, samples_seen=step*global_batch,
                     GPU_hours=total_wall*world/3600, **raw_final)
    if args.suite == "final" and best_step < step and best_ADE < final["ADE_m"]:
        with torch.no_grad():
            for n, p in policy.named_parameters():
                if p.requires_grad:
                    p.copy_(best_state[n].to(device=p.device, dtype=p.dtype))
        final, details, explanation = evaluate(policy, val, device, rank, world, args.microbatch, explain=True)
        total_eval += final["eval_s"]
        total_explain += explanation["explain_s"]
        emit("BEST_CHECKPOINT_SELECTED", selected_step=best_step, ADE_m=final["ADE_m"], raw_final=raw_final)
    else:
        best_step = step
    changes = {n: float((p.detach().cpu()-before[n]).abs().max())
               for n, p in policy.named_parameters() if p.requires_grad}
    frozen_change = float((frozen_parameter.detach().flatten()[:8192].float().cpu()-frozen_before).abs().max())
    head_digest = hashlib.sha256(policy.head[-1].weight.detach().float().cpu().numpy().tobytes()).hexdigest()
    rank_digests = [None]*world
    dist.all_gather_object(rank_digests, head_digest)
    assert len(set(rank_digests)) == 1, "DDP heads diverged across ranks"
    assert frozen_change == 0, "Frozen pretrained weight sample changed"
    peak = torch.tensor(torch.cuda.max_memory_allocated()/1024**3, device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    # Profiler instrumentation is excluded from the throughput measurement.
    warmup_steps = min(10, len(rows)//3) if mode == "train" else args.profile_steps+5
    warm = rows[warmup_steps:]
    assert warm, "No steps remain after warmup and profiler instrumentation"
    train_s = sum(r["step_s"] for r in warm)
    system = dict(throughput_samples_s=len(warm)*global_batch/train_s,
        throughput_measured_steps=len(warm), throughput_excluded_steps=warmup_steps,
        throughput_scope="max per-rank optimizer-step wall time; excludes evaluation and rank0 profiler steps",
        step_p50_s=float(np.median([r["step_s"] for r in warm])),
        step_p95_s=float(np.percentile([r["step_s"] for r in warm], 95)),
        data_wait_fraction=sum(r["data_wait_s"] for r in warm)/train_s,
        forward_fraction=sum(r["forward_s"] for r in warm)/train_s,
        backward_fraction=sum(r["backward_s"] for r in warm)/train_s,
        optimizer_fraction=sum(r["optimizer_s"] for r in warm)/train_s,
        H2D_fraction=sum(r["H2D_s"] for r in warm)/train_s,
        peak_allocated_GiB=float(peak), steps=step, samples_seen=step*global_batch,
        train_step_total_s=sum(r["step_s"] for r in rows), train_window_s=total_wall,
        eval_total_s=total_eval, explain_total_s=total_explain)
    for key in ("loss_trajectory", "loss_action", "loss_rationale"):
        system[key+"_first10"] = float(np.mean([r[key] for r in rows[:10]]))
        system[key+"_last10"] = float(np.mean([r[key] for r in rows[-10:]]))
    if rank == 0:
        if (out/"trace.json").exists():
            system["profile"] = trace_summary(out/"trace.json", profile_wall)
        checkpoint_start = time.monotonic()
        checkpoint = dict(trainable_state={n: p.detach().cpu() for n, p in policy.named_parameters() if p.requires_grad},
            config=config, model_revision=args.model_revision, case=name, global_batch=global_batch,
            step=best_step, training_completed_step=step, seed=args.seed, cached_input_manifest_sha256=args.data_hash)
        torch.save(checkpoint, out/"checkpoint.pt")
        temporary = args.live/f"{name}.checkpoint.pt.upload"
        with temporary.open("wb") as output, (out/"checkpoint.pt").open("rb") as source:
            while chunk := source.read(8*1024**2):
                output.write(chunk)
        temporary.replace(args.live/f"{name}.checkpoint.pt")
        system["checkpoint_s"] = time.monotonic()-checkpoint_start
        reload_probe = None
        if args.suite == "final":
            tick = time.monotonic()
            restored, saved = restore_policy(args.model, out/"checkpoint.pt", device)
            assert saved["model_revision"] == args.model_revision
            assert saved["cached_input_manifest_sha256"] == args.data_hash
            batch = move({k: v.unsqueeze(0) for k, v in val[0].items()}, device)
            policy.eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                original = policy(batch)
                reproduced = restored(batch)
            deltas = [float((a.float()-b.float()).abs().max()) for a, b in zip(original, reproduced)]
            assert max(deltas) <= 1e-5, f"Checkpoint reload changed predictions: {deltas}"
            reload_probe = dict(verified=True, max_abs_output_deltas=deltas,
                val_token=val.records[0]["token"], reload_and_probe_s=time.monotonic()-tick)
            del restored, saved
            emit("CHECKPOINT_RELOAD_VERIFIED", **reload_probe)
        curves.append(raw_curve)
        result = dict(environment=environment, initial=initial, final=final,
            raw_final=raw_final, selected_checkpoint_step=best_step,
            selected_checkpoint_samples_seen=best_step*global_batch,
            checkpoint_reload_probe=reload_probe,
            system=system, explanation=explanation, curves=curves,
            parameter_changes=changes, changed_parameter_tensors=sum(v > 0 for v in changes.values()),
            frozen_weight_probe=dict(name=frozen_name, sampled_values=8192, max_abs_change=frozen_change),
            rank_head_sha256=rank_digests,
            checkpoint_sha256=hashlib.sha256((out/"checkpoint.pt").read_bytes()).hexdigest())
        atomic_json(out/"result.json", result)
        atomic_json(out/"curves.json", curves)
        atomic_json(out/"predictions.json", details)
        atomic_json(out/"steps.json", rows)
        atomic_json(args.live/f"{name}.result.json", result)
        atomic_json(args.live/f"{name}.predictions.json", details)
        emit("CASE_COMPLETE", case=name, mode=mode, system=system, initial=initial, final=final)
    dist.barrier()
    del iterator, loader, ddp, policy, optimizer, trainable, lora, head, before, trace
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["diagnose8", "diagnose16", "sanity", "final", "scale32", "scale64"], required=True)
    parser.add_argument("--case", choices=list(CASES), default="lora_head")
    parser.add_argument("--case-seconds", type=int, default=1200)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--profile-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    rank, world, args.local_rank = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    torch.cuda.set_device(args.local_rank)
    torch.set_num_threads(2)
    device = torch.device("cuda", args.local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=600))
    local = Path(os.environ["DRIVING_LOCAL"])
    args.output = local/"results"/"training"
    args.live = Path(os.environ["DRIVING_STORE"])/"runs"/os.environ["DRIVING_RUN"]/"live"
    if args.local_rank == 0:
        extract(Path(os.environ["DRIVING_STORE"])/"assets/model.tar", local)
        extract(Path(os.environ["DRIVING_STORE"])/"assets/cache.tar", local)
        args.output.mkdir(exist_ok=True)
        args.live.mkdir(exist_ok=True)
    dist.barrier()
    args.model = local/"model"
    args.data_hash = hashlib.sha256((local/"cache/manifest.json").read_bytes()).hexdigest()
    args.model_revision = (args.model/"MODEL_REVISION").read_text().strip()
    args.hosts = [None]*world
    dist.all_gather_object(args.hosts, socket.gethostname())
    hashes = [None]*world
    dist.all_gather_object(hashes, args.data_hash)
    assert len(set(hashes)) == 1 and len(set(args.hosts)) == world//torch.cuda.device_count()
    train, val = CachedData(local/"cache", "train"), CachedData(local/"cache", "val")
    synthetic = CachedData(local/"cache", "train", synthetic=True)
    try:
        if args.suite in ("diagnose8", "diagnose16", "scale32", "scale64", "sanity"):
            for mode, data in (("synthetic", synthetic), ("cached", train)):
                run_case(args, f"profile_{mode}", CASES["lora_head"], data, val, device,
                         rank, world, 1800, mode=mode)
        if args.suite == "diagnose8":
            for name in CASES:
                run_case(args, name, CASES[name], train, val, device, rank, world, args.case_seconds)
        elif args.suite == "diagnose16":
            run_case(args, "lora_head", CASES["lora_head"], train, val, device, rank, world, args.case_seconds)
            run_case(args, "large_batch", CASES["lora_head"], train, val, device, rank, world,
                     args.case_seconds, global_batch=args.global_batch*2)
        elif args.suite in ("final", "scale32", "scale64"):
            run_case(args, args.case, CASES[args.case], train, val, device, rank, world, args.case_seconds)
        elif args.suite == "sanity":
            run_case(args, "lora_head", CASES["lora_head"], train, val, device, rank, world, 60)
        if rank == 0:
            atomic_json(args.live/"suite_complete.json",
                        dict(suite=args.suite, completed_epoch_s=time.time(), world=world))
            emit("DRIVING_SUITE_COMPLETE", suite=args.suite)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
