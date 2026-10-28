"""GPU smoke check of the pinned Qwen API, bf16 LoRA gradients and frozen weights."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from train import CASES, Policy, objective, run_case


def tiny_config():
    config = Qwen2_5_VLConfig(vocab_size=512, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
        image_token_id=510, video_token_id=511, vision_start_token_id=509,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]},
        vision_config=dict(depth=1, hidden_size=128, intermediate_size=256,
                           num_heads=4, out_hidden_size=128, fullatt_block_indexes=[0]))
    config._attn_implementation = "sdpa"
    return config


class PipelineData(Dataset):
    def __init__(self):
        self.records = [dict(token=str(i), scene="test", trajectory=[[1., 0.]]*6,
            action=1, teacher=dict(action=1, evidence=[1., 0., 0., 0., 0., 0.]),
            future_occupied_circles=[[] for _ in range(6)]) for i in range(8)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        masks = torch.zeros(6, 64, dtype=torch.bool)
        for camera in range(6):
            masks[camera, camera*8:camera*8+8] = True
        return dict(embeddings=torch.randn(64, 128, generator=generator).half(),
            attention=torch.ones(64, dtype=torch.bool),
            positions=torch.arange(64).view(1, -1).expand(3, -1),
            camera_masks=masks, state=torch.zeros(6), trajectory=torch.tensor([[1., 0.]]*6),
            action=torch.tensor(1), teacher_action=torch.tensor(1),
            teacher_valid=torch.tensor(True), teacher_evidence=torch.tensor([1., 0., 0., 0., 0., 0.]),
            record_index=torch.tensor(index))


def pipeline_check():
    def pinned_tiny(*args, **kwargs):
        # A pretrained load returns the same frozen weights on every invocation.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(707)
            # Match HF from_pretrained: loading leaves the entire model in eval mode.
            return Qwen2_5_VLForConditionalGeneration(tiny_config()).to(torch.bfloat16).eval()

    with tempfile.TemporaryDirectory(prefix="driving-pipeline-") as directory:
        root = Path(directory)
        args = SimpleNamespace(global_batch=4, microbatch=2, seed=321, model="unused",
            local_rank=0, hosts=["test"], data_hash="test", model_revision="test",
            output=root/"output", live=root/"live", suite="sanity",
            deadline_epoch=float("inf"), profile_steps=4)
        args.output.mkdir()
        args.live.mkdir()
        dist.init_process_group("nccl", init_method="file://"+str(root/"init"), rank=0, world_size=1)
        try:
            with patch.object(Qwen2_5_VLForConditionalGeneration, "from_pretrained",
                    side_effect=pinned_tiny):
                for mode in ("synthetic", "train"):
                    args.suite = "final" if mode == "train" else "sanity"
                    run_case(args, mode, CASES["lora_head"], PipelineData(), PipelineData(),
                        torch.device("cuda", 0), 0, 1, 2, mode=mode)
                    result = json.loads((args.live/f"{mode}.result.json").read_text())
                    checkpoint = torch.load(args.live/f"{mode}.checkpoint.pt", weights_only=True)
                    environment = result["environment"]
                    assert all(environment[flag] is True for flag in (
                        "policy_training", "language_training",
                        "gradient_checkpointing_enabled", "gradient_checkpointing_active"))
                    assert checkpoint["step"] == result["selected_checkpoint_step"]
                    assert result["changed_parameter_tensors"] > 0
                    if mode == "synthetic":
                        assert result["system"]["profile"]["kernel_count"] > 0
                        assert result["system"]["throughput_excluded_steps"] == 9
                    else:
                        assert result["explanation"]["interventions"] == 48
                        assert result["checkpoint_reload_probe"]["verified"]
                    print("GPU_PIPELINE_OK", mode, result["system"]["steps"], flush=True)
        finally:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.is_available()
    config = tiny_config()
    for name in ("frozen_head", "lora_head", "state_only", "image_only"):
        tiny = Qwen2_5_VLForConditionalGeneration(config).to(dtype=torch.bfloat16)
        with patch.object(Qwen2_5_VLForConditionalGeneration, "from_pretrained", return_value=tiny):
            policy = Policy("unused", CASES[name]).cuda()
        batch = dict(embeddings=torch.randn(2, 64, 128, device="cuda", dtype=torch.float16),
            attention=torch.ones(2, 64, device="cuda", dtype=torch.bool),
            positions=torch.arange(64, device="cuda").view(1, 1, -1).expand(2, 3, -1),
            camera_masks=torch.zeros(2, 6, 64, device="cuda", dtype=torch.bool),
            state=torch.randn(2, 6, device="cuda"), trajectory=torch.randn(2, 6, 2, device="cuda"),
            action=torch.tensor([1, 2], device="cuda"), teacher_action=torch.tensor([0, 1], device="cuda"),
            teacher_valid=torch.ones(2, device="cuda", dtype=torch.bool),
            teacher_evidence=torch.zeros(2, 6, device="cuda"))
        for camera in range(6):
            batch["camera_masks"][:, camera, camera*8:camera*8+8] = True
        frozen_name, frozen = next((n, p) for n, p in policy.named_parameters() if not p.requires_grad)
        before = frozen.detach().clone()
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-3)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = policy(batch)
                loss, components = objective(prediction, batch, CASES[name])
            loss.backward()
            assert all(p.grad is None for p in policy.language.parameters() if not p.requires_grad)
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in policy.parameters() if p.requires_grad)
            if name == "lora_head":
                assert any(p.grad.abs().max() > 0 for n, p in policy.named_parameters() if n.endswith(".b"))
            optimizer.step()
        assert torch.equal(frozen, before)
        policy.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            original = policy(batch)
            counterfactual = policy(batch, remove_camera=0)
        assert all(torch.isfinite(t).all() for t in counterfactual)
        if name == "state_only":
            assert all(torch.equal(a, b) for a, b in zip(original, counterfactual)), (
                "A camera intervention must preserve the state-only image exclusion")
        print("PINNED_GPU_RUNTIME_OK", name, float(loss.detach()), frozen_name, flush=True)
    print("DRIVING_RUNTIME_CHECK_COMPLETE", torch.__version__, torch.cuda.get_device_name(), flush=True)
    if args.pipeline:
        pipeline_check()


if __name__ == "__main__":
    main()
