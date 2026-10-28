"""Publish real vision embeddings, multimodal positions and offline teacher labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import time

import numpy as np
from PIL import Image
import torch

from contracts import ACTION_NAMES, CAMERAS, HORIZON, build_records, self_check
from download_assets import publish


def extract(archive, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar:
            path = (directory/member.name).resolve()
            assert path.is_relative_to(directory.resolve()) and not member.issym() and not member.islnk()
            tar.extract(member, directory)


def prompt(processor, teacher=False, speed=0):
    question = ("Return JSON only with action (one of " + ", ".join(ACTION_NAMES) +
        "), evidence (list of camera names), rationale (short sentence), "
        "trajectory (six [x,y] coordinates in meters at 0.5,1,1.5,2,2.5,3 seconds; x forward, y left). "
        f"Current ego speed {speed:.2f} m/s. Choose a safe maneuver using current evidence only."
        if teacher else "Observe these six labeled camera views. Summarize the scene for a driving policy.")
    content = []
    for camera in CAMERAS:
        content.extend([dict(type="text", text=camera+": "), dict(type="image")])
    content.append(dict(type="text", text=question))
    return processor.apply_chat_template([dict(role="user", content=content)],
                                        tokenize=False, add_generation_prompt=True)


def parse_teacher(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        row = json.loads(match.group())
        assert row["action"] in ACTION_NAMES and isinstance(row["rationale"], str)
        assert isinstance(row["evidence"], list) and all(c in CAMERAS for c in row["evidence"])
        trajectory = np.asarray(row.get("trajectory", []), dtype=np.float64)
        valid_trajectory = trajectory.shape == (6, 2) and np.isfinite(trajectory).all()
        return dict(action=ACTION_NAMES.index(row["action"]), rationale=row["rationale"],
                    evidence=[float(c in row["evidence"]) for c in CAMERAS],
                    trajectory=trajectory.tolist() if valid_trajectory else None)
    except (ValueError, KeyError, AssertionError, TypeError):
        return None


def main():
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-limit", type=int, default=10000)
    args = parser.parse_args()
    self_check()
    started = time.monotonic()
    local = Path(os.environ["DRIVING_LOCAL"])
    store = Path(os.environ["DRIVING_STORE"])
    results = local/"results"
    live = store/"runs"/os.environ["DRIVING_RUN"]/"live"
    live.mkdir(exist_ok=True)
    extract(store/"assets/model.tar", local)
    extract(store/"assets/nuscenes-cameras.tar.gz", local/"data")
    records, manifest = build_records(local/"data")
    processor = AutoProcessor.from_pretrained(local/"model", min_pixels=112*112, max_pixels=112*112)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        local/"model", torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    assert model.config.hidden_size > 0
    dataset = local/"cache"
    dataset.mkdir()
    vision_seconds, teacher_seconds, token_count, lengths = [], [], [], []
    max_length = 256
    valid = 0
    for index, row in enumerate(records):
        images = [Image.open(local/"data"/p).convert("RGB").resize((112, 112)) for p in row["images"]]
        batch = processor(text=[prompt(processor)], images=images, return_tensors="pt")
        batch = {key: value.cuda() for key, value in batch.items()}
        torch.cuda.synchronize()
        tick = time.monotonic()
        with torch.no_grad():
            embeddings = model.model.embed_tokens(batch["input_ids"])
            visual = model.visual(batch["pixel_values"].to(model.visual.dtype),
                                  grid_thw=batch["image_grid_thw"])
            mask = batch["input_ids"] == model.config.image_token_id
            assert mask.sum() == len(visual)
            embeddings[mask] = visual
            positions, _ = model.get_rope_index(
                batch["input_ids"], image_grid_thw=batch["image_grid_thw"],
                attention_mask=batch["attention_mask"])
        torch.cuda.synchronize()
        vision_seconds.append(time.monotonic()-tick)
        length = embeddings.shape[1]
        assert length <= max_length, length
        lengths.append(length)
        values = torch.zeros(max_length, model.config.hidden_size, dtype=torch.float16)
        values[:length] = embeddings[0].cpu().half()
        position = torch.zeros(3, max_length, dtype=torch.int32)
        position[:, :length] = positions[:, 0].cpu().int()
        attention = torch.zeros(max_length, dtype=torch.bool)
        attention[:length] = True
        camera_masks = torch.zeros(6, max_length, dtype=torch.bool)
        image_locations = torch.where(mask[0])[0].cpu()
        offsets = torch.prod(batch["image_grid_thw"], -1).cpu() // model.visual.spatial_merge_size**2
        offset = 0
        for camera, count in enumerate(offsets.tolist()):
            camera_masks[camera, image_locations[offset:offset+count]] = True
            offset += count
        assert offset == len(image_locations)
        cached = dict(embeddings=values, positions=position, attention=attention,
                      camera_masks=camera_masks)
        torch.save(cached, dataset/f'{row["token"]}.pt')
        row["cached_file"] = f'{row["token"]}.pt'
        row["length"] = length
        teacher = None
        if index < args.teacher_limit:
            teacher_batch = processor(text=[prompt(processor, True, row["state"][-1])],
                                      images=images, return_tensors="pt")
            teacher_batch = {key: value.cuda() for key, value in teacher_batch.items()}
            torch.cuda.synchronize()
            tick = time.monotonic()
            with torch.inference_mode():
                output = model.generate(**teacher_batch, max_new_tokens=256, do_sample=False)
            torch.cuda.synchronize()
            teacher_seconds.append(time.monotonic()-tick)
            generated = output[:, teacher_batch["input_ids"].shape[1]:]
            token_count.append(generated.shape[1])
            text = processor.batch_decode(generated, skip_special_tokens=True)[0]
            teacher = parse_teacher(text)
            row["teacher_text"] = text
            if teacher:
                valid += 1
        row["teacher"] = teacher
        for image in images:
            image.close()
        if index % 10 == 0:
            progress = dict(done=index+1, total=len(records), teacher_valid=valid,
                elapsed_s=time.monotonic()-started, teacher_total_s=sum(teacher_seconds),
                teacher_examples=[dict(token=r["token"], text=r.get("teacher_text"), parsed=r["teacher"])
                                  for r in records[:min(index+1, 3)]])
            temporary = live/"cache_progress.json.upload"
            temporary.write_text(json.dumps(progress))
            temporary.replace(live/"cache_progress.json")
            print("CACHE_PROGRESS", json.dumps(progress), flush=True)
    (dataset/"records.json").write_text(json.dumps(records))
    manifest.update(dict(model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        model_revision=(local/"model/MODEL_REVISION").read_text().strip(),
        hidden_size=model.config.hidden_size, max_length=max_length,
        sequence_lengths=dict(min=min(lengths), max=max(lengths), mean=float(np.mean(lengths))),
        input_cache="frozen visual embeddings + frozen text embeddings, original 3-axis MRoPE positions",
        teacher_count=len(teacher_seconds), teacher_valid=valid, teacher_invalid=len(teacher_seconds)-valid,
        teacher_input="current six 112px cameras and current speed only; no future labels",
        teacher_model="same pinned Qwen2.5-VL-3B baseline; weak self-distillation, not external oracle",
        teacher_output="action, six evidence flags, natural language rationale and optional predicted trajectory",
        teacher_trajectory_valid=sum(bool(r["teacher"] and r["teacher"].get("trajectory")) for r in records),
        teacher_total_s=sum(teacher_seconds), teacher_s_per_example=float(np.mean(teacher_seconds)),
        teacher_tokens=sum(token_count), teacher_tokens_per_s=sum(token_count)/sum(teacher_seconds),
        vision_total_s=sum(vision_seconds), vision_s_per_example=float(np.mean(vision_seconds)),
        preparation_wall_s=time.monotonic()-started))
    baselines = {}
    for split in ("train", "val"):
        distances = [np.linalg.norm(
            np.array(r["state"][:2])[None, :]*HORIZON[:, None]-np.array(r["trajectory"]), axis=-1)
            for r in records if r["split"] == split]
        baselines[split] = dict(ADE_m=float(np.mean(distances)),
            FDE_m=float(np.mean([d[-1] for d in distances])),
            miss_rate_2m=float(np.mean([d[-1] > 2 for d in distances])))
    manifest["constant_velocity_baseline"] = baselines
    teacher_distances = [np.linalg.norm(np.array(r["teacher"]["trajectory"])-np.array(r["trajectory"]), axis=-1)
                         for r in records if r["split"] == "val" and r["teacher"] and r["teacher"].get("trajectory")]
    manifest["teacher_val_trajectory"] = dict(count=len(teacher_distances),
        ADE_m=float(np.mean(teacher_distances)) if teacher_distances else None,
        FDE_m=float(np.mean([d[-1] for d in teacher_distances])) if teacher_distances else None)
    (dataset/"manifest.json").write_text(json.dumps(manifest, indent=2))
    # Save thumbnails for real-image evidence and later qualitative report.
    previews = dataset/"previews"
    previews.mkdir()
    for row in [r for r in records if r["split"] == "val"][::8]:
        canvas = Image.new("RGB", (672, 112))
        for camera, path in enumerate(row["images"]):
            with Image.open(local/"data"/path) as image:
                canvas.paste(image.convert("RGB").resize((112, 112)), (camera*112, 0))
        canvas.save(previews/f'{row["token"]}.jpg')
    publish(dataset/"records.json", store/"assets/records.json")
    preview_archive = local/"previews.tar.gz"
    with tarfile.open(preview_archive, "w:gz") as tar:
        tar.add(previews, arcname="previews")
    publish(preview_archive, store/"assets/previews.tar.gz")
    archive = local/"cache.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(dataset, arcname="cache")
    publication = publish(archive, store/"assets/cache.tar")
    manifest["publication"] = publication
    manifest["preparation_wall_s"] = time.monotonic()-started
    manifest["cache_other_s"] = max(0., manifest["preparation_wall_s"] -
                                   manifest["teacher_total_s"] - manifest["vision_total_s"])
    (results/"cache_manifest.json").write_text(json.dumps(manifest, indent=2))
    publish(results/"cache_manifest.json", store/"assets/cache_manifest.json")
    print("DRIVING_CACHE_COMPLETE", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
