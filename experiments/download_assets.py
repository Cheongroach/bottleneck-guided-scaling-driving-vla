"""Download pinned public inputs and publish atomic archives to the run's HDFS mount."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
DATA_URL = "https://www.nuscenes.org/data/v1.0-mini.tgz"
DATA_BYTES = 4167696325


def publish(source, destination):
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".upload")
    shutil.copyfile(source, temporary)
    assert temporary.stat().st_size == source.stat().st_size
    temporary.rename(destination)
    return {
        "file": destination.name,
        "bytes": source.stat().st_size,
        "sha256": hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
    }


def main():
    from huggingface_hub import snapshot_download

    started = time.time()
    root = Path(os.environ["DRIVING_LOCAL"])
    store = Path(os.environ["DRIVING_STORE"]) / "assets"
    root.mkdir(parents=True, exist_ok=True)
    store.mkdir(parents=True, exist_ok=True)

    def model():
        target = root / "model"
        snapshot_download(
            MODEL_ID, revision=MODEL_REVISION, local_dir=target, max_workers=4,
            ignore_patterns=[".gitattributes"],
        )
        (target / "MODEL_REVISION").write_text(MODEL_REVISION + "\n")
        index = json.loads((target / "model.safetensors.index.json").read_text())
        for name in set(index["weight_map"].values()):
            assert (target / name).stat().st_size > 0, name
        archive = root / "model.tar"
        with tarfile.open(archive, "w") as tar:
            for path in sorted(target.iterdir()):
                if path.is_file():
                    tar.add(path, arcname="model/" + path.name)
        result = publish(archive, store / archive.name)
        print("MODEL_READY", json.dumps(result), flush=True)
        return result

    def dataset():
        source = root / "nuscenes-official.tgz"
        subprocess.run([
            "curl", "-fL", "--connect-timeout", "30", "--max-time", "5400",
            "--speed-time", "120", "--speed-limit", "1024",
            DATA_URL, "-o", str(source),
        ], check=True)
        assert source.stat().st_size == DATA_BYTES
        output = root / "nuscenes-cameras.tar.gz"
        counts = {"metadata": 0, "camera_images": 0}
        with tarfile.open(source, "r|gz") as incoming, tarfile.open(output, "w:gz") as outgoing:
            for member in incoming:
                name = member.name.removeprefix("./")
                keep_meta = name.startswith("v1.0-mini/") and name.endswith(".json")
                keep_image = name.startswith("samples/CAM_") and name.endswith(".jpg")
                if member.isfile() and (keep_meta or keep_image):
                    member.name = name
                    outgoing.addfile(member, incoming.extractfile(member))
                    counts["metadata" if keep_meta else "camera_images"] += 1
        assert counts["metadata"] >= 10 and counts["camera_images"] >= 2000, counts
        result = publish(output, store / output.name)
        result.update(counts)
        result["source_sha256"] = hashlib.file_digest(source.open("rb"), "sha256").hexdigest()
        print("DATASET_READY", json.dumps(result), flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        model_future = executor.submit(model)
        dataset_future = executor.submit(dataset)
        result = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_url": DATA_URL,
            "dataset_split": "v1.0-mini",
            "model": model_future.result(),
            "dataset": dataset_future.result(),
            "elapsed_s": time.time() - started,
        }
    (root / "download_manifest.json").write_text(json.dumps(result, indent=2))
    publish(root / "download_manifest.json", store / "download_manifest.json")
    print("DRIVING_DOWNLOAD_COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
