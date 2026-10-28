"""Merlin submission helpers; each create uses the identical payload for preflight."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

CLI = "/home/tiger/.merlin-cli/bin/merlin-cli"
CP = "cn"
HDFS_ROOT = "hdfs://haruna/home/byte_search_lifeservice/zhangyang.exp/driving_scaling/20261028T0203"
STORE = "/mnt/hdfs/driving"
IMAGE_VID = "d7iu6hbc77u4ebjdshsg"
QUEUES = {
    "hl": (20, "compute-124-hl-cloudnative-ai-search.douyin.life.nlp-guarantee"),
    "lf": (22, "compute-124-lf-cloudnative-ai-search.douyin.life.nlp-guarantee"),
}
TMP = Path("/tmp/driving-session")
START_EPOCH = 1793124180
LAST_START_EPOCH = START_EPOCH + 10*3600
FINISH_EPOCH = START_EPOCH + 14*3600
DEPENDENCIES = """
python3 -m pip install --upgrade --target "$DRIVING_LOCAL/deps" --index-url https://bytedpypi.byted.org/simple 'transformers==4.51.3' 'tokenizers==0.21.4' 'huggingface-hub==0.36.0' 'numpy==1.26.4' 'pillow==11.3.0' 2>&1 | tee "$DRIVING_LOCAL/results/deps.log"
export PYTHONPATH="$DRIVING_LOCAL/deps:$DRIVING_LOCAL/source"
python3 -m pip freeze > "$DRIVING_LOCAL/results/packages-final.txt"
"""


def wait_assets(names, seconds=7200):
    encoded = json.dumps(names)
    return f"""
python3 - <<'ASSET_WAIT'
import json,os,time
from pathlib import Path
names=json.loads('{encoded}')
root=Path(os.environ['DRIVING_STORE'])/'assets'
deadline=min(time.time()+{seconds}, {FINISH_EPOCH}-600)
while not all((root/name).exists() for name in names):
 if time.time()>deadline: raise TimeoutError('Inputs not available within budget: '+str(names))
 time.sleep(15)
print('INPUTS_AVAILABLE',names,flush=True)
ASSET_WAIT
"""


def cache_request(attempt=1):
    root = Path(__file__).parent
    names = ["cache_inputs.py", "contracts.py", "download_assets.py"]
    commands = wait_assets(["model.tar", "nuscenes-cameras.tar.gz"]) + DEPENDENCIES + """
python3 "$DRIVING_LOCAL/source/cache_inputs.py" 2>&1 | tee "$DRIVING_LOCAL/results/cache.log"
"""
    name = "driving-cache-hl-20261028" if attempt == 1 else f"driving-cache-hl-a{attempt}-20261028"
    return payload(name, [root/n for n in names], commands,
                   "hl", 1, 1, 14400,
                   "Official mini scene split; frozen 6-camera Qwen vision embeddings + current-only offline teacher labels; 4h bound.")


def training_request(suite, world, seconds, case="lora_head", attempt=1, dc="lf",
                     wall_limit_s=None, deadline_epoch=None):
    root = Path(__file__).parent
    names = ["train.py", "cache_inputs.py", "contracts.py", "download_assets.py", "launch_distributed.py"]
    microbatch = 4 if world <= 16 else 2 if world == 32 else 1
    name = f"driving-{suite}-{world}h20-a{attempt}-20261028"
    gpus, nodes = min(8, world), max(1, world//8)
    global_batch = 4 if suite == "sanity" else 64
    deadline = min(FINISH_EPOCH, deadline_epoch or FINISH_EPOCH)
    commands = wait_assets(["model.tar", "cache.tar"], 9000) + DEPENDENCIES + f"""
python3 "$DRIVING_LOCAL/source/launch_distributed.py" --nodes {nodes} --gpus {gpus} -- --suite {suite} --case {case} --case-seconds {seconds} --global-batch {global_batch} --microbatch {microbatch} --deadline-epoch {deadline} 2>&1 | tee "$DRIVING_LOCAL/results/training.log"
"""
    bound = min(32000, max(1800, int(deadline-time.time()-60)))
    if wall_limit_s is not None:
        if wall_limit_s < 1800:
            raise ValueError("The task wall limit must allow preparation and a short probe")
        bound = min(bound, int(wall_limit_s))
    return payload(name, [root/n for n in names], commands, dc, nodes, gpus, bound,
        f"{suite}; {world} H20 in {dc}; {global_batch} global batch, LoRA q/v r8 + residual trajectory/action/evidence heads; {seconds}s probe windows; deadline bounded.")


def cli(*command, payload=None, timeout=180):
    args = [CLI, "--control-plane", CP, *command]
    if payload is not None:
        args += ["--json", json.dumps(payload, ensure_ascii=False)]
    result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    if result.stderr:
        print(result.stderr, flush=True)
    if result.returncode:
        raise RuntimeError(f"CLI failed ({result.returncode}): {result.stdout[:4000]}")
    data = json.loads(result.stdout)
    if data.get("ResponseBase", {}).get("code", 0):
        raise RuntimeError(json.dumps(data, ensure_ascii=False))
    if data.get("error"):
        raise RuntimeError(json.dumps(data, ensure_ascii=False))
    return data


def script_entrypoint(files, commands, name, seconds, expected_pods=1):
    blob = {}
    for file in files:
        content = Path(file).read_bytes()
        blob[Path(file).name] = {
            "data": base64.b64encode(gzip.compress(content, mtime=0)).decode(),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    encoded = base64.b64encode(json.dumps(blob).encode()).decode()
    return f"""#!/bin/bash
exec timeout --signal=TERM --kill-after=60s {seconds}s /bin/bash -s <<'DRIVING_INNER'
set -euo pipefail
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY="$HTTP_PROXY"
export NO_PROXY=.byted.org,localhost,127.0.0.1
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HF_HOME=/tmp/driving-hf-home HF_HUB_DISABLE_XET=1
export DRIVING_LOCAL=/tmp/driving-{name}
export DRIVING_STORE={STORE}
export DRIVING_RUN={name}
export DRIVING_EXPECTED_PODS={expected_pods}
mkdir -p "$DRIVING_LOCAL/source" "$DRIVING_LOCAL/results" "$DRIVING_STORE/runs/{name}"
finish() {{
 rc=$?
 trap - EXIT
 set +e
 printf '%s\\n' "$rc" > "$DRIVING_LOCAL/results/exit_code.txt"
 pod="${{MY_POD_NAME:-local}}"
 tar -czf "$DRIVING_STORE/runs/{name}/$pod.tar.gz.upload" -C "$DRIVING_LOCAL" results
 copy_rc=$?
 if [[ "$copy_rc" == 0 ]]; then
   mv "$DRIVING_STORE/runs/{name}/$pod.tar.gz.upload" "$DRIVING_STORE/runs/{name}/$pod.tar.gz"
   copy_rc=$?
 fi
 if [[ "$rc" == 0 && "$copy_rc" == 0 ]]; then
   python3 - <<'ARCHIVE_BARRIER'
import os,time
from pathlib import Path
root=Path(os.environ["DRIVING_STORE"])/"runs"/os.environ["DRIVING_RUN"]
deadline=min(time.time()+240, {FINISH_EPOCH}-60)
expected=int(os.environ["DRIVING_EXPECTED_PODS"])
while len(list(root.glob("*.tar.gz"))) < expected:
 if time.time()>deadline: raise TimeoutError("Missing completed pod archives")
 time.sleep(2)
print("ARCHIVE_BARRIER_COMPLETE",expected,flush=True)
ARCHIVE_BARRIER
   barrier_rc=$?
   if [[ "$barrier_rc" != 0 ]]; then rc=$barrier_rc; fi
 fi
 echo "DRIVING_EXIT run={name} original_rc=$rc copy_rc=$copy_rc"
 if [[ "$rc" == 0 && "$copy_rc" != 0 ]]; then rc=$copy_rc; fi
 exit "$rc"
}}
trap finish EXIT
trap 'exit 124' TERM
python3 - <<'UNPACK'
import base64,gzip,hashlib,json,os
from pathlib import Path
root=Path(os.environ['DRIVING_LOCAL'])/'source'
for name,item in json.loads(base64.b64decode('{encoded}')).items():
 data=gzip.decompress(base64.b64decode(item['data']))
 assert hashlib.sha256(data).hexdigest()==item['sha256']
 (root/name).write_bytes(data)
UNPACK
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv > "$DRIVING_LOCAL/results/gpus.csv"
python3 -m pip freeze > "$DRIVING_LOCAL/results/packages-initial.txt"
{commands}
DRIVING_INNER
"""


def payload(name, files, commands, dc, nodes, gpus, seconds, description):
    cluster, queue = QUEUES[dc]
    return {
        "name": name, "namespace": "/user/zhangyang.exp", "job_type": "train",
        "description": description,
        "attachments": [{
            "kind": "hdfs_volume",
            "hdfs_volume_attachment": {"hdfs_volumes": [{
                "access_mode": "RW", "mnt": STORE, "path": HDFS_ROOT,
                "roles": ["worker"],
            }]},
        }],
        "job_config": {"job_template_config": {
            "image_meta": {"image_source": "vid", "image_vid": IMAGE_VID},
            "entrypoint_full_script": script_entrypoint(files, commands, name, seconds, nodes),
        }},
        "resource_config": {
            "resource_backend": "arnold",
            "arnold_resource_config": {
                "cluster_ids": [cluster], "cluster_select_mode": "single", "group_id": 2079,
                "roles": [{
                    "name": "worker", "num": nodes, "gpu": gpus, "gpu_type": "NVIDIA_H20",
                    "cpu": 20 * gpus, "memory_mb": 230400 * gpus, "ports": 2,
                    "queue_name": queue,
                    "advanced": {
                        "is_complete_trial_on_task_no_restart": True,
                        "is_fail_trial_on_pod_failure": True, "pod_no_restart_state": "any",
                    },
                }],
            },
        },
    }


def submit(request):
    TMP.mkdir(parents=True, exist_ok=True)
    name = request["name"]
    original = TMP / f"{name}.create.json"
    original.write_text(json.dumps(request, indent=2))
    preview = subprocess.run(
        [CLI, "--control-plane", CP, "job-v2", "runs", "create",
         "--from-file", str(original), "--dry-run"],
        text=True, capture_output=True, timeout=180, cwd=TMP,
    )
    if preview.returncode:
        raise RuntimeError(f"Dry-run failed: {preview.stdout}\n{preview.stderr}")
    (TMP / f"{name}.dry-run.json").write_text(preview.stdout)
    precheck = cli("job-v2", "check-items", "check", payload={
        "node_type": "merlin_job_run", "check_point": "create",
        "job_run_context": request,
    })
    (TMP / f"{name}.precheck.json").write_text(json.dumps(precheck, indent=2))
    print("PREFLIGHT", json.dumps(precheck, ensure_ascii=False), flush=True)
    if precheck["data"]["status"] == "error":
        raise RuntimeError("Preflight rejected the request; submission stopped.")
    checked = dict(request)
    checked["request_context"] = {
        **request.get("request_context", {}), "precheck_sid": precheck["data"]["sid"],
    }
    checked_file = TMP / f"{name}.checked.json"
    checked_file.write_text(json.dumps(checked, indent=2))
    result = cli("job-v2", "runs", "create", payload=checked, timeout=240)
    (TMP / f"{name}.result.json").write_text(json.dumps(result, indent=2))
    print("CREATED", json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-download", action="store_true")
    args = parser.parse_args()
    if args.prepare_download:
        source = Path(__file__).parent / "download_assets.py"
        commands = """
python3 -m pip install --target "$DRIVING_LOCAL/deps" --index-url https://bytedpypi.byted.org/simple 'huggingface-hub==0.36.0' 2>&1 | tee "$DRIVING_LOCAL/results/deps.log"
export PYTHONPATH="$DRIVING_LOCAL/deps:$DRIVING_LOCAL/source"
python3 "$DRIVING_LOCAL/source/download_assets.py" 2>&1 | tee "$DRIVING_LOCAL/results/download.log"
"""
        submit(payload(
            "driving-assets-hl-20261028", [source], commands, "hl", 1, 1, 7200,
            "Pinned Qwen2.5-VL-3B and official nuScenes mini camera preparation; 1 H20; 2h bound; no training quality claim.",
        ))
