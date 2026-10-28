"""Preflight and submit one bounded H20 probe using the existing assets."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from merlin_platform import CLI, CP, DEPENDENCIES, TMP, payload


def invoke(arguments, output, timeout=180):
    response = subprocess.run([CLI, "--control-plane", CP, *arguments],
                              capture_output=True, text=True, timeout=timeout, cwd=TMP)
    output.write_text(response.stdout)
    output.with_suffix(".stderr.txt").write_text(response.stderr)
    if response.returncode:
        raise RuntimeError(f"CLI {arguments[:3]} failed; private details: {output}")
    if "--dry-run" in arguments:
        return None
    data = json.loads(response.stdout)
    if data.get("error") or data.get("ResponseBase", {}).get("code", 0):
        raise RuntimeError(f"Platform rejected request; private details: {output}")
    return data


def main():
    root = Path(__file__).parent
    assert TMP.is_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument("--recover-name", help="Record an already-created task without submitting again")
    args = parser.parse_args()
    if args.recover_name:
        name = args.recover_name
        request = json.loads((TMP / f"{name}.checked.json").read_text())
        created = json.loads((TMP / f"{name}.result.json").read_text())
        checked = json.loads((TMP / f"{name}.precheck.json").read_text())
        submitted_epoch = (TMP / f"{name}.checked.json").stat().st_mtime
        return save_submission(root, name, request, created, checked, submitted_epoch)
    name = "driving-single-gpu-hl-" + time.strftime("%Y%m%d%H%M%S", time.gmtime())
    commands = """
export DRIVING_BENCH_START_EPOCH=$(date +%s)
export DRIVING_BENCH_DEADLINE=$((DRIVING_BENCH_START_EPOCH + 1100))
test -f "$DRIVING_STORE/assets/model.tar"
test -f "$DRIVING_STORE/assets/cache.tar"
""" + DEPENDENCIES + """
python3 "$DRIVING_LOCAL/source/benchmark_single_gpu.py" 2>&1 | tee "$DRIVING_LOCAL/results/benchmark.log"
"""
    files = [root / n for n in ("train.py", "cache_inputs.py", "contracts.py",
                               "download_assets.py", "benchmark_single_gpu.py")]
    request = payload(
        name, files, commands, "hl", 1, 1, 1180,
        "One H20 only; <=20min complete-task bound; same-host real cached LoRA+head "
        "micro2/4/8 x checkpoint on/off; repeated timings and numerical checks; no training quality claim.",
    )
    entry = request["job_config"]["job_template_config"]["entrypoint_full_script"]
    request["job_config"]["job_template_config"]["entrypoint_full_script"] = entry.replace(
        "--kill-after=60s", "--kill-after=10s", 1)
    role = request["resource_config"]["arnold_resource_config"]["roles"][0]
    assert role["num"] == role["gpu"] == 1 and role["cpu"] == 20
    assert request["resource_config"]["arnold_resource_config"]["cluster_ids"] == [20]
    original = TMP / f"{name}.create.json"
    original.write_text(json.dumps(request, indent=2))
    schema = json.loads((TMP / "single-gpu-create-schema.json").read_text())
    try:
        import jsonschema
    except ImportError:
        pass
    else:
        jsonschema.validate(request, schema)
    invoke(["job-v2", "runs", "create", "--from-file", str(original), "--dry-run"],
           TMP / f"{name}.dry-run.json")
    precheck_request = TMP / f"{name}.precheck-request.json"
    precheck_request.write_text(json.dumps(dict(node_type="merlin_job_run", check_point="create",
                                               job_run_context=request)))
    checked = invoke(["job-v2", "check-items", "check", "--from-file", str(precheck_request)],
                     TMP / f"{name}.precheck.json")
    data = checked["data"]
    print(json.dumps(dict(event="PREFLIGHT", status=data["status"],
                          messages=[dict(message=r.get("message"), suggestion=r.get("suggestion"))
                                    for r in data.get("results", [])]), ensure_ascii=False), flush=True)
    if data["status"] == "error":
        raise RuntimeError("Preflight failed; task not submitted")
    request["request_context"] = {**request.get("request_context", {}),
                                  "precheck_sid": data["sid"]}
    final = TMP / f"{name}.checked.json"
    final.write_text(json.dumps(request, indent=2))
    submitted_epoch = time.time()
    created = invoke(["job-v2", "runs", "create", "--from-file", str(final)],
                     TMP / f"{name}.result.json", timeout=240)
    save_submission(root, name, request, created, checked, submitted_epoch)


def save_submission(root, name, request, created, checked, submitted_epoch):
    result = created.get("data", created)
    assert isinstance(result["sid"], str)
    data = checked["data"]
    files = [root / n for n in ("train.py", "cache_inputs.py", "contracts.py",
                               "download_assets.py", "benchmark_single_gpu.py")]
    safe = dict(sid=result["sid"], name=name, world=1, dc="hl",
                submitted_epoch_s=submitted_epoch, hard_stop_epoch_s=submitted_epoch + 1200,
                entry_timeout_s=1180, kill_grace_s=10, suite="single_gpu",
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                precheck_status=data["status"], precheck_sid=data["sid"])
    (root.parent / "results" / "single_gpu_submission.json").write_text(
        json.dumps(safe, indent=2, ensure_ascii=False))
    print(json.dumps(dict(event="CREATED", **safe), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
