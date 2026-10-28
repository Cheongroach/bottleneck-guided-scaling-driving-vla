"""Save whitelisted terminal evidence without publishing platform payloads."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

from merlin_platform import CLI, CP, TMP

ROOT = Path(__file__).resolve().parent.parent / "results"


def call(command, request, name):
    response = subprocess.run(
        [CLI, "--control-plane", CP, *command, "--json", json.dumps(request)],
        capture_output=True, text=True, timeout=90)
    raw = TMP / (name + ".json")
    raw.write_text(response.stdout)
    raw.with_suffix(".stderr.txt").write_text(response.stderr)
    if response.returncode:
        raise RuntimeError(f"CLI failed; private evidence: {raw}")
    data = json.loads(response.stdout)
    if data.get("error") or data.get("ResponseBase", {}).get("code", 0):
        raise RuntimeError(f"Platform error; private evidence: {raw}")
    return data, hashlib.sha256(response.stdout.encode()).hexdigest()


def epoch(timestamp):
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()


def main():
    job = json.loads((ROOT / "single_gpu_submission.json").read_text())
    response, status_sha = call(["job-v2", "runs", "get"], {"sid": job["sid"]},
                                "single-gpu-terminal-status-raw")
    data = response["data"]
    assert data["status"] in {"success", "done"}
    trial = data["runtime_context"]["trial_id"]
    response, pod_sha = call(["job-v2", "run-trial-pods", "list"],
                             {"filter": {"trial_sid_any": [trial]}}, "single-gpu-terminal-pods-raw")
    pods = [
        {k: p.get(k) for k in ("sid", "pod_name", "start_at", "stop_at", "status", "exit_status")}
        for p in response["data"]
    ]
    assert len(pods) == 1 and pods[0]["exit_status"] == 0 and pods[0]["stop_at"]
    start, stop = epoch(pods[0]["start_at"]), epoch(pods[0]["stop_at"])
    created = epoch(data["created_at"])
    assert 0 < stop - created <= 1200
    safe = dict(sid=job["sid"], trial_id=trial, status=data["status"],
                created_at=data["created_at"], pods=pods, pod_wall_s=stop - start,
                creation_to_exit_s=stop - created, submit_to_exit_s=stop - job["submitted_epoch_s"],
                source_response_sha256=dict(status=status_sha, pods=pod_sha),
                resource_release_evidence="One pod has stop_at, exit_status=0 and job terminal success",
                allocated_gpus=1)
    (ROOT / "single_gpu_terminal.json").write_text(json.dumps(safe, indent=2))
    print(json.dumps(safe, indent=2))


if __name__ == "__main__":
    main()
