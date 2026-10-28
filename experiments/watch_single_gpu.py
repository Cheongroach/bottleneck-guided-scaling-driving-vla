"""Monitor the authorized single-GPU task, enforce its cap, collect artifacts."""
import json
from pathlib import Path
import subprocess
import time

from merlin_platform import CLI, CP, HDFS_ROOT, TMP

ROOT = Path(__file__).resolve().parent.parent / "results"
SUCCESS = {"success", "done", "succeeded", "completed"}
TERMINAL = SUCCESS | {"failed", "stopped", "failed_to_launch"}


def call(command, request, output):
    response = subprocess.run(
        [CLI, "--control-plane", CP, *command, "--json", json.dumps(request)],
        capture_output=True, text=True, timeout=90,
    )
    output.write_text(response.stdout)
    output.with_suffix(".stderr.txt").write_text(response.stderr)
    if response.returncode:
        raise RuntimeError(f"CLI failed; private response in {output}")
    value = json.loads(response.stdout)
    if value.get("error") or value.get("ResponseBase", {}).get("code", 0):
        raise RuntimeError(f"Platform error; private response in {output}")
    return value


def hdfs(*arguments, timeout=300):
    response = subprocess.run(["hdfs", "dfs", *map(str, arguments)], timeout=timeout,
                              capture_output=True)
    if response.returncode:
        raise RuntimeError(response.stderr.decode()[-1000:])
    return response.stdout


def main():
    job = json.loads((ROOT / "single_gpu_submission.json").read_text())
    directory = ROOT / "jobs" / job["name"]
    directory.mkdir(exist_ok=True)
    (directory / "live").mkdir(exist_ok=True)
    status = None
    while True:
        data = call(["job-v2", "runs", "get"], {"sid": job["sid"]},
                    TMP / "single-gpu-status-raw.json")["data"]
        runtime = data.get("runtime_context", {})
        safe = dict(sid=job["sid"], name=job["name"], status=data["status"],
                    created_at=data.get("created_at"), trial_id=runtime.get("trial_id"),
                    trial_status=runtime.get("trial_status"), observed_epoch_s=time.time(),
                    world=1, hard_stop_epoch_s=job["hard_stop_epoch_s"])
        (directory / "status.json").write_text(json.dumps(safe, indent=2))
        if data["status"] != status:
            print(json.dumps(dict(event="STATUS", **safe)), flush=True)
            status = data["status"]
        if status in TERMINAL:
            break
        remaining = job["hard_stop_epoch_s"] - time.time()
        if remaining <= 5:
            result = call(["job-v2", "runs", "stop"],
                          {"sid": job["sid"], "stop_reason": "Authorized single-GPU 20-minute cap"},
                          TMP / "single-gpu-stop-raw.json")
            (directory / "cap_stop.json").write_text(
                json.dumps(dict(sid=job["sid"], requested_epoch_s=time.time())))
            print(json.dumps(dict(event="CAP_STOP", sid=job["sid"])), flush=True)
            # Recheck termination immediately; do not resubmit or extend budget.
            data = call(["job-v2", "runs", "get"], {"sid": job["sid"]},
                        TMP / "single-gpu-status-raw.json")["data"]
            if data["status"] not in TERMINAL:
                raise RuntimeError("Stop requested; termination needs final verification")
            safe.update(status=data["status"], observed_epoch_s=time.time())
            (directory / "status.json").write_text(json.dumps(safe, indent=2))
            status = data["status"]
            break
        time.sleep(min(15, max(1, remaining - 5)))

    path = f'{HDFS_ROOT}/runs/{job["name"]}'
    listing = hdfs("-ls", path).decode()
    archives = [line.split()[-1] for line in listing.splitlines()
                if line.split() and line.split()[-1].endswith(".tar.gz")]
    for source in archives:
        target = directory / source.rsplit("/", 1)[-1]
        if not target.exists():
            hdfs("-get", source, target)
    try:
        content = hdfs("-cat", f"{path}/live/single_gpu.result.json")
    except RuntimeError:
        content = None
    if content:
        (directory / "live/single_gpu.result.json").write_bytes(content)
    job.update(status=status, terminal_observed_epoch_s=time.time(),
               trial_id=safe["trial_id"], archives=[str(directory / a.rsplit("/", 1)[-1])
                                                   for a in archives])
    (directory / "job.json").write_text(json.dumps(job, indent=2))
    (ROOT / "single_gpu_submission.json").write_text(json.dumps(job, indent=2))
    print(json.dumps(dict(event="COLLECTED", sid=job["sid"], status=status,
                          archives=len(archives), complete_result=bool(content))), flush=True)
    if status not in SUCCESS or len(archives) != 1 or not content:
        raise RuntimeError("Single-GPU probe incomplete; inspect evidence before retry")


if __name__ == "__main__":
    main()
