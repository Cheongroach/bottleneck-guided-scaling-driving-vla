"""Submit/observe/collect a separate, bounded one-H20 GC diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import jsonschema

from launch_single_gpu import invoke
from merlin_platform import DEPENDENCIES, FINISH_EPOCH, HDFS_ROOT, TMP, payload
from watch_single_gpu import call, hdfs, TERMINAL

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "results/gc_probe_submission.json"


def submit():
    if STATE.exists():
        raise RuntimeError("Existing GC probe state; observe or collect it")
    name = "driving-gc1-hl-" + time.strftime("%Y%m%d%H%M%S", time.gmtime())
    started = time.time()
    deadline = int(started + 1180)
    files = [ROOT / "experiments" / n for n in (
        "train.py", "cache_inputs.py", "contracts.py", "download_assets.py", "benchmark_gc.py")]
    commands = f"""
export DRIVING_BENCH_DEADLINE={deadline - 30}
test -f "$DRIVING_STORE/assets/model.tar"
test -f "$DRIVING_STORE/assets/cache.tar"
""" + DEPENDENCIES + """
python3 "$DRIVING_LOCAL/source/benchmark_gc.py" 2>&1 | tee "$DRIVING_LOCAL/results/benchmark.log"
"""
    request = payload(name, files, commands, "hl", 1, 1, 1160,
        "One H20 only, 20min creation-to-stop cap; instrument Python GC and paired "
        "GC enabled/disabled at fixed micro2 checkpoint-on; checkpoint-off control; no training-quality claim.")
    entry = request["job_config"]["job_template_config"]["entrypoint_full_script"]
    entry = entry.replace("--kill-after=60s", "--kill-after=10s", 1).replace(str(FINISH_EPOCH), str(deadline))
    request["job_config"]["job_template_config"]["entrypoint_full_script"] = entry
    jsonschema.validate(request, json.loads((TMP / "single-gpu-create-schema.json").read_text()))
    original = TMP / f"{name}.create.json"
    original.write_text(json.dumps(request, indent=2))
    invoke(["job-v2", "runs", "create", "--from-file", str(original), "--dry-run"],
           TMP / f"{name}.dry-run.json")
    pre = TMP / f"{name}.precheck-request.json"
    pre.write_text(json.dumps(dict(node_type="merlin_job_run", check_point="create", job_run_context=request)))
    checked = invoke(["job-v2", "check-items", "check", "--from-file", str(pre)],
                     TMP / f"{name}.precheck.json")["data"]
    print(json.dumps(dict(event="PREFLIGHT", status=checked["status"], messages=[
        dict(message=x.get("message"), suggestion=x.get("suggestion")) for x in checked.get("results", [])]),
        ensure_ascii=False), flush=True)
    if checked["status"] == "error":
        raise RuntimeError("Preflight failed; no task submitted")
    request["request_context"] = {**request.get("request_context", {}), "precheck_sid": checked["sid"]}
    final = TMP / f"{name}.checked.json"
    final.write_text(json.dumps(request, indent=2))
    created = invoke(["job-v2", "runs", "create", "--from-file", str(final)],
                     TMP / f"{name}.result.json")
    created = created.get("data", created)
    safe = dict(sid=created["sid"], name=name, world=1, dc="hl", submitted_epoch_s=time.time(),
                hard_stop_epoch_s=deadline + 10, entry_timeout_s=1160,
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                precheck_status=checked["status"], precheck_sid=checked["sid"])
    STATE.write_text(json.dumps(safe, indent=2))
    print(json.dumps(dict(event="CREATED", **safe)), flush=True)


def observe(collect=False):
    job = json.loads(STATE.read_text())
    data = call(["job-v2", "runs", "get"], {"sid": job["sid"]},
                TMP / f"{job['name']}.status-raw.json")["data"]
    if data["status"] not in TERMINAL and time.time() > job["hard_stop_epoch_s"] - 10:
        call(["job-v2", "runs", "stop"], {"sid": job["sid"], "stop_reason": "GC diagnostic 20-minute cap"},
             TMP / f"{job['name']}.stop-raw.json")
        data = call(["job-v2", "runs", "get"], {"sid": job["sid"]},
                    TMP / f"{job['name']}.release-raw.json")["data"]
    runtime = data.get("runtime_context", {})
    safe = dict(sid=job["sid"], name=job["name"], status=data["status"],
                trial_id=runtime.get("trial_id"), trial_status=runtime.get("trial_status"),
                observed_epoch_s=time.time())
    print(json.dumps(safe), flush=True)
    job.update(safe)
    STATE.write_text(json.dumps(job, indent=2))
    if collect:
        if data["status"] not in TERMINAL:
            raise RuntimeError("Task not terminal; collect later")
        directory = ROOT / "results/jobs" / job["name"]
        directory.mkdir(exist_ok=True)
        (directory / "live").mkdir(exist_ok=True)
        listing = hdfs("-ls", f"{HDFS_ROOT}/runs/{job['name']}").decode()
        sources = [s.split()[-1] for s in listing.splitlines() if s.split() and s.split()[-1].endswith(".tar.gz")]
        for source in sources:
            target = directory / source.rsplit("/", 1)[-1]
            if not target.exists():
                hdfs("-get", source, target)
        content = hdfs("-cat", f"{HDFS_ROOT}/runs/{job['name']}/live/gc_probe.result.json")
        (directory / "live/gc_probe.result.json").write_bytes(content)
        job["archives"] = [str(directory / s.rsplit("/", 1)[-1]) for s in sources]
        STATE.write_text(json.dumps(job, indent=2))
        (directory / "job.json").write_text(json.dumps(job, indent=2))
        print(json.dumps(dict(event="COLLECTED", archives=len(sources))), flush=True)


def watch():
    while True:
        observe()
        job = json.loads(STATE.read_text())
        if job["status"] in TERMINAL:
            observe(collect=True)
            return
        time.sleep(min(15, max(1, job["hard_stop_epoch_s"] - time.time() - 10)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("submit", "observe", "collect", "watch"))
    arg = parser.parse_args().action
    if arg == "submit":
        submit()
    elif arg == "watch":
        watch()
    else:
        observe(arg == "collect")
