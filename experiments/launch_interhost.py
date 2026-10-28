"""Bounded pair of two-GPU jobs, with host exclusion and a verified launch gate."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import jsonschema

from launch_single_gpu import invoke
from merlin_platform import DEPENDENCIES, FINISH_EPOCH, HDFS_ROOT, TMP, payload
from watch_single_gpu import call, hdfs, SUCCESS, TERMINAL

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
STATE = OUT / "interhost_probe_submission.json"


def save(value):
    temporary = STATE.with_suffix(".upload")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(STATE)


def event(kind, **values):
    print(json.dumps(dict(event=kind, **values), ensure_ascii=False), flush=True)


def request_for(state, rank, masks):
    name = f'{state["coord_name"]}-r{rank}'
    files = [ROOT / "experiments" / n for n in (
        "train.py", "cache_inputs.py", "contracts.py", "download_assets.py",
        "launch_distributed.py", "benchmark_communication.py")]
    deadline = int(state["hard_stop_epoch_s"]) - 20
    commands = f"""
export ARNOLD_ID={rank}
export DRIVING_RUN={state["coord_name"]}
export DRIVING_BENCH_DEADLINE={deadline}
mkdir -p "$DRIVING_STORE/runs/$DRIVING_RUN"
test -f "$DRIVING_STORE/assets/model.tar"
test -f "$DRIVING_STORE/assets/cache.tar"
""" + DEPENDENCIES + """
python3 - <<'EXTRACT'
import os
from pathlib import Path
from cache_inputs import extract
for name in ("model.tar", "cache.tar"):
 extract(Path(os.environ["DRIVING_STORE"])/"assets"/name, Path(os.environ["DRIVING_LOCAL"]))
EXTRACT
python3 - <<'PLACEMENT_GATE'
import os,time
from pathlib import Path
gate=Path(os.environ["DRIVING_STORE"])/"runs"/os.environ["DRIVING_RUN"]/"placement_verified.json"
while not gate.exists():
 if time.time()>float(os.environ["DRIVING_BENCH_DEADLINE"])-600:
  raise TimeoutError("Distinct-host placement not verified in preparation budget")
 time.sleep(2)
print("DISTINCT_HOST_PLACEMENT_GATE_PASSED",flush=True)
PLACEMENT_GATE
for channels in auto 8 16; do
 mkdir -p "$DRIVING_LOCAL/results/communication-$channels"
 export NCCL_DEBUG_FILE="$DRIVING_LOCAL/results/communication-$channels/nccl-%h-%p.log"
 if [[ "$channels" != "auto" ]]; then
  unset NCCL_MIN_CTAS NCCL_MAX_CTAS
  export NCCL_MIN_NCHANNELS="$channels" NCCL_MAX_NCHANNELS="$channels"
 fi
 python3 "$DRIVING_LOCAL/source/launch_distributed.py" --nodes 2 --gpus 2 --entry-script benchmark_communication.py -- --channel-mode "$channels" 2>&1 | tee "$DRIVING_LOCAL/results/communication-$channels/benchmark.log"
done
"""
    commands = commands.replace("for channels in auto 8 16;",
                                "for channels in " + " ".join(state["modes"]) + ";")
    if state.get("network_only"):
        files = [ROOT / "experiments" / n for n in ("launch_distributed.py", "benchmark_network.py")]
        gate = commands[commands.index("python3 - <<'PLACEMENT_GATE'"):commands.index("for channels in")]
        gate = gate.replace('float(os.environ["DRIVING_BENCH_DEADLINE"])-600',
                            'float(os.environ["DRIVING_BENCH_DEADLINE"])-90')
        commands = f"""
export ARNOLD_ID={rank}
export DRIVING_RUN={state["coord_name"]}
export DRIVING_BENCH_DEADLINE={deadline}
mkdir -p "$DRIVING_STORE/runs/$DRIVING_RUN" "$DRIVING_LOCAL/results/network"
""" + gate + """
export NCCL_DEBUG_FILE="$DRIVING_LOCAL/results/network/nccl-%h-%p.log"
python3 "$DRIVING_LOCAL/source/launch_distributed.py" --nodes 2 --gpus 1 --entry-script benchmark_network.py 2>&1 | tee "$DRIVING_LOCAL/results/network/benchmark.log"
"""
    seconds = min(1180, max(1, int(state["hard_stop_epoch_s"] - time.time() - 15)))
    request = payload(name, files, commands, state["dc"], 1, state["gpus_per_worker"], seconds,
                      "One half of a verified inter-host H20 probe; <=20min original round cap; "
                      "mask rank0 host for rank1; no training quality claim; "
                      + ("collectives only, two GPUs total." if state.get("network_only")
                         else "four GPUs total, real resident model specimens."))
    if masks:
        request["attachments"].append({
            "kind": "arnold_cluster_control",
            "arnold_cluster_control_attachment": {"mask_hosts": masks}})
    entry = request["job_config"]["job_template_config"]["entrypoint_full_script"]
    entry = entry.replace("--kill-after=60s", "--kill-after=10s", 1)
    entry = entry.replace(str(FINISH_EPOCH), str(deadline))
    # Coordination uses a shared path, but each one-pod job archives independently.
    entry = entry.replace(
        'root=Path(os.environ["DRIVING_STORE"])/"runs"/os.environ["DRIVING_RUN"]',
        f'root=Path(os.environ["DRIVING_STORE"])/"runs"/"{name}"')
    request["job_config"]["job_template_config"]["entrypoint_full_script"] = entry
    return request, files


def submit(state, rank, masks=()):
    request, files = request_for(state, rank, list(masks))
    name = request["name"]
    schema = json.loads((TMP / "single-gpu-create-schema.json").read_text())
    jsonschema.validate(request, schema)
    original = TMP / f"{name}.create.json"
    original.write_text(json.dumps(request, indent=2))
    invoke(["job-v2", "runs", "create", "--from-file", str(original), "--dry-run"],
           TMP / f"{name}.dry-run.json")
    pre = TMP / f"{name}.precheck-request.json"
    pre.write_text(json.dumps(dict(node_type="merlin_job_run", check_point="create",
                                  job_run_context=request)))
    checked = invoke(["job-v2", "check-items", "check", "--from-file", str(pre)],
                     TMP / f"{name}.precheck.json")["data"]
    messages = [dict(message=r.get("message"), suggestion=r.get("suggestion"))
                for r in checked.get("results", [])]
    event("PREFLIGHT", rank=rank, status=checked["status"], messages=messages)
    if checked["status"] == "error":
        raise RuntimeError("Preflight rejected; no submission")
    request["request_context"] = {
        **request.get("request_context", {}), "precheck_sid": checked["sid"]}
    final = TMP / f"{name}.checked.json"
    final.write_text(json.dumps(request, indent=2))
    if time.time() > state["placement_deadline_epoch_s"] - 30:
        raise TimeoutError("Placement submission budget exhausted")
    submitted = time.time()
    created = invoke(["job-v2", "runs", "create", "--from-file", str(final)],
                     TMP / f"{name}.result.json", timeout=120)
    created = created.get("data", created)
    job = dict(sid=created["sid"], name=name, rank=rank, dc=state["dc"],
               workers=1, gpus_per_worker=state["gpus_per_worker"], submitted_epoch_s=submitted,
               precheck_status=checked["status"], precheck_sid=checked["sid"],
               masked_host_sha256=[hashlib.sha256(h.encode()).hexdigest() for h in masks],
               source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in files})
    state["jobs"].append(job)
    save(state)
    event("CREATED", sid=job["sid"], name=name, rank=rank, mask_count=len(masks))
    return job


def refresh(job):
    data = call(["job-v2", "runs", "get"], {"sid": job["sid"]},
                TMP / f'{job["name"]}.status-raw.json')["data"]
    if data["status"] != job.get("status"):
        event("STATUS", sid=job["sid"], rank=job["rank"], status=data["status"])
    job.update(status=data["status"], created_at=data.get("created_at"),
               observed_epoch_s=time.time())
    trial = data.get("runtime_context", {}).get("trial_id")
    if not trial:
        return []
    job["trial_id"] = trial
    pods = call(["job-v2", "run-trial-pods", "list"],
                {"filter": {"trial_sid_any": [trial]}},
                TMP / f'{job["name"]}.pods-raw.json')["data"]
    job["pods"] = [
        dict(**{k: p.get(k) for k in ("sid", "pod_name", "start_at", "stop_at",
                                     "status", "exit_status")},
             physical_host_sha256=hashlib.sha256(
                 str(p.get("host_v6") or p.get("host")).encode()).hexdigest())
        for p in pods]
    return pods


def host_values(pods):
    return sorted({str(p[k]) for p in pods for k in ("host", "host_v6") if p.get(k)})


def stop_all(state, reason):
    for job in state["jobs"]:
        if job.get("status") in TERMINAL:
            continue
        call(["job-v2", "runs", "stop"], {"sid": job["sid"], "stop_reason": reason},
             TMP / f'{job["name"]}.stop-raw.json')
        event("STOP_REQUESTED", sid=job["sid"], reason=reason)
    save(state)


def epoch(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def collect(state):
    for job in state["jobs"]:
        directory = OUT / "jobs" / job["name"]
        directory.mkdir(exist_ok=True)
        archives = []
        try:
            listing = hdfs("-ls", f'{HDFS_ROOT}/runs/{job["name"]}').decode()
            paths = [line.split()[-1] for line in listing.splitlines()
                     if line.split() and line.split()[-1].endswith(".tar.gz")]
            for source in paths:
                target = directory / source.rsplit("/", 1)[-1]
                if not target.exists():
                    hdfs("-get", source, target)
                archives.append(str(target))
        except RuntimeError:
            event("ARCHIVES_UNAVAILABLE", sid=job["sid"])
        job["archives"] = archives
        (directory / "job.json").write_text(json.dumps(job, indent=2))
    pods = [p for job in state["jobs"] for p in job.get("pods", [])]
    charged_pods = [(job["gpus_per_worker"], p)
                    for attempt in state.get("previous_attempts", []) + [state]
                    for job in attempt["jobs"] for p in job.get("pods", [])]
    complete_times = all(p.get("start_at") and p.get("stop_at") for p in pods) and bool(pods)
    terminal = dict(
        jobs=[{k: job.get(k) for k in ("sid", "name", "rank", "status", "trial_id",
                                     "created_at", "pods", "archives")} for job in state["jobs"]],
        distinct_hosts=len({p["physical_host_sha256"] for p in pods}),
        GPU_hours=sum(gpus * (epoch(p["stop_at"]) - epoch(p["start_at"])) / 3600
                      for gpus, p in charged_pods) if complete_times else None,
        round_to_last_exit_s=max(epoch(p["stop_at"]) for p in pods) - state["started_epoch_s"]
        if complete_times else None,
        all_terminal=all(j.get("status") in TERMINAL for j in state["jobs"]),
        all_exit_zero=bool(pods) and all(p.get("exit_status") == 0 for p in pods),
        previous_attempts=state.get("previous_attempts", []),
        resource_release_evidence="Job terminal states and per-pod stop_at/exit_status",
        outcome=state["outcome"])
    (OUT / "interhost_probe_terminal.json").write_text(json.dumps(terminal, indent=2))
    save(state)
    event("COLLECTED", **{k: v for k, v in terminal.items() if k != "jobs"})


def main():
    assert TMP.is_dir() and OUT.is_dir() and (OUT / "jobs").is_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry-dc", choices=("hl",),
                        help="One alternate queue attempt inside the original twenty-minute budget")
    parser.add_argument("--network-only", action="store_true",
                        help="Final one-GPU-per-host collective probe after unsuccessful model placement")
    args = parser.parse_args()
    previous = None
    if STATE.exists():
        previous = json.loads(STATE.read_text())
        retry_count = len(previous.get("previous_attempts", []))
        if not args.retry_dc or previous["outcome"] != "incomplete" or retry_count >= 2 or (
                retry_count and not args.network_only):
            raise RuntimeError("Existing pair is not eligible for one bounded placement retry")
        for job in previous["jobs"]:
            refresh(job)
            if job["status"] not in TERMINAL or any(not p.get("stop_at") for p in job.get("pods", [])):
                raise RuntimeError("Previous placement attempt has not released its workers")
        (OUT / f"interhost_placement_attempt{retry_count+1}.json").write_text(json.dumps(previous, indent=2))
    elif args.retry_dc:
        raise RuntimeError("Retry requires an existing stopped placement attempt")
    started = previous["started_epoch_s"] if previous else time.time()
    if time.time() > started + 1200 - (180 if args.network_only else 700):
        raise TimeoutError("Insufficient original budget for another placement and measurement")
    dc = args.retry_dc or "lf"
    world = 2 if args.network_only else 4
    state = dict(coord_name=f"driving-interhost{world}-{dc}-" + time.strftime("%Y%m%d%H%M%S", time.gmtime()),
                 started_epoch_s=started, hard_stop_epoch_s=started + 1200,
                 placement_deadline_epoch_s=min(time.time() + (240 if previous else 480),
                                               started + 1200 - (120 if args.network_only else 480)),
                 dc=dc, modes=[] if args.network_only else ["8", "16"] if previous else ["auto", "8", "16"],
                 world=world, gpus_per_worker=world // 2, network_only=args.network_only,
                 previous_attempts=previous.get("previous_attempts", []) + [
                     {k: v for k, v in previous.items() if k != "previous_attempts"}] if previous else [],
                 maximum_concurrent_gpus=world, jobs=[], outcome="placing")
    save(state)
    try:
        first = submit(state, 0)
        while True:
            first_pods = refresh(first)
            save(state)
            if host_values(first_pods):
                break
            if first["status"] in TERMINAL:
                raise RuntimeError("First job terminated before placement")
            if time.time() > state["placement_deadline_epoch_s"] - 120:
                raise TimeoutError("First host placement budget exhausted")
            time.sleep(10)
        masks = host_values(first_pods)
        event("FIRST_HOST_FOUND", sid=first["sid"], host_sha256=first["pods"][0]["physical_host_sha256"])
        second = submit(state, 1, masks)
        while True:
            first_pods, second_pods = refresh(first), refresh(second)
            save(state)
            if host_values(second_pods):
                if set(host_values(first_pods)) & set(host_values(second_pods)):
                    raise RuntimeError("Host exclusion ignored; co-located pair rejected")
                hashes = {p["physical_host_sha256"] for j in state["jobs"] for p in j["pods"]}
                if len(hashes) != 2:
                    raise RuntimeError("Physical host identity not unambiguously distinct")
                gate = dict(verified_epoch_s=time.time(), physical_host_sha256=sorted(hashes),
                            jobs=[j["sid"] for j in state["jobs"]], mask_hosts_applied=True)
                gate_path = TMP / f'{state["coord_name"]}.placement_verified.json'
                gate_path.write_text(json.dumps(gate))
                remote = f'{HDFS_ROOT}/runs/{state["coord_name"]}'
                hdfs("-mkdir", "-p", remote)
                hdfs("-put", "-f", gate_path, f"{remote}/placement_verified.json")
                state.update(outcome="measuring", placement=gate)
                save(state)
                event("DISTINCT_HOSTS_VERIFIED", jobs=gate["jobs"], physical_hosts=2)
                break
            if any(j["status"] in TERMINAL for j in state["jobs"]):
                raise RuntimeError("A paired job terminated before distinct placement")
            if time.time() > state["placement_deadline_epoch_s"]:
                raise TimeoutError("Different-host placement deadline reached")
            time.sleep(10)
        while True:
            for job in state["jobs"]:
                refresh(job)
            save(state)
            if all(j["status"] in TERMINAL for j in state["jobs"]):
                state["outcome"] = "success" if all(j["status"] in SUCCESS for j in state["jobs"]) else "incomplete"
                break
            if any(j["status"] in TERMINAL and j["status"] not in SUCCESS for j in state["jobs"]):
                raise RuntimeError("One paired worker failed; stopping its peer")
            if time.time() >= state["hard_stop_epoch_s"] - 15:
                raise TimeoutError("Authorized twenty-minute round cap")
            time.sleep(10)
    except Exception as exc:
        state.update(outcome="incomplete", error=f"{type(exc).__name__}: {exc}")
        event("ROUND_INCOMPLETE", reason=state["error"])
        stop_all(state, "Bounded distinct-host probe: " + str(exc))
    finally:
        for job in state["jobs"]:
            refresh(job)
        collect(state)


if __name__ == "__main__":
    main()
