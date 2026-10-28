"""Persistent bounded orchestration: gather evidence, gates, final training, report."""
import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import urllib.request

from merlin_platform import (CLI, HDFS_ROOT, START_EPOCH, LAST_START_EPOCH, FINISH_EPOCH, TMP,
                             QUEUES, cli, submit, training_request)
from report import create_report, load_results

SOURCE = Path(__file__).parent
ROOT = SOURCE.parent/"results"
SUCCESS = {"done", "success", "succeeded", "completed"}
FAILURE = {"failed", "failed_to_launch", "stopped"}
TERMINAL = SUCCESS | FAILURE


def atomic(path, value):
    temporary = path.with_suffix(".upload")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def hdfs(*arguments, timeout=90):
    result = subprocess.run(["hdfs", "dfs", *map(str, arguments)], capture_output=True,
                            timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode()[-1000:])
    return result.stdout


class Controller:
    def __init__(self):
        ROOT.mkdir(exist_ok=True)
        (ROOT/"jobs").mkdir(exist_ok=True)
        (ROOT/"assets").mkdir(exist_ok=True)
        self.path = ROOT/"controller_state.json"
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = dict(start_epoch=START_EPOCH, last_start_epoch=LAST_START_EPOCH,
                finish_epoch=FINISH_EPOCH, events=[], phase="inputs", jobs=[
                    dict(sid="0a1970d0e4c5012b", suite="assets", world=1, dc="hl",
                         name="driving-assets-hl-20261028", attempt=1),
                    dict(sid="ea31c8d50f664e3a", suite="cache", world=1, dc="hl",
                         name="driving-cache-hl-20261028", attempt=1)])
        self.save()

    def save(self):
        self.state["updated_epoch_s"] = time.time()
        atomic(self.path, self.state)

    def event(self, event, **values):
        row = dict(event=event, timestamp=datetime.now(timezone.utc).isoformat(), **values)
        self.state["events"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        self.save()

    def directory(self, job):
        root = ROOT/"jobs"/job["name"]
        root.mkdir(exist_ok=True)
        (root/"live").mkdir(exist_ok=True)
        return root

    def query(self, job):
        response = cli("job-v2", "runs", "get", payload={"sid": job["sid"]})
        data = response["data"]
        previous = job.get("status")
        job.update(status=data["status"], trial=data.get("runtime_context", {}).get("trial_id"),
                   created_at=data.get("created_at"), trial_status=data.get("runtime_context", {}).get("trial_status"))
        if job["status"] == "running":
            job.setdefault("first_running_epoch_s", time.time())
        if job["status"] in TERMINAL:
            job.setdefault("observed_terminal_epoch_s", time.time())
        directory = self.directory(job)
        limit_notice = TMP/(job["sid"]+".wall-limit.json")
        if limit_notice.exists() and json.loads(limit_notice.read_text()).get("sid") == job["sid"]:
            job["user_time_limit_reached"] = True
        atomic(directory/"status.json", response)
        atomic(directory/"job.json", job)
        if previous != job["status"]:
            self.event("JOB_STATUS", sid=job["sid"], suite=job["suite"],
                       status=job["status"], trial_status=job["trial_status"])
        if (time.time() > LAST_START_EPOCH and not job.get("user_wall_limit_s") and
                job["status"] in {"started", "launching", "queued"}):
            self.stop(job, "last-start deadline passed while staging")
        if job["status"] in FAILURE and job.get("user_time_limit_reached"):
            if not job.get("failure_reviewed"):
                self.collect(job)
                self.logs(job, failure=True)
                job["failure_reviewed"] = True
                self.event("USER_WALL_LIMIT_TERMINAL", sid=job["sid"], status=job["status"])
            return
        if job["status"] in FAILURE and not job.get("failure_reviewed"):
            self.logs(job, failure=True)
            job["failure_reviewed"] = True
            self.event("REQUIRES_FIX", sid=job["sid"], suite=job["suite"],
                       evidence=str(directory), status=job["status"])
            self.state["phase"] = "needs_fix"
            self.save()
            raise RuntimeError(f'{job["suite"]} failed; evidence collected at {directory}')
        if job["status"] in FAILURE and not job.get("deadline_stopped"):
            raise RuntimeError(f'{job["suite"]} remains failed; fix and supersede before continuing')
        if job["status"] in SUCCESS and not job.get("collected"):
            self.collect(job, archives=True)
            self.logs(job)
            job["collected"] = True
            self.event("ARTIFACTS_COLLECTED", sid=job["sid"], suite=job["suite"])
        elif job["status"] == "running" and time.time()-job.get("last_collect_epoch", 0) > 600:
            self.collect(job)
            job["last_collect_epoch"] = time.time()

    def logs(self, job, failure=False):
        directory = self.directory(job)
        pods = cli("job-v2", "run-trial-pods", "list",
                   payload={"filter": {"trial_sid_any": [job["trial"]]}})
        atomic(directory/"pods.json", pods)
        if failure:
            exited = cli("job-v2", "run-trials", "exit-info",
                         payload={"trials_list": {"filter": {"job_run_sid_any": [job["sid"]]}}})
            atomic(directory/"exit-info.json", exited)
        for pod in pods.get("data", []):
            for stream in ("stdout", "stderr"):
                response = cli("job-v2", "run-trial-local-logs", "get",
                    payload={"trial_pod_sid": pod["sid"], "stream": stream})
                url = response.get("data", {}).get("url")
                if url:
                    content = urllib.request.urlopen(url, timeout=60).read()
                    (directory/f'{pod["pod_name"]}.{stream}.log').write_bytes(content)
                else:
                    self.event("LOCAL_LOG_UNAVAILABLE", sid=job["sid"], pod=pod["sid"], stream=stream)

    def collect(self, job, archives=False):
        directory = self.directory(job)
        path = f'{HDFS_ROOT}/runs/{job["name"]}'
        try:
            listing = hdfs("-ls", f"{path}/live").decode()
        except RuntimeError:
            listing = ""
        for line in listing.splitlines():
            parts = line.split()
            if len(parts) < 8:
                continue
            source = parts[-1]
            name = source.rsplit("/", 1)[-1]
            if name.endswith(".json"):
                target = directory/"live"/name
                temporary = target.with_suffix(".upload")
                temporary.write_bytes(hdfs("-cat", source))
                temporary.replace(target)
        if archives:
            listing = hdfs("-ls", path).decode()
            archives_found = []
            for line in listing.splitlines():
                parts = line.split()
                if len(parts) < 8 or not parts[-1].endswith(".tar.gz"):
                    continue
                source = parts[-1]
                target = directory/source.rsplit("/", 1)[-1]
                if not target.exists():
                    hdfs("-get", source, target, timeout=600)
                archives_found.append(str(target))
                with tarfile.open(target) as tar:
                    for member in tar:
                        if member.isfile() and member.name.endswith(("result.json", "exit_code.txt", "cache_manifest.json")):
                            content = tar.extractfile(member).read()
                            if member.name.endswith("exit_code.txt") and content.strip() != b"0":
                                raise RuntimeError(f"Archived nonzero exit code: {target}: {content!r}")
            expected = max(1, job["world"]//8)
            if len(archives_found) != expected:
                raise RuntimeError(f"Completed pod archive count {len(archives_found)} != {expected}: {job['sid']}")
            job["archives"] = archives_found
            if job["suite"] == "final":
                checkpoint = ROOT/"final_checkpoint.pt"
                source = f'{path}/live/{job["case"]}.checkpoint.pt'
                if not checkpoint.exists():
                    hdfs("-get", source, checkpoint, timeout=300)
                result = json.loads((directory/"live"/f'{job["case"]}.result.json').read_text())
                digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                assert digest == result["checkpoint_sha256"], "Final checkpoint transfer hash mismatch"
                assert (result.get("checkpoint_reload_probe") or {}).get("verified"), "Final checkpoint reload unverified"
                atomic(ROOT/"reproduce.json", dict(checkpoint="final_checkpoint.pt", sha256=digest,
                    model_revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
                    base_model_archive=f"{HDFS_ROOT}/assets/model.tar",
                    cached_input_archive=f"{HDFS_ROOT}/assets/cache.tar",
                    cache_manifest_sha256=result["environment"].get("cache_manifest_sha256"),
                    case=job["case"], selected_step=result.get("selected_checkpoint_step"),
                    checkpoint_reload_probe=result["checkpoint_reload_probe"],
                    source_directory=str(SOURCE), checkpoint_scope="trainable LoRA/head weights; frozen base stored separately"))

    def assets(self):
        for name in ("download_manifest.json", "cache_manifest.json", "records.json", "previews.tar.gz"):
            target = ROOT/"assets"/name
            if not target.exists():
                try:
                    hdfs("-get", f"{HDFS_ROOT}/assets/{name}", target, timeout=300)
                except RuntimeError:
                    pass

    def refresh_cache_source(self, job):
        # Before inputs are ready, replace the reverse-index contract and preview export.
        if job.get("source_refreshed") or job["status"] != "running":
            return
        files = {name: base64.b64encode(gzip.compress((SOURCE/name).read_bytes())).decode()
                 for name in ("contracts.py", "cache_inputs.py")}
        encoded = base64.b64encode(json.dumps(files).encode()).decode()
        code = ("import os,json,base64,gzip; from pathlib import Path; "
                f"files=json.loads(base64.b64decode('{encoded}')); "
                f"root=Path('/tmp/driving-{job['name']}/source'); "
                "[(root/n).write_bytes(gzip.decompress(base64.b64decode(v))) for n,v in files.items()]; "
                "print('CACHE_SOURCE_REFRESHED',list(files),flush=True)")
        command = [CLI, "--control-plane", "cn", "job-v2", "runs", "execute-script", "--json",
                   json.dumps(dict(job_run_id=job["sid"], exec="python3 -c '"+code+"'",
                                   exec_timeout_seconds=60))]
        # Base64 uses no single quotes; code strings above need shell-safe quoting.
        command[-1] = json.dumps(dict(job_run_id=job["sid"],
            exec="python3 -c "+shell_quote(code), exec_timeout_seconds=60))
        response = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (self.directory(job)/"source-refresh.log").write_text(response.stdout+response.stderr)
        if response.returncode:
            raise RuntimeError("Cache source refresh failed: "+response.stderr[-1000:])
        job["source_refreshed"] = {n: hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in files}
        self.event("CACHE_SOURCE_REFRESHED", sid=job["sid"], hashes=job["source_refreshed"])

    def latest(self, suite):
        return next((j for j in reversed(self.state["jobs"]) if j["suite"] == suite and not j.get("superseded")), None)

    def completed(self, suite):
        job = self.latest(suite)
        return bool(job and job.get("status") in SUCCESS and job.get("collected"))

    def launch(self, suite, world, seconds, case="lora_head", dc="lf"):
        if time.time() > LAST_START_EPOCH-600:
            raise RuntimeError("No new task: last-start budget has less than ten minutes")
        attempt = sum(j["suite"] == suite for j in self.state["jobs"])+1
        request = training_request(suite, world, seconds, case, attempt, dc)
        response = submit(request)
        job = dict(sid=response["sid"], name=request["name"], suite=suite, world=world,
                   seconds=seconds, case=case, dc=dc, attempt=attempt, submitted_epoch_s=time.time())
        self.state["jobs"].append(job)
        self.event("TASK_SUBMITTED", **job)
        self.query(job)
        return job

    def queue_inventory(self):
        response = cli("resource", "my-group-resources", "list", payload={
            "filter": {"group_sid": "2079", "gpu_type": "NVIDIA-H20"},
            "page_size": 100})
        atomic(ROOT/"latest_capacity.json", response)
        rows = response.get("data", [])
        inventory = {}
        for dc, (cluster, queue) in QUEUES.items():
            matched = [r for r in rows if r.get("queue_name") == queue and
                       str(r.get("cluster_sid")) == str(cluster)]
            inventory[dc] = dict(
                H20=sum(float(r.get("rest_gpu", {}).get("NVIDIA-H20", 0)) for r in matched),
                cpu=sum(float(r.get("rest_cpu", 0)) for r in matched),
                memory_mb=sum(float(r.get("rest_memory", 0)) for r in matched))
        request_id = response.get("ResponseBase", {}).get("meta", {}).get("request_id")
        return inventory, request_id

    @staticmethod
    def fits(capacity, world):
        return (capacity["H20"] >= world and capacity["cpu"] >= world*20 and
                capacity["memory_mb"] >= world*230400)

    def diagnostic_dc(self):
        job = self.latest("diagnose8")
        return self.state.get("diagnostic_dc") or (job or {}).get("dc") or "lf"

    def choose_diagnostic_queue(self):
        inventory, request_id = self.queue_inventory()
        eligible = [dc for dc in QUEUES if self.fits(inventory[dc], 8)]
        self.event("DIAGNOSTIC_QUEUE_CAPACITY", inventory=inventory, requested=8,
                   control_plane="cn", request_id=request_id)
        if not eligible:
            return None
        # Prefer simultaneous 8+16 coverage; otherwise use the largest eligible pool.
        dc = max(eligible, key=lambda d:(self.fits(inventory[d], 24), inventory[d]["H20"], d == "lf"))
        self.state["diagnostic_dc"] = dc
        self.event("DIAGNOSTIC_QUEUE_SELECTED", dc=dc, cluster=QUEUES[dc][0], queue=QUEUES[dc][1],
                   capacity=inventory[dc], reason="One authorized H20 queue, locked before the first diagnosis")
        return dc

    def capacity(self, world, dc=None):
        dc = dc or self.diagnostic_dc()
        inventory, request_id = self.queue_inventory()
        self.event("CAPACITY", available_H20=inventory[dc]["H20"], available_cpu=inventory[dc]["cpu"],
                   available_memory_mb=inventory[dc]["memory_mb"], requested=world, dc=dc,
                   control_plane="cn", request_id=request_id)
        return self.fits(inventory[dc], world)

    def stop(self, job, reason):
        response = cli("job-v2", "runs", "stop", payload={"sid": job["sid"], "stop_reason": reason})
        atomic(self.directory(job)/"stop.json", response)
        job["deadline_stopped"] = True
        self.event("TASK_STOPPED", sid=job["sid"], reason=reason)

    def advance(self):
        requested = self.state.get("requires_user_decision_after")
        if requested:
            job = self.latest(requested)
            ended = self.completed(requested) or (
                job and job.get("user_time_limit_reached") and job.get("status") in TERMINAL)
            if ended:
                self.state["phase"] = "awaiting_user_decision"
                if not self.state.get("user_validation_ready_notified"):
                    self.state["user_validation_ready_notified"] = True
                    self.state["decisions"] = create_report(ROOT)
                    self.event("USER_VALIDATION_READY", sid=job["sid"], suite=requested,
                               status=job["status"], report=str(ROOT/"report.md"),
                               continuation="User decides whether additional tasks run")
            return
        self.assets()
        cache_job = self.latest("cache")
        self.refresh_cache_source(cache_job)
        if not self.completed("cache"):
            return
        if not self.latest("sanity"):
            self.launch("sanity", 1, 60, dc="hl")
            self.state["phase"] = "sanity"
            return
        if not self.completed("sanity"):
            return
        if not self.latest("diagnose8"):
            if time.time()-self.state.get("diagnose8_capacity_checked_epoch", 0) < 300:
                return
            self.state["diagnose8_capacity_checked_epoch"] = time.time()
            dc = self.choose_diagnostic_queue()
            if dc is None:
                return
            self.launch("diagnose8", 8, 1800, dc=dc)
            self.state["phase"] = "diagnosis"
        if not self.latest("diagnose16"):
            if time.time()-self.state.get("diagnose16_capacity_checked_epoch", 0) < 300:
                return
            self.state["diagnose16_capacity_checked_epoch"] = time.time()
            if self.capacity(16):
                self.launch("diagnose16", 16, 3600, dc=self.diagnostic_dc())
                self.state["phase"] = "diagnosis"
            elif not self.state.get("diagnose16_deferred"):
                self.state["diagnose16_deferred"] = True
                self.event("DIAGNOSE16_DEFERRED", dc=self.diagnostic_dc(),
                           reason="Requires 16 currently available H20 on the same diagnostic queue")
            return
        if not self.completed("diagnose8") or not self.completed("diagnose16"):
            return
        gates = create_report(ROOT)
        self.state["decisions"] = gates
        if not self.state.get("scale32_decided"):
            self.state["scale32_decided"] = True
            if gates.get("allow32") and self.capacity(32) and time.time() < LAST_START_EPOCH-7200:
                self.launch("scale32", 32, 3600, dc=self.diagnostic_dc())
                self.state["phase"] = "scale32"
                return
            self.event("SCALE32_SKIPPED", gates=gates.get("gate32"),
                       reason="Gate or current single-queue capacity or deadline did not permit validation")
        if self.latest("scale32") and not self.completed("scale32"):
            return
        if not self.state.get("scale64_decided"):
            gates = create_report(ROOT)
            self.state["scale64_decided"] = True
            if gates.get("allow64") and self.capacity(64) and time.time() < LAST_START_EPOCH-3600:
                self.launch("scale64", 64, 1800, dc=self.diagnostic_dc())
                self.state["phase"] = "scale64"
                return
            self.event("SCALE64_SKIPPED", gates=gates.get("gate64"),
                       reason="Requires successful 32 validation, stricter quality/cost gates, capacity and deadline")
        if self.latest("scale64") and not self.completed("scale64"):
            return
        if not self.latest("final"):
            if time.time()-self.state.get("final_capacity_checked_epoch", 0) < 300:
                return
            self.state["final_capacity_checked_epoch"] = time.time()
            candidates = [r for r in load_results(ROOT) if r["suite"] == "diagnose8" and
                          r["environment"]["mode"] == "train"]
            if not candidates:
                raise RuntimeError("No verified algorithm probe available for final training")
            best = min(candidates, key=lambda r:r["final"]["ADE_m"])
            case = best["environment"]["case"]
            world = 32 if self.completed("scale32") and self.capacity(32) else 16 if self.capacity(16) else 8
            if not self.capacity(world):
                if not self.state.get("final_deferred"):
                    self.state["final_deferred"] = True
                    self.event("FINAL_DEFERRED", dc=self.diagnostic_dc(),
                               reason="Wait for current queue resources to release before final training")
                return
            seconds = min(12600, int(FINISH_EPOCH-time.time()-1800))
            if seconds < 1800:
                raise RuntimeError("Less than 30 minutes of final training remains in deadline")
            self.event("FINAL_SELECTED", case=case, metric="lowest validation ADE after equal 30min probes",
                       ADE_m=best["final"]["ADE_m"], world=world, seconds=seconds)
            self.launch("final", world, seconds, case=case, dc=self.diagnostic_dc())
            self.state["phase"] = "final"
            return
        if self.completed("final"):
            create_report(ROOT, final=True)
            self.state["phase"] = "complete"
            self.state["completed_epoch_s"] = time.time()
            self.event("DRIVING_PROJECT_COMPLETE", report=str(ROOT/"report.md"),
                       elapsed_h=(time.time()-START_EPOCH)/3600)
            create_report(ROOT, final=True)

    def loop(self):
        while self.state.get("phase") != "complete":
            if time.time() > FINISH_EPOCH-300:
                for job in self.state["jobs"]:
                    if job.get("status") not in TERMINAL and not job.get("superseded"):
                        self.stop(job, "14-hour finish deadline")
                self.state["phase"] = "deadline"
                self.save()
                create_report(ROOT, final=True)
                raise RuntimeError("Deadline reached; incomplete experiments identified in report")
            for job in list(self.state["jobs"]):
                if not job.get("superseded") and not (job.get("status") in SUCCESS and job.get("collected")):
                    if (job.get("hard_stop_epoch_s") and job.get("status") not in TERMINAL and
                            not job.get("user_time_limit_reached") and
                            time.time() >= job["hard_stop_epoch_s"]-120):
                        job["user_time_limit_reached"] = True
                        self.save()
                        self.stop(job, "User requested a maximum two-hour validation task")
                    self.query(job)
            self.advance()
            self.save()
            if (self.state.get("phase") == "awaiting_user_decision" and
                    all(j.get("status") in TERMINAL for j in self.state["jobs"]
                        if not j.get("superseded"))):
                create_report(ROOT)
                return
            if self.state.get("phase") != "complete":
                time.sleep(45)


def shell_quote(value):
    return "'"+value.replace("'", "'\"'\"'")+"'"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    controller = Controller()
    try:
        if args.once:
            for job in controller.state["jobs"]:
                if not job.get("superseded"):
                    controller.query(job)
            controller.advance()
            controller.save()
        else:
            controller.loop()
    except Exception as exc:
        controller.event("CONTROLLER_ERROR", error=repr(exc))
        create_report(ROOT)
        raise
