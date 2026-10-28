"""Audit completed campaign artifacts independently of the controller exit code."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile

REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
CASES = {"trajectory_only", "trajectory_action", "frozen_head", "state_only",
         "image_only", "lora_head", "lora_no_rationale"}
PROFILES = {"profile_synthetic", "profile_cached"}
SUCCESS = {"done", "success", "succeeded", "completed"}


def step_evidence(rows, global_batch, excluded):
    measured = rows[excluded:]
    duration = sum(r["step_s"] for r in measured)
    return dict(
        steps=len(rows),
        contiguous_steps=[r["step"] for r in rows] == list(range(1, len(rows)+1)),
        throughput_samples_s=len(measured)*global_batch/duration,
        data_wait_fraction=sum(r["data_wait_s"] for r in measured)/duration,
        train_step_total_s=sum(r["step_s"] for r in rows),
        throughput_measured_steps=len(measured),
    )


def kernel_evidence(trace):
    kernels = [e for e in trace["traceEvents"]
               if e.get("cat") == "kernel" and e.get("ph") == "X"]
    communication = [e for e in kernels if "nccl" in e.get("name", "").lower()]

    def occupied_time(events):
        boundaries = sorted(point for e in events
                            for point in ((e["ts"], 1), (e["ts"]+e["dur"], -1)))
        active, previous, total = 0, None, 0
        for timestamp, change in boundaries:
            if active:
                total += timestamp-previous
            active += change
            previous = timestamp
        return total/1e6

    return dict(kernel_count=len(kernels), nccl_kernel_count=len(communication),
                gpu_kernel_union_s=occupied_time(kernels),
                nccl_union_s=occupied_time(communication))


def audit(project):
    root = project/"results"
    checks = []

    def check(requirement, passed, evidence):
        checks.append(dict(requirement=requirement, passed=bool(passed), evidence=evidence))

    def read(path):
        return json.loads(path.read_text())

    def finite(value):
        return isinstance(value, (int, float)) and math.isfinite(value)

    def close(a, b):
        return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8)

    state = read(root/"controller_state.json")
    check("Controller completed the campaign", state.get("phase") == "complete",
          dict(phase=state.get("phase"), completed_epoch_s=state.get("completed_epoch_s")))
    check("Campaign completed within fourteen hours",
          finite(state.get("completed_epoch_s")) and state["completed_epoch_s"] <= state["finish_epoch"],
          {k: state.get(k) for k in ("start_epoch", "finish_epoch", "completed_epoch_s")})
    active = {j["suite"]: j for j in state["jobs"] if not j.get("superseded")}
    for suite in ("assets", "cache", "sanity", "diagnose8", "diagnose16", "final"):
        check(f"{suite} succeeded and was collected",
              suite in active and active[suite].get("status") in SUCCESS and
              active[suite].get("collected") is True, active.get(suite))
    worlds = dict(assets=1, cache=1, sanity=1, diagnose8=8, diagnose16=16,
                  scale32=32, scale64=64)
    for suite, job in active.items():
        expected = worlds.get(suite)
        check(f"{suite} requested the required GPU count",
              job["world"] == expected if expected else job["world"] in (8, 16, 32),
              dict(actual=job["world"], expected=expected or [8, 16, 32]))

    cache = read(root/"assets/cache_manifest.json")
    records_path = root/"assets/records.json"
    records = read(records_path)
    scenes = {split: {r["scene"] for r in records if r["split"] == split} for split in ("train", "val")}
    check("Official eight/two scene split has no scene leakage",
          len(scenes["train"]) == 8 and len(scenes["val"]) == 2 and not scenes["train"] & scenes["val"] and
          all(scenes[s] == set(cache[s+"_scenes"]) for s in scenes),
          {s: sorted(names) for s, names in scenes.items()})
    check("Every sample has six cameras and six two-dimensional targets",
          records and all(len(r["images"]) == 6 and len(r["trajectory"]) == 6 and
                          all(len(p) == 2 for p in r["trajectory"]) for r in records),
          dict(records=len(records), train=cache["train"], val=cache["val"]))
    offsets = [abs((t-r["timestamp_s"])-(i+1)*.5)
               for r in records for i, t in enumerate(r["future_times_s"])]
    margin = min(r["future_times_s"][0]-r["input_max_timestamp_s"] for r in records)
    check("All targets preserve half-second offsets and follow every camera input",
          margin > 0 and max(offsets) <= 1e-6,
          dict(minimum_future_input_margin_s=margin, maximum_offset_error_s=max(offsets)))
    check("The real pinned Qwen checkpoint was used", cache.get("model_revision") == REVISION, REVISION)
    check("Teacher and visual cache costs were measured",
          all(finite(cache.get(k)) and cache[k] > 0
              for k in ("teacher_total_s", "vision_total_s", "preparation_wall_s")) and
          cache.get("teacher_valid", 0) > 0,
          {k: cache.get(k) for k in ("teacher_count", "teacher_valid", "teacher_invalid",
                                   "teacher_total_s", "vision_total_s", "preparation_wall_s")})
    teacher = read(root/"teacher_diagnostics.json")
    check("Teacher diagnostics describe the exact published records",
          teacher["records_sha256"] == hashlib.sha256(records_path.read_bytes()).hexdigest() and
          teacher["overall"]["samples"] == len(records) == cache["train"]+cache["val"] and
          all(teacher["splits"][s]["samples"] == cache[s] for s in scenes) and
          teacher["overall"]["parsed_teachers"] == cache["teacher_valid"],
          dict(records_sha256=teacher["records_sha256"], overall=teacher["overall"]))

    suites = {}
    for suite, job in active.items():
        directory = root/"jobs"/job["name"]
        live = directory/"live"
        suites[suite] = {p.name.removesuffix(".result.json"): read(p)
                         for p in live.glob("*.result.json")}
        archives = list(directory.glob("*.tar.gz"))
        exits, training, duplicates = [], {}, []
        for archive in archives:
            with tarfile.open(archive) as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    if member.name.endswith("exit_code.txt"):
                        exits.append(tar.extractfile(member).read().strip().decode())
                    parts = Path(member.name).parts
                    if (len(parts) == 4 and parts[:2] == ("results", "training") and
                            parts[-1] in ("result.json", "steps.json", "trace.json")):
                        artifact = training.setdefault(parts[2], {})
                        if parts[-1] in artifact:
                            duplicates.append(member.name)
                        artifact[parts[-1]] = json.load(tar.extractfile(member))
        check(f"{suite} published one successful archive per pod",
              len(archives) == max(1, job["world"]//8) and len(exits) == len(archives) and set(exits) == {"0"},
              dict(archives=[p.name for p in archives], exit_codes=exits))
        if suite in ("assets", "cache"):
            continue
        marker_path = live/"suite_complete.json"
        marker = read(marker_path) if marker_path.exists() else {}
        check(f"{suite} finished before the campaign deadline",
              marker.get("suite") == suite and finite(marker.get("completed_epoch_s")) and
              marker["completed_epoch_s"] <= state["finish_epoch"], marker)
        check(f"{suite} started before the ten-hour deadline",
              finite(job.get("first_running_epoch_s")) and
              job["first_running_epoch_s"] <= state["last_start_epoch"], job.get("first_running_epoch_s"))
        check(f"{suite} has one archived record for every published case",
              not duplicates and set(training) == set(suites[suite]),
              dict(archived_cases=sorted(training), published_cases=sorted(suites[suite]),
                   duplicates=duplicates))
        for case, result in suites[suite].items():
            env, system = result["environment"], result["system"]
            archived = training.get(case, {})
            check(f"{suite}/{case} live summary matches the completed archive",
                  archived.get("result.json") == result, dict(archive_files=sorted(archived)))
            rows = archived.get("steps.json", [])
            excluded = system.get("throughput_excluded_steps", 0)
            usable = (rows and isinstance(excluded, int) and 0 <= excluded < len(rows) and
                      all(finite(r.get("step_s")) and r["step_s"] > 0 and
                          finite(r.get("data_wait_s")) and r["data_wait_s"] >= 0 for r in rows))
            measured = step_evidence(rows, env["global_batch"], excluded) if usable else {}
            check(f"{suite}/{case} throughput and data wait match archived optimizer steps",
                  measured.get("contiguous_steps") is True and all(
                      close(measured.get(k), system.get(k)) for k in (
                          "steps", "throughput_samples_s", "data_wait_fraction",
                          "train_step_total_s", "throughput_measured_steps")), measured)
            if suite != "sanity":
                check(f"{suite}/{case} proved the configured training and checkpointing modes",
                      env.get("policy_training") is True and env.get("language_training") is True and
                      env.get("gradient_checkpointing_enabled") is bool(env["config"]["lora"]) and
                      env.get("gradient_checkpointing_active") is bool(env["config"]["lora"]),
                      {k: env.get(k) for k in ("policy_training", "language_training",
                                               "gradient_checkpointing_enabled", "gradient_checkpointing_active")})
            check(f"{suite}/{case} ran on the requested H20 ranks",
                  env["world"] == job["world"] and "H20" in env["device"] and
                  len(env["hosts"]) == env["world"] and len(set(env["hosts"])) == max(1, env["world"]//8),
                  dict(world=env["world"], device=env["device"], hosts=env["hosts"]))
            digests = result.get("rank_head_sha256", [])
            check(f"{suite}/{case} synchronized heads and preserved frozen weights",
                  len(digests) == env["world"] and len(set(digests)) == 1 and
                  result["frozen_weight_probe"]["max_abs_change"] == 0 and
                  result.get("changed_parameter_tensors", 0) > 0,
                  dict(rank_head_sha256=digests, frozen=result["frozen_weight_probe"],
                       changed_tensors=result.get("changed_parameter_tensors")))
            check(f"{suite}/{case} recorded throughput and training cost",
                  all(finite(system.get(k)) and system[k] > 0 for k in
                      ("throughput_samples_s", "train_step_total_s", "peak_allocated_GiB")) and
                  system["samples_seen"] == system["steps"]*env["global_batch"],
                  {k: system.get(k) for k in ("throughput_samples_s", "steps", "samples_seen",
                                             "train_window_s", "eval_total_s", "explain_total_s")})
            if case in PROFILES:
                trace = archived.get("trace.json")
                kernels = kernel_evidence(trace) if trace else {}
                check(f"{suite}/{case} kernel and NCCL time match the archived CUDA trace",
                      kernels.get("kernel_count", 0) > 0 and all(
                          close(value, system.get("profile", {}).get(key))
                          for key, value in kernels.items()), kernels)
                check(f"{suite}/{case} excluded profiler and subsequent warmup",
                      system["steps"] >= 60 and system["throughput_excluded_steps"] >= 21 and
                      system["throughput_measured_steps"] == system["steps"]-system["throughput_excluded_steps"] and
                      finite(system.get("profile", {}).get("nccl_wall_fraction")),
                      {k: system.get(k) for k in ("steps", "throughput_excluded_steps",
                                                 "throughput_measured_steps", "profile")})
            if env["mode"] == "train":
                quality = result["final"]
                check(f"{suite}/{case} evaluated the full validation set",
                      quality["samples"] == cache["val"] and all(finite(quality.get(k)) for k in
                          ("ADE_m", "FDE_m", "miss_rate_2m", "action_accuracy", "collision_proxy",
                           "cached_policy_latency_p50_ms", "cached_policy_latency_p95_ms")) and
                      len(result.get("curves", [])) >= 2,
                      dict(quality=quality, curve_points=len(result.get("curves", []))))
                explanation = result.get("explanation") or {}
                check(f"{suite}/{case} completed all six interventions on the predefined subset",
                      explanation.get("examples") == min(48, cache["val"]) and
                      explanation.get("interventions") == min(48, cache["val"])*6,
                      explanation)

    comparisons = dict(sanity=PROFILES | {"lora_head"}, diagnose8=CASES | PROFILES,
                       diagnose16=PROFILES | {"lora_head", "large_batch"})
    for suite in ("final", "scale32", "scale64"):
        if suite in active:
            comparisons[suite] = {active[suite]["case"]} | (PROFILES if suite != "final" else set())
    for suite, expected in comparisons.items():
        check(f"{suite} covered every required comparison",
              set(suites.get(suite, {})) == expected, sorted(suites.get(suite, {})))
        if suite == "final":
            continue
        seconds = dict(sanity=60, diagnose8=1800, diagnose16=3600,
                       scale32=3600, scale64=1800)[suite]
        for case, result in suites.get(suite, {}).items():
            if case not in PROFILES:
                check(f"{suite}/{case} received its specified training window",
                      result["system"]["train_window_s"] >= .98*seconds, result["system"]["train_window_s"])
    eight, sixteen = suites.get("diagnose8", {}), suites.get("diagnose16", {})
    check("8/16 diagnosis used the same queue and cluster",
          active.get("diagnose8", {}).get("dc") == active.get("diagnose16", {}).get("dc") and
          active.get("diagnose8", {}).get("dc") in ("hl", "lf"),
          {s: active.get(s, {}).get("dc") for s in ("diagnose8", "diagnose16")})
    for case in PROFILES:
        if case in eight and case in sixteen:
            a, b = eight[case]["environment"], sixteen[case]["environment"]
            check(f"{case} matched batch, seed, configuration, and cache",
                  a["global_batch"] == b["global_batch"] == 64 and a["seed"] == b["seed"] and
                  a["config"] == b["config"] and a["cache_manifest_sha256"] == b["cache_manifest_sha256"],
                  dict(eight=a, sixteen=b))
    gates = read(root/"decisions.json")
    check("All seven gate32 criteria have measured evidence",
          gates.get("ready") and len(gates.get("gate32", [])) == 7 and
          all(g["value"] is not None for g in gates["gate32"]), gates.get("gate32"))
    if "profile_cached" in eight and "profile_cached" in sixteen:
        efficiency = sixteen["profile_cached"]["system"]["throughput_samples_s"]/eight["profile_cached"]["system"]["throughput_samples_s"]/2
        check("Cached scaling efficiency was independently recomputed",
              close(efficiency, (gates.get("gate32") or [{}])[0].get("value")), efficiency)
    diagnostic = [r for side in (eight, sixteen) for r in side.values() if r["environment"]["mode"] == "train"]
    train_s = sum(r["system"]["train_step_total_s"] for r in diagnostic)
    eval_s = sum(r["system"]["eval_total_s"] for r in diagnostic)
    overhead = (cache["teacher_total_s"]+eval_s)/(cache["teacher_total_s"]+eval_s+train_s)
    check("Teacher/eval gate excludes subsequent final training",
          close(overhead, gates.get("teacher_eval_fraction")), overhead)
    for suite, gate in (("scale32", "allow32"), ("scale64", "allow64")):
        if suite in active:
            check(f"{suite} was permitted by its evidence gate", gates.get(gate) is True, gates.get(gate))
        else:
            omissions = [e for e in state["events"] if e["event"] == suite.upper()+"_SKIPPED"]
            check(f"{suite} omission was explicitly recorded", bool(omissions), omissions)
    final_job = active.get("final", {})
    final = suites.get("final", {}).get(final_job.get("case"))
    if final:
        seconds = final_job.get("seconds")
        check("Final training completed its allocated window within the protocol bounds",
              finite(seconds) and 1800 <= seconds <= 12600 and
              final["system"]["train_window_s"] >= .98*seconds,
              dict(allocated_s=seconds, measured_s=final["system"]["train_window_s"]))
        selected = min((k for k in eight if k in CASES), key=lambda k: eight[k]["final"]["ADE_m"])
        check("Final configuration was selected from all equal-duration ablations",
              final_job["case"] == selected, dict(selected=selected, actual=final_job["case"]))
        reproduce = read(root/"reproduce.json")
        digest = hashlib.sha256((root/"final_checkpoint.pt").read_bytes()).hexdigest()
        probe = final.get("checkpoint_reload_probe") or {}
        check("Transferred final weights match their training hash and fresh-base reload",
              digest == final["checkpoint_sha256"] == reproduce["sha256"] and
              reproduce["model_revision"] == REVISION and probe.get("verified") is True and
              len(probe.get("max_abs_output_deltas", [])) == 4 and max(probe["max_abs_output_deltas"]) <= 1e-5,
              dict(sha256=digest, reload_probe=probe))
    required = ("report.md", "metrics.json", "decision_table.csv", "bottleneck_decisions.csv",
                "quality_curves.png", "scaling.png", "qualitative.png", "teacher_diagnostics.json",
                "final_checkpoint.pt", "reproduce.json")
    check("Measured report, curves, decision tables, teacher diagnostics, and weights exist",
          all((root/n).is_file() and (root/n).stat().st_size > 0 for n in required),
          {n: (root/n).is_file() for n in required})
    manuscript = (project/"docs/manuscript.md").read_text()
    check("Manuscript includes completed results",
          "实验仍在执行" not in manuscript and "实验进行中" not in manuscript, "docs/manuscript.md")
    return dict(complete=all(c["passed"] for c in checks), passed=sum(c["passed"] for c in checks),
                total=len(checks), checks=checks,
                remaining_publication_check="Verify GitHub remote commit and uploaded files separately")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    args = parser.parse_args()
    result = audit(args.project.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["complete"] else 1)
