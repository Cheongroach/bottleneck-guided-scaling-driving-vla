"""Publish selected experiment evidence only after the full local completion audit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile

from verify_results import audit


PUBLIC_FILES = (
    "report.md", "metrics.json", "decisions.json", "decision_table.csv",
    "bottleneck_decisions.csv", "teacher_diagnostics.json", "quality_curves.png",
    "scaling.png", "qualitative.png", "final_checkpoint.pt", "reproduce.json",
)
TRAINING_FILES = {"result.json", "predictions.json", "curves.json", "steps.json", "trace.json"}
JOB_KEYS = (
    "sid", "name", "suite", "world", "dc", "seconds", "case", "attempt",
    "status", "collected", "submitted_epoch_s", "first_running_epoch_s",
    "observed_terminal_epoch_s", "hard_stop_epoch_s", "user_wall_limit_s",
    "cross_queue_validation",
)
SECRET_KEYS = {
    "access_token", "refresh_token", "id_token", "authorization", "jwt",
    "password", "secret", "client_secret", "api_key", "private_key", "credentials",
}
SECRET_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{20,}"),
    re.compile(r"""(?i)["']?(?:access_token|refresh_token|client_secret|api_key|password)["']?\s*[:=]\s*["']?[A-Za-z0-9._~+/-]{20,}"""),
    re.compile(r"(?i)[?&](?:token|signature|x-amz-signature|x-tos-signature)=[A-Za-z0-9%._~-]{20,}"),
)


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8*1024**2), b""):
            sha.update(chunk)
    return sha.hexdigest()


def inspect_text(content, label):
    """Fail with a filename only; never echo a detected credential."""
    text = content.decode("utf-8")
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        raise ValueError(f"Credential-like content rejected: {label}")
    if label.endswith(".json"):
        value = json.loads(text)

        def visit(node):
            if isinstance(node, dict):
                for key, child in node.items():
                    if key.lower().replace("-", "_") in SECRET_KEYS and child:
                        raise ValueError(f"Sensitive JSON field rejected: {label}")
                    visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)

        visit(value)


def package(project):
    root = project/"results"
    verification = audit(project)
    if not verification["complete"]:
        failed = [c["requirement"] for c in verification["checks"] if not c["passed"]]
        raise RuntimeError("Campaign incomplete: "+"; ".join(failed))
    state = json.loads((root/"controller_state.json").read_text())
    active = [j for j in state["jobs"] if not j.get("superseded")]
    prepared = {}
    archives = []

    def add(relative, content):
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Public artifact path escapes its destination")
        if relative in prepared:
            raise ValueError(f"Duplicate public artifact: {relative}")
        if relative.endswith((".json", ".md", ".csv", ".py", ".txt")):
            inspect_text(content, relative)
        prepared[relative] = content

    for name in PUBLIC_FILES:
        content = (root/name).read_bytes()
        add("results/"+name, content)
    forecast = root/"end_to_end_forecast.json"
    if forecast.exists():
        add("results/"+forecast.name, forecast.read_bytes())
    for name in ("download_manifest.json", "cache_manifest.json", "records.json"):
        add("results/evidence/assets/"+name, (root/"assets"/name).read_bytes())
    for job in active:
        directory = root/"jobs"/job["name"]
        metadata = {k: job[k] for k in JOB_KEYS if k in job}
        metadata["archives"] = []
        for path in sorted(directory.glob("*.tar.gz")):
            source_archive = dict(name=path.name, sha256=digest(path), bytes=path.stat().st_size)
            metadata["archives"].append(source_archive)
            archives.append(dict(suite=job["suite"], **source_archive))
            with tarfile.open(path) as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    parts = Path(member.name).parts
                    if len(parts) == 4 and parts[:2] == ("results", "training") and parts[-1] in TRAINING_FILES:
                        content = tar.extractfile(member).read()
                        relative = f"results/evidence/{job['name']}/{parts[2]}/{parts[3]}"
                        if parts[-1] in ("trace.json", "steps.json"):
                            inspect_text(content, relative)
                            add(relative+".gz", gzip.compress(content, mtime=0))
                        else:
                            add(relative, content)
        add(f"results/evidence/{job['name']}/job.json",
            (json.dumps(metadata, indent=2)+"\n").encode())
        marker = directory/"live"/"suite_complete.json"
        if marker.exists():
            add(f"results/evidence/{job['name']}/suite_complete.json", marker.read_bytes())
    # This snapshot contains no platform response, startup log, or environment dump.
    add("results/completion_audit.json", (json.dumps(
        verification, indent=2, ensure_ascii=False)+"\n").encode())
    for directory in ("experiments", "docs"):
        for path in sorted((project/directory).iterdir()):
            if path.is_file() and path.suffix in (".py", ".md", ".txt"):
                inspect_text(path.read_bytes(), str(path.relative_to(project)))
                add(str(path.relative_to(project)), path.read_bytes())
    for name in ("README.md", "plan.md", ".gitignore"):
        content = (project/name).read_bytes()
        inspect_text(content, name)
        add(name, content)
    manifest = dict(
        campaign_completed_epoch_s=state["completed_epoch_s"],
        archives=archives,
        files={name: dict(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
               for name, content in sorted(prepared.items())},
        excluded="Raw platform responses, pod startup logs, archives, and base-model/data binaries",
        scope="Selected measured evidence, final trainable weights, documentation, and source",
    )
    add("results/publication_manifest.json", (json.dumps(manifest, indent=2)+"\n").encode())
    # Validate everything before writing the public evidence tree.
    for relative, content in prepared.items():
        path = project/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return dict(files=len(prepared), evidence_files=sum("/evidence/" in p for p in prepared),
                archive_sources=len(archives), audit_checks=verification["total"],
                manifest="results/publication_manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.project.resolve()), indent=2))
