"""Read historical Grafana panels using the official trial URL and pod interval."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parent.parent
TMP = Path("/tmp/driving-session")
CLI = "/home/tiger/.merlin-cli/bin/merlin-cli"
RUNS = {
    16: "driving-diagnose16-16h20-a1-20261028",
    32: "driving-scale32-32h20-a1-20261028",
}


def stamp(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, choices=RUNS, required=True)
    parser.add_argument("--panel", type=int, required=True)
    parser.add_argument("--role-dashboard", action="store_true")
    args = parser.parse_args()
    link = json.loads((TMP / f"goal-grafana{args.world}-link.json").read_text())
    pods = json.loads((ROOT / "results/jobs" / RUNS[args.world] / "pods.json").read_text())["data"]
    start = min(stamp(p["start_at"]) for p in pods) - 30
    end = max(stamp(p["stop_at"]) for p in pods) + 30
    parsed = urlsplit(link["grafana_url"])
    query = [(k, v) for k, v in parse_qsl(parsed.query)
             if k not in ("from", "to", "var-target_arnold_role")]
    query += [("from", str(int(start * 1000))), ("to", str(int(end * 1000)))]
    if args.role_dashboard:
        # This alternate Arnold dashboard was inspected with `grafana describe`.
        parsed = parsed._replace(path="/d/JiidrBwGzrole/")
        query += [("var-target_arnold_role", "worker")]
    url = urlunsplit(parsed._replace(query=urlencode(query)))
    target = TMP / f"goal-grafana{args.world}-p{args.panel}.json"
    result = subprocess.run(
        [CLI, "--control-plane", "cn", "grafana", "panel-data", "--json",
         json.dumps(dict(url=url, panel_id=args.panel))],
        capture_output=True, text=True, timeout=120)
    target.write_text(result.stdout)
    target.with_suffix(".stderr.txt").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"Panel {args.panel} query failed; see {target}")
    data = json.loads(result.stdout)
    series = [s for item in data.get("total", []) if isinstance(item.get("data"), list)
              for s in item["data"] if isinstance(s, dict) and s.get("dps")]
    print(json.dumps(dict(world=args.world, panel=args.panel, readable_series=len(series),
                          bytes=target.stat().st_size, file=str(target)), ensure_ascii=False))


if __name__ == "__main__":
    main()
