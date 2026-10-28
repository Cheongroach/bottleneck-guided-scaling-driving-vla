"""Bounded per-pod torchrun launch using advertised Arnold worker networking."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--entry-script", default="train.py")
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    entry_script = Path(__file__).parent / args.entry_script
    if Path(args.entry_script).name != args.entry_script or not entry_script.is_file():
        raise ValueError("entry-script must name a bundled source file")
    pod = os.environ.get("MY_POD_NAME", socket.gethostname())
    match = re.search(r"worker[-_](\d+)$", pod)
    rank_text = os.environ.get("ARNOLD_ID", match.group(1) if match else "0")
    rank = int(re.search(r"(\d+)$", rank_text).group(1))
    if not 0 <= rank < args.nodes:
        raise ValueError(f"node rank {rank} outside {args.nodes}")
    store = Path(os.environ["DRIVING_STORE"])/"runs"/os.environ["DRIVING_RUN"]
    rendezvous = store/"rendezvous.json"
    if rank == 0:
        address = os.environ.get("ARNOLD_WORKER_0_HOST") or os.environ.get("MLP_WORKER_0_HOST")
        if not address:
            try:
                networks = json.loads(subprocess.check_output(["ip", "-j", "addr"], text=True))
                addresses = [a["local"] for n in networks for a in n.get("addr_info", [])
                             if a.get("scope") == "global" and a["family"] == "inet6"]
                address = addresses[0] if addresses else socket.gethostbyname(socket.gethostname())
            except (OSError, subprocess.CalledProcessError):
                address = socket.gethostbyname(socket.gethostname())
        port = os.environ.get("ARNOLD_WORKER_0_PORT") or os.environ.get("PORT0") or "29501"
        port = str(port).split(",")[0].split(":")[0]
        record = dict(address=address.strip("[]"), port=int(port), pod=pod,
                      written_epoch_s=time.time(), nodes=args.nodes)
        temporary = rendezvous.with_suffix(".upload")
        temporary.write_text(json.dumps(record))
        temporary.rename(rendezvous)
    deadline = time.monotonic()+600
    while not rendezvous.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("rank0 rendezvous not published within 10 minutes")
        time.sleep(2)
    record = json.loads(rendezvous.read_text())
    assert time.time()-record["written_epoch_s"] < 1800
    address = record["address"]
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ.setdefault("NCCL_DEBUG_FILE", str(Path(os.environ["DRIVING_LOCAL"])/"results"/"nccl-%h-%p.log"))
    command = ["python3", "-m", "torch.distributed.run", f"--nnodes={args.nodes}",
               f"--nproc_per_node={args.gpus}", f"--node_rank={rank}",
               f"--master_addr={address}", f'--master_port={record["port"]}',
               "--max_restarts=0", str(entry_script)]
    command.extend(args.training_args[1:] if args.training_args[:1] == ["--"] else args.training_args)
    print("DISTRIBUTED_LAUNCH", json.dumps(dict(node_rank=rank, nodes=args.nodes,
          gpus_per_node=args.gpus, rendezvous=record, command=command)), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
