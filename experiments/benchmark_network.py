"""Two-rank actual-host NCCL probe without model preparation or training claims."""
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert world == 2 and torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    out = Path(os.environ["DRIVING_LOCAL"]) / "results" / "network"
    out.mkdir(exist_ok=True)
    metadata = dict(rank=rank, world=world, gpu=torch.cuda.get_device_name(),
                    host_boot_sha256=hashlib.sha256(
                        Path("/proc/sys/kernel/random/boot_id").read_bytes()).hexdigest(),
                    torch=torch.__version__, nccl_version=torch.cuda.nccl.version(),
                    channel_setting="platform default",
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in Path(__file__).parent.glob("*.py")},
                    timing_scope="all_reduce Python call through CUDA completion; rank alignment barrier outside timing",
                    measurement="NCCL collectives only; no model, training throughput, or convergence result")
    results = []
    for count in (3, 6, 2382640):
        if time.time() > float(os.environ["DRIVING_BENCH_DEADLINE"]) - 60:
            raise TimeoutError("Reserve time for archive")
        tensor = torch.empty(count, dtype=torch.float32, device="cuda")
        seconds = []
        for index in range(120):
            tensor.fill_(rank + 1)
            torch.cuda.synchronize()
            dist.barrier()
            start = time.perf_counter()
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            duration = time.perf_counter() - start
            if index >= 20:
                seconds.append(duration)
        assert bool((tensor == 3).all())
        results.append(dict(bytes=count * 4, warmup=20, iterations=100,
                            seconds=seconds, all_reduce_sum_verified=True))
    (out / f"rank{rank}.json").write_text(json.dumps(dict(environment=metadata, collectives=results), indent=2))
    dist.barrier()
    if rank == 0:
        print(json.dumps(dict(event="NETWORK_PROBE_DONE", world=world,
                              payload_bytes=[r["bytes"] for r in results])), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
