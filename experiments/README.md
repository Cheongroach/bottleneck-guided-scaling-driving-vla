# Driving-VLA experiment code

The code uses cached current-frame nuScenes mini embeddings, a frozen
Qwen2.5-VL language backbone, rank-8 Q/V LoRA, and trajectory/action/evidence
heads. The probes distinguish complete update wall time, CUDA-event elapsed
time, kernel intervals, and collective residence time.

- `train.py`, `contracts.py`, and `cache_inputs.py` implement training and the
  scene-isolated input contract.
- `benchmark_single_gpu.py` compares microbatch 2/4/8 and activation
  checkpointing on/off on resident real specimens.
- `benchmark_gc.py` records generation-specific GC wall/thread CPU time and
  compares automatic GC, disabled GC, and checkpointing disabled.
- `benchmark_communication.py` varies collective channels, checkpointing,
  and statistics cadence with fixed global batch.
- `benchmark_network.py` and `launch_interhost.py` measure all-reduce on
  explicitly verified physical hosts.
- `analyze_*.py` rebuild evidence from archived outputs; `verify_results.py`
  checks completion and numerical/evaluation gates.
- `../results/execution-roofline/assets/` contains the matrix probe, trace
  correlation analysis, report builder, and ECharts configuration.

Install CPU analysis dependencies from `report-requirements.txt`. Training
requires a CUDA-enabled PyTorch environment and the pinned data/model
artifacts. Raw job archives are local inputs under `results/jobs/`; they are
not committed to Git. Missing artifacts are errors, not passing gates.

The matrix probe expects `roofline-model-config.json` beside its source; copy
the supplied `model-config.json` to that name in the remote source bundle.
It uses generated matrix values with real model shapes and makes no training
quality claim.

Platform launchers in this snapshot retain the original campaign's internal
queue, HDFS, and deadline configuration. Do not rerun the old campaign
controller for a new probe: use a fresh bounded deadline and verify the
resource request. Follow-up diagnostics are limited to at most four GPUs
concurrently; the matrix/counter preflight needs only one.

Kernel interval coverage is not SM utilization. Algorithmic matrix bytes are
not measured DRAM traffic. Nsight Compute counters must be collected before
claiming actual HBM intensity or scheduler utilization.
