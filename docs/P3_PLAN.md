# P3 plan selected from measured gap

Status: branch selection is deliberately deferred until P2 has at least two applicable existing
runtime observations or an explicit reproducible compatibility blocker. The selection rule is
machine-readable in the queue and must not be replaced by intuition.

## Branch selector

| P2 classification | P3 branch | First falsifiable experiment |
|---|---|---|
| A `SUCCESS_WITH_EXISTING_OSS` | P3A orchestrator | Reproduce the winning placement/KV policy across all target contexts |
| B `MEMORY_GAP_ONLY` | P3B memory-gap isolation | Move one named component at a time to host/secondary tier and measure traffic |
| C `SPEED_GAP_ONLY` | P3C existing-backend profiling | Profile two existing kernels on the same Linear shape before any custom code |
| D `QUALITY_GAP` | P3D representation research | Compare one safer representation or mixed-precision allocation with quality gate |
| E `MULTIPLE_GAPS` | P3E Pareto decomposition | Choose the single largest measured bottleneck and hold other axes fixed |

## Guardrails

- No custom CUDA/Triton kernel is permitted in P3 until the same checkpoint and shape have been
  tested with at least two existing applicable approaches.
- A P3 result must carry `status` in `PLANNED`, `RUNNING`, `PASS`, `FAIL`, or `INCONCLUSIVE`.
- One Linear compute-for-memory proof is mandatory; it must report memory saved, extra compute,
  host/device traffic, correctness error, and whether the trade is beneficial at batch 1.
- A target miss is a result, not a failure of the research process. The next step must follow
  the measured bottleneck rather than silently changing the target.
