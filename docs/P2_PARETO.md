# P2 quality / VRAM / speed Pareto frontier

Status: pending full-model records. A point may enter this frontier only when it has a fixed
checkpoint revision, measured persistent and peak VRAM, measured decode speed, and a quality-gate
record. Synthetic records, layer fixtures, and third-party card claims are excluded.

## Required point fields

| Dimension | Required evidence |
|---|---|
| Quality | Same prompt/corpus mini-suite, reference outputs, and pass/fail thresholds |
| VRAM | Persistent and peak allocator/device measurements, context and KV/state precision |
| Speed | Warm-up policy, prefill tok/s, TTFT, steady decode tok/s, batch 1 |
| Reproducibility | Runtime version, model/artifact revision, command, hardware, raw log |

## Frontier procedure

1. Group only comparable text-only runs from the same Qwen3.8-27B base revision.
2. Reject points with failed correctness or quality gate.
3. A surviving point dominates another when it is no slower, no larger in peak VRAM, and no
   worse in quality, with one strict improvement.
4. Preserve dominated points in raw results; they remain useful for gap diagnosis.

No Pareto frontier is asserted yet. The P2 runner and result schema are the evidence path.
