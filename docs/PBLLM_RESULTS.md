# P8 PB-LLM reference track

[PB-LLM](https://github.com/hahnyuan/PB-LLM) is used as an MIT-licensed
algorithmic reference only. Its partially binarized idea is represented in the
planner as binary weights plus a high-bit salient subset, a bitmap, and group
scales. The repository does not vendor PB-LLM code and does not claim a full
Qwen3.8-27B runtime.

## Planned sweep

The layer-level sweep is:

| Binary fraction | High-bit salient fraction | High-bit setting | Group |
|---:|---:|---:|---:|
| 90% | 10% | 8-bit | 64 |
| 95% | 5% | 8-bit | 64 |
| 97.5% | 2.5% | 8-bit | 64 |

The planner currently models these as approximately 3.05, 2.65, and 2.45
physical bpw respectively under its explicit storage assumptions. The values
include a one-bit-per-parameter bitmap and FP16 scale bytes; they are not a
PB-LLM serialization dump. This accounting makes the metadata cost visible
instead of treating the binary fraction as free.

## Required evidence before promotion

- layer reconstruction error for q/k/v/o, MLP, and `lm_head` where feasible;
- comparison with HQQ/GemLite 1-bit and 2-bit on identical shapes;
- exception/index/scale storage measured from the actual artifact;
- kernel latency and workspace measured separately from reconstruction;
- no full-model run until a layer-level Pareto point passes the P8 KPI.

Current status is `REFERENCE_ONLY`; quality, speed, 8k fit, and model
intelligence are `NOT_RUN`.
