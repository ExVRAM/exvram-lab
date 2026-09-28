# P8 — Frequent-Group 1-Bit / Binary Representation

P8 is an isolated research track. It does not replace the measured P4/P5/P6
runtime results and it does not change the current production recipe.

## Research question

Can a dense Qwen3.8-27B layer use binary or partially binary weights while
spending additional GPU compute on reconstruction, scales, metadata, and
salient exceptions? The target is a practical representation near 1.2–1.6
effective bits per weight on RTX 5060 8 GB / SM120, batch 1, with quality loss
measured on the same tensor and with latency compared to a 2-bit baseline.

This repository currently contains the protocol, manifest, external adapter
boundary, and physical-footprint planner. It does not claim a measured 1-bit
quality or speed result.

## External-first rule

The first implementations are external projects, not vendored code:

- [HQQ](https://github.com/dropbox/hqq), Apache-2.0, commit
  `d88a488ec8aa2d58362ef2038a52bca862db2e74`: quantization/reconstruction
  boundary for 1/2-bit layer experiments.
- [GemLite](https://github.com/dropbox/gemlite), Apache-2.0, commit
  `89d9bc705c5dfca9115d3a5620f97a17ba0111a`: candidate low-bit kernels and
  SM120 performance path.
- [PB-LLM](https://github.com/hahnyuan/PB-LLM), MIT, commit
  `fe85da943d9df48ab6455d698f75406bb0bfefbc`: partially binarized reference.
- [BiLLM](https://github.com/Aaronhuang-778/BiLLM), MIT, `main` commit
  `dc137ebbf62d4b31e8a82ba6bf9e18a51a298dcb`: layer-feasibility reference.

GPL projects remain process-level references only. No GPL source is copied into
the Apache-2.0 codebase. A custom low-level kernel requires a documented gap
against at least two existing approaches on the same shape and protocol.

## Layer matrix

The manifest uses the real Qwen3.8-27B dimensions from the recorded model
configuration:

- attention: q/k/v/o projections;
- MLP: gate/up/down projections;
- `lm_head` when the external runtime can handle its vocabulary-sized shape.

HQQ is swept at groups 8/16/32/64/128. Groups 8 and 16 are quality/memory-only
points. GemLite fast-path candidates are restricted to groups 32/64/128.

Every candidate is compared with FP16/BF16, HQQ 2-bit, GemLite 2-bit, and the
current IQ2_XXS artifact estimate. The same matrix shape is required for every
comparison.

## Physical accounting

The planner reports more than nominal weight bits:

```text
physical bytes = packed weights
               + scales + zeros
               + exception values + indices/bitmap/metadata
               + packing/alignment padding
               + persistent buffers
```

`physical_bpw` is computed from total persistent bytes and the unpadded logical
parameter count. It is a layout estimate until a real HQQ/GemLite serialization
and allocator snapshot is captured.

## Quality and performance gates

Quality uses tensor-level MSE, relative MSE, cosine similarity, maximum absolute
error, and maximum relative error. These are reconstruction proxies, not claims
about intelligence or instruction following.

GemLite measurements must include M=1, 8, 32, and 128, median/p95 latency,
effective GB/s, workspace, VRAM, and GPU utilization. RTX 5060 autotuning is a
separate result. The 2-bit comparison must use the same shapes and warm-up policy.

The research KPI is `physical_bpw <= 1.6` with latency no worse than `1.2x` the
measured 2-bit baseline. The stretch target is `<=1.4 bpw` and `<=1.1x` latency.
No full-model quantization or training starts until a layer-level Pareto point
passes this gate.

## Reproducible entry point

The CPU-safe planner can be run without CUDA:

```powershell
.venv\\Scripts\\exvram.exe plan-p8-binary `
  --manifest experiments\\manifests\\p8_binary_qwen38.json `
  --output experiments\\results\\p8_binary_plan.json
```

The output is explicitly `synthetic: true`, `measurement_status: PLANNED`, and
`decision_gate.status: NOT_RUN`. Actual GPU work requires an idle, safe VRAM
slot and an environment containing the pinned external integrations.
