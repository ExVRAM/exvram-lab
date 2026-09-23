# Gap analysis

## Evidence boundary

The 2026-09-23 run used an RTX 5060 with 7.959 GiB reported VRAM and compute capability 12.0.
Two isolated CUDA environments were used; the global CPU-only Torch installation was preserved.
The raw evidence is split between the GemLite/Torch matrix and the corrected ExLlamaV3 run:

- [`p1_real_gpu_layer_benchmark.jsonl`](../experiments/results/p1_real_gpu_layer_benchmark.jsonl)
  contains 48 measured Torch/GemLite records plus three ExLlama records from the first corrected
  integration pass.
- [`exl3_real_gpu_attention_q_m1_v3.jsonl`](../experiments/results/exl3_real_gpu_attention_q_m1_v3.jsonl)
  contains the final ExLlama M=1/8/32 records with corrected workspace accounting.

All these records are real CUDA layer timings with `synthetic: false`. They use generated matrices,
not a dense 27B checkpoint. P2 provenance/runtime preflight is recorded in
[`p2_preflight.jsonl`](../experiments/results/p2_preflight.jsonl); no full-model quality,
KV-cache, 8k-context or decode result is claimed until full-model rows are measured.

## Existing backend comparison

| Backend | Real path exercised | Current evidence | Remaining limitation | Custom kernel? |
|---|---|---|---|---|
| ExLlamaV3 / EXL3 | official `quantize_exl3` + `LinearEXL3.forward` | K=4 4096×4096 CUDA layer at M=1/8/32 | no checkpoint/qmap, no calibrated quality, no full model | no |
| GemLite | official `GemLiteLinear.pack` + Triton kernel | W2/W4 across four 4096-based shapes at M=1/8/32 | generated affine fixture, no model format, ptxas warning | no |
| Torch FP16/BF16 | CUDA matmul | 24 baseline records | dense baseline only | no |
| llama.cpp | official b10964 CUDA 13.3 binary sees CUDA0 after DLL bundle repair | device probe is measured; full-model runner is ready | GGUF artifact still downloading/awaiting load | no |
| BitNet GPU | not exercised | format-specific integration boundary verified | generic dense 27B adapter not established | no |
| CUTLASS | not exercised | optional building-block boundary verified | no custom integration or copied code | no |

The current measured gap is not yet a custom-kernel justification. GemLite W2 and EXL3 K4 differ
in representation, quantizer and runtime path; comparing their raw milliseconds is useful for
research triage but not a quality-preserving winner declaration. A custom kernel remains NO-GO
until two existing approaches are compared with the same checkpoint format, matrix shapes, quality
gate and resource budget, and the bottleneck is proved in profiling evidence.

## Research consequence

The environmental blocker from the original CPU-only attempt is resolved for layer work. The
scientific blocker remains: a concrete checkpoint, completed artifact download, and reproducible
quality protocol are required before the planner can claim a 27B fit or the target of at least
25 decode tokens/s. P2 classification is therefore `INCONCLUSIVE` until two applicable
full-model paths have measured rows.
