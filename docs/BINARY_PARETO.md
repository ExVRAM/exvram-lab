# P8 binary Pareto ledger

This ledger separates modeled storage from measured quality and speed. Empty
measurement cells are intentional; they must not be read as zero or as a
performance claim.

## Current modeled model-level storage

The values below come from `src/exvram/p8_binary.py` using the Qwen3.8-27B
parameter count in the P8 manifest, FP16 scales/zeros, 32-byte component
alignment, and no runtime-specific hidden buffers. They are not serialized
HQQ/GemLite artifacts.

| Method | Group | Modeled physical bpw | Quality | M=1 latency | Status |
|---|---:|---:|---|---:|---|
| HQQ 1-bit | 32 | ~2.00 | not run | not run | planned |
| HQQ 1-bit | 64 | ~1.50 | not run | not run | planned |
| HQQ 1-bit | 128 | ~1.25 | not run | not run | planned |
| HQQ 2-bit | 64 | ~2.50 | not run | not run | planned |
| PB-LLM layout, 90% binary / 10% high-bit | 64 | ~3.05 | reference only | not run | reference |
| PB-LLM layout, 95% binary / 5% high-bit | 64 | ~2.65 | reference only | not run | reference |
| PB-LLM layout, 97.5% binary / 2.5% high-bit | 64 | ~2.45 | reference only | not run | reference |
| IQ2_XXS | artifact | nominal 2.0625; artifact estimate ~2.09 | not run here | not run here | reference |

PB-LLM values include a bitmap, FP16 group scales, and 8-bit salient
exceptions. A real implementation may use different scale, index, packing, and
workspace layouts; those bytes must replace the model estimate before a decision.

## Decision table

| Gate | Required evidence | Current state |
|---|---|---|
| Representation | Exact serialized weights and persistent buffers | open |
| Reconstruction | MSE/relative MSE/cosine/max error for every listed layer | open |
| Kernel speed | GemLite M=1/8/32/128 median and p95 | open |
| 2-bit control | Same shape, runtime, warm-up, and measurement policy | open |
| Full model | Fit, 8k KV, quality, and decode | intentionally deferred |
| Custom kernel | Gap versus at least two existing approaches | not justified |

The only current conclusion is that group size materially changes modeled
overhead. It is not evidence that 1-bit inference is faster or that it preserves
model quality.
