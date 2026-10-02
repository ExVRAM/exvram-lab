# P8 BiLLM reference track

Status: frozen on `experiment/p8-low-bit` (`9c93171`). This track is not the product path.

[BiLLM](https://github.com/Aaronhuang-778/BiLLM) is an MIT-licensed external
reference for 1-bit model feasibility. ExVRAM does not copy its training or
runtime code and does not present the paper's reported average bit width as a
Qwen3.8-27B serialized footprint.

## Scope

BiLLM is limited to a layer-feasibility comparison in P8. The required first
step is to test whether representative Qwen3.8-27B projection shapes can be
reconstructed with useful error and whether an existing runtime can execute the
result. Full-model training, conversion, and serving are explicitly out of
scope until HQQ/GemLite layer evidence supports them.

## Current status

The plan records BiLLM as `reference_only` with no quality tensor, latency,
VRAM, or 8k result. The paper/reference number must not be compared directly
with HQQ or GemLite physical bpw until all metadata, scales, exceptions,
padding, and persistent buffers are measured using the same accounting rule.
