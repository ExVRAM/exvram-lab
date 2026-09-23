# Compute-for-memory proof

Status: P3 gate pending. The lab thesis is not established by a memory calculator alone. It
requires one real Linear-layer comparison on the target GPU using at least two existing OSS
paths before considering first-party low-level code.

## Required controlled comparison

- Same input/output shape, batch and dtype.
- Reference: Torch or a mature dense kernel with the full representation resident.
- Candidate: existing EXL3, GemLite, BitNet, CUTLASS, or llama.cpp-compatible path where the
  representation is actually supported.
- Report packed bytes, scale/codebook bytes, workspace, persistent allocator bytes, and any
  CPU↔GPU transfer bytes.
- Report latency, effective compute, correctness error, and whether the candidate wins under
  batch 1 decode-like M=1 as well as a prefill-like M>1.

P1 already contains real RTX 5060 layer observations for Torch, GemLite W2/W4, and ExLlama EXL3
fixtures. They are not same-checkpoint evidence and do not authorize a custom kernel. The P3
proof will reference the full-model bottleneck and append raw records under
`experiments/results/`.
