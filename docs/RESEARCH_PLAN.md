# Research plan and backlog

## Target

The first target is an RTX 5060 8 GiB / SM120-class environment, a dense approximately 27B
checkpoint, 8k context, batch 1, and a research target of at least 25 decode tokens/s with the
smallest measurable quality loss. These are targets, not results. No target is claimed until a
real matching GPU, fixed checkpoint, and reproducible protocol produce a result record.

## P0 — baselines and memory map

- Select one concrete dense checkpoint and record its model license, hash, tokenizer, and shape.
- Run llama.cpp and ExLlamaV3 baselines where supported.
- Measure weights, scales, KV cache, temporary workspace, allocator reserve, and offload.
- Validate the planner against runtime telemetry; keep theoretical and observed values separate.

## P1 — layer-level comparison of existing kernels

- Compare ExLlamaV3, GemLite, CUTLASS-based primitives, BitNet-compatible kernels, and llama.cpp
  baselines on representative GEMV/GEMM shapes.
- Measure correctness, dequantization cost, launch overhead, residency, and sustained throughput.
- Do not write a custom low-level kernel until at least two existing approaches show a documented
  gap on the same shape and measurement protocol.

## P2 — full-model fit and quality

- Test candidate weight/KV combinations on the concrete checkpoint. The first machine-readable
  queue is generated from `experiments/manifests/qwen38_p2.json` with:
  `exvram plan-p2 --manifest ... --queue-output ... --database ...`.
- Record queue states as `PLANNED`, `RUNNING`, `PASS`, `FAIL`, or `INCONCLUSIVE`; persist both
  JSONL and SQLite rows so a failed runtime probe is not lost.
- Measure actual model fit, persistent/peak VRAM, host RAM/offload, contexts 128/2k/8k,
  prefill/decode speed, TTFT, KV/recurrent-state precision, optional vision/MTP components,
  and CPU↔GPU traffic where the runtime exposes it.
- Add perplexity/task checks with a fixed corpus and reference outputs.
- Treat model, tokenizer, and dataset licenses as separate reviews.
- Reject configurations that fit only by violating quality or reproducibility gates.

## P3 — 8k context and throughput target

- Re-run the best P2 configurations at 8k context and batch 1.
- Measure warm and steady-state decode separately, with a documented tokenization and sampling
  policy.
- Report whether the >=25 tok/s target is met, missed, or unmeasured. Never infer it from the
  synthetic microbenchmark.

The P3 branch is selected from the measured P2 classification: A existing OSS success, B memory
gap, C speed gap, D quality gap, or E multiple gaps. Until P2 has measured points, P3 remains
planned rather than being silently optimized against a proxy.

## Backlog priority

- P0: baselines and memory map.
- P1: layer-level comparison of existing kernels.
- P2: full-model fit, residency, quality, and Pareto frontier.
- P3: 8k context and the >=25 tok/s target after the measured gap is isolated.
- P4: optional ExVRAM runtime/orchestrator only after P3 demonstrates that composing existing
  runtimes is not enough.
