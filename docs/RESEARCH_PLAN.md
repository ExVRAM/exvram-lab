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

## Minimal-refusal track

This track is parallel to P2/P3. It does not replace the FULL_GPU performance work, and it
does not start with a new uncensoring algorithm or with SFT/RLHF/DPO.

Order:

1. Finish the current FULL_GPU 27B fit and measure decode.
2. Keep the lowest acceptable low-bit representation for the official checkpoint.
3. Use the published Heretic BF16 checkpoint `JonathanColetti/Qwen3.8-27B-Uncensored` as the
   behaviour source. See [UNCENSORED_OSS_MATRIX.md](UNCENSORED_OSS_MATRIX.md).
4. Requantize that BF16 checkpoint with llama.cpp, Unsloth, or ExLlamaV3. Do not abliterate
   an already low-bit file. Disk placement is [REQUANT_DISK_PLAN.md](REQUANT_DISK_PLAN.md):
   `E:` only. `F:` is removable on this machine even though Windows reports it as fixed.
5. Compare the matched quant on VRAM, tok/s, 8k fit, refusal rate, over-refusal rate, and
   capability. The scorecard is [UNCENSORED_PARETO.md](UNCENSORED_PARETO.md).

The 2026-09-24 gate is `REQUANTIZATION_REQUIRED`. Heretic stays an external AGPL tool for a
later re-export only if the published Coletti point fails a matched quality or refusal gate.
Fine-tuning stays behind that measurement.

`REFUSAL_RATE` is the fraction of refusals on sensitive and standard-refusal prompts.
`OVERREFUSAL_RATE` is the fraction of refusals on neutral, benign-but-suspicious, and
harmless-control prompts. Both come from supplied responses. A drop in refusal rate is not
success by itself; the figure that matters is refusal reduction against measured capability
loss.

## Backlog priority

- P0: baselines and memory map.
- P1: layer-level comparison of existing kernels.
- P2: full-model fit, residency, quality, and Pareto frontier.
- P3: 8k context and the >=25 tok/s target after the measured gap is isolated.
- P4: optional ExVRAM runtime/orchestrator only after P3 demonstrates that composing existing
  runtimes is not enough.
- Minimal-refusal: requantize the published Heretic BF16 checkpoint at the matched quant.
  No first-party abliteration unless that checkpoint fails a measured gate.
