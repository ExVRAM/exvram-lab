# Performance Pareto

One measured llama.cpp point and one failed ExLlamaV3 fit. This is not a frontier.

| Point | Quant bytes | GPU model | Host model | Decode tok/s | Prefill tok/s | 8k | Quality | Evidence |
|---|---:|---:|---:|---:|---:|---|---|---|
| llama.cpp `b10964` IQ2_XXS, c128, 32 predict | 8,881,268,448 | 5673 MiB | 2557 MiB | 3.8 | 31.3 | not run | NOT_RUN | historical fit log |
| llama.cpp same binary, short prompt, 16 predict | 8,881,268,448 | framebuffer peak 7191 MiB | not re-logged | 5.5 | 15.9 | not run | NOT_RUN | 2026-09-24 rerun |
| ExLlamaV3 EXL3 4.00bpw | ~16.8 GB on disk | fit failed | n/a | none | none | not run | NOT_RUN | `Insufficient VRAM` |

Stage checkpoints P3-S1 through P3-S5 (≥5, ≥10, ≥15, ≥20, ≥25 tok/s) are not cleared. The
5.5 tok/s short-prompt generation is a different workload from the 3.8 tok/s row and is not
recorded as P3-S1, because GPU residency was still incomplete and SM utilization stayed low.

Raw row: `experiments/results/p3_llamacpp_iq2_decode_diag.jsonl`.

The next Pareto point has to be a quant that fits. A higher nominal bpw that still offloads
is not a win on this GPU.

## P5 full-GPU context/KV points

These rows use the same Qwen3.8-27B IQ2_XXS checkpoint, llama.cpp b10982, 65/65 GPU layers,
and the standardized streaming runner. Quality is `NOT_RUN`; the rows are performance and
memory evidence, not a quality-preserving frontier.

| Point | KV | Context | Decode tok/s | Prefill tok/s | Peak VRAM MiB | Quality | Evidence |
|---|---|---:|---:|---:|---:|---|---|
| P5 q4 | q4_0 / q4_0 | 2048 | 19.00 | 20.58 | 7525 | NOT_RUN | `p5_context_scaling_q4.jsonl` |
| P5 q4 | q4_0 / q4_0 | 8192 | 29.04 | 32.58 | 7641 | NOT_RUN | `p5_context_scaling_q4.jsonl` |
| P5 q8 | q8_0 / q8_0 | 2048 | 30.17 | 33.29 | 7574 | NOT_RUN | `p5_kv_tradeoff_q8.jsonl` |
| P5 q8 | q8_0 / q8_0 | 8192 | 18.74 | 20.49 | 7687 | NOT_RUN | `p5_kv_tradeoff_q8.jsonl` |
| P5 f16 | f16 / f16 | 2048 | 30.42 | 33.50 | 7607 | NOT_RUN | `p5_kv_tradeoff_f16.jsonl` |
| P5 f16 | f16 / f16 | 8192 | 19.60 | 15.54 | 7764 | NOT_RUN | `p5_kv_tradeoff_f16.jsonl` |

The q4 c8192 value clears the numerical 25 tok/s target in one run, but it is provisional:
the c256-c4096 q4 curve is about 19 tok/s and matched Q8/F16 c8192 runs were slower. See
[`docs/CONTEXT_SCALING.md`](CONTEXT_SCALING.md), [`docs/KV_TRADEOFF.md`](KV_TRADEOFF.md),
and [`docs/8K_PERFORMANCE.md`](8K_PERFORMANCE.md).

## P6 reproducibility controls

P6 uses the median of five measured runs after one warmup. These are short-prompt controls and
must not be mistaken for actual 8k occupancy:

| Configured ctx | Occupied ctx | KV | Median decode tok/s | CV | Quality |
|---:|---:|---|---:|---:|---|
| 256 | 256 | q4_0/q4_0 | 30.65 | 1.63% | NOT_RUN |
| 512 | 266 | q4_0/q4_0 | 31.28 | 1.36% | NOT_RUN |
| 1024 | 266 | q4_0/q4_0 | 32.98 | 3.33% | NOT_RUN |
| 4096 | 266 | q4_0/q4_0 | 18.98 | 4.65% | NOT_RUN |
| 8192 | 266 | q4_0/q4_0 | 17.67 | 0.74% | NOT_RUN |

The P6 filled-context c8192 gate is still open; no row above confirms the 27B/8GB/actual-8k/
>=25 tok/s target.
