# P6 reproducibility

Status: **occupied ~8k decode is measured and repeatable on q4_0 KV. Quality is not run. q8_0 and f16 KV at this occupancy are not in the headline series.**

Headline metric is the median decode tok/s after one warmup, not the fastest run. A run counts only when llama-server reports 65/65 layers on CUDA and the client records `cpu_decoder_weight_offload=false`. Host-mapped token embeddings stay at 397.85 MiB (`CUDA_Host` model buffer). That is not decoder-weight offload.

Runtime held constant for the series below: Unsloth UD-IQ2_XXS SHA256 `e792d8fb3142fe6d9171876d6da0f71f05a71028718debc72dbec93ff645e67d`, llama.cpp b10982 commit `fc82583e6`, `--fit off`, `--gpu-layers 999`, Flash Attention on, q4_0 K/V, batch 32, ubatch 1, seed 42, temperature 0, reasoning off, `--ctx-checkpoints 0`, `--cache-ram 0`, 256 new tokens.

## What 29.04 tok/s was

The P5 c8192 point of 29.04 tok/s was one decode measurement. Its prompt was 5240 tokens inside a configured window of 8192, so final occupancy was about 5496, not 8192. P6 reproduced a nearby rate only after the prompt itself was filled to 8192 tokens. The number was a real single-run decode, not an empty-window artifact, and it was not by itself a repeated full-occupancy result.

## Window-size test

Same 10-token prompt. Only `--ctx-size` changes. With context checkpoints left on, configured 4096/8192 decode fell to about 19 and 17 tok/s and some series aborted. With `--ctx-checkpoints 0` and `--cache-ram 0`, configured window size does not move decode off ~30.6 tok/s.

| configured ctx | prompt | generated | median decode | CV | occupancy |
|---:|---:|---:|---:|---:|---:|
| 256 | 10 | 246 | 30.77 | 8.6% | 256 |
| 512 | 10 | 256 | 30.66 | 0.25% | 266 |
| 1024 | 10 | 256 | 30.76 | 0.28% | 266 |
| 2048 | 10 | 256 | 30.76 | 0.20% | 266 |
| 4096 | 10 | 256 | 30.61 | 0.53% | 266 |
| 8192 | 10 | 256 | 30.61 | 0.12% | 266 |

The c256 CV is above 5% because the first measured run was 25.07 tok/s and the other four were 30.75–30.82. The median is 30.77. That first-run dip is the instability, not the window size.

## Occupied-context test

The prompt is filled to the target, then 256 tokens are generated. The server window is target + 256 so generation is not truncated. `tokens_in_cache_before_decode` is the occupancy that matters.

Rows below are the latest uncontended five-run series (prefill stayed near 33–36 tok/s). The JSONL also contains a slower cluster, prefill about 19 tok/s and decode about 17–20, from sessions where the GPU was shared. Those rows stay in the file and are not the headline.

| tokens in cache before decode | configured ctx | median decode | CV | median prefill | median p50/p95/p99 ITL ms |
|---:|---:|---:|---:|---:|---|
| 256 | 512 | 33.69 | 0.34% | 36.5 | 29.5 / 30.3 / 30.8 |
| 511 | 768 | 33.43 | 0.20% | 36.4 | 29.7 / 30.5 / 32.5 |
| 1024 | 1280 | 33.35 | 0.05% | 36.5 | 29.8 / 30.4 / 30.7 |
| 2048 | 2304 | 30.11 | 1.5% | 33.0 | 33.1 / 33.7 / 34.5 |
| 4096 | 4352 | 31.77 | 4.6% | see raw | see raw |
| 8192 | 8448 | 30.10 | 2.0% | 33.6 | 32.6 / 34.2 / 35.5 |

Occupied 8192, 27 Sep 2026, five runs after one warmup: 30.10, 29.62, 29.55, 31.07, 30.21 tok/s. All 65/65, Flash Attention on, graph splits 2. CUDA model buffer 6521.13 MiB, CUDA_Host embeddings 397.85 MiB, q4 KV 148.5 MiB, recurrent state 149.62 MiB. Peak framebuffer 7565–7767 MiB (max 7767). Max GPU utilization 99–100%. Max memory-controller utilization 59–61%. Max power about 147 W. TTFT is the prefill of 8192 tokens, median about 244 s.

An earlier five-run occupied-8192 group (26 Sep, 26.96–31.07, median 28.42) sits in the same band.

## KV at occupied 8192

Same prompt (8192), same 256 generated tokens, `--fit off`. No CPU decoder offload.

| KV | runs | median decode | notes | KV MiB | peak VRAM |
|---|---:|---:|---|---:|---:|
| q4_0/q4_0 | 5 | 30.10 | CV 2.0% | 148.5 | 7767 |
| q8_0/q8_0 | 5 | 20.31 | CV above 5%. One run was 4.72 tok/s; the other four were 19.66, 20.31, 20.35, 27.13 | 280.5 | 7788 |
| f16/f16 | — | — | not completed in this write-up | — | — |

Among the two completed types, q4_0 is FASTEST_KV, LOWEST_MEMORY_KV, and BEST_BALANCED_KV. q8_0 fits (65/65) and is slower.

Raw: `experiments/results/p6_reproducibility.jsonl`.
