# P5 KV precision trade-off

Measured on the RTX 5060 8 GB with the same IQ2_XXS Qwen3.8-27B GGUF and the
same P5 runner. Quality was not run, so this is a memory/latency trade-off only.

| KV K/V | ctx | KV buffer MiB | recurrent state MiB | peak VRAM MiB | min free VRAM MiB | prefill tok/s | decode tok/s | p50 / p95 ms | min free RAM MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q4_0 / q4_0 | 2048 | 36 | 149.62 | 7525 | 372 | 20.58 | 19.00 | 53.63 / 59.14 | 971 |
| q8_0 / q8_0 | 2048 | 68 | 149.62 | 7574 | 323 | 33.29 | 30.17 | 33.07 / 33.94 | 443 |
| f16 / f16 | 2048 | 128 | 149.62 | 7607 | 290 | 33.50 | 30.42 | 32.80 / 33.46 | 889 |
| q4_0 / q4_0 | 8192 | 144 | 149.62 | 7641 | 256 | 32.58 | 29.04 | 34.33 / 35.20 | 228 |
| q8_0 / q8_0 | 8192 | 272 | 149.62 | 7687 | 210 | 20.49 | 18.74 | 53.90 / 60.06 | 23 |
| f16 / f16 | 8192 | 512 | 149.62 | 7764 | 133 | 15.54 | 19.60 | 50.33 / 53.74 | 22 |

All six runs loaded `65/65` layers on CUDA0, used `flash_attn = enabled`, and
reported `kv_unified = false`. q4 and q8 used 3943 graph nodes; f16 used 3751.
The difference is a runtime graph choice, not a copied or custom kernel.

## Decision

F16 KV fits at c8192 on this exact machine, but leaves only 133 MiB of measured
VRAM headroom and 22 MiB of available system RAM. It is therefore not the safe
default. q4 KV has the smallest footprint; q8 KV is a middle point, but its
measured 8k decode was slower in this run. q4 remains the primary 8k candidate
until repeated measurements and quality results establish a winner.

Raw records:

- `experiments/results/p5_context_scaling_q4.jsonl`
- `experiments/results/p5_kv_tradeoff_q8.jsonl`
- `experiments/results/p5_kv_tradeoff_f16.jsonl`

The exact FA/KV/state lines are retained in the corresponding ignored P5 logs.
