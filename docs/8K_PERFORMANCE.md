# P5 8k performance result

## Result

The base IQ2_XXS Qwen3.8-27B GGUF reached an actual context of 8192 with all
65 layers on the RTX 5060 CUDA device. The q4_0 KV run measured 29.04 decode
tok/s, 32.58 prefill tok/s, 160.84 s TTFT, 34.33 ms p50 and 35.20 ms p95
inter-token time. Peak VRAM was 7641 MiB of 8151 MiB.

This clears the numerical `>=25 tok/s` target in one standardized run, but it is
not yet a release claim: the result is not repeated, the quality gate is
`NOT_RUN`, and the matched Q8/F16 runs measured 18.74/19.60 decode tok/s. The
P5 c256-c4096 q4 curve was about 19 tok/s, so the c8192 speed increase must be
treated as a variance/outlier investigation item.

| setting | prefill tok/s | decode tok/s | TTFT ms | p50 / p95 ms | peak VRAM MiB | GPU util max | power max W |
|---|---:|---:|---:|---:|---:|---:|---:|
| q4 KV, c8192 | 32.58 | 29.04 | 160842.6 | 34.33 / 35.20 | 7641 | 99% | 146.11 |
| q8 KV, c8192 | 20.49 | 18.74 | 255778.4 | 53.90 / 60.06 | 7687 | 99% | 115.99 |
| f16 KV, c8192 | 15.54 | 19.60 | 337171.3 | 50.33 / 53.74 | 7764 | 100% | 109.76 |

## Reproduction

Use the recipe in `recipes/rtx5060_8gb_qwen38_27b_8k.yaml` and the runner:

```powershell
.\.venv\Scripts\python.exe benchmark\run_p5_context_scaling.py `
  --server <llama-server.exe> `
  --model <Qwen3.8-27B-UD-IQ2_XXS.gguf> `
  --contexts 8192 --cache-type-k q4_0 --cache-type-v q4_0 `
  --decode-tokens 256 --allow-removable-storage
```

The command writes a JSONL record, a server log and an NVIDIA telemetry CSV.
Quality evaluation is intentionally separate and must not be inferred from the
speed run.

## Open gates

1. Repeat q4 c8192 under a controlled idle system and report variance.
2. Run the quality suite against an unquantized or accepted reference.
3. Verify long-context correctness, not only that the server accepts `n_ctx=8192`.
4. Profile only the winning c8192 configuration; do not write a custom decoder or
   kernel before the existing-runtime gap is demonstrated.
