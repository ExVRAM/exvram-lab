# P5 8k performance result

## Result

The P5 q4 c8192 point of 29.04 tok/s was **not reproduced**. P6 repeated the
same short prompt five times at configured c8192 and measured a median of
17.67 tok/s, mean 17.67 tok/s, CV 0.74%, with 266 occupied tokens. This is a
configured-context control, not a real filled 8k quality result.

The old P5 prompt had 5240 prompt tokens plus 256 generated tokens, so its final
occupancy was about 5496, not 8192. A true filled-context smoke reached the
correct occupancy target but its prefill was too slow to complete safely under
the existing `ubatch=1` path. The 8k target therefore remains unconfirmed.

| setting | prefill tok/s | decode tok/s | TTFT ms | p50 / p95 ms | peak VRAM MiB | GPU util max | power max W |
|---|---:|---:|---:|---:|---:|---:|---:|
| q4 KV, c8192, P5 single run | 32.58 | 29.04 | 160842.6 | 34.33 / 35.20 | 7641 | 99% | 146.11 |
| q4 KV, c8192, P6 short median | not comparable | 17.67 | measured per run | median p50/p95 in raw JSONL | ~7.6 GiB | 99% max | see telemetry |
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

1. Run a safe filled-context protocol with a runtime configuration that completes
   prefill without hour-scale stalls.
2. Run the quality suite against an unquantized or accepted reference.
3. Verify long-context correctness, not only that the server accepts `n_ctx=8192`.
4. Profile only the winning c8192 configuration; do not write a custom decoder or
   kernel before the existing-runtime gap is demonstrated.
