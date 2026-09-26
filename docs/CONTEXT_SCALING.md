# P5 context scaling

Status: measured on 2026-09-25 with the official llama.cpp CUDA runtime
`b10982`, commit `fc82583e6`, and
`Qwen3.8-27B-UD-IQ2_XXS.gguf` (SHA256
`e792d8fb3142fe6d9171876d6da0f71f05a71028718debc72dbec93ff645e67d`).

## Why the old c256/c512 numbers differed

The earlier P4 values were not a controlled context curve. The c256 run used the
direct CLI path and was runtime-clamped to an actual context of 256 with a short
prompt; the c512 control used a different prompt/run shape and a longer decode.
They also had different warmup/telemetry conditions. The observed difference
therefore cannot be attributed to context size alone.

P5 uses one server runner, the same model/runtime, seed 42, temperature 0,
`flash_attn=on`, GPU layers 999, CUDA0, batch 32, ubatch 1, one slot, warmup,
streaming output and 256 requested generated tokens. The prompt family is
deterministic and grows to leave room for generation. c256 is structurally
limited to 246 generated tokens because the 10-token prompt plus generation
fills the 256-token context.

## Standardized curve

| requested/actual ctx | prompt / generated | prefill tok/s | decode tok/s | TTFT ms | p50 / p95 inter-token ms | peak VRAM MiB | min free RAM MiB | quality |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 256 / 256 | 10 / 246 | 17.83 | 19.06 | 581.6 | 53.67 / 58.82 | 7479 | 16 | NOT_RUN |
| 512 / 512 | 180 / 256 | 20.36 | 19.02 | 8870.0 | 53.86 / 59.19 | 7483 | 7 | NOT_RUN |
| 1024 / 1024 | 519 / 256 | 20.57 | 19.06 | 25255.6 | 53.92 / 59.71 | 7495 | 213 | NOT_RUN |
| 2048 / 2048 | 1191 / 256 | 20.58 | 19.00 | 57887.6 | 53.63 / 59.14 | 7525 | 971 | NOT_RUN |
| 4096 / 4096 | 2544 / 256 | 25.10 | 18.96 | 101370.8 | 54.27 / 59.96 | 7584 | 511 | NOT_RUN |
| 8192 / 8192 | 5240 / 256 | 32.58 | 29.04 | 160842.6 | 34.33 / 35.20 | 7641 | 228 | NOT_RUN |

The c8192 q4 decode point is above the 25 tok/s research target in this single
run, but it is an outlier against the c256-c4096 q4 curve and against the Q8/F16
8k runs. It is recorded as an observation, not as a stable guarantee. Repeat
runs and the quality gate are still required.

## Runtime and memory path

The P5 logs prove `flash_attn = enabled` for every cache format tested; no
fallback message was observed. The model is fully resident from the runtime's
point of view (`offloaded 65/65 layers to GPU`), with a small memory-mapped CPU
model buffer of 397.85 MiB and a CUDA0 model buffer of 6521.13 MiB.

The GGUF metadata identifies a hybrid Qwen35 layout: 24 attention heads, 4 KV
heads, full attention interval 4, and SSM/GDN state metadata. llama.cpp reports
16 KV-cache layers and a separate CUDA0 recurrent-state buffer of 149.62 MiB.
The KV cache is not unified (`kv_unified = false`).

Raw records: `experiments/results/p5_context_scaling_q4.jsonl`.
Per-run logs and NVML samples are under the ignored `work/p5_context_scaling_q4/`
directory. The standardized runner is
`benchmark/run_p5_context_scaling.py`.

## Interpretation

The main c256/c512 discrepancy is resolved as a measurement-protocol problem,
not evidence that c512 inherently halves decode speed. Under the P5 protocol
those two points are 19.06 and 19.02 tok/s. The 8k result is memory-feasible
with q4 KV, but its latency/quality claim remains provisional until repeated and
quality-gated.
