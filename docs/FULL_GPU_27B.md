# P4 FULL_GPU 27B

## Result

The first real Qwen3.8-27B GGUF run reached `FULL_GPU_CONFIRMED` on the RTX 5060. The
runner reported `65/65 layers` on CUDA0 and a CUDA model buffer of `6521.13 MiB`.
No decoder layer was placed on the CPU. The `397.85 MiB CPU_Mapped model buffer` is
the memory-mapped host view of the GGUF file, not a CPU decoder-layer allocation.

The requested c128 run exposed a llama.cpp/model constraint: the runtime created an
actual `n_ctx=256` even when `-c 128` was supplied. With the 22-token prompt, a
256-token total budget therefore produced 234 generated tokens. A separate ctx512
control run produced 256 generated tokens, so the minimum-token requirement was met
without mislabeling that control run as c128.

| Run | Requested context | Actual context | Prompt | Generated | Prefill | Decode | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c128 request | 128 | 256 (runtime clamp) | 22 | 234 | 30.20 tok/s | 31.39 tok/s | 7611 MiB |
| token-count control | 512 | 512 | 22 | 256 | 16.80 tok/s | 18.11 tok/s | 7592 MiB |

The c128-request run is the primary FULL_GPU result. The ctx512 row exists only to
prove a complete 256-generated-token pass; it is not a c128 claim.

## Hardware and artifact

- GPU: NVIDIA GeForce RTX 5060, 8151 MiB, compute capability 12.0 / SM120.
- Driver: 610.88; PCIe Gen4 x8 at measurement time.
- Model: `Qwen3.8-27B-UD-IQ2_XXS.gguf`, 26.90B parameters, IQ2_XXS, 2.0625 bpw.
- Artifact source: `unsloth/Qwen3.8-27B-GGUF` at revision
  `4ca720788d1e01f1bff70c033e0d0028fd02e502`.
- Artifact size: `7266070528` bytes.
- Artifact SHA256:
  `e792d8fb3142fe6d9171876d6da0f71f05a71028718debc72dbec93ff645e67d`.
- Model metadata license: Apache-2.0.

## Reproduction profile

The run used the official llama.cpp Windows CUDA binary `b10982` / commit
`fc82583e6`, `-ngl 999`, `--device CUDA0`, `--split-mode none`, `--fit off`, Flash
Attention, q4_0 K/V cache, warmup enabled, `n_batch=32`, `n_ubatch=1`, one active
sequence, temperature 0, reasoning off, and no speculative decoding or MTP.

The logical batch is 32 only because the CLI needs room for the prompt; the physical
decode microbatch is 1. The earlier literal `batch=1` attempt hit llama.cpp's
`n_tokens_all <= n_batch` assertion before model execution.

## Limits of this result

- The requested 25 tok/s target is exceeded by the c256 runtime-clamped decode
  (31.39 tok/s), but not by the ctx512 control (18.11 tok/s).
- No 8k run was performed yet.
- No quality gate was performed, so this result says nothing about accuracy loss.
- Per-token p95 latency was not exposed by this llama.cpp CLI capture; aggregate
  decode timing and 500 ms `nvidia-smi` samples are recorded instead.
- Raw records and source logs are listed in
  `experiments/results/p4_full_gpu_27b.jsonl` and the adjacent ignored `p4_*` logs.
