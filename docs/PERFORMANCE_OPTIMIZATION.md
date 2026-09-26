# Performance optimization log

Optimization past the baseline is stopped. The 2026-09-24 classification is
`CPU_OFFLOAD_BOUND` for bartowski IQ2_XXS on llama.cpp `b10964`. See
[PERFORMANCE_BASELINE.md](PERFORMANCE_BASELINE.md).

## Why the later phases did not run

| Phase | State | Why |
|---|---|---|
| Flash Attention KV sweep | not run | FA on/off was not logged, and `nvcc` is absent, so the quant kernels cannot be rebuilt and checked |
| Full-GPU quant search | blocked | The only local GGUF is 8,881,268,448 bytes, larger than 8151 MiB. The local EXL3 4.00bpw artifact is about 16.8 GB and already failed with `Insufficient VRAM` |
| Batch / ubatch sweep | not run | Offload dominates; a grid search would not create a full-GPU point |
| ExLlamaV3 full-model | failed earlier | Six recorded attempts, no successful decode. Last errors are VRAM fit, not a tok/s |
| Nsight timeline | not run | `nsys` and `ncu` are not installed |
| Speculative decoding, MTP, DFlash | not run | No full-GPU target residency yet. Extra draft weights would need a memory check first |
| Prefill / TTFT program | partial | One short-prompt row exists (15.9 tok/s prompt, 5.5 tok/s generation). Cold/warm TTFT was not split |
| Quality suite | not run | No speed candidate has beaten the offload diagnosis |
| Custom CUDA kernel | not opened | The gate in the research plan is still closed |

## Bandwidth

`estimated_weight_bandwidth_gbs` is not reported as a ceiling measurement. nvidia-smi's
memory-controller utilization while the framebuffer was above 6000 MiB averaged 3.31%. That
is a utilization percent, not DRAM GB/s. The official RTX 5060 bandwidth figure was not
re-verified in this session, so utilization versus that spec stays `UNAVAILABLE`.

A derived product of the historical 5673 MiB GPU-resident model slice times 3.8 tok/s would
describe only the bytes that were on the GPU, and only if every decode reread them. That
assumption is not checked. It is not stored as a measured bandwidth.

## Next change that could move decode

Obtain one 27B quant whose on-disk model bytes, plus KV and compute buffers, fit inside the
8151 MiB device with margin. Then repeat the same prompt, predict length, and `dmon` capture.
Until that point, raising batch size or writing a kernel does not address the host-resident
2557 MiB.
