# Performance baseline

Status date: 2026-09-24. This file records what was measured on the RTX 5060. It does not
treat ~3.8 tok/s as a tuned GPU baseline.

## Artifact

| Field | Value | Status |
|---|---|---|
| Model | `Qwen3.8-27B` | manifest |
| Quant | bartowski IQ2_XXS | runtime banner: `IQ2_XXS - 2.0625 bpw` |
| GGUF size | 8,881,268,448 bytes | file size |
| llama.cpp | build `b10964`, commit `b29c606e2`, version `0.4.1-dev` | `llama-cli --version` |
| Binary CUDA | prebuilt Windows CUDA 13.3 bundle | `llama-cli --list-devices` sees CUDA0 |
| `nvcc` | not installed | environment probe |
| Driver | 610.88 | `nvidia-smi` |
| GPU | RTX 5060, CC 12.0, 8151 MiB | `nvidia-smi` |
| Binary strings | `sm_120` present in `ggml-cuda.dll` and the CUDA 13 cuBLAS DLLs | string scan, not a cubin audit |
| `n_gpu_layers` | `auto` | launch flag |
| K / V cache | `q4_0` / `q4_0` | launch flag |
| Context / batch | 128 / batch and ubatch not overridden | launch flag |
| Flash Attention | not present in the captured log | unknown |
| CUDA graphs | not present in the captured log | unknown |
| mmap / mlock | not present in the captured log | unknown |
| Nsight Systems / Compute | not installed | environment probe |

The card's 2.56 bpw figure is provenance only. The runtime ftype line says 2.0625 bpw.

## Historical c128 capture

The earlier text-only run, 32 predicted tokens, returned decode **3.8 tok/s** and prefill
**31.3 tok/s** after 284.83 s wall time. Its final llama.cpp fit snapshot was:

| Placement | Model MiB | Context MiB | Compute MiB |
|---|---:|---:|---:|
| CUDA0 RTX 5060 | 5673 | 112 | 71 |
| Host | 2557 | 41 | 7 |

Device free memory in that snapshot was 744 MiB of 8150 MiB. About 2557 MiB of the model stayed
on the host. Intermediate fit lines in the same log moved the model between GPU and host before
settling there. That is partial offload, not full GPU residency.

## Bounded rerun on 2026-09-24

Same binary, same GGUF, same `auto` offload and `q4_0` KV, context 128, 16 predicted tokens,
short prompt. Exit code 0. Wall time 199.6 s.

| Metric | Value |
|---|---:|
| Prompt | 15.9 tok/s |
| Generation | 5.5 tok/s |

`nvidia-smi dmon` during that process, 198 one-second samples:

| Window | SM util | Memory-controller util | Framebuffer |
|---|---:|---:|---:|
| Whole run max | 29% | 20% | 7191 MiB |
| Samples with framebuffer ≥ 6000 MiB (n=104) | mean 6.95%, max 29% | mean 3.31%, max 20% | mean 6966 MiB |

Low SM and memory-controller utilization while the framebuffer is near full means the GPU is
resident but stalled. This rerun is not a replacement for the 3.8 tok/s row: the prompt and
predict length differ. Both rows are real. Neither is an 8k or quality result.

PCIe at idle before the rerun was Gen4 x8. The reported maximum is Gen4 x16. Link width was not
sampled during decode. Display was active. Power limit 145 W. Clocks at idle were not the
boost clocks (max SM clock reported 3090 MHz).

## Classification of the ~3.8 tok/s result

`CPU_OFFLOAD_BOUND`

Reasons:

- The GGUF is 8,881,268,448 bytes, larger than the 8151 MiB device, so this artifact cannot be
  fully GPU-resident.
- The fit log left 2557 MiB of model weights on the host.
- While framebuffer use was high, SM utilization averaged about 7% and the memory controller
  about 3%. That is not a GPU compute ceiling and not a GPU DRAM ceiling.

Not selected: `GPU_COMPUTE_BOUND`, `GPU_MEMORY_BOUND`. `PCIE_BOUND` is plausible but not
separated from CPU execution, because PCIe byte counters and a CUDA trace were not collected.
Flash Attention fallback remains `UNKNOWN`.

No further kernel tuning is justified on this binary until a quant that can reside on the GPU,
or a measured full-GPU fit, exists.
