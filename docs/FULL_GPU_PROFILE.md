# FULL_GPU Profile

This profile is based on two real llama.cpp runs and 500 ms `nvidia-smi` samples.
It is not a synthetic estimate.

## Hardware

| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 |
| VRAM reported | 8151 MiB |
| Compute capability | 12.0 / SM120 |
| Driver | 610.88 |
| PCIe | Gen4 x8 current, x16 maximum |
| Power limit | 145 W |

## Residency and memory map

Both runs reported:

```text
offloaded 65/65 layers to GPU
CPU_Mapped model buffer size = 397.85 MiB
CUDA0 model buffer size      = 6521.13 MiB
```

The c256 primary run reported:

```text
CUDA0 total 8150 = free 58 + (self 6681 = model 6521 + context 154 + compute 6) + unaccounted 1410 MiB
Host                398 = model 397 + context 0 + compute 0 MiB
CUDA0 KV buffer     4.50 MiB (K q4_0 2.25 + V q4_0 2.25)
```

The ctx512 token-count control reported a 9.00 MiB CUDA KV buffer and 158 MiB
context contribution. No CPU context or compute buffer was reported.

## Telemetry

| Run | Samples | Peak VRAM | Minimum free | Max GPU util | Max memory util | Max SM clock | Max power |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c128 request / actual c256 | 30 | 7611 MiB | 286 MiB | 98% | 57% | 2820 MHz | 145.5 W |
| ctx512 / 256 generated | 67 | 7592 MiB | 305 MiB | 99% | 46% | 2835 MHz | 107.65 W |

The peak values include Windows display allocation and the normal CUDA/runtime
overhead. A separate llama.cpp process from another data volume appeared during an
earlier interrupted attempt; it was not part of the accepted measurements
and was not terminated by ExVRAM.

## Timing

The accepted primary run used warmup, Flash Attention, q4_0 K/V, `n_batch=32`, and
`n_ubatch=1`. Aggregate llama.cpp timings are the source of truth:

- actual c256: 22 prompt tokens at 30.20 tok/s; 234 generated at 31.39 tok/s;
  total 8,151.58 ms;
- ctx512 control: 22 prompt tokens at 16.80 tok/s; 256 generated at 18.11 tok/s;
  total 15,392.14 ms.

The CLI did not emit per-token timestamps, so p95 token latency is left unreported
rather than inferred from aggregate averages.
