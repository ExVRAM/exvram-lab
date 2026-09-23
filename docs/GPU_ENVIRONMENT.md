# GPU environment record

Captured on 2026-09-23 in the existing `exvram-lab` repository. The global interpreter still uses
CPU-only Torch, but its editable `exvram` package points at this project directory.

## Hardware and host

| Field | Observed value | Source/status |
|---|---|---|
| OS | Windows 10 Pro 25H2, build 26200.9457 | registry/platform telemetry |
| GPU | NVIDIA GeForce RTX 5060 | `nvidia-smi` and Torch |
| Driver | 610.88 | `nvidia-smi` |
| VRAM | 8151 MiB / 7.959 GiB | `nvidia-smi`; Torch reports 8,546,353,152 bytes |
| Compute capability | 12.0 / SM120 | Torch and NVIDIA GPU table |
| Python | 3.11.9 | interpreter telemetry |
| `nvcc` | not found | environment probe |
| Nsight Compute (`ncu`) | not found | environment probe |

The RTX 5060 compute capability is independently listed as 12.0 in [NVIDIA's GPU table](https://developer.nvidia.com/cuda/gpus).

## Isolated runtimes

| Environment | Packages used | Purpose | Result |
|---|---|---|---|
| `.venv-gpu` | Torch `2.14.0+cu130`, CUDA runtime `13.0`; GemLite `0.6.0.post1`; `triton-windows` `3.8.0.post28` | Torch baselines and GemLite | CUDA available; GemLite kernels measured |
| `.venv-exllama` | Torch `2.10.0+cu128`, CUDA runtime `12.8`; ExLlamaV3 `1.5.1+cu128.torch2.10.0`; `triton-windows` `3.6.0.post26` | Official EXL3 fixture | CUDA available; `LinearEXL3` measured |
| global interpreter | Torch `2.14.0+cpu`, no CUDA runtime | preserved CPU workflow | unchanged; not used for GPU results |

The CUDA-enabled Torch smoke test allocated and synchronized CUDA tensors, reported the RTX 5060,
and successfully executed FP16 and BF16 matrix multiplications. PyTorch's official installation
selector and Blackwell guidance are documented at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
and in the [PyTorch 2.12 release notes](https://pytorch.org/blog/pytorch-2-12-release-blog/).

## Windows Triton note

A Unicode Windows project path caused Triton's cache creation to fail with a permission/path
decoding error. Measurements were run in a PowerShell process with a temporary drive-letter
mapping and an explicit cache directory:

```text
subst X: <repository-root>
TRITON_CACHE_DIR=X:\work\triton-cache
```

The mapping is process-local and is not a repository or system configuration change. GemLite emitted
a `ptxas-blackwell.exe` not-found warning because no standalone CUDA toolkit is installed, but the
Triton runtime path still launched real CUDA kernels and produced measured records. This caveat is
attached to the measurements; it is not hidden as a successful full-toolkit setup.

`pip check` reports that GemLite's metadata requires a distribution named `triton`, while the tested
Windows compatibility package is distributed as `triton-windows` and provides the imported `triton`
module. The runtime import and kernel execution were verified; this packaging mismatch is an
environment note to resolve before publishing a lockfile.

## Reproduction boundary

The repository code uses optional imports only. External runtime source is not copied into ExVRAM.
The exact installation/wheel provenance and license boundary are recorded in [THIRD_PARTY.md](../THIRD_PARTY.md).
The GPU environments are intentionally ignored by Git; a clean clone can recreate them without
affecting the global interpreter.

## Removable-storage safety

Earlier full-model artifacts and the llama.cpp binary were staged in an external data directory
outside this repository. That volume is not required for CPU/synthetic workflows. Do not run a long
benchmark against removable storage if the device is disconnecting or causing system lag. Move the
specific model and runtime artifacts to a stable data volume first, keep the process foregrounded,
and use a bounded timeout before resuming GPU measurements.
