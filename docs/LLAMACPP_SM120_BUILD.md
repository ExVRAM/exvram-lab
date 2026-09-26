# llama.cpp SM120 build

Status: **not built from source**. `nvcc` is not on this machine, and Nsight is not installed.
A native Release build with an explicit SM120 architecture flag was therefore not produced.

## What was checked

| Check | Result |
|---|---|
| CUDA Toolkit `nvcc` | not found |
| CMake | 4.3.2 is installed |
| MSVC `cl` | not on `PATH` |
| Upstream commit built here | none |
| Existing binary | llama.cpp `b10964`, commit `b29c606e2`, Clang 20.1.8, Windows x86_64 |
| CUDA backend | `llama-cli --list-devices` lists `CUDA0: NVIDIA GeForce RTX 5060 (8150 MiB)` |
| `sm_120` string | present in `ggml-cuda.dll` and the bundled CUDA 13 cuBLAS libraries |

A string hit is not a build manifest. It does not prove that every kernel was compiled as
native `sm_120`, that CUDA graphs were enabled, or that Flash Attention quant kernels for
`q4_0` were compiled in. Those flags have to come from a source configure log, which this
machine cannot produce until `nvcc` and a C++ toolchain are installed.

## Required before the next build

- CUDA Toolkit with `nvcc` 12.8 or newer.
- A host C++ compiler visible on `PATH`.
- Configure for Release, native SM120, CUDA graphs on, and the specific Flash Attention quant
  pairs under test. Do not turn on a blanket FA-quant set without naming the K/V pairs.
- Record the exact upstream SHA, cmake cache, and the runtime device line after the binary
  starts.

Until that build exists, speed comparisons against this prebuilt bundle stay labeled as
prebuilt-CUDA-13.3, not as an ExVRAM SM120 performance build.
