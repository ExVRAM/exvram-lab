# exvram-cuda

Windows launcher for the ExVRAM llama.cpp recipe. It starts the CUDA build of `llama-server` with the same placement flags as `benchmark/run_p6_reproducibility.py`: every decoder layer on `CUDA0`, `--fit off`, direct I/O, q4_0 KV, flash attention, and no checkpoint or RAM cache. Kernels stay in `ggml-cuda.dll`.

`--embedding-gpu on` (the default) adds `--override-tensor token_embd.weight=CUDA0`, the same switch `plan-gpu-vram` emits for a profile whose embedding is on the GPU. The process then requires graph splits of 1. `--embedding-gpu off` matches the measured 27B run, which left the token embedding in a host buffer.

The executable looks upward from the working directory for:

- `.cache/llama/runtime-b10982-cuda13.4/llama-server.exe`
- `models/qwen35-2b/Qwen3.5-2B-Q4_K_M.gguf`

Both stay outside Git. Override them with `--runtime` and `--model`. Default context is 8448, the same window as the VRAM planner.

## Build

From a Visual Studio 2022 x64 developer prompt, in this directory:

```bat
cl /nologo /EHsc /std:c++20 /W4 /O2 /utf-8 /Fe:exvram-cuda.exe main.cpp winhttp.lib ws2_32.lib
```

CMake, from the repository root after `vcvars64.bat`:

```bat
cmake -S cpp/cuda_launch -B cpp/cuda_launch/build -G "NMake Makefiles"
cmake --build cpp/cuda_launch/build
```

`build/` is gitignored.

## Run

```bat
exvram-cuda.exe --prompt "What is 2+2? Reply with one number." --predict 16
```

Without `--prompt` the process stays up, reads lines, and stops on `/exit` or Ctrl+C. A successful start prints equal `gpu_layers` counts, a CUDA0 model buffer, and `ggml_cuda_loaded: yes`.
