# GPU Python Environment

The GPU-only Python environment is isolated from the CPU/dev environment and lives
outside the tracked source tree in a local `.venv-gpu` directory.

| Component | Verified value |
| --- | --- |
| Host Python used to create env | 3.11.9 |
| PyTorch | 2.10.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| GPU | NVIDIA GeForce RTX 5060, SM120 |
| ExLlamaV3 | 1.5.1+cu128.torch2.10.0 official Windows wheel |
| GemLite | 0.6.0.post1 |
| triton-windows | 3.8.0.post28 |

The imports for `torch`, `exllamav3`, and `gemlite` succeed with CUDA available. The
official ExLlamaV3 wheel SHA256 is
`cab2d227383f648b8c4d6ff05b6510ca809d17b5b928fece870b5500d2e5a990`.

The current Triton package contains `ptxas.exe` but does not provide
`ptxas-blackwell.exe`; Triton therefore warns during import. This is recorded as an
environment caveat, not silently treated as an EXL3 performance result. The cache
locations used to keep generated files off the removable drive are:

```powershell
$env:TRITON_CACHE_DIR = '<project>\.cache\triton'
$env:TORCH_EXTENSIONS_DIR = '<project>\.cache\torch_extensions'
```

The standalone GGUF measurement used the official llama.cpp Windows CUDA 13.4
runtime b10982. Its two downloaded archive SHA256 values were verified before
extraction and are retained in the acquisition notes.
