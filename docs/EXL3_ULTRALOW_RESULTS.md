# EXL3 Ultra-Low Results

## Status

No full-model EXL3 result is claimed yet. The official ExLlamaV3 Windows wheel was
installed and imported in the isolated GPU environment, but the EXL3 model artifact
was deliberately not downloaded before the GGUF FULL_GPU gate completed.

Planned artifact: official `turboderp/Qwen3.8-27B-exl3` model repository. Its exact
revision, files, hashes, and model terms must be recorded before download.

## Runtime prepared

- ExLlamaV3 wheel: `1.5.1+cu128.torch2.10.0`, SHA256
  `cab2d227383f648b8c4d6ff05b6510ca809d17b5b928fece870b5500d2e5a990`.
- PyTorch: `2.10.0+cu128`.
- CUDA runtime reported by PyTorch: 12.8.
- Windows Triton package: `3.8.0.post28`.

The environment imports on SM120. Triton reports that `ptxas-blackwell.exe` is not
present in the current package; this is an environment limitation to resolve or
document before interpreting an EXL3 kernel result. No EXL3 tok/s, VRAM, fit, or
quality number is present in this repository yet.

## Gate before EXL3

1. Acquire the official model revision and verify its hashes.
2. Run a minimal ExLlamaV3 load without CPU decoder offload.
3. Record actual layer residency, VRAM peak, prefill, decode, and quality against the
   verified GGUF reference.
4. Keep EXL3 code external/optional; no runtime source is copied into ExVRAM.
