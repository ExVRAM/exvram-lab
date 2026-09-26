# OSS gap search

This is the pre-custom-code audit for the current target. ExVRAM does not copy any of these
projects into the Apache-2.0 codebase.

| Project | Relevant capability | RTX 5060 / SM120 status | License boundary | Decision |
|---|---|---|---|---|
| ExLlamaV3 | EXL3 quantization, GPU inference, cache/offload path | Layer path measured; target full-model path has compatibility/fit failures | MIT; model/artifact terms separate | Keep as primary candidate |
| GemLite | Reusable low-bit Triton matmul kernels | W2/W4 layer fixtures measured on RTX 5060; not a full-model loader | Apache-2.0 | Keep as layer candidate |
| CUTLASS | CUDA/Tensor Core primitives, including Blackwell families | SM120 support is upstream capability, not ExVRAM measurement | BSD-3-Clause for applicable core; inspect component terms | Use only through an adapter/build boundary |
| BitNet GPU | W2A8/ternary format-specific kernels | Not a generic dense Qwen3.8-27B path in this repo | MIT code; model terms separate | Reference only until format matches |
| llama.cpp | GGUF runtime and broad low-bit formats | Target c128 full-model smoke measured; decode 3.8 tok/s | MIT; model/quant/runtime terms separate | Keep as full-model baseline |
| Marlin | FP16 x INT4 GEMM | Existing project documents Ampere/Ada focus; no SM120 result here | Verify exact upstream revision before reuse | Reference only; no copy |
| FlashInfer | Attention and serving kernels | Relevant SM120 work exists upstream, but no current ExVRAM checkpoint path | Apache-2.0 project with component-level license review required | Reference only |
| GPTQModel | Quantization/runtime integration and Marlin/EXL3/GGUF backends | Windows support is documented upstream; no target measurement here | Apache-2.0 core; optional vendored components can have separate terms | Evaluate as external integration, never vendor blindly |

## Gate result

`NO_CUSTOM_CODE`.

The existing paths are not yet compared on one identical checkpoint and quality protocol. The
first justified optimization is orchestration plus measurement: fix compatibility, run bounded
context/offload/KV experiments, and update this table with exact revisions and numbers. A custom
kernel can be reopened only after a same-shape, same-format, profiler-backed gap survives this
search.
