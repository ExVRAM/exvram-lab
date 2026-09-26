# Open-source reuse matrix

| Project | Role in ExVRAM Lab | License boundary | Current use |
|---|---|---|---|
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | EXL3 runtime, model loading, cache/offload experiments | MIT code; model/artifact terms separate | Installed 1.5.1 and measured at P1 layer level; P2 full-model runner added |
| [GemLite](https://github.com/dropbox/gemlite) | Triton low-bit layer kernels | Apache-2.0 | Installed and measured W2/W4 fixtures at P1 |
| [CUTLASS](https://github.com/NVIDIA/cutlass) | NVIDIA CUDA/Tensor Core building blocks | Review exact component/EULA before reuse | Reference only; no copied code |
| [BitNet](https://github.com/microsoft/BitNet) | W2A8 reference and possible adapter | MIT code; model terms separate | Reference only; no copied code |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF baseline and CUDA runtime | MIT code; quant/model terms separate | Official b10964 CUDA 13.3 binary staged; full-model P2 pending artifact |
| [Ollama](https://github.com/ollama/ollama) | Installed-model CLI baseline and smoke runtime | MIT code; model terms separate | Local qwen2.5-coder:7b smoke verified; no model download through ExVRAM |
| [Marlin](https://github.com/IST-DASLab/marlin) | INT4 GEMM reference | Revision-level license and SM120 applicability review required | No reuse approval yet | Reference only; no copied code |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | Attention and serving kernels | Apache-2.0 project; component terms require review | No current ExVRAM checkpoint path | Reference only; no copied code |
| [GPTQModel](https://github.com/ModelCloud/GPTQModel) | External quantization/runtime integration | Apache-2.0 core; optional components may differ | No immutable revision selected | Evaluate externally; do not vendor blindly |
| [QTIP](https://github.com/Cornell-RelaxML/qtip) | GPL research reference only | GPL-3.0; not compatible for copied Apache-2.0 code | Never vendor or copy into this repository |

## Model artifacts

The official `Qwen/Qwen3.8-27B` checkpoint is Apache-2.0. Third-party EXL3/GGUF conversion
artifacts are recorded separately in
[`experiments/manifests/qwen38_p2.json`](../experiments/manifests/qwen38_p2.json). An artifact
without an explicit license statement is marked as `artifact_terms_unspecified...`; that label
is not a commercial-compatibility approval. The repository does not redistribute model weights.

Minimal-refusal checkpoints and the external abliteration tools are recorded in
[UNCENSORED_OSS_MATRIX.md](UNCENSORED_OSS_MATRIX.md). Heretic is AGPL-3.0 and is not vendored.
The 2026-09-24 gate is requantization of a published Heretic BF16 checkpoint, not a new
first-party uncensoring method.
