# Third-party projects and license boundary

This file records the integration boundary as checked on 2026-09-23. ExVRAM Lab does not copy
third-party source code into this repository. External runtimes are optional integrations or
submodules to be selected later, with their own installation and license obligations.

| Project | Repository | Version/revision checked | License | Reused component | Modified status | Commercial compatibility |
|---|---|---|---|---|---|---|
| ExLlamaV3 | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | `1.5.1+cu128.torch2.10.0` official Windows wheel; SHA-256 `cab2d227383f648b8c4d6ff05b6510ca809d17b5b928fece870b5500d2e5a990` | MIT | Optional EXL3 quantizer/runtime API | Not copied or modified | Compatible with commercial use subject to MIT notice and upstream/model terms |
| GemLite | [dropbox/gemlite](https://github.com/dropbox/gemlite) | PyPI `0.6.0.post1`, measured in `.venv-gpu` | Apache-2.0 | Optional Triton low-bit matmul kernels | Not copied or modified | Compatible with commercial use subject to Apache notices/patent terms |
| triton-windows | [woct0rdho/triton-windows](https://github.com/woct0rdho/triton-windows) | `3.8.0.post28` with GemLite; `3.6.0.post26` with ExLlamaV3 | MIT package license | Windows compatibility distribution providing the `triton` module | Not copied or modified | Compatible subject to its MIT notice; upstream Triton and CUDA terms remain separate |
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch) | `2.14.0+cu130` GemLite env; `2.10.0+cu128` ExLlama env | BSD-style | Optional CUDA tensor/runtime and timing API | Not copied or modified | Compatible subject to PyTorch and CUDA redistribution terms |
| CUTLASS | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | `4.8.0` release shown by upstream on 2026-09-23 | BSD-3-Clause for applicable core components | Optional CUDA/CuTe building blocks after component-level review | Not copied or modified | Compatible for applicable BSD component; verify EULA-governed files separately |
| BitNet | [microsoft/BitNet](https://github.com/microsoft/BitNet) | `1.0` README version / default branch observed 2026-09-23 | MIT | Optional reference/GPU-kernel integration for BitNet-family models | Not copied or modified | Compatible with commercial use subject to MIT notice and model terms |
| llama.cpp | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | official Windows x64 CUDA 13.3 nightly `b10964`, commit `b29c606e28a01b1bc8c1351026a0fa6e616bf6c0` | MIT | Optional GGUF baseline executable/API | Not copied or modified | Compatible with commercial use subject to MIT notice, CUDA redistribution terms, and model/artifact terms |
| QTIP (reference only) | [Cornell-RelaxML/qtip](https://github.com/Cornell-RelaxML/qtip) | Default branch observed 2026-09-23 | GPL-3.0 | Research/reference comparison only | No GPL code copied | Not included in the Apache-2.0 codebase; use only as an external process/reference unless legal review says otherwise |
| Qwen3.8-27B base checkpoint | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | HF revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Apache-2.0 | P2 target model identity/reference | Not redistributed or copied into Git | Model terms must be followed separately from ExVRAM code |
| Qwen3.8-27B EXL3 conversion | [thelastspark/Qwen3.8-27B-exl3](https://huggingface.co/thelastspark/Qwen3.8-27B-exl3) | branch `4.00bpw`, HF revision `04a468b234ed309ef46b098b3451942f08333af1` | Apache-2.0 metadata | External P2 test artifact | Not redistributed or copied into Git | Use subject to artifact/model terms; runner records provenance |
| Qwen3.8-27B GGUF conversion | [ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) | HF revision `efbb3b1f70a21d97fd4495240648405f7228554f` | Apache-2.0 metadata | External P2 GGUF reference | Not redistributed or copied into Git | Use subject to artifact/model terms |
| Qwen3.8-27B IQ2 GGUF conversion | [bartowski/Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF) | HF revision `0c92138c51f112d2f0dc84d5f6d1ebe4b9912b9a`, IQ2_XXS | Artifact terms not explicitly declared in inspected card; base model Apache-2.0 | External low-memory P2 candidate | Not redistributed or copied into Git | Not a commercial-compatibility approval; perform legal review before redistribution |

## Important boundary notes

- A project's code license does not grant rights to model weights, tokenizer files, CUDA SDK
  components, or generated model artifacts. Those must be reviewed separately.
- CUTLASS is a repository containing components with different terms. This project does not copy
  the Python DSL or any EULA-governed material; a future adapter must record the exact component
  and revision it uses.
- Before publishing a lockfile, binary, wheel, or submodule update, replace branch/date entries
  with immutable commit IDs and refresh this table.

The measured P1 records set `upstream_revision_tested` to the exact package/runtime strings used for
GemLite and ExLlamaV3. The records are layer fixtures, not claims about a model checkpoint, model
license, full-model quality, or full-model performance. llama.cpp, BitNet and CUTLASS remain optional
integration references and were not exercised in this run. Before a release, replace mutable branch
references with immutable upstream commit IDs and refresh this table.
