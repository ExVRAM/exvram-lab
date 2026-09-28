from __future__ import annotations

from ..contracts import AdapterInfo
from .base import ExternalAdapter


def all_adapters() -> tuple[ExternalAdapter, ...]:
    return (
        ExternalAdapter(
            AdapterInfo(
                "exllamav3",
                "https://github.com/turboderp-org/exllamav3",
                "Primary optional local-GPU inference integration candidate.",
                "optional-python-runtime",
                "MIT",
                caution=(
                    "Install a CUDA/Torch-compatible release separately; "
                    "no source is copied here."
                ),
            ),
            import_names=("exllamav3",),
            installation_note="Follow ExLlamaV3's official release or source-install instructions.",
        ),
        ExternalAdapter(
            AdapterInfo(
                "gemlite",
                "https://github.com/dropbox/gemlite",
                "Optional low-bit Triton matmul kernel candidate.",
                "optional-python-kernel",
                "Apache-2.0",
                caution="Kernel support and performance must be measured on the target GPU.",
            ),
            import_names=("gemlite",),
            installation_note="Install GemLite from its official repository/package instructions.",
        ),
        ExternalAdapter(
            AdapterInfo(
                "hqq",
                "https://github.com/dropbox/hqq",
                "Optional HQQ 1/2-bit group-wise quantization boundary.",
                "optional-python-quantizer",
                "Apache-2.0",
                caution=(
                    "HQQ measurements must use the external implementation; ExVRAM does not "
                    "reimplement its quantizer."
                ),
            ),
            import_names=("hqq",),
            installation_note="Install HQQ from its official repository/package instructions.",
        ),
        ExternalAdapter(
            AdapterInfo(
                "cutlass",
                "https://github.com/NVIDIA/cutlass",
                "Optional CUDA/CUTLASS building-block integration candidate.",
                "optional-cuda-building-block",
                "BSD-3-Clause for applicable core components",
                caution=(
                    "CUTLASS distributions can contain separately governed components; "
                    "verify the exact component license before redistribution."
                ),
            ),
            import_names=("cutlass",),
            installation_note=(
                "Use the official CUTLASS checkout/build integration for the selected component."
            ),
        ),
        ExternalAdapter(
            AdapterInfo(
                "bitnet",
                "https://github.com/microsoft/BitNet",
                "Optional reference/integration candidate for ternary/low-bit kernels.",
                "external-cpp-gpu-reference",
                "MIT",
                caution=(
                    "BitNet's model format and kernels target BitNet-family models; "
                    "this is not a generic 27B adapter yet."
                ),
            ),
            import_names=("bitnet",),
            executable_names=("bitnet",),
            installation_note="Build/use BitNet through its official repository instructions.",
        ),
        ExternalAdapter(
            AdapterInfo(
                "llamacpp",
                "https://github.com/ggml-org/llama.cpp",
                "Optional baseline runtime for independent GGUF comparisons.",
                "external-executable-baseline",
                "MIT",
                caution=(
                    "Model conversion and model licenses remain separate from "
                    "llama.cpp's code license."
                ),
            ),
            executable_names=("llama-cli", "llama-server"),
            installation_note=(
                "Install/build llama.cpp using its official instructions, then expose "
                "llama-cli or llama-server on PATH."
            ),
        ),
        ExternalAdapter(
            AdapterInfo(
                "ollama",
                "https://github.com/ollama/ollama",
                "Optional baseline runtime for locally installed Ollama models.",
                "external-cli-runtime",
                "MIT",
                caution=(
                    "ExVRAM never downloads models through this adapter; model licenses and "
                    "Ollama distribution terms remain separate."
                ),
            ),
            executable_names=("ollama",),
            installation_note=(
                "Install Ollama using its official instructions; only models already shown by "
                "ollama list are eligible for an ExVRAM smoke run."
            ),
        ),
    )


def get_adapter(name: str) -> ExternalAdapter:
    for adapter in all_adapters():
        if adapter.info.name == name:
            return adapter
    known = ", ".join(item.info.name for item in all_adapters())
    raise KeyError(f"unknown adapter {name}; known adapters: {known}")
