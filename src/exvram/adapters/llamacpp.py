"""Command builder for the official llama-bench executable."""

from __future__ import annotations

from pathlib import Path


def build_llama_bench_command(
    executable: str, model: str | Path, *, repetitions: int = 5, gpu_layers: int = 99
) -> list[str]:
    return [
        executable,
        "-m",
        str(model),
        "-r",
        str(repetitions),
        "-ngl",
        str(gpu_layers),
        "-o",
        "jsonl",
        "--no-warmup",
    ]

