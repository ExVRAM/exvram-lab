"""Plan the unfinished Q4 needles. This module does not start a model."""

from __future__ import annotations

import math

from .breakpoint import missing_long_context

Q4_SHA256 = "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223"
Q4_BYTES = 1_280_835_840
Q4_ROLE = "qwen35_2b_q4_k_m"
SHORT_OUTPUT = "experiments/results/p9_qwen35_2b_q4.jsonl"
LONG_OUTPUT = "experiments/results/p9_qwen35_2b_q4_long.jsonl"
MODEL_RELATIVE = "models/qwen35-2b/Qwen3.5-2B-Q4_K_M.gguf"
SERVER_RELATIVE = ".cache/llama/runtime-b10982-cuda13.4/llama-server.exe"
BENCHMARK_RAM_MIB = 2048


def original_q4_gap() -> tuple[str, ...]:
    have = {f"needle:2048:{index}" for index in range(15)}
    have.update(f"needle:4096:{index}" for index in range(5))
    return tuple(missing_long_context(have))


def resume_command() -> list[str]:
    return [
        "python",
        "-m",
        "benchmark.run_p9_quality",
        "--server",
        SERVER_RELATIVE,
        "--model",
        MODEL_RELATIVE,
        "--model-sha256",
        Q4_SHA256,
        "--model-role",
        Q4_ROLE,
        "--output",
        SHORT_OUTPUT,
        "--long-output",
        LONG_OUTPUT,
    ]


def q4_resume_plan(
    measured_long_ids: set[str],
    *,
    ram_available_mib: int | None = None,
) -> dict[str, object]:
    missing = missing_long_context(measured_long_ids)
    gap = set(original_q4_gap())
    if ram_available_mib is None:
        resource = "not_measured"
    elif ram_available_mib < BENCHMARK_RAM_MIB:
        resource = "defer"
    else:
        resource = "within_floor"
    return {
        "launch": False,
        "reason": "owner permission is required",
        "missing": missing,
        "outside_original_gap": [item for item in missing if item not in gap],
        "resource": resource,
        "command": resume_command(),
        "appends_one_row_at_a_time": True,
    }


def paired_wald(candidate: list[bool], reference: list[bool]) -> dict[str, float]:
    if len(candidate) != len(reference) or not candidate:
        raise ValueError("paired flags must be the same non-empty length")
    both = list(zip(candidate, reference, strict=True))
    gained = sum(1 for left, right in both if left and not right)
    lost = sum(1 for left, right in both if right and not left)
    diff = (sum(candidate) - sum(reference)) / len(both)
    variance = ((gained + lost) - (gained - lost) ** 2 / len(both)) / len(both) ** 2
    margin = 1.96 * math.sqrt(variance)
    return {"diff": diff, "low": diff - margin, "high": diff + margin}
