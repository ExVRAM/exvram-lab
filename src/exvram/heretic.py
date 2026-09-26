"""External command boundary for Heretic. ExVRAM does not vendor or run its optimizer."""

from __future__ import annotations

import shutil
from typing import Any

HERETIC_REPOSITORY = "https://github.com/p-e-w/heretic"
HERETIC_REVISION_CHECKED = "3521f8648a0dccf6e12a92666862632235fac7e6"
HERETIC_LICENSE = "AGPL-3.0"


def build_heretic_command(
    executable: str,
    model: str,
    *,
    evaluate_model: str | None = None,
) -> list[str]:
    if not executable.strip():
        raise ValueError("executable is required")
    if not model.strip():
        raise ValueError("model is required")
    if evaluate_model is not None and not evaluate_model.strip():
        raise ValueError("evaluate_model must be a non-empty model id")
    if evaluate_model:
        return [executable, "--model", model, "--evaluate-model", evaluate_model]
    return [executable, model]


def plan_heretic(
    model: str,
    *,
    evaluate_model: str | None = None,
    executable: str = "heretic",
) -> dict[str, Any]:
    resolved = shutil.which(executable)
    return {
        "tool": "heretic",
        "repository": HERETIC_REPOSITORY,
        "revision_checked": HERETIC_REVISION_CHECKED,
        "license": HERETIC_LICENSE,
        "copied_code": False,
        "executed": False,
        "role": "external_research_tool",
        "availability": "available" if resolved else "unavailable",
        "resolved_executable": resolved,
        "command": build_heretic_command(
            executable, model, evaluate_model=evaluate_model
        ),
    }
