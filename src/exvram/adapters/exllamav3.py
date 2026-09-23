"""Thin, optional ExLlamaV3 API probe; the runtime remains an external dependency."""

from __future__ import annotations

import importlib.util
from typing import Any


def api_status() -> dict[str, Any]:
    available = importlib.util.find_spec("exllamav3") is not None
    return {
        "available": available,
        "project": "ExLlamaV3",
        "entrypoint": "exllamav3.modules.linear.Linear",
        "detail": (
            "requires a concrete EXL3 checkpoint/qmap fixture"
            if available
            else "module unavailable"
        ),
    }


def load_linear_api() -> Any:
    """Return the upstream Linear class without vendoring or wrapping its implementation."""
    from exllamav3.modules.linear import Linear

    return Linear
