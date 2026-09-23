"""CUTLASS is treated as an optional CUDA building-block/reference boundary."""

from __future__ import annotations

import importlib.util


def api_status() -> dict[str, object]:
    available = importlib.util.find_spec("cutlass") is not None
    return {
        "available": available,
        "project": "CUTLASS",
        "entrypoint": "selected component from the official CUTLASS checkout",
        "detail": "component-level license review required before redistribution",
    }

