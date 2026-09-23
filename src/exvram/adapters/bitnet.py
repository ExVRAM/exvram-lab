"""BitNet is probed as an external format-specific runtime, not a generic Linear backend."""

from __future__ import annotations

import importlib.util


def api_status() -> dict[str, object]:
    available = importlib.util.find_spec("bitnet") is not None
    return {
        "available": available,
        "project": "BitNet",
        "entrypoint": "official GPU kernel under the BitNet repository",
        "detail": "format-specific; not assumed compatible with the dense placeholder model",
    }

