"""Thin GemLite constructor boundary following the upstream public README API."""

from __future__ import annotations

import importlib.util
from typing import Any


def api_status() -> dict[str, Any]:
    available = importlib.util.find_spec("gemlite") is not None
    return {
        "available": available,
        "project": "GemLite",
        "entrypoint": "gemlite.GemLiteLinear",
        "detail": (
            "supports pack/forward fixture when installed"
            if available
            else "module unavailable"
        ),
    }


def create_linear(
    weight_bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
    *,
    input_dtype: str = "FP16",
    output_dtype: str = "FP16",
) -> Any:
    """Construct the upstream GemLite module; packing remains caller-owned."""
    from gemlite import DType, GemLiteLinear

    return GemLiteLinear(
        weight_bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        input_dtype=getattr(DType, input_dtype),
        output_dtype=getattr(DType, output_dtype),
    )
