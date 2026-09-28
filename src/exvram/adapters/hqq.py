"""Thin optional HQQ boundary; HQQ source is never vendored by ExVRAM."""

from __future__ import annotations

import importlib.metadata
import importlib.util
from typing import Any


def api_status() -> dict[str, Any]:
    available = importlib.util.find_spec("hqq") is not None
    try:
        version = importlib.metadata.version("hqq")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "available": available,
        "project": "HQQ",
        "entrypoint": "hqq.core.quantize.HQQLinear",
        "version": version,
        "detail": (
            "requires external torch and HQQ quantization API"
            if available
            else "module unavailable"
        ),
    }


def build_quantize_config(nbits: int, group_size: int) -> Any:
    """Construct the upstream HQQ config without reimplementing quantization."""
    from hqq.core.quantize import BaseQuantizeConfig

    return BaseQuantizeConfig(nbits=nbits, group_size=group_size)


def quantize_linear(linear: Any, nbits: int, group_size: int, **kwargs: Any) -> Any:
    """Call upstream HQQLinear; external HQQ must be installed by the operator."""
    from hqq.core.quantize import HQQLinear

    return HQQLinear(
        linear,
        quant_config=build_quantize_config(nbits, group_size),
        **kwargs,
    )
