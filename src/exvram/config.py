from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import SCHEMA_VERSION, ExperimentConfig
from .errors import ExVRAMError


def default_experiment() -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "id": "default-27b-8gb-2bit",
            "adapter": "exllamav3",
            "candidate_adapters": ["gemlite", "cutlass", "bitnet", "llamacpp"],
            "model": {
                "name": "dense-27b-placeholder",
                "params_b": 27,
                "layers": 32,
                "kv_heads": 8,
                "head_dim": 128,
            },
            "vram_gib": 8,
            "context_tokens": 8192,
            "batch_size": 1,
            "weight_bits": 2,
            "kv_dtype_bytes": 2,
            "overhead_fraction": 0.08,
            "target_toks_s": 25,
            "notes": ["Planning target only; no hardware result is implied."],
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExVRAMError(f"cannot read JSON config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExVRAMError("config root must be an object")
    version = payload.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ExVRAMError(f"unsupported config schema_version: {version}")
    return payload


def load_experiment(path: str | None = None, index: int = 0) -> ExperimentConfig:
    if path is None:
        return default_experiment()
    payload = _load_json(Path(path))
    experiments = payload.get("experiments")
    if experiments is not None:
        if not isinstance(experiments, list) or not experiments:
            raise ExVRAMError("config.experiments must be a non-empty list")
        if index < 0 or index >= len(experiments):
            raise ExVRAMError(f"experiment index {index} is outside 0..{len(experiments) - 1}")
        raw = experiments[index]
    else:
        raw = payload
    if not isinstance(raw, dict):
        raise ExVRAMError("experiment entry must be an object")
    return ExperimentConfig.from_mapping(raw)
