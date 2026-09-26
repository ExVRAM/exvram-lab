"""Small, explicit contracts shared by the CLI, adapters, and result writers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .errors import ExVRAMError

SCHEMA_VERSION = 1


def _positive_number(
    mapping: Mapping[str, Any], name: str, *, integer: bool = False
) -> float | int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ExVRAMError(f"{name} must be a positive number")
    if integer and int(value) != value:
        raise ExVRAMError(f"{name} must be a positive integer")
    return int(value) if integer else float(value)


def _nonnegative_number(
    mapping: Mapping[str, Any], name: str, *, default: float = 0.0
) -> float:
    value = mapping.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ExVRAMError(f"{name} must be a non-negative number")
    return float(value)


def _optional_nonnegative(mapping: Mapping[str, Any], name: str) -> float | None:
    if name not in mapping or mapping[name] is None:
        return None
    return _nonnegative_number(mapping, name)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_b: float
    layers: int
    kv_heads: int
    head_dim: int
    weight_group_size: int = 128
    scale_dtype_bytes: int = 2
    input_embedding_params_b: float | None = None
    input_embedding_bits: float = 2.0
    lm_head_params_b: float | None = None
    lm_head_bits: float = 6.0
    small_tensors_mib: float | None = None
    codebook_mib: float | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ModelSpec":
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ExVRAMError("model.name must be a non-empty string")
        model = cls(
            name=name,
            params_b=float(_positive_number(raw, "params_b")),
            layers=int(_positive_number(raw, "layers", integer=True)),
            kv_heads=int(_positive_number(raw, "kv_heads", integer=True)),
            head_dim=int(_positive_number(raw, "head_dim", integer=True)),
            weight_group_size=int(raw.get("weight_group_size", 128)),
            scale_dtype_bytes=int(raw.get("scale_dtype_bytes", 2)),
            input_embedding_params_b=_optional_nonnegative(raw, "input_embedding_params_b"),
            input_embedding_bits=float(raw.get("input_embedding_bits", 2.0)),
            lm_head_params_b=_optional_nonnegative(raw, "lm_head_params_b"),
            lm_head_bits=float(raw.get("lm_head_bits", 6.0)),
            small_tensors_mib=_optional_nonnegative(raw, "small_tensors_mib"),
            codebook_mib=_optional_nonnegative(raw, "codebook_mib"),
        )
        embedding_params = model.input_embedding_params_b or 0.0
        lm_head_params = model.lm_head_params_b or 0.0
        if embedding_params + lm_head_params > model.params_b:
            raise ExVRAMError("input/lm_head parameter counts exceed model.params_b")
        if model.input_embedding_bits <= 0 or model.lm_head_bits <= 0:
            raise ExVRAMError("input_embedding_bits and lm_head_bits must be positive")
        return model


@dataclass(frozen=True)
class ExperimentConfig:
    id: str
    adapter: str
    model: ModelSpec
    vram_gib: float = 8.0
    context_tokens: int = 8192
    batch_size: int = 1
    weight_bits: float = 2.0
    kv_dtype_bytes: float = 2.0
    overhead_fraction: float = 0.08
    target_toks_s: float = 25.0
    target_peak_vram_gib: float = 7.5
    kv_layer_types: tuple[tuple[str, float], ...] = (("full_attention", 1.0),)
    candidate_adapters: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    seed: int = 17

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        experiment_id = raw.get("id")
        adapter = raw.get("adapter")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ExVRAMError("experiment.id must be a non-empty string")
        if not isinstance(adapter, str) or not adapter.strip():
            raise ExVRAMError("experiment.adapter must be a non-empty string")
        context_tokens = int(_positive_number(raw, "context_tokens", integer=True))
        batch_size = int(_positive_number(raw, "batch_size", integer=True))
        overhead_fraction = float(raw.get("overhead_fraction", 0.08))
        if not 0 <= overhead_fraction < 1:
            raise ExVRAMError("overhead_fraction must be in [0, 1)")
        candidates = raw.get("candidate_adapters", [])
        notes = raw.get("notes", [])
        layer_types = raw.get(
            "kv_layer_types", [{"name": "full_attention", "fraction": 1.0}]
        )
        if not isinstance(candidates, list) or not all(
            isinstance(item, str) for item in candidates
        ):
            raise ExVRAMError("candidate_adapters must be a list of strings")
        if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
            raise ExVRAMError("notes must be a list of strings")
        if not isinstance(layer_types, list) or not layer_types:
            raise ExVRAMError("kv_layer_types must be a non-empty list")
        parsed_layer_types: list[tuple[str, float]] = []
        for item in layer_types:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise ExVRAMError("each kv_layer_types entry needs a string name")
            fraction = item.get("fraction")
            if (
                isinstance(fraction, bool)
                or not isinstance(fraction, (int, float))
                or fraction <= 0
            ):
                raise ExVRAMError("each kv_layer_types fraction must be positive")
            parsed_layer_types.append((item["name"], float(fraction)))
        if abs(sum(item[1] for item in parsed_layer_types) - 1.0) > 1e-6:
            raise ExVRAMError("kv_layer_types fractions must sum to 1")
        target_peak = float(raw.get("target_peak_vram_gib", 7.5))
        if target_peak <= 0:
            raise ExVRAMError("target_peak_vram_gib must be positive")
        return cls(
            id=experiment_id,
            adapter=adapter,
            model=ModelSpec.from_mapping(raw.get("model", {})),
            vram_gib=float(_positive_number(raw, "vram_gib")),
            context_tokens=context_tokens,
            batch_size=batch_size,
            weight_bits=float(_positive_number(raw, "weight_bits")),
            kv_dtype_bytes=float(_positive_number(raw, "kv_dtype_bytes")),
            overhead_fraction=overhead_fraction,
            target_toks_s=float(_positive_number(raw, "target_toks_s")),
            target_peak_vram_gib=target_peak,
            kv_layer_types=tuple(parsed_layer_types),
            candidate_adapters=tuple(candidates),
            notes=tuple(notes),
            seed=int(raw.get("seed", 17)),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["candidate_adapters"] = list(self.candidate_adapters)
        result["notes"] = list(self.notes)
        result["kv_layer_types"] = [
            {"name": name, "fraction": fraction} for name, fraction in self.kv_layer_types
        ]
        return result


@dataclass(frozen=True)
class AdapterAvailability:
    name: str
    state: str
    detail: str
    detected_via: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdapterInfo:
    name: str
    project_url: str
    description: str
    integration_kind: str
    license: str
    copied_code: bool = False
    caution: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResultRecord:
    schema_version: int
    benchmark_id: str
    benchmark_kind: str
    synthetic: bool
    timestamp_utc: str
    config: dict[str, Any]
    hardware: dict[str, Any]
    metrics: dict[str, Any]
    quality_gate: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
