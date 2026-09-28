"""P8 binary-representation planning and physical-footprint accounting."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

P8_SCHEMA_VERSION = 1
P8_GROUP_SIZES = (32, 64, 128)
P8_QUALITY_ONLY_GROUP_SIZES = (8, 16)
P8_BATCH_SIZES = (1, 8, 32, 128)


@dataclass(frozen=True)
class P8MatrixShape:
    name: str
    in_features: int
    out_features: int
    layer_kind: str
    source: str = "Qwen/Qwen3.8-27B config"

    @property
    def parameter_count(self) -> int:
        return self.in_features * self.out_features

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "P8MatrixShape":
        name = raw.get("name")
        layer_kind = raw.get("layer_kind")
        in_features = raw.get("in_features")
        out_features = raw.get("out_features")
        if not isinstance(name, str) or not name:
            raise ValueError("P8 shape name must be a non-empty string")
        if not isinstance(layer_kind, str) or not layer_kind:
            raise ValueError(f"P8 shape {name} needs layer_kind")
        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError(f"P8 shape {name} needs positive in_features")
        if not isinstance(out_features, int) or out_features <= 0:
            raise ValueError(f"P8 shape {name} needs positive out_features")
        return cls(name, in_features, out_features, layer_kind, raw.get("source", cls.source))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "parameter_count": self.parameter_count}


@dataclass(frozen=True)
class PhysicalFootprint:
    logical_weight_bytes: int
    binary_weight_bytes: int
    scale_bytes: int
    zero_bytes: int
    exception_bytes: int
    metadata_bytes: int
    packing_padding_bytes: int
    alignment_padding_bytes: int
    persistent_buffer_bytes: int
    total_persistent_bytes: int
    physical_bpw: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _align(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _footprint(
    *,
    parameter_count: int,
    padded_elements: int,
    bits: float,
    scale_count: int = 0,
    scale_dtype_bytes: int = 0,
    zero_count: int = 0,
    zero_dtype_bytes: int = 0,
    exception_bytes: int = 0,
    metadata_bytes: int = 0,
    persistent_buffer_bytes: int = 0,
    alignment_bytes: int = 32,
) -> PhysicalFootprint:
    if parameter_count <= 0 or padded_elements < parameter_count:
        raise ValueError("padded_elements must cover a positive parameter_count")
    if bits <= 0:
        raise ValueError("bits must be positive")
    logical_weight_bytes = math.ceil(parameter_count * bits / 8)
    binary_weight_bytes = math.ceil(padded_elements * bits / 8)
    packing_padding_bytes = binary_weight_bytes - logical_weight_bytes
    scale_bytes = scale_count * scale_dtype_bytes
    zero_bytes = zero_count * zero_dtype_bytes
    raw_components = (
        binary_weight_bytes,
        scale_bytes,
        zero_bytes,
        exception_bytes,
        metadata_bytes,
    )
    aligned_components = tuple(_align(value, alignment_bytes) for value in raw_components)
    alignment_padding_bytes = sum(aligned_components) - sum(raw_components)
    total = sum(aligned_components) + persistent_buffer_bytes
    return PhysicalFootprint(
        logical_weight_bytes=logical_weight_bytes,
        binary_weight_bytes=binary_weight_bytes,
        scale_bytes=scale_bytes,
        zero_bytes=zero_bytes,
        exception_bytes=exception_bytes,
        metadata_bytes=metadata_bytes,
        packing_padding_bytes=packing_padding_bytes,
        alignment_padding_bytes=alignment_padding_bytes,
        persistent_buffer_bytes=persistent_buffer_bytes,
        total_persistent_bytes=total,
        physical_bpw=total * 8 / parameter_count,
    )


def grouped_footprint(
    shape: P8MatrixShape,
    bits: int,
    group_size: int,
    *,
    scale_dtype_bytes: int = 2,
    zero_dtype_bytes: int = 2,
    metadata_bytes_per_group: int = 0,
    alignment_bytes: int = 32,
    persistent_buffer_bytes: int = 0,
) -> PhysicalFootprint:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    groups_per_row = _ceil_div(shape.in_features, group_size)
    groups = shape.out_features * groups_per_row
    padded_elements = shape.out_features * groups_per_row * group_size
    return _footprint(
        parameter_count=shape.parameter_count,
        padded_elements=padded_elements,
        bits=bits,
        scale_count=groups,
        scale_dtype_bytes=scale_dtype_bytes,
        zero_count=groups,
        zero_dtype_bytes=zero_dtype_bytes,
        metadata_bytes=groups * metadata_bytes_per_group,
        persistent_buffer_bytes=persistent_buffer_bytes,
        alignment_bytes=alignment_bytes,
    )


def fixed_precision_footprint(
    shape: P8MatrixShape, dtype_bytes: int, *, alignment_bytes: int = 32
) -> PhysicalFootprint:
    return _footprint(
        parameter_count=shape.parameter_count,
        padded_elements=shape.parameter_count,
        bits=dtype_bytes * 8,
        alignment_bytes=alignment_bytes,
    )


def pbllm_footprint(
    shape: P8MatrixShape,
    salient_fraction: float,
    high_bits: int,
    *,
    group_size: int = 64,
    scale_dtype_bytes: int = 2,
    metadata_index_bytes: int = 0,
    alignment_bytes: int = 32,
) -> PhysicalFootprint:
    if not 0 < salient_fraction < 1:
        raise ValueError("salient_fraction must be between zero and one")
    if high_bits <= 1:
        raise ValueError("high_bits must exceed the binary base")
    groups = shape.out_features * _ceil_div(shape.in_features, group_size)
    exceptions = math.ceil(shape.parameter_count * salient_fraction)
    bitmap_bytes = math.ceil(shape.parameter_count / 8)
    index_bytes = exceptions * metadata_index_bytes
    return _footprint(
        parameter_count=shape.parameter_count,
        padded_elements=shape.parameter_count,
        bits=1,
        scale_count=groups,
        scale_dtype_bytes=scale_dtype_bytes,
        exception_bytes=math.ceil(exceptions * high_bits / 8),
        metadata_bytes=bitmap_bytes + index_bytes,
        alignment_bytes=alignment_bytes,
    )


def parameter_count_footprint(
    parameter_count: int,
    bits: float,
    group_size: int | None = None,
    *,
    scale_dtype_bytes: int = 2,
    zero_dtype_bytes: int = 2,
    alignment_bytes: int = 32,
) -> PhysicalFootprint:
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    groups = _ceil_div(parameter_count, group_size) if group_size else 0
    return _footprint(
        parameter_count=parameter_count,
        padded_elements=parameter_count,
        bits=bits,
        scale_count=groups,
        scale_dtype_bytes=scale_dtype_bytes if group_size else 0,
        zero_count=groups,
        zero_dtype_bytes=zero_dtype_bytes if group_size else 0,
        alignment_bytes=alignment_bytes,
    )


def quality_metrics(reference: Sequence[float], reconstructed: Sequence[float]) -> dict[str, float]:
    if len(reference) != len(reconstructed) or not reference:
        raise ValueError("reference and reconstructed must have equal non-zero length")
    errors = [float(actual) - float(expected) for expected, actual in zip(reference, reconstructed)]
    mse = sum(error * error for error in errors) / len(errors)
    reference_energy = sum(float(value) * float(value) for value in reference) / len(reference)
    dot = sum(float(expected) * float(actual) for expected, actual in zip(reference, reconstructed))
    ref_norm = math.sqrt(sum(float(value) ** 2 for value in reference))
    actual_norm = math.sqrt(sum(float(value) ** 2 for value in reconstructed))
    cosine = dot / (ref_norm * actual_norm) if ref_norm and actual_norm else 0.0
    max_abs = max(abs(error) for error in errors)
    max_ref = max(max(abs(float(value)) for value in reference), 1e-12)
    return {
        "mse": mse,
        "relative_mse": mse / max(reference_energy, 1e-12),
        "cosine_similarity": cosine,
        "max_absolute_error": max_abs,
        "max_relative_error": max_abs / max_ref,
    }


def representative_tensor(size: int = 256, seed: int = 17) -> tuple[float, ...]:
    """Return deterministic values for harness tests; this is not a quantizer or model data."""
    if size <= 0:
        raise ValueError("size must be positive")
    return tuple(
        math.sin((index + seed) * 0.17) * 0.8
        + math.cos((index + seed) * 0.037) * 0.2
        for index in range(size)
    )


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_status() -> dict[str, dict[str, Any]]:
    result = {}
    for name, package in (
        ("hqq", "hqq"),
        ("gemlite", "gemlite"),
        ("pbllm", "pbllm"),
        ("billm", "billm"),
    ):
        result[name] = {
            "available": importlib.util.find_spec(package) is not None,
            "package": package,
            "version": _version(package),
            "execution": "external_optional" if name in {"hqq", "gemlite"} else "reference_only",
        }
    return result


def _record(
    shape: P8MatrixShape,
    method: str,
    bits: float,
    group_size: int | None,
    footprint: PhysicalFootprint,
    *,
    status: str,
    notes: list[str],
    dependency: str | None = None,
) -> dict[str, Any]:
    return {
        "method": method,
        "bits": bits,
        "group_size": group_size,
        "matrix": shape.to_dict(),
        "footprint": footprint.to_dict(),
        "measurement_status": status,
        "dependency": dependency,
        "quality": {
            "status": "NOT_RUN",
            "metrics": None,
            "note": "No reconstructed external tensor was supplied.",
        },
        "performance": {
            "status": "NOT_RUN",
            "batch_sizes": list(P8_BATCH_SIZES),
            "latency_ms_median": None,
            "latency_ms_p95": None,
            "effective_gib_per_s": None,
            "workspace_bytes": None,
            "vram_bytes": None,
            "gpu_utilization_percent": None,
        },
        "notes": notes,
    }


def _shapes(manifest: dict[str, Any]) -> tuple[P8MatrixShape, ...]:
    raw_shapes = manifest.get("shapes")
    if not isinstance(raw_shapes, list) or not raw_shapes:
        raise ValueError("P8 manifest needs a non-empty shapes list")
    return tuple(P8MatrixShape.from_mapping(item) for item in raw_shapes)


def _model_estimates(manifest: dict[str, Any]) -> dict[str, Any]:
    model = manifest.get("model", {})
    params_b = float(model["parameters_b"])
    parameter_count = round(params_b * 1_000_000_000)
    estimates: dict[str, Any] = {}
    for bits, group in ((1, 32), (1, 64), (1, 128), (2, 64)):
        footprint = parameter_count_footprint(parameter_count, bits, group)
        estimates[f"hqq_{bits}bit_g{group}"] = footprint.to_dict()
    for salient_fraction in (0.10, 0.05, 0.025):
        binary_fraction = 1.0 - salient_fraction
        estimates[f"pbllm_{binary_fraction:.3f}_binary_high8_g64"] = pbllm_footprint(
            P8MatrixShape("model", parameter_count, 1, "model_estimate"),
            salient_fraction,
            8,
            group_size=64,
        ).to_dict()
    reference = manifest.get("references", {}).get("iq2_xxs", {})
    artifact_bytes = reference.get("artifact_bytes")
    artifact_bpw = None
    if isinstance(artifact_bytes, int) and artifact_bytes > 0:
        artifact_bpw = artifact_bytes * 8 / parameter_count
    estimates["iq2_xxs_reference"] = {
        "nominal_bpw": reference.get("nominal_bpw"),
        "artifact_effective_bpw": artifact_bpw,
        "artifact_bytes": artifact_bytes,
        "status": "reference_estimate",
    }
    return estimates


def build_p8_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    shapes = _shapes(manifest)
    dependencies = dependency_status()
    records: list[dict[str, Any]] = []
    for shape in shapes:
        records.append(
            _record(
                shape,
                "fp16",
                16,
                None,
                fixed_precision_footprint(shape, 2),
                status="reference_estimate",
                notes=["Same matrix shape; no external quantizer or kernel was used."],
            )
        )
        records.append(
            _record(
                shape,
                "bf16",
                16,
                None,
                fixed_precision_footprint(shape, 2),
                status="reference_estimate",
                notes=["Same storage width as FP16; compute dtype remains a separate measurement."],
            )
        )
        for bits in (1, 2):
            for group_size in (*P8_QUALITY_ONLY_GROUP_SIZES, *P8_GROUP_SIZES):
                note = (
                    "Quality/memory-only group; not a GemLite fast-path candidate."
                    if group_size in P8_QUALITY_ONLY_GROUP_SIZES
                    else "Requires HQQ external quantization and reconstruction measurement."
                )
                records.append(
                    _record(
                        shape,
                        f"hqq-{bits}bit",
                        bits,
                        group_size,
                        grouped_footprint(shape, bits, group_size),
                        status="planned" if dependencies["hqq"]["available"] else "unavailable",
                        dependency="hqq",
                        notes=[
                            note,
                            "Physical bpw is a layout estimate until HQQ serialization "
                            "is captured.",
                        ],
                    )
                )
        for bits in (1, 2):
            for group_size in P8_GROUP_SIZES:
                records.append(
                    _record(
                        shape,
                        f"gemlite-{bits}bit",
                        bits,
                        group_size,
                        grouped_footprint(shape, bits, group_size),
                        status="planned" if dependencies["gemlite"]["available"] else "unavailable",
                        dependency="gemlite",
                        notes=[
                            "M=1,8,32,128 are required performance points.",
                            "No CUDA timing was run by the plan command.",
                        ],
                    )
                )
        for salient_fraction in (0.10, 0.05, 0.025):
            binary_fraction = 1.0 - salient_fraction
            records.append(
                _record(
                    shape,
                    "pbllm",
                    1,
                    64,
                    pbllm_footprint(shape, salient_fraction, 8),
                    status="reference_only",
                    dependency="pbllm",
                    notes=[
                        f"binary_fraction={binary_fraction:.3f}; "
                        f"salient_fraction={salient_fraction:.3f}; bitmap and high-bit "
                        "exceptions are included.",
                        "PB-LLM code is not copied or executed by the plan command.",
                    ],
                )
            )
        records.append(
            _record(
                shape,
                "billm-paper-reference",
                1.08,
                None,
                _footprint(
                    parameter_count=shape.parameter_count,
                    padded_elements=shape.parameter_count,
                    bits=1.08,
                ),
                status="reference_only",
                dependency="billm",
                notes=[
                    "The paper-reported 1.08 average bpw is not a Qwen3.8 serialized footprint.",
                    "No BiLLM layer reconstruction was run.",
                ],
            )
        )
        iq2_bpw = float(manifest["references"]["iq2_xxs"]["nominal_bpw"])
        records.append(
            _record(
                shape,
                "iq2-xxs-reference",
                iq2_bpw,
                None,
                _footprint(
                    parameter_count=shape.parameter_count,
                    padded_elements=shape.parameter_count,
                    bits=iq2_bpw,
                ),
                status="reference_estimate",
                notes=["Nominal GGML reference; not a layer serialization dump."],
            )
        )
    return {
        "schema_version": P8_SCHEMA_VERSION,
        "benchmark_kind": "p8_binary_layer_plan",
        "track": "P8",
        "synthetic": True,
        "measurement_status": "PLANNED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_id": manifest.get("id", "p8-binary-unknown"),
        "dependencies": dependencies,
        "model": manifest.get("model", {}),
        "shapes": [shape.to_dict() for shape in shapes],
        "model_estimates": _model_estimates(manifest),
        "records": records,
        "decision_gate": {
            "status": "NOT_RUN",
            "result": None,
            "reason": "No external HQQ/GemLite reconstruction or CUDA timing was performed.",
        },
        "notes": [
            "This is a planning/footprint record, not a model-quality or tokens/second claim.",
            "External source is never vendored; use the exact revisions in the manifest.",
            "No custom quantizer, CUDA kernel, Triton kernel, or full-model quantization was run.",
        ],
    }


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("P8 manifest root must be an object")
    return payload
