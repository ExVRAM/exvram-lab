"""Bounded configuration search and evidence-aware recommendations."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..contracts import ExperimentConfig
from ..errors import ExVRAMError
from ..hardware import HardwareInfo
from ..memory import estimate_memory


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExVRAMError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ExVRAMError(f"{name} must be a positive number")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExVRAMError(f"{name} must be a positive integer")
    return value


def _unique_strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ExVRAMError(f"{name} must be a non-empty list of strings")
    result = tuple(item.strip() for item in value)
    if any(not item for item in result) or len(set(result)) != len(result):
        raise ExVRAMError(f"{name} must contain unique non-empty strings")
    return result


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()


@dataclass(frozen=True)
class KVOption:
    name: str
    bytes_per_element: float
    backends: tuple[str, ...]
    support_status: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["backends"] = list(self.backends)
        return payload


@dataclass(frozen=True)
class ResidencyOption:
    name: str
    cpu_offload_mib: float | None
    support_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    backends: tuple[str, ...]
    options: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["backends"] = list(self.backends)
        return payload


@dataclass(frozen=True)
class SearchSpace:
    search_id: str
    model_id: str
    base_config: str
    backends: tuple[str, ...]
    quant_bits: tuple[float, ...]
    contexts: tuple[int, ...]
    kv_options: tuple[KVOption, ...]
    residency_options: tuple[ResidencyOption, ...]
    runtime_profiles: tuple[RuntimeProfile, ...]
    screen_quant_bits: tuple[float, ...]
    screen_contexts: tuple[int, ...]
    max_candidates: int
    expand_top_n: int
    target_context: int
    target_decode_tokens_per_second: float
    target_peak_vram_gib: float
    quality_preference: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SearchSpace":
        if raw.get("schema_version", 1) != 1:
            raise ExVRAMError("unsupported search schema_version")
        backends = _unique_strings(raw.get("backends"), "backends")
        quant_values = raw.get("quant_bits")
        if not isinstance(quant_values, list) or not quant_values:
            raise ExVRAMError("quant_bits must be a non-empty list")
        quant_bits = tuple(_positive_float(value, "quant_bits entry") for value in quant_values)
        if len(set(quant_bits)) != len(quant_bits):
            raise ExVRAMError("quant_bits must be unique")
        context_values = raw.get("contexts")
        if not isinstance(context_values, list) or not context_values:
            raise ExVRAMError("contexts must be a non-empty list")
        contexts = tuple(_positive_int(value, "contexts entry") for value in context_values)
        if len(set(contexts)) != len(contexts):
            raise ExVRAMError("contexts must be unique")

        kv_options: list[KVOption] = []
        for item in raw.get("kv_precisions", []):
            if not isinstance(item, dict):
                raise ExVRAMError("kv_precisions entries must be objects")
            supported = tuple(item.get("backends", backends))
            if not supported or not all(isinstance(value, str) for value in supported):
                raise ExVRAMError("kv_precisions backends must be strings")
            kv_options.append(
                KVOption(
                    _string(item.get("name"), "kv precision name"),
                    _positive_float(item.get("bytes_per_element"), "kv bytes_per_element"),
                    supported,
                    _string(item.get("support_status", "candidate"), "kv support_status"),
                )
            )
        if not kv_options:
            raise ExVRAMError("kv_precisions must contain at least one option")

        residency_options: list[ResidencyOption] = []
        for item in raw.get("residency", []):
            if not isinstance(item, dict):
                raise ExVRAMError("residency entries must be objects")
            offload = item.get("cpu_offload_mib")
            if offload is not None:
                if (
                    isinstance(offload, bool)
                    or not isinstance(offload, (int, float))
                    or offload < 0
                ):
                    raise ExVRAMError("cpu_offload_mib must be a non-negative number")
                offload = float(offload)
            residency_options.append(
                ResidencyOption(
                    _string(item.get("name"), "residency name"),
                    offload,
                    _string(item.get("support_status", "candidate"), "residency support_status"),
                )
            )
        if not residency_options:
            raise ExVRAMError("residency must contain at least one option")

        runtime_profiles: list[RuntimeProfile] = []
        for item in raw.get("runtime_profiles", []):
            if not isinstance(item, dict) or not isinstance(item.get("options"), dict):
                raise ExVRAMError("runtime_profiles entries need an options object")
            profile_backends = tuple(item.get("backends", backends))
            if not profile_backends or not all(
                isinstance(value, str) for value in profile_backends
            ):
                raise ExVRAMError("runtime profile backends must be strings")
            runtime_profiles.append(
                RuntimeProfile(
                    _string(item.get("name"), "runtime profile name"),
                    profile_backends,
                    dict(item["options"]),
                )
            )
        if not runtime_profiles:
            raise ExVRAMError("runtime_profiles must contain at least one option")

        policy = raw.get("policy", {})
        if not isinstance(policy, dict):
            raise ExVRAMError("policy must be an object")
        screen_quant = tuple(
            _positive_float(value, "screen_quant_bits entry")
            for value in policy.get("screen_quant_bits", [quant_bits[0]])
        )
        screen_contexts = tuple(
            _positive_int(value, "screen_contexts entry")
            for value in policy.get("screen_contexts", [contexts[0]])
        )
        if not set(screen_quant) <= set(quant_bits):
            raise ExVRAMError("screen_quant_bits must be included in quant_bits")
        if not set(screen_contexts) <= set(contexts):
            raise ExVRAMError("screen_contexts must be included in contexts")
        target_context = _positive_int(
            raw.get("target_context", max(contexts)), "target_context"
        )
        if target_context not in contexts:
            raise ExVRAMError("target_context must be included in contexts")
        return cls(
            _string(raw.get("search_id"), "search_id"),
            _string(raw.get("model_id"), "model_id"),
            _string(raw.get("base_config"), "base_config"),
            backends,
            quant_bits,
            contexts,
            tuple(kv_options),
            tuple(residency_options),
            tuple(runtime_profiles),
            screen_quant,
            screen_contexts,
            _positive_int(policy.get("max_candidates", 32), "max_candidates"),
            _positive_int(policy.get("expand_top_n", len(backends)), "expand_top_n"),
            target_context,
            _positive_float(raw.get("target_decode_tokens_per_second", 25), "target decode"),
            _positive_float(raw.get("target_peak_vram_gib", 7.5), "target peak VRAM"),
            _string(raw.get("quality_preference", "measured_quality_first"), "quality_preference"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "search_id": self.search_id,
            "model_id": self.model_id,
            "base_config": self.base_config,
            "backends": list(self.backends),
            "quant_bits": list(self.quant_bits),
            "contexts": list(self.contexts),
            "kv_precisions": [item.to_dict() for item in self.kv_options],
            "residency": [item.to_dict() for item in self.residency_options],
            "runtime_profiles": [item.to_dict() for item in self.runtime_profiles],
            "policy": {
                "screen_quant_bits": list(self.screen_quant_bits),
                "screen_contexts": list(self.screen_contexts),
                "max_candidates": self.max_candidates,
                "expand_top_n": self.expand_top_n,
            },
            "target_context": self.target_context,
            "target_decode_tokens_per_second": self.target_decode_tokens_per_second,
            "target_peak_vram_gib": self.target_peak_vram_gib,
            "quality_preference": self.quality_preference,
        }


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    stage: str
    backend: str
    quant_bits: float
    context_tokens: int
    kv_precision: str
    kv_dtype_bytes: float
    residency: str
    cpu_offload_mib: float | None
    runtime_profile: str
    runtime_options: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_search_space(path: str | Path) -> SearchSpace:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExVRAMError(f"cannot read search config {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExVRAMError("search config root must be an object")
    return SearchSpace.from_mapping(payload)


def load_measurements(results_dir: str | Path) -> list[dict[str, Any]]:
    directory = Path(results_dir)
    if not directory.exists():
        return []
    records: list[dict[str, Any]] = []
    for source in sorted(directory.glob("*.jsonl")):
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExVRAMError(
                    f"invalid measurement JSONL at {source}:{line_number}: {exc}"
                ) from exc
            if isinstance(payload, dict):
                payload = dict(payload)
                payload["_source"] = str(source).replace("\\", "/")
                records.append(payload)
    return records


def _supported_kv(space: SearchSpace, backend: str) -> tuple[KVOption, ...]:
    return tuple(item for item in space.kv_options if backend in item.backends)


def _supported_profiles(space: SearchSpace, backend: str) -> tuple[RuntimeProfile, ...]:
    return tuple(item for item in space.runtime_profiles if backend in item.backends)


def _candidate(
    search_id: str,
    stage: str,
    backend: str,
    quant_bits: float,
    context_tokens: int,
    kv: KVOption,
    residency: ResidencyOption,
    profile: RuntimeProfile,
) -> Candidate:
    candidate_id = "-".join(
        (
            "search",
            _slug(search_id),
            _slug(backend),
            f"w{quant_bits:g}".replace(".", "p"),
            f"c{context_tokens}",
            _slug(kv.name),
            _slug(residency.name),
            _slug(profile.name),
        )
    )
    return Candidate(
        candidate_id,
        stage,
        backend,
        quant_bits,
        context_tokens,
        kv.name,
        kv.bytes_per_element,
        residency.name,
        residency.cpu_offload_mib,
        profile.name,
        dict(profile.options),
    )


def _append_unique(items: list[Candidate], candidate: Candidate, seen: set[str]) -> None:
    if candidate.candidate_id not in seen:
        items.append(candidate)
        seen.add(candidate.candidate_id)


def _measured_backends(records: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    names = {
        str(record.get("backend"))
        for record in records
        if record.get("benchmark_kind") == "full_model_shootout"
        and record.get("status") == "PASS"
        and record.get("measurement_status") == "measured"
    }
    return tuple(sorted(names))


def generate_candidates(
    space: SearchSpace,
    *,
    stage: str = "adaptive",
    records: Iterable[Mapping[str, Any]] = (),
) -> tuple[Candidate, ...]:
    if stage not in {"screen", "deep", "adaptive"}:
        raise ExVRAMError("search stage must be screen, deep or adaptive")
    candidates: list[Candidate] = []
    seen: set[str] = set()
    first_residency = space.residency_options[0]
    for backend in space.backends:
        kv_options = _supported_kv(space, backend)
        profiles = _supported_profiles(space, backend)
        if not kv_options or not profiles:
            continue
        for quant_bits in space.screen_quant_bits:
            for context_tokens in space.screen_contexts:
                _append_unique(
                    candidates,
                    _candidate(
                        space.search_id,
                        "screen",
                        backend,
                        quant_bits,
                        context_tokens,
                        kv_options[0],
                        first_residency,
                        profiles[0],
                    ),
                    seen,
                )
    if stage == "screen":
        return tuple(candidates[: space.max_candidates])

    measured = _measured_backends(records)
    priority = [backend for backend in measured if backend in space.backends]
    priority.extend(backend for backend in space.backends if backend not in priority)
    priority = priority[: space.expand_top_n]
    deep_contexts = tuple(
        sorted(
            (context for context in space.contexts if context not in space.screen_contexts),
            key=lambda context: context != space.target_context,
        )
    )
    for backend in priority:
        kv_options = _supported_kv(space, backend)
        profiles = _supported_profiles(space, backend)
        if not kv_options or not profiles:
            continue
        for quant_bits in space.quant_bits:
            for context_tokens in deep_contexts:
                for kv in kv_options:
                    for residency in space.residency_options:
                        for profile in profiles:
                            _append_unique(
                                candidates,
                                _candidate(
                                    space.search_id,
                                    "deep",
                                    backend,
                                    quant_bits,
                                    context_tokens,
                                    kv,
                                    residency,
                                    profile,
                                ),
                                seen,
                            )
                            if len(candidates) >= space.max_candidates:
                                return tuple(candidates)
    return tuple(candidates[: space.max_candidates])


def _fit_peak_mib(record: Mapping[str, Any]) -> float | None:
    rows = record.get("metrics", {}).get("fit_memory_breakdown", {}).get("rows", [])
    values = [
        row.get("self")
        for row in rows
        if row.get("kind") == "device" and row.get("self") is not None
    ]
    return max((float(value) for value in values), default=None)


def _full_model_evidence(
    candidate: Candidate, model_id: str, records: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    record_list = list(records)
    matches = [
        record
        for record in record_list
        if record.get("benchmark_kind") == "full_model_shootout"
        and record.get("backend") == candidate.backend
        and record.get("model_id") == model_id
        and record.get("context_tokens") == candidate.context_tokens
        and record.get("search_candidate_id") == candidate.candidate_id
    ]
    passes = [
        record
        for record in matches
        if record.get("status") == "PASS" and record.get("measurement_status") == "measured"
    ]
    if passes:
        record = max(
            passes,
            key=lambda item: float(item.get("metrics", {}).get("decode_tokens_per_second", 0) or 0),
        )
        metrics = record.get("metrics", {})
        return {
            "status": "measured",
            "source": record.get("_source"),
            "decode_tokens_per_second": metrics.get("decode_tokens_per_second"),
            "prefill_tokens_per_second": metrics.get("prefill_tokens_per_second"),
            "peak_vram_mib": _fit_peak_mib(record),
            "quality_status": record.get("quality_gate", {}).get("status", "NOT_RUN"),
        }
    if matches:
        last = matches[-1]
        return {
            "status": "failed",
            "source": last.get("_source"),
            "error": last.get("error", "matching full-model run did not pass"),
            "quality_status": last.get("quality_gate", {}).get("status", "NOT_RUN"),
        }
    related = [
        record
        for record in record_list
        if record.get("benchmark_kind") == "full_model_shootout"
        and record.get("backend") == candidate.backend
        and record.get("model_id") == model_id
        and record.get("context_tokens") == candidate.context_tokens
    ]
    if related:
        return {
            "status": "unavailable",
            "source": None,
            "reason": "related result exists but candidate identity is not recorded",
            "related_sources": list(dict.fromkeys(record.get("_source") for record in related)),
            "quality_status": "NOT_RUN",
        }
    return {
        "status": "unavailable",
        "source": None,
        "reason": "no matching measured full-model result",
        "quality_status": "NOT_RUN",
    }


def _candidate_assessment(
    candidate: Candidate,
    space: SearchSpace,
    base_config: ExperimentConfig,
    hardware: HardwareInfo,
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    evidence = _full_model_evidence(candidate, space.model_id, records)
    config = replace(
        base_config,
        id=candidate.candidate_id,
        adapter=candidate.backend,
        context_tokens=candidate.context_tokens,
        weight_bits=candidate.quant_bits,
        kv_dtype_bytes=candidate.kv_dtype_bytes,
    )
    budget = estimate_memory(config).to_dict()
    if candidate.residency == "full_gpu":
        modeled_peak = budget["modeled_total_gib"]
        memory_status = budget["evidence_status"]
    else:
        modeled_peak = None
        memory_status = "unavailable"
    hardware_fit = None
    if modeled_peak is not None:
        hardware_fit = modeled_peak <= space.target_peak_vram_gib
    return {
        "candidate": candidate.to_dict(),
        "memory": {
            "status": memory_status,
            "modeled_peak_vram_gib": modeled_peak,
            "target_peak_vram_gib": space.target_peak_vram_gib,
            "fits_target_peak": hardware_fit,
            "budget_evidence": budget["evidence_status"],
        },
        "hardware": {
            "device_name": hardware.device_name,
            "compute_capability": hardware.compute_capability,
            "vram_gib": hardware.vram_gib,
        },
        "evidence": evidence,
    }


def build_search_plan(
    space: SearchSpace,
    base_config: ExperimentConfig,
    hardware: HardwareInfo,
    records: Iterable[Mapping[str, Any]],
    *,
    stage: str = "adaptive",
) -> dict[str, Any]:
    record_list = list(records)
    candidates = generate_candidates(space, stage=stage, records=record_list)
    assessments = [
        _candidate_assessment(candidate, space, base_config, hardware, record_list)
        for candidate in candidates
    ]
    return {
        "schema_version": 1,
        "benchmark_kind": "adaptive_configuration_search_plan",
        "search_id": space.search_id,
        "model_id": space.model_id,
        "stage": stage,
        "status": "PLANNED",
        "evidence_policy": (
            "Measured full-model evidence is required before a speed recommendation."
        ),
        "search_space": space.to_dict(),
        "hardware": hardware.to_dict(),
        "candidate_count": len(assessments),
        "measured_full_model_backends": list(_measured_backends(record_list)),
        "candidates": assessments,
    }


def _unmeasured_recommendation(
    mode: str, assessment: dict[str, Any] | None, reason: str
) -> dict[str, Any]:
    return {
        "mode": mode,
        "status": "MODELED_NOT_MEASURED" if assessment else "NO_MEASURED_TARGET_RESULT",
        "candidate": assessment["candidate"] if assessment else None,
        "memory": assessment["memory"] if assessment else None,
        "evidence": assessment["evidence"] if assessment else None,
        "reason": reason,
    }


def recommend_configurations(
    space: SearchSpace,
    base_config: ExperimentConfig,
    hardware: HardwareInfo,
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    record_list = list(records)
    candidates = generate_candidates(space, records=record_list)
    assessments = [
        _candidate_assessment(candidate, space, base_config, hardware, record_list)
        for candidate in candidates
        if candidate.context_tokens == space.target_context
    ]
    measured = [
        item
        for item in assessments
        if item["evidence"]["status"] == "measured"
        and item["evidence"].get("decode_tokens_per_second") is not None
    ]
    quality_pass = [
        item
        for item in measured
        if str(item["evidence"].get("quality_status", "")).lower() == "pass"
    ]
    full_gpu = [item for item in assessments if item["candidate"]["residency"] == "full_gpu"]
    fastest = max(
        measured,
        key=lambda item: item["evidence"]["decode_tokens_per_second"],
        default=None,
    )
    best_quality = max(
        quality_pass,
        key=lambda item: item["evidence"]["decode_tokens_per_second"],
        default=None,
    )
    lowest_vram = min(
        (item for item in full_gpu if item["memory"]["modeled_peak_vram_gib"] is not None),
        key=lambda item: item["memory"]["modeled_peak_vram_gib"],
        default=None,
    )
    balanced_pool = [item for item in full_gpu if item["memory"]["fits_target_peak"] is True]
    balanced = max(balanced_pool, key=lambda item: item["candidate"]["quant_bits"], default=None)
    return {
        "schema_version": 1,
        "benchmark_kind": "hardware_aware_configuration_recommendation",
        "search_id": space.search_id,
        "model_id": space.model_id,
        "hardware": hardware.to_dict(),
        "target": {
            "context_tokens": space.target_context,
            "decode_tokens_per_second": space.target_decode_tokens_per_second,
            "peak_vram_gib": space.target_peak_vram_gib,
            "quality_preference": space.quality_preference,
        },
        "recommendations": {
            "FASTEST": (
                {
                    "mode": "FASTEST",
                    "status": "MEASURED",
                    "candidate": fastest["candidate"],
                    "memory": fastest["memory"],
                    "evidence": fastest["evidence"],
                    "reason": "Highest measured target-context decode rate.",
                }
                if fastest
                else _unmeasured_recommendation(
                    "FASTEST", None, "No measured full-model target-context decode result exists."
                )
            ),
            "BEST_QUALITY": (
                {
                    "mode": "BEST_QUALITY",
                    "status": "MEASURED",
                    "candidate": best_quality["candidate"],
                    "memory": best_quality["memory"],
                    "evidence": best_quality["evidence"],
                    "reason": "Highest measured target-context result with a passing quality gate.",
                }
                if best_quality
                else _unmeasured_recommendation(
                    "BEST_QUALITY",
                    max(full_gpu, key=lambda item: item["candidate"]["quant_bits"], default=None),
                    (
                        "No target-context quality gate has passed; the fallback is a modeled "
                        "high-bit candidate."
                    ),
                )
            ),
            "LOWEST_VRAM": _unmeasured_recommendation(
                "LOWEST_VRAM",
                lowest_vram,
                (
                    "Lowest modeled full-GPU target-context budget; residency and peak are not "
                    "measured."
                ),
            ),
            "BALANCED": _unmeasured_recommendation(
                "BALANCED",
                balanced or lowest_vram,
                (
                    "No measured target-context Pareto point exists; fallback uses the modeled "
                    "budget only."
                ),
            ),
        },
        "candidates_considered": len(assessments),
        "measured_target_context_candidates": len(measured),
        "quality_passing_target_context_candidates": len(quality_pass),
        "evidence_policy": (
            "Null speed is intentional when no matching full-model measurement exists."
        ),
    }


def write_json_payload(payload: Mapping[str, Any], path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(output)
