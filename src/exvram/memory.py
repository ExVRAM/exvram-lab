"""Transparent component-level VRAM accounting and existing-upstream option planning."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

from .contracts import ExperimentConfig
from .errors import ExVRAMError

GIB = 1024**3
MIB = 1024**2

# These shares reproduce the legacy 8% reserve. They are planning assumptions, not telemetry.
EVIDENCE_STATUSES = ("measured", "synthetic", "modeled", "unavailable", "INCONCLUSIVE")
OVERHEAD_SHARES = {
    "cuda_context": 0.1875,
    "allocator_reserve": 0.3125,
    "runtime_scratch_workspace": 0.15625,
    "attention_workspace": 0.125,
    "kernel_specific_workspace": 0.125,
    "graph_capture_overhead": 0.0,
    "safety_reserve": 0.09375,
}


def _gib(byte_count: float) -> float:
    return byte_count / GIB


def _optional_parameters(value: float | None) -> float:
    if value is None:
        return 0.0
    return value * 1_000_000_000


def _small_tensor_parameters(value: float | None) -> float:
    if value is None:
        return 0.0
    return value * MIB / 2


def _supplied_evidence(supplied: bool) -> str:
    return "modeled" if supplied else "unavailable"


def _budget_evidence(components: tuple[MemoryComponent, ...] | list[MemoryComponent]) -> str:
    statuses = {component.evidence_status for component in components}
    if statuses <= {"modeled"}:
        return "modeled"
    return "INCONCLUSIVE"


@dataclass(frozen=True)
class MemoryComponent:
    name: str
    category: str
    theoretical_bytes: float
    modeled_bytes: float
    precision: str
    bpw: float | None
    residency: str
    source: str
    movable_or_compressible: bool
    estimated_saving_gib: float
    notes: str
    evidence_status: str = "modeled"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["theoretical_gib"] = _gib(self.theoretical_bytes)
        result["modeled_gib"] = _gib(self.modeled_bytes)
        return result


@dataclass(frozen=True)
class MemoryOptimization:
    id: str
    category: str
    description: str
    vram_saving_gib: float
    new_total_gib: float
    reaches_target_peak: bool
    quality_risk: str
    speed_risk: str
    upstream_available: str
    upstream_projects: tuple[str, ...]
    source: str
    notes: str
    evidence_status: str = "modeled"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["upstream_projects"] = list(self.upstream_projects)
        return result


@dataclass(frozen=True)
class MemoryBudget:
    vram_gib: float
    target_peak_vram_gib: float
    raw_packed_weights_gib: float
    scale_metadata_gib: float
    estimated_weight_storage_gib: float
    kv_cache_gib: float
    runtime_overhead_gib: float
    theoretical_total_gib: float
    modeled_total_gib: float
    headroom_gib: float
    target_headroom_gib: float
    fits_modeled_budget: bool
    fits_target_peak_budget: bool
    legacy_modeled_total_gib: float
    assumptions: tuple[str, ...]
    components: tuple[MemoryComponent, ...]
    evidence_status: str = "modeled"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["assumptions"] = list(self.assumptions)
        result["components"] = [component.to_dict() for component in self.components]
        result["ranked_consumers"] = [
            component.to_dict()
            for component in sorted(
                self.components, key=lambda component: component.modeled_bytes, reverse=True
            )
        ]
        counts: dict[str, int] = {}
        for component in self.components:
            counts[component.evidence_status] = counts.get(component.evidence_status, 0) + 1
        result["evidence_status_counts"] = counts
        return result


def _quantized_bytes(
    parameters: float, bits: float, group_size: int, scale_dtype_bytes: int
) -> tuple[float, float]:
    raw = parameters * bits / 8
    if bits >= 16:
        return raw, 0.0
    groups = math.ceil(parameters / group_size)
    return raw, groups * scale_dtype_bytes


def _component(
    name: str,
    category: str,
    byte_count: float,
    precision: str,
    bpw: float | None,
    residency: str,
    source: str,
    movable_or_compressible: bool,
    notes: str,
    evidence_status: str = "modeled",
) -> MemoryComponent:
    if evidence_status not in EVIDENCE_STATUSES:
        raise ExVRAMError(f"unknown evidence_status {evidence_status}")
    return MemoryComponent(
        name=name,
        category=category,
        theoretical_bytes=byte_count,
        modeled_bytes=byte_count,
        precision=precision,
        bpw=bpw,
        residency=residency,
        source=source,
        movable_or_compressible=movable_or_compressible,
        estimated_saving_gib=0.0,
        notes=notes,
        evidence_status=evidence_status,
    )


def _weight_components(config: ExperimentConfig) -> tuple[list[MemoryComponent], float, float]:
    model = config.model
    total_parameters = model.params_b * 1_000_000_000
    embedding_parameters = _optional_parameters(model.input_embedding_params_b)
    lm_head_parameters = _optional_parameters(model.lm_head_params_b)
    small_parameters = _small_tensor_parameters(model.small_tensors_mib)
    transformer_parameters = (
        total_parameters - embedding_parameters - lm_head_parameters - small_parameters
    )
    if transformer_parameters < 0:
        raise ExVRAMError("component parameter counts exceed model.params_b")

    transformer_raw, transformer_scales = _quantized_bytes(
        transformer_parameters, config.weight_bits, model.weight_group_size, model.scale_dtype_bytes
    )
    embedding_raw, embedding_scales = _quantized_bytes(
        embedding_parameters,
        model.input_embedding_bits,
        model.weight_group_size,
        model.scale_dtype_bytes,
    )
    lm_head_raw, lm_head_scales = _quantized_bytes(
        lm_head_parameters, model.lm_head_bits, model.weight_group_size, model.scale_dtype_bytes
    )
    total_raw = transformer_raw + embedding_raw + lm_head_raw
    total_scales = transformer_scales + embedding_scales + lm_head_scales
    source = "model manifest when component parameter counts are supplied; zero otherwise"
    components = [
        _component(
            "packed_transformer_weights",
            "weights",
            transformer_raw,
            f"{config.weight_bits:g}-bit packed",
            config.weight_bits,
            "VRAM",
            source,
            True,
            "Main transformer linear weights only; excludes scales and runtime padding.",
            _supplied_evidence(transformer_parameters > 0),
        ),
        _component(
            "input_embeddings",
            "weights",
            embedding_raw,
            f"{model.input_embedding_bits:g}-bit packed",
            model.input_embedding_bits,
            "VRAM",
            source,
            True,
            "Can be offloaded only if the selected runtime supports embedding residency changes.",
            _supplied_evidence(model.input_embedding_params_b is not None),
        ),
        _component(
            "lm_head",
            "weights",
            lm_head_raw,
            f"{model.lm_head_bits:g}-bit packed",
            model.lm_head_bits,
            "VRAM",
            source,
            True,
            "Output projection; tied/untied status must come from a concrete checkpoint.",
            _supplied_evidence(model.lm_head_params_b is not None),
        ),
        _component(
            "norms_biases_small_fp16",
            "small_tensors",
            (model.small_tensors_mib or 0.0) * MIB,
            "FP16/FP32 small tensors",
            16.0,
            "VRAM",
            "model manifest; omitted means the checkpoint size was not supplied",
            True,
            "Needs checkpoint metadata for a non-zero estimate.",
            _supplied_evidence(model.small_tensors_mib is not None),
        ),
        _component(
            "quantization_scales",
            "quantization_metadata",
            total_scales,
            f"{model.scale_dtype_bytes * 8:g}-bit scales",
            model.scale_dtype_bytes * 8,
            "VRAM",
            "one scale per configured weight group",
            True,
            "Does not include runtime-specific padding or codebook indirection.",
        ),
        _component(
            "codebooks_metadata",
            "quantization_metadata",
            (model.codebook_mib or 0.0) * MIB,
            "runtime metadata",
            None,
            "VRAM",
            "model manifest; default zero for placeholder model",
            True,
            "Runtime-specific codebooks and headers must be measured from artifacts.",
            _supplied_evidence(model.codebook_mib is not None),
        ),
    ]
    return components, total_raw, total_scales


def _kv_components(config: ExperimentConfig) -> tuple[list[MemoryComponent], float]:
    model = config.model
    total_bytes = (
        2
        * model.layers
        * config.context_tokens
        * model.kv_heads
        * model.head_dim
        * config.kv_dtype_bytes
        * config.batch_size
    )
    components: list[MemoryComponent] = []
    counted_bytes = 0.0
    for name, fraction in config.kv_layer_types:
        if name == "full_attention":
            components.append(
                _component(
                    f"kv_cache_{name}",
                    "kv_cache",
                    total_bytes * fraction,
                    f"{config.kv_dtype_bytes * 8:g}-bit KV",
                    config.kv_dtype_bytes * 8,
                    "VRAM",
                    "analytic KV formula and configured layer-type fraction",
                    True,
                    "Quantization can change quality and attention kernel behavior.",
                )
            )
            counted_bytes += total_bytes * fraction
            continue
        components.append(
            _component(
                f"kv_cache_{name}",
                "kv_cache",
                0.0,
                "unspecified",
                None,
                "VRAM",
                "no byte formula for this layer type",
                True,
                "Hybrid or recurrent state is not estimated from the full-attention formula.",
                "unavailable",
            )
        )
    return components, counted_bytes


def _runtime_components(config: ExperimentConfig) -> list[MemoryComponent]:
    overhead_bytes = config.vram_gib * config.overhead_fraction * GIB
    labels = {
        "cuda_context": "CUDA context and driver allocations",
        "allocator_reserve": "allocator reserve / fragmentation",
        "runtime_scratch_workspace": "runtime scratch/workspace",
        "attention_workspace": "attention workspace",
        "kernel_specific_workspace": "kernel-specific workspace",
        "graph_capture_overhead": "graph capture overhead (if enabled)",
        "safety_reserve": "safety reserve",
    }
    components = []
    for name, fraction in OVERHEAD_SHARES.items():
        components.append(
            _component(
                name,
                "runtime_reserve",
                overhead_bytes * fraction,
                "runtime allocation / reserve",
                None,
                "VRAM" if name != "safety_reserve" else "optional",
                "configured overhead_fraction split; not hardware telemetry",
                name not in {"cuda_context", "safety_reserve"},
                labels[name],
            )
        )
    return components


def estimate_memory(config: ExperimentConfig) -> MemoryBudget:
    weight_components, total_raw, total_scales = _weight_components(config)
    kv_components, total_kv_bytes = _kv_components(config)
    components = [*weight_components, *kv_components, *_runtime_components(config)]
    annotated_components = []
    for component in components:
        saving = 0.0
        component_gib = _gib(component.modeled_bytes)
        if component.name == "input_embeddings":
            saving = component_gib
        elif component.name == "lm_head":
            saving = _lm_head_saving(config, 2.25)
        elif component.name.startswith("kv_cache_"):
            saving = component_gib * 0.75
        elif component.name == "quantization_scales":
            saving = component_gib * 0.25
        elif component.name == "codebooks_metadata":
            saving = component_gib * 0.25
        elif component.name == "runtime_scratch_workspace":
            saving = component_gib
        annotated_components.append(replace(component, estimated_saving_gib=saving))
    components = annotated_components
    theoretical_total = sum(component.theoretical_bytes for component in components)
    modeled_total = sum(component.modeled_bytes for component in components)
    overhead = sum(
        component.modeled_bytes
        for component in components
        if component.category == "runtime_reserve"
    )
    legacy_parameters = config.model.params_b * 1_000_000_000
    legacy_raw = legacy_parameters * config.weight_bits / 8
    legacy_scales = math.ceil(legacy_parameters / config.model.weight_group_size) * (
        config.model.scale_dtype_bytes
    )
    legacy_modeled_total = _gib(legacy_raw + legacy_scales + total_kv_bytes + overhead)
    assumptions = (
        "raw packed weights exclude runtime-specific headers, padding, and kernel workspaces",
        "scale metadata is approximated as one scale per configured weight group",
        "KV estimate uses 2 x layers x tokens x KV heads x head dimension x element bytes",
        "runtime reserve is a planning split of overhead_fraction, not a measurement",
        "small tensors are removed from packed weights at 2 bytes per element",
        "evidence_status modeled is arithmetic; unavailable means a size was not supplied",
        "a budget with any unavailable component is INCONCLUSIVE, not a measured fit",
    )
    evidence_status = _budget_evidence(components)
    return MemoryBudget(
        vram_gib=config.vram_gib,
        target_peak_vram_gib=config.target_peak_vram_gib,
        raw_packed_weights_gib=_gib(total_raw),
        scale_metadata_gib=_gib(total_scales),
        estimated_weight_storage_gib=_gib(total_raw + total_scales),
        kv_cache_gib=_gib(total_kv_bytes),
        runtime_overhead_gib=_gib(overhead),
        theoretical_total_gib=_gib(theoretical_total),
        modeled_total_gib=_gib(modeled_total),
        headroom_gib=config.vram_gib - _gib(modeled_total),
        target_headroom_gib=config.target_peak_vram_gib - _gib(modeled_total),
        fits_modeled_budget=_gib(modeled_total) <= config.vram_gib,
        fits_target_peak_budget=_gib(modeled_total) <= config.target_peak_vram_gib,
        legacy_modeled_total_gib=legacy_modeled_total,
        assumptions=assumptions,
        components=tuple(components),
        evidence_status=evidence_status,
    )


def _component_gib(budget: MemoryBudget, name: str) -> float:
    return next(
        (
            _gib(component.modeled_bytes)
            for component in budget.components
            if component.name == name
        ),
        0.0,
    )


def _make_option(
    config: ExperimentConfig,
    budget: MemoryBudget,
    *,
    option_id: str,
    category: str,
    description: str,
    saving_gib: float,
    quality_risk: str,
    speed_risk: str,
    upstream_available: str,
    upstream_projects: tuple[str, ...],
    source: str,
    notes: str,
) -> MemoryOptimization:
    saving_gib = max(0.0, saving_gib)
    new_total = max(0.0, budget.modeled_total_gib - saving_gib)
    return MemoryOptimization(
        id=option_id,
        category=category,
        description=description,
        vram_saving_gib=saving_gib,
        new_total_gib=new_total,
        reaches_target_peak=new_total <= config.target_peak_vram_gib,
        quality_risk=quality_risk,
        speed_risk=speed_risk,
        upstream_available=upstream_available,
        upstream_projects=upstream_projects,
        source=source,
        notes=notes,
        evidence_status=budget.evidence_status,
    )


def _lm_head_saving(config: ExperimentConfig, target_bits: float) -> float:
    model = config.model
    if (
        model.lm_head_params_b is None
        or model.lm_head_params_b <= 0
        or target_bits >= model.lm_head_bits
    ):
        return 0.0
    current_raw, current_scales = _quantized_bytes(
        model.lm_head_params_b * 1_000_000_000,
        model.lm_head_bits,
        model.weight_group_size,
        model.scale_dtype_bytes,
    )
    target_raw, target_scales = _quantized_bytes(
        model.lm_head_params_b * 1_000_000_000,
        target_bits,
        model.weight_group_size,
        model.scale_dtype_bytes,
    )
    return _gib(current_raw + current_scales - target_raw - target_scales)


def _embedding_quantization_saving(config: ExperimentConfig, target_bits: float) -> float:
    model = config.model
    if (
        model.input_embedding_params_b is None
        or model.input_embedding_params_b <= 0
        or target_bits >= model.input_embedding_bits
    ):
        return 0.0
    current_raw, current_scales = _quantized_bytes(
        model.input_embedding_params_b * 1_000_000_000,
        model.input_embedding_bits,
        model.weight_group_size,
        model.scale_dtype_bytes,
    )
    target_raw, target_scales = _quantized_bytes(
        model.input_embedding_params_b * 1_000_000_000,
        target_bits,
        model.weight_group_size,
        model.scale_dtype_bytes,
    )
    return _gib(current_raw + current_scales - target_raw - target_scales)


def suggest_optimizations(
    config: ExperimentConfig, budget: MemoryBudget
) -> list[MemoryOptimization]:
    """Generate conservative options; no option is treated as a measured result."""
    options = [
        _make_option(
            config,
            budget,
            option_id="embeddings-to-pinned-ram",
            category="residency",
            description="Move input embeddings out of VRAM when the runtime can page them safely.",
            saving_gib=_component_gib(budget, "input_embeddings"),
            quality_risk="low if numerically identical and transfers are correct",
            speed_risk="medium to high; token lookup can cross the PCIe boundary",
            upstream_available="conditional",
            upstream_projects=("ExLlamaV3",),
            source="component estimate plus external-runtime capability check",
            notes="Zero saving means the placeholder config has no embedding manifest size.",
        ),
        _make_option(
            config,
            budget,
            option_id="quantized-embeddings-4bit",
            category="weight_precision",
            description="Quantize input embeddings to 4-bit storage.",
            saving_gib=_embedding_quantization_saving(config, 4.0),
            quality_risk="medium; embedding quantization needs task/perplexity evidence",
            speed_risk="low to medium; lookup/dequantization overhead is runtime-specific",
            upstream_available="conditional",
            upstream_projects=("ExLlamaV3", "GemLite"),
            source="weight-group arithmetic; not a tested backend result",
            notes="Requires a concrete checkpoint and runtime support for embedding tensors.",
        ),
    ]
    for bits in (4.0, 3.0, 2.25):
        options.append(
            _make_option(
                config,
                budget,
                option_id=f"lm-head-{bits:g}bit",
                category="weight_precision",
                description=f"Reduce lm_head storage to {bits:g} bits per weight.",
                saving_gib=_lm_head_saving(config, bits),
                quality_risk="medium to high; output projection quality must be measured",
                speed_risk="medium; dequantization and output bandwidth can change",
                upstream_available="conditional",
                upstream_projects=("ExLlamaV3", "llama.cpp"),
                source="weight-group arithmetic; not a tested backend result",
                notes="No saving is claimed until lm_head size and tied-weight status are known.",
            )
        )
    current_kv = budget.kv_cache_gib
    for bits in (8.0, 6.0, 4.0, 3.0, 2.0):
        target_bytes = bits / 8
        saving = current_kv * max(0.0, 1.0 - target_bytes / config.kv_dtype_bytes)
        options.append(
            _make_option(
                config,
                budget,
                option_id=f"kv-q{bits:g}",
                category="kv_cache",
                description=f"Quantize KV cache to Q{bits:g} equivalent storage.",
                saving_gib=saving,
                quality_risk="low at Q8, increasing from Q6 through Q2",
                speed_risk="low to medium; cache dequantization/kernel support is runtime-specific",
                upstream_available="reported_upstream_capability",
                upstream_projects=("ExLlamaV3",),
                source="KV element-size arithmetic; exact cache layout unmeasured",
                notes="The upstream runtime must be tested with the concrete model and context.",
            )
        )
    options.extend(
        [
            _make_option(
                config,
                budget,
                option_id="mixed-precision-by-tensor-sensitivity",
                category="mixed_precision",
                description=(
                    "Assign lower bits only to tensors that pass a quality sensitivity gate."
                ),
                saving_gib=0.0,
                quality_risk="unknown until calibration/evaluation",
                speed_risk="medium; heterogeneous dispatch can add overhead",
                upstream_available="conditional",
                upstream_projects=("ExLlamaV3", "GemLite"),
                source="requires layer sensitivity measurements",
                notes=(
                    "Planner cannot invent a savings fraction without a checkpoint "
                    "and allocation map."
                ),
            ),
            _make_option(
                config,
                budget,
                option_id="reduce-codebook-metadata",
                category="metadata",
                description=(
                    "Use a more compact upstream-supported metadata/codebook representation."
                ),
                saving_gib=_component_gib(budget, "codebooks_metadata") * 0.25,
                quality_risk="low to medium; depends on representation",
                speed_risk="low to medium; extra indirection may cost bandwidth",
                upstream_available="conditional",
                upstream_projects=("ExLlamaV3", "GemLite"),
                source="25% sensitivity scenario over configured codebook bytes",
                notes="This is a scenario, not evidence that a particular backend achieves 25%.",
            ),
            _make_option(
                config,
                budget,
                option_id="disable-unused-multimodal-components",
                category="model_scope",
                description=(
                    "Do not load vision/multimodal modules when the checkpoint does not need them."
                ),
                saving_gib=0.0,
                quality_risk="none for a dense text-only checkpoint; not applicable otherwise",
                speed_risk="low",
                upstream_available="conditional",
                upstream_projects=("ExLlamaV3", "llama.cpp"),
                source="no multimodal component is present in the placeholder config",
                notes="Savings become measurable only after a concrete model manifest is supplied.",
            ),
            _make_option(
                config,
                budget,
                option_id="reuse-scratch-buffers",
                category="workspace",
                description="Reuse runtime scratch allocations across adjacent operations.",
                saving_gib=_component_gib(budget, "runtime_scratch_workspace"),
                quality_risk="low if buffer lifetimes are correct",
                speed_risk="low to medium; synchronization/lifetime constraints may appear",
                upstream_available="conditional",
                upstream_projects=("ExLlamaV3", "GemLite", "CUTLASS"),
                source="modeled scratch component; not allocator telemetry",
                notes="The current placeholder uses the configured reserve split.",
            ),
        ]
    )
    return sorted(options, key=lambda item: (not item.reaches_target_peak, item.new_total_gib))


def compare_memory_configurations(config: ExperimentConfig) -> dict[str, Any]:
    budget = estimate_memory(config)
    options = suggest_optimizations(config, budget)
    by_id = {option.id: option for option in options}
    combinations = []
    for kv_id in ("kv-q4", "kv-q3", "kv-q2"):
        kv = by_id[kv_id]
        scratch = by_id["reuse-scratch-buffers"]
        saving = kv.vram_saving_gib + scratch.vram_saving_gib
        combinations.append(
            _make_option(
                config,
                budget,
                option_id=f"{kv_id}+reuse-scratch-buffers",
                category="combined_scenario",
                description=f"{kv.description} Then reuse scratch buffers.",
                saving_gib=saving,
                quality_risk=f"{kv.quality_risk}; scratch reuse requires correctness validation",
                speed_risk=kv.speed_risk,
                upstream_available="conditional",
                upstream_projects=("ExLlamaV3", "GemLite", "CUTLASS"),
                source="sum of two non-overlapping modeled components",
                notes="Scenario only; it is not a hardware benchmark.",
            )
        )
    return {
        "schema_version": 1,
        "target_peak_vram_gib": config.target_peak_vram_gib,
        "baseline": budget.to_dict(),
        "options": [option.to_dict() for option in options],
        "combined_scenarios": [option.to_dict() for option in combinations],
        "evidence_status": budget.evidence_status,
    }
