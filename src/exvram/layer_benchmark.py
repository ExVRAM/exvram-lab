"""Real CUDA layer timing when CUDA-enabled Torch is available.

The runner never falls back to CPU silently. A missing CUDA runtime or external backend becomes an
explicit unavailable/not-exercised record in the JSONL evidence.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import statistics
from datetime import datetime, timezone
from typing import Any

from .hardware import HardwareInfo
from .memory import GIB

LAYER_BENCHMARK_SCHEMA_VERSION = 1
DEFAULT_LAYER_SHAPES = (
    {"name": "attention_q_proj", "in_features": 4096, "out_features": 4096},
    {"name": "attention_o_proj", "in_features": 4096, "out_features": 4096},
    {"name": "mlp_up_proj", "in_features": 4096, "out_features": 11008},
    {"name": "mlp_down_proj", "in_features": 11008, "out_features": 4096},
)
DEFAULT_BACKENDS = (
    "torch-fp16",
    "torch-bf16",
    "exl3",
    "gemlite-2bit",
    "gemlite-4bit",
    "bitnet-w2a8",
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base_record(
    hardware: HardwareInfo,
    backend: str,
    layer: dict[str, int | str],
    *,
    status: str,
    notes: list[str],
    batch_size: int = 1,
    error: str | None = None,
) -> dict[str, Any]:
    result = {
        "schema_version": LAYER_BENCHMARK_SCHEMA_VERSION,
        "benchmark_kind": "layer_microbenchmark",
        "synthetic": False,
        "measurement_status": status,
        "timestamp_utc": _timestamp(),
        "backend": backend,
        "upstream_revision_tested": None,
        "layer": layer,
        "batch_size": batch_size,
        "warmup": 0,
        "repeats": 0,
        "hardware": hardware.to_dict(),
        "metrics": {},
        "correctness": {},
        "notes": notes,
    }
    if error:
        result["error"] = error
    return result


def _torch_status() -> tuple[Any | None, str | None]:
    if importlib.util.find_spec("torch") is None:
        return None, "PyTorch is not installed"
    try:
        import torch
    except Exception as exc:  # optional backend probe
        return None, f"PyTorch import failed: {type(exc).__name__}: {exc}"
    if not torch.cuda.is_available():
        return None, "PyTorch is installed but has no CUDA-enabled runtime"
    return torch, None


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))]


def _run_torch_layer(
    torch: Any,
    hardware: HardwareInfo,
    backend: str,
    layer: dict[str, int | str],
    *,
    dtype: Any,
    warmup: int,
    repeats: int,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    in_features = int(layer["in_features"])
    out_features = int(layer["out_features"])
    torch.manual_seed(seed)
    activation = torch.randn((batch_size, in_features), device=device, dtype=dtype)
    weight = torch.randn((out_features, in_features), device=device, dtype=dtype)
    weight_bytes = weight.numel() * weight.element_size()
    for _ in range(warmup):
        output = activation.matmul(weight.t())
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    elapsed_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = activation.matmul(weight.t())
            end.record()
            end.synchronize()
            elapsed_ms.append(float(start.elapsed_time(end)))
    torch.cuda.synchronize(device)
    reference = activation.float().matmul(weight.float().t())
    difference = output.float() - reference
    max_abs = float(difference.abs().max().item())
    rmse = float(difference.square().mean().sqrt().item())
    max_reference = max(float(reference.abs().max().item()), 1e-8)
    median_ms = float(statistics.median(elapsed_ms))
    seconds = max(median_ms / 1000.0, 1e-12)
    flops = 2 * in_features * out_features
    allocated = int(torch.cuda.memory_allocated(device))
    peak = int(torch.cuda.max_memory_allocated(device))
    return {
        "schema_version": LAYER_BENCHMARK_SCHEMA_VERSION,
        "benchmark_kind": "layer_microbenchmark",
        "synthetic": False,
        "measurement_status": "measured",
        "timestamp_utc": _timestamp(),
        "backend": backend,
        "upstream_revision_tested": "torch-" + str(torch.__version__),
        "layer": {
            **layer,
            "batch_size": batch_size,
            "dtype": str(dtype),
        },
        "batch_size": batch_size,
        "warmup": warmup,
        "repeats": repeats,
        "hardware": hardware.to_dict(),
        "metrics": {
            "latency_ms_median": median_ms,
            "latency_ms_p95": _p95(elapsed_ms),
            "kernel_time_ms": median_ms,
            "effective_weight_bytes_read": weight_bytes,
            "effective_weight_gib_per_s": weight_bytes / seconds / GIB,
            "effective_compute_tflops": flops / seconds / 1e12,
            "useful_compute_per_memory_traffic": flops / max(weight_bytes, 1),
            "vram_allocated_bytes": allocated,
            "vram_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "vram_peak_bytes": peak,
            "workspace_bytes_estimate": max(0, peak - allocated),
            "throughput_rows_per_second": batch_size / seconds,
            "tokens_per_second_equivalent": None,
        },
        "correctness": {
            "reference": "same operation in FP32 on CUDA",
            "max_absolute_error": max_abs,
            "rmse": rmse,
            "max_relative_error": max_abs / max_reference,
        },
        "notes": [
            "Layer timing only; this is not full-model tokens/second.",
            "Effective bandwidth counts theoretical weight bytes, not profiler traffic.",
        ],
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gemlite_revision() -> str:
    gemlite_version = _distribution_version("gemlite") or "unknown"
    triton_version = _distribution_version("triton-windows") or _distribution_version("triton")
    suffix = f"; triton-{triton_version}" if triton_version else ""
    return f"gemlite-{gemlite_version}{suffix}"


def _quantize_per_group(
    torch: Any, weight: Any, bits: int, group_size: int
) -> tuple[Any, Any, Any]:
    if weight.shape[1] % group_size:
        raise ValueError("in_features must be divisible by group_size for GemLite quantization")
    qmax = float((1 << bits) - 1)
    grouped = weight.float().view(weight.shape[0], weight.shape[1] // group_size, group_size)
    minimum = grouped.amin(dim=-1)
    maximum = grouped.amax(dim=-1)
    scales = ((maximum - minimum) / qmax).clamp_min(1e-6)
    zeros = torch.round(-minimum / scales).clamp(0, qmax)
    quantized = torch.round(grouped / scales.unsqueeze(-1) + zeros.unsqueeze(-1))
    quantized = quantized.clamp(0, qmax).to(torch.uint8).view(weight.shape)
    return quantized.contiguous(), scales.contiguous(), zeros.contiguous()


def _parameter_bytes(module: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, parameter in module.named_parameters():
        result[name] = int(parameter.numel() * parameter.element_size())
    return result


def _run_gemlite_layer(
    torch: Any,
    hardware: HardwareInfo,
    backend: str,
    layer: dict[str, int | str],
    *,
    warmup: int,
    repeats: int,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    try:
        from gemlite import DType, GemLiteLinear
    except Exception as exc:
        return _base_record(
            hardware,
            backend,
            layer,
            status="unavailable",
            notes=["GemLite import failed; no timing was fabricated."],
            batch_size=batch_size,
            error=f"{type(exc).__name__}: {exc}",
        )

    device = torch.device("cuda")
    in_features = int(layer["in_features"])
    out_features = int(layer["out_features"])
    bits = int(backend.rsplit("-", 1)[1].replace("bit", ""))
    group_size = 64
    torch.manual_seed(seed)
    activation = torch.randn((batch_size, in_features), device=device, dtype=torch.float16)
    weight = torch.randn((out_features, in_features), device=device, dtype=torch.float16)
    quantized, scales, zeros = _quantize_per_group(torch, weight, bits, group_size)
    gemlite_linear = GemLiteLinear(
        bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        input_dtype=DType.FP16,
        output_dtype=DType.FP16,
    )
    gemlite_linear.pack(quantized, scales.to(torch.float16), zeros.to(torch.float16))
    torch.cuda.synchronize(device)

    # Keep the correctness reference off the GPU while timing the packed operator.
    with torch.inference_mode():
        reference_cpu = activation.float().matmul(weight.float().t()).cpu()
        dequantized = (
            (quantized.float().view(out_features, in_features // group_size, group_size)
             - zeros.unsqueeze(-1))
            * scales.unsqueeze(-1)
        ).view(out_features, in_features)
        quantized_reference_cpu = activation.float().matmul(dequantized.t()).cpu()
    del dequantized
    del quantized, scales, zeros, weight
    torch.cuda.empty_cache()
    persistent = _parameter_bytes(gemlite_linear)
    persistent_bytes = sum(persistent.values())
    theoretical_packed_bytes = math.ceil(out_features * in_features * bits / 8)
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    for _ in range(warmup):
        output = gemlite_linear(activation)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    elapsed_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = gemlite_linear(activation)
            end.record()
            end.synchronize()
            elapsed_ms.append(float(start.elapsed_time(end)))
    torch.cuda.synchronize(device)
    output_cpu = output.float().cpu()
    difference = output_cpu - reference_cpu
    kernel_difference = output_cpu - quantized_reference_cpu
    max_abs = float(difference.abs().max().item())
    rmse = float(difference.square().mean().sqrt().item())
    kernel_max_abs = float(kernel_difference.abs().max().item())
    kernel_rmse = float(kernel_difference.square().mean().sqrt().item())
    max_reference = max(float(reference_cpu.abs().max().item()), 1e-8)
    median_ms = float(statistics.median(elapsed_ms))
    seconds = max(median_ms / 1000.0, 1e-12)
    flops = 2 * in_features * out_features * batch_size
    peak = int(torch.cuda.max_memory_allocated(device))
    output_bytes = int(output.numel() * output.element_size())
    weight_bytes = persistent.get("W_q", 0)
    scale_bytes = persistent.get("scales", 0)
    zero_bytes = persistent.get("zeros", 0)
    metadata_bytes = persistent.get("metadata", 0) + persistent.get("orig_shape", 0)
    return {
        "schema_version": LAYER_BENCHMARK_SCHEMA_VERSION,
        "benchmark_kind": "layer_microbenchmark",
        "synthetic": False,
        "measurement_status": "measured",
        "timestamp_utc": _timestamp(),
        "backend": backend,
        "upstream_revision_tested": _gemlite_revision(),
        "layer": {**layer, "batch_size": batch_size, "dtype": "torch.float16"},
        "batch_size": batch_size,
        "warmup": warmup,
        "repeats": repeats,
        "hardware": hardware.to_dict(),
        "metrics": {
            "latency_ms_median": median_ms,
            "latency_ms_p95": _p95(elapsed_ms),
            "kernel_time_ms": median_ms,
            "throughput_rows_per_second": batch_size / seconds,
            "effective_weight_bytes_read": weight_bytes,
            "effective_weight_gib_per_s": weight_bytes / seconds / GIB,
            "effective_compute_tflops": flops / seconds / 1e12,
            "useful_compute_per_memory_traffic": flops / max(weight_bytes, 1),
            "parameter_count": out_features * in_features,
            "nominal_weight_bpw": bits,
            "theoretical_packed_weight_bytes": theoretical_packed_bytes,
            "actual_stored_weight_bytes": weight_bytes,
            "scale_bytes": scale_bytes,
            "zero_bytes": zero_bytes,
            "metadata_bytes": metadata_bytes,
            "lookup_codebook_bytes": 0,
            "alignment_padding_bytes": max(0, weight_bytes - theoretical_packed_bytes),
            "persistent_bytes": persistent_bytes,
            "effective_physical_bpw": persistent_bytes * 8 / (out_features * in_features),
            "vram_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "vram_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "vram_peak_bytes": peak,
            "workspace_bytes_estimate": max(0, peak - baseline_allocated - output_bytes),
            "tokens_per_second_equivalent": None,
        },
        "correctness": {
            "reference": "original FP32 CUDA weight matmul",
            "max_absolute_error": max_abs,
            "rmse": rmse,
            "max_relative_error": max_abs / max_reference,
            "kernel_vs_dequantized_max_absolute_error": kernel_max_abs,
            "kernel_vs_dequantized_rmse": kernel_rmse,
        },
        "notes": [
            "Real GemLite Triton execution on CUDA; this is not a full-model tokens/second result.",
            "Stored bytes include GemLite packed weights, scales, zero points, and metadata.",
            "The generated quantization fixture is per-group asymmetric affine quantization.",
        ],
    }


def _run_exl3_layer(
    torch: Any,
    hardware: HardwareInfo,
    layer: dict[str, int | str],
    *,
    warmup: int,
    repeats: int,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    """Exercise the official ExLlamaV3 EXL3 path with a small calibration-free fixture.

    The fixture deliberately uses ExLlamaV3's own quantizer and LinearEXL3 class. It is not a
    model quality result: no checkpoint or calibration corpus is available in this repository.
    """
    try:
        from exllamav3.modules.quant.exl3 import LinearEXL3
        from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    except Exception as exc:
        return _base_record(
            hardware,
            "exl3",
            layer,
            status="unavailable",
            notes=["ExLlamaV3 import failed; no timing was fabricated."],
            batch_size=batch_size,
            error=f"{type(exc).__name__}: {exc}",
        )

    device = torch.device("cuda")
    in_features = int(layer["in_features"])
    out_features = int(layer["out_features"])
    if in_features % 16 or out_features % 128:
        return _base_record(
            hardware,
            "exl3",
            layer,
            status="unsupported",
            notes=["This EXL3 fixture requires K divisible by 16 and N divisible by 128."],
            batch_size=batch_size,
            error="layer dimensions are not compatible with the official EXL3 quantizer",
        )

    torch.manual_seed(seed)
    # ExLlamaV3 expects row-major (in_features, out_features) weights.
    weight = torch.randn((in_features, out_features), device=device, dtype=torch.float32)
    weight_cpu = weight.cpu()
    h_data = {
        "H": torch.empty((in_features, in_features), device="meta"),
        "L": None,
        "finalized": False,
        "diag": None,
        "q_fallback": True,
        "device": device,
        "count": 0,
        "num_total": 0,
    }
    quant_args = {
        "K": 4,
        "devices": [f"cuda:{torch.cuda.current_device()}"],
        "seed": seed,
        "sigma_reg": 0.025,
        "apply_out_scales": False,
        "no_refit": True,
    }
    try:
        _, proxy_error, packed = quantize_exl3(
            weight,
            h_data,
            quant_args,
            return_weight_q=False,
            progress_str=None,
            verbose=False,
        )
        exl3_linear = LinearEXL3(
            None,
            in_features,
            out_features,
            suh=packed["suh"],
            svh=packed["svh"],
            trellis=packed["trellis"],
            mcg=packed.get("mcg"),
            mul1=packed.get("mul1"),
        )
    except Exception as exc:
        return _base_record(
            hardware,
            "exl3",
            layer,
            status="unavailable",
            notes=["ExLlamaV3 fixture packing failed; no timing was fabricated."],
            batch_size=batch_size,
            error=f"{type(exc).__name__}: {exc}",
        )

    activation = torch.randn((batch_size, in_features), device=device, dtype=torch.float16)
    with torch.inference_mode():
        reference_cpu = activation.cpu().float().matmul(weight_cpu)
    del weight, weight_cpu
    torch.cuda.empty_cache()
    params = {"reconstruct": False}
    torch.cuda.synchronize(device)
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    for _ in range(warmup):
        output = exl3_linear.forward(activation, params)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    elapsed_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = exl3_linear.forward(activation, params)
            end.record()
            end.synchronize()
            elapsed_ms.append(float(start.elapsed_time(end)))
    torch.cuda.synchronize(device)
    output_cpu = output.float().cpu()
    difference = output_cpu - reference_cpu
    max_abs = float(difference.abs().max().item())
    rmse = float(difference.square().mean().sqrt().item())
    max_reference = max(float(reference_cpu.abs().max().item()), 1e-8)
    median_ms = float(statistics.median(elapsed_ms))
    seconds = max(median_ms / 1000.0, 1e-12)
    flops = 2 * in_features * out_features * batch_size
    persistent = {
        name: int(tensor.numel() * tensor.element_size())
        for name, tensor in packed.items()
    }
    persistent_bytes = sum(persistent.values())
    theoretical_packed_bytes = math.ceil(in_features * out_features * 4 / 8)
    output_bytes = int(output.numel() * output.element_size())
    peak = int(torch.cuda.max_memory_allocated(device))
    return {
        "schema_version": LAYER_BENCHMARK_SCHEMA_VERSION,
        "benchmark_kind": "layer_microbenchmark",
        "synthetic": False,
        "measurement_status": "measured",
        "timestamp_utc": _timestamp(),
        "backend": "exl3",
        "upstream_revision_tested": (
            f"exllamav3-{_distribution_version('exllamav3') or '1.5.1-wheel'}"
            f"; torch-{torch.__version__}; exl3-K4"
        ),
        "layer": {**layer, "batch_size": batch_size, "dtype": "torch.float16"},
        "batch_size": batch_size,
        "warmup": warmup,
        "repeats": repeats,
        "hardware": hardware.to_dict(),
        "metrics": {
            "latency_ms_median": median_ms,
            "latency_ms_p95": _p95(elapsed_ms),
            "kernel_time_ms": median_ms,
            "throughput_rows_per_second": batch_size / seconds,
            "effective_weight_bytes_read": persistent.get("trellis", 0),
            "effective_weight_gib_per_s": persistent.get("trellis", 0) / seconds / GIB,
            "effective_compute_tflops": flops / seconds / 1e12,
            "useful_compute_per_memory_traffic": flops / max(persistent.get("trellis", 0), 1),
            "parameter_count": in_features * out_features,
            "nominal_weight_bpw": 4,
            "theoretical_packed_weight_bytes": theoretical_packed_bytes,
            "actual_stored_weight_bytes": persistent.get("trellis", 0),
            "scale_bytes": 0,
            "zero_bytes": 0,
            "metadata_bytes": persistent_bytes - persistent.get("trellis", 0),
            "lookup_codebook_bytes": (
                persistent.get("mcg", 0) + persistent.get("mul1", 0)
            ),
            "alignment_padding_bytes": max(
                0, persistent.get("trellis", 0) - theoretical_packed_bytes
            ),
            "persistent_bytes": persistent_bytes,
            "effective_physical_bpw": persistent_bytes * 8 / (in_features * out_features),
            "vram_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "vram_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "vram_peak_bytes": peak,
            "workspace_bytes_estimate": max(
                0, peak - baseline_allocated - output_bytes
            ),
            "tokens_per_second_equivalent": None,
        },
        "correctness": {
            "reference": "original FP32 CUDA weight matmul",
            "max_absolute_error": max_abs,
            "rmse": rmse,
            "max_relative_error": max_abs / max_reference,
            "exl3_proxy_error": float(proxy_error),
        },
        "notes": [
            "Real ExLlamaV3 LinearEXL3 execution on CUDA; this is not a full-model result.",
            "The fixture uses K=4 calibration-free EXL3 packing because no model checkpoint "
            "is present.",
            "Stored bytes include official EXL3 trellis and sign/scale metadata tensors.",
        ],
    }


def _external_backend_record(
    hardware: HardwareInfo, backend: str, layer: dict[str, int | str], batch_size: int
) -> dict[str, Any]:
    module = "gemlite" if backend.startswith("gemlite") else None
    if module and importlib.util.find_spec(module) is None:
        return _base_record(
            hardware,
            backend,
            layer,
            status="unavailable",
            notes=["External backend is not installed; no timing was fabricated."],
            batch_size=batch_size,
            error=f"Python module {module} is unavailable",
        )
    if backend == "exl3" and importlib.util.find_spec("exllamav3") is None:
        return _base_record(
            hardware,
            backend,
            layer,
            status="unavailable",
            notes=["ExLlamaV3 is not installed; no timing was fabricated."],
            batch_size=batch_size,
            error="Python module exllamav3 is unavailable",
        )
    if backend == "bitnet-w2a8":
        return _base_record(
            hardware,
            backend,
            layer,
            status="not_applicable",
            notes=[
                "BitNet GPU kernels target BitNet-family formats; generic dense-layer "
                "compatibility is not established.",
                "No timing was fabricated.",
            ],
            batch_size=batch_size,
            error="BitNet W2A8 shape/format adapter is not established for this placeholder model",
        )
    return _base_record(
        hardware,
        backend,
        layer,
        status="not_exercised",
        notes=[
            "API/package probe succeeded, but no checkpoint-backed packing fixture is available.",
            "No timing was fabricated.",
        ],
        batch_size=batch_size,
        error="benchmark fixture requires a concrete upstream packed tensor",
    )


def run_layer_benchmark(
    hardware: HardwareInfo,
    *,
    backends: tuple[str, ...] = DEFAULT_BACKENDS,
    layers: tuple[dict[str, int | str], ...] = DEFAULT_LAYER_SHAPES,
    warmup: int = 10,
    repeats: int = 50,
    seed: int = 17,
    batch_sizes: tuple[int, ...] = (1,),
) -> list[dict[str, Any]]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be >= 0 and repeats must be positive")
    if not batch_sizes or any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")
    torch, torch_error = _torch_status()
    results: list[dict[str, Any]] = []
    for backend in backends:
        for layer in layers:
            for batch_size in batch_sizes:
                if backend in {"torch-fp16", "torch-bf16"}:
                    if torch is None:
                        record = _base_record(
                            hardware,
                            backend,
                            layer,
                            status="unavailable",
                            notes=["No CUDA timing was performed."],
                            batch_size=batch_size,
                            error=torch_error,
                        )
                        record["warmup"] = warmup
                        record["repeats"] = repeats
                        results.append(record)
                        continue
                    dtype = torch.float16 if backend == "torch-fp16" else torch.bfloat16
                    if backend == "torch-bf16" and not torch.cuda.is_bf16_supported():
                        results.append(
                            _base_record(
                                hardware,
                                backend,
                                layer,
                                status="unsupported",
                                notes=["CUDA device/runtime does not report BF16 support."],
                                batch_size=batch_size,
                                error="torch.cuda.is_bf16_supported() is false",
                            )
                        )
                        continue
                    results.append(
                        _run_torch_layer(
                            torch,
                            hardware,
                            backend,
                            layer,
                            dtype=dtype,
                            warmup=warmup,
                            repeats=repeats,
                            seed=seed,
                            batch_size=batch_size,
                        )
                    )
                    continue
                if torch is None and (backend.startswith("gemlite") or backend == "exl3"):
                    results.append(
                        _base_record(
                            hardware,
                            backend,
                            layer,
                            status="unavailable",
                            notes=[
                                "CUDA-enabled Torch is required before this low-bit backend "
                                "can be exercised."
                            ],
                            batch_size=batch_size,
                            error=torch_error,
                        )
                    )
                    continue
                if backend.startswith("gemlite"):
                    results.append(
                        _run_gemlite_layer(
                            torch,
                            hardware,
                            backend,
                            layer,
                            warmup=warmup,
                            repeats=repeats,
                            seed=seed,
                            batch_size=batch_size,
                        )
                    )
                    continue
                if backend == "exl3" and torch is not None:
                    results.append(
                        _run_exl3_layer(
                            torch,
                            hardware,
                            layer,
                            warmup=warmup,
                            repeats=repeats,
                            seed=seed,
                            batch_size=batch_size,
                        )
                    )
                    continue
                results.append(_external_backend_record(hardware, backend, layer, batch_size))
    return results
