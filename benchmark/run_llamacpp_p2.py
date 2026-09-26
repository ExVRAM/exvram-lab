"""Run one full-model text-only-input point through an external llama.cpp binary."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from exvram.storage import StorageSafetyError, assert_storage_safe

_PROMPT_RE = re.compile(r"prompt eval time\s*=\s*([0-9.]+) ms /\s*([0-9]+) tokens")
_DECODE_RE = re.compile(r"eval time\s*=\s*([0-9.]+) ms /\s*([0-9]+) runs")
_RATE_RE = re.compile(
    r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]",
    re.IGNORECASE,
)
_GPU_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU", re.IGNORECASE)
_BUFFER_RE = re.compile(
    r"(?P<kind>CUDA\d+[^=]*)buffer size\s*=\s*(?P<mib>[0-9.]+) MiB",
    re.IGNORECASE,
)
_FIT_MEMORY_RE = re.compile(
    r'\{\s*"type"\s*:\s*"fit_memory_breakdown"\s*,',
    re.IGNORECASE,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, help="path to llama-cli.exe")
    parser.add_argument("--model", required=True, help="path to a full GGUF model")
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--gpu-layers", default="auto")
    parser.add_argument("--cache-type-k", default="q4_0")
    parser.add_argument("--cache-type-v", default="q4_0")
    parser.add_argument(
        "--search-candidate-id",
        default=None,
        help="exact candidate identity from experiments/search/*.json",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-removable-storage",
        action="store_true",
        help="allow model/runtime/output paths on a removable volume",
    )
    return parser


def _prompt(context: int) -> str:
    seed = (
        "ExVRAM Lab measures reproducible local language model inference. "
        "Continue this plain text benchmark without using image input. "
    )
    return (seed * max(1, context // 12 + 1))[: context * 5]


def _parse_metrics(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    prompt = _PROMPT_RE.search(text)
    decode = _DECODE_RE.search(text)
    gpu = _GPU_RE.search(text)
    if prompt:
        milliseconds, tokens = prompt.groups()
        metrics["prompt_eval_ms"] = float(milliseconds)
        metrics["prompt_tokens"] = int(tokens)
        metrics["prefill_tokens_per_second"] = (
            int(tokens) / (float(milliseconds) / 1000.0) if float(milliseconds) else None
        )
    if decode:
        milliseconds, tokens = decode.groups()
        metrics["decode_eval_ms"] = float(milliseconds)
        metrics["decode_tokens"] = int(tokens)
        metrics["decode_tokens_per_second"] = (
            int(tokens) / (float(milliseconds) / 1000.0) if float(milliseconds) else None
        )
    rate = _RATE_RE.search(text)
    if rate:
        prompt_rate, decode_rate = rate.groups()
        metrics["prefill_tokens_per_second"] = float(prompt_rate)
        metrics["decode_tokens_per_second"] = float(decode_rate)
    if gpu:
        metrics["gpu_layers"] = int(gpu.group(1))
        metrics["total_layers"] = int(gpu.group(2))
    buffers = [
        {"kind": match.group("kind").strip(), "mib": float(match.group("mib"))}
        for match in _BUFFER_RE.finditer(text)
    ]
    if buffers:
        metrics["reported_cuda_buffers"] = buffers
    fit_memory_breakdown = None
    decoder = json.JSONDecoder()
    for match in _FIT_MEMORY_RE.finditer(text):
        try:
            payload, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if payload.get("type") == "fit_memory_breakdown":
            fit_memory_breakdown = payload.get("data")
    if isinstance(fit_memory_breakdown, dict):
        metrics["fit_memory_breakdown"] = fit_memory_breakdown
    return metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        args.binary,
        "--model",
        args.model,
        "--ctx-size",
        str(args.context),
        "--gpu-layers",
        args.gpu_layers,
        "--cache-type-k",
        args.cache_type_k,
        "--cache-type-v",
        args.cache_type_v,
        "--seed",
        "17",
        "--n-predict",
        str(args.decode_tokens),
        "--prompt",
        _prompt(args.context),
        "--single-turn",
        "--no-display-prompt",
        "--log-jsonl",
        "--perf",
    ]
    record: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": f"p2-llamacpp-bartowski-qwen38-iq2-xxs-c{args.context}",
        "phase": "P2",
        "benchmark_kind": "full_model_shootout",
        "backend": "llamacpp",
        "model_id": "Qwen3.8-27B",
        "artifact_ref": "bartowski/Qwen3.8-27B-GGUF@IQ2_XXS",
        "synthetic": False,
        "measurement_status": "planned",
        "status": "PLANNED",
        "context_tokens": args.context,
        "batch_size": 1,
        "text_only_input": True,
        "vision_input": False,
        "mtp_enabled": False,
        "runtime_config": {
            "gpu_layers": args.gpu_layers,
            "cache_type_k": args.cache_type_k,
            "cache_type_v": args.cache_type_v,
            "decode_tokens": args.decode_tokens,
        },
        "command": command,
    }
    if args.search_candidate_id:
        record["search_candidate_id"] = args.search_candidate_id
    started = time.perf_counter()
    try:
        assert_storage_safe(
            [args.binary, args.model, args.output],
            allow_removable=args.allow_removable_storage,
        )
        model_bytes = Path(args.model).stat().st_size
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.timeout,
            check=False,
        )
        elapsed = time.perf_counter() - started
        combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
        metrics = _parse_metrics(combined)
        record.update(
            {
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
                "measurement_status": "measured" if completed.returncode == 0 else "failed",
                "status": "PASS" if completed.returncode == 0 and metrics else "FAIL",
                "metrics": {
                    **metrics,
                    "model_file_bytes": model_bytes,
                    "artifact_file_bpw_estimate": model_bytes * 8.0 / 27_000_000_000.0,
                },
                "raw_stdout_tail": (completed.stdout or "")[-12000:],
                "raw_stderr_tail": (completed.stderr or "")[-12000:],
            }
        )
    except StorageSafetyError as exc:
        record.update(
            {
                "measurement_status": "blocked",
                "status": "INCONCLUSIVE",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    except subprocess.TimeoutExpired as exc:
        record.update(
            {
                "measurement_status": "timeout",
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": f"llama.cpp exceeded timeout {args.timeout}s",
            }
        )
    except OSError as exc:
        record.update(
            {
                "measurement_status": "unavailable",
                "status": "INCONCLUSIVE",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    record["quality_gate"] = {
        "status": "NOT_RUN",
        "reason": "P2 throughput smoke only; quality suite is separate.",
    }
    return record


def main() -> int:
    args = _parser().parse_args()
    payload = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
