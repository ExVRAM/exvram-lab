"""Optional boundary for the official Ollama CLI; no Ollama source is vendored."""

from __future__ import annotations

import re
import subprocess
import time
from typing import Any

_DURATION_RE = re.compile(
    r"(?P<label>total duration|load duration|prompt eval duration|eval duration):\s*"
    r"(?P<value>[0-9.]+)\s*(?P<unit>ns|µs|us|ms|s)",
    re.IGNORECASE,
)
_COUNT_RE = re.compile(
    r"(?P<label>prompt eval count|eval count):\s*(?P<count>[0-9]+)\s*token",
    re.IGNORECASE,
)


def build_ollama_run_command(
    executable: str,
    model: str,
    prompt: str | None = None,
    *,
    verbose: bool = True,
    keepalive: str = "5m",
) -> list[str]:
    command = [executable, "run", model]
    if prompt is not None:
        command.append(prompt)
    if verbose:
        command.append("--verbose")
    if keepalive:
        command.extend(["--keepalive", keepalive])
    return command


def parse_ollama_list(text: str) -> tuple[str, ...]:
    models: list[str] = []
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0].lower() != "name" and len(fields) >= 2:
            models.append(fields[0])
    return tuple(models)


def _duration_seconds(value: str, unit: str) -> float:
    scale = {"ns": 1e-9, "µs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1.0}
    return float(value) * scale[unit.lower()]


def parse_ollama_verbose(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for match in _DURATION_RE.finditer(text):
        label = match.group("label").lower().replace(" ", "_")
        metrics[label] = _duration_seconds(match.group("value"), match.group("unit"))
    for match in _COUNT_RE.finditer(text):
        label = match.group("label").lower().replace(" ", "_")
        metrics[label] = int(match.group("count"))
    if metrics.get("prompt_eval_count") and metrics.get("prompt_eval_duration"):
        metrics["prefill_tokens_per_second"] = (
            metrics["prompt_eval_count"] / metrics["prompt_eval_duration"]
        )
    if metrics.get("eval_count") and metrics.get("eval_duration"):
        metrics["decode_tokens_per_second"] = metrics["eval_count"] / metrics["eval_duration"]
    return metrics


def _base_record(model: str, command: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark_kind": "external_runtime_smoke",
        "backend": "ollama",
        "model_id": model,
        "command": command,
        "synthetic": False,
        "text_only_input": True,
        "vision_input": False,
        "quality_gate": {
            "status": "NOT_RUN",
            "reason": "Runtime smoke only; model quality is a separate gate.",
        },
    }


def run_ollama(
    model: str,
    prompt: str,
    *,
    executable: str = "ollama",
    timeout: int = 120,
    keepalive: str = "5m",
) -> dict[str, Any]:
    list_command = [executable, "list"]
    command = build_ollama_run_command(executable, model, prompt, keepalive=keepalive)
    record = _base_record(model, command)
    try:
        listed = subprocess.run(
            list_command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
        if listed.returncode != 0:
            record.update(
                {
                    "status": "INCONCLUSIVE",
                    "measurement_status": "unavailable",
                    "error_type": "OllamaListFailed",
                    "error": (listed.stderr or listed.stdout).strip(),
                }
            )
            return record
        installed = parse_ollama_list(listed.stdout)
        if model not in installed:
            record.update(
                {
                    "status": "INCONCLUSIVE",
                    "measurement_status": "not_installed",
                    "error_type": "ModelNotInstalled",
                    "error": f"model {model!r} is not listed by ollama; no download was attempted",
                }
            )
            return record
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
        record.update(
            {
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "measurement_status": "measured" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "elapsed_seconds": time.perf_counter() - started,
                "metrics": parse_ollama_verbose(combined),
                "response_preview": (completed.stdout or "").strip()[-4000:],
            }
        )
        return record
    except subprocess.TimeoutExpired as exc:
        record.update(
            {
                "status": "FAIL",
                "measurement_status": "timeout",
                "error_type": type(exc).__name__,
                "error": f"ollama exceeded timeout {timeout}s",
            }
        )
        return record
    except OSError as exc:
        record.update(
            {
                "status": "INCONCLUSIVE",
                "measurement_status": "unavailable",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return record
