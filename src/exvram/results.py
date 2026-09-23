from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import SCHEMA_VERSION, ExperimentConfig, ResultRecord
from .hardware import HardwareInfo
from .quality import not_run_quality_gate


def new_result(
    *,
    config: ExperimentConfig,
    hardware: HardwareInfo,
    benchmark_kind: str,
    synthetic: bool,
    metrics: dict[str, Any],
    notes: list[str] | None = None,
) -> ResultRecord:
    return ResultRecord(
        schema_version=SCHEMA_VERSION,
        benchmark_id=config.id,
        benchmark_kind=benchmark_kind,
        synthetic=synthetic,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        config=config.to_dict(),
        hardware=hardware.to_dict(),
        metrics=metrics,
        quality_gate=not_run_quality_gate(),
        notes=notes or [],
    )


def write_json(record: ResultRecord, path: str | None) -> str | None:
    if path is None:
        return None
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return str(output)


def write_jsonl(record: ResultRecord, path: str | None) -> str | None:
    if path is None:
        return None
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    return str(output)


def append_jsonl_payload(payload: dict[str, Any], path: str | None) -> str | None:
    """Append a machine-readable record that is not the model-result dataclass."""
    if path is None:
        return None
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return str(output)


def write_result(record: ResultRecord, path: str | None) -> str | None:
    """Write pretty JSON or append one JSONL record based on the file suffix."""
    if path and Path(path).suffix.lower() == ".jsonl":
        return write_jsonl(record, path)
    return write_json(record, path)
