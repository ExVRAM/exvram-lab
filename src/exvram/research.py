"""Small standard-library research queue and SQLite/JSONL persistence layer."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import ExVRAMError

RESEARCH_STATUSES = ("PLANNED", "RUNNING", "PASS", "FAIL", "INCONCLUSIVE")
RESEARCH_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class QueueItem:
    """A planned or executed experiment, with stable machine-readable identity."""

    experiment_id: str
    phase: str
    backend: str
    model_id: str
    artifact_ref: str
    contexts: tuple[int, ...]
    status: str = "PLANNED"
    prerequisites: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ExVRAMError("experiment_id must be non-empty")
        if self.status not in RESEARCH_STATUSES:
            raise ExVRAMError(f"unsupported research status: {self.status}")
        if not self.contexts or any(context <= 0 for context in self.contexts):
            raise ExVRAMError("contexts must contain positive integers")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contexts"] = list(self.contexts)
        payload["prerequisites"] = list(self.prerequisites)
        payload["notes"] = list(self.notes)
        payload["schema_version"] = RESEARCH_SCHEMA_VERSION
        return payload


def validate_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the minimum provenance fields used by the P2 planner."""
    if payload.get("schema_version") != RESEARCH_SCHEMA_VERSION:
        raise ExVRAMError("unsupported research manifest schema_version")
    model = payload.get("model")
    candidates = payload.get("candidates")
    if not isinstance(model, dict) or not isinstance(candidates, list) or not candidates:
        raise ExVRAMError("manifest needs model and non-empty candidates")
    for key in ("id", "source_url", "revision", "license"):
        if not isinstance(model.get(key), str) or not model[key].strip():
            raise ExVRAMError(f"manifest.model.{key} must be a non-empty string")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ExVRAMError("manifest candidates must be objects")
        for key in ("id", "backend", "source_url", "revision", "license"):
            if not isinstance(candidate.get(key), str) or not candidate[key].strip():
                raise ExVRAMError(f"manifest candidate {key} must be a non-empty string")
        size = candidate.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ExVRAMError(f"manifest candidate {candidate['id']} needs positive bytes")
    return dict(payload)


def load_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExVRAMError(f"cannot read research manifest {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExVRAMError("research manifest root must be an object")
    return validate_manifest(payload)


def default_p2_queue(manifest: Mapping[str, Any]) -> list[QueueItem]:
    """Create the initial P2 matrix without claiming that an item has run."""
    validate_manifest(manifest)
    model_id = str(manifest["model"]["id"])
    queue: list[QueueItem] = []
    for candidate in manifest["candidates"]:
        candidate_id = str(candidate["id"])
        backend = str(candidate["backend"])
        compatibility = str(candidate.get("compatibility", "unverified"))
        slug = candidate_id.split("@", 1)[0].replace("/", "-")
        queue.append(
            QueueItem(
                experiment_id=f"p2-{backend}-{slug}-fit-throughput",
                phase="P2",
                backend=backend,
                model_id=model_id,
                artifact_ref=candidate_id,
                contexts=(128, 2048, 8192),
                prerequisites=("artifact-present", "runtime-compatible"),
                notes=(
                    "Full-model measurement; text-only input mode; no vision input.",
                    f"Manifest compatibility state: {compatibility}.",
                ),
            )
        )
    return queue


def write_queue_jsonl(items: Iterable[QueueItem], path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")
    return str(output)


def _connect(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(path))
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            phase TEXT NOT NULL,
            backend TEXT NOT NULL,
            model_id TEXT NOT NULL,
            artifact_ref TEXT NOT NULL,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def upsert_items(items: Iterable[Mapping[str, Any] | QueueItem], path: str | Path) -> int:
    """Persist queue/results idempotently; old rows are not deleted."""
    rows = []
    for item in items:
        payload = item.to_dict() if isinstance(item, QueueItem) else dict(item)
        status = payload.get("status")
        if status not in RESEARCH_STATUSES:
            raise ExVRAMError(f"unsupported research status: {status}")
        required = ("experiment_id", "phase", "backend", "model_id", "artifact_ref")
        if any(not isinstance(payload.get(key), str) for key in required):
            raise ExVRAMError("research row is missing identity fields")
        rows.append(
            (
                payload["experiment_id"],
                payload["phase"],
                payload["backend"],
                payload["model_id"],
                payload["artifact_ref"],
                status,
                json.dumps(payload, sort_keys=True),
                _utc_now(),
            )
        )
    if not rows:
        return 0
    db = Path(path)
    db.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect(db)
    try:
        connection.executemany(
            """
            INSERT INTO experiments
              (experiment_id, phase, backend, model_id, artifact_ref, status,
               payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(experiment_id) DO UPDATE SET
              phase=excluded.phase,
              backend=excluded.backend,
              model_id=excluded.model_id,
              artifact_ref=excluded.artifact_ref,
              status=excluded.status,
              payload_json=excluded.payload_json,
              updated_at=excluded.updated_at
            """,
            rows,
        )
        connection.commit()
    finally:
        connection.close()
    return len(rows)


def read_queue_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ExVRAMError(f"cannot read research queue {source}: {exc}") from exc
    result = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExVRAMError(f"invalid JSONL at {source}:{line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ExVRAMError(f"research row at line {line_number} is not an object")
        result.append(payload)
    return result


def read_db(path: str | Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    connection = _connect(path)
    try:
        rows = connection.execute(
            "SELECT payload_json FROM experiments ORDER BY experiment_id"
        ).fetchall()
    finally:
        connection.close()
    return [json.loads(row[0]) for row in rows]
