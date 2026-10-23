"""Read stored P9 rows. This module does not start a model."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Mapping

from .p9_eval import QUALITY_PASS, aggregate_score

_NUMBER = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")
LONG_LENGTHS = (2048, 4096, 8192)
LONG_PER_LENGTH = 15


def _gold_number(answer: str) -> str | None:
    if "####" not in answer:
        return None
    match = _NUMBER.search(answer.split("####", 1)[1])
    return match.group(0).replace(",", "") if match else None


def _first_marker(text: str) -> str | None:
    if "####" not in text:
        return None
    match = _NUMBER.search(text.split("####", 1)[1])
    return match.group(0).replace(",", "") if match else None


def _repeated_line(text: str) -> int:
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 40]
    if not lines:
        return 0
    return max(Counter(lines).values())


def math_failure_class(row: Mapping[str, Any]) -> str:
    """Label one stored GSM8K row. A correct row is ``ok``."""
    if row.get("correct"):
        return "ok"
    prediction = str(row.get("prediction") or "")
    timings = row.get("timings") or {}
    decoding = row.get("decoding") or {}
    capped = int(timings.get("predicted_n") or 0) >= int(decoding.get("n_predict") or 0) > 0
    gold = _gold_number(str(row.get("gold") or ""))
    first = _first_marker(prediction)
    if first is not None and first == gold:
        return "marker_present_scorer_miss"
    if _repeated_line(prediction) >= 3:
        return "repetition"
    if "####" not in prediction and capped:
        return "cap_without_marker"
    if "####" not in prediction:
        return "finished_without_marker"
    if capped:
        return "cap_with_wrong_marker"
    return "finished_wrong_number"


def count_classes(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("category") != "math":
            continue
        label = math_failure_class(row)
        counts[label] = counts.get(label, 0) + 1
    return counts


def minimum_long_correct(
    finished: Mapping[str, float],
    reference_aggregate: float,
    long_n: int = 45,
) -> int | None:
    """Smallest long-context correct count that meets the locked aggregate ratio."""
    if long_n <= 0 or reference_aggregate <= 0:
        return None
    floor = QUALITY_PASS["aggregate_min_ratio"] * reference_aggregate
    keys = (
        "reasoning_score",
        "math_score",
        "coding_score",
        "factual_score",
        "instruction_score",
    )
    for correct in range(long_n + 1):
        scores = {key: finished[key] for key in keys}
        scores["long_context_score"] = correct / long_n
        aggregate = aggregate_score(scores)
        if aggregate is not None and aggregate >= floor:
            return correct
    return None


def missing_long_context(sample_ids: set[str]) -> list[str]:
    missing = [
        f"needle:{length}:{index}"
        for length in LONG_LENGTHS
        for index in range(LONG_PER_LENGTH)
        if f"needle:{length}:{index}" not in sample_ids
    ]
    return missing
