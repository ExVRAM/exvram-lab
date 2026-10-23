"""Gross-failure check on stored P9 rows. Not a retention proof."""

from __future__ import annotations

from typing import Any, Mapping

from .breakpoint import _repeated_line


def diagnose(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    samples = [row for row in rows if row.get("record_type", "sample") == "sample"]
    empty = sum(1 for row in samples if not str(row.get("prediction") or "").strip())
    repeated = sum(1 for row in samples if _repeated_line(str(row.get("prediction") or "")) >= 3)
    by_category: dict[str, list[Mapping[str, Any]]] = {}
    for row in samples:
        category = str(row.get("category") or "")
        if not category:
            continue
        by_category.setdefault(category, []).append(row)
    scores = {}
    dead = []
    for category, group in sorted(by_category.items()):
        correct = sum(1 for row in group if row.get("correct") is True)
        scores[category] = {"correct": correct, "n": len(group)}
        if len(group) >= 10 and correct == 0:
            dead.append(category)
    gross = empty > 0 or repeated >= 10 or bool(dead)
    return {
        "samples": len(samples),
        "empty_predictions": empty,
        "repeated_predictions": repeated,
        "dead_categories": dead,
        "scores": scores,
        "status": "GROSS_FAIL" if gross else "NO_GROSS_FAILURE",
        "proves_quality": False,
    }
