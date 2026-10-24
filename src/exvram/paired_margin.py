"""Paired P9 intervals. This module does not start a model."""

from __future__ import annotations

import math
import random
from typing import Mapping

from .p9_eval import QUALITY_PASS, aggregate_score, quality_verdict

CATEGORIES = (
    "reasoning",
    "math",
    "coding",
    "factual",
    "instruction",
    "long_context",
)


def _ratio(category: str) -> float:
    if category == "reasoning":
        return float(QUALITY_PASS["reasoning_min_ratio"])
    if category == "coding":
        return float(QUALITY_PASS["coding_min_ratio"])
    return float(QUALITY_PASS["category_min_ratio"])


def margin_call(low: float, high: float, threshold: float) -> str:
    if low >= threshold:
        return "INSIDE_MARGIN"
    if high < threshold:
        return "OUTSIDE_MARGIN"
    return "STRADDLES_MARGIN"


def wald_interval(pairs: list[tuple[bool, bool]]) -> tuple[float, float, float]:
    count = len(pairs)
    if count == 0:
        raise ValueError("paired flags must not be empty")
    gained = sum(1 for cand, ref in pairs if cand and not ref)
    lost = sum(1 for cand, ref in pairs if ref and not cand)
    diff = (sum(cand for cand, _ref in pairs) - sum(ref for _cand, ref in pairs)) / count
    variance = ((gained + lost) - (gained - lost) ** 2 / count) / count**2
    margin = 1.96 * math.sqrt(variance)
    return diff, diff - margin, diff + margin


def bootstrap_aggregate_delta(
    groups: list[list[tuple[bool, bool]]],
    *,
    draws: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """2.5 and 97.5 percentiles. Seed 42 matches the stored Q8 aggregate low."""
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(draws):
        candidate: list[float] = []
        reference: list[float] = []
        for pairs in groups:
            count = len(pairs)
            drawn = [pairs[rng.randrange(count)] for _ in range(count)]
            candidate.append(sum(item[0] for item in drawn) / count)
            reference.append(sum(item[1] for item in drawn) / count)
        deltas.append(sum(candidate) / len(groups) - sum(reference) / len(groups))
    deltas.sort()
    return deltas[249], deltas[9749]


def paired_study(
    reference_rows: list[Mapping[str, object]],
    candidate_rows: list[Mapping[str, object]],
) -> dict[str, object]:
    reference = _by_category(reference_rows)
    candidate = _by_category(candidate_rows)
    groups: list[list[tuple[bool, bool]]] = []
    categories: dict[str, dict[str, object]] = {}
    mismatches = 0
    ref_scores: dict[str, float] = {}
    cand_scores: dict[str, float] = {}
    for name in CATEGORIES:
        ids = sorted(set(reference[name]) & set(candidate[name]))
        pairs: list[tuple[bool, bool]] = []
        for sample_id in ids:
            left = reference[name][sample_id]
            right = candidate[name][sample_id]
            if left.get("prompt_sha256") != right.get("prompt_sha256"):
                mismatches += 1
            elif left.get("gold") != right.get("gold"):
                mismatches += 1
            pairs.append((bool(right["correct"]), bool(left["correct"])))
        groups.append(pairs)
        ref_correct = sum(item[1] for item in pairs)
        cand_correct = sum(item[0] for item in pairs)
        count = len(pairs)
        ref_scores[f"{name}_score"] = ref_correct / count
        cand_scores[f"{name}_score"] = cand_correct / count
        diff, low, high = wald_interval(pairs)
        threshold = (_ratio(name) - 1.0) * ref_scores[f"{name}_score"]
        categories[name] = {
            "n": count,
            "reference_correct": ref_correct,
            "candidate_correct": cand_correct,
            "delta": diff,
            "low": low,
            "high": high,
            "threshold": threshold,
            "call": margin_call(low, high, threshold),
        }
    ref_aggregate = aggregate_score(ref_scores)
    cand_aggregate = aggregate_score(cand_scores)
    if ref_aggregate is None or cand_aggregate is None:
        raise ValueError("aggregate requires all six categories")
    low, high = bootstrap_aggregate_delta(groups)
    threshold = (float(QUALITY_PASS["aggregate_min_ratio"]) - 1.0) * ref_aggregate
    calls = [str(item["call"]) for item in categories.values()]
    calls.append(margin_call(low, high, threshold))
    if "OUTSIDE_MARGIN" in calls:
        study = "OUTSIDE_MARGIN"
    elif "STRADDLES_MARGIN" in calls:
        study = "INCONCLUSIVE"
    else:
        study = "INSIDE_MARGIN"
    return {
        "prompt_or_gold_mismatches": mismatches,
        "reference_aggregate": ref_aggregate,
        "candidate_aggregate": cand_aggregate,
        "point_verdict": quality_verdict(cand_scores, ref_scores),
        "aggregate_low": low,
        "aggregate_high": high,
        "aggregate_threshold": threshold,
        "aggregate_call": margin_call(low, high, threshold),
        "study_verdict": study,
        "categories": categories,
    }


def _by_category(
    rows: list[Mapping[str, object]],
) -> dict[str, dict[str, Mapping[str, object]]]:
    grouped: dict[str, dict[str, Mapping[str, object]]] = {name: {} for name in CATEGORIES}
    for row in rows:
        if row.get("record_type") == "summary" or not row.get("sample_id"):
            continue
        category = str(row.get("category"))
        if category in grouped:
            grouped[category][str(row["sample_id"])] = row
    return grouped
