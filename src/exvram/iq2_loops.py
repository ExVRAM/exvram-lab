"""Classify stored IQ2 repetition. This module does not start a model."""

from __future__ import annotations

from collections import Counter
from typing import Mapping

PROBE_IDS = ("gsm8k:807", "gsm8k:548", "gsm8k:450", "gsm8k:48")
CONTROL_IDS = (
    "gsm8k:52",
    "gsm8k:400",
    "gsm8k:848",
    "gsm8k:849",
    "gsm8k:923",
    "gsm8k:1085",
    "logical_deduction_three_objects:71",
    "logical_deduction_three_objects:212",
    "logical_deduction_three_objects:213",
)


def twelve_gram(text: str) -> tuple[str, int]:
    words = text.split()
    if len(words) < 12:
        return "", 0
    counts = Counter(tuple(words[i : i + 12]) for i in range(len(words) - 11))
    phrase, count = counts.most_common(1)[0]
    return " ".join(phrase), count


def loop_shape(text: str) -> str:
    """`short_cycle` is a 12-word window with at most three distinct words."""
    phrase, count = twelve_gram(text)
    if count < 4:
        return "no_12gram_loop"
    if len(set(phrase.split())) <= 3:
        return "short_cycle"
    return "repeated_phrase"


def math_loop_counts(rows: list[Mapping[str, object]]) -> dict[str, int]:
    counts = {"wrong": 0, "short_cycle": 0, "repeated_phrase": 0, "no_12gram_loop": 0}
    for row in rows:
        if row.get("category") != "math" or row.get("correct"):
            continue
        counts["wrong"] += 1
        counts[loop_shape(str(row.get("prediction") or ""))] += 1
    return counts


def gsm8k_prompt(question: str) -> str:
    return (
        question.rstrip()
        + "\nSolve the problem. Put the final number on its own line as #### <number>."
    )


def probe_plan() -> dict[str, object]:
    """One-factor reproduction. The stored rows are the comparison, not this plan."""
    overlap = sorted(set(PROBE_IDS) & set(CONTROL_IDS))
    return {
        "launch": False,
        "factor": "none_on_the_first_rerun",
        "probe_ids": PROBE_IDS,
        "control_ids": CONTROL_IDS,
        "overlap": overlap,
        "arms_in_order": (
            "qwen35_2b_iq2",
            "qwen35_2b_bf16",
            "qwen35_2b_q8_0_gpu",
            "qwen35_2b_q4_k_m",
        ),
        "output": "experiments/results/p9_iq2_probe.jsonl",
        "reuse_scored_files": False,
        "change_together": (),
    }
