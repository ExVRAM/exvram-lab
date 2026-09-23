"""Quality-gate scaffolding; actual model evaluation is deliberately not faked."""

from __future__ import annotations

from typing import Any


def not_run_quality_gate() -> dict[str, Any]:
    return {
        "status": "not_run",
        "reason": "No reference model and evaluation corpus were supplied.",
        "required_evidence": [
            "same tokenizer and prompt set for candidate and reference",
            "fixed seed and decoding parameters",
            "perplexity or task metrics with a documented corpus",
            "acceptance threshold selected before looking at candidate results",
        ],
    }


def evaluate_quality_gate(
    *, candidate: dict[str, float], reference: dict[str, float], max_delta: float
) -> dict[str, Any]:
    """Compare named scalar metrics; missing metrics fail closed."""
    missing = sorted(set(reference) - set(candidate))
    if missing:
        return {"status": "fail", "reason": "missing candidate metrics", "missing": missing}
    deltas = {name: candidate[name] - value for name, value in reference.items()}
    failures = {name: delta for name, delta in deltas.items() if abs(delta) > max_delta}
    return {
        "status": "fail" if failures else "pass",
        "max_delta": max_delta,
        "deltas": deltas,
        "failures": failures,
    }
