import json
import unittest
from pathlib import Path

from exvram.q4_resume import original_q4_gap, paired_wald, q4_resume_plan

LONG = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "results"
    / "p9_qwen35_2b_q4_long.jsonl"
)
SHORT = {
    "bf16": "p9_qwen35_2b_bf16.jsonl",
    "q4": "p9_qwen35_2b_q4.jsonl",
}


class Q4ResumeTests(unittest.TestCase):
    def test_the_stored_gap_is_the_25_needles(self):
        gap = original_q4_gap()
        self.assertEqual(len(gap), 25)
        self.assertEqual(gap[0], "needle:4096:5")
        self.assertEqual(gap[-1], "needle:8192:14")
        self.assertNotIn("needle:4096:4", gap)

    def test_plan_never_launches_and_defers_under_the_ram_floor(self):
        ids = _historical_partial_ids()
        plan = q4_resume_plan(ids, ram_available_mib=1944)
        self.assertFalse(plan["launch"])
        self.assertEqual(plan["resource"], "defer")
        self.assertEqual(plan["missing"], list(original_q4_gap()))
        self.assertEqual(plan["outside_original_gap"], [])
        ready = q4_resume_plan(ids, ram_available_mib=4283)
        self.assertEqual(ready["resource"], "within_floor")
        self.assertFalse(ready["launch"])

    def test_finished_long_file_does_not_ask_for_another_launch(self):
        plan = q4_resume_plan(_long_ids(), ram_available_mib=4283)
        self.assertEqual(plan["missing"], [])
        self.assertEqual(plan["outside_original_gap"], [])
        self.assertFalse(plan["launch"])

    def test_q4_math_interval_is_entirely_below_zero(self):
        paired = _paired("math")
        interval = paired_wald(paired["q4"], paired["bf16"])
        self.assertLess(interval["high"], 0)
        self.assertAlmostEqual(interval["diff"], -0.1)

    def test_q4_reasoning_interval_crosses_zero(self):
        paired = _paired("reasoning")
        interval = paired_wald(paired["q4"], paired["bf16"])
        self.assertLess(interval["low"], 0)
        self.assertGreater(interval["high"], 0)
        self.assertEqual(interval["diff"], 0)


def _historical_partial_ids() -> set[str]:
    gap = set(original_q4_gap())
    return {
        f"needle:{length}:{index}"
        for length in (2048, 4096, 8192)
        for index in range(15)
        if f"needle:{length}:{index}" not in gap
    }


def _long_ids() -> set[str]:
    return {
        json.loads(line)["sample_id"]
        for line in LONG.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _paired(category: str) -> dict[str, list[bool]]:
    root = LONG.parent
    by_id: dict[str, dict[str, bool]] = {}
    for name, filename in SHORT.items():
        rows = [
            json.loads(line)
            for line in (root / filename).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            if row.get("category") != category:
                continue
            by_id.setdefault(row["sample_id"], {})[name] = bool(row["correct"])
    shared = sorted(sample_id for sample_id, flags in by_id.items() if set(flags) == set(SHORT))
    return {name: [by_id[sample_id][name] for sample_id in shared] for name in SHORT}
