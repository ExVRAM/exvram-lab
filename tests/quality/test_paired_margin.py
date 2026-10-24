import json
import unittest
from pathlib import Path

from exvram.paired_margin import paired_study

ROOT = Path(__file__).resolve().parents[2] / "experiments" / "results"


def _rows(*names: str) -> list[dict]:
    rows: list[dict] = []
    for name in names:
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


class PairedMarginTests(unittest.TestCase):
    def test_q8_bootstrap_matches_the_stored_interval(self):
        stored = json.loads((ROOT / "p9_qwen35_2b_q8_gpu_paired.json").read_text(encoding="utf-8"))
        study = paired_study(
            _rows("p9_qwen35_2b_bf16.jsonl", "p9_qwen35_2b_bf16_long.jsonl"),
            _rows("p9_qwen35_2b_q8_gpu.jsonl", "p9_qwen35_2b_q8_gpu_long.jsonl"),
        )
        self.assertEqual(study["aggregate_low"], stored["aggregate_bootstrap95"][0])
        self.assertAlmostEqual(study["aggregate_high"], stored["aggregate_bootstrap95"][1])
        self.assertEqual(study["point_verdict"], "PASS")
        self.assertEqual(study["study_verdict"], "INCONCLUSIVE")

    def test_closed_q4_suite_passes_the_point_rule(self):
        study = paired_study(
            _rows("p9_qwen35_2b_bf16.jsonl", "p9_qwen35_2b_bf16_long.jsonl"),
            _rows("p9_qwen35_2b_q4.jsonl", "p9_qwen35_2b_q4_long.jsonl"),
        )
        self.assertEqual(study["prompt_or_gold_mismatches"], 0)
        categories = study["categories"]
        self.assertEqual(categories["long_context"]["candidate_correct"], 45)
        self.assertEqual(categories["math"]["call"], "INSIDE_MARGIN")
        self.assertLess(categories["math"]["high"], 0)
        self.assertEqual(categories["reasoning"]["call"], "STRADDLES_MARGIN")
        self.assertEqual(categories["coding"]["call"], "STRADDLES_MARGIN")
        self.assertEqual(study["aggregate_call"], "INSIDE_MARGIN")
        self.assertEqual(study["point_verdict"], "PASS")
        self.assertEqual(study["study_verdict"], "INCONCLUSIVE")
