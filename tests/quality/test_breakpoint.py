import unittest

from exvram.breakpoint import (
    count_classes,
    math_failure_class,
    minimum_long_correct,
    missing_long_context,
)
from exvram.p9_eval import aggregate_score, quality_verdict


def _row(prediction: str, gold: str, correct: bool, predicted_n: int = 100, n_predict: int = 768):
    return {
        "category": "math",
        "prediction": prediction,
        "gold": gold,
        "correct": correct,
        "timings": {"predicted_n": predicted_n},
        "decoding": {"n_predict": n_predict},
    }


class BreakpointTests(unittest.TestCase):
    def test_repetition_is_not_called_a_right_answer(self):
        gold = "working\n#### 4"
        prediction = "We need to determine the number of inches in the mosaic.\n" * 4
        label = math_failure_class(_row(prediction, gold, False, predicted_n=768))
        self.assertEqual(label, "repetition")

    def test_a_cap_without_a_marker_is_its_own_class(self):
        label = math_failure_class(_row("no marker here", "#### 4", False, predicted_n=768))
        self.assertEqual(label, "cap_without_marker")

    def test_a_finished_wrong_number_stays_a_model_miss(self):
        label = math_failure_class(_row("#### 5", "#### 4", False, predicted_n=20))
        self.assertEqual(label, "finished_wrong_number")

    def test_first_marker_matching_gold_is_not_a_scorer_miss_when_marked_wrong(self):
        label = math_failure_class(_row("#### 4\nlater #### 9", "#### 4", False, predicted_n=30))
        self.assertEqual(label, "marker_present_scorer_miss")

    def test_counts_and_missing_needles(self):
        counts = count_classes([_row("#### 4", "#### 4", True)])
        self.assertEqual(counts, {"ok": 1})
        missing = missing_long_context({"needle:2048:0"})
        self.assertEqual(len(missing), 44)
        self.assertNotIn("needle:2048:0", missing)
        self.assertIn("needle:8192:14", missing)

    def test_q4_aggregate_floor_needs_36_of_45(self):
        finished = {
            "reasoning_score": 29 / 60,
            "math_score": 71 / 100,
            "coding_score": 12 / 40,
            "factual_score": 67 / 100,
            "instruction_score": 41 / 50,
        }
        reference = {
            "reasoning_score": 29 / 60,
            "math_score": 81 / 100,
            "coding_score": 13 / 40,
            "factual_score": 72 / 100,
            "instruction_score": 43 / 50,
            "long_context_score": 1.0,
        }
        needed = minimum_long_correct(finished, aggregate_score(reference))
        self.assertEqual(needed, 36)
        for correct, verdict in ((35, "FAIL"), (36, "PASS")):
            candidate = dict(finished)
            candidate["long_context_score"] = correct / 45
            self.assertEqual(quality_verdict(candidate, reference), verdict)
