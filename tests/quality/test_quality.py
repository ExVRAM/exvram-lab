import unittest

from exvram.quality import evaluate_quality_gate, not_run_quality_gate


class QualityGateTests(unittest.TestCase):
    def test_scaffold_is_not_presented_as_a_pass(self):
        self.assertEqual(not_run_quality_gate()["status"], "not_run")

    def test_missing_candidate_metric_fails_closed(self):
        result = evaluate_quality_gate(
            candidate={"ppl": 10.0}, reference={"ppl": 9.0, "task": 0.8}, max_delta=0.1
        )
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["missing"], ["task"])
