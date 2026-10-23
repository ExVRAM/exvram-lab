import unittest

from exvram.diagnostic_gate import diagnose


class DiagnosticGateTests(unittest.TestCase):
    def test_a_dead_category_is_a_gross_failure_and_not_a_proof(self):
        rows = [
            {"record_type": "sample", "category": "math", "correct": False, "prediction": "no"}
            for _ in range(10)
        ]
        report = diagnose(rows)
        self.assertEqual(report["status"], "GROSS_FAIL")
        self.assertEqual(report["dead_categories"], ["math"])
        self.assertFalse(report["proves_quality"])

    def test_mixed_rows_are_not_a_gross_failure(self):
        rows = [
            {"category": "math", "correct": True, "prediction": "#### 1"},
            {"category": "math", "correct": False, "prediction": "#### 2"},
        ]
        report = diagnose(rows)
        self.assertEqual(report["status"], "NO_GROSS_FAILURE")
        self.assertEqual(report["scores"]["math"], {"correct": 1, "n": 2})
