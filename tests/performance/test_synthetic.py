import unittest

from exvram.config import default_experiment
from exvram.hardware import detect_hardware
from exvram.synthetic import run_synthetic_microbenchmark


class SyntheticBenchmarkTests(unittest.TestCase):
    def test_result_is_marked_synthetic_and_has_no_tok_per_second_claim(self):
        result = run_synthetic_microbenchmark(
            default_experiment(), detect_hardware(), iterations=2, work_units=10
        )
        payload = result.to_dict()
        self.assertTrue(payload["synthetic"])
        self.assertEqual(payload["benchmark_kind"], "synthetic_microbenchmark")
        self.assertNotIn("tokens_per_second", payload["metrics"])
