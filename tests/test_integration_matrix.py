import json
import unittest
from pathlib import Path

from exvram.config import load_experiment


class MatrixTests(unittest.TestCase):
    def test_baseline_matrix_has_five_weight_points(self):
        path = Path(__file__).parents[1] / "experiments" / "weights" / "baseline_27b_8gb.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["experiments"]), 5)
        self.assertEqual(load_experiment(str(path), 0).context_tokens, 8192)

