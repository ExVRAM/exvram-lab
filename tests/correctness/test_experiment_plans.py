import json
import unittest
from pathlib import Path


class ExperimentPlanTests(unittest.TestCase):
    def test_planned_curves_are_bounded_and_not_results(self):
        root = Path(__file__).parents[2]
        plans = (
            root / "experiments" / "residency" / "offload_curve.json",
            root / "experiments" / "residency" / "vram_pressure.json",
            root / "experiments" / "kv_cache" / "weight_kv_tradeoff.json",
        )
        for path in plans:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["status"], "PLANNED")
            self.assertLessEqual(payload["execution_policy"]["bounded_timeout_seconds"], 300)
            self.assertTrue(payload["execution_policy"]["stable_storage_required"])
            self.assertIn("metrics", payload)


if __name__ == "__main__":
    unittest.main()
