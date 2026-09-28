import json
import unittest
from pathlib import Path

from exvram.p8_binary import (
    P8MatrixShape,
    build_p8_plan,
    grouped_footprint,
    load_manifest,
    pbllm_footprint,
    quality_metrics,
    representative_tensor,
)


class P8BinaryTests(unittest.TestCase):
    def test_qwen_shapes_and_physical_group_overhead(self):
        shape = P8MatrixShape("q_proj", 5120, 6144, "attention")
        self.assertEqual(shape.parameter_count, 5120 * 6144)
        footprints = [grouped_footprint(shape, 1, group) for group in (32, 64, 128)]
        self.assertGreater(footprints[0].physical_bpw, footprints[1].physical_bpw)
        self.assertGreater(footprints[1].physical_bpw, footprints[2].physical_bpw)
        self.assertAlmostEqual(footprints[0].physical_bpw, 2.0, places=6)
        self.assertAlmostEqual(footprints[1].physical_bpw, 1.5, places=6)
        self.assertAlmostEqual(footprints[2].physical_bpw, 1.25, places=6)
        self.assertGreater(footprints[1].scale_bytes, 0)
        self.assertGreater(footprints[1].zero_bytes, 0)

    def test_pbllm_footprint_counts_exceptions_and_metadata(self):
        shape = P8MatrixShape("down_proj", 17408, 5120, "mlp")
        footprint = pbllm_footprint(shape, salient_fraction=0.05, high_bits=8)
        self.assertGreater(footprint.exception_bytes, 0)
        self.assertGreater(footprint.metadata_bytes, 0)
        self.assertGreater(footprint.physical_bpw, 1.0)

    def test_quality_metrics_are_deterministic_and_bounded(self):
        reference = representative_tensor(size=32)
        identical = quality_metrics(reference, reference)
        self.assertAlmostEqual(identical["mse"], 0.0)
        self.assertAlmostEqual(identical["cosine_similarity"], 1.0)

        reconstructed = tuple(value * 0.99 for value in reference)
        metrics = quality_metrics(reference, reconstructed)
        self.assertGreater(metrics["mse"], 0.0)
        self.assertLessEqual(metrics["cosine_similarity"], 1.0)
        self.assertGreaterEqual(metrics["cosine_similarity"], 0.99)

    def test_plan_is_explicitly_unmeasured_and_uses_manifest(self):
        manifest_path = Path("experiments/manifests/p8_binary_qwen38.json")
        manifest = load_manifest(manifest_path)
        plan = build_p8_plan(manifest)
        self.assertEqual(plan["track"], "P8")
        self.assertTrue(plan["synthetic"])
        self.assertEqual(plan["measurement_status"], "PLANNED")
        self.assertEqual(plan["decision_gate"]["status"], "NOT_RUN")
        self.assertEqual(plan["model"]["id"], "Qwen3.8-27B")
        self.assertGreater(len(plan["records"]), 0)
        self.assertTrue(all(record["quality"]["status"] == "NOT_RUN" for record in plan["records"]))
        self.assertTrue(
            all(record["performance"]["status"] == "NOT_RUN" for record in plan["records"])
        )
        self.assertAlmostEqual(
            plan["model_estimates"]["pbllm_0.900_binary_high8_g64"]["physical_bpw"],
            3.05,
            places=3,
        )

        json.dumps(plan)


if __name__ == "__main__":
    unittest.main()
