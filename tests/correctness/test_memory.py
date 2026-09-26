import unittest
from pathlib import Path

from exvram.config import default_experiment, load_experiment
from exvram.memory import compare_memory_configurations, estimate_memory


class MemoryTests(unittest.TestCase):
    def test_kv_estimate_is_explicit_and_positive(self):
        budget = estimate_memory(default_experiment())
        self.assertAlmostEqual(budget.kv_cache_gib, 1.0, places=3)
        self.assertGreater(budget.raw_packed_weights_gib, 6.0)
        self.assertGreater(budget.estimated_weight_storage_gib, budget.raw_packed_weights_gib)

    def test_more_weight_bits_never_reduce_modeled_weight_storage(self):
        low = default_experiment()
        high = type(low)(**{**low.__dict__, "weight_bits": 4.0})
        self.assertGreater(
            estimate_memory(high).estimated_weight_storage_gib,
            estimate_memory(low).estimated_weight_storage_gib,
        )

    def test_detailed_components_reproduce_the_legacy_total(self):
        budget = estimate_memory(default_experiment())
        names = {component.name for component in budget.components}
        self.assertIn("cuda_context", names)
        self.assertIn("kv_cache_full_attention", names)
        self.assertAlmostEqual(budget.modeled_total_gib, budget.legacy_modeled_total_gib, places=9)

    def test_q4_kv_plus_scratch_reuse_crosses_the_planning_target(self):
        comparison = compare_memory_configurations(default_experiment())
        scenarios = {item["id"]: item for item in comparison["combined_scenarios"]}
        self.assertLessEqual(scenarios["kv-q4+reuse-scratch-buffers"]["new_total_gib"], 7.5)
        self.assertTrue(scenarios["kv-q4+reuse-scratch-buffers"]["reaches_target_peak"])

    def test_missing_manifest_rows_are_unavailable_and_budget_is_inconclusive(self):
        budget = estimate_memory(default_experiment())
        by_name = {component.name: component.evidence_status for component in budget.components}
        self.assertEqual(by_name["packed_transformer_weights"], "modeled")
        self.assertEqual(by_name["quantization_scales"], "modeled")
        self.assertEqual(by_name["kv_cache_full_attention"], "modeled")
        self.assertEqual(by_name["runtime_scratch_workspace"], "modeled")
        self.assertEqual(by_name["input_embeddings"], "unavailable")
        self.assertEqual(by_name["lm_head"], "unavailable")
        self.assertEqual(by_name["norms_biases_small_fp16"], "unavailable")
        self.assertEqual(by_name["codebooks_metadata"], "unavailable")
        self.assertNotIn("measured", by_name.values())
        self.assertEqual(budget.evidence_status, "INCONCLUSIVE")
        counts = budget.to_dict()["evidence_status_counts"]
        self.assertEqual(sum(counts.values()), len(budget.components))

    def test_supplied_manifest_rows_stay_modeled(self):
        base = default_experiment()
        model = type(base.model)(
            **{
                **base.model.__dict__,
                "input_embedding_params_b": 0.5,
                "lm_head_params_b": 0.5,
                "small_tensors_mib": 32,
                "codebook_mib": 8,
            }
        )
        budget = estimate_memory(type(base)(**{**base.__dict__, "model": model}))
        self.assertTrue(
            all(component.evidence_status == "modeled" for component in budget.components)
        )
        self.assertEqual(budget.evidence_status, "modeled")
        comparison = compare_memory_configurations(
            type(base)(**{**base.__dict__, "model": model})
        )
        self.assertEqual(comparison["evidence_status"], "modeled")
        self.assertEqual(comparison["options"][0]["evidence_status"], "modeled")

    def test_qwen38_bf16_weight_map_matches_published_checkpoint_bytes(self):
        config = load_experiment(
            str(Path(__file__).parents[2] / "experiments" / "weights" / "qwen38_27b_bf16.json"),
            0,
        )
        budget = estimate_memory(config)
        by_name = {component.name: component for component in budget.components}
        self.assertEqual(by_name["input_embeddings"].evidence_status, "modeled")
        self.assertEqual(by_name["lm_head"].evidence_status, "modeled")
        self.assertEqual(by_name["norms_biases_small_fp16"].evidence_status, "modeled")
        self.assertEqual(by_name["codebooks_metadata"].evidence_status, "modeled")
        self.assertEqual(by_name["codebooks_metadata"].modeled_bytes, 0)
        self.assertEqual(by_name["kv_cache_linear_attention"].evidence_status, "unavailable")
        self.assertEqual(budget.evidence_status, "INCONCLUSIVE")
        weight_bytes = sum(
            component.modeled_bytes
            for component in budget.components
            if component.category in {"weights", "small_tensors", "quantization_metadata"}
        )
        self.assertEqual(weight_bytes, 55562855904)
