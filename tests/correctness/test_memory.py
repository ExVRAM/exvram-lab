import unittest

from exvram.config import default_experiment
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
