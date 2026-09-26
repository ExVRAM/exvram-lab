import unittest
from pathlib import Path

from exvram.config import load_experiment
from exvram.hardware import HardwareInfo
from exvram.optimizer.engine import (
    SearchSpace,
    generate_candidates,
    load_measurements,
    load_search_space,
    recommend_configurations,
)


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).parents[2]
        self.space = load_search_space(root / "experiments" / "search" / "rtx5060_27b_search.json")
        self.base = load_experiment(self.space.base_config)
        self.hardware = HardwareInfo(
            os="test", python="3.13", device_name="RTX 5060", vram_bytes=8 * 1024**3,
            compute_capability="12.0", cuda_available=True
        )

    def test_adaptive_plan_is_bounded_and_deterministic(self):
        first = generate_candidates(self.space, records=())
        second = generate_candidates(self.space, records=())
        self.assertEqual(first, second)
        self.assertEqual(len(first), self.space.max_candidates)
        self.assertEqual(len({item.candidate_id for item in first}), len(first))
        self.assertTrue(all(item.context_tokens in self.space.contexts for item in first))
        self.assertIn(2048, {item.context_tokens for item in first})
        self.assertIn(8192, {item.context_tokens for item in first})

    def test_old_backend_result_is_not_reused_for_new_candidate(self):
        records = load_measurements(Path(__file__).parents[2] / "experiments" / "results")
        result = recommend_configurations(self.space, self.base, self.hardware, records)
        self.assertGreater(result["candidates_considered"], 0)
        fastest = result["recommendations"]["FASTEST"]
        self.assertEqual(fastest["status"], "NO_MEASURED_TARGET_RESULT")
        self.assertIsNone(fastest["candidate"])

    def test_exact_candidate_identity_unlocks_measured_recommendation(self):
        raw = {
            "search_id": "test-search",
            "model_id": "Qwen3.8-27B",
            "base_config": "unused.json",
            "backends": ["llamacpp"],
            "quant_bits": [2.0],
            "contexts": [128],
            "target_context": 128,
            "kv_precisions": [
                {"name": "q4", "bytes_per_element": 0.5, "backends": ["llamacpp"]}
            ],
            "residency": [{"name": "full_gpu", "cpu_offload_mib": 0}],
            "runtime_profiles": [
                {"name": "default", "backends": ["llamacpp"], "options": {}}
            ],
        }
        space = SearchSpace.from_mapping(raw)
        candidate = generate_candidates(space, stage="screen")[0]
        records = [
            {
                "benchmark_kind": "full_model_shootout",
                "backend": "llamacpp",
                "model_id": "Qwen3.8-27B",
                "context_tokens": 128,
                "search_candidate_id": candidate.candidate_id,
                "status": "PASS",
                "measurement_status": "measured",
                "metrics": {"decode_tokens_per_second": 4.2},
                "quality_gate": {"status": "NOT_RUN"},
            }
        ]
        result = recommend_configurations(space, self.base, self.hardware, records)
        fastest = result["recommendations"]["FASTEST"]
        self.assertEqual(fastest["status"], "MEASURED")
        self.assertEqual(fastest["evidence"]["decode_tokens_per_second"], 4.2)


if __name__ == "__main__":
    unittest.main()
