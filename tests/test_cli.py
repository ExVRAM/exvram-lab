import io
import json
import unittest
from contextlib import redirect_stdout

from exvram.cli import main


class CLITests(unittest.TestCase):
    def test_list_adapters_is_json(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["list-adapters"]), 0)
        payload = json.loads(output.getvalue())
        names = {item["name"] for item in payload["adapters"]}
        self.assertEqual(names, {"exllamav3", "gemlite", "cutlass", "bitnet", "llamacpp"})

    def test_plan_experiment_contains_memory_and_adapter_plans(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["plan-experiment"]), 0)
        payload = json.loads(output.getvalue())
        self.assertIn("memory_budget", payload)
        self.assertEqual(len(payload["adapter_plans"]), 5)

    def test_compare_memory_configurations_contains_target_scenarios(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["compare-memory-configurations"]), 0)
        payload = json.loads(output.getvalue())
        scenario = next(
            item
            for item in payload["combined_scenarios"]
            if item["id"] == "kv-q4+reuse-scratch-buffers"
        )
        self.assertTrue(scenario["reaches_target_peak"])

    def test_layer_benchmark_does_not_fallback_to_cpu(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                main(
                    [
                        "run-layer-benchmark",
                        "--backend",
                        "torch-fp16",
                        "--layer",
                        "attention_q_proj",
                        "--warmup",
                        "0",
                        "--repeats",
                        "1",
                    ]
                ),
                0,
            )
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["results"][0]["benchmark_kind"], "layer_microbenchmark")
        self.assertFalse(payload["results"][0]["synthetic"])

    def test_layer_benchmark_accepts_prefill_batch_sizes(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                main(
                    [
                        "run-layer-benchmark",
                        "--backend",
                        "torch-fp16",
                        "--layer",
                        "attention_q_proj",
                        "--warmup",
                        "0",
                        "--repeats",
                        "1",
                        "--batch-size",
                        "1",
                        "--batch-size",
                        "8",
                    ]
                ),
                0,
            )
        payload = json.loads(output.getvalue())
        self.assertEqual([item["batch_size"] for item in payload["results"]], [1, 8])

    def test_exl3_reports_cuda_unavailable_without_fabricating_timing(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                main(
                    [
                        "run-layer-benchmark",
                        "--backend",
                        "exl3",
                        "--layer",
                        "attention_q_proj",
                        "--warmup",
                        "0",
                        "--repeats",
                        "1",
                    ]
                ),
                0,
            )
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["results"][0]["measurement_status"], "unavailable")
        self.assertEqual(payload["results"][0]["metrics"], {})
