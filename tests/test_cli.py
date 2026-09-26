import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from exvram.cli import main
from exvram.hardware import HardwareInfo


class CLITests(unittest.TestCase):
    def test_list_adapters_is_json(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["list-adapters"]), 0)
        payload = json.loads(output.getvalue())
        names = {item["name"] for item in payload["adapters"]}
        self.assertEqual(
            names, {"exllamav3", "gemlite", "cutlass", "bitnet", "llamacpp", "ollama"}
        )

    def test_plan_experiment_contains_memory_and_adapter_plans(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["plan-experiment"]), 0)
        payload = json.loads(output.getvalue())
        self.assertIn("memory_budget", payload)
        self.assertEqual(len(payload["adapter_plans"]), 5)

    def test_plan_search_is_bounded_and_machine_readable(self):
        output = io.StringIO()
        config = str(Path("experiments/search/rtx5060_27b_search.json"))
        hardware = HardwareInfo(os="test", python="3.13", device_name="RTX 5060")
        with patch("exvram.cli.detect_hardware", return_value=hardware), redirect_stdout(output):
            self.assertEqual(main(["plan-search", "--config", config, "--stage", "screen"]), 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["benchmark_kind"], "adaptive_configuration_search_plan")
        self.assertGreater(payload["candidate_count"], 0)
        self.assertEqual(payload["status"], "PLANNED")

    def test_run_ollama_writes_external_runtime_record(self):
        fake_record = {
            "backend": "ollama",
            "model_id": "qwen2.5-coder:7b",
            "status": "PASS",
            "measurement_status": "measured",
            "synthetic": False,
        }
        output = io.StringIO()
        with patch("exvram.cli.run_ollama", return_value=fake_record) as run, redirect_stdout(
            output
        ):
            self.assertEqual(main(["run-ollama", "--model", "qwen2.5-coder:7b"]), 0)
        run.assert_called_once()
        self.assertEqual(json.loads(output.getvalue())["backend"], "ollama")

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
