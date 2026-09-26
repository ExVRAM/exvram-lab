import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_llamacpp_runner():
    path = Path(__file__).parents[2] / "benchmark" / "run_llamacpp_p2.py"
    spec = importlib.util.spec_from_file_location("run_llamacpp_p2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_exllama_runner():
    path = Path(__file__).parents[2] / "benchmark" / "run_exllama_p2.py"
    spec = importlib.util.spec_from_file_location("run_exllama_p2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FullModelRunnerTests(unittest.TestCase):
    def test_llamacpp_perf_lines_are_parsed_into_explicit_metrics(self):
        runner = _load_llamacpp_runner()
        output = """
        llama_perf_context_print: prompt eval time = 125.00 ms / 128 tokens
        llama_perf_context_print: eval time = 800.00 ms / 32 runs
        offloaded 64 / 64 layers to GPU
        CUDA0 buffer size = 7012.50 MiB
        """

        metrics = runner._parse_metrics(output)

        self.assertEqual(metrics["prompt_tokens"], 128)
        self.assertEqual(metrics["decode_tokens"], 32)
        self.assertAlmostEqual(metrics["prefill_tokens_per_second"], 1024.0)
        self.assertAlmostEqual(metrics["decode_tokens_per_second"], 40.0)
        self.assertEqual(metrics["gpu_layers"], 64)
        self.assertEqual(metrics["total_layers"], 64)
        self.assertEqual(metrics["reported_cuda_buffers"][0]["mib"], 7012.5)

    def test_llamacpp_b10964_rate_summary_is_parsed(self):
        runner = _load_llamacpp_runner()

        metrics = runner._parse_metrics("[ Prompt: 38.0 t/s | Generation: 4.3 t/s ]")

        self.assertEqual(metrics["prefill_tokens_per_second"], 38.0)
        self.assertEqual(metrics["decode_tokens_per_second"], 4.3)

    def test_llamacpp_fit_memory_breakdown_is_normalized(self):
        runner = _load_llamacpp_runner()
        output = (
            'Loading model... {"type":"fit_memory_breakdown","data":'
            '{"unit":"MiB","rows":[{"kind":"device","name":"CUDA0",'
            '"description":"RTX 5060","total":8150,"free":744,"self":5857,'
            '"model":5673,"context":112,"compute":71,"unaccounted":1549}]}'
            '}'
        )

        metrics = runner._parse_metrics(output)

        self.assertEqual(metrics["fit_memory_breakdown"]["unit"], "MiB")
        row = metrics["fit_memory_breakdown"]["rows"][0]
        self.assertEqual(row["description"], "RTX 5060")
        self.assertEqual(row["free"], 744)

    def test_llamacpp_runner_blocks_storage_before_model_stat(self):
        runner = _load_llamacpp_runner()
        args = runner._parser().parse_args(
            [
                "--binary",
                r"R:\tools\llama-cli.exe",
                "--model",
                r"R:\models\model.gguf",
                "--context",
                "128",
                "--output",
                r"PROJECT:\blocked.jsonl",
                "--search-candidate-id",
                "rtx5060-27b-llamacpp-w2.0-c128-kv-fp16-residency-full_gpu-runtime-default",
            ]
        )

        with patch.object(
            runner,
            "assert_storage_safe",
            side_effect=runner.StorageSafetyError("removable storage"),
        ):
            record = runner.run(args)

        self.assertEqual(record["status"], "INCONCLUSIVE")
        self.assertEqual(record["measurement_status"], "blocked")
        self.assertEqual(record["error_type"], "StorageSafetyError")
        self.assertEqual(
            record["search_candidate_id"],
            "rtx5060-27b-llamacpp-w2.0-c128-kv-fp16-residency-full_gpu-runtime-default",
        )

    def test_exllama_record_preserves_exact_search_candidate_identity(self):
        runner = _load_exllama_runner()
        args = runner._parser().parse_args(
            [
                "--model-dir",
                r"PROJECT:\models\qwen",
                "--context",
                "8192",
                "--output",
                r"PROJECT:\results.jsonl",
                "--search-candidate-id",
                "rtx5060-27b-exllamav3-w4.0-c8192-kv-q4-residency-edge_offload-runtime-cache_tuned",
            ]
        )

        record = runner._record_base(args)

        self.assertEqual(
            record["search_candidate_id"],
            "rtx5060-27b-exllamav3-w4.0-c8192-kv-q4-residency-edge_offload-runtime-cache_tuned",
        )


if __name__ == "__main__":
    unittest.main()
