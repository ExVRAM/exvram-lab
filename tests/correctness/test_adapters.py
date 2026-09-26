import unittest
from unittest.mock import Mock, patch

from exvram.adapters.llamacpp import build_llama_bench_command
from exvram.adapters.ollama import (
    build_ollama_run_command,
    parse_ollama_list,
    parse_ollama_verbose,
    run_ollama,
)


class AdapterBoundaryTests(unittest.TestCase):
    def test_llama_bench_command_uses_jsonl_and_gpu_offload_flags(self):
        command = build_llama_bench_command("llama-bench", "model.gguf")
        self.assertIn("-o", command)
        self.assertIn("jsonl", command)
        self.assertIn("-ngl", command)

    def test_ollama_command_does_not_add_download_behavior(self):
        command = build_ollama_run_command(
            "ollama", "qwen2.5-coder:7b", "Reply briefly", keepalive="5m"
        )
        self.assertEqual(command[:4], ["ollama", "run", "qwen2.5-coder:7b", "Reply briefly"])
        self.assertIn("--verbose", command)
        self.assertNotIn("pull", command)

    def test_ollama_list_and_verbose_output_are_parsed(self):
        models = parse_ollama_list(
            "NAME              ID              SIZE      MODIFIED\n"
            "qwen2.5-coder:7b dae161e27b0e    4.7 GB    7 days ago\n"
        )
        metrics = parse_ollama_verbose(
            "prompt eval count: 10 token(s)\n"
            "prompt eval duration: 500ms\n"
            "eval count: 20 token(s)\n"
            "eval duration: 2s\n"
        )
        self.assertEqual(models, ("qwen2.5-coder:7b",))
        self.assertEqual(metrics["prefill_tokens_per_second"], 20.0)
        self.assertEqual(metrics["decode_tokens_per_second"], 10.0)

    def test_ollama_does_not_download_missing_model(self):
        listed = Mock(returncode=0, stdout="NAME ID SIZE MODIFIED\n", stderr="")
        with patch("exvram.adapters.ollama.subprocess.run", return_value=listed) as run:
            record = run_ollama("missing:model", "Reply briefly", executable="ollama")
        self.assertEqual(record["status"], "INCONCLUSIVE")
        self.assertEqual(record["measurement_status"], "not_installed")
        self.assertEqual(run.call_count, 1)
