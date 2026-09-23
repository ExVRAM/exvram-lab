import unittest

from exvram.adapters.llamacpp import build_llama_bench_command


class AdapterBoundaryTests(unittest.TestCase):
    def test_llama_bench_command_uses_jsonl_and_gpu_offload_flags(self):
        command = build_llama_bench_command("llama-bench", "model.gguf")
        self.assertIn("-o", command)
        self.assertIn("jsonl", command)
        self.assertIn("-ngl", command)

