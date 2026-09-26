import json
import re
import unittest
from pathlib import Path


class PublicResultTests(unittest.TestCase):
    def test_llamacpp_public_summary_has_no_host_absolute_paths(self):
        path = (
            Path(__file__).parents[2]
            / "experiments"
            / "results"
            / "p2_llamacpp_full_model_public.jsonl"
        )
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

        self.assertEqual(len(records), 1)
        record = records[0]
        serialized = json.dumps(record, sort_keys=True)
        self.assertIsNone(re.search(r"[A-Za-z]:[\\/]", serialized))
        self.assertTrue(record["storage_paths_redacted"])
        self.assertEqual(record["status"], "PASS")
        self.assertEqual(record["metrics"]["decode_tokens_per_second"], 3.8)
        self.assertEqual(record["quality_gate"]["status"], "NOT_RUN")

    def test_ollama_public_summary_is_measured_and_redacted(self):
        path = Path(__file__).parents[2] / "experiments" / "results" / "ollama_smoke_public.jsonl"
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

        self.assertEqual(len(records), 1)
        record = records[0]
        serialized = json.dumps(record, sort_keys=True)
        self.assertIsNone(re.search(r"[A-Za-z]:[\\/]", serialized))
        self.assertTrue(record["storage_paths_redacted"])
        self.assertEqual(record["status"], "PASS")
        self.assertFalse(record["synthetic"])
        self.assertEqual(record["quality_gate"]["status"], "NOT_RUN")
        self.assertEqual(record["scope"], "7B local baseline; not the dense 27B target")

    def test_iq2_decode_diag_is_offload_bound_and_has_no_paths(self):
        path = (
            Path(__file__).parents[2]
            / "experiments"
            / "results"
            / "p3_llamacpp_iq2_decode_diag.jsonl"
        )
        record = json.loads(path.read_text(encoding="utf-8"))
        serialized = json.dumps(record, sort_keys=True)
        self.assertIsNone(re.search(r"[A-Za-z]:[\\/]", serialized))
        self.assertEqual(record["classification"], "CPU_OFFLOAD_BOUND")
        self.assertFalse(record["synthetic"])
        self.assertGreater(
            record["metrics"]["model_file_bytes"],
            record["metrics"]["device_total_mib"] * 1024 * 1024,
        )
        self.assertGreater(record["metrics"]["cpu_model_mib_final_fit"], 0)
        self.assertLess(record["metrics"]["sm_util_percent_mean_while_fb_ge_6000"], 15)


if __name__ == "__main__":
    unittest.main()
