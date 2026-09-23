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


if __name__ == "__main__":
    unittest.main()
