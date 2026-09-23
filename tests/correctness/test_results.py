import json
import tempfile
import unittest
from pathlib import Path

from exvram.config import default_experiment
from exvram.hardware import detect_hardware
from exvram.results import new_result, write_result


class ResultWriterTests(unittest.TestCase):
    def test_jsonl_appends_one_parseable_record(self):
        record = new_result(
            config=default_experiment(),
            hardware=detect_hardware(),
            benchmark_kind="test",
            synthetic=True,
            metrics={"value": 1},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            write_result(record, str(path))
            write_result(record, str(path))
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["benchmark_kind"], "test")

