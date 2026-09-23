import tempfile
import unittest
from pathlib import Path

from exvram.errors import ExVRAMError
from exvram.research import (
    default_p2_queue,
    load_manifest,
    read_db,
    read_queue_jsonl,
    upsert_items,
    write_queue_jsonl,
)


class ResearchPersistenceTests(unittest.TestCase):
    def test_manifest_creates_stable_planned_queue(self):
        manifest = load_manifest("experiments/manifests/qwen38_p2.json")
        queue = default_p2_queue(manifest)
        self.assertGreaterEqual(len(queue), 3)
        self.assertTrue(all(item.status == "PLANNED" for item in queue))
        self.assertTrue(all("/" not in item.experiment_id for item in queue))

    def test_queue_and_sqlite_round_trip(self):
        manifest = load_manifest("experiments/manifests/qwen38_p2.json")
        queue = default_p2_queue(manifest)
        with tempfile.TemporaryDirectory() as directory:
            queue_path = Path(directory) / "queue.jsonl"
            database_path = Path(directory) / "research.sqlite"
            write_queue_jsonl(queue, queue_path)
            upsert_items(queue, database_path)
            queue_rows = read_queue_jsonl(queue_path)
            db_rows = read_db(database_path)
        self.assertEqual(len(queue_rows), len(queue))
        self.assertEqual(len(db_rows), len(queue))
        self.assertEqual({row["status"] for row in db_rows}, {"PLANNED"})

    def test_invalid_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ExVRAMError):
                upsert_items(
                    [
                        {
                            "experiment_id": "bad",
                            "phase": "P2",
                            "backend": "test",
                            "model_id": "model",
                            "artifact_ref": "artifact",
                            "status": "DONE",
                        }
                    ],
                    Path(directory) / "research.sqlite",
                )
