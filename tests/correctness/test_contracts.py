import unittest

from exvram.config import default_experiment
from exvram.errors import ExVRAMError


class ContractTests(unittest.TestCase):
    def test_default_config_round_trips_to_json_shape(self):
        config = default_experiment()
        payload = config.to_dict()
        self.assertEqual(payload["id"], "default-27b-8gb-2bit")
        self.assertEqual(payload["model"]["name"], "dense-27b-placeholder")
        self.assertIsInstance(payload["candidate_adapters"], list)

    def test_invalid_context_is_rejected(self):
        raw = default_experiment().to_dict()
        raw["context_tokens"] = 0
        with self.assertRaises(ExVRAMError):
            type(default_experiment()).from_mapping(raw)

