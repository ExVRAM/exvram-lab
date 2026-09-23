import unittest

from exvram.hardware import detect_hardware


class HardwareTests(unittest.TestCase):
    def test_detection_returns_a_serializable_snapshot_without_cuda_assumption(self):
        snapshot = detect_hardware()
        payload = snapshot.to_dict()
        self.assertIn("cuda_available", payload)
        self.assertIn("cuda_runtime_available", payload)
        self.assertIn("warnings", payload)
        self.assertIn("vram_gib", payload)
