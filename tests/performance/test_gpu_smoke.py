import unittest

try:
    import torch
except ImportError:  # optional GPU smoke dependency
    torch = None


CUDA_AVAILABLE = torch is not None and bool(torch.cuda.is_available())


@unittest.skipUnless(CUDA_AVAILABLE, "optional GPU smoke test requires a CUDA-enabled PyTorch")
class CudaSmokeTests(unittest.TestCase):
    def test_cuda_can_allocate_and_synchronize(self):
        tensor = torch.zeros((1,), device="cuda")
        torch.cuda.synchronize()
        self.assertEqual(tensor.device.type, "cuda")

