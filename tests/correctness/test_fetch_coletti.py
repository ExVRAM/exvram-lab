import importlib.util
import unittest
from pathlib import Path

from exvram.storage import StorageSafetyError


def _load_fetcher():
    path = Path(__file__).parents[2] / "benchmark" / "fetch_coletti_bf16.py"
    spec = importlib.util.spec_from_file_location("fetch_coletti_bf16", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fetcher from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ColettiFetchGuardTests(unittest.TestCase):
    def test_f_drive_is_rejected_even_when_windows_reports_it_fixed(self):
        fetcher = _load_fetcher()
        with self.assertRaises(StorageSafetyError):
            fetcher._reject_forbidden_volume(Path(r"F:\exvram-coletti"))

    def test_e_drive_is_not_on_the_operator_ban(self):
        fetcher = _load_fetcher()
        self.assertEqual(fetcher._reject_forbidden_volume(Path(r"E:\project")), "E:")

    def test_main_stops_before_download_on_f(self):
        fetcher = _load_fetcher()
        self.assertEqual(fetcher.main(["--local-dir", r"F:\exvram-coletti"]), 2)
