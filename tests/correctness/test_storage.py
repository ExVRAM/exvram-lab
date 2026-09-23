import unittest

from exvram.storage import DRIVE_REMOVABLE, StorageSafetyError, assert_storage_safe


class StorageSafetyTests(unittest.TestCase):
    def test_removable_path_is_rejected_without_override(self):
        with self.assertRaises(StorageSafetyError):
            assert_storage_safe(
                [r"R:\models\model.gguf"],
                drive_type_fn=lambda _: DRIVE_REMOVABLE,
            )

    def test_explicit_override_allows_removable_path(self):
        assert_storage_safe(
            [r"R:\models\model.gguf"],
            allow_removable=True,
            drive_type_fn=lambda _: DRIVE_REMOVABLE,
        )

    def test_fixed_and_relative_paths_are_allowed(self):
        assert_storage_safe(
            [r"R:\project\model.gguf", "models/model.gguf"],
            drive_type_fn=lambda path: 3 if path.startswith("E:") else None,
        )


if __name__ == "__main__":
    unittest.main()
