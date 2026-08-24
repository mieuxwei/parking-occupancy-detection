import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data_paths import (
    DATA_ROOT_ENV,
    DatasetRootError,
    require_data_root,
    resolve_cnrpark_image,
    resolve_pklot_image,
)


class RequireDataRootTests(unittest.TestCase):
    def test_unset_variable_has_setup_instruction(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(DatasetRootError, "export PARKING_DATA_ROOT"):
                require_data_root()

    def test_relative_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(DatasetRootError, "absolute path"):
            require_data_root({DATA_ROOT_ENV: "relative/datasets"})

    def test_repository_path_is_rejected(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(DatasetRootError, "inside the Git repository"):
            require_data_root({DATA_ROOT_ENV: str(repository)}, repository)

    def test_existing_external_directory_is_accepted(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            result = require_data_root({DATA_ROOT_ENV: directory}, repository)
            self.assertEqual(result, Path(directory).resolve())

    def test_relative_image_url_resolves_below_external_root(self) -> None:
        root = Path("/external/datasets")
        result = resolve_cnrpark_image("CNRPark/A/free/example.jpg", root)
        self.assertEqual(
            result,
            root / "cnrpark_ext/extracted/CNRPark/A/free/example.jpg",
        )

    def test_unsafe_image_url_is_rejected(self) -> None:
        with self.assertRaisesRegex(DatasetRootError, "Unsafe image_url"):
            resolve_cnrpark_image("../outside.jpg", Path("/external/datasets"))

    def test_relative_pklot_path_resolves_below_external_root(self) -> None:
        root = Path("/external/datasets")
        result = resolve_pklot_image(
            "PKLot/PKLotSegmented/PUC/Sunny/2012-09-17/Empty/example.jpg",
            root,
        )
        self.assertEqual(
            result,
            root
            / "pklot/extracted/PKLot/PKLotSegmented/PUC/Sunny/2012-09-17/Empty/example.jpg",
        )


if __name__ == "__main__":
    unittest.main()
