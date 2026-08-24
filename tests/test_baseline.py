import csv
import gzip
import tempfile
import unittest
from pathlib import Path, PurePosixPath

import torch
from PIL import Image

from src.dataset import (
    PKLotAdaptationDataset,
    PKLotOccupancyDataset,
    ParkingOccupancyDataset,
    build_transform,
)
from src.evaluate_cross_domain import ConfusionAccumulator
from src.simple_cnn import SimpleCNN
from src.resnet18_transfer import build_resnet18, freeze_backbone, unfreeze_all
from src.train_baseline import binary_metrics
from src.prepare_pklot_manifest import parse_patch_path
from src.prepare_pklot_adaptation_split import assign_date_groups
from src.analyze_errors import make_error_row, rank_errors, representative_errors


class BaselineTests(unittest.TestCase):
    def test_dataset_filters_quality_subset_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "cnrpark_ext/extracted/CNR-EXT/example.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("L", (100, 150), color=128).save(image_path)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as destination:
                writer = csv.DictWriter(
                    destination,
                    fieldnames=[
                        "image_url",
                        "occupancy",
                        "subset",
                        "quality_status",
                        "date_split",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_url": "CNR-EXT/example.jpg",
                        "occupancy": "1",
                        "subset": "CNR-EXT",
                        "quality_status": "include",
                        "date_split": "train",
                    }
                )
            dataset = ParkingOccupancyDataset(
                manifest,
                root,
                split="train",
                transform=build_transform(False, 64),
            )
            tensor, label = dataset[0]
            self.assertEqual(tuple(tensor.shape), (3, 64, 64))
            self.assertEqual(label, 1)

    def test_model_output_shape(self) -> None:
        output = SimpleCNN()(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 2))

    def test_resnet18_output_and_freezing(self) -> None:
        model = build_resnet18(pretrained=False)
        freeze_backbone(model)
        self.assertTrue(all(parameter.requires_grad for parameter in model.fc.parameters()))
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in model.named_parameters()
                if not name.startswith("fc.")
            )
        )
        output = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 2))
        unfreeze_all(model)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_binary_metrics(self) -> None:
        metrics = binary_metrics([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["f1_occupied"], 0.5)

    def test_parse_pklot_patch_path(self) -> None:
        row = parse_patch_path(
            PurePosixPath(
                "PKLot/PKLotSegmented/PUC/Sunny/2012-09-17/Occupied/"
                "2012-09-17_08_33_55#036.jpg"
            )
        )
        self.assertEqual(row["physical_location"], "PUCPR")
        self.assertEqual(row["source_frame_id"], "PUC/2012-09-17/08:33:55")
        self.assertEqual(row["slot_id"], 36)
        self.assertEqual(row["occupancy"], 1)
        self.assertEqual(row["domain_role"], "cross_domain_evaluation")

    def test_pklot_dataset_reads_gzip_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = (
                root
                / "pklot/extracted/PKLot/PKLotSegmented/PUC/Sunny/2012-09-17/Empty/example.jpg"
            )
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (80, 120), color=(64, 96, 128)).save(image_path)
            manifest = root / "manifest.csv.gz"
            with gzip.open(manifest, "wt", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(
                    target,
                    fieldnames=[
                        "image_url",
                        "occupancy",
                        "site",
                        "physical_location",
                        "weather",
                        "domain_role",
                        "adaptation_split",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_url": "PKLot/PKLotSegmented/PUC/Sunny/2012-09-17/Empty/example.jpg",
                        "occupancy": 0,
                        "site": "PUC",
                        "physical_location": "PUCPR",
                        "weather": "sunny",
                        "domain_role": "cross_domain_evaluation",
                        "adaptation_split": "adaptation_train",
                    }
                )
            dataset = PKLotOccupancyDataset(
                manifest,
                root,
                transform=build_transform(False, 64),
            )
            tensor, label, site, location, weather = dataset[0]
            self.assertEqual(tuple(tensor.shape), (3, 64, 64))
            self.assertEqual((label, site, location, weather), (0, "PUC", "PUCPR", "sunny"))
            adaptation = PKLotAdaptationDataset(
                manifest,
                root,
                split="adaptation_train",
                transform=build_transform(False, 64),
            )
            adaptation_tensor, adaptation_label = adaptation[0]
            self.assertEqual(tuple(adaptation_tensor.shape), (3, 64, 64))
            self.assertEqual(adaptation_label, 0)

    def test_confusion_accumulator(self) -> None:
        accumulator = ConfusionAccumulator()
        for label, prediction in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            accumulator.update(label, prediction, 0.25)
        metrics = accumulator.metrics()
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["f1_occupied"], 0.5)
        self.assertEqual(metrics["loss"], 0.25)

    def test_adaptation_date_assignment_is_deterministic_and_disjoint(self) -> None:
        dates = {
            "PUC": {f"2012-09-{day:02d}" for day in range(1, 21)},
            "UFPR04": {f"2013-01-{day:02d}" for day in range(1, 21)},
        }
        first = assign_date_groups(dates, seed=42, train_fraction=0.05, validation_fraction=0.05)
        second = assign_date_groups(dates, seed=42, train_fraction=0.05, validation_fraction=0.05)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {(site, date) for site, values in dates.items() for date in values})
        for site in dates:
            site_splits = [split for (assigned_site, _), split in first.items() if assigned_site == site]
            self.assertEqual(site_splits.count("adaptation_train"), 1)
            self.assertEqual(site_splits.count("adaptation_validation"), 1)
            self.assertEqual(site_splits.count("heldout_evaluation"), 18)

    def test_error_rows_and_ranking_are_deterministic(self) -> None:
        metadata = {
            "image_id": "PUC/2012-09-12/06:05:16#001",
            "image_url": "PKLot/example.jpg",
            "site": "PUC",
            "physical_location": "PUCPR",
            "weather": "cloudy",
            "capture_date": "2012-09-12",
            "capture_time": "06:05:16",
            "source_frame_id": "PUC/2012-09-12/06:05:16",
            "slot_id": "1",
        }
        lower = make_error_row(metadata, 0, 1, 0.8, 0.8)
        higher = {**lower, "image_id": "PUC/example#002", "confidence": 0.9}
        self.assertEqual(lower["error_type"], "false_positive")
        self.assertEqual(rank_errors([lower, higher], "false_positive", 1), [higher])
        self.assertEqual(representative_errors([lower, higher]), [higher])


if __name__ == "__main__":
    unittest.main()
