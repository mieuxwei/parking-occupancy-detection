import csv
import gzip
import tempfile
import unittest
from pathlib import Path, PurePosixPath

import torch
import os
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
from src.inference import (
    decode_uploaded_image,
    load_locked_final_summary,
    prediction_from_probabilities,
    resolve_checkpoint_path,
)
from src.prepare_v2_protocol import assign_v2_date_groups
from src.v2_training import (
    balanced_site_label_sampler,
    build_v2_transform,
    build_efficientnet_b0,
    validation_rank,
)


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

    def test_demo_decodes_supported_image_and_rejects_invalid_upload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.png"
            Image.new("RGB", (40, 60), color=(10, 20, 30)).save(image_path)
            image = decode_uploaded_image(image_path.read_bytes())
            self.assertEqual((image.mode, image.size), ("RGB", (40, 60)))
        with self.assertRaisesRegex(ValueError, "valid decodable image"):
            decode_uploaded_image(b"not an image")

    def test_demo_prediction_and_checkpoint_resolution(self) -> None:
        result = prediction_from_probabilities(torch.tensor([0.25, 0.75]))
        self.assertEqual(result["label"], "OCCUPIED")
        self.assertAlmostEqual(result["confidence"], 0.75)
        threshold_result = prediction_from_probabilities(torch.tensor([0.5, 0.5]))
        self.assertEqual(threshold_result["label"], "OCCUPIED")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "selected.pt"
            checkpoint.touch()
            self.assertEqual(resolve_checkpoint_path(checkpoint, root), checkpoint.resolve())
            previous = os.environ.pop("PARKING_MODEL_PATH", None)
            try:
                with self.assertRaisesRegex(FileNotFoundError, "PARKING_MODEL_PATH") as error:
                    resolve_checkpoint_path(repository_root=root)
                self.assertNotIn(str(root), str(error.exception))
            finally:
                if previous is not None:
                    os.environ["PARKING_MODEL_PATH"] = previous

    def test_demo_reads_locked_final_summary_without_evaluation(self) -> None:
        summary = load_locked_final_summary(Path(__file__).resolve().parents[1])
        self.assertEqual(summary["samples"], 154669)
        self.assertAlmostEqual(summary["accuracy"], 0.9988944132308348)
        self.assertAlmostEqual(summary["occupied_f1"], 0.9989121376177722)
        self.assertEqual(summary["ufpr04_occupied_recall"], 1.0)

    def test_demo_uses_locked_v2_inference_preprocessing(self) -> None:
        operations = build_v2_transform(training=False, input_size=224).transforms
        self.assertEqual(
            [operation.__class__.__name__ for operation in operations],
            ["PadToSquare", "Resize", "ToTensor", "Normalize"],
        )
        self.assertEqual(operations[1].size, (224, 224))
        self.assertEqual(operations[1].interpolation.value, "bilinear")
        self.assertTrue(operations[1].antialias)
        self.assertEqual(tuple(operations[3].mean), (0.485, 0.456, 0.406))
        self.assertEqual(tuple(operations[3].std), (0.229, 0.224, 0.225))

    def test_v2_protocol_is_deterministic_and_keeps_final_out_of_development(self) -> None:
        dates = {
            "PUC": {f"2012-09-{day:02d}" for day in range(1, 16)},
            "UFPR04": {f"2013-01-{day:02d}" for day in range(1, 16)},
        }
        v1_splits = {
            (site, date): (
                "adaptation_train" if index == 0 else "heldout_evaluation"
            )
            for site, site_dates in dates.items()
            for index, date in enumerate(sorted(site_dates))
        }
        kwargs = {
            "dates_by_site": dates,
            "v1_split_by_date": v1_splits,
            "protocol_id": "test-v2",
            "train_dates_per_site": 4,
            "validation_dates_per_site": 2,
            "final_fraction": 0.20,
        }
        first = assign_v2_date_groups(**kwargs)
        second = assign_v2_date_groups(**kwargs)
        self.assertEqual(first, second)
        for site in dates:
            site_assignments = {
                date: split
                for (assigned_site, date), split in first.items()
                if assigned_site == site
            }
            self.assertEqual(list(site_assignments.values()).count("v2_train"), 4)
            self.assertEqual(list(site_assignments.values()).count("v2_validation"), 2)
            self.assertEqual(
                list(site_assignments.values()).count("v2_fresh_final_evaluation"),
                3,
            )
            for date, split in site_assignments.items():
                if split == "v2_fresh_final_evaluation":
                    self.assertEqual(v1_splits[(site, date)], "heldout_evaluation")

    def test_v2_sampler_balances_site_label_cells(self) -> None:
        samples = [
            ("a.jpg", 0, "PUC"),
            ("b.jpg", 0, "PUC"),
            ("c.jpg", 1, "PUC"),
            ("d.jpg", 1, "UFPR04"),
        ]
        sampler, counts = balanced_site_label_sampler(samples, seed=105)
        self.assertEqual(len(sampler), len(samples))
        self.assertEqual(
            counts,
            {"PUC:0": 2, "PUC:1": 1, "UFPR04:1": 1},
        )

    def test_v2_validation_rank_and_efficientnet_output(self) -> None:
        metrics = {
            "overall": {"f1_occupied": 0.8},
            "selection": {
                "macro_site_f1_occupied": 0.7,
                "minimum_site_recall_occupied": 0.6,
            },
        }
        self.assertEqual(validation_rank(metrics), (0.7, 0.6, 0.8))
        model = build_efficientnet_b0(pretrained=False)
        self.assertEqual(tuple(model(torch.zeros(1, 3, 64, 64)).shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
