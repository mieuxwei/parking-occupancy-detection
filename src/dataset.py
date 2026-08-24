"""Manifest-backed CNRPark+EXT dataset with external image storage."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, functional
from torchvision import transforms

from src.data_paths import resolve_cnrpark_image, resolve_pklot_image


TRAIN_RGB_MEAN = (0.412173, 0.402667, 0.353757)
TRAIN_RGB_STD = (0.175863, 0.171592, 0.179682)
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)


class PadToSquare:
    """Symmetrically edge-pad a PIL image to a square."""

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        horizontal = side - width
        vertical = side - height
        padding = [
            horizontal // 2,
            vertical // 2,
            horizontal - horizontal // 2,
            vertical - vertical // 2,
        ]
        return functional.pad(image, padding, padding_mode="edge")


def build_transform(
    training: bool,
    input_size: int = 128,
    mean: tuple[float, float, float] = TRAIN_RGB_MEAN,
    std: tuple[float, float, float] = TRAIN_RGB_STD,
) -> Callable[[Image.Image], Tensor]:
    operations: list[Callable] = [
        PadToSquare(),
        transforms.Resize(
            (input_size, input_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
    ]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(
                    degrees=5,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=tuple(round(value * 255) for value in mean),
                ),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transforms.Compose(operations)


def build_resnet_transform(
    training: bool, input_size: int = 224
) -> Callable[[Image.Image], Tensor]:
    return build_transform(
        training,
        input_size=input_size,
        mean=IMAGENET_RGB_MEAN,
        std=IMAGENET_RGB_STD,
    )


class ParkingOccupancyDataset(Dataset[tuple[Tensor, int]]):
    """Load eligible image paths for one manifest split without copying files."""

    def __init__(
        self,
        manifest: Path,
        data_root: Path,
        split: str,
        transform: Callable[[Image.Image], Tensor],
        protocol: str = "date",
        subset: str = "CNR-EXT",
    ) -> None:
        if split not in {"train", "validation", "test", "cross_subset_test"}:
            raise ValueError(f"Unsupported split: {split}")
        split_column = {"date": "date_split", "camera": "camera_split"}.get(protocol)
        if split_column is None:
            raise ValueError(f"Unsupported protocol: {protocol}")

        with manifest.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            required = {"image_url", "occupancy", "subset", split_column, "quality_status"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
            self.samples = [
                (row["image_url"], int(row["occupancy"]))
                for row in reader
                if row["subset"] == subset
                and row["quality_status"] == "include"
                and row[split_column] == split
            ]

        if not self.samples:
            raise ValueError(
                f"No eligible samples for subset={subset}, protocol={protocol}, split={split}"
            )
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image_url, label = self.samples[index]
        image_path = resolve_cnrpark_image(image_url, self.data_root)
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label


class PKLotOccupancyDataset(Dataset[tuple[Tensor, int, str, str, str]]):
    """Load PKLot cross-domain patches and grouping metadata from gzip CSV."""

    def __init__(
        self,
        manifest: Path,
        data_root: Path,
        transform: Callable[[Image.Image], Tensor],
        adaptation_split: str | None = None,
    ) -> None:
        opener = gzip.open if manifest.suffix == ".gz" else open
        with opener(manifest, "rt", newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            required = {
                "image_url",
                "occupancy",
                "site",
                "physical_location",
                "weather",
                "domain_role",
            }
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
            if adaptation_split is not None and "adaptation_split" not in (
                reader.fieldnames or []
            ):
                raise ValueError("Manifest is missing column: adaptation_split")
            self.samples = [
                (
                    row["image_url"],
                    int(row["occupancy"]),
                    row["site"],
                    row["physical_location"],
                    row["weather"],
                )
                for row in reader
                if row["domain_role"] == "cross_domain_evaluation"
                and (
                    adaptation_split is None
                    or row["adaptation_split"] == adaptation_split
                )
            ]
        if not self.samples:
            raise ValueError("No eligible PKLot cross-domain evaluation samples")
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str, str, str]:
        image_url, label, site, physical_location, weather = self.samples[index]
        image_path = resolve_pklot_image(image_url, self.data_root)
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label, site, physical_location, weather


class PKLotAdaptationDataset(Dataset[tuple[Tensor, int]]):
    """Load one audited PKLot adaptation split without returning metadata."""

    def __init__(
        self,
        manifest: Path,
        data_root: Path,
        split: str,
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        if split not in {
            "adaptation_train",
            "adaptation_validation",
            "heldout_evaluation",
        }:
            raise ValueError(f"Unsupported PKLot adaptation split: {split}")
        opener = gzip.open if manifest.suffix == ".gz" else open
        with opener(manifest, "rt", newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            required = {"image_url", "occupancy", "adaptation_split"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
            self.samples = [
                (row["image_url"], int(row["occupancy"]))
                for row in reader
                if row["adaptation_split"] == split
            ]
        if not self.samples:
            raise ValueError(f"No eligible PKLot samples for {split}")
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image_url, label = self.samples[index]
        image_path = resolve_pklot_image(image_url, self.data_root)
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label
