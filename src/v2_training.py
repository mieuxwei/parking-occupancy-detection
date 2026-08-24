"""Locked datasets, transforms, metrics, and models for V2 robustness training."""

from __future__ import annotations

import csv
import gzip
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import InterpolationMode

from src.data_paths import resolve_pklot_image
from src.dataset import IMAGENET_RGB_MEAN, IMAGENET_RGB_STD, PadToSquare
from src.train_baseline import binary_metrics


ALLOWED_V2_DEVELOPMENT_SPLITS = {"v2_train", "v2_validation"}


def build_v2_transform(training: bool, input_size: int = 224) -> Callable:
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
                    degrees=7,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=tuple(round(value * 255) for value in IMAGENET_RGB_MEAN),
                ),
                transforms.ColorJitter(
                    brightness=0.20,
                    contrast=0.20,
                    saturation=0.10,
                    hue=0.02,
                ),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_RGB_MEAN, IMAGENET_RGB_STD),
        ]
    )
    return transforms.Compose(operations)


class V2DevelopmentDataset(Dataset):
    """Load only precommitted V2 train or validation rows and their site."""

    def __init__(
        self,
        manifest: Path,
        data_root: Path,
        split: str,
        transform: Callable,
        return_site: bool,
    ) -> None:
        if split not in ALLOWED_V2_DEVELOPMENT_SPLITS:
            raise ValueError(
                f"V2 training code only permits {sorted(ALLOWED_V2_DEVELOPMENT_SPLITS)}; "
                f"got {split}"
            )
        opener = gzip.open if manifest.suffix == ".gz" else open
        with opener(manifest, "rt", newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            required = {"image_url", "occupancy", "site", "v2_split"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
            self.samples = [
                (row["image_url"], int(row["occupancy"]), row["site"])
                for row in reader
                if row["v2_split"] == split
            ]
        if not self.samples:
            raise ValueError(f"No samples found for allowed development split {split}")
        self.data_root = data_root
        self.transform = transform
        self.return_site = return_site

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_url, label, site = self.samples[index]
        image_path = resolve_pklot_image(image_url, self.data_root)
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        if self.return_site:
            return tensor, label, site
        return tensor, label


def balanced_site_label_sampler(
    samples: list[tuple[str, int, str]], seed: int
) -> tuple[WeightedRandomSampler, dict[str, int]]:
    counts = Counter((site, label) for _, label, site in samples)
    weights = [1.0 / counts[(site, label)] for _, label, site in samples]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(samples),
        replacement=True,
        generator=generator,
    )
    serialized_counts = {
        f"{site}:{label}": count for (site, label), count in sorted(counts.items())
    }
    return sampler, serialized_counts


def build_efficientnet_b0(pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.20), nn.Linear(features, 2))
    return model


def freeze_efficientnet_backbone(model: nn.Module) -> None:
    model.requires_grad_(False)
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_model(model: nn.Module) -> None:
    model.requires_grad_(True)


def validation_rank(metrics: dict[str, Any]) -> tuple[float, float, float]:
    selection = metrics["selection"]
    return (
        float(selection["macro_site_f1_occupied"]),
        float(selection["minimum_site_recall_occupied"]),
        float(metrics["overall"]["f1_occupied"]),
    )


@torch.inference_mode()
def evaluate_v2_validation(
    model: nn.Module,
    loader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    labels_all: list[int] = []
    predictions_all: list[int] = []
    labels_by_site: dict[str, list[int]] = {}
    predictions_by_site: dict[str, list[int]] = {}
    loss_sum = 0.0
    for inputs, labels, sites in loader:
        labels_device = labels.to(device)
        logits = model(inputs.to(device))
        loss_sum += criterion(logits, labels_device).item()
        predictions = logits.argmax(dim=1).cpu().tolist()
        label_values = labels.tolist()
        labels_all.extend(label_values)
        predictions_all.extend(predictions)
        for label, prediction, site in zip(label_values, predictions, sites):
            labels_by_site.setdefault(site, []).append(label)
            predictions_by_site.setdefault(site, []).append(prediction)
    overall = binary_metrics(labels_all, predictions_all)
    overall["loss"] = loss_sum / len(labels_all)
    by_site = {
        site: binary_metrics(labels_by_site[site], predictions_by_site[site])
        for site in sorted(labels_by_site)
    }
    expected_sites = {"PUC", "UFPR04", "UFPR05"}
    if set(by_site) != expected_sites:
        raise ValueError(f"Expected validation sites {sorted(expected_sites)}, got {sorted(by_site)}")
    macro_site_f1 = sum(value["f1_occupied"] for value in by_site.values()) / len(
        by_site
    )
    minimum_site_recall = min(value["recall_occupied"] for value in by_site.values())
    return {
        "overall": overall,
        "by_site": by_site,
        "selection": {
            "macro_site_f1_occupied": macro_site_f1,
            "minimum_site_recall_occupied": minimum_site_recall,
            "overall_f1_occupied": overall["f1_occupied"],
            "rank_tuple": [macro_site_f1, minimum_site_recall, overall["f1_occupied"]],
        },
    }


def train_v2_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    candidate_id: str,
    epoch: int,
    progress_interval: int,
    freeze_batch_norm: bool = False,
) -> float:
    model.train()
    if freeze_batch_norm:
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    sample_count = 0
    started = time.monotonic()
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        if sample_count == len(loader.dataset) or (
            sample_count % progress_interval < batch_size
        ):
            print(
                json.dumps(
                    {
                        "candidate": candidate_id,
                        "epoch": epoch,
                        "train_processed": sample_count,
                        "train_total": len(loader.dataset),
                        "running_loss": loss_sum / sample_count,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    return loss_sum / sample_count

