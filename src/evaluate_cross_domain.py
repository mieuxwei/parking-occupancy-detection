"""Evaluate the frozen selected ResNet18 on CNR-EXT test and PKLot."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from src.data_paths import require_data_root
from src.dataset import (
    PKLotOccupancyDataset,
    ParkingOccupancyDataset,
    build_resnet_transform,
)
from src.resnet18_transfer import build_resnet18
from src.train_baseline import evaluate, select_device


class ConfusionAccumulator:
    def __init__(self) -> None:
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0
        self.loss_sum = 0.0

    def update(self, label: int, prediction: int, loss: float) -> None:
        if label == 0 and prediction == 0:
            self.tn += 1
        elif label == 0 and prediction == 1:
            self.fp += 1
        elif label == 1 and prediction == 0:
            self.fn += 1
        elif label == 1 and prediction == 1:
            self.tp += 1
        else:
            raise ValueError(f"Expected binary label/prediction, got {label}/{prediction}")
        self.loss_sum += loss

    def metrics(self) -> dict[str, Any]:
        total = self.tn + self.fp + self.fn + self.tp
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "accuracy": (self.tp + self.tn) / total if total else 0.0,
            "precision_occupied": precision,
            "recall_occupied": recall,
            "f1_occupied": f1,
            "confusion_matrix": [[self.tn, self.fp], [self.fn, self.tp]],
            "samples": total,
            "loss": self.loss_sum / total if total else 0.0,
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def evaluate_pklot(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    total_samples: int,
    progress_interval: int,
    progress_label: str = "pklot",
) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    overall = ConfusionAccumulator()
    grouped: dict[str, defaultdict[str, ConfusionAccumulator]] = {
        "physical_location": defaultdict(ConfusionAccumulator),
        "site": defaultdict(ConfusionAccumulator),
        "weather": defaultdict(ConfusionAccumulator),
    }
    started = time.monotonic()
    processed = 0
    for inputs, labels, sites, physical_locations, weather_values in loader:
        inputs = inputs.to(device)
        labels_device = labels.to(device)
        logits = model(inputs)
        losses = criterion(logits, labels_device).cpu().tolist()
        predictions = logits.argmax(dim=1).cpu().tolist()
        label_values = labels.tolist()
        for label, prediction, loss, site, location, weather in zip(
            label_values,
            predictions,
            losses,
            sites,
            physical_locations,
            weather_values,
        ):
            overall.update(label, prediction, loss)
            grouped["site"][site].update(label, prediction, loss)
            grouped["physical_location"][location].update(label, prediction, loss)
            grouped["weather"][weather].update(label, prediction, loss)
        processed += len(label_values)
        if processed == total_samples or processed % progress_interval < len(label_values):
            print(
                json.dumps(
                    {
                        f"{progress_label}_processed": processed,
                        f"{progress_label}_total": total_samples,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    return {
        "overall": overall.metrics(),
        "by_physical_location": {
            key: value.metrics() for key, value in sorted(grouped["physical_location"].items())
        },
        "by_site": {
            key: value.metrics() for key, value in sorted(grouped["site"].items())
        },
        "by_weather": {
            key: value.metrics() for key, value in sorted(grouped["weather"].items())
        },
        "duration_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cnr_manifest", type=Path)
    parser.add_argument("pklot_manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-interval", type=int, default=50000)
    args = parser.parse_args()

    started = time.monotonic()
    device = select_device(args.device)
    data_root = require_data_root()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    checkpoint_config = checkpoint["config"]
    if checkpoint_config.get("architecture") != "ResNet18":
        raise ValueError("Checkpoint is not the selected ResNet18 model")
    if checkpoint_config.get("test_split_used") is not False:
        raise ValueError("Checkpoint training record does not preserve the test boundary")
    model = build_resnet18(
        pretrained=False,
        dropout=float(checkpoint_config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.requires_grad_(False)
    model.to(device).eval()
    transform = build_resnet_transform(False, args.input_size)

    cnr_test = ParkingOccupancyDataset(
        args.cnr_manifest,
        data_root,
        split="test",
        protocol="date",
        transform=transform,
    )
    pklot = PKLotOccupancyDataset(args.pklot_manifest, data_root, transform)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "persistent_workers": args.workers > 0,
        "pin_memory": device.type == "cuda",
    }
    cnr_loader = DataLoader(cnr_test, shuffle=False, **loader_options)
    pklot_loader = DataLoader(pklot, shuffle=False, **loader_options)
    config = {
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_best_epoch": checkpoint["best_epoch"],
        "checkpoint_locked_before_pklot": True,
        "weights_or_hyperparameters_changed": False,
        "cnr_test_opened_once": True,
        "pklot_used_for_model_selection": False,
        "input_size": args.input_size,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": str(device),
        "cnr_test_samples": len(cnr_test),
        "pklot_samples": len(pklot),
    }
    print(json.dumps(config, indent=2), flush=True)

    cnr_started = time.monotonic()
    cnr_metrics = evaluate(
        model,
        cnr_loader,
        nn.CrossEntropyLoss(),
        device,
    )
    cnr_duration = time.monotonic() - cnr_started
    print(json.dumps({"cnr_test": cnr_metrics, "duration_seconds": cnr_duration}), flush=True)
    pklot_metrics = evaluate_pklot(
        model,
        pklot_loader,
        device,
        len(pklot),
        args.progress_interval,
    )
    cross = pklot_metrics["overall"]
    result = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "cnr_ext_test": cnr_metrics,
        "cnr_ext_test_duration_seconds": cnr_duration,
        "pklot": pklot_metrics,
        "cross_domain_drop_from_cnr_test": {
            "accuracy_absolute": cnr_metrics["accuracy"] - cross["accuracy"],
            "f1_occupied_absolute": cnr_metrics["f1_occupied"] - cross["f1_occupied"],
        },
        "total_duration_seconds": time.monotonic() - started,
        "versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
        },
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved result to {args.result}", flush=True)


if __name__ == "__main__":
    main()
