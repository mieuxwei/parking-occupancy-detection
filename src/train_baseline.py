"""Train the SimpleCNN baseline on the primary date-grouped split."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data_paths import require_data_root
from src.dataset import (
    TRAIN_RGB_MEAN,
    TRAIN_RGB_STD,
    ParkingOccupancyDataset,
    build_transform,
)
from src.simple_cnn import SimpleCNN


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    tn = sum(label == 0 and pred == 0 for label, pred in zip(labels, predictions))
    fp = sum(label == 0 and pred == 1 for label, pred in zip(labels, predictions))
    fn = sum(label == 1 and pred == 0 for label, pred in zip(labels, predictions))
    tp = sum(label == 1 and pred == 1 for label, pred in zip(labels, predictions))
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision_occupied": precision,
        "recall_occupied": recall,
        "f1_occupied": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "samples": total,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    freeze_batch_norm: bool = False,
) -> float:
    model.train()
    if freeze_batch_norm:
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
    loss_sum = 0.0
    sample_count = 0
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
    return loss_sum / sample_count


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())
    metrics = binary_metrics(labels_all, predictions_all)
    metrics["loss"] = loss_sum / len(labels_all)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = select_device(args.device)
    data_root = require_data_root()
    train_dataset = ParkingOccupancyDataset(
        args.manifest,
        data_root,
        split="train",
        protocol="date",
        transform=build_transform(True, args.input_size),
    )
    validation_dataset = ParkingOccupancyDataset(
        args.manifest,
        data_root,
        split="validation",
        protocol="date",
        transform=build_transform(False, args.input_size),
    )

    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "persistent_workers": args.workers > 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )

    model = SimpleCNN(dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = {
        "architecture": "SimpleCNN",
        "protocol": "date",
        "subset": "CNR-EXT",
        "train_split": "train",
        "validation_split": "validation",
        "test_split_used": False,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "input_size": args.input_size,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss": "CrossEntropyLoss",
        "class_weights": None,
        "dropout": args.dropout,
        "early_stopping_patience": args.patience,
        "seed": args.seed,
        "workers": args.workers,
        "device": str(device),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "parameter_count": parameter_count,
        "normalization_mean": TRAIN_RGB_MEAN,
        "normalization_std": TRAIN_RGB_STD,
        "augmentation": {
            "horizontal_flip_probability": 0.5,
            "rotation_degrees": 5,
            "brightness": 0.15,
            "contrast": 0.15,
        },
    }
    print(json.dumps(config, indent=2), flush=True)

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.monotonic()
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        validation_metrics = evaluate(model, validation_loader, criterion, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation_metrics,
            "duration_seconds": time.monotonic() - epoch_started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

        current_f1 = validation_metrics["f1_occupied"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "best_epoch": best_epoch,
                    "validation_metrics": validation_metrics,
                },
                args.checkpoint,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}", flush=True)
                break

    best_record = next(record for record in history if record["epoch"] == best_epoch)
    result = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation": best_record["validation"],
        "history": history,
        "total_duration_seconds": time.monotonic() - started,
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved result to {args.result}", flush=True)


if __name__ == "__main__":
    main()
