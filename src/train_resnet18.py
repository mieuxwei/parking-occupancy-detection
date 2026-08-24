"""Train pretrained ResNet18 on the primary date-grouped split."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

from src.data_paths import require_data_root
from src.dataset import (
    IMAGENET_RGB_MEAN,
    IMAGENET_RGB_STD,
    ParkingOccupancyDataset,
    build_resnet_transform,
)
from src.resnet18_transfer import build_resnet18, freeze_backbone, unfreeze_all
from src.train_baseline import evaluate, seed_everything, select_device, train_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--head-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-4)
    parser.add_argument("--classifier-learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
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
        transform=build_resnet_transform(True, args.input_size),
    )
    validation_dataset = ParkingOccupancyDataset(
        args.manifest,
        data_root,
        split="validation",
        protocol="date",
        transform=build_resnet_transform(False, args.input_size),
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

    model = build_resnet18(pretrained=True, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = {
        "architecture": "ResNet18",
        "pretrained_weights": ResNet18_Weights.DEFAULT.name,
        "protocol": "date",
        "subset": "CNR-EXT",
        "train_split": "train",
        "validation_split": "validation",
        "test_split_used": False,
        "head_epochs": args.head_epochs,
        "finetune_epochs": args.finetune_epochs,
        "batch_size": args.batch_size,
        "input_size": args.input_size,
        "optimizer": "AdamW",
        "head_learning_rate": args.head_learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate,
        "classifier_learning_rate": args.classifier_learning_rate,
        "weight_decay": args.weight_decay,
        "loss": "CrossEntropyLoss",
        "class_weights": None,
        "dropout": args.dropout,
        "freeze_batch_norm_during_head_training": True,
        "seed": args.seed,
        "workers": args.workers,
        "device": str(device),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "parameter_count": parameter_count,
        "normalization_mean": IMAGENET_RGB_MEAN,
        "normalization_std": IMAGENET_RGB_STD,
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
    started = time.monotonic()
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    freeze_backbone(model)
    phases = [
        (
            "head",
            args.head_epochs,
            AdamW(
                model.fc.parameters(),
                lr=args.head_learning_rate,
                weight_decay=args.weight_decay,
            ),
        )
    ]

    # Build the fine-tuning optimizer after restoring gradients to all layers.
    if args.finetune_epochs:
        unfreeze_all(model)
        backbone_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("fc.")
        ]
        phases.append(
            (
                "finetune",
                args.finetune_epochs,
                AdamW(
                    [
                        {
                            "params": backbone_parameters,
                            "lr": args.backbone_learning_rate,
                        },
                        {
                            "params": model.fc.parameters(),
                            "lr": args.classifier_learning_rate,
                        },
                    ],
                    weight_decay=args.weight_decay,
                ),
            )
        )

    global_epoch = 0
    for phase_name, phase_epochs, optimizer in phases:
        if phase_name == "head":
            freeze_backbone(model)
        else:
            unfreeze_all(model)
        for _ in range(phase_epochs):
            global_epoch += 1
            epoch_started = time.monotonic()
            train_loss = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                freeze_batch_norm=phase_name == "head",
            )
            validation_metrics = evaluate(model, validation_loader, criterion, device)
            record = {
                "epoch": global_epoch,
                "phase": phase_name,
                "train_loss": train_loss,
                "validation": validation_metrics,
                "duration_seconds": time.monotonic() - epoch_started,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if validation_metrics["f1_occupied"] > best_f1:
                best_f1 = validation_metrics["f1_occupied"]
                best_epoch = global_epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "best_epoch": best_epoch,
                        "validation_metrics": validation_metrics,
                    },
                    args.checkpoint,
                )

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
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
        },
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved result to {args.result}", flush=True)


if __name__ == "__main__":
    main()
