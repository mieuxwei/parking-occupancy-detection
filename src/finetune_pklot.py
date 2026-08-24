"""Fine-tune the selected ResNet18 on the audited PKLot adaptation pool."""

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

from src.data_paths import require_data_root
from src.dataset import (
    PKLotAdaptationDataset,
    PKLotOccupancyDataset,
    build_resnet_transform,
)
from src.evaluate_cross_domain import evaluate_pklot, file_sha256
from src.resnet18_transfer import build_resnet18, freeze_backbone, unfreeze_all
from src.train_baseline import evaluate, seed_everything, select_device, train_epoch


def metric_changes(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, float]:
    return {
        "accuracy_absolute": after["accuracy"] - before["accuracy"],
        "f1_occupied_absolute": after["f1_occupied"] - before["f1_occupied"],
        "precision_occupied_absolute": (
            after["precision_occupied"] - before["precision_occupied"]
        ),
        "recall_occupied_absolute": after["recall_occupied"] - before["recall_occupied"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("base_checkpoint", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--head-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--classifier-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-interval", type=int, default=50000)
    args = parser.parse_args()

    seed_everything(args.seed)
    started = time.monotonic()
    device = select_device(args.device)
    data_root = require_data_root()
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    base_config = base["config"]
    if base_config.get("architecture") != "ResNet18" or base.get("best_epoch") != 3:
        raise ValueError("Expected the selected Milestone 6 epoch-3 ResNet18")
    transform_train = build_resnet_transform(True, args.input_size)
    transform_eval = build_resnet_transform(False, args.input_size)
    train_dataset = PKLotAdaptationDataset(
        args.manifest,
        data_root,
        "adaptation_train",
        transform_train,
    )
    validation_dataset = PKLotAdaptationDataset(
        args.manifest,
        data_root,
        "adaptation_validation",
        transform_eval,
    )
    heldout_dataset = PKLotOccupancyDataset(
        args.manifest,
        data_root,
        transform_eval,
        adaptation_split="heldout_evaluation",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    heldout_loader = DataLoader(
        heldout_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    config = {
        "base_checkpoint": args.base_checkpoint.name,
        "base_checkpoint_sha256": file_sha256(args.base_checkpoint),
        "base_checkpoint_best_epoch": base["best_epoch"],
        "adaptation_manifest": args.manifest.name,
        "adaptation_manifest_sha256": file_sha256(args.manifest),
        "head_epochs": args.head_epochs,
        "finetune_epochs": args.finetune_epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "input_size": args.input_size,
        "head_learning_rate": args.head_learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate,
        "classifier_learning_rate": args.classifier_learning_rate,
        "weight_decay": args.weight_decay,
        "loss": "CrossEntropyLoss",
        "class_weights": None,
        "selection_metric": "adaptation_validation_f1_occupied",
        "seed": args.seed,
        "workers": args.workers,
        "device": str(device),
        "adaptation_train_samples": len(train_dataset),
        "adaptation_validation_samples": len(validation_dataset),
        "heldout_evaluation_samples": len(heldout_dataset),
        "cnr_ext_test_reopened": False,
    }
    print(json.dumps(config, indent=2), flush=True)

    model = build_resnet18(pretrained=False, dropout=float(base_config["dropout"]))
    model.load_state_dict(base["model_state_dict"])
    model.requires_grad_(False).to(device).eval()
    before = evaluate_pklot(
        model,
        heldout_loader,
        device,
        len(heldout_dataset),
        args.progress_interval,
        progress_label="before_heldout",
    )
    partial_result = {
        "status": "frozen_baseline_complete",
        "config": config,
        "heldout_before": before,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(partial_result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"heldout_before": before["overall"]}), flush=True)

    criterion = nn.CrossEntropyLoss()
    freeze_backbone(model)
    phases: list[tuple[str, int, AdamW]] = [
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
    if args.finetune_epochs:
        unfreeze_all(model)
        backbone = [
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
                        {"params": backbone, "lr": args.backbone_learning_rate},
                        {
                            "params": model.fc.parameters(),
                            "lr": args.classifier_learning_rate,
                        },
                    ],
                    weight_decay=args.weight_decay,
                ),
            )
        )
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    global_epoch = 0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
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
            validation = evaluate(
                model,
                validation_loader,
                criterion,
                device,
            )
            record = {
                "epoch": global_epoch,
                "phase": phase_name,
                "train_loss": train_loss,
                "validation": validation,
                "duration_seconds": time.monotonic() - epoch_started,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if validation["f1_occupied"] > best_f1:
                best_f1 = validation["f1_occupied"]
                best_epoch = global_epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "best_epoch": best_epoch,
                        "validation_metrics": validation,
                    },
                    args.checkpoint,
                )

    selected = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    selected_model = build_resnet18(
        pretrained=False,
        dropout=float(base_config["dropout"]),
    )
    selected_model.load_state_dict(selected["model_state_dict"])
    selected_model.requires_grad_(False).to(device).eval()
    after = evaluate_pklot(
        selected_model,
        heldout_loader,
        device,
        len(heldout_dataset),
        args.progress_interval,
        progress_label="after_heldout",
    )
    result = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation": selected["validation_metrics"],
        "heldout_before": before,
        "heldout_after": after,
        "heldout_improvement": metric_changes(before["overall"], after["overall"]),
        "site_improvement": {
            site: metric_changes(
                before["by_site"][site],
                after["by_site"][site],
            )
            for site in sorted(before["by_site"])
        },
        "fine_tuned_checkpoint_sha256": file_sha256(args.checkpoint),
        "total_duration_seconds": time.monotonic() - started,
        "versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
        },
    }
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved result to {args.result}", flush=True)


if __name__ == "__main__":
    main()
