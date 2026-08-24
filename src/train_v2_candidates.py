"""Train and validation-select exactly the two precommitted V2 candidates."""

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
from src.evaluate_cross_domain import file_sha256
from src.resnet18_transfer import build_resnet18
from src.train_baseline import seed_everything, select_device
from src.v2_training import (
    V2DevelopmentDataset,
    balanced_site_label_sampler,
    build_efficientnet_b0,
    build_v2_transform,
    evaluate_v2_validation,
    freeze_efficientnet_backbone,
    train_v2_epoch,
    unfreeze_model,
    validation_rank,
)


PRECOMMITTED_CONFIG_SHA256 = "57fb8133760cc7eded11eb77e9c1ce5aa67d5379bfbfe33539b279e95a024957"
EXPECTED_CANDIDATES = [
    "v2a_balanced_resnet18",
    "v2b_balanced_efficientnet_b0",
]


def optimizer_for_full_finetune(
    model: nn.Module,
    classifier_prefix: str,
    backbone_learning_rate: float,
    classifier_learning_rate: float,
    weight_decay: float,
) -> AdamW:
    backbone = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(classifier_prefix)
    ]
    classifier = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(classifier_prefix)
    ]
    return AdamW(
        [
            {"params": backbone, "lr": backbone_learning_rate},
            {"params": classifier, "lr": classifier_learning_rate},
        ],
        weight_decay=weight_decay,
    )


def save_partial(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def run_candidate(
    candidate: dict[str, Any],
    experiment_config: dict[str, Any],
    manifest: Path,
    data_root: Path,
    v1_checkpoint: Path,
    checkpoint_path: Path,
    device: torch.device,
    result_path: Path,
    global_result: dict[str, Any],
    progress_interval: int,
) -> dict[str, Any]:
    candidate_id = candidate["id"]
    common = experiment_config["common_training"]
    seed = int(common["seed"])
    seed_everything(seed)
    train_dataset = V2DevelopmentDataset(
        manifest,
        data_root,
        "v2_train",
        build_v2_transform(True, int(experiment_config["preprocessing"]["input_size"])),
        return_site=False,
    )
    validation_dataset = V2DevelopmentDataset(
        manifest,
        data_root,
        "v2_validation",
        build_v2_transform(False, int(experiment_config["preprocessing"]["input_size"])),
        return_site=True,
    )
    if len(train_dataset) != experiment_config["data"]["train_samples"]:
        raise ValueError("V2 train sample count differs from precommit")
    if len(validation_dataset) != experiment_config["data"]["validation_samples"]:
        raise ValueError("V2 validation sample count differs from precommit")
    sampler, sampler_cell_counts = balanced_site_label_sampler(train_dataset.samples, seed)
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(common["train_batch_size"]),
        sampler=sampler,
        num_workers=int(common["workers"]),
        persistent_workers=int(common["workers"]) > 0,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(common["eval_batch_size"]),
        shuffle=False,
        num_workers=int(common["workers"]),
        persistent_workers=int(common["workers"]) > 0,
    )

    if candidate_id == "v2a_balanced_resnet18":
        v1 = torch.load(v1_checkpoint, map_location="cpu", weights_only=True)
        model = build_resnet18(pretrained=False, dropout=0.20)
        model.load_state_dict(v1["model_state_dict"])
        classifier_prefix = "fc."
        phases = [
            {
                "name": "full_finetune",
                "epochs": int(candidate["full_finetune_epochs"]),
                "backbone_learning_rate": float(candidate["backbone_learning_rate"]),
                "classifier_learning_rate": float(candidate["classifier_learning_rate"]),
            }
        ]
    elif candidate_id == "v2b_balanced_efficientnet_b0":
        model = build_efficientnet_b0(pretrained=True)
        classifier_prefix = "classifier."
        phases = [
            {
                "name": "head",
                "epochs": int(candidate["head_only_epochs"]),
                "learning_rate": float(candidate["head_learning_rate"]),
            },
            {
                "name": "full_finetune",
                "epochs": int(candidate["full_finetune_epochs"]),
                "backbone_learning_rate": float(candidate["backbone_learning_rate"]),
                "classifier_learning_rate": float(candidate["classifier_learning_rate"]),
            },
        ]
    else:
        raise ValueError(f"Unexpected candidate outside precommit: {candidate_id}")

    model.to(device)
    history: list[dict[str, Any]] = []
    best_rank = (-1.0, -1.0, -1.0)
    best_epoch = 0
    best_validation: dict[str, Any] | None = None
    global_epoch = 0
    candidate_started = time.monotonic()
    candidate_config = {
        **candidate,
        "manifest": manifest.name,
        "manifest_sha256": file_sha256(manifest),
        "train_split": "v2_train",
        "validation_split": "v2_validation",
        "fresh_final_dataset_instantiated": False,
        "fresh_final_images_opened": 0,
        "error_artifacts_read": False,
        "sampler": experiment_config["sampling"],
        "sampler_cell_counts": sampler_cell_counts,
        "preprocessing": experiment_config["preprocessing"],
        "augmentation": experiment_config["augmentation"],
        "common_training": common,
        "device": str(device),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    print(json.dumps({"candidate_config": candidate_config}, indent=2), flush=True)
    for phase in phases:
        if phase["name"] == "head":
            freeze_efficientnet_backbone(model)
            optimizer = AdamW(
                model.classifier.parameters(),
                lr=phase["learning_rate"],
                weight_decay=float(common["weight_decay"]),
            )
            freeze_batch_norm = True
        else:
            unfreeze_model(model)
            optimizer = optimizer_for_full_finetune(
                model,
                classifier_prefix,
                phase["backbone_learning_rate"],
                phase["classifier_learning_rate"],
                float(common["weight_decay"]),
            )
            freeze_batch_norm = False
        for _ in range(phase["epochs"]):
            global_epoch += 1
            epoch_started = time.monotonic()
            train_loss = train_v2_epoch(
                model,
                train_loader,
                optimizer,
                device,
                candidate_id,
                global_epoch,
                progress_interval,
                freeze_batch_norm=freeze_batch_norm,
            )
            validation = evaluate_v2_validation(model, validation_loader, device)
            record = {
                "epoch": global_epoch,
                "phase": phase["name"],
                "train_loss": train_loss,
                "validation": validation,
                "duration_seconds": time.monotonic() - epoch_started,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            rank = validation_rank(validation)
            if rank > best_rank:
                best_rank = rank
                best_epoch = global_epoch
                best_validation = validation
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "candidate_id": candidate_id,
                        "architecture": candidate["architecture"],
                        "experiment_config_sha256": PRECOMMITTED_CONFIG_SHA256,
                        "v2_manifest_sha256": experiment_config["data"][
                            "v2_manifest_sha256"
                        ],
                        "best_epoch": best_epoch,
                        "best_phase": phase["name"],
                        "validation_metrics": best_validation,
                        "fresh_final_dataset_instantiated": False,
                        "fresh_final_images_opened": 0,
                        "error_artifacts_read": False,
                    },
                    checkpoint_path,
                )
            partial_candidate = {
                "status": "training",
                "candidate_config": candidate_config,
                "history": history,
                "best_epoch": best_epoch,
                "best_validation": best_validation,
                "checkpoint": checkpoint_path.name,
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "duration_seconds": time.monotonic() - candidate_started,
            }
            existing = [
                item
                for item in global_result.get("candidates", [])
                if item.get("candidate_config", {}).get("id") != candidate_id
            ]
            global_result["candidates"] = existing + [partial_candidate]
            save_partial(result_path, global_result)
    if best_validation is None:
        raise RuntimeError(f"No validation result for {candidate_id}")
    selected_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if selected_checkpoint["best_epoch"] != best_epoch:
        raise ValueError("Fresh checkpoint load does not match selected epoch")
    return {
        "status": "complete",
        "candidate_config": candidate_config,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "validation_rank": list(best_rank),
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "duration_seconds": time.monotonic() - candidate_started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_config", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("v1_checkpoint", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("selection_lock", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-interval", type=int, default=5000)
    args = parser.parse_args()

    started = time.monotonic()
    config_sha = file_sha256(args.experiment_config)
    if config_sha != PRECOMMITTED_CONFIG_SHA256:
        raise ValueError(
            f"Experiment config SHA-256 changed: expected {PRECOMMITTED_CONFIG_SHA256}, "
            f"got {config_sha}"
        )
    experiment_config = json.loads(args.experiment_config.read_text(encoding="utf-8"))
    if experiment_config["status"] != "precommitted_no_training":
        raise ValueError("Experiment config is not the precommitted no-training record")
    if file_sha256(args.manifest) != experiment_config["data"]["v2_manifest_sha256"]:
        raise ValueError("V2 manifest SHA-256 differs from precommit")
    if file_sha256(args.v1_checkpoint) != experiment_config["v1_baseline"][
        "checkpoint_sha256"
    ]:
        raise ValueError("V1 checkpoint SHA-256 differs from frozen baseline")
    candidate_ids = [item["id"] for item in experiment_config["candidates"]]
    if candidate_ids != EXPECTED_CANDIDATES:
        raise ValueError(f"Expected exactly the two candidates {EXPECTED_CANDIDATES}")
    if args.selection_lock.exists():
        raise FileExistsError("Selection lock already exists; refusing to overwrite it")

    device = select_device(args.device)
    data_root = require_data_root()
    result: dict[str, Any] = {
        "status": "training",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_config": args.experiment_config.name,
        "experiment_config_sha256": config_sha,
        "v2_manifest": args.manifest.name,
        "v2_manifest_sha256": file_sha256(args.manifest),
        "v1_checkpoint_sha256": file_sha256(args.v1_checkpoint),
        "device": str(device),
        "seed": experiment_config["common_training"]["seed"],
        "fresh_final_dataset_instantiated": False,
        "fresh_final_images_opened": 0,
        "error_artifacts_read": False,
        "candidates": [],
    }
    save_partial(args.result, result)
    completed_candidates = []
    for candidate in experiment_config["candidates"]:
        checkpoint_path = args.output_directory / f"{candidate['id']}.pt"
        if checkpoint_path.exists():
            raise FileExistsError(f"Refusing to overwrite candidate checkpoint: {checkpoint_path}")
        completed = run_candidate(
            candidate,
            experiment_config,
            args.manifest,
            data_root,
            args.v1_checkpoint,
            checkpoint_path,
            device,
            args.result,
            result,
            args.progress_interval,
        )
        completed_candidates.append(completed)
        result["candidates"] = completed_candidates
        save_partial(args.result, result)

    selected = max(
        completed_candidates,
        key=lambda item: tuple(item["validation_rank"]),
    )
    selected_checkpoint_path = args.output_directory / selected["checkpoint"]
    selection_lock = {
        "status": "selected_checkpoint_locked_fresh_final_unopened",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_source": "v2_validation only",
        "selection_criterion": [
            "macro_site_f1_occupied",
            "minimum_site_recall_occupied",
            "overall_f1_occupied",
        ],
        "selected_candidate": selected["candidate_config"]["id"],
        "selected_architecture": selected["candidate_config"]["architecture"],
        "selected_checkpoint": str(selected_checkpoint_path),
        "selected_checkpoint_sha256": file_sha256(selected_checkpoint_path),
        "selected_epoch": selected["best_epoch"],
        "selected_validation": selected["best_validation"],
        "candidate_validation_ranks": {
            item["candidate_config"]["id"]: item["validation_rank"]
            for item in completed_candidates
        },
        "experiment_config_sha256": config_sha,
        "v2_manifest_sha256": file_sha256(args.manifest),
        "fresh_final_dataset_instantiated": False,
        "fresh_final_images_opened": 0,
        "error_artifacts_read": False,
        "streamlit_demo_updated": False,
    }
    args.selection_lock.parent.mkdir(parents=True, exist_ok=True)
    args.selection_lock.write_text(
        json.dumps(selection_lock, indent=2) + "\n", encoding="utf-8"
    )
    result.update(
        {
            "status": "complete_selected_fresh_final_unopened",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidates": completed_candidates,
            "selection_lock": args.selection_lock.name,
            "selection_lock_sha256": file_sha256(args.selection_lock),
            "selected_candidate": selection_lock["selected_candidate"],
            "selected_checkpoint": selection_lock["selected_checkpoint"],
            "selected_checkpoint_sha256": selection_lock[
                "selected_checkpoint_sha256"
            ],
            "total_duration_seconds": time.monotonic() - started,
            "versions": {
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "numpy": np.__version__,
            },
        }
    )
    save_partial(args.result, result)
    print(json.dumps(selection_lock, indent=2), flush=True)
    print(f"Saved completed result to {args.result}", flush=True)


if __name__ == "__main__":
    main()
