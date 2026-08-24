"""Run the one-time, locked V1 versus V2 PKLot fresh-final comparison."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data_paths import require_data_root, resolve_pklot_image
from src.evaluate_cross_domain import ConfusionAccumulator
from src.resnet18_transfer import build_resnet18
from src.train_baseline import select_device
from src.v2_training import build_v2_transform


EXPECTED_CONFIG_SHA256 = "57fb8133760cc7eded11eb77e9c1ce5aa67d5379bfbfe33539b279e95a024957"
EXPECTED_SELECTION_SHA256 = "60743fb7594e05cd7c2726a2a1029853f99473cfe20179e15cae1c78ee084e9e"
FINAL_SPLIT = "v2_fresh_final_evaluation"
FINAL_SAMPLES = 154_669
THRESHOLD = 0.5


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


class FreshFinalDataset(Dataset):
    """Dataset with no configurable split: it can expose only the locked final rows."""

    def __init__(self, manifest: Path, data_root: Path, transform: Callable) -> None:
        opener = gzip.open if manifest.suffix == ".gz" else open
        with opener(manifest, "rt", newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            required = {"image_url", "occupancy", "site", "weather", "v2_split"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
            self.samples = [
                (row["image_url"], int(row["occupancy"]), row["site"], row["weather"])
                for row in reader
                if row["v2_split"] == FINAL_SPLIT
            ]
        if len(self.samples) != FINAL_SAMPLES:
            raise ValueError(
                f"Locked fresh-final sample count must be {FINAL_SAMPLES}, got {len(self.samples)}"
            )
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_url, label, site, weather = self.samples[index]
        with Image.open(resolve_pklot_image(image_url, self.data_root)) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label, site, weather


def gate_result(v1: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    ufpr04_gain = (
        v2["by_site"]["UFPR04"]["recall_occupied"]
        - v1["by_site"]["UFPR04"]["recall_occupied"]
    )
    overall_f1_delta = v2["overall"]["f1_occupied"] - v1["overall"]["f1_occupied"]
    # A tiny tolerance prevents binary floating-point representation from
    # turning an exact precommitted boundary (for example 0.82 - 0.80) into a fail.
    tolerance = 1e-12
    recall_pass = ufpr04_gain + tolerance >= 0.02
    f1_pass = overall_f1_delta + tolerance >= -0.005
    return {
        "ufpr04_recall_occupied_absolute_gain": ufpr04_gain,
        "ufpr04_minimum_gain": 0.02,
        "ufpr04_gate_pass": recall_pass,
        "overall_f1_occupied_absolute_delta": overall_f1_delta,
        "overall_f1_maximum_decline": 0.005,
        "overall_f1_gate_pass": f1_pass,
        "robustness_gate_pass": recall_pass and f1_pass,
    }


def summarized(accumulator: ConfusionAccumulator) -> dict[str, Any]:
    metrics = accumulator.metrics()
    metrics["false_positives"] = accumulator.fp
    metrics["false_negatives"] = accumulator.fn
    return metrics


def finalize_metrics(
    overall: ConfusionAccumulator,
    by_site: dict[str, ConfusionAccumulator],
    by_weather: dict[str, ConfusionAccumulator],
    inference_seconds: float,
) -> dict[str, Any]:
    site_metrics = {key: summarized(value) for key, value in sorted(by_site.items())}
    return {
        "overall": summarized(overall),
        "by_site": site_metrics,
        "by_weather": {
            key: summarized(value) for key, value in sorted(by_weather.items())
        },
        "macro_site_f1_occupied": float(
            np.mean([value["f1_occupied"] for value in site_metrics.values()])
        ),
        "inference_seconds": inference_seconds,
    }


@torch.inference_mode()
def compare_models(
    models: dict[str, nn.Module],
    loader: DataLoader,
    device: torch.device,
    progress_interval: int,
) -> tuple[dict[str, Any], float]:
    for model in models.values():
        model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    accumulators = {
        name: {
            "overall": ConfusionAccumulator(),
            "site": defaultdict(ConfusionAccumulator),
            "weather": defaultdict(ConfusionAccumulator),
            "inference_seconds": 0.0,
        }
        for name in models
    }
    started = time.monotonic()
    processed = 0
    for inputs, labels, sites, weather_values in loader:
        inputs_device = inputs.to(device)
        labels_device = labels.to(device)
        label_values = labels.tolist()
        for name, model in models.items():
            inference_started = time.monotonic()
            logits = model(inputs_device)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = (probabilities >= THRESHOLD).to(torch.int64)
            losses = criterion(logits, labels_device)
            prediction_values = predictions.cpu().tolist()
            loss_values = losses.cpu().tolist()
            accumulators[name]["inference_seconds"] += time.monotonic() - inference_started
            for label, prediction, loss, site, weather in zip(
                label_values, prediction_values, loss_values, sites, weather_values
            ):
                accumulators[name]["overall"].update(label, prediction, loss)
                accumulators[name]["site"][site].update(label, prediction, loss)
                accumulators[name]["weather"][weather].update(label, prediction, loss)
        processed += len(label_values)
        if processed == FINAL_SAMPLES or processed % progress_interval < len(label_values):
            print(
                json.dumps(
                    {
                        "fresh_final_processed": processed,
                        "fresh_final_total": FINAL_SAMPLES,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    if processed != FINAL_SAMPLES:
        raise RuntimeError(f"Expected {FINAL_SAMPLES} evaluated samples, got {processed}")
    return {
        name: finalize_metrics(
            values["overall"],
            values["site"],
            values["weather"],
            values["inference_seconds"],
        )
        for name, values in accumulators.items()
    }, time.monotonic() - started


def load_locked_resnet(checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_resnet18(pretrained=False, dropout=0.20)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.requires_grad_(False)
    return model.to(device).eval()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("v1_checkpoint", type=Path)
    parser.add_argument("selection_lock", type=Path)
    parser.add_argument("opened_audit", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-interval", type=int, default=10_000)
    args = parser.parse_args()

    if args.opened_audit.exists() or args.result.exists():
        raise FileExistsError("Fresh-final was already opened or completed; refusing to rerun")

    config_sha = require_sha(args.config, EXPECTED_CONFIG_SHA256, "experiment config")
    selection_sha = require_sha(args.selection_lock, EXPECTED_SELECTION_SHA256, "selection lock")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    if selection["status"] != "selected_checkpoint_locked_fresh_final_unopened":
        raise ValueError("Selection lock is not in the unopened state")
    if selection["selected_candidate"] != "v2a_balanced_resnet18":
        raise ValueError("Locked candidate is not V2-A balanced ResNet18")
    if selection["fresh_final_dataset_instantiated"] or selection["fresh_final_images_opened"] != 0:
        raise ValueError("Selection lock does not certify an unopened fresh-final dataset")

    manifest_sha = require_sha(
        args.manifest, config["data"]["v2_manifest_sha256"], "V2 manifest"
    )
    v1_sha = require_sha(
        args.v1_checkpoint, config["v1_baseline"]["checkpoint_sha256"], "V1 checkpoint"
    )
    v2_checkpoint = Path(selection["selected_checkpoint"])
    v2_sha = require_sha(v2_checkpoint, selection["selected_checkpoint_sha256"], "V2 checkpoint")
    if config["data"]["fresh_final_split"] != FINAL_SPLIT:
        raise ValueError("Fresh-final split name differs from the precommitment")
    if config["data"]["fresh_final_samples"] != FINAL_SAMPLES:
        raise ValueError("Fresh-final sample count differs from the precommitment")
    if config["fresh_final_evaluation"]["threshold"] != THRESHOLD:
        raise ValueError("Threshold differs from the precommitment")

    device = select_device(args.device)
    data_root = require_data_root()
    transform = build_v2_transform(training=False, input_size=224)
    dataset = FreshFinalDataset(args.manifest, data_root, transform)
    models = {
        "v1_finetuned_resnet18": load_locked_resnet(args.v1_checkpoint, device),
        "v2a_balanced_resnet18": load_locked_resnet(v2_checkpoint, device),
    }
    runtime = {
        "device": str(device),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "batch_size": args.batch_size,
        "workers": args.workers,
    }
    opened_at = datetime.now(timezone.utc).isoformat()
    opened = {
        "status": "fresh_final_opened_one_time_evaluation_in_progress",
        "opened_at_utc": opened_at,
        "split": FINAL_SPLIT,
        "samples": len(dataset),
        "manifest_sha256": manifest_sha,
        "experiment_config_sha256": config_sha,
        "selection_lock_sha256": selection_sha,
        "models": {
            "v1_finetuned_resnet18": {"checkpoint": str(args.v1_checkpoint), "sha256": v1_sha},
            "v2a_balanced_resnet18": {"checkpoint": str(v2_checkpoint), "sha256": v2_sha},
        },
        "threshold": THRESHOLD,
        "preprocessing": config["preprocessing"],
        "label_definitions": {"0": "EMPTY", "1": "OCCUPIED"},
        "retraining_or_recalibration_allowed": False,
        "runtime": runtime,
    }
    atomic_json(args.opened_audit, opened)
    print(json.dumps(opened, indent=2), flush=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda",
    )
    metrics, evaluation_seconds = compare_models(
        models, loader, device, args.progress_interval
    )
    gate = gate_result(
        metrics["v1_finetuned_resnet18"], metrics["v2a_balanced_resnet18"]
    )
    completed_at = datetime.now(timezone.utc).isoformat()
    result = {
        "status": "milestone_10_5_fresh_final_comparison_complete_locked",
        "opened_at_utc": opened_at,
        "completed_at_utc": completed_at,
        "protocol": {
            "split": FINAL_SPLIT,
            "samples": len(dataset),
            "same_samples_and_order": True,
            "threshold": THRESHOLD,
            "preprocessing": config["preprocessing"],
            "label_definitions": {"0": "EMPTY", "1": "OCCUPIED"},
            "evaluation_code": "src/evaluate_v2_fresh_final.py",
            "model_selection_or_recalibration_after_opening": False,
        },
        "artifact_hashes": {
            "experiment_config_sha256": config_sha,
            "selection_lock_sha256": selection_sha,
            "fresh_final_manifest_sha256": manifest_sha,
            "v1_checkpoint_sha256": v1_sha,
            "v2_checkpoint_sha256": v2_sha,
        },
        "runtime": {**runtime, "joint_evaluation_seconds": evaluation_seconds},
        "models": metrics,
        "comparison_v2_minus_v1": {
            key: metrics["v2a_balanced_resnet18"]["overall"][key]
            - metrics["v1_finetuned_resnet18"]["overall"][key]
            for key in ["accuracy", "precision_occupied", "recall_occupied", "f1_occupied"]
        },
        "robustness_gate": gate,
        "boundaries": {
            "retrained": False,
            "threshold_recalibrated": False,
            "efficientnet_evaluated": False,
            "splits_changed": False,
            "streamlit_demo_modified": False,
            "milestone_11_started": False,
            "future_model_update_from_fresh_final_allowed": False,
        },
    }
    atomic_json(args.result, result)
    opened["status"] = "fresh_final_opened_one_time_evaluation_complete"
    opened["completed_at_utc"] = completed_at
    opened["result"] = str(args.result)
    opened["result_sha256"] = file_sha256(args.result)
    atomic_json(args.opened_audit, opened)
    print(json.dumps({"result": str(args.result), "robustness_gate": gate}, indent=2))


if __name__ == "__main__":
    main()
