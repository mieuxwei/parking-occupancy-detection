"""Analyze errors from the locked PKLot fine-tuned held-out evaluation."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image, ImageDraw
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.data_paths import require_data_root, resolve_pklot_image
from src.dataset import build_resnet_transform
from src.evaluate_cross_domain import ConfusionAccumulator, file_sha256
from src.resnet18_transfer import build_resnet18
from src.train_baseline import select_device


ERROR_FIELDS = [
    "image_id",
    "image_url",
    "true_label",
    "predicted_label",
    "error_type",
    "confidence",
    "probability_occupied",
    "site",
    "physical_location",
    "weather",
    "capture_date",
    "capture_time",
    "source_frame_id",
    "slot_id",
]


class PKLotErrorAnalysisDataset(Dataset[tuple[Tensor, int, int]]):
    """Return held-out image tensors with a stable row index into metadata."""

    def __init__(
        self,
        manifest: Path,
        data_root: Path,
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        opener = gzip.open if manifest.suffix == ".gz" else open
        with opener(manifest, "rt", newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            required = {
                "image_id",
                "image_url",
                "occupancy",
                "site",
                "physical_location",
                "weather",
                "capture_date",
                "capture_time",
                "source_frame_id",
                "slot_id",
                "adaptation_split",
            }
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
            self.rows = [
                row for row in reader if row["adaptation_split"] == "heldout_evaluation"
            ]
        if not self.rows:
            raise ValueError("No heldout_evaluation samples in adaptation manifest")
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        row = self.rows[index]
        image_path = resolve_pklot_image(row["image_url"], self.data_root)
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["occupancy"]), index


def make_error_row(
    metadata: dict[str, str],
    label: int,
    prediction: int,
    confidence: float,
    probability_occupied: float,
) -> dict[str, str | int | float]:
    """Combine one wrong prediction with portable manifest metadata."""

    return {
        "image_id": metadata["image_id"],
        "image_url": metadata["image_url"],
        "true_label": label,
        "predicted_label": prediction,
        "error_type": "false_positive" if label == 0 else "false_negative",
        "confidence": confidence,
        "probability_occupied": probability_occupied,
        "site": metadata["site"],
        "physical_location": metadata["physical_location"],
        "weather": metadata["weather"],
        "capture_date": metadata["capture_date"],
        "capture_time": metadata["capture_time"],
        "source_frame_id": metadata["source_frame_id"],
        "slot_id": metadata["slot_id"],
    }


def accumulator_summary(
    accumulators: dict[str, ConfusionAccumulator],
) -> dict[str, dict[str, Any]]:
    return {key: value.metrics() for key, value in sorted(accumulators.items())}


def rank_errors(
    errors: list[dict[str, Any]], error_type: str, limit: int
) -> list[dict[str, Any]]:
    """Return the most confidently wrong examples with deterministic tie-breaking."""

    candidates = [row for row in errors if row["error_type"] == error_type]
    return sorted(
        candidates,
        key=lambda row: (-float(row["confidence"]), str(row["image_id"])),
    )[:limit]


def representative_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select one most-confident error per type/site/weather combination."""

    representatives: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in errors:
        key = (str(row["error_type"]), str(row["site"]), str(row["weather"]))
        current = representatives.get(key)
        if current is None or (
            float(row["confidence"]), str(row["image_id"])
        ) > (float(current["confidence"]), str(current["image_id"])):
            representatives[key] = row
    return [representatives[key] for key in sorted(representatives)]


def create_contact_sheet(
    rows: list[dict[str, Any]],
    data_root: Path,
    destination: Path,
    columns: int = 4,
) -> None:
    """Create a compact derivative preview; source images remain external."""

    if not rows:
        return
    tile_width, image_height, caption_height = 300, 240, 64
    tile_height = image_height + caption_height
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, sheet_rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        image_path = resolve_pklot_image(str(row["image_url"]), data_root)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            scale = min(
                (tile_width - 12) / image.width,
                (image_height - 12) / image.height,
            )
            image = image.resize(
                (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        image_x = x + (tile_width - image.width) // 2
        image_y = y + (image_height - image.height) // 2
        sheet.paste(image, (image_x, image_y))
        caption = (
            f"{row['site']} | {row['weather']} | {row['capture_date']}\n"
            f"true={row['true_label']} pred={row['predicted_label']} "
            f"conf={float(row['confidence']):.4f}\n{row['image_id']}"
        )
        draw.multiline_text((x + 5, y + image_height + 2), caption, fill="black", spacing=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=90, optimize=True)


@torch.inference_mode()
def analyze(
    model: nn.Module,
    loader: DataLoader,
    dataset: PKLotErrorAnalysisDataset,
    device: torch.device,
    progress_interval: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    criterion = nn.CrossEntropyLoss(reduction="none")
    overall = ConfusionAccumulator()
    grouped: dict[str, defaultdict[str, ConfusionAccumulator]] = {
        "site": defaultdict(ConfusionAccumulator),
        "weather": defaultdict(ConfusionAccumulator),
        "capture_date": defaultdict(ConfusionAccumulator),
    }
    errors: list[dict[str, Any]] = []
    processed = 0
    started = time.monotonic()
    model.eval()
    for inputs, labels, indices in loader:
        logits = model(inputs.to(device))
        probabilities = logits.softmax(dim=1).cpu()
        predictions = probabilities.argmax(dim=1)
        losses = criterion(logits, labels.to(device)).cpu()
        for label, prediction, probability, loss, index in zip(
            labels.tolist(),
            predictions.tolist(),
            probabilities.tolist(),
            losses.tolist(),
            indices.tolist(),
        ):
            row = dataset.rows[index]
            confidence = float(max(probability))
            overall.update(label, prediction, loss)
            grouped["site"][row["site"]].update(label, prediction, loss)
            grouped["weather"][row["weather"]].update(label, prediction, loss)
            grouped["capture_date"][row["capture_date"]].update(label, prediction, loss)
            if label != prediction:
                errors.append(
                    make_error_row(
                        row,
                        label,
                        prediction,
                        confidence,
                        float(probability[1]),
                    )
                )
        processed += len(labels)
        if processed == len(dataset) or processed % progress_interval < len(labels):
            print(
                json.dumps(
                    {
                        "processed": processed,
                        "total": len(dataset),
                        "errors_so_far": len(errors),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    metrics = {
        "overall": overall.metrics(),
        "by_site": accumulator_summary(grouped["site"]),
        "by_weather": accumulator_summary(grouped["weather"]),
        "by_capture_date": accumulator_summary(grouped["capture_date"]),
        "duration_seconds": time.monotonic() - started,
    }
    return errors, metrics


def write_error_manifest(rows: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if destination.suffix == ".gz" else open
    with opener(destination, "wt", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=ERROR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("fine_tuning_result", type=Path)
    parser.add_argument("error_manifest", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--image-dir", type=Path, default=Path("images"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--worst-per-type", type=int, default=12)
    parser.add_argument("--progress-interval", type=int, default=50000)
    args = parser.parse_args()

    started = time.monotonic()
    data_root = require_data_root()
    device = select_device(args.device)
    prior_result = json.loads(args.fine_tuning_result.read_text(encoding="utf-8"))
    checkpoint_sha = file_sha256(args.checkpoint)
    if checkpoint_sha != prior_result["fine_tuned_checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 differs from the completed Milestone 8 result")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if file_sha256(args.manifest) != config["adaptation_manifest_sha256"]:
        raise ValueError("Adaptation manifest SHA-256 differs from the training record")
    if config.get("cnr_ext_test_reopened") is not False:
        raise ValueError("Fine-tuning record does not preserve the CNR-EXT test boundary")

    dataset = PKLotErrorAnalysisDataset(
        args.manifest,
        data_root,
        build_resnet_transform(False, args.input_size),
    )
    expected_samples = int(config["heldout_evaluation_samples"])
    if len(dataset) != expected_samples:
        raise ValueError(f"Expected {expected_samples} held-out samples, found {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda",
    )
    model = build_resnet18(pretrained=False, dropout=0.20)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.requires_grad_(False).to(device).eval()
    errors, metrics = analyze(model, loader, dataset, device, args.progress_interval)

    if metrics["overall"]["confusion_matrix"] != prior_result["heldout_after"]["overall"][
        "confusion_matrix"
    ]:
        raise ValueError("Recomputed held-out confusion matrix differs from Milestone 8")
    errors.sort(key=lambda row: str(row["image_id"]))
    write_error_manifest(errors, args.error_manifest)
    worst_fp = rank_errors(errors, "false_positive", args.worst_per_type)
    worst_fn = rank_errors(errors, "false_negative", args.worst_per_type)
    representatives = representative_errors(errors)
    fp_sheet = args.image_dir / "pklot_false_positive_worst_cases.jpg"
    fn_sheet = args.image_dir / "pklot_false_negative_worst_cases.jpg"
    representative_sheet = args.image_dir / "pklot_error_representatives.jpg"
    create_contact_sheet(worst_fp, data_root, fp_sheet)
    create_contact_sheet(worst_fn, data_root, fn_sheet)
    create_contact_sheet(representatives, data_root, representative_sheet)

    false_positives = sum(row["error_type"] == "false_positive" for row in errors)
    false_negatives = sum(row["error_type"] == "false_negative" for row in errors)
    date_ranking = sorted(
        metrics["by_capture_date"].items(),
        key=lambda item: (
            -(item[1]["confusion_matrix"][0][1] + item[1]["confusion_matrix"][1][0]),
            item[0],
        ),
    )
    result = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "checkpoint": args.checkpoint.name,
            "checkpoint_sha256": checkpoint_sha,
            "adaptation_manifest": args.manifest.name,
            "adaptation_manifest_sha256": file_sha256(args.manifest),
            "split": "heldout_evaluation",
            "threshold": 0.5,
            "weights_changed": False,
            "threshold_changed": False,
            "cnr_ext_test_reopened": False,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "input_size": args.input_size,
            "device": str(device),
        },
        "heldout_metrics_recomputed": metrics,
        "error_counts": {
            "total": len(errors),
            "false_positive": false_positives,
            "false_negative": false_negatives,
            "false_negative_share": false_negatives / len(errors),
        },
        "error_manifest": {
            "file": args.error_manifest.name,
            "sha256": file_sha256(args.error_manifest),
            "rows": len(errors),
            "contains_absolute_paths": False,
        },
        "most_confident_false_positives": worst_fp,
        "most_confident_false_negatives": worst_fn,
        "representative_errors_by_type_site_weather": representatives,
        "dates_with_most_errors": [
            {
                "capture_date": date,
                "errors": values["confusion_matrix"][0][1]
                + values["confusion_matrix"][1][0],
                "false_positive": values["confusion_matrix"][0][1],
                "false_negative": values["confusion_matrix"][1][0],
                "samples": values["samples"],
            }
            for date, values in date_ranking[:10]
        ],
        "contact_sheets": [
            fp_sheet.name,
            fn_sheet.name,
            representative_sheet.name,
        ],
        "total_duration_seconds": time.monotonic() - started,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["error_counts"], indent=2), flush=True)
    print(f"Saved result to {args.result}", flush=True)


if __name__ == "__main__":
    main()
