"""Run full image QA and metadata EDA for an external CNRPark+EXT manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from data_paths import DatasetRootError, require_data_root, resolve_cnrpark_image


def inspect_image(row: dict[str, str], root: Path) -> dict[str, Any]:
    image_url = row["image_url"]
    result: dict[str, Any] = {
        "image_url": image_url,
        "occupancy": row.get("occupancy", ""),
        "date_split": row.get("date_split", ""),
        "camera_split": row.get("camera_split", ""),
    }
    try:
        path = resolve_cnrpark_image(image_url, root)
        payload = path.read_bytes()
    except (DatasetRootError, OSError) as error:
        result["read_error"] = f"{type(error).__name__}: {error}"
        return result

    result["bytes"] = len(payload)
    result["sha256"] = hashlib.sha256(payload).hexdigest()

    try:
        with Image.open(io.BytesIO(payload)) as image:
            result["pillow_format"] = image.format or "UNKNOWN"
            result["pillow_mode"] = image.mode
            result["pillow_size"] = f"{image.width}x{image.height}"
            image.verify()
    except Exception as error:  # Pillow exposes several format-specific errors.
        result["pillow_error"] = f"{type(error).__name__}: {error}"

    try:
        encoded = np.frombuffer(payload, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            result["opencv_error"] = "imdecode returned None"
        else:
            result["opencv_shape"] = "x".join(str(value) for value in decoded.shape)
            result["opencv_dtype"] = str(decoded.dtype)
            result["pixel_min"] = int(decoded.min())
            result["pixel_max"] = int(decoded.max())
            result["pixel_std"] = float(decoded.std())
            if result["date_split"] == "train":
                if decoded.ndim == 2:
                    rgb = np.repeat(decoded[:, :, None], 3, axis=2)
                else:
                    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                pixels = rgb.reshape(-1, 3).astype(np.float64)
                result["training_pixel_count"] = len(pixels)
                result["training_channel_sum"] = pixels.sum(axis=0).tolist()
                result["training_channel_squared_sum"] = np.square(pixels).sum(
                    axis=0
                ).tolist()
    except Exception as error:
        result["opencv_error"] = f"{type(error).__name__}: {error}"
    return result


def counter(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    return dict(sorted(Counter(row.get(column, "") for row in rows).items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    try:
        root = require_data_root()
    except DatasetRootError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    with args.manifest.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if "image_url" not in (reader.fieldnames or []):
            print("ERROR: manifest is missing the image_url column", file=sys.stderr)
            return 2
        rows = list(reader)

    qa_counters = {
        "pillow_formats": Counter(),
        "pillow_modes": Counter(),
        "pillow_sizes": Counter(),
        "opencv_shapes": Counter(),
        "opencv_dtypes": Counter(),
    }
    error_counts = Counter()
    error_samples: list[dict[str, str]] = []
    file_sizes: list[int] = []
    hashes: dict[str, dict[str, Any]] = {}
    training_pixel_count = 0
    training_channel_sum = np.zeros(3, dtype=np.float64)
    training_channel_squared_sum = np.zeros(3, dtype=np.float64)
    uniform_images = 0
    black_images = 0
    near_uniform_images = 0
    content_warning_samples: list[dict[str, Any]] = []
    black_image_samples: list[dict[str, Any]] = []

    worker = partial(inspect_image, root=root)
    index = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for chunk_start in range(0, len(rows), 2_000):
            chunk = rows[chunk_start : chunk_start + 2_000]
            for result in executor.map(worker, chunk):
                index += 1
                for error_name in ("read_error", "pillow_error", "opencv_error"):
                    if error_name in result:
                        error_counts[error_name] += 1
                        if len(error_samples) < 20:
                            error_samples.append(
                                {
                                    "image_url": result["image_url"],
                                    "error_type": error_name,
                                    "error": result[error_name],
                                }
                            )

                if "bytes" in result:
                    file_sizes.append(result["bytes"])
                if "training_pixel_count" in result:
                    training_pixel_count += result["training_pixel_count"]
                    training_channel_sum += result["training_channel_sum"]
                    training_channel_squared_sum += result[
                        "training_channel_squared_sum"
                    ]
                if "pixel_std" in result:
                    is_uniform = result["pixel_min"] == result["pixel_max"]
                    is_black = result["pixel_max"] == 0
                    is_near_uniform = result["pixel_std"] < 1.0
                    uniform_images += int(is_uniform)
                    black_images += int(is_black)
                    near_uniform_images += int(is_near_uniform)
                    if is_black:
                        black_image_samples.append(
                            {
                                "image_url": result["image_url"],
                                "occupancy": result["occupancy"],
                                "date_split": result["date_split"],
                                "camera_split": result["camera_split"],
                            }
                        )
                    if is_near_uniform and len(content_warning_samples) < 50:
                        content_warning_samples.append(
                            {
                                "image_url": result["image_url"],
                                "occupancy": result["occupancy"],
                                "date_split": result["date_split"],
                                "camera_split": result["camera_split"],
                                "pixel_min": result["pixel_min"],
                                "pixel_max": result["pixel_max"],
                                "pixel_std": result["pixel_std"],
                            }
                        )
                for result_key, counter_name in (
                    ("pillow_format", "pillow_formats"),
                    ("pillow_mode", "pillow_modes"),
                    ("pillow_size", "pillow_sizes"),
                    ("opencv_shape", "opencv_shapes"),
                    ("opencv_dtype", "opencv_dtypes"),
                ):
                    if result_key in result:
                        qa_counters[counter_name][result[result_key]] += 1

                digest = result.get("sha256")
                if digest:
                    info = hashes.setdefault(
                        digest,
                        {
                            "count": 0,
                            "date_splits": set(),
                            "camera_splits": set(),
                            "sample_image_urls": [],
                        },
                    )
                    info["count"] += 1
                    if result["date_split"]:
                        info["date_splits"].add(result["date_split"])
                    if result["camera_split"]:
                        info["camera_splits"].add(result["camera_split"])
                    if len(info["sample_image_urls"]) < 5:
                        info["sample_image_urls"].append(result["image_url"])

                if index % 10_000 == 0 or index == len(rows):
                    print(f"Scanned {index:,}/{len(rows):,} images", flush=True)

    duplicate_groups = [info for info in hashes.values() if info["count"] > 1]
    main_splits = {"train", "validation", "test"}
    cross_date_groups = [
        info
        for info in duplicate_groups
        if len(info["date_splits"].intersection(main_splits)) > 1
    ]
    cross_camera_groups = [
        info
        for info in duplicate_groups
        if len(info["camera_splits"].intersection(main_splits)) > 1
    ]

    def duplicate_sample(info: dict[str, Any]) -> dict[str, Any]:
        return {
            "count": info["count"],
            "date_splits": sorted(info["date_splits"]),
            "camera_splits": sorted(info["camera_splits"]),
            "sample_image_urls": info["sample_image_urls"],
        }

    training_mean = training_channel_sum / training_pixel_count
    training_variance = (
        training_channel_squared_sum / training_pixel_count - np.square(training_mean)
    )
    training_std = np.sqrt(np.maximum(training_variance, 0))

    summary = {
        "rows": len(rows),
        "metadata": {
            "subsets": counter(rows, "subset"),
            "occupancy": counter(rows, "occupancy"),
            "weather": counter(rows, "weather"),
            "cameras": counter(rows, "camera"),
            "capture_dates": counter(rows, "capture_date"),
            "date_split": counter(rows, "date_split"),
            "camera_split": counter(rows, "camera_split"),
        },
        "image_qa": {
            "errors": dict(sorted(error_counts.items())),
            "error_samples": error_samples,
            "file_size_bytes": {
                "minimum": min(file_sizes) if file_sizes else None,
                "maximum": max(file_sizes) if file_sizes else None,
                "mean": sum(file_sizes) / len(file_sizes) if file_sizes else None,
                "total": sum(file_sizes),
            },
            "date_train_rgb_statistics": {
                "pixel_count": training_pixel_count,
                "mean_0_1": (training_mean / 255).tolist(),
                "std_0_1": (training_std / 255).tolist(),
            },
            "content_warnings": {
                "uniform_images": uniform_images,
                "black_images": black_images,
                "black_image_samples": black_image_samples,
                "near_uniform_images_std_lt_1": near_uniform_images,
                "samples": content_warning_samples,
            },
            **{
                name: dict(sorted(values.items()))
                for name, values in qa_counters.items()
            },
        },
        "exact_duplicates": {
            "unique_sha256": len(hashes),
            "duplicate_groups": len(duplicate_groups),
            "files_in_duplicate_groups": sum(
                info["count"] for info in duplicate_groups
            ),
            "cross_date_split_groups": len(cross_date_groups),
            "cross_camera_split_groups": len(cross_camera_groups),
            "cross_date_split_samples": [
                duplicate_sample(info) for info in cross_date_groups[:20]
            ],
            "cross_camera_split_samples": [
                duplicate_sample(info) for info in cross_camera_groups[:20]
            ],
        },
        "versions": {
            "pillow": Image.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    total_errors = sum(error_counts.values())
    print(f"Wrote QA summary to {args.output}")
    print(
        f"Decode errors: {total_errors:,}; exact duplicate groups: "
        f"{len(duplicate_groups):,}; cross-date split groups: "
        f"{len(cross_date_groups):,}"
    )
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
