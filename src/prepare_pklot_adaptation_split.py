"""Create a deterministic date-grouped PKLot adaptation protocol."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPLITS = ("adaptation_train", "adaptation_validation", "heldout_evaluation")


def stable_group_order(site: str, dates: set[str], seed: int) -> list[str]:
    return sorted(
        dates,
        key=lambda date: hashlib.sha256(
            f"{seed}:{site}:{date}".encode("utf-8")
        ).hexdigest(),
    )


def assign_date_groups(
    dates_by_site: dict[str, set[str]],
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[tuple[str, str], str]:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Adaptation fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Adaptation fractions must leave a held-out evaluation set")
    assignments: dict[tuple[str, str], str] = {}
    for site, dates in sorted(dates_by_site.items()):
        ordered = stable_group_order(site, dates, seed)
        train_count = max(1, int(len(ordered) * train_fraction + 0.5))
        validation_count = max(1, int(len(ordered) * validation_fraction + 0.5))
        if train_count + validation_count >= len(ordered):
            raise ValueError(f"Not enough date groups for held-out site {site}")
        for index, date in enumerate(ordered):
            if index < train_count:
                split = "adaptation_train"
            elif index < train_count + validation_count:
                split = "adaptation_validation"
            else:
                split = "heldout_evaluation"
            assignments[(site, date)] = split
    return assignments


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "image_id",
            "site",
            "capture_date",
            "source_frame_id",
            "occupancy",
            "weather",
            "domain_role",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        raise ValueError("PKLot manifest is empty")
    if any(row["domain_role"] != "cross_domain_evaluation" for row in rows):
        raise ValueError("Unexpected non-evaluation row in PKLot manifest")
    return fields, rows


def write_gzip_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    args = parser.parse_args()

    fields, rows = read_rows(args.manifest)
    dates_by_site: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        dates_by_site[row["site"]].add(row["capture_date"])
    assignments = assign_date_groups(
        dates_by_site,
        args.seed,
        args.train_fraction,
        args.validation_fraction,
    )
    split_field = "adaptation_split"
    if split_field in fields:
        raise ValueError(f"Manifest already contains {split_field}")
    output_fields = fields + [split_field]
    date_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    frame_splits: dict[str, set[str]] = defaultdict(set)
    image_ids: set[str] = set()
    counts: dict[str, dict[str, Counter[str]]] = {
        split: {
            "site": Counter(),
            "weather": Counter(),
            "occupancy": Counter(),
        }
        for split in SPLITS
    }
    row_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    for key, split in assignments.items():
        group_counts[split] += 1
    for row in rows:
        split = assignments[(row["site"], row["capture_date"])]
        row[split_field] = split
        image_id = row["image_id"]
        if image_id in image_ids:
            raise ValueError(f"Duplicate image_id: {image_id}")
        image_ids.add(image_id)
        date_splits[(row["site"], row["capture_date"])].add(split)
        frame_splits[row["source_frame_id"]].add(split)
        row_counts[split] += 1
        for field in counts[split]:
            counts[split][field][row[field]] += 1
    date_leaks = sum(len(splits) > 1 for splits in date_splits.values())
    frame_leaks = sum(len(splits) > 1 for splits in frame_splits.values())
    if date_leaks or frame_leaks:
        raise ValueError(
            f"Split leakage detected: date_groups={date_leaks}, frames={frame_leaks}"
        )

    write_gzip_csv(args.output, output_fields, rows)
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": args.manifest.name,
        "manifest": args.output.name,
        "manifest_size_bytes": args.output.stat().st_size,
        "manifest_sha256": file_sha256(args.output),
        "protocol": {
            "group_key": ["site", "capture_date"],
            "selection": "SHA-256 rank within site",
            "seed": args.seed,
            "train_fraction_of_dates_per_site": args.train_fraction,
            "validation_fraction_of_dates_per_site": args.validation_fraction,
            "metric_based_date_selection": False,
        },
        "rows": len(rows),
        "unique_image_ids": len(image_ids),
        "unique_source_frames": len(frame_splits),
        "site_date_groups": len(assignments),
        "row_counts": dict(sorted(row_counts.items())),
        "site_date_group_counts": dict(sorted(group_counts.items())),
        "date_assignments": {
            split: {
                site: sorted(
                    date
                    for (assigned_site, date), assigned_split in assignments.items()
                    if assigned_site == site and assigned_split == split
                )
                for site in sorted(dates_by_site)
            }
            for split in SPLITS
        },
        "counts": {
            split: {
                field: dict(sorted(counter.items()))
                for field, counter in split_counts.items()
            }
            for split, split_counts in counts.items()
        },
        "leakage_checks": {
            "site_date_groups_in_multiple_splits": date_leaks,
            "source_frames_in_multiple_splits": frame_leaks,
            "duplicate_image_ids": len(rows) - len(image_ids),
            "passed": True,
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
