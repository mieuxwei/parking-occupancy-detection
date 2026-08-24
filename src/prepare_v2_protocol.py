"""Precommit the date-grouped PKLot V2 robustness protocol without training."""

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

from src.evaluate_cross_domain import file_sha256


V2_SPLITS = (
    "v2_train",
    "v2_validation",
    "v2_fresh_final_evaluation",
    "unused_reserve",
)
DEVELOPMENT_SPLITS = {"v2_train", "v2_validation"}


def stable_order(
    site: str,
    dates: set[str],
    protocol_id: str,
    namespace: str,
) -> list[str]:
    return sorted(
        dates,
        key=lambda date: hashlib.sha256(
            f"{protocol_id}:{namespace}:{site}:{date}".encode("utf-8")
        ).hexdigest(),
    )


def assign_v2_date_groups(
    dates_by_site: dict[str, set[str]],
    v1_split_by_date: dict[tuple[str, str], str],
    protocol_id: str,
    train_dates_per_site: int,
    validation_dates_per_site: int,
    final_fraction: float,
) -> dict[tuple[str, str], str]:
    """Reserve final dates first, then choose fixed-count development dates."""

    if not protocol_id.strip():
        raise ValueError("protocol_id must not be empty")
    if train_dates_per_site < 1 or validation_dates_per_site < 1:
        raise ValueError("Development date counts must be positive")
    if not 0 < final_fraction < 1:
        raise ValueError("final_fraction must be between zero and one")
    assignments: dict[tuple[str, str], str] = {}
    for site, all_dates in sorted(dates_by_site.items()):
        old_heldout = {
            date
            for date in all_dates
            if v1_split_by_date[(site, date)] == "heldout_evaluation"
        }
        final_count = max(1, int(len(all_dates) * final_fraction + 0.5))
        if final_count > len(old_heldout):
            raise ValueError(f"Not enough old held-out dates for final site {site}")
        final_dates = set(
            stable_order(site, old_heldout, protocol_id, "fresh-final")[:final_count]
        )
        development_candidates = all_dates.difference(final_dates)
        needed = train_dates_per_site + validation_dates_per_site
        if needed >= len(development_candidates):
            raise ValueError(f"Not enough dates for development and reserve at {site}")
        ordered_development = stable_order(
            site,
            development_candidates,
            protocol_id,
            "development",
        )
        train_dates = set(ordered_development[:train_dates_per_site])
        validation_dates = set(
            ordered_development[
                train_dates_per_site : train_dates_per_site
                + validation_dates_per_site
            ]
        )
        for date in all_dates:
            if date in final_dates:
                split = "v2_fresh_final_evaluation"
            elif date in train_dates:
                split = "v2_train"
            elif date in validation_dates:
                split = "v2_validation"
            else:
                split = "unused_reserve"
            assignments[(site, date)] = split
    return assignments


def write_gzip_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v1_adaptation_manifest", type=Path)
    parser.add_argument("v2_manifest", type=Path)
    parser.add_argument("date_assignments", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--protocol-id", default="pklot-v2-robustness-v1")
    parser.add_argument("--train-dates-per-site", type=int, default=4)
    parser.add_argument("--validation-dates-per-site", type=int, default=2)
    parser.add_argument("--final-fraction", type=float, default=0.20)
    args = parser.parse_args()

    opener = gzip.open if args.v1_adaptation_manifest.suffix == ".gz" else open
    with opener(
        args.v1_adaptation_manifest,
        "rt",
        newline="",
        encoding="utf-8",
    ) as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        required = {
            "image_id",
            "image_url",
            "site",
            "capture_date",
            "source_frame_id",
            "occupancy",
            "weather",
            "adaptation_split",
        }
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("V1 adaptation manifest is empty")

    dates_by_site: dict[str, set[str]] = defaultdict(set)
    v1_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (row["site"], row["capture_date"])
        dates_by_site[row["site"]].add(row["capture_date"])
        v1_splits[key].add(row["adaptation_split"])
    if any(len(values) != 1 for values in v1_splits.values()):
        raise ValueError("V1 adaptation manifest leaks a site-date across splits")
    v1_split_by_date = {key: next(iter(values)) for key, values in v1_splits.items()}
    assignments = assign_v2_date_groups(
        dates_by_site,
        v1_split_by_date,
        args.protocol_id,
        args.train_dates_per_site,
        args.validation_dates_per_site,
        args.final_fraction,
    )

    output_fields = fields + ["v2_split"]
    image_ids: set[str] = set()
    frame_splits: dict[str, set[str]] = defaultdict(set)
    row_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter(assignments.values())
    counts: dict[str, dict[str, Counter[str]]] = {
        split: {"site": Counter(), "weather": Counter(), "occupancy": Counter()}
        for split in V2_SPLITS
    }
    for row in rows:
        split = assignments[(row["site"], row["capture_date"])]
        row["v2_split"] = split
        if row["image_id"] in image_ids:
            raise ValueError(f"Duplicate image_id: {row['image_id']}")
        image_ids.add(row["image_id"])
        frame_splits[row["source_frame_id"]].add(split)
        row_counts[split] += 1
        for field in counts[split]:
            counts[split][field][row[field]] += 1
    frame_leaks = sum(len(values) > 1 for values in frame_splits.values())
    final_v1_development_overlap = sum(
        split == "v2_fresh_final_evaluation"
        and v1_split_by_date[key] != "heldout_evaluation"
        for key, split in assignments.items()
    )
    if frame_leaks or final_v1_development_overlap:
        raise ValueError(
            "Leakage detected: "
            f"frames={frame_leaks}, final_v1_dev={final_v1_development_overlap}"
        )

    write_gzip_csv(args.v2_manifest, output_fields, rows)
    args.date_assignments.parent.mkdir(parents=True, exist_ok=True)
    with args.date_assignments.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["site", "capture_date", "v1_split", "v2_split"],
            lineterminator="\n",
        )
        writer.writeheader()
        for site, date in sorted(assignments):
            writer.writerow(
                {
                    "site": site,
                    "capture_date": date,
                    "v1_split": v1_split_by_date[(site, date)],
                    "v2_split": assignments[(site, date)],
                }
            )

    summary: dict[str, Any] = {
        "status": "precommitted_no_training",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": args.protocol_id,
        "source_manifest": args.v1_adaptation_manifest.name,
        "source_manifest_sha256": file_sha256(args.v1_adaptation_manifest),
        "v2_manifest": args.v2_manifest.name,
        "v2_manifest_sha256": file_sha256(args.v2_manifest),
        "date_assignments": args.date_assignments.name,
        "date_assignments_sha256": file_sha256(args.date_assignments),
        "error_manifest_read_or_used": False,
        "assignment": {
            "group_key": ["site", "capture_date"],
            "method": "SHA-256 rank within site; reserve final before development",
            "protocol_id": args.protocol_id,
            "train_dates_per_site": args.train_dates_per_site,
            "validation_dates_per_site": args.validation_dates_per_site,
            "final_fraction_of_all_dates_per_site": args.final_fraction,
            "final_candidates": "V1 heldout_evaluation dates only",
            "metric_or_error_based_selection": False,
        },
        "rows": len(rows),
        "unique_image_ids": len(image_ids),
        "unique_source_frames": len(frame_splits),
        "site_date_groups": len(assignments),
        "row_counts": dict(sorted(row_counts.items())),
        "site_date_group_counts": dict(sorted(group_counts.items())),
        "date_assignments_by_split": {
            split: {
                site: sorted(
                    date
                    for (assigned_site, date), assigned_split in assignments.items()
                    if assigned_site == site and assigned_split == split
                )
                for site in sorted(dates_by_site)
            }
            for split in V2_SPLITS
        },
        "counts": {
            split: {
                field: dict(sorted(counter.items()))
                for field, counter in split_counts.items()
            }
            for split, split_counts in counts.items()
        },
        "leakage_checks": {
            "site_date_groups_in_multiple_v2_splits": 0,
            "source_frames_in_multiple_v2_splits": frame_leaks,
            "duplicate_image_ids": len(rows) - len(image_ids),
            "fresh_final_dates_used_by_v1_adaptation": final_v1_development_overlap,
            "fresh_final_dates_used_by_v2_development": 0,
            "passed": True,
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
