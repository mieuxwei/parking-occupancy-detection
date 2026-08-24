"""Create and audit leakage-aware split manifests for CNRPark+EXT metadata."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


REQUIRED_COLUMNS = {
    "camera",
    "datetime",
    "day",
    "image_url",
    "month",
    "occupancy",
    "slot_id",
    "weather",
    "year",
}

# Each primary split contains one sunny, one overcast, and one rainy date.
DATE_VALIDATION = {"2015-11-22", "2015-12-03", "2016-01-08"}
DATE_TEST = {"2015-11-12", "2015-11-29", "2016-01-14"}
CAMERA_VALIDATION = "02"
CAMERA_TEST = "09"
DEFAULT_EXCLUSIONS = (
    Path(__file__).resolve().parents[1] / "data" / "IMAGE_EXCLUSIONS.csv"
)


def capture_date(row: dict[str, str]) -> str:
    return f"{int(row['year']):04d}-{int(row['month']):02d}-{int(row['day']):02d}"


def dataset_subset(camera: str) -> str:
    return "CNRPark" if camera in {"A", "B"} else "CNR-EXT"


def date_split(row: dict[str, str]) -> str:
    if row["subset"] == "CNRPark":
        return "cross_subset_test"
    if row["capture_date"] in DATE_VALIDATION:
        return "validation"
    if row["capture_date"] in DATE_TEST:
        return "test"
    return "train"


def camera_split(row: dict[str, str], reserved_slots: set[str]) -> str:
    if row["subset"] == "CNRPark":
        return "cross_subset_test"
    if row["camera"] == CAMERA_VALIDATION:
        return "validation"
    if row["camera"] == CAMERA_TEST:
        return "test"
    if row["slot_id"] in reserved_slots:
        return "excluded_overlap"
    return "train"


def read_metadata(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        missing = REQUIRED_COLUMNS.difference(fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("Metadata file contains no data rows")
    return rows, fieldnames


def read_exclusions(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not {"image_url", "reason"}.issubset(reader.fieldnames or []):
            raise ValueError("Exclusion config requires image_url and reason columns")
        exclusions = {row["image_url"]: row["reason"] for row in reader}
    if not exclusions:
        raise ValueError("Exclusion config contains no rows")
    return exclusions


def enrich_rows(rows: list[dict[str, str]], exclusions: dict[str, str]) -> None:
    for row in rows:
        row["subset"] = dataset_subset(row["camera"])
        row["capture_date"] = capture_date(row)
        # Group all parking-space patches captured at the same event. For
        # CNR-EXT this deliberately spans cameras with a shared timestamp.
        row["frame_group"] = f"{row['subset']}:{row['datetime']}"
        row["date_split"] = date_split(row)

    validation_slots = {
        row["slot_id"] for row in rows if row["camera"] == CAMERA_VALIDATION
    }
    test_slots = {row["slot_id"] for row in rows if row["camera"] == CAMERA_TEST}
    if validation_slots.intersection(test_slots):
        raise ValueError("Validation and test cameras share global slot IDs")
    reserved_slots = validation_slots.union(test_slots)

    for row in rows:
        row["camera_split"] = camera_split(row, reserved_slots)
        row["quality_status"] = exclusions.get(row["image_url"], "include")
        if row["quality_status"] != "include":
            row["date_split"] = "excluded_quality"
            row["camera_split"] = "excluded_quality"

    unmatched = set(exclusions).difference(row["image_url"] for row in rows)
    if unmatched:
        raise ValueError(f"Exclusions not found in metadata: {sorted(unmatched)[:5]}")


def assert_disjoint_groups(
    rows: list[dict[str, str]], split_column: str, group_column: str
) -> None:
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = row[split_column]
        if split in {"train", "validation", "test"}:
            groups[split].add(row[group_column])

    for left, right in combinations(("train", "validation", "test"), 2):
        overlap = groups[left].intersection(groups[right])
        if overlap:
            raise ValueError(
                f"{split_column} leaks {len(overlap)} {group_column} values "
                f"between {left} and {right}"
            )


def audit(rows: list[dict[str, str]]) -> None:
    image_urls = [row["image_url"] for row in rows]
    if len(image_urls) != len(set(image_urls)):
        raise ValueError("Duplicate image_url values found")

    invalid_labels = {row["occupancy"] for row in rows}.difference({"0", "1"})
    if invalid_labels:
        raise ValueError(f"Invalid occupancy labels: {sorted(invalid_labels)}")

    assert_disjoint_groups(rows, "date_split", "capture_date")
    assert_disjoint_groups(rows, "date_split", "frame_group")
    assert_disjoint_groups(rows, "camera_split", "camera")
    assert_disjoint_groups(rows, "camera_split", "slot_id")


def split_summary(rows: list[dict[str, str]], split_column: str) -> dict[str, object]:
    summary: dict[str, object] = {}
    for split in (
        "train",
        "validation",
        "test",
        "excluded_overlap",
        "excluded_quality",
        "cross_subset_test",
    ):
        selected = [row for row in rows if row[split_column] == split]
        summary[split] = {
            "rows": len(selected),
            "occupancy": dict(sorted(Counter(row["occupancy"] for row in selected).items())),
            "weather": dict(sorted(Counter(row["weather"] for row in selected).items())),
            "cameras": sorted({row["camera"] for row in selected}),
            "dates": sorted({row["capture_date"] for row in selected}),
            "frame_groups": len({row["frame_group"] for row in selected}),
        }
    return summary


def build_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "subsets": dict(sorted(Counter(row["subset"] for row in rows).items())),
        "occupancy": dict(sorted(Counter(row["occupancy"] for row in rows).items())),
        "weather": dict(sorted(Counter(row["weather"] for row in rows).items())),
        "unique_image_urls": len({row["image_url"] for row in rows}),
        "unique_frame_groups": len({row["frame_group"] for row in rows}),
        "missing_occupant_changed": sum(
            not row.get("occupant_changed", "").strip() for row in rows
        ),
        "date_protocol": split_summary(rows, "date_split"),
        "camera_protocol": split_summary(rows, "camera_split"),
        "audit": "passed",
    }


def write_manifest(
    path: Path, rows: list[dict[str, str]], original_fields: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    added_fields = [
        "subset",
        "capture_date",
        "frame_group",
        "date_split",
        "camera_split",
        "quality_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=original_fields + added_fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    args = parser.parse_args()

    rows, original_fields = read_metadata(args.metadata)
    exclusions = read_exclusions(args.exclusions)
    enrich_rows(rows, exclusions)
    audit(rows)
    write_manifest(args.manifest, rows, original_fields)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(build_summary(rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows):,} rows; leakage audit passed.")


if __name__ == "__main__":
    main()
