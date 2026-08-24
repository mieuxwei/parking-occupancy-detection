"""Build a compressed, relative-path manifest for official PKLot patches."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterator

from src.data_paths import require_data_root, storage_layout


FILENAME_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})"
    r"#(?P<slot>\d+)\.jpg$",
    re.IGNORECASE,
)
SITE_LOCATIONS = {"PUC": "PUCPR", "UFPR04": "UFPR", "UFPR05": "UFPR"}
LABELS = {"Empty": 0, "Occupied": 1}
FIELDS = [
    "image_id",
    "image_url",
    "site",
    "physical_location",
    "weather",
    "capture_date",
    "capture_time",
    "source_frame_id",
    "slot_id",
    "occupancy",
    "domain_role",
]
EXCLUSION_FIELDS = ["image_id", "image_url", "reason", "content_sha256"]


def parse_patch_path(relative_path: PurePosixPath) -> dict[str, str | int]:
    """Parse an official PKLotSegmented path into portable manifest fields."""

    parts = relative_path.parts
    if len(parts) != 7 or parts[:2] != ("PKLot", "PKLotSegmented"):
        raise ValueError(f"Unexpected PKLot patch path: {relative_path}")
    _, _, site, weather, capture_date, label, filename = parts
    if site not in SITE_LOCATIONS:
        raise ValueError(f"Unexpected PKLot site in {relative_path}: {site}")
    if label not in LABELS:
        raise ValueError(f"Unexpected PKLot label in {relative_path}: {label}")
    match = FILENAME_PATTERN.fullmatch(filename)
    if match is None or match.group("date") != capture_date:
        raise ValueError(f"Unexpected PKLot filename in {relative_path}: {filename}")
    capture_time = ":".join(
        [match.group("hour"), match.group("minute"), match.group("second")]
    )
    source_frame_id = f"{site}/{capture_date}/{capture_time}"
    return {
        "image_id": f"{source_frame_id}#{match.group('slot')}",
        "image_url": relative_path.as_posix(),
        "site": site,
        "physical_location": SITE_LOCATIONS[site],
        "weather": weather.lower(),
        "capture_date": capture_date,
        "capture_time": capture_time,
        "source_frame_id": source_frame_id,
        "slot_id": int(match.group("slot")),
        "occupancy": LABELS[label],
        "domain_role": "cross_domain_evaluation",
    }


def iter_rows(extraction_root: Path) -> Iterator[dict[str, str | int]]:
    segmented_root = extraction_root / "PKLot" / "PKLotSegmented"
    if not segmented_root.is_dir():
        raise FileNotFoundError(
            f"Official PKLotSegmented directory is missing: {segmented_root}"
        )
    paths = sorted(
        path
        for path in segmented_root.rglob("*.jpg")
        if not path.name.startswith("._")
    )
    if not paths:
        raise ValueError(f"No PKLot segmented JPGs found under {segmented_root}")
    for path in paths:
        relative = PurePosixPath(path.relative_to(extraction_root).as_posix())
        yield parse_patch_path(relative)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_conflicting_ids(
    extraction_root: Path,
) -> tuple[set[str], int]:
    first_labels: dict[str, int] = {}
    conflicts: set[str] = set()
    source_rows = 0
    for row in iter_rows(extraction_root):
        source_rows += 1
        image_id = str(row["image_id"])
        label = int(row["occupancy"])
        previous = first_labels.setdefault(image_id, label)
        if previous != label:
            conflicts.add(image_id)
    return conflicts, source_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--exclusions", type=Path, required=True)
    args = parser.parse_args()

    root = require_data_root()
    extraction_root = storage_layout(root)["pklot_extracted"]
    conflicting_ids, source_rows = find_conflicting_ids(extraction_root)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.exclusions.parent.mkdir(parents=True, exist_ok=True)
    counters: dict[str, Counter[str | int]] = {
        "site": Counter(),
        "physical_location": Counter(),
        "weather": Counter(),
        "occupancy": Counter(),
        "capture_date": Counter(),
    }
    frame_ids: set[str] = set()
    image_ids: set[str] = set()
    row_count = 0
    excluded_rows = 0
    conflict_hashes: dict[str, set[str]] = {}

    with args.manifest.open("wb") as raw, args.exclusions.open(
        "w", encoding="utf-8", newline=""
    ) as exclusion_file:
        exclusion_writer = csv.DictWriter(
            exclusion_file, fieldnames=EXCLUSION_FIELDS, lineterminator="\n"
        )
        exclusion_writer.writeheader()
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(
                compressed, encoding="utf-8", newline=""
            ) as text:
                writer = csv.DictWriter(text, fieldnames=FIELDS, lineterminator="\n")
                writer.writeheader()
                for row in iter_rows(extraction_root):
                    image_id = str(row["image_id"])
                    if image_id in conflicting_ids:
                        image_path = extraction_root.joinpath(
                            *PurePosixPath(str(row["image_url"])).parts
                        )
                        content_hash = sha256(image_path)
                        conflict_hashes.setdefault(image_id, set()).add(content_hash)
                        exclusion_writer.writerow(
                            {
                                "image_id": image_id,
                                "image_url": row["image_url"],
                                "reason": "conflicting_occupancy_labels",
                                "content_sha256": content_hash,
                            }
                        )
                        excluded_rows += 1
                        continue
                    if image_id in image_ids:
                        raise ValueError(f"Duplicate non-conflicting PKLot image_id: {image_id}")
                    image_ids.add(image_id)
                    frame_ids.add(str(row["source_frame_id"]))
                    for field in counters:
                        counters[field][row[field]] += 1
                    writer.writerow(row)
                    row_count += 1

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": args.manifest.name,
        "manifest_compression": "gzip",
        "manifest_size_bytes": args.manifest.stat().st_size,
        "manifest_sha256": sha256(args.manifest),
        "source_rows": source_rows,
        "rows": row_count,
        "excluded_rows": excluded_rows,
        "conflicting_image_ids": len(conflicting_ids),
        "conflicting_ids_with_identical_content": sum(
            len(hashes) == 1 for hashes in conflict_hashes.values()
        ),
        "unique_image_ids": len(image_ids),
        "unique_source_frames": len(frame_ids),
        "domain_role": "cross_domain_evaluation",
        "used_for_model_selection": False,
        "counts": {
            field: {str(key): value for key, value in sorted(counter.items())}
            for field, counter in counters.items()
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
