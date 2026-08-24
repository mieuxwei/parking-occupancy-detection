"""Verify every CNRPark+EXT metadata image path on external storage."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from data_paths import DatasetRootError, require_data_root, resolve_cnrpark_image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument(
        "--max-reported-missing",
        type=int,
        default=10,
        help="Maximum missing or unsafe paths to print",
    )
    args = parser.parse_args()

    try:
        root = require_data_root()
    except DatasetRootError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    checked = 0
    missing_count = 0
    unsafe_count = 0
    missing: list[str] = []
    unsafe: list[str] = []
    with args.metadata.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if "image_url" not in (reader.fieldnames or []):
            print("ERROR: metadata is missing the image_url column", file=sys.stderr)
            return 2

        for row in reader:
            image_url = row["image_url"]
            checked += 1
            try:
                image_path = resolve_cnrpark_image(image_url, root)
            except DatasetRootError:
                unsafe_count += 1
                if len(unsafe) < args.max_reported_missing:
                    unsafe.append(image_url)
                continue
            if not image_path.is_file():
                missing_count += 1
                if len(missing) < args.max_reported_missing:
                    missing.append(image_url)

    if unsafe_count or missing_count:
        print(
            f"FAILED: checked {checked:,} metadata rows; "
            f"found {unsafe_count:,} unsafe and {missing_count:,} missing paths.",
            file=sys.stderr,
        )
        for image_url in unsafe:
            print(f"UNSAFE: {image_url}", file=sys.stderr)
        for image_url in missing:
            print(f"MISSING: {image_url}", file=sys.stderr)
        return 1

    print(f"PASS: all {checked:,} metadata image paths exist on external storage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
