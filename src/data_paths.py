"""Validate and describe external dataset storage paths."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Mapping


DATA_ROOT_ENV = "PARKING_DATA_ROOT"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SETUP_EXAMPLE = (
    'export PARKING_DATA_ROOT="/absolute/path/to/external-ssd/parking-datasets"'
)


class DatasetRootError(RuntimeError):
    """Raised when external dataset storage is not configured safely."""


def require_data_root(
    environment: Mapping[str, str] | None = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Return a validated external dataset root or raise a clear error."""

    env = os.environ if environment is None else environment
    raw_value = env.get(DATA_ROOT_ENV, "").strip()
    if not raw_value:
        raise DatasetRootError(
            f"{DATA_ROOT_ENV} is not set. Configure an absolute directory on "
            f"the external SSD before using image data, for example:\n"
            f"  {SETUP_EXAMPLE}"
        )

    root = Path(raw_value).expanduser()
    if not root.is_absolute():
        raise DatasetRootError(
            f"{DATA_ROOT_ENV} must be an absolute path on the external SSD.\n"
            f"Example:\n  {SETUP_EXAMPLE}"
        )

    root = root.resolve()
    repository_root = repository_root.resolve()
    try:
        root.relative_to(repository_root)
    except ValueError:
        pass
    else:
        raise DatasetRootError(
            f"{DATA_ROOT_ENV} points inside the Git repository ({root}). "
            "Choose a directory on the external SSD instead."
        )

    if not root.exists():
        raise DatasetRootError(
            f"{DATA_ROOT_ENV} does not exist: {root}. Create the directory on "
            "the external SSD, then run this check again."
        )
    if not root.is_dir():
        raise DatasetRootError(f"{DATA_ROOT_ENV} is not a directory: {root}")
    if not os.access(root, os.R_OK | os.W_OK):
        raise DatasetRootError(
            f"{DATA_ROOT_ENV} is not readable and writable: {root}"
        )
    return root


def storage_layout(root: Path) -> dict[str, Path]:
    """Return planned paths without creating directories or copying images."""

    cnrpark = root / "cnrpark_ext"
    pklot = root / "pklot"
    return {
        "cnrpark_archives": cnrpark / "archives",
        "cnrpark_extracted": cnrpark / "extracted",
        "pklot_archives": pklot / "archives",
        "pklot_extracted": pklot / "extracted",
    }


def resolve_cnrpark_image(relative_image_url: str, root: Path) -> Path:
    """Resolve a portable manifest path below external CNRPark extraction root."""

    relative_path = PurePosixPath(relative_image_url)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise DatasetRootError(
            f"Unsafe image_url must be a source-relative path: {relative_image_url}"
        )
    if not relative_path.parts:
        raise DatasetRootError("image_url is empty")
    return storage_layout(root)["cnrpark_extracted"].joinpath(*relative_path.parts)


def resolve_pklot_image(relative_image_url: str, root: Path) -> Path:
    """Resolve a portable manifest path below external PKLot extraction root."""

    relative_path = PurePosixPath(relative_image_url)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise DatasetRootError(
            f"Unsafe image_url must be a source-relative path: {relative_image_url}"
        )
    if not relative_path.parts:
        raise DatasetRootError("image_url is empty")
    return storage_layout(root)["pklot_extracted"].joinpath(*relative_path.parts)


def main() -> int:
    try:
        root = require_data_root()
    except DatasetRootError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    free_bytes = shutil.disk_usage(root).free
    print(f"{DATA_ROOT_ENV} is valid: {root}")
    print(f"Free space: {free_bytes:,} bytes")
    print("Planned storage paths (not created by this check):")
    for name, path in storage_layout(root).items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
