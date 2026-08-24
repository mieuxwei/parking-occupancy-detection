# Dataset Storage

Large image datasets must live outside this Git repository on external storage.
The repository contains code, documentation, small metadata, data manifests, and
split assignments only.

## Configure the External Dataset Root

Set `PARKING_DATA_ROOT` to an existing, writable, absolute directory on the
external SSD. No private machine-specific path is stored in this repository.

```bash
export PARKING_DATA_ROOT="/absolute/path/to/external-ssd/parking-datasets"
python3 src/data_paths.py
```

Add the same `export` command, with the actual SSD directory, to the local shell
profile if the setting should persist across terminal sessions. Do not commit the
real path in a tracked file.

If the variable is missing, the check exits with status 2 and prints an explicit
setup instruction. It also rejects relative paths, paths inside this repository,
missing directories, non-directories, and locations that are not readable and
writable.

## External SSD Layout

The storage check reports these planned locations but does not create them:

```text
$PARKING_DATA_ROOT/
├── cnrpark_ext/
│   ├── archives/
│   └── extracted/
└── pklot/
    ├── archives/
    └── extracted/
```

Archives are preserved under `archives/`. Extraction keeps the source archive
structure under `extracted/`. The project must not create physical
`train/`, `validation/`, or `test/` image copies.

## Repository `data/` Policy

The repository's existing `data/` structure remains unchanged and is limited to:

- the small official metadata CSV under ignored local storage;
- tracked provenance and checksum records such as `data/DATA_MANIFEST.md`;
- generated split manifests and summaries under ignored local storage; and
- small future configuration files.

Image archives, extracted images, full-frame datasets, and generated image copies
must never be placed under repository `data/`.

## Portable Split Manifests

`data/processed/cnrpark_ext_split_manifest.csv` records the source-relative
`image_url`, identifiers, metadata, and split assignment. It does not contain
images and does not store a private absolute SSD path. Future data loading will
resolve each relative image path against the configured external root and the
verified extracted archive layout at runtime.

This keeps the same manifest portable across machines while avoiding duplicated
train/validation/test image trees.

## Current Image Archives

The following official assets are stored under
`$PARKING_DATA_ROOT/cnrpark_ext/archives/` and extracted on the external SSD:

- [`CNRPark-Patches-150x150.zip`](https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNRPark-Patches-150x150.zip) — 36,596,809 bytes
- [`CNR-EXT-Patches-150x150.zip`](https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNR-EXT-Patches-150x150.zip) — 449,502,403 bytes

Do not download the 1.1 GB full-frame archive, published split bundle, or trained
models for the initial EDA/classification workflow.

All 157,549 metadata image paths have been verified against the extracted tree.
On this exFAT volume, the extracted small JPGs occupy about 19 GB because of the
filesystem allocation unit. macOS-generated `._*.jpg` AppleDouble sidecars were
removed after extraction; they are not dataset images.

## PKLot Storage

The official `PKLot.tar.gz` is stored under
`$PARKING_DATA_ROOT/pklot/archives/` and its original structure is extracted
under `$PARKING_DATA_ROOT/pklot/extracted/`. The archive contains both full-frame
JPG/XML pairs and official `PKLotSegmented` parking-space patches, so no derived
patch copies or physical train/validation/test directories are needed.

The repository stores only the compressed relative-path manifest, its summary,
and a small conflict exclusion list. On exFAT, the extracted dataset occupies
about 91 GiB because hundreds of thousands of small files each consume a full
allocation unit. Generated `._*` AppleDouble sidecars are not dataset assets and
were removed after verification.

Use the portable path audit at any time with:

```bash
python3 src/audit_image_paths.py 'data/raw/cnrpark_ext/CNRPark+EXT.csv'
```

Full decode and content QA uses the ignored local virtual environment:

```bash
.venv/bin/python src/audit_images.py \
  'data/processed/cnrpark_ext_split_manifest.csv' \
  'data/processed/cnrpark_ext_image_qa.json'
```
