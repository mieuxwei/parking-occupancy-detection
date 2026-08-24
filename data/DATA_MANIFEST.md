# Data Manifest

Large dataset files belong under the external `PARKING_DATA_ROOT`. Repository
`data/` is limited to metadata, manifests, summaries, and small config. This file
records provenance and integrity information without embedding a private SSD
path.

## CNRPark+EXT Metadata

- Local path: `data/raw/cnrpark_ext/CNRPark+EXT.csv`
- Source: [official GitHub release asset](https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNRPark%2BEXT.csv)
- Retrieved: 2026-08-21
- Reported asset size: 18,132,695 bytes
- Local size: 18,132,695 bytes
- SHA-256: `26416454cb8b184006431d0c149d31beabb8ef65d8eb2212cf4a39914a470dc8`
- License: Open Data Commons Open Database License (ODbL) 1.0
- Status: downloaded and integrity fingerprint recorded

No CNRPark full-frame archive, published split bundle, or supplied trained model
has been downloaded.

## External Image Assets

The following archives were downloaded to
`$PARKING_DATA_ROOT/cnrpark_ext/archives/` on 2026-08-21:

- [`CNRPark-Patches-150x150.zip`](https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNRPark-Patches-150x150.zip) — 36,596,809 bytes
  - SHA-256: `2dc9b13892595e599e1f5025ddcce08ef7287babbb978670580c60abd86c5fdb`
- [`CNR-EXT-Patches-150x150.zip`](https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNR-EXT-Patches-150x150.zip) — 449,502,403 bytes
  - SHA-256: `84169527af6ad6aaf51729663b59283bc5f28368ff9ddd7856198eec226056d6`

Both ZIP integrity checks passed. They were extracted under
`$PARKING_DATA_ROOT/cnrpark_ext/extracted/`, producing 12,584 CNRPark JPGs and
144,965 CNR-EXT JPGs. All 157,549 metadata paths were found. Extracted images
remain on the external SSD, and split manifests reference relative paths rather
than copying images into split-specific directories.

macOS created one `._*.jpg` AppleDouble sidecar per image during extraction to
the exFAT SSD. All 157,549 generated sidecars were removed after their one-to-one
relationship was confirmed; the actual JPGs and source ZIP archives were retained.

## PKLot Official Archive

The official archive was downloaded directly to
`$PARKING_DATA_ROOT/pklot/archives/` on 2026-08-22:

- [`PKLot.tar.gz`](https://www.inf.ufpr.br/vri/databases/PKLot.tar.gz)
  - HTTP Content-Length and local size: 4,898,276,304 bytes
  - SHA-256: `e89bbc1dc735298c478688d50c7a682fb3b0076a87b6634923132709f2d2fa9b`
  - License: Creative Commons Attribution 4.0 International
  - Integrity: complete gzip/tar traversal passed; zero unsafe archive paths

The archive was extracted without restructuring under
`$PARKING_DATA_ROOT/pklot/extracted/`. It contains 12,417 full-frame JPGs,
12,416 XML files, and 695,851 official segmented parking-space JPGs. One
full-frame image lacks a matching XML annotation:
`PUCPR/Sunny/2012-11-06/2012-11-06_18_48_46.jpg`.

macOS generated 721,277 `._*` AppleDouble sidecars during extraction to exFAT.
After confirming the real-file counts, only these sidecars were removed; the
source archive and all real JPG/XML files remain. Extraction occupies about
91 GiB on this filesystem.

The compressed local manifest `data/processed/pklot_manifest.csv.gz` contains
695,695 eligible relative-path rows. It excludes both copies of 78 image IDs
found under conflicting Empty and Occupied directories (156 paths total). All
78 pairs have identical file content. The excluded paths and hashes are recorded
in `data/PKLOT_EXCLUSIONS.csv`; the original images remain on external storage.

## Quality Exclusions

Full Pillow/OpenCV QA found no decode errors. It identified 39 uniformly black
placeholder images, including a 38-file exact-duplicate group with conflicting
labels. Their relative paths are tracked in `data/IMAGE_EXCLUSIONS.csv`; the
files remain on external storage but are assigned `excluded_quality` in every
split protocol.
