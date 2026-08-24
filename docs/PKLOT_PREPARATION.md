# PKLot Cross-Domain Preparation

Milestone 7 data preparation completed on 2026-08-22. No model inference,
fine-tuning, or CNR-EXT test evaluation was performed during this step.

## Acquisition and Integrity

The official [`PKLot.tar.gz`](https://www.inf.ufpr.br/vri/databases/PKLot.tar.gz)
was downloaded directly to external storage. The official server and local file
both reported 4,898,276,304 bytes. Local SHA-256:

```text
e89bbc1dc735298c478688d50c7a682fb3b0076a87b6634923132709f2d2fa9b
```

A complete gzip/tar traversal passed. The archive has 721,277 entries under the
single `PKLot` root and contains no absolute or parent-traversal paths. It is
licensed under CC BY 4.0 and requires attribution to the PKLot authors and paper.

## Extracted Assets

The original archive structure is preserved on the external SSD:

- 12,417 full-frame JPGs
- 12,416 XML annotations
- 695,851 official segmented parking-space JPGs

One full frame lacks its XML pair:
`PUCPR/Sunny/2012-11-06/2012-11-06_18_48_46.jpg`. Cross-domain classification
will use the official segmented patches, so this frame is not part of the patch
manifest.

Extraction to exFAT generated 721,277 macOS AppleDouble sidecars and temporarily
consumed almost all free space. The generated `._*` files were removed after
their scope was verified. All actual JPG/XML files and the source archive remain.
The extracted dataset occupies about 91 GiB on this filesystem.

## Manifest

`src/prepare_pklot_manifest.py` generates:

- `data/processed/pklot_manifest.csv.gz`
- `data/processed/pklot_manifest_summary.json`
- `data/PKLOT_EXCLUSIONS.csv`

The gzip manifest is 5,763,824 bytes and has SHA-256:

```text
fd0a38bb56a7cf21ca98ad231a7ac463a9f22c778a15726c5054403406b5ef48
```

It contains 695,695 unique eligible image IDs and 12,508 source frames. Every
`image_url` is relative to the PKLot extraction root. It records site, physical
location, weather, capture date/time, source frame, slot, occupancy, and domain
role without copying images.

Observed eligible distribution:

| Group | Samples |
| --- | ---: |
| PUCPR (`PUC`) | 424,067 |
| UFPR04 | 105,843 |
| UFPR05 | 165,785 |
| Empty | 357,993 |
| Occupied | 337,702 |
| Cloudy | 229,131 |
| Rainy | 99,943 |
| Sunny | 366,621 |

The archive contains 695,851 segmented patches, 48 fewer than the 695,899 count
reported in the PKLot paper. The manifest uses observed files rather than filling
the gap with synthetic rows.

## Label Conflict Exclusions

There are 78 duplicate `(site, frame, slot)` IDs stored once under `Empty` and
once under `Occupied`, totaling 156 paths. For every pair, the two JPG hashes are
identical. Both paths are excluded because the directory labels conflict and
there is no trustworthy basis for choosing one label. The files remain on the
external SSD and are listed with hashes in `data/PKLOT_EXCLUSIONS.csv`.

## Cross-Domain Boundary

All eligible PKLot rows are marked `cross_domain_evaluation` and
`used_for_model_selection` is false. The ResNet18 architecture, weights, and
checkpoint were selected using CNR-EXT before PKLot acquisition. PKLot must not
be used for hyperparameter selection in Milestone 7.

Milestone 8 must define a new date-grouped adaptation/test protocol before using
any PKLot labels for fine-tuning. No physical train/validation/test image folders
will be created.

## Reproduction

With `PARKING_DATA_ROOT` configured and the official archive extracted:

```bash
.venv/bin/python -m src.prepare_pklot_manifest \
  'data/processed/pklot_manifest.csv.gz' \
  'data/processed/pklot_manifest_summary.json' \
  --exclusions 'data/PKLOT_EXCLUSIONS.csv'
```
