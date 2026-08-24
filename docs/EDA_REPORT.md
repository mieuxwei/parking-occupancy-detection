# EDA and Image Quality Report

Milestone 4 completed on 2026-08-21. The scan covered every CNRPark+EXT patch on
the external SSD. No model was created or trained.

## Image Integrity

| Check | Result |
| --- | ---: |
| Metadata rows scanned | 157,549 |
| Pillow decode / verify errors | 0 |
| OpenCV decode errors | 0 |
| JPEG files | 157,549 |
| Exact duplicate groups | 1 |
| Files in the duplicate group | 38 |
| Exact duplicate groups crossing active date splits | 0 |
| Exact duplicate groups crossing active camera splits | 0 |
| Quality exclusions | 39 |
| Eligible images after exclusions | 157,510 |

The duplicate group contains 38 byte-identical, uniformly black placeholders
with conflicting labels. A second uniformly black image uses a different JPEG
encoding. All 39 are recorded in `data/IMAGE_EXCLUSIONS.csv` with reason
`uniform_black_placeholder` and assigned `excluded_quality` in both protocols.
The source files remain on the SSD for auditability.

There are 178 images with pixel standard deviation below 1.0, including the 39
excluded black images. The remaining low-variance images are retained because
they may represent difficult low-light scenes; low variance alone is not enough
evidence to relabel or discard them.

## Dimensions and Color Modes

| Property | Count |
| --- | ---: |
| 150×150 | 153,901 (97.685%) |
| Other dimensions | 3,648 (2.315%) |
| RGB | 157,434 (99.927%) |
| Grayscale (`L`) | 115 (0.073%) |
| OpenCV dtype `uint8` | 157,549 |

Non-150×150 shapes range from 113–150 pixels on the shorter side. The loader
must convert grayscale images to RGB and perform size normalization in memory.
No resized image copies will be written.

## Primary CNR-EXT Dataset

The primary date-grouped experiment uses CNR-EXT only after quality exclusions.

| Category | Count | Share |
| --- | ---: | ---: |
| Total eligible | 144,926 | 100% |
| Empty / free (`0`) | 65,648 | 45.298% |
| Occupied / busy (`1`) | 79,278 | 54.702% |
| Sunny | 63,175 | 43.591% |
| Overcast | 44,219 | 30.511% |
| Rainy | 37,532 | 25.897% |

The primary subset contains 23 acquisition dates, 9 cameras, 164 global slot IDs,
and 320 camera-slot pairs. The class imbalance is modest; no automatic
oversampling or class weighting is planned initially. Precision, recall, F1, and
the confusion matrix will still be reported alongside accuracy.

## Final Primary Date Split

| Assignment | Rows | Empty | Occupied |
| --- | ---: | ---: | ---: |
| Train | 106,833 | 46,706 | 60,127 |
| Validation | 18,938 | 9,543 | 9,395 |
| Test | 19,155 | 9,399 | 9,756 |
| Excluded quality | 39 | 36 | 3 |

CNRPark cameras A/B remain a separate 12,584-row cross-subset diagnostic.

## Training-Only RGB Statistics

Statistics were computed only from the 106,833 eligible primary training images,
after converting grayscale inputs to three channels. Validation and test pixels
were not used.

- Mean: `[0.412173, 0.402667, 0.353757]`
- Standard deviation: `[0.175863, 0.171592, 0.179682]`

These values are suitable for the future custom CNN baseline. A future pretrained
ImageNet model should use the normalization expected by its published weights
instead of silently reusing these values.

## Preprocessing Decision

At data-loading time:

1. Filter to `quality_status == include` and the requested split protocol.
2. Resolve `image_url` against `PARKING_DATA_ROOT`; do not copy the image.
3. Decode the JPEG and convert every image to RGB.
4. Pad non-square patches symmetrically to a square while preserving aspect ratio.
5. Resize in memory to the model's declared input size.
6. Convert to a tensor and apply the normalization declared in the experiment
   configuration.

Validation and test transforms must be deterministic. The initial training
augmentation candidates are horizontal flip, small rotations, and mild
brightness/contrast variation. Vertical flips, aggressive crops, and strong
geometric transforms are not planned because they do not match the capture setup
or may remove relevant vehicle evidence.

No augmentation has been implemented yet; its exact parameters belong with the
Milestone 5 baseline configuration.

## Reproduction

```bash
python3 src/prepare_splits.py \
  'data/raw/cnrpark_ext/CNRPark+EXT.csv' \
  'data/processed/cnrpark_ext_split_manifest.csv' \
  'data/processed/cnrpark_ext_split_summary.json'

.venv/bin/python src/audit_images.py \
  'data/processed/cnrpark_ext_split_manifest.csv' \
  'data/processed/cnrpark_ext_image_qa.json'
```

`PARKING_DATA_ROOT` must be set before the image scan. Generated CSV/JSON outputs
are ignored by Git; this report and the exclusion config are the tracked research
record.
