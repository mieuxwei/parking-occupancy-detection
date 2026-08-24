# Data Split and Leakage Audit

Milestone 3 completed on 2026-08-21 using the official CNRPark+EXT metadata CSV.
No image archive was downloaded.

## Metadata Inventory

| Item | Observed value |
| --- | ---: |
| Metadata rows | 157,549 |
| Unique image paths | 157,549 |
| Duplicate image paths | 0 |
| CNR-EXT rows | 144,965 |
| CNRPark rows | 12,584 |
| Empty / free (`0`) | 69,866 |
| Occupied / busy (`1`) | 87,683 |
| Sunny (`S`) | 75,762 |
| Overcast (`O`) | 44,243 |
| Rainy (`R`) | 37,544 |
| Unique cross-camera timestamp groups | 3,416 |
| Missing `occupant_changed` values | 69,931 |

The metadata fields are `camera`, `datetime`, `day`, `hour`, `image_url`,
`minute`, `month`, `occupancy`, `slot_id`, `weather`, `year`, and
`occupant_changed`. The incomplete `occupant_changed` field is not used for
grouping or splitting.

## Protocol A — Date-Grouped Primary Split

This is the primary protocol for the future in-domain baseline. It uses only the
CNR-EXT subset. Every date—and therefore every frame/timestamp group on that
date—belongs to exactly one split. Validation and test each contain one sunny,
one overcast, and one rainy date.

| Split | Dates | Rows | Empty | Occupied |
| --- | --- | ---: | ---: | ---: |
| Train | 17 dates | 106,833 | 46,706 | 60,127 |
| Validation | 2015-11-22, 2015-12-03, 2016-01-08 | 18,938 | 9,543 | 9,395 |
| Test | 2015-11-12, 2015-11-29, 2016-01-14 | 19,155 | 9,399 | 9,756 |
| Excluded quality | 6 dates | 39 | 36 | 3 |
| Cross-subset test | CNRPark: 2015-07-03 and 2015-07-08 | 12,584 | 4,182 | 8,402 |

The 17 training dates are recorded in the generated summary file. The CNRPark
rows are kept out of training and serve as a separate cross-subset diagnostic;
they are not a different physical parking location.

## Protocol B — Camera-and-Slot Holdout

This secondary protocol measures viewpoint generalization within the CNR parking
area. Camera `02` is validation and camera `09` is test. CNR-EXT uses global
slot IDs, and adjacent cameras share some slots. To prevent those physical slots
from crossing split boundaries, all matching rows from potential training
cameras are marked `excluded_overlap`.

| Assignment | Cameras | Rows | Empty | Occupied |
| --- | --- | ---: | ---: | ---: |
| Train | 01, 03, 04, 05, 06, 07, 08 | 116,706 | 54,166 | 62,540 |
| Validation | 02 | 4,095 | 1,454 | 2,641 |
| Test | 09 | 12,942 | 5,584 | 7,358 |
| Excluded shared slots | 01, 03, 04, 08 | 11,183 | 4,444 | 6,739 |
| Excluded quality | 06, 07, 08, 09 | 39 | 36 | 3 |
| Cross-subset test | A, B | 12,584 | 4,182 | 8,402 |

Camera and global slot IDs are disjoint across train, validation, and test. This
protocol is still a camera-domain test inside one parking area, not evidence of
cross-location generalization.

## Automated Leakage Checks

`src/prepare_splits.py` fails before writing output if it detects:

- missing required metadata columns;
- duplicate `image_url` values;
- occupancy labels outside `0` and `1`;
- capture dates shared by primary train/validation/test;
- frame/timestamp groups shared by primary train/validation/test;
- cameras shared by camera-protocol train/validation/test; or
- global slot IDs shared by camera-protocol train/validation/test.

`data/IMAGE_EXCLUSIONS.csv` additionally assigns confirmed invalid images to
`excluded_quality` before the leakage checks. The tracked config currently lists
39 uniformly black placeholders found during full image QA.

The audit passed for all 157,549 metadata rows.

## Local Outputs

The following generated files are intentionally ignored by Git:

- `data/processed/cnrpark_ext_split_manifest.csv`
- `data/processed/cnrpark_ext_split_summary.json`
- `data/processed/cnrpark_ext_image_qa.json`

The manifest contains metadata, IDs, source-relative image paths, and split
assignments only. It does not copy image files or contain an absolute external
SSD path. Images will be resolved through `PARKING_DATA_ROOT` at runtime; see
`docs/DATA_STORAGE.md`.

Reproduce them from the repository root with:

```bash
python3 src/prepare_splits.py \
  'data/raw/cnrpark_ext/CNRPark+EXT.csv' \
  'data/processed/cnrpark_ext_split_manifest.csv' \
  'data/processed/cnrpark_ext_split_summary.json'
```

## Remaining Limitations

- All metadata paths now resolve to extracted image files on the external SSD.
- Pillow/OpenCV decode integrity, dimensions, modes, exact byte duplicates, and
  uniform-image warnings have been audited; see `docs/EDA_REPORT.md`.
- Date choices are fixed before model development and must not be adjusted based
  on future test performance.
- PKLot cross-location splits remain deferred until that dataset is acquired.
