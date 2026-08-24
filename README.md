# Parking Occupancy Detection

## Project Overview

This computer vision project aims to classify a cropped parking-space image as
`EMPTY` or `OCCUPIED`. The project will study both in-domain classification and
cross-domain generalization across parking lots, cameras, lighting, and weather
conditions.

## Research Questions

- Can a simple CNN provide an effective baseline?
- Does ResNet18 transfer learning outperform a CNN trained from scratch?
- How much does performance drop when a model is evaluated on an unseen parking lot?
- How do weather, lighting, shadows, occlusion, and camera angle affect predictions?
- Can fine-tuning with a small amount of target-domain data improve performance?
- Which image conditions are most frequently associated with errors?

## Planned Datasets

- Primary dataset: CNRPark+EXT
- Secondary external domain: PKLot

Milestone 2 selected CNRPark+EXT as the practical starting point because it
provides official cropped patches and rich camera, time, weather, and slot
metadata. PKLot is reserved for later cross-location evaluation. See the
[dataset research report](docs/DATASET_RESEARCH.md) for the comparison,
limitations, licenses, and deferred acquisition plan. Large datasets will not be
committed to this repository.

The leakage-aware date and camera/slot protocols are documented in the
[data split report](docs/DATA_SPLIT.md).

## Dataset Storage

Code stays in this GitHub repository. Large image archives and extracted images
must be stored on an external SSD through the `PARKING_DATA_ROOT` environment
variable; no machine-specific SSD path is committed. The repository `data/`
directory is reserved for metadata, manifests, summaries, and small config only.

```bash
export PARKING_DATA_ROOT="/absolute/path/to/external-ssd/parking-datasets"
python3 src/data_paths.py
```

The split manifest stores relative image paths and IDs. It does not copy images
or create physical train/validation/test folders. See the
[dataset storage guide](docs/DATA_STORAGE.md).

Image integrity, distributions, quality exclusions, training-only RGB statistics,
and preprocessing decisions are recorded in the
[EDA and image quality report](docs/EDA_REPORT.md).

The Simple CNN configuration, training history, and validation metrics are in the
[baseline report](docs/SIMPLE_CNN_BASELINE.md). The test split remained untouched
through all model selection and was opened once in Milestone 7.

The ResNet18 transfer-learning configuration, training history, and baseline
comparison are in the [transfer-learning report](docs/RESNET18_TRANSFER.md).
PKLot acquisition, integrity checks, manifest construction, and label-conflict
handling are documented in the
[PKLot preparation report](docs/PKLOT_PREPARATION.md).
Frozen-model in-domain and cross-domain metrics are in the
[cross-domain evaluation report](docs/CROSS_DOMAIN_EVALUATION.md).
The audited target adaptation split and fine-tuning before/after results are in
the [adaptation split report](docs/PKLOT_ADAPTATION_SPLIT.md) and
[fine-tuning report](docs/PKLOT_FINETUNING.md).
The locked-model false-positive/false-negative, site, weather, date, and visual
inspection findings are in the [error-analysis report](docs/ERROR_ANALYSIS.md).

## Methodology

The planned workflow is dataset research, leakage-aware data splitting,
exploratory data analysis, preprocessing, baseline training, transfer learning,
cross-domain evaluation, fine-tuning, and error analysis. Group-based or
domain-based splits will be preferred over random image splits.

## Tech Stack

- Python
- PyTorch and torchvision
- OpenCV
- NumPy and Pandas
- scikit-learn
- Matplotlib
- Jupyter Notebook
- Git and GitHub

## Repository Structure

```text
parking-occupancy-detection/
├── data/          # Metadata, manifests, summaries, and small config only
├── notebooks/     # Exploration and experiment notebooks
├── src/           # Reusable project source code
├── models/        # Local model checkpoints and related artifacts
├── results/       # Evaluation outputs and experiment summaries
├── images/        # Documentation and presentation images
├── tests/         # Automated tests
├── docs/          # Supporting project documentation
└── app/           # Future demonstration application
```

## Project Status

Milestone 9 — Error Analysis is complete. On the disjoint 615,653-sample PKLot
held-out set, frozen ResNet18 accuracy/F1 were 0.762163/0.809347 and fine-tuned
accuracy/F1 were 0.986387/0.986545. The target adaptation protocol used six
train dates and six validation dates with zero date/frame leakage. CNR-EXT test
was not reopened. Of 8,381 remaining errors, 7,341 were false negatives; UFPR04
accounted for 6,117 of them. The next milestone is a minimal inference demo.
