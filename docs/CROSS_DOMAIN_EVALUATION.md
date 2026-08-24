# Cross-Domain Evaluation

Milestone 7 completed on 2026-08-22. The epoch-3 ResNet18 checkpoint was frozen
before PKLot acquisition and evaluation. No weights, preprocessing, or
hyperparameters were changed after seeing CNR-EXT test or PKLot results.

## Protocol

| Setting | Value |
| --- | --- |
| Model | ResNet18, selected in Milestone 6 |
| Checkpoint SHA-256 | `0b8a9e51bfdef525410781c4d57813ef3a12bee508ab9843e0ba9d393bd78b60` |
| Input | 224×224 RGB |
| Preprocessing | Deterministic pad-to-square, resize, ImageNet normalization |
| In-domain set | CNR-EXT date-grouped test, 19,155 samples |
| Cross-domain set | PKLot eligible manifest, 695,695 samples |
| Batch size | 128 |
| Device | Apple MPS |
| Model selection after evaluation | None |

The reserved CNR-EXT test split was opened once only after architecture,
checkpoint, and evaluation settings were fixed. PKLot was not used for model
selection. All results below come from the same frozen checkpoint.

## In-Domain and Cross-Domain Results

| Dataset | Accuracy | Occupied precision | Occupied recall | Occupied F1 |
| --- | ---: | ---: | ---: | ---: |
| CNR-EXT test | 0.988567 | 0.996150 | 0.981345 | 0.988692 |
| PKLot overall | 0.743097 | 0.656489 | 0.987442 | 0.788652 |
| Absolute drop | 0.245470 | — | — | 0.200040 |

CNR-EXT test confusion matrix:

```text
[[9362,   37],
 [ 182, 9574]]
```

PKLot confusion matrix:

```text
[[183508, 174485],
 [  4241, 333461]]
```

The cross-domain model retains very high occupied recall but produces 174,485
false positives. The dominant shift is therefore a strong tendency to classify
empty PKLot spaces as occupied, rather than failure to detect occupied spaces.

## Results by Site

| Site | Samples | Accuracy | Occupied precision | Occupied recall | Occupied F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PUCPR (`PUC`) | 424,067 | 0.741685 | 0.639578 | 0.998434 | 0.779697 |
| UFPR04 | 105,843 | 0.512920 | 0.472203 | 0.999740 | 0.641438 |
| UFPR05 | 165,785 | 0.893663 | 0.872170 | 0.959713 | 0.913850 |

UFPR04 is the most difficult site and accounts for 51,542 false positives out
of 105,843 samples. UFPR05 transfers substantially better, demonstrating that
the domain gap varies strongly with camera viewpoint and site appearance.

## Results by Weather

| Weather | Samples | Accuracy | Occupied precision | Occupied recall | Occupied F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cloudy | 229,131 | 0.701476 | 0.563055 | 0.983895 | 0.716231 |
| Rainy | 99,943 | 0.821768 | 0.782302 | 0.997104 | 0.876738 |
| Sunny | 366,621 | 0.747663 | 0.671604 | 0.985818 | 0.798926 |

Cloudy conditions have the lowest occupied F1. These weather groups are also
correlated with site, date, illumination, and viewpoint, so the table does not
establish weather as an isolated causal factor.

## Interpretation

The 24.547-percentage-point accuracy drop and 20.004-percentage-point occupied-F1
drop confirm that strong in-domain CNR-EXT performance does not transfer directly
to a new parking dataset. The high-recall, low-precision pattern suggests a
systematic visual-domain mismatch affecting the empty class. Site-level variation
shows that one aggregate PKLot score is insufficient to describe generalization.

No threshold calibration or target-domain adaptation was attempted. Such changes
would use PKLot information and belong to a separately defined fine-tuning or
calibration experiment.

## Artifacts and Reproduction

- Result JSON: `results/cross_domain_evaluation.json`
- Evaluation code: `src/evaluate_cross_domain.py`
- PKLot manifest: `data/processed/pklot_manifest.csv.gz`

```bash
.venv/bin/python -m src.evaluate_cross_domain \
  'data/processed/cnrpark_ext_split_manifest.csv' \
  'data/processed/pklot_manifest.csv.gz' \
  'models/resnet18_transfer.pt' \
  'results/cross_domain_evaluation.json' \
  --batch-size 128 \
  --input-size 224 \
  --workers 2 \
  --device mps
```

PKLot inference took 1,880.18 seconds and the complete evaluation took 1,921.42
seconds. Group sample counts and confusion matrices were checked against the
overall 695,695-sample result.
