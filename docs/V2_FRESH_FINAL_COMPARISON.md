# Milestone 10.5 Fresh-Final Comparison

## Locked Protocol

The one-time comparison completed on 2026-08-24. Both immutable ResNet18
models were evaluated in the same deterministic pass over exactly 154,669
`v2_fresh_final_evaluation` samples. They used identical 224×224 edge-padding,
bilinear resizing, ImageNet normalization, `EMPTY=0` / `OCCUPIED=1` labels, and
an occupied-probability threshold of 0.5.

No retraining, threshold recalibration, augmentation, split change, candidate
switch, or model update occurred after fresh final opened. EfficientNet-B0 was
not evaluated. The existing Streamlit demo was not changed.

## Results

| Model | Accuracy | Precision | Recall | Occupied F1 | Macro-site F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 fine-tuned ResNet18 | 0.991834 | 0.998594 | 0.985315 | 0.991910 | 0.973232 | 109 | 1,154 |
| V2-A balanced ResNet18 | 0.998894 | 0.998766 | 0.999058 | 0.998912 | 0.998763 | 97 | 74 |
| V2 − V1 | +0.007060 | +0.000172 | +0.013743 | +0.007002 | +0.025531 | -12 | -1,080 |

Confusion matrices use `[[TN, FP], [FN, TP]]`:

- V1: `[[75977, 109], [1154, 77429]]`
- V2-A: `[[75989, 97], [74, 78509]]`

## Per-site Occupied Recall

| Site | V1 | V2-A | Absolute change |
|---|---:|---:|---:|
| PUC | 0.997855 | 0.999690 | +0.001836 |
| UFPR04 | 0.859053 | 1.000000 | +0.140947 |
| UFPR05 | 0.996480 | 0.997955 | +0.001475 |

## Precommitted Robustness Gate

| Condition | Observed | Required | Result |
|---|---:|---:|---|
| UFPR04 occupied-recall absolute gain | +0.140947 | ≥ +0.020000 | Pass |
| Overall occupied-F1 absolute change | +0.007002 | ≥ -0.005000 | Pass |

The combined robustness gate passed. This result is a terminal comparison, not
an input to further model development.

## Audit Artifacts

- Exact result: `results/v2_fresh_final_comparison.json` (SHA-256
  `44652a36777b9d5267e205b6af6d664c9d2a63ea787619fab45ed7764b0997a6`)
- One-time opening audit: `data/V2_FRESH_FINAL_OPENED.json` (SHA-256
  `2036e7df562168e0bfa634f999a00e7848d8e42634e2a42c38e833236e7db3d0`)
- Fresh-final manifest: SHA-256
  `4aa2e976d67983b44708d304343ed6c72d7f054f294106ebf3b14e93dd00c5d7`
- V1 checkpoint: SHA-256
  `e61137bf2fbc259a2b7bd22ecc9840dda8c2668d9f73b7352bbaa3d47809e7ce`
- V2-A checkpoint: SHA-256
  `97b039fa7d4125e993903c4d1b485a7bc8e58d47cf7917c5fef8515e6982d5f9`
- Selection lock: SHA-256
  `60743fb7594e05cd7c2726a2a1029853f99473cfe20179e15cae1c78ee084e9e`

The run used Apple MPS, Python 3.9.6, PyTorch 2.8.0, torchvision 0.23.0,
NumPy 2.0.2, batch size 128, and two data-loader workers. The joint evaluation
took 726.74 seconds. Exact per-site and per-weather metrics are retained in the
result JSON.
