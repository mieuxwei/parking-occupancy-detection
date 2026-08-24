# Milestone 10.5 — V2 Training and Validation Selection

## Status and boundaries

Both precommitted V2 candidates completed on 2026-08-24. Exactly one checkpoint
was selected using `v2_validation` only and locked before any fresh-final
inference.

- Training images: `v2_train` only, 78,376 sampled draws per epoch
- Selection images: `v2_validation` only, 42,148 natural-order samples
- Fresh-final dataset instantiated: no
- Fresh-final images opened: 0
- V1 error artifacts read or used: no
- CNR-EXT test reopened: no
- Streamlit Demo updated: no
- Milestone 11 started: no

The executed experiment-config SHA-256 is
`57fb8133760cc7eded11eb77e9c1ce5aa67d5379bfbfe33539b279e95a024957`.
The V2 manifest SHA-256 is
`4aa2e976d67983b44708d304343ed6c72d7f054f294106ebf3b14e93dd00c5d7`.

## Shared configuration

Both candidates used seed 105, Apple MPS, batch sizes 32/128, two workers,
AdamW, weight decay `1e-4`, unweighted cross-entropy, identical mild
augmentation, and a replacement sampler balanced by `(site, occupancy)` cell.
Validation retained its natural distribution and no augmentation.

The exact configuration, including cell counts, learning rates, preprocessing,
and augmentation, is embedded in `results/v2_training_selection.json` and
matches `data/PKLOT_V2_EXPERIMENT_CONFIG.json`.

Runtime versions:

- PyTorch 2.8.0
- torchvision 0.23.0
- NumPy 2.0.2
- Device: MPS

## Training history

### V2-A — Balanced ResNet18

Initialized from the immutable V1 PKLot fine-tuned checkpoint. Both epochs were
full fine-tuning with backbone/classifier learning rates `5e-6`/`2e-5`.

| Epoch | Train loss | Accuracy | Precision | Recall | F1 | Macro-site F1 | Min-site recall | Confusion matrix |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.009764 | 0.999525 | 0.999564 | 0.999346 | 0.999455 | 0.999442 | 0.999024 | `[[23783, 8], [12, 18345]]` |
| 2 | 0.002908 | 0.999478 | 0.999238 | 0.999564 | 0.999401 | 0.999489 | 0.999512 | `[[23777, 14], [8, 18349]]` |

Epoch 2 won because macro-site occupied F1 is the primary criterion, even though
epoch 1 has slightly higher overall F1. Epoch 2 UFPR04 validation metrics are:

- Accuracy: 0.999706
- Precision occupied: 0.998712
- Recall occupied: 1.000000
- F1 occupied: 0.999356
- Confusion matrix: `[[5257, 2], [0, 1551]]`

Checkpoint:

- File: `models/v2a_balanced_resnet18.pt`
- Size: 44,790,987 bytes
- SHA-256:
  `97b039fa7d4125e993903c4d1b485a7bc8e58d47cf7917c5fef8515e6982d5f9`
- Candidate runtime: 2,037.25 seconds

### V2-B — Balanced EfficientNet-B0

Initialized with torchvision ImageNet-1K V1 weights. Epoch 1 trained only the
head at `1e-4`; epochs 2–3 fully fine-tuned with backbone/classifier learning
rates `1e-5`/`5e-5`.

| Epoch | Phase | Train loss | Accuracy | Precision | Recall | F1 | Macro-site F1 | Min-site recall | Confusion matrix |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Head | 0.077188 | 0.984578 | 0.966932 | 0.998747 | 0.982582 | 0.987268 | 0.993167 | `[[23164, 627], [23, 18334]]` |
| 2 | Full | 0.031921 | 0.995919 | 0.991248 | 0.999455 | 0.995334 | 0.997472 | 0.998536 | `[[23629, 162], [10, 18347]]` |
| 3 | Full | 0.005564 | 0.998909 | 0.998096 | 0.999401 | 0.998748 | 0.999104 | 0.998048 | `[[23756, 35], [11, 18346]]` |

Epoch 3 is the EfficientNet-B0 best. Its UFPR04 validation metrics are:

- Accuracy: 0.999853
- Precision occupied: 0.999356
- Recall occupied: 1.000000
- F1 occupied: 0.999678
- Confusion matrix: `[[5258, 1], [0, 1551]]`

Checkpoint:

- File: `models/v2b_balanced_efficientnet_b0.pt`
- Size: 16,343,221 bytes
- SHA-256:
  `d3baa8374b012954aa24892bd53313a6d0735cc87b4ae3d4d8bd6ce688650185`
- Candidate runtime: 3,536.19 seconds

## Validation-only selection

The precommitted lexicographic criterion was:

1. macro-site occupied F1;
2. minimum site occupied recall;
3. overall occupied F1.

| Candidate | Best epoch | Macro-site F1 | Min-site recall | Overall F1 |
|---|---:|---:|---:|---:|
| V2-A ResNet18 | 2 | 0.999489 | 0.999512 | 0.999401 |
| V2-B EfficientNet-B0 | 3 | 0.999104 | 0.998048 | 0.998748 |

V2-A ResNet18 wins on the primary criterion and is the one selected V2 model.
The selection record is locked in `data/V2_SELECTED_CHECKPOINT.json`, SHA-256
`60743fb7594e05cd7c2726a2a1029853f99473cfe20179e15cae1c78ee084e9e`.

Total training/validation/selection runtime was 5,621.12 seconds (93.69
minutes). Runtime is descriptive and was not a selection criterion.

## Next boundary

Stop here for review. The next authorized operation, only after explicit user
approval, is the one-time fair comparison of immutable V1 and locked V2-A on the
same 154,669-sample `v2_fresh_final_evaluation` set. No model changes,
recalibration, candidate switching, or Demo update may occur after opening it.
