# Simple CNN Baseline

Milestone 5 completed on 2026-08-21. This report contains validation results only;
the primary date-grouped test split was not loaded or evaluated.

## Experiment Configuration

| Setting | Value |
| --- | --- |
| Dataset | CNR-EXT |
| Protocol | Primary date-grouped split |
| Train samples | 106,833 |
| Validation samples | 18,938 |
| Input | 128×128 RGB |
| Model | SimpleCNN |
| Trainable parameters | 98,178 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Loss | Cross-entropy, no class weights |
| Batch size | 64 |
| Epoch limit | 5 |
| Early-stopping patience | 2 |
| Dropout | 0.30 |
| Seed | 42 |
| Device | Apple MPS |
| Checkpoint criterion | Validation occupied-class F1 |

Training augmentation used horizontal flip (`p=0.5`), rotation within ±5°,
brightness jitter 0.15, and contrast jitter 0.15. Validation preprocessing was
deterministic. Both used runtime RGB conversion, edge padding to square, resize,
and the training-only normalization documented in `docs/EDA_REPORT.md`.

## Training History

| Epoch | Train loss | Validation loss | Validation accuracy | Validation F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.117598 | 0.079601 | 0.969955 | 0.969168 |
| 2 | 0.072765 | 0.068413 | 0.974496 | 0.974318 |
| **3** | **0.061842** | **0.055419** | **0.980885** | **0.980611** |
| 4 | 0.053440 | 0.064290 | 0.975077 | 0.974536 |
| 5 | 0.049278 | 0.056848 | 0.977242 | 0.976712 |

Epoch 3 was selected by the predefined validation F1 criterion. Later epochs did
not replace the checkpoint. Total training time was 565.85 seconds.

## Best Validation Result

- Accuracy: `0.980885`
- Precision, occupied class: `0.986954`
- Recall, occupied class: `0.974348`
- F1, occupied class: `0.980611`
- Validation loss: `0.055419`

Confusion matrix, with true labels as rows and predictions as columns
(`empty`, `occupied`):

```text
[[9422, 121],
 [ 241, 9154]]
```

These numbers are not final test performance. Repeated comparison across epochs
uses validation information, so the untouched test split must remain reserved
for a later final evaluation.

## Artifacts

- Tracked metrics/config: `results/simple_cnn_baseline.json`
- Local ignored checkpoint: `models/simple_cnn_baseline.pt`
- Checkpoint size: 406,393 bytes
- Checkpoint SHA-256:
  `82ade7ce2c19bdd20415787ffa42340a85d61b23a61269ed75189b6bf1c1fb1c`

The checkpoint was loaded into a fresh `SimpleCNN` instance after training and
its config, best epoch, sample counts, and `test_split_used: false` flag were
verified.

## Reproduction

With `PARKING_DATA_ROOT` configured:

```bash
.venv/bin/python -m src.train_baseline \
  'data/processed/cnrpark_ext_split_manifest.csv' \
  'models/simple_cnn_baseline.pt' \
  'results/simple_cnn_baseline.json' \
  --epochs 5 \
  --batch-size 64 \
  --input-size 128 \
  --workers 2 \
  --seed 42 \
  --device mps
```

PyTorch MPS operations are seeded but may not be bit-for-bit deterministic across
hardware or library versions. The PyTorch and NumPy versions used for training
are stored in the result JSON.
