# ResNet18 Transfer Learning

Milestone 6 completed on 2026-08-21. This report contains validation results
only. The primary date-grouped test split was not loaded or evaluated.

## Experiment Configuration

| Setting | Value |
| --- | --- |
| Dataset | CNR-EXT |
| Protocol | Primary date-grouped split |
| Train samples | 106,833 |
| Validation samples | 18,938 |
| Input | 224×224 RGB |
| Model | ResNet18, ImageNet-1K V1 weights |
| Parameters | 11,177,538 |
| Optimizer | AdamW |
| Head learning rate | 0.001 |
| Fine-tune backbone learning rate | 0.0001 |
| Fine-tune classifier learning rate | 0.0005 |
| Weight decay | 0.0001 |
| Loss | Cross-entropy, no class weights |
| Batch size | 32 |
| Training schedule | 1 head epoch + 2 full fine-tune epochs |
| Dropout | 0.20 |
| Seed | 42 |
| Device | Apple MPS |
| Checkpoint criterion | Validation occupied-class F1 |

The classifier head was trained first while the backbone parameters and
BatchNorm statistics were frozen. All layers were then unfrozen for two epochs
with separate backbone and classifier learning rates. Training augmentation used
horizontal flip (`p=0.5`), rotation within ±5°, brightness jitter 0.15, and
contrast jitter 0.15. Images were padded to square, resized, converted to RGB,
and normalized using ImageNet mean and standard deviation.

## Training History

| Epoch | Phase | Train loss | Validation loss | Validation accuracy | Validation F1 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Head | 0.059041 | 0.046944 | 0.983208 | 0.983091 |
| 2 | Fine-tune | 0.028825 | 0.024757 | 0.994614 | 0.994551 |
| **3** | **Fine-tune** | **0.014299** | **0.015482** | **0.995248** | **0.995199** |

Epoch 3 was selected by validation occupied-class F1. Total training time was
2,556.16 seconds (about 42.6 minutes).

## Best Validation Result

- Accuracy: `0.995248`
- Precision, occupied class: `0.997434`
- Recall, occupied class: `0.992975`
- F1, occupied class: `0.995199`
- Validation loss: `0.015482`

Confusion matrix, with true labels as rows and predictions as columns
(`empty`, `occupied`):

```text
[[9519, 24],
 [  66, 9329]]
```

These are validation metrics, not final test performance. The untouched test
split remains reserved for a later evaluation.

## Baseline Comparison

Both models used the same 106,833 training and 18,938 validation assignments.

| Model | Validation accuracy | Validation occupied F1 |
| --- | ---: | ---: |
| SimpleCNN | 0.980885 | 0.980611 |
| ResNet18 transfer learning | **0.995248** | **0.995199** |
| Absolute change | +0.014363 | +0.014589 |

ResNet18 improved validation accuracy by 1.436 percentage points and occupied
F1 by 1.459 percentage points. This comparison does not establish performance
on the held-out test split or on a different parking location.

## Artifacts

- Tracked metrics/config: `results/resnet18_transfer.json`
- Local ignored checkpoint: `models/resnet18_transfer.pt`
- Checkpoint size: 44,790,475 bytes
- Checkpoint SHA-256:
  `0b8a9e51bfdef525410781c4d57813ef3a12bee508ab9843e0ba9d393bd78b60`

The checkpoint was loaded into a fresh, non-pretrained ResNet18 instance after
training. Its state dictionary, best epoch, validation sample count, and
`test_split_used: false` flag were verified.

## Reproduction

With `PARKING_DATA_ROOT` configured, set `TORCH_HOME` to a suitable external
cache directory so pretrained weights do not occupy repository storage:

```bash
export TORCH_HOME="$PARKING_DATA_ROOT/torch_cache"
.venv/bin/python -m src.train_resnet18 \
  'data/processed/cnrpark_ext_split_manifest.csv' \
  'models/resnet18_transfer.pt' \
  'results/resnet18_transfer.json' \
  --head-epochs 1 \
  --finetune-epochs 2 \
  --batch-size 32 \
  --input-size 224 \
  --workers 2 \
  --seed 42 \
  --device mps
```

MPS operations are seeded but may not be bit-for-bit deterministic across
hardware or library versions. Exact library versions are stored in the result
JSON.
