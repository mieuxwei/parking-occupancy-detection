# PKLot Fine-Tuning

Milestone 8 completed on 2026-08-22. A small, date-grouped PKLot adaptation pool
was used to fine-tune the already selected ResNet18. Frozen before and fine-tuned
after metrics use the same disjoint 615,653-sample held-out set.

## Data Boundary

| Split | Site-date groups | Samples |
| --- | ---: | ---: |
| Adaptation train | 6 | 47,742 |
| Adaptation validation | 6 | 32,300 |
| Held-out evaluation | 89 | 615,653 |

The split was selected by deterministic SHA-256 ranking within each site using
seed 42. Date groups, source frames, and image IDs do not cross splits. Exact
dates and audit results are documented in `docs/PKLOT_ADAPTATION_SPLIT.md`.

The CNR-EXT test split was not reopened. The full-PKLot Milestone 7 metric was
not used as the before score because it contains the adaptation dates.

## Precommitted Configuration

| Setting | Value |
| --- | --- |
| Initialization | Milestone 6 epoch-3 ResNet18 |
| Head phase | 1 epoch, learning rate 0.0001 |
| Full fine-tune phase | 1 epoch |
| Backbone learning rate | 0.00001 |
| Classifier learning rate | 0.00005 |
| Optimizer | AdamW |
| Weight decay | 0.0001 |
| Batch size | 32 |
| Input | 224×224 RGB |
| Class weights | None |
| Checkpoint criterion | Adaptation-validation occupied F1 |

No epochs, thresholds, class weights, or learning rates were changed after the
held-out before result was observed.

## Training History

| Epoch | Phase | Train loss | Validation loss | Validation accuracy | Validation F1 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Head | 0.410037 | 0.187031 | 0.927678 | 0.861497 |
| **2** | **Full fine-tune** | **0.033367** | **0.052831** | **0.983591** | **0.971022** |

Epoch 2 was selected by the precommitted validation F1 criterion.

## Held-Out Before and After

| Metric | Frozen before | Fine-tuned after | Absolute change |
| --- | ---: | ---: | ---: |
| Accuracy | 0.762163 | 0.986387 | +0.224224 |
| Occupied precision | 0.685431 | 0.996626 | +0.311196 |
| Occupied recall | 0.987956 | 0.976665 | -0.011291 |
| Occupied F1 | 0.809347 | 0.986545 | +0.177197 |
| Loss | 0.555694 | 0.047830 | -0.507864 |

Frozen confusion matrix:

```text
[[158431, 142636],
 [  3789, 310797]]
```

Fine-tuned confusion matrix:

```text
[[300027,   1040],
 [  7341, 307245]]
```

Fine-tuning reduced false positives from 142,636 to 1,040. The tradeoff was a
small occupied-recall decline, with false negatives increasing from 3,789 to
7,341. Overall occupied F1 still improved by 17.720 percentage points.

## Results by Site

| Site | Accuracy before | Accuracy after | F1 before | F1 after |
| --- | ---: | ---: | ---: | ---: |
| PUC | 0.767891 | 0.997215 | 0.809697 | 0.997179 |
| UFPR04 | 0.526791 | 0.933939 | 0.657149 | 0.921485 |
| UFPR05 | 0.894738 | 0.991828 | 0.914749 | 0.993071 |

UFPR04 improved most, but its occupied recall decreased from 0.999715 to
0.854565. This residual site-specific tradeoff should be examined in Milestone 9
rather than addressed by post-hoc tuning in this experiment.

## Results by Weather

| Weather | Accuracy before | Accuracy after | F1 before | F1 after |
| --- | ---: | ---: | ---: | ---: |
| Cloudy | 0.706848 | 0.990685 | 0.726347 | 0.988083 |
| Rainy | 0.836847 | 0.993158 | 0.891219 | 0.994880 |
| Sunny | 0.779367 | 0.981413 | 0.830039 | 0.982778 |

Weather remains correlated with site, date, illumination, and viewpoint, so
these values are descriptive rather than causal weather effects.

## Interpretation and Limits

The experiment demonstrates that labeled target-domain examples from six dates
can strongly improve performance on different dates from the same PKLot sites
and camera views. It does not demonstrate equal improvement on a new parking
location or camera unseen by both source training and target adaptation.

The adaptation train set contains 47,742 patches, or about 6.9% of eligible
PKLot patches. Although it uses only six dates, temporal correlation produces
many individual parking-space samples. Claims should therefore describe this as
small date-group adaptation rather than few-shot image learning.

## Artifacts

- Result: `results/pklot_finetuning.json`
- Local ignored checkpoint: `models/resnet18_pklot_finetuned.pt`
- Checkpoint size: 44,791,307 bytes
- Checkpoint SHA-256:
  `e61137bf2fbc259a2b7bd22ecc9840dda8c2668d9f73b7352bbaa3d47809e7ce`
- Adaptation manifest SHA-256:
  `c0965e66278f78290bd0d830db6dabc560b097cefd82de5d6c2c983bde1ab280`

The selected checkpoint was loaded into a fresh model and its best epoch and
validation metrics were verified. Before and after group counts and confusion
matrices were checked against all 615,653 held-out samples.

## Reproduction

```bash
.venv/bin/python -m src.finetune_pklot \
  'data/processed/pklot_adaptation_manifest.csv.gz' \
  'models/resnet18_transfer.pt' \
  'models/resnet18_pklot_finetuned.pt' \
  'results/pklot_finetuning.json' \
  --head-epochs 1 \
  --finetune-epochs 1 \
  --train-batch-size 32 \
  --eval-batch-size 128 \
  --head-learning-rate 0.0001 \
  --backbone-learning-rate 0.00001 \
  --classifier-learning-rate 0.00005 \
  --workers 2 \
  --seed 42 \
  --device mps
```

The complete before/train/after run took 4,401.66 seconds, approximately 73.4
minutes.
