# PKLot Adaptation Split

Milestone 8 protocol was precommitted on 2026-08-22 before any target-domain
weight update. Dates were not chosen using Milestone 7 site or weather metrics.

## Deterministic Assignment

- Group key: `(site, capture_date)`
- Ordering: SHA-256 of `42:<site>:<date>`, ranked independently within each site
- Adaptation train: approximately 5% of dates per site
- Adaptation validation: approximately 5% of dates per site
- Held-out evaluation: all remaining dates
- Physical image copies: none

The protocol assigns two train dates and two validation dates for each of PUC,
UFPR04, and UFPR05. All other dates remain held out.

| Split | Site-date groups | Samples | Empty | Occupied |
| --- | ---: | ---: | ---: | ---: |
| Adaptation train | 6 | 47,742 | 34,027 | 13,715 |
| Adaptation validation | 6 | 32,300 | 22,899 | 9,401 |
| Held-out evaluation | 89 | 615,653 | 301,067 | 314,586 |

## Adaptation Dates

| Site | Train dates | Validation dates |
| --- | --- | --- |
| PUC | 2012-09-16, 2012-11-10 | 2012-09-11, 2012-10-15 |
| UFPR04 | 2012-12-26, 2013-01-19 | 2012-12-11, 2012-12-21 |
| UFPR05 | 2013-03-18, 2013-03-22 | 2013-03-09, 2013-04-16 |

## Leakage Audit

- Site-date groups in multiple splits: 0
- Source frames in multiple splits: 0
- Duplicate image IDs: 0
- Total manifest rows preserved: 695,695

The generated gzip manifest SHA-256 is:

```text
c0965e66278f78290bd0d830db6dabc560b097cefd82de5d6c2c983bde1ab280
```

## Precommitted Fine-Tuning Configuration

The experiment will initialize from the unchanged Milestone 6 epoch-3 ResNet18
checkpoint. The configuration was fixed before evaluating the new held-out
subset:

| Setting | Value |
| --- | --- |
| Head-only phase | 1 epoch |
| Head learning rate | 0.0001 |
| Full fine-tune phase | 1 epoch |
| Backbone learning rate | 0.00001 |
| Classifier learning rate | 0.00005 |
| Optimizer | AdamW |
| Weight decay | 0.0001 |
| Batch size | 32 |
| Input | 224×224 RGB |
| Augmentation | Same flip/rotation/brightness/contrast policy as Milestone 6 |
| Selection criterion | Adaptation-validation occupied F1 |
| Class weights | None |

The frozen before score and fine-tuned after score must use exactly the same
615,653 held-out samples. CNR-EXT test will not be reopened. The Milestone 7
full-PKLot score is not used as the fine-tuning baseline because it includes the
adaptation dates.

## Reproduction

```bash
.venv/bin/python -m src.prepare_pklot_adaptation_split \
  'data/processed/pklot_manifest.csv.gz' \
  'data/processed/pklot_adaptation_manifest.csv.gz' \
  'data/processed/pklot_adaptation_split_summary.json' \
  --seed 42 \
  --train-fraction 0.05 \
  --validation-fraction 0.05
```
