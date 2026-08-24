# Milestone 10.5 — Model Robustness Improvement Protocol

## Status

Protocol and experiment configuration precommitted on 2026-08-24. No V2 model
has been trained, selected, or evaluated. Milestone 11 has not started.

## V1 preservation boundary

All completed V1 checkpoints, result JSON files, local adaptation/error
manifests, and milestone reports are registered by path, byte size, and SHA-256
in `data/V1_ARTIFACT_LOCK.json`. `src/freeze_v1_artifacts.py` creates the
registry once; later runs verify it and fail if any registered artifact changed.

V1 metrics remain the completed experiment results. Milestone 10.5 will not
overwrite them, reinterpret the previously analyzed 615,653-sample held-out
result, reopen CNR-EXT test, or update the Streamlit application.

The existing PKLot error manifest, aggregate error result, and error contact
sheets are explicit forbidden inputs for V2 training, validation, sampling, and
candidate selection. They motivated the high-level research question—UFPR04
false negatives—but do not supply samples or weights.

## Fresh-final limitation

Every eligible PKLot image already received a V1 prediction in Milestone 7, and
the former 615,653-sample held-out set was subsequently analyzed. Therefore no
PKLot subset can honestly be described as never previously inferred by V1.

Here, **fresh final** means a newly precommitted set that is isolated from all V2
development and contains only dates that were also excluded from V1 adaptation
train/validation. Dates were selected by a new SHA-256 protocol identifier, not
by V1/V2 predictions, errors, site metrics, or date metrics. Existing
error-manifest rows are not read or reused to form the split.

## Date-group assignment

- Source: the clean V1 adaptation manifest, SHA-256
  `c0965e66278f78290bd0d830db6dabc560b097cefd82de5d6c2c983bde1ab280`
- Protocol ID: `pklot-v2-robustness-v1`
- Group key: `(site, capture_date)`
- Final reservation: approximately 20% of all dates per site, selected only from
  prior V1 `heldout_evaluation` dates
- Development: four train dates and two validation dates per site, selected
  after final dates are removed
- Remaining groups: `unused_reserve`
- Physical image copies: none

| Split | Site-date groups | Samples | Empty | Occupied |
|---|---:|---:|---:|---:|---:|
| V2 train | 12 | 78,376 | 41,738 | 36,638 |
| V2 validation | 6 | 42,148 | 23,791 | 18,357 |
| V2 fresh final | 21 | 154,669 | 76,086 | 78,583 |
| Unused reserve | 62 | 420,502 | 216,378 | 204,124 |

The tracked 101-row `data/PKLOT_V2_DATE_SPLIT.csv` is the authoritative date
assignment. The full ignored local manifest is
`data/processed/pklot_v2_protocol_manifest.csv.gz`, SHA-256
`4aa2e976d67983b44708d304343ed6c72d7f054f294106ebf3b14e93dd00c5d7`.

## Leakage audit

- Site-date groups spanning V2 splits: 0
- Source frames spanning V2 splits: 0
- Duplicate image IDs: 0
- Fresh-final dates previously used by V1 adaptation: 0
- Fresh-final dates used by V2 development: 0

The final split contains 90,781 PUC, 23,135 UFPR04, and 40,753 UFPR05 samples.
Natural class/site/weather distributions are preserved for validation and final
evaluation; only the training sampler is balanced.

## Precommitted V2 development

The exact machine-readable configuration is
`data/PKLOT_V2_EXPERIMENT_CONFIG.json`.

### Site-and-class-balanced sampling

Training uses replacement sampling with inverse frequency for each
`(site, occupancy)` cell. One epoch still contains 78,376 draws. This prevents
PUC's larger patch count from dominating batches and gives UFPR04 occupied
examples a controlled, metadata-defined increase in exposure. Validation and
final evaluation retain their natural distributions. Loss class weights are not
combined with the sampler.

### Augmentation

Both candidates use edge padding, 224×224 resize, ImageNet normalization,
horizontal flip probability 0.5, rotation up to 7 degrees, and mild brightness,
contrast, saturation, and hue jitter. Random crops and vertical flips are
forbidden because they can remove occupancy evidence or create implausible
parking geometry.

### Candidates

1. `v2a_balanced_resnet18`: initialize from the immutable V1 PKLot checkpoint
   and run two full fine-tuning epochs with backbone learning rate `5e-6` and
   classifier learning rate `2e-5`.
2. `v2b_balanced_efficientnet_b0`: the one allowed alternative backbone,
   initialized with torchvision ImageNet-1K V1 weights. Run one head epoch and
   two full epochs. Split, sampling, augmentation, loss, batch size, seed, and
   selection rules remain identical to V2-A.

Both candidates use AdamW, weight decay `1e-4`, train/eval batch sizes 32/128,
two workers, seed 105, and unweighted cross-entropy.

## Validation-only selection

Candidates and epochs are ranked without final data:

1. macro-average occupied F1 across PUC, UFPR04, and UFPR05;
2. minimum per-site occupied recall;
3. overall occupied F1.

Exactly one candidate must be selected after both planned candidates finish.
Its checkpoint/config hashes must be recorded before opening fresh final.

## One-time fair comparison

After the V2 candidate is locked, the immutable V1 checkpoint and selected V2
checkpoint will be evaluated in the same order on the same 154,669 fresh-final
samples at threshold 0.5. Report overall, site, and weather metrics plus
confusion matrices. Neither model may be selected, recalibrated, or retrained
after the final set is opened.

The precommitted robustness gate is:

- UFPR04 occupied recall improves by at least 0.02 absolute; and
- overall occupied F1 declines by no more than 0.005 absolute.

The gate is reported as pass/fail only; it cannot be used to choose or revise a
model after final evaluation.

## Reproduction of the protocol only

```bash
.venv/bin/python -m src.freeze_v1_artifacts data/V1_ARTIFACT_LOCK.json

.venv/bin/python -m src.prepare_v2_protocol \
  data/processed/pklot_adaptation_manifest.csv.gz \
  data/processed/pklot_v2_protocol_manifest.csv.gz \
  data/PKLOT_V2_DATE_SPLIT.csv \
  results/v2_protocol.json \
  --protocol-id pklot-v2-robustness-v1 \
  --train-dates-per-site 4 \
  --validation-dates-per-site 2 \
  --final-fraction 0.20
```

These commands read metadata only. They do not load images, train a model, open
fresh final for inference, or modify the Demo.
