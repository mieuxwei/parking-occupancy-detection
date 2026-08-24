# Milestone 9 — Error Analysis

## Scope and safeguards

This milestone analyzes the already locked Milestone 8 fine-tuned ResNet18 on
the same 615,653 PKLot `heldout_evaluation` samples. It did not retrain the
model, change weights, tune the 0.5 decision threshold, reopen CNR-EXT test, or
use held-out labels for model selection.

Before inference, `src/analyze_errors.py` verifies both the fine-tuned
checkpoint SHA-256 and adaptation manifest SHA-256 against the completed
Milestone 8 records. The recomputed confusion matrix must exactly match the
saved Milestone 8 matrix or the analysis fails.

## Reproduction

Set `PARKING_DATA_ROOT` to the external dataset directory, without committing
its machine-specific path, then run:

```bash
python -m src.analyze_errors \
  data/processed/pklot_adaptation_manifest.csv.gz \
  models/resnet18_pklot_finetuned.pt \
  results/pklot_finetuning.json \
  data/processed/pklot_error_manifest.csv.gz \
  results/error_analysis.json \
  --image-dir images \
  --batch-size 128 \
  --workers 2 \
  --device auto
```

The compressed local error manifest contains only the 8,381 wrong predictions.
Its fields are image ID, relative image path, true and predicted labels, error
type, confidence, occupied probability, site, physical location, weather,
capture date/time, source frame ID, and slot ID. It contains no absolute SSD
path and no copied dataset image.

## Reproduced held-out result

| Metric | Value |
|---|---:|
| Samples | 615,653 |
| Accuracy | 0.986387 |
| Occupied precision | 0.996626 |
| Occupied recall | 0.976665 |
| Occupied F1 | 0.986545 |
| False positive | 1,040 |
| False negative | 7,341 |
| Total errors | 8,381 |

The confusion matrix is `[[300027, 1040], [7341, 307245]]`, exactly matching
Milestone 8. False negatives account for 87.59% of all remaining errors. This is
the main post-adaptation tradeoff: false positives were strongly reduced, but
missed occupied spaces now dominate the residual error set.

## Site findings

| Site | Samples | FP | FN | Error rate | Occupied recall |
|---|---:|---:|---:|---:|---:|
| PUC | 374,505 | 217 | 826 | 0.2785% | 0.995540 |
| UFPR04 | 92,718 | 8 | 6,117 | 6.6061% | 0.854565 |
| UFPR05 | 148,430 | 815 | 398 | 0.8172% | 0.995442 |

UFPR04 contributes 73.08% of all errors and 83.33% of all false negatives,
despite representing only 15.06% of held-out samples. Its 0.854565 occupied
recall is the clearest remaining domain-specific weakness. In contrast, UFPR05
contributes 78.37% of false positives but maintains high occupied recall.

## Weather findings

| Weather | Samples | FP | FN | Error rate | Occupied recall |
|---|---:|---:|---:|---:|---:|
| Cloudy | 216,956 | 115 | 1,906 | 0.9315% | 0.977757 |
| Rainy | 89,450 | 105 | 507 | 0.6842% | 0.991545 |
| Sunny | 309,247 | 820 | 4,928 | 1.8587% | 0.970828 |

Sunny images contribute 68.58% of all errors. This is an association, not proof
that sunlight caused each error: weather is confounded with site, date, camera,
and scene composition. The available manifest has no reliable shadow,
occlusion, or illumination-intensity annotations, so those conditions are not
reported as quantitative categories.

## Date concentration

The ten dates with the most errors are saved in `results/error_analysis.json`.
The largest count occurs on 2012-12-14: 870 false negatives among 4,526 samples,
an error rate of 19.22%. The top dates are concentrated in the UFPR04 capture
period, consistent with the site-level recall result. Date-level held-out
findings are descriptive and were not used to revise the model.

## Visual inspection

Three small contact sheets were created from the external source images:

- `images/pklot_false_positive_worst_cases.jpg`
- `images/pklot_false_negative_worst_cases.jpg`
- `images/pklot_error_representatives.jpg`

The first two show the 12 most confidently wrong predictions of each type. The
third includes the most confident available error for each observed
error-type/site/weather combination.

Several highest-confidence false positives visibly contain vehicles even though
their manifest/folder label is `EMPTY`; several highest-confidence false
negatives show mostly pavement or parking boundaries despite an `OCCUPIED`
label. These are potential annotation, crop-boundary, or source alignment
ambiguities. Visual inspection alone cannot determine which one is responsible,
so no labels were changed and these samples remain in the error manifest for
future source-XML audit. Other representatives include partial vehicles,
edge-dominated crops, low-detail dark patches, and non-vehicle foreground
objects. These are qualitative examples, not measured failure-cause rates.

## Artifacts and limitations

- Tracked aggregate result: `results/error_analysis.json`
- Local ignored error manifest: `data/processed/pklot_error_manifest.csv.gz`
- Error manifest rows: 8,381
- Error manifest SHA-256:
  `c3ae7d4c7652a0fc4e393ec39431295eb619f6459d4d4f3d6f8631d096d97afe`
- The contact sheets are derivative previews only; no train/validation/test
  image-copy directories were created.
- Site and weather statistics use supplied metadata. Lighting, shadows, and
  occlusion cannot be quantified without a separate audited annotation effort.
- The error set is from held-out dates at PKLot sites seen during adaptation; it
  is not an entirely unseen-site evaluation.

