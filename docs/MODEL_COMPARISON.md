# Model Comparison / 模型比較

This report preserves every completed result in its original evaluation context.
Rows from different datasets or split protocols are not treated as a single
leaderboard. 本文件保留各階段原始評估語境；不同資料集或 split 的數值不可直接視為同一排行榜。

## 1. CNR-EXT validation: architecture baseline

Both models used the same 18,938-sample date-grouped validation split.

| Model | Accuracy | Occupied precision | Occupied recall | Occupied F1 |
|---|---:|---:|---:|---:|
| SimpleCNN, epoch 3 | 0.980885 | 0.986954 | 0.974348 | 0.980611 |
| ResNet18 transfer learning, epoch 3 | 0.995248 | 0.997434 | 0.992975 | 0.995199 |

Result: transfer learning improved occupied F1 by +0.014589 on the same
validation assignments.

## 2. Frozen ResNet18: source-to-target domain shift

This is one immutable model evaluated across two domains, not a comparison of
two trained candidates.

| Evaluation domain | Samples | Accuracy | Occupied F1 |
|---|---:|---:|---:|
| CNR-EXT one-time test | 19,155 | 0.988567 | 0.988692 |
| PKLot zero-shot | 695,695 | 0.743097 | 0.788652 |
| Absolute target-domain drop | — | -0.245470 | -0.200040 |

## 3. PKLot target adaptation: same V1 held-out set

Both rows use the same 615,653 held-out samples. These results are complete and
were not reinterpreted during V2 development.

| Model state | Accuracy | Precision | Recall | Occupied F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Frozen source ResNet18 | 0.762163 | 0.685431 | 0.987956 | 0.809347 | 142,636 | 3,789 |
| V1 target-adapted ResNet18 | 0.986387 | 0.996626 | 0.976665 | 0.986545 | 1,040 | 7,341 |

Target adaptation improved occupied F1 by +0.177197, while the remaining errors
became false-negative dominated. UFPR04 occupied recall was 0.854565 in the
separate locked error analysis.

## 4. V2 validation-only candidate selection

Both precommitted candidates used only `v2_train` for training and the same
42,148-sample `v2_validation` set for selection.

| Candidate | Selected epoch | Accuracy | Occupied F1 | Macro-site F1 | Minimum-site recall |
|---|---:|---:|---:|---:|---:|
| V2-A balanced ResNet18 | 2 | 0.999478 | 0.999401 | 0.999489 | 0.999512 |
| V2-B balanced EfficientNet-B0 | 3 | 0.998909 | 0.998748 | 0.999104 | 0.998048 |

The precommitted lexicographic criterion selected V2-A on primary macro-site
occupied F1. EfficientNet-B0 was retained for audit and was not evaluated on
fresh final.

## 5. One-time fresh-final comparison

The immutable V1 and selected V2-A models used the same 154,669 samples, order,
preprocessing, label definitions, evaluation code, and threshold 0.5.

| Model | Accuracy | Precision | Recall | Occupied F1 | Macro-site F1 | UFPR04 recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 target-adapted ResNet18 | 0.991834 | 0.998594 | 0.985315 | 0.991910 | 0.973232 | 0.859053 | 109 | 1,154 |
| V2-A balanced ResNet18 | **0.998894** | **0.998766** | **0.999058** | **0.998912** | **0.998763** | **1.000000** | **97** | **74** |

V2-A passed both precommitted robustness conditions: UFPR04 occupied recall
improved by +0.140947, and overall occupied F1 improved by +0.007002. The result
is terminal and cannot be used to revise either model.

## Final production decision / 最終模型決策

The production demo uses `models/v2a_balanced_resnet18.pt`, SHA-256
`97b039fa7d4125e993903c4d1b485a7bc8e58d47cf7917c5fef8515e6982d5f9`.
The 224×224 edge-pad/ImageNet-normalized preprocessing and threshold 0.5 remain
locked. 最終展示模型為 V2-A ResNet18；未重新訓練、未調整 threshold，也未重開
fresh-final evaluation。

