# Project Abstract / 專案摘要

**Independent AI Research Project · Completed and Frozen**

單一停車格裁切影像的跨場域占用分類研究：由 domain shift、錯誤分析與場域適應，走向預先鎖定規則的一次性最終評估。研究與模型已凍結；不包含車位定位或即時監控。

### Leakage-Aware Cross-Domain Parking-Space Occupancy Classification with Transfer Learning

This project develops a reproducible and auditable computer-vision workflow for
classifying a single cropped parking space as `EMPTY` or `OCCUPIED`. The study
focuses on source-to-target domain shift, target adaptation, and site-level
robustness. Rather than reporting accuracy alone, it tracks occupied precision,
recall, F1, confusion matrices, per-site metrics, and macro-site F1, while
treating leakage prevention as a central experimental requirement.

CNRPark+EXT and PKLot serve as the source and target datasets. Large archives
and extracted images remain on an external SSD; the GitHub repository stores
only portable relative-path manifests, metadata, split assignments,
configurations, hashes, code, and results. Splits are grouped by date, site, and
source frame to keep temporally or spatially related crops from crossing
development and evaluation boundaries. Preparation also excluded 39 uniform
black placeholders and both copies of 78 conflicting-label duplicate PKLot IDs.

On the same date-grouped CNR-EXT validation set, a 98,178-parameter SimpleCNN
achieved 0.980611 occupied F1, while an ImageNet-pretrained ResNet18 achieved
0.995199. Only after selection was locked did the ResNet18 enter the one-time
CNR-EXT test, where it achieved 0.988567 accuracy and 0.988692 occupied F1. The
same frozen model then underwent zero-shot evaluation on 695,695 PKLot patches;
accuracy fell to 0.743097 and occupied F1 to 0.788652, exposing a 0.200040
absolute F1 domain-shift penalty.

A precommitted PKLot site/date adaptation protocol next compared the frozen and
fine-tuned model on the same 615,653 held-out samples. Target adaptation raised
occupied F1 from 0.809347 to 0.986545, but false negatives became the dominant
residual error. A locked error-analysis replay reproduced the exact confusion
matrix and found 7,341 false negatives among 8,381 errors. UFPR04 contributed
6,117 false negatives and reached only 0.854565 occupied recall. Weather was
reported as an association—not a causal factor—because it is confounded with
site, date, camera, and scene composition.

The final stage introduced a separately precommitted V2 robustness protocol.
To avoid overfitting to analyzed mistakes, the V1 held-out error manifest,
aggregate error analysis, and contact sheets were prohibited as training or
selection inputs. V2 fixed seed 105, group splits, site/occupancy-balanced
sampling, mild augmentation, two candidates, and a validation-only
lexicographic criterion before training. A balanced ResNet18 and a controlled
EfficientNet-B0 trained only on `v2_train` and were selected only on
`v2_validation`. ResNet18 won with 0.999489 macro-site F1 and its checkpoint
hash was locked before fresh-final inference.

The immutable V1 and selected V2-A were finally compared once on the same
154,669 fresh-final samples using identical order, preprocessing, labels,
evaluation code, and threshold 0.5. V2-A achieved 0.998894 accuracy, 0.998912
occupied F1, 0.998763 macro-site F1, and 1.000000 UFPR04 occupied recall. False
negatives fell from 1,154 to 74. Its +0.140947 UFPR04 recall gain and +0.007002
overall F1 change passed both precommitted robustness gates.

Deliverables include complete experiment histories, portable manifests,
artifact hashes, automated tests, error contact sheets, research
visualizations, and a Streamlit single-crop demo using V2-A. Limitations remain
explicit: the system does not localize parking spaces in full-lot images,
softmax confidence is uncalibrated, site/weather metadata cannot support causal
claims, and “fresh final” is development-disjoint rather than never historically
inferred by V1. Earlier metrics and failures remain visible because preserving
the research progression is part of the contribution.
