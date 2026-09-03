# Final Research Summary / 最終研究摘要

## 中文摘要

本專題研究單一停車格裁切影像的 `EMPTY`／`OCCUPIED` 二元分類，核心問題不是只追求高準確率，而是建立一條可稽核的跨場域研究流程：資料依日期、來源 frame 與場域分組，避免高度相似影像洩漏；大型影像只存放於外接 SSD；repository 僅保存可攜式 manifest、設定、程式、hash 與結果。

研究從 CNRPark+EXT 的 98,178-parameter SimpleCNN baseline 開始。在相同 18,938 筆 validation split 上，SimpleCNN occupied F1 為 0.980611，ImageNet transfer-learning ResNet18 提升至 0.995199。鎖定 ResNet18 後，一次性 CNR-EXT test occupied F1 為 0.988692；直接移至 PKLot 的 zero-shot occupied F1 降至 0.788652，量化出 0.200040 的跨場域落差。

接著以事先承諾的 PKLot site/date group split 進行 target adaptation。V1 在同一 615,653 筆 held-out set 上，occupied F1 從 0.809347 提升至 0.986545；但 error analysis 顯示 8,381 個錯誤中有 7,341 個 false negatives，UFPR04 貢獻 6,117 個 FN，occupied recall 僅 0.854565。這項弱點成為 V2 robustness protocol 的研究目標，但既有 error manifest、contact sheets 與 held-out aggregate analysis 均被禁止作為 V2 訓練或選模輸入。

V2 事先鎖定 site/label-balanced sampling、溫和 augmentation、seed、split、選模規則與 fresh-final gate。V2-A ResNet18 與 V2-B EfficientNet-B0 僅使用 `v2_train` 訓練，並僅使用 `v2_validation` 選模；V2-A 以較高 macro-site F1 0.999489 勝出。最後，V1 與 V2-A 在同一 154,669 筆 fresh-final samples 上進行唯一一次比較。V2-A 達到 Accuracy 0.998894、occupied F1 0.998912、UFPR04 occupied recall 1.000000，並同時通過「UFPR04 recall 至少提升 +0.02」與「overall F1 不得下降超過 0.005」兩項 precommitted gate。

本專題的主要成果不只是最終分數，而是完整保留 baseline、domain shift、adaptation、error analysis、controlled candidate selection 與 terminal evaluation 的研究脈絡。最終凍結展示 使用鎖定的 V2-A checkpoint，維持 224×224 edge padding、bilinear resize、ImageNet normalization 與 threshold 0.5。

## English Summary

This project studies binary `EMPTY`/`OCCUPIED` classification for a single
cropped parking-space image. Its central contribution is an auditable
cross-domain workflow rather than an isolated high score: images are grouped by
date, source frame, and site to reduce near-duplicate leakage; large archives
remain on external storage; and the repository retains portable manifests,
precommitted configurations, hashes, code, and exact results.

The study began with a 98,178-parameter SimpleCNN on CNRPark+EXT. On the same
18,938-sample validation split, occupied F1 increased from 0.980611 for
SimpleCNN to 0.995199 for an ImageNet-pretrained ResNet18. After model selection
was locked, the one-time CNR-EXT test occupied F1 was 0.988692. Zero-shot
transfer of that immutable model to PKLot reduced occupied F1 to 0.788652, an
absolute cross-domain drop of 0.200040.

A precommitted PKLot site/date protocol then evaluated target adaptation. On the
same 615,653 held-out samples, V1 occupied F1 improved from 0.809347 to
0.986545. However, 7,341 of 8,381 residual errors were false negatives, with
UFPR04 contributing 6,117 and reaching only 0.854565 occupied recall. This
documented weakness motivated a new V2 robustness protocol, while the analyzed
held-out error manifest, contact sheets, and aggregate error analysis were
explicitly excluded from V2 training and selection.

V2 precommitted balanced site/label sampling, mild augmentation, seeds, splits,
selection criteria, and a final robustness gate. A balanced ResNet18 and a
controlled EfficientNet-B0 candidate trained only on `v2_train` and were ranked
only on `v2_validation`; ResNet18 won on the primary macro-site F1 criterion.
The immutable V1 and selected V2-A were then compared once on the same 154,669
fresh-final samples. V2-A achieved 0.998894 accuracy, 0.998912 occupied F1, and
1.000000 UFPR04 occupied recall, passing both precommitted robustness gates.

The final frozen demo uses the locked V2-A checkpoint with unchanged 224px
edge padding, bilinear resizing, ImageNet normalization, and threshold 0.5.
Earlier results remain visible because the research progression—and the
boundaries between development, selection, and final evaluation—is part of the
deliverable.

## Contributions / 專題貢獻

- Leakage-aware group splitting and portable external-storage manifests.
- A complete source-domain → zero-shot target → adaptation → robustness study.
- Validation-only model selection with checkpoint/config SHA-256 locks.
- Explicit error-analysis safeguards and a one-time terminal comparison.
- A minimal, reproducible single-crop Streamlit demo using the selected V2-A.

## Limitations / 研究限制

- The demo classifies a pre-cropped space; it does not localize spaces in a full lot.
- CNRPark+EXT represents one physical parking area; camera holdout is not true cross-location generalization.
- PKLot site/date findings do not establish causal effects of weather, shadow, or lighting.
- Softmax confidence is not calibrated.
- “Fresh final” is disjoint from V1 adaptation and V2 development, but PKLot images had historically received V1 inference before this protocol; the result is not claimed as never previously inferred by V1.
- The final result is closed and may not be used for another model update.


## Full Research Progression

| Stage | Dataset / boundary | Main result | Research decision |
|---|---|---|---|
| 1. SimpleCNN baseline | CNR-EXT validation, 18,938 | Accuracy 0.980885; F1 0.980611 | Established a small from-scratch baseline |
| 2. ResNet18 transfer learning | Same CNR-EXT validation | Accuracy 0.995248; F1 0.995199 | Transfer learning won on the fair split |
| 3. In-domain evaluation | CNR-EXT one-time test, 19,155 | Accuracy 0.988567; F1 0.988692 | Locked source-domain performance |
| 4. Zero-shot domain shift | PKLot, 695,695 | Accuracy 0.743097; F1 0.788652 | Revealed a 0.200040 F1 drop |
| 5. V1 target adaptation | PKLot held-out, 615,653 | Accuracy 0.986387; F1 0.986545 | Recovered most target-domain performance |
| 6. Error analysis | Same completed V1 held-out result | 8,381 errors; 7,341 FN; UFPR04 recall 0.854565 | Identified a site-specific FN weakness |
| 7. V2 robustness protocol | New disjoint site/date groups | Balanced site/label sampling, mild augmentation, locked gate | Excluded analyzed V1 errors from development |
| 8. V2 candidate selection | `v2_validation`, 42,148 | ResNet18 macro-site F1 0.999489 vs EfficientNet 0.999104 | Selected V2-A using validation only |
| 9. One-time V1 vs V2 | Fresh final, 154,669 | V2-A Accuracy 0.998894; F1 0.998912; UFPR04 recall 1.000000 | Both precommitted robustness gates passed |

Metrics from different rows belong to different protocols and are not a single
leaderboard. 不同階段的數值保留其原始 dataset/split 語境，只有標明相同 split 的
比較才是直接公平比較。
