# Graduate-Application Project Abstract / 研究所備審專題摘要

## 中文版

### 基於遷移學習與洩漏防護協議之跨場域停車格占用辨識

本專題旨在建立一套可重現、可稽核的電腦視覺研究流程，判斷單一停車格裁切影像為空位（`EMPTY`）或占用（`OCCUPIED`），並探討模型從來源停車場移轉至不同目標停車場時的 domain shift、target adaptation 與 robustness。相較於只報告單一 accuracy，本研究同時追蹤 occupied precision、recall、F1、confusion matrix、per-site metrics 與 macro-site F1，並將資料洩漏防護視為實驗設計的核心。

資料採用 CNRPark+EXT 與 PKLot。所有大型影像與 archive 存放於外接 SSD，GitHub repository 僅保留相對路徑 manifest、metadata、split assignment、設定檔、hash 與結果。切分以 date、site、source frame 等群組為單位，避免同一時序或原始影像衍生的高度相似停車格跨越 train、validation 與 evaluation。資料準備亦包含影像解碼檢查、39 個均勻黑色 placeholder 排除，以及 78 組 PKLot conflicting-label duplicate IDs 的雙邊排除。

第一階段在相同的 CNR-EXT date-grouped validation split 上比較從零訓練的 SimpleCNN 與 ImageNet-pretrained ResNet18。SimpleCNN occupied F1 為 0.980611；ResNet18 提升至 0.995199。模型選定後才開啟一次性 CNR-EXT test，ResNet18 達到 Accuracy 0.988567 與 occupied F1 0.988692。相同 frozen model 在 695,695 筆 PKLot 進行 zero-shot evaluation 時，Accuracy 降至 0.743097、occupied F1 降至 0.788652，清楚量化跨停車場 domain shift。

第二階段事先承諾 PKLot site/date adaptation protocol。在相同 615,653 筆 held-out samples 上，target adaptation 使 occupied F1 由 0.809347 提升至 0.986545；然而 false negatives 增至主要殘餘錯誤。後續 error analysis 在不改權重與 threshold 的前提下重現相同 confusion matrix，發現 8,381 個錯誤中有 7,341 個 FN，UFPR04 占 6,117 個，occupied recall 僅 0.854565。由於 weather 與 site/date 高度混雜，本研究只將其視為關聯，不宣稱光線或天氣的因果效果。

第三階段建立 V2 robustness protocol。為避免針對已分析錯誤過度調整，V1 held-out error manifest、aggregate error analysis 與 contact sheets 均被禁止作為訓練或選模輸入。V2 事先固定 split、seed 105、site/occupancy-balanced sampling、溫和 augmentation、候選模型與 validation-only lexicographic selection criterion。V2-A balanced ResNet18 與 V2-B EfficientNet-B0 僅使用 `v2_train` 訓練、僅使用 `v2_validation` 選模；V2-A 以 macro-site F1 0.999489 勝出並在 fresh-final 開啟前鎖定 checkpoint SHA-256。

最終一次性評估以同一 154,669 筆 fresh-final samples、相同順序、前處理、label definition、evaluation code 與 threshold 0.5，比較 immutable V1 與 selected V2-A。V2-A 達到 Accuracy 0.998894、occupied F1 0.998912、macro-site F1 0.998763 與 UFPR04 occupied recall 1.000000，false negatives 由 V1 的 1,154 降至 74。UFPR04 recall 絕對提升 +0.140947，overall F1 亦提升 +0.007002，因此通過兩項 precommitted robustness gate。

本專題最終產出包含完整 experiment history、portable manifests、artifact hashes、automated tests、error contact sheets、研究視覺化與使用 V2-A 的 Streamlit 單張裁切影像 demo。研究限制包括：系統不執行完整停車場的車位定位、softmax confidence 未校準、場域與天氣 metadata 無法支持因果推論，且 fresh-final 雖與 V1 adaptation 及 V2 development disjoint，PKLot 圖片在較早研究階段曾接受 V1 inference。這些限制與所有早期較低分數均被保留，以呈現完整且可信的研究演進。

## English Version

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

