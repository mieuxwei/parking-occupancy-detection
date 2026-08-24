# CV / Resume Project Description

## 中文精簡版

**跨場域停車格占用辨識｜PyTorch、Transfer Learning、Domain Adaptation**

- 建立 CNRPark+EXT → PKLot 的完整跨場域研究流程，使用 date/site/source-frame group split、SHA-256 artifact locks 與 precommitted evaluation protocol 控制資料洩漏與選模偏誤。
- 比較 SimpleCNN、ResNet18 與 controlled EfficientNet-B0；selected V2-A ResNet18 在 154,669 筆一次性 fresh-final set 達到 Accuracy 0.998894、occupied F1 0.998912、UFPR04 recall 1.000000。
- 量化 zero-shot occupied-F1 domain shift 0.988692 → 0.788652，並透過 target adaptation、8,381-error analysis 與 balanced V2 protocol 將 UFPR04 fresh-final recall 從 0.859053 提升至 1.000000。
- 完成 external-SSD dataset pipeline、portable manifests、automated tests、研究視覺化與 V2-A Streamlit inference demo。

## English concise version

**Cross-Domain Parking Occupancy Detection | PyTorch, Transfer Learning, Domain Adaptation**

- Built an end-to-end CNRPark+EXT → PKLot research pipeline with grouped date/site/source-frame splits, SHA-256 artifact locks, and precommitted evaluation boundaries.
- Compared SimpleCNN, ResNet18, and a controlled EfficientNet-B0 candidate; selected V2-A achieved 0.998894 accuracy, 0.998912 occupied F1, and 1.000000 UFPR04 recall on a one-time 154,669-sample fresh-final set.
- Quantified a zero-shot occupied-F1 domain shift from 0.988692 to 0.788652, then used target adaptation, 8,381-error analysis, and balanced V2 development to raise fresh-final UFPR04 recall from 0.859053 to 1.000000.
- Delivered external-SSD data tooling, portable manifests, automated tests, research visualizations, and a V2-A Streamlit inference demo.

## Technical Skills / 技術能力

| Area | Skills demonstrated |
|---|---|
| Computer vision | Binary image classification, transfer learning, augmentation, ImageNet normalization, crop-based inference |
| Deep learning | PyTorch, torchvision, SimpleCNN, ResNet18, EfficientNet-B0, AdamW, frozen-head and full fine-tuning |
| Experimental design | Group-based splits, cross-domain evaluation, target adaptation, validation-only selection, precommitted robustness gates |
| Evaluation | Accuracy, precision, recall, occupied F1, macro-site F1, confusion matrices, per-site/per-weather analysis |
| Reproducibility | Deterministic seeds, JSON configs, SHA-256 checkpoint/artifact locks, immutable final evaluation boundaries |
| Data engineering | External-SSD storage, gzip manifests, relative-path resolution, archive/image integrity audits, duplicate/conflict handling |
| Analysis and communication | Error manifests, contact sheets, SVG research visualizations, bilingual technical writing |
| Product demonstration | Streamlit, safe in-memory uploads, configurable checkpoint paths, CPU/MPS inference, browser-based UI validation |
| Software quality | Modular Python, unittest coverage, input validation, Git/GitHub repository organization |

