# Project Highlights / 專案重點

**Independent AI Research Project · Completed and Frozen**

以資料洩漏防護、跨場域分析與預先鎖定評估規則，完成單一停車格占用分類研究。以下為技術成果摘要，不代表完整停車場偵測或營運部署。

**Cross-Domain Parking Occupancy Detection | PyTorch, Transfer Learning, Domain Adaptation**

- Built an end-to-end CNRPark+EXT → PKLot research pipeline with grouped date/site/source-frame splits, SHA-256 artifact locks, and precommitted evaluation boundaries.
- Compared SimpleCNN, ResNet18, and a controlled EfficientNet-B0 candidate; selected V2-A achieved 0.998894 accuracy, 0.998912 occupied F1, and 1.000000 UFPR04 recall on a one-time 154,669-sample fresh-final set.
- Quantified a zero-shot occupied-F1 domain shift from 0.988692 to 0.788652, then used target adaptation, 8,381-error analysis, and balanced V2 development to raise fresh-final UFPR04 recall from 0.859053 to 1.000000.
- Delivered external-SSD data tooling, portable manifests, automated tests, research visualizations, and a V2-A Streamlit inference demo.

## Technical Contributions

| Area | Skills demonstrated |
|---|---|
| Computer vision | Binary image classification, transfer learning, augmentation, ImageNet normalization, crop-based inference |
| Deep learning | PyTorch, torchvision, SimpleCNN, ResNet18, EfficientNet-B0, AdamW, frozen-head and full fine-tuning |
| Experimental design | Group-based splits, cross-domain evaluation, target adaptation, validation-only selection, precommitted robustness gates |
| Evaluation | Accuracy, precision, recall, occupied F1, macro-site F1, confusion matrices, per-site/per-weather analysis |
| Reproducibility | Deterministic seeds, JSON configs, SHA-256 checkpoint/artifact locks, immutable final evaluation boundaries |
| Data engineering | External-SSD storage, gzip manifests, relative-path resolution, archive/image integrity audits, duplicate/conflict handling |
| Analysis and communication | Error manifests, contact sheets, SVG research visualizations, bilingual technical writing |
| Interactive demonstration | Streamlit, safe in-memory uploads, configurable checkpoint paths, CPU/MPS inference, browser-based UI validation |
| Software quality | Modular Python, unittest coverage, input validation, Git/GitHub repository organization |
