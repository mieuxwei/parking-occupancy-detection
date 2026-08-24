# PROJECT_PLAN.md

# Cross-Domain Parking Space Occupancy Detection Using Transfer Learning
## 基於遷移學習之跨場域智慧停車位占用辨識系統

## 1. Project Overview

本專題目標為建立一套 Computer Vision 模型，自動判斷單一停車格影像為 EMPTY 或 OCCUPIED。

研究重點不只在分類準確率，也包含：
- 不同停車場之間的泛化能力
- 不同攝影機角度、光線與天氣條件的影響
- Transfer Learning 是否優於從零訓練 CNN
- Cross-domain performance drop
- Fine-tuning 是否能改善跨場域辨識
- Error Analysis

本專題預計作為 AI / Computer Vision 研究所備審與實習作品集之一，因此必須保留完整研究流程、可重現實驗、模型比較與展示成果。

## 2. Main Research Questions

1. Simple CNN 能否作為有效 baseline？
2. ResNet18 Transfer Learning 是否優於從零訓練 CNN？
3. 模型在不同停車場之間是否會出現明顯 performance drop？
4. 天氣、光線、陰影、遮擋是否影響辨識？
5. 少量 target-domain 資料 Fine-tuning 能否改善 cross-domain performance？
6. 模型最容易在哪些影像情境誤判？

## 3. Planned Datasets

優先評估：
- PKLot
- CNRPark / CNRPark-EXT

第一階段先確認：
- Dataset structure
- Label 定義
- Parking lot / camera / weather metadata
- Dataset size
- License / usage conditions
- 是否已有 cropped parking-space images
- 是否可依來源場域進行 train / validation / test split

大型 Dataset 不直接 commit 到 GitHub。

## 4. Research Tracks

### Track A — In-Domain Occupancy Classification

Dataset
→ Data Cleaning
→ EDA
→ Simple CNN
→ ResNet18 Transfer Learning
→ Model Evaluation
→ Error Analysis

### Track B — Cross-Domain Generalization

Source Domain Training
→ Target Domain Testing
→ Performance Drop Analysis
→ Fine-tuning
→ Re-evaluation

## 5. Data Leakage Rules

必須避免高度相似影像同時出現在 train / validation / test。

特別注意：
- 同一張原始停車場影像裁切出的多個停車格，不應分散到不同 split。
- 同一連續時間序列中的極相似畫面，不應同時出現在 train 與 test。
- 若 dataset 提供 parking lot / camera / date / weather metadata，優先依來源群組切分。
- Cross-domain 實驗應保留未見過的場域作為 test domain。

主要 split 應採 group-based 或 domain-based split，而不是單純 random image split。

## 6. Planned Models

### Baseline
- Simple CNN

### Transfer Learning
- ResNet18

### Optional Later Model
- EfficientNet

目前不要一次加入過多模型。

## 7. Evaluation Metrics

主要指標：
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Cross-domain 額外比較：
- In-domain F1
- Cross-domain F1
- Performance drop
- Fine-tuning before / after improvement

禁止只報告 Accuracy。

## 8. Error Analysis

後期分析：
- False Positive
- False Negative
- 陰影
- 強光
- 雨天
- 低光
- 車輛部分遮擋
- 停車格邊界不清
- 不同攝影機角度
- 不同停車場

若資料沒有可靠 metadata，不可自行猜測失敗原因。

## 9. Transfer Learning / Fine-tuning

1. 使用 source domain 訓練模型
2. 在 target domain 直接測試
3. 記錄 cross-domain performance drop
4. 使用少量 target-domain 標註資料 Fine-tuning
5. 再次評估
6. 比較 Fine-tuning 前後差異

## 10. Planned Tech Stack

- Python
- PyTorch
- torchvision
- OpenCV
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Jupyter Notebook
- Git
- GitHub
- Codex

目前不使用 TensorFlow。

## 11. Repository Structure

```text
parking-occupancy-detection/
├── README.md
├── PROJECT_PLAN.md
├── HANDOFF.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
├── models/
├── results/
├── images/
├── tests/
├── docs/
└── app/
```

實際結構可依開發狀況調整，不需為符合文件範例而破壞良好架構。

## 12. GitHub Rules

不要 commit：
- 大型 Dataset
- 大型 model weights
- API keys
- secrets
- .env
- .venv
- cache files

`.gitignore` 至少包含：

```text
.venv/
__pycache__/
.DS_Store
.ipynb_checkpoints/
.env
data/raw/
data/processed/
models/*.pt
models/*.pth
models/*.ckpt
```

需要保留資料夾時可使用 `.gitkeep`。

## 13. Milestones

### Milestone 1 — Project Setup
- Repository structure
- README
- PROJECT_PLAN
- HANDOFF
- requirements.txt
- .gitignore

### Milestone 2 — Dataset Research
- 評估 PKLot
- 評估 CNRPark / CNRPark-EXT
- 確認 labels / metadata / size / license
- 選定主要 dataset
- 設計 acquisition strategy

### Milestone 3 — Data Split & Leakage Audit
- 設計 group/domain split
- 確認 train / validation / test
- 建立 leakage checks
- 建立 dataset summary

### Milestone 4 — EDA & Data Preparation
- Class balance
- Image size
- Weather / camera / parking lot distribution
- Preprocessing
- Data augmentation strategy

### Milestone 5 — Simple CNN Baseline
- 建立 baseline
- 保存 training config
- Accuracy / Precision / Recall / F1
- Confusion Matrix

### Milestone 6 — ResNet18 Transfer Learning
- Transfer learning
- 與 CNN baseline 比較
- 保持相同 split

### Milestone 7 — Cross-Domain Evaluation
- 未見場域測試
- Performance drop 分析
- In-domain vs cross-domain 比較

### Milestone 8 — Fine-tuning
- 少量 target-domain 資料
- Fine-tuning
- Before / after comparison

### Milestone 9 — Error Analysis
- Worst cases
- False Positive / False Negative
- Weather / lighting / occlusion
- Domain-specific failure cases

### Milestone 10 — Demo
- 建立簡單 inference interface
- 上傳停車格圖片
- 輸出 EMPTY / OCCUPIED
- 顯示 confidence
- Demo video / GIF

### Milestone 10.5 — Model Robustness Improvement
- Freeze and preserve all completed V1 artifacts and results
- Precommit a new leakage-safe V2 development and fresh-final protocol
- Focus model development on the documented UFPR04 false-negative weakness
- Evaluate balanced site/label sampling and reasonable augmentation
- Allow at most one controlled alternative backbone such as EfficientNet-B0
- Keep the analyzed V1 held-out result immutable and exclude its error manifest
  from all V2 training and model selection
- Compare locked V1 and selected V2 once on the same fresh final date groups
- Keep the existing Streamlit demo unchanged until a V2 model is selected

### Milestone 11 — Portfolio Finalization
- Bilingual polished README and research progression table
- Architecture/workflow, domain-shift, and V1-versus-V2 visualizations
- Boundary-aware model comparison table and error-analysis summary
- V2-A production demo, screenshots, and GIF
- Bilingual final research summary and graduate-application abstract
- Short CV/resume description and technical-skills summary
- Repository QR code
- Final HANDOFF and integrity audit

## 14. Current Priority

目前專題已完成 Milestone 11 Portfolio Finalization。最終 production model
為 selected V2-A balanced ResNet18；fresh-final evaluation 已關閉，不得用於
後續調參或模型更新。

Final scope remains cropped parking-space classification. YOLO、完整停車場自動
框選、fresh-final reopening 與新一輪 modeling 不屬於本專題已完成範圍。

## 15. Instructions for Codex

每次工作前：

1. 閱讀 `PROJECT_PLAN.md`
2. 閱讀 `HANDOFF.md`
3. 檢查 repository 現況
4. 只執行目前指定 Milestone
5. 不要提前做後續模型
6. 不得虛構 dataset 欄位或 metrics
7. 不得使用容易造成 leakage 的 random image split
8. 每個模型必須使用一致 split 才能公平比較
9. 每個 Stage 完成後更新 HANDOFF.md
10. 清楚記錄新增檔案、修改檔案、測試、限制與下一步

## 16. HANDOFF Requirements

每次階段完成後至少記錄：

- Date
- Current Stage
- Completed tasks
- New files
- Modified files
- Dataset source
- Dataset size
- Label definition
- Split strategy
- Leakage risk
- Model results（若有）
- Tests
- Known limitations
- Next recommended step
- User decisions required

## 17. Final Deliverables

1. GitHub Repository
2. Dataset / Split 說明
3. Simple CNN baseline
4. ResNet18 Transfer Learning
5. Cross-domain evaluation
6. Fine-tuning experiment
7. Error Analysis
8. Demo
9. Demo Video / GIF
10. Research Summary
11. 研究所備審專題摘要

## 18. Project Status

✅ Complete

Current Stage:
> Milestone 11 — Portfolio Finalization completed

Next Stage:
> None planned; repository is ready for review and publication
