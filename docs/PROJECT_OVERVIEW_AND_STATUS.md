# 專案介紹、目前狀態與技術總覽

個人自主 AI 研究作品｜已完成並凍結

研究主線：發現跨場域落差 → 分析錯誤 → 改善場域穩健性 → 預先鎖定規則的最終評估。

## 1. 專案簡介

**Parking Occupancy Detection — Cross-Domain Robustness Study** 是一個電腦視覺研究專案，目標是將一張已裁切的單一停車格影像分類為：

- `EMPTY`：停車格為空
- `OCCUPIED`：停車格已被車輛占用

本專案不只訓練一個分類器，也研究模型從原始資料場域移到不同停車場後所產生的 **domain shift（跨場域分布差異）**。完整研究歷程涵蓋基準模型、遷移學習、跨資料集測試、目標場域適應、錯誤分析、模型穩健性改善，以及預先鎖定規則的一次性最終評估。

公開展示：[Streamlit Live Demo](https://parking-occupancy-detection-hk9l6wzyvtkrqjr6tkvftc.streamlit.app/)

> Demo 的輸入必須是「單一、已裁切的停車格影像」。系統不包含自動車位定位、整張停車場偵測、CCTV 串流或即時停車管理功能。

## 2. 專案目前狀態

| 項目 | 目前狀態 |
|---|---|
| 研究與模型開發 | 已完成並凍結 |
| Portfolio Finalization（Milestone 11） | 已完成 |
| 最終模型 | V2-A Balanced ResNet18 |
| 最終 checkpoint | `models/v2a_balanced_resnet18.pt` |
| 決策 threshold | `0.5`（已鎖定） |
| 前處理 | 已鎖定，不再調整 |
| Fresh-final evaluation | 已完成且關閉，不再重新開啟 |
| 公開 GitHub repository | 已整理為作品集形式 |
| 公開 Streamlit Demo | 已上線 |
| Demo 體驗 | Try a Sample（10 張展示素材）與 Upload Your Own |
| 後續研究訓練 | 無；目前為公開展示與文件維護階段 |

最終 fresh-final 結果來自一次性、154,669 筆樣本的鎖定評估：

| Metric | V2-A 結果 |
|---|---:|
| Accuracy | 99.8894% |
| Occupied Precision | 99.8766% |
| Occupied Recall | 99.9058% |
| Occupied F1 | 99.8912% |
| UFPR04 Occupied Recall | 100.0000% |
| False Positives | 97 |
| False Negatives | 74 |

Confusion matrix：`[[75989, 97], [74, 78509]]`

這些數值是已完成研究的終端結果，不會再用於調整模型、重新選模、校正 threshold 或開啟新一輪訓練。

## 3. 研究動機與核心問題

停車格分類在單一資料集內可能得到很高的準確率，但實際移到另一個停車場時，攝影機角度、光線、天候、地面材質、車位線與車型分布都可能改變。因此，本專案關注以下問題：

1. 小型 SimpleCNN 能否建立合理的 baseline？
2. ImageNet 預訓練的 ResNet18 是否能在相同 split 上改善結果？
3. 模型從 CNRPark+EXT 移到 PKLot 時會產生多大的 domain shift？
4. Target-domain adaptation 能否有效恢復跨場域表現？
5. 剩餘錯誤集中在哪些 site 與錯誤類型？
6. 能否透過事先承諾的平衡取樣與合理 augmentation，改善 UFPR04 的 false-negative 弱點，同時避免整體 F1 下降？

## 4. 完整研究歷程

```text
CNRPark+EXT 資料準備
        ↓
SimpleCNN baseline
        ↓
ImageNet ResNet18 transfer learning
        ↓
CNR-EXT in-domain test
        ↓
PKLot zero-shot evaluation → 發現 domain shift
        ↓
PKLot target adaptation（V1）
        ↓
Error analysis → 發現 UFPR04 false-negative 弱點
        ↓
預先鎖定 V2 robustness protocol
        ↓
V2-A ResNet18 vs V2-B EfficientNet-B0（validation-only selection）
        ↓
一次性 fresh-final V1 vs V2-A comparison
        ↓
鎖定 V2-A，建立公開 Streamlit Demo
```

重要研究結果：

- ResNet18 transfer learning 在相同 CNR-EXT validation split 上優於 SimpleCNN。
- 原始 ResNet18 從 CNR-EXT 直接移到 PKLot 時，Occupied F1 由 `0.988692` 降至 `0.788652`，顯示明顯跨場域落差。
- V1 target adaptation 將 PKLot held-out Occupied F1 提升至 `0.986545`。
- Error analysis 發現 V1 的主要弱點是 false negatives，尤其集中於 UFPR04。
- V2 採用 site/label-balanced sampling、溫和 augmentation 與 validation-only selection。
- V2-A 在 fresh-final 上通過預先承諾的 robustness gate，並成為最終 凍結展示模型。

不同階段使用不同 dataset 與 split protocol，相關 metrics 必須保留原始語境，不能全部視為同一排行榜直接比較。

## 5. 本專案應用的技術

### 5.1 程式語言與核心框架

| 技術 | 專案用途 |
|---|---|
| Python | 資料處理、模型訓練、評估、推論、測試與 Demo |
| PyTorch | 神經網路建模、訓練、checkpoint 儲存與 CPU/MPS 推論 |
| torchvision | ResNet18、EfficientNet-B0、影像 transformation 與 ImageNet normalization |
| Streamlit | 建立公開互動式 AI Portfolio Demo |

### 5.2 電腦視覺與深度學習

| 技術 | 實際應用 |
|---|---|
| Binary image classification | 將單一停車格分類為 EMPTY 或 OCCUPIED |
| Convolutional Neural Network | 建立 98,178-parameter SimpleCNN baseline |
| Transfer learning | 使用 ImageNet 預訓練 ResNet18，加速收斂並改善分類表現 |
| Fine-tuning | 由凍結 backbone 到全模型微調，並進行 PKLot target adaptation |
| ResNet18 | Source-domain、V1 adaptation 與最終 V2-A 凍結展示 backbone |
| EfficientNet-B0 | V2 階段唯一受控的 alternative backbone，僅用 validation 選模 |
| Data augmentation | V2 使用合理且溫和的影像變換改善場域穩健性 |
| Balanced sampling | 以 site/label cell 平衡取樣，減少特定場域或類別支配訓練 |

### 5.3 影像處理與資料工具

| 技術 | 專案用途 |
|---|---|
| Pillow | JPEG、PNG、WebP 解碼、RGB 轉換及上傳影像驗證 |
| OpenCV | 影像處理與研究階段的視覺資料工具 |
| NumPy | 陣列、數值運算與評估資料處理 |
| Pandas | 研究環境的表格分析依賴；核心 manifest 流程另使用 Python CSV/JSON 工具 |
| Matplotlib | 研究環境的繪圖依賴；目前公開研究圖表由專案 SVG 工具產生 |
| scikit-learn | 研究環境的機器學習分析依賴；鎖定核心 metrics 由專案 Python 函式計算 |

### 5.4 跨場域研究方法

- **Source-domain training：** 使用 CNRPark+EXT 建立初始模型。
- **In-domain evaluation：** 在相同資料來源的隔離 test split 評估模型。
- **Zero-shot cross-domain evaluation：** 不經 PKLot 訓練，直接測量模型遷移到新場域時的表現下降。
- **Target-domain adaptation：** 使用隔離的 PKLot development data 微調模型。
- **Site/date grouped splitting：** 依 site、日期、camera、slot 或 source frame 分組，降低相似影像跨 split 所造成的 leakage。
- **Macro-site metrics：** 除整體 metrics 外，同時觀察不同停車場的表現，避免大場域掩蓋小場域弱點。
- **Error analysis：** 分析 FP、FN、site、日期與影像案例，但禁止將已分析的 held-out errors 回流至 V2 開發。
- **Precommitted protocol：** 在訓練前先鎖定 split、seed、設定、選模標準與 robustness gate。
- **One-time terminal evaluation：** Fresh-final 僅開啟一次，結果不得再作為後續模型調整依據。

### 5.5 可重現性與研究治理

- 以 JSON、CSV 與 Markdown 保存實驗設定、manifest、split 與結果。
- 記錄 random seeds、training history、runtime/device 與 confusion matrix。
- 使用 SHA-256 鎖定 checkpoint、config、selection artifact 與最終結果。
- 大型資料集只放在外接儲存裝置，透過 `PARKING_DATA_ROOT` 設定位置。
- Repository 只保存 metadata、相對路徑、程式碼、小型結果與 凍結展示 checkpoint。
- 不建立 train/validation/test 圖片副本，降低儲存浪費與資料洩漏風險。
- 使用 Python `unittest` 驗證資料路徑、split、metrics、模型輸出、推論與 robustness gate。
- 使用 Git 與 GitHub 保存版本與公開研究紀錄。

### 5.6 鎖定展示推論契約與 Demo

最終推論流程保持鎖定：

```text
JPEG / PNG / WebP 單一停車格影像
        ↓
格式與 10 MB 大小驗證
        ↓
轉換為 RGB
        ↓
Symmetric edge padding 成正方形
        ↓
224 × 224 bilinear resize
        ↓
ImageNet mean/std normalization
        ↓
V2-A Balanced ResNet18（CPU）
        ↓
Softmax class scores
        ↓
Occupied score ≥ 0.5 → OCCUPIED，否則 EMPTY
```

Demo 具備：

- 10 張 purpose-created sample images，供使用者立即體驗，並非評估證據。
- 自行上傳 JPEG、PNG 或 WebP。
- 顯示 EMPTY／OCCUPIED、predicted-class confidence 與兩個 class scores。
- Checkpoint SHA-256、candidate ID、architecture 與 config hash 驗證。
- 不需要 dataset、外接 SSD、API key 或 secret。
- 所有上傳影像只在記憶體中處理，不寫入 repository。

## 6. 資料集與儲存架構

| Dataset | 研究角色 |
|---|---|
| CNRPark+EXT | Source-domain baseline、transfer learning 與 in-domain evaluation |
| PKLot | Zero-shot domain-shift、target adaptation、error analysis 與 V2 robustness study |

大型 image archive 不存放在 GitHub repository，而是由外接 SSD 管理。`data/` 只保存 portable manifest、metadata、small config、artifact lock 與 split assignment。

```bash
export PARKING_DATA_ROOT="/absolute/path/to/external-ssd/parking-datasets"
```

公開 Demo 不需要設定 `PARKING_DATA_ROOT`，也不會存取任何訓練、驗證或 final evaluation image。

## 7. Repository 結構

```text
parking-occupancy-detection/
├── app/          # Streamlit Demo 與 purpose-created sample manifest
├── data/         # Metadata、split、config 與 artifact locks
├── docs/         # 研究報告、模型比較及專案摘要
├── images/       # 圖表、Demo 素材、截圖與 GIF
├── models/       # 鎖定的 V2-A 凍結展示 checkpoint
├── results/      # 完整實驗與評估 JSON
├── src/          # 資料、訓練、評估、推論與視覺化程式
└── tests/        # 自動化測試
```

唯一公開權重為 V2-A：44,790,987 bytes（42.72 MiB）。保留在一般 Git 中，讓 clone 後可以直接啟動展示；暫不使用 Git LFS，也不新增其他模型權重。

## 8. 如何執行公開 Demo

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r app/requirements.txt
streamlit run app/app.py
```

預設會載入 `models/v2a_balanced_resnet18.pt`，並以 CPU 執行。程式會驗證 checkpoint 是否為鎖定的 V2-A artifact，不允許靜默替換成其他模型。

## 9. 專案限制

- 僅支援已裁切的單一停車格分類，不包含 parking-space localization。
- 不支援整張停車場影像偵測、CCTV 串流或即時監控。
- Softmax confidence 未進行 probability calibration。
- Weather、site 與 date 可能互相混雜，因此不能把相關性解讀為因果關係。
- CNRPark+EXT 主要代表單一實體停車區域，camera holdout 不等同完整的 cross-location generalization。
- Fresh-final 與 V1 adaptation、V2 development 隔離，但 PKLot 圖片在更早階段曾接受 V1 inference；不宣稱從未被 V1 推論過。
- Fresh-final 已關閉，不能再用於模型更新或 threshold 調整。

研究凍結不等於封存 repository；一般文件與軟體維護仍可進行。Linux CPU CI 已準備，執行狀態以 GitHub Actions 對應 commit 的實際結果為準，不預先宣稱通過。

## 10. 延伸閱讀

- [README：專案入口與完整研究故事](../README.md)
- [Final Research Summary](FINAL_RESEARCH_SUMMARY.md)
- [Model Comparison](MODEL_COMPARISON.md)
- [Cross-Domain Evaluation](CROSS_DOMAIN_EVALUATION.md)
- [Error Analysis](ERROR_ANALYSIS.md)
- [V2 Training and Selection](V2_TRAINING_SELECTION.md)
- [V2 Fresh-Final Comparison](V2_FRESH_FINAL_COMPARISON.md)
- [Final Streamlit Demo](FINAL_DEMO.md)
