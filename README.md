# 🐶 Pet Photo Sorter Pro (M4 Optimized)

這是一個專為寵物家庭打造的「專業級」影像管理管家。針對 Apple Silicon (M4/M3/M2/M1) 進行深度效能優化，結合 macOS 原生視覺技術，讓數萬張照片的整理從沉重的任務變成一種享受。

---

## 🌟 核心進化 (M4 Pro 特色)

### 1. 極速效能優化 (Apple Silicon Native)
- **MPS 硬體加速**：全面啟用 Metal Performance Shaders，讓 YOLO 與 CLIP 運行在 Mac 的 GPU/NPU 上，效能提升 5-10 倍。
- **非同步批次處理 (Batch Inference)**：採用多執行緒預載技術與批次推論（一次處理 16 張），徹底榨乾 M4 晶片的處理能力。

### 2. macOS 原生美感評分系統
- **Apple Vision 整合**：呼叫 macOS 內建的 `VNGenerateImageAestheticsScoresRequest` 演算法，瞬間為每一張照片打分 (0.0~1.0)。
- **漏斗式篩選**：
    - **精選組 (Score > 0.6)**：視為珍貴瞬間，啟動 Ollama AI 撰寫深情日記。
    - **普通組**：標準分類與歸檔。
    - **低分組 (Score < 0.2)**：自動移至 `低分` 資料夾，節省系統資源。

### 3. Finder 深度整合 (Metadata & Tags)
- **自動標籤**：根據狗狗 ID 自動打上 Finder 顏色標籤（二季：藍色、四季：橘色、精選：紅色）。
- **分數註解**：美感分數會直接寫入檔案的「Finder 註解」欄位，讓您在 Finder 列表就能直接排序精選照片。

### 4. 進階 Obsidian 紀念書
- **星等視覺化**：筆記中自動顯示 ★★★★★ 星等評分。
- **豐富元數據**：包含拍攝設備 (iPhone 17 Pro 預設)、驗證狀態與美感得分。

---

## 🚀 快速開始 (macOS 推薦)

### 1. 安裝環境與原生依賴
```bash
# 安裝基礎套件
pip install ultralytics transformers torch torchvision pillow opencv-python numpy

# 安裝 macOS 原生優化套件 (僅限 Mac 使用者)
pip install osxmetadata pyobjc-framework-Vision pyobjc-framework-Quartz
```

### 2. 啟動 Ollama
```bash
ollama pull qwen2.5-vl:7b
```

### 3. 設定 `config.json`
您可以調整 `batch_size` (建議 16) 與 `aesthetic_score_high` 等參數，來決定哪些照片值得讓 AI 寫日記。

### 4. 執行
```bash
python pet_sorter.py
```

---

## 📂 輸出結構範例

```text
照片-Pet_分類/
├── 二季/ (Finder 藍色標籤 🔵)
│   ├── 動作_奔跑/ (包含 [S_85] 前綴的高分照片)
│   └── ...
├── 低分/ (美感分數過低的存檔)
├── Obsidian_Logs/ 
│   └── 二季/
│       └── 2024/
│           └── [S_85]_二季_奔跑.md (含星等 ★★★★☆ 與 AI 日記)
└── scan_cache.json (智慧掃描快取)
```

---

## 🛠 技術規格
- **硬體加速**：Apple MPS (Metal) / NVIDIA CUDA / CPU
- **美感評分**：macOS Vision Framework (Aesthetics Scores)
- **元數據**：XATTR / Finder Info (via osxmetadata)
- **AI 核心**：YOLOv8, CLIP, Ollama

---

## 📜 授權與感謝
本專案專為熱愛寵物的開發者打造。在 Windows 上執行時會自動退回標準模式，不影響基本功能。

*(註：本專案中的「二季」與「四季」為開發者家中的愛犬名稱。如果您喜歡這套針對 Mac 優化的系統，歡迎給個 Star！)*