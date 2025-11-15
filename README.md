# 🐶 Pet Photo Classifier

自動分類「二季 / 四季」照片，並整合 YOLO + CLIP 做驗證、毛色比對與潛在誤判標註。

---

## 快速開始

1. 安裝需求
```bash
pip install -r requirements.txt
```
或
```bash
pip install ultralytics transformers torch torchvision pillow opencv-python numpy
```

2. 執行（使用專案根目錄）
```bash
python pet_sorter.py
```
或使用腳本版本：
```bash
python scripts/pet_sorter.py
```

注意：程式會嘗試載入 YOLO 權重 [yolov8n.pt](yolov8n.pt)。

---

## 功能總覽

- 使用 [`get_dog_bbox`](pet_sorter.py)（YOLO）偵測狗與人。
- 使用亮度邏輯和 [`classify_color_with_clip`](pet_sorter.py)（CLIP）判定 ID（淺色->二季、深色->四季）。
- 使用 [`verify_is_dog_with_clip`](pet_sorter.py) 避免誤判（熊、狐狸等）。
- 若有低信心或 ID 修正，複製至「潛在誤判或修正」資料夾，並在檔名前綴標註原因。
- 支援依日期/動作分資料夾與以拍照時間重新命名。

---

## 執行流程（概要）

1. 讀取來源資料夾（由 [`Config`](pet_sorter.py).SOURCE_DIR 提供）
2. 預載已處理圖片哈希（[`preload_hashes`](pet_sorter.py)）
3. 對每張圖執行 [`process_file`](pet_sorter.py)：
   - 提取時間（EXIF 或檔名／檔案修改時間）
   - 計算 aHash（[`calculate_aHash`](pet_sorter.py)）以避免重複
   - YOLO 偵測與裁切（[`crop_image`](pet_sorter.py)）
   - 縮放以加速推理（[`resize_for_ai`](pet_sorter.py)）
   - CLIP 驗證與毛色/動作判斷（[`verify_is_dog_with_clip`](pet_sorter.py)、[`classify_color_with_clip`](pet_sorter.py)、[`classify_action`](pet_sorter.py)）
   - 建構目標路徑並複製（[`construct_path_and_copy`](pet_sorter.py)）

---

## 主要設定（摘錄）

在程式中的 `Config` 類別（[`Config`](pet_sorter.py)）可調整：
- SOURCE_DIR / OUTPUT_DIR
- BRIGHTNESS_THRESHOLD（亮度分界）
- CLIP_VERIFY_MULTIPLIER、CLIP_MIN_DOG_PROBABILITY
- ENABLE_DUPLICATE_CHECK、SKIP_NO_DOG、OVERWRITE_EXISTING
- UNCERTAINTY_FOLDER_NAME（預設 "潛在誤判或修正"）

---

## 故障排除

- CLIP 載不動：改用 CPU 或選更小模型，或確認 transformers / torch 版本。
  參考：[`initialize_models`](pet_sorter.py)
- YOLO 偵測不到：嘗試提升模型大小（yolov8s / yolov8m）或調整圖片品質。
- 中文路徑問題：程式使用 cv2.imdecode 與 numpy.fromfile 來支援。

---

## 輸出結構範例

Pet/
├─ 二季/  
│  ├─ YYYY/MM/  
├─ 四季/  
├─ 潛在誤判或修正/  
└─ logs/（含 misclassified_report.txt）

---

## 日誌與報表

執行會輸出詳細 log（INFO / WARNING / ERROR），錯誤或疑似誤判會記錄於 logs/misclassified_report.txt。

---

## 參考程式檔案

- 主程式與核心實作：[`pet_sorter.py`](pet_sorter.py)（含函式：[`process_file`](pet_sorter.py)、[`calculate_aHash`](pet_sorter.py)、[`get_dog_bbox`](pet_sorter.py) 等）
- 簡化腳本：[`scripts/pet_sorter.py`](scripts/pet_sorter.py)
- 權重檔案：[yolov8n.pt](yolov8n.pt)
- 依賴清單：[requirements.txt](requirements.txt)

---

如果你要我替你：
- 加入更完整的範例輸出（logs/misclassified_report.txt 範例），或
- 將 README 加入執行範例截圖、或
- 產生 release notes 範本，

告訴我想要加入的內容即可，我會幫你產生對應段落與範例。
