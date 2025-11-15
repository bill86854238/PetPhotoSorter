```markdown
# 🐶 Pet Photo Classifier  
自動分類二季、四季的照片，並加入 CLIP 驗證、ID 修正與「潛在誤判或修正」備援資料夾機制。

---

## 📌 專案用途  
將大量手機備份照片自動分類為二季 / 四季，同時依照動作、顏色、時間等額外維度進行整理。  
搭配 YOLO + CLIP 雙模型，可降低誤判，並建立「潛在誤判或修正」資料夾方便人工檢查。

---

## 🧩 功能摘要  

### 1️⃣ 狗狗偵測與分類  
- 使用 **YOLOv8** 偵測照片中的狗  
- 亮度邏輯區分：
  - 淺色 → 二季  
  - 深色 → 四季  

### 2️⃣ CLIP 驗證與修正  
- CLIP 驗證是否真的為狗（避免熊、狐狸等誤判）  
- CLIP 進行毛色分析，比對初判結果後 **自動 ID 修正**  
- 若初判與 CLIP 判斷不同 → 加上標籤 `[二季被四季修正]`  

### 3️⃣ 潛在誤判備援資料夾  
自動複製疑似誤判圖至：  
```

潛在誤判或修正/

```

標籤邏輯：  
| 觸發條件 | 標籤範例 | 說明 |
| ---- | ---- | ---- |
| CLIP 低信心 | `[低信心]` | 狗機率 < 設定閾值 |
| ID 修正 | `[二季被四季修正]` | 初判與 CLIP 衝突 |
| 兩者皆有 | `[二季被四季修正_低信心]` | 同時發生 |

---

## 📁 輸出資料夾結構  

```

Pet/
├─ 二季/
│   ├─ 2024/
│   └─ 2025/
├─ 四季/
│   ├─ 2024/
│   └─ 2025/
├─ 潛在誤判或修正/
└─ logs/

````

---

## ⚙️ 安裝需求  

### 安裝模型相關套件  
```bash
pip install ultralytics transformers torch torchvision pillow opencv-python numpy
````

### 使用的模型

* YOLOv8n（物件偵測）
* CLIP (openai/clip-vit-base-patch32)

---

## 🛠️ Config 設定說明（部分）

```python
class Config:
    SOURCE_DIR = Path(r"\\192.168.50.143\home\Photos\MobileBackup") 
    OUTPUT_DIR = Path(r"D:/Pet")

    BRIGHTNESS_THRESHOLD = 185
    DOG_ID_BRIGHT_NAME = "二季"
    DOG_ID_DARK_NAME = "四季"

    CLIP_VERIFY_MULTIPLIER = 1.2
    CLIP_MIN_DOG_PROBABILITY = 0.6

    UNCERTAINTY_FOLDER_NAME = "潛在誤判或修正"
```

---

## 🚀 執行方式

```bash
python main.py
```

執行後會自動：

* 掃描來源資料夾
* 依照分類結果建立資料夾
* 將照片重新命名、複製、標註必要前綴
* 對可疑案例加上標籤並放入備援資料夾

---

## 🧪 典型流程

1. 讀取照片
2. YOLO 偵測狗
3. 初判毛色 → 二季/四季
4. CLIP 驗證是否為狗
5. CLIP 毛色分析 → 修正 ID（若不一致）
6. 若低信心或 ID 修正 → 複製到備援資料夾
7. 依照拍照日期重新命名 → 輸出完成

---

## 📜 Log 記錄

每次執行會生成：

```
logs/
  └─ misclassified_report.txt
```

內容包含：

* 誤判原因
* CLIP 機率
* ID 修正紀錄

---

## 🧷 FAQ

### Q1. CLIP 模型載不動 / 記憶體不足？

可將模型改為更小版本或改用 CPU 模式執行。

### Q2. YOLO 偵測不到狗？

確認圖片是否太暗、模糊或角度怪異。
可考慮升級模型至 yolov8s 或 yolov8m。

---

## 🐾 支援後續功能（可加入）

* 臉部辨識（區分二季 / 四季臉部特徵）
* 動作分類更細緻（睡姿、站姿、玩玩具、奔跑等）
* 自動產生相簿（HTML / PDF）

---

如果你需要我 **幫你加上程式片段**、**產生範例日誌**、或 **寫入 GitHub release 用語**，隨時跟我說！

```markdown
```
