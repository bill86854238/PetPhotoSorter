from pathlib import Path
import sys
import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from PIL import Image

# 將專案根目錄加入 Python 模組路徑
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import config.settings as cfg

# --- 初始化 YOLO 模型 ---
model = YOLO("yolov8n.pt")

# 自動建立輸出根目錄
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# --- 核心函數 ---
def smart_brightness(img):
    """計算圖像最亮 20% 像素的平均亮度"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat = gray.flatten()
    flat.sort()
    top20 = flat[int(len(flat)*0.8):]
    return np.mean(top20)

def classify_action(img):
    """依照寵物姿勢高寬比判斷動作"""
    h, w = img.shape[:2]
    ratio = h / w
    if ratio < cfg.ACTION_RATIO_LIE_DOWN:
        return "躺下"
    elif ratio > cfg.ACTION_RATIO_STAND:
        return "站立"
    else:
        return "坐玩耍"

def get_photo_datetime(filepath):
    """嘗試讀取 EXIF 拍照日期，失敗則用檔案修改時間"""
    try:
        img = Image.open(filepath)
        exif = img.getexif()
        dt_str = exif.get(36867)  # DateTimeOriginal
        if dt_str:
            return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
    except:
        pass
    t = os.path.getmtime(filepath)
    return datetime.fromtimestamp(t)

# --- 主程式 ---
print(f"--- 開始處理目錄: {cfg.SOURCE_DIR} ---")
brightness_results = {}

for filepath in Path(cfg.SOURCE_DIR).rglob("*"):
    if not filepath.is_file() or filepath.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    filename = filepath.name
    filepath_str = str(filepath)
    print(f"\n處理: {filepath_str}")

    # YOLO 偵測
    results = model(filepath_str, verbose=False)
    dogs_detected = [box for box in results[0].boxes if int(box.cls) == 16]

    if not dogs_detected:
        if not cfg.SKIP_NO_DOG:
            # 沒有偵測到狗，複製到 unknown
            dest = os.path.join(cfg.OUTPUT_DIR, "unknown", filename)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if not os.path.exists(dest) or cfg.OVERWRITE_EXISTING:
                shutil.copy2(filepath_str, dest)
            brightness_results[filename] = "N/A (No Dog Detected)"
            print("  - 未偵測到狗 -> unknown")
        else:
            print("  - 未偵測到狗 -> 跳過")
        continue

    img = cv2.imread(filepath_str)
    photo_dt = get_photo_datetime(filepath)

    for idx, box in enumerate(dogs_detected, start=1):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if y1 >= y2 or x1 >= x2:
            continue

        dog_crop = img[y1:y2, x1:x2]
        brightness = smart_brightness(dog_crop)
        target = "二季" if brightness > cfg.BRIGHTNESS_THRESHOLD else "四季"

        # 判斷是否分類動作
        if cfg.ENABLE_ACTION_CLASSIFY:
            action = classify_action(dog_crop)
        else:
            action = ""

        # 判斷是否依日期分類
        if cfg.ENABLE_DATE_CLASSIFY:
            year_month = photo_dt.strftime("%Y/%m")
            base_dir = os.path.join(cfg.OUTPUT_DIR, target, year_month)
        else:
            base_dir = os.path.join(cfg.OUTPUT_DIR, target)

        if action:
            base_dir = os.path.join(base_dir, action)

        os.makedirs(base_dir, exist_ok=True)

        # 改檔名為年月日時分秒（可選）
        if cfg.RENAME_BY_DATETIME:
            dest_filename = photo_dt.strftime("%Y%m%d_%H%M%S")
            if len(dogs_detected) > 1:
                dest_filename += f"_dog{idx}"
            dest_filename += filepath.suffix
        else:
            dest_filename = filename

        dest = os.path.join(base_dir, dest_filename)
        if not os.path.exists(dest) or cfg.OVERWRITE_EXISTING:
            shutil.copy2(filepath_str, dest)

        brightness_results[dest_filename] = f"{brightness:.2f} -> {target}/{action}" if action else f"{brightness:.2f} -> {target}"
        print(f"  - 亮度: {brightness:.2f} -> 分類: {target}/{action}" if action else f"  - 亮度: {brightness:.2f} -> 分類: {target}")

print("\n--- 完成處理 ---")
print("--- 分類結果 ---")
for fname, result in brightness_results.items():
    print(f"  {fname}: {result}")
