import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
import config.settings as cfg

# --- 初始化 ---
model = YOLO("yolov8n.pt")

# 創建輸出目錄
for dog_color in ["二季", "四季", "unknown"]:
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, dog_color), exist_ok=True)

# --- 核心函數 ---
def smart_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat = gray.flatten()
    flat.sort()
    top20 = flat[int(len(flat)*0.8):]
    return np.mean(top20)

def classify_action(img):
    h, w = img.shape[:2]
    ratio = h / w
    if ratio < cfg.ACTION_RATIO_LIE_DOWN:
        return "躺下"
    elif ratio > cfg.ACTION_RATIO_STAND:
        return "站立"
    else:
        return "坐玩耍"

# --- 主程式 ---
print(f"--- 開始處理目錄: {cfg.SOURCE_DIR} ---")
brightness_results = {}

for filename in os.listdir(cfg.SOURCE_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    filepath = os.path.join(cfg.SOURCE_DIR, filename)
    print(f"\n處理: {filename}")
    results = model(filepath, verbose=False)

    if len(results[0].boxes) == 0:
        dest = os.path.join(cfg.OUTPUT_DIR, "unknown", filename)
        if not os.path.exists(dest) or cfg.OVERWRITE_EXISTING:
            shutil.copy2(filepath, dest)
        brightness_results[filename] = "N/A (No Object Detected)"
        print("  - 未檢測到物體 -> unknown")
        continue

    assigned = False
    for box in results[0].boxes:
        cls = int(box.cls)
        if cls != 16:  # 只關注狗
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        img = cv2.imread(filepath)
        if y1 >= y2 or x1 >= x2:
            continue

        dog_crop = img[y1:y2, x1:x2]
        brightness = smart_brightness(dog_crop)
        action = classify_action(dog_crop)

        target = "二季" if brightness > cfg.BRIGHTNESS_THRESHOLD else "四季"
        action_dir = os.path.join(cfg.OUTPUT_DIR, target, action)
        os.makedirs(action_dir, exist_ok=True)

        dest = os.path.join(action_dir, filename)
        if not os.path.exists(dest) or cfg.OVERWRITE_EXISTING:
            shutil.copy2(filepath, dest)

        assigned = True
        brightness_results[filename] = f"{brightness:.2f} -> {target}/{action}"
        print(f"  - 亮度: {brightness:.2f} -> 分類: {target}/{action}")
        break

    if not assigned:
        dest = os.path.join(cfg.OUTPUT_DIR, "unknown", filename)
        if not os.path.exists(dest) or cfg.OVERWRITE_EXISTING:
            shutil.copy2(filepath, dest)
        brightness_results[filename] = "N/A (No Dog Detected)"
        print("  - 偵測到物體但沒有狗 -> unknown")

print("\n--- 完成處理 ---")
print("--- 分類結果 ---")
for fname, result in brightness_results.items():
    print(f"  {fname}: {result}")
