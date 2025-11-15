from pathlib import Path
import sys
import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from PIL import Image

# 新增 CLIP 相關的函式庫
from transformers import CLIPProcessor, CLIPModel
import torch

# ==============================================================================
# 設定區塊 (Config Block)
# ==============================================================================
class MockConfig:
    # 檔案路徑設定 (根據您的專案路徑設定)
    SOURCE_DIR = r"\\192.168.50.143\home\Photos\MobileBackup" # <-- 您的網路輸入路徑
    OUTPUT_DIR = r"D:/Pet"                                   # <-- 您的本地輸出路徑
    
    # --- 狗狗 ID 分類設定 (以二季/四季為 ID) ---
    # 亮度/顏色 初步判斷閾值 (用於將圖片分類到狗狗 ID)
    BRIGHTNESS_THRESHOLD = 185 
    # 亮度高或淺色特徵對應的狗狗 ID
    DOG_ID_BRIGHT_NAME = "二季"
    # 亮度低或深色特徵對應的狗狗 ID
    DOG_ID_DARK_NAME = "四季"
    
    # 功能開關
    SKIP_NO_DOG = True             
    OVERWRITE_EXISTING = False     
    ENABLE_ACTION_CLASSIFY = True  # 動作分類已啟用 (透過 CLIP 實現)
    ENABLE_DATE_CLASSIFY = False   # ** 已關閉：腳本執行日期分類 (只使用批次資料夾) **
    RENAME_BY_DATETIME = True      # 檔名會加上動作標籤
    ENABLE_COLOR_CLASSIFY = True   # 毛色分類已啟用 (透過 CLIP 實現，用於輔助狗狗 ID 判斷)
    ENABLE_GEAR_CLASSIFY = False   
    
    # 【手動開關】: 若設為 False，則輸出路徑中不會有「戶外/室內」資料夾。
    ENABLE_ENV_CLASSIFY = False    # ** 已關閉：環境分類 **
    
    SKIP_IF_HUMAN_FACE = False 
    
    # --- 其他配置 (保留) ---
    CROP_SHRINK_RATIO = 0.10 
    
    # 環境判斷設定 (HSV 範圍)
    ENV_OUTDOOR_PIXEL_RATIO = 0.1 
    GREEN_LOWER = np.array([40, 40, 40])
    GREEN_UPPER = np.array([80, 255, 255])
    BLUE_LOWER = np.array([100, 40, 40])
    BLUE_UPPER = np.array([140, 255, 255])

# 假設 cfg 已被正確載入
cfg = MockConfig() 

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# --- 初始化 CLIP 模型 (用於動作/毛色/語義判斷) ---
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_INITIALIZED = False
device = "cpu"

try:
    # 嘗試使用 CUDA，否則使用 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CLIP 運行設備: {device}")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    CLIP_INITIALIZED = True
    print("CLIP 模型載入成功。")
except Exception as e:
    print(f"錯誤：無法載入 CLIP 模型。錯誤訊息: {e}")
    print("動作分類、毛色分類與誤判過濾功能將被禁用。")
    CLIP_INITIALIZED = False


# --- 初始化 YOLO 模型 ---
try:
    # 確保 yolov8n.pt 文件存在，如果不存在，YOLO 會自動下載
    model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"錯誤：無法載入 YOLO 模型。錯誤訊息: {e}")
    sys.exit(1)


# 自動建立輸出根目錄
try:
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
except Exception as e:
    print(f"錯誤：無法建立或存取輸出目錄 {cfg.OUTPUT_DIR}。錯誤訊息: {e}")
    sys.exit(1)


# --- 核心函數 ---

def smart_brightness(img):
    """計算圖像最亮 20% 像素的平均亮度 (用於狗狗 ID 初判)"""
    if img is None or img.size == 0: return 0
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flat = gray.flatten()
        flat.sort()
        top20_index = int(len(flat) * 0.8)
        top20 = flat[top20_index:]
        return np.mean(top20) if top20.size > 0 else 0
    except Exception: return 0


def classify_action(dog_crop, cfg, filepath_name):
    """使用 CLIP 模型判斷狗的動作"""
    if not CLIP_INITIALIZED or dog_crop is None or dog_crop.size == 0:
        return "動作_無效"

    try:
        dog_crop_rgb = cv2.cvtColor(dog_crop, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(dog_crop_rgb)
    except Exception:
        return "動作_圖像轉換失敗"

    action_descriptions = [
        "A dog is standing and looking around.",      # 站立 (Standing)
        "A dog is sitting with its front paws down.",  # 坐下 (Sitting)
        "A dog is lying down, resting or sleeping.",   # 躺臥 (Lying Down/Sleeping)
        "A dog is running, jumping, or moving quickly." # 活動 (Active/Moving)
    ]

    inputs = clip_processor(text=action_descriptions, images=image_pil, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clip_model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).squeeze().cpu().numpy()
    
    max_index = np.argmax(probs)
    action_label = action_descriptions[max_index]

    if "standing" in action_label.lower():
        return "動作_站立"
    elif "sitting" in action_label.lower():
        return "動作_坐下"
    elif "lying down" in action_label.lower() or "resting" in action_label.lower() or "sleeping" in action_label.lower():
        return "動作_躺臥"
    elif "running" in action_label.lower() or "moving" in action_label.lower():
        return "動作_活動"
    
    return "動作_中立" 


def verify_is_dog_with_clip(dog_crop):
    """使用 CLIP 模型驗證偵測到的物體是否真的是狗"""
    if not CLIP_INITIALIZED: return True

    if dog_crop is None or dog_crop.size == 0: return True

    try:
        dog_crop_rgb = cv2.cvtColor(dog_crop, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(dog_crop_rgb)
    except Exception: return True

    verification_descriptions = [
        "A photograph of a pet dog (Labrador, Husky, Poodle, etc.).",
        "A photograph of a bear (Black bear, brown bear, panda bear, etc.).",
        "A photograph of a red panda (Ailurus fulgens).",
        "A photograph of a wild animal (fox, raccoon, squirrel, etc.)."
    ]

    inputs = clip_processor(text=verification_descriptions, images=image_pil, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clip_model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).squeeze().cpu().numpy()

    dog_prob = probs[0]
    
    max_non_dog_prob = max(probs[1:])
    
    if max_non_dog_prob > dog_prob * 1.2:
        non_dog_index = np.argmax(probs[1:]) + 1
        non_dog_label = verification_descriptions[non_dog_index]
        print(f"  [CLIP 驗證警告]: 狗機率 {dog_prob:.2f}。最高非狗機率 {max_non_dog_prob:.2f} ({non_dog_label[:20]}...)。判定為誤判。")
        return False
    
    print(f"  [CLIP 驗證通過]: 狗機率 {dog_prob:.2f}, 非狗機率最高為 {max_non_dog_prob:.2f}。")
    return True


def classify_color_with_clip(dog_crop, cfg, filepath_name):
    """使用 CLIP 模型判斷狗的顏色特徵 (深色 vs 淺色)"""
    if not CLIP_INITIALIZED or dog_crop is None or dog_crop.size == 0:
        return "未知顏色", None

    try:
        dog_crop_rgb = cv2.cvtColor(dog_crop, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(dog_crop_rgb)
    except Exception as e:
        return "圖像轉換失敗", {"file": filepath_name, "錯誤": f"圖像轉換失敗: {e}"}

    text_descriptions = [
        "A dog with predominantly dark, black, or deep brown fur.",     # 深色
        "A dog with predominantly light, yellow, tan, or white fur.",   # 淺色
        "A multi-colored dog (brindle, black and tan, etc.)",           # 多色
        "A generic dog photograph"                                      # 通用
    ]

    inputs = clip_processor(text=text_descriptions, images=image_pil, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clip_model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).squeeze().cpu().numpy()

    dark_mix_prob = probs[0]
    light_mix_prob = probs[1]

    color_label = "CLIP多色/通用"
    
    if dark_mix_prob > light_mix_prob * 1.1: 
        color_label = "CLIP深色米克斯"
    elif light_mix_prob > dark_mix_prob * 1.1:
        color_label = "CLIP淺色米克斯"

    diag_data = {
        "file": filepath_name,
        "CLIP_Dark_Prob": f"{dark_mix_prob:.4f}",
        "CLIP_Light_Prob": f"{light_mix_prob:.4f}",
        "CLIP_Scores": {desc: f"{p:.4f}" for desc, p in zip(text_descriptions, probs)},
        "判斷依據": f"深色機率 {dark_mix_prob:.2f} vs 淺色機率 {light_mix_prob:.2f}"
    }

    return color_label, diag_data


def get_photo_datetime(filepath):
    """嘗試讀取 EXIF 拍照日期，失敗則用檔案修改時間"""
    try:
        img = Image.open(filepath)
        exif = img.getexif()
        dt_str = exif.get(36867) 
        if dt_str:
            return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    
    try:
        t = os.path.getmtime(filepath)
        return datetime.fromtimestamp(t)
    except Exception:
        return datetime.now()


def classify_environment(img, box, cfg):
    """判斷背景是否為戶外 (檢查綠色/藍色佔比)"""
    if not cfg.ENABLE_ENV_CLASSIFY: return "" # 如果關閉則直接返回空字串

    if img is None or img.size == 0: return "環境判斷錯誤"
        
    try:
        h, w, _ = img.shape
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        mask = np.ones((h, w), dtype=np.uint8) * 255
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
        
        background_pixels = np.sum(mask == 255)
        if background_pixels == 0: return "室內/特寫" 

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        green_mask = cv2.inRange(hsv, cfg.GREEN_LOWER, cfg.GREEN_UPPER)
        blue_mask = cv2.inRange(hsv, cfg.BLUE_LOWER, cfg.BLUE_UPPER)
        
        outdoor_mask = cv2.bitwise_or(green_mask, blue_mask)
        outdoor_mask = cv2.bitwise_and(outdoor_mask, outdoor_mask, mask=mask)
        
        outdoor_pixels = np.sum(outdoor_mask > 0)
        outdoor_ratio = outdoor_pixels / background_pixels
        
        if outdoor_ratio > cfg.ENV_OUTDOOR_PIXEL_RATIO:
            return "戶外"
        else:
            return "室內"
    except Exception: return "環境判斷錯誤"

# --- 預備階段：初始化診斷檔案與設定執行批次資料夾 ---

# ** 取得腳本執行當下的日期時間，並格式化為單一資料夾名稱 (YYYYMMDD_HHMMSS) **
runtime_dt = datetime.now()
RUNTIME_DATETIME_STR = runtime_dt.strftime("%Y%m%d_%H%M%S")
print(f"腳本執行批次資料夾名稱: {RUNTIME_DATETIME_STR}")

# ** 設定輸出路徑和診斷報告路徑 **
target_script_dir = Path(cfg.OUTPUT_DIR) / RUNTIME_DATETIME_STR
failure_report_path = os.path.join(target_script_dir, "diagnostic_clip_analysis.txt")

# 確保批次輸出目錄存在
try:
    os.makedirs(target_script_dir, exist_ok=True)
except Exception as e:
    print(f"錯誤：無法建立或存取批次輸出目錄 {target_script_dir}。錯誤訊息: {e}")
    sys.exit(1)


# ** 腳本複製 (NEW: 提前到這裡執行) **
try:
    if len(sys.argv) > 0 and Path(sys.argv[0]).is_file():
        current_script_path = Path(sys.argv[0])
        target_script_path = target_script_dir / current_script_path.name
        
        shutil.copy2(current_script_path, target_script_path)
        print(f"** 腳本已複製到批次目錄: {target_script_path} **")
    else:
        print("警告: 無法找到執行中的腳本檔案 (dog_classifier_full.py) 來複製。請手動將腳本複製到批次目錄中以供記錄。")
except Exception as e:
    print(f"錯誤：複製腳本檔案失敗。錯誤: {e}")


# 初始化診斷報告檔案
try:
    with open(failure_report_path, 'w', encoding='utf-8') as f:
        f.write(f"--- CLIP 診斷性數據列表 (開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
        f.write("此報告用於追蹤 CLIP 驗證、毛色分類及動作分類的詳細分數。\n\n")
    print(f"已初始化診斷報告檔案: {failure_report_path}")
except Exception as e:
    print(f"錯誤：無法寫入診斷報告檔案 {failure_report_path}。錯誤: {e}")

# --- 主程式：開始處理檔案 ---

print(f"--- 開始處理目錄: {cfg.SOURCE_DIR} ---")

if not Path(cfg.SOURCE_DIR).exists():
    print(f"錯誤：來源目錄 {cfg.SOURCE_DIR} 不存在。請檢查您的網路路徑。")
    sys.exit(1)


for filepath in Path(cfg.SOURCE_DIR).rglob("*"):
    # 只處理圖片檔案
    if not filepath.is_file() or filepath.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    filename = filepath.name
    filepath_str = str(filepath)
    print(f"\n處理檔案: {filename}") 

    # YOLO 偵測
    try:
        results = model(filepath_str, verbose=False)
        dogs_detected = [box for box in results[0].boxes if int(box.cls) == 16]
        people_detected = [box for box in results[0].boxes if int(box.cls) == 0] 
    except Exception as e:
        print(f"  - YOLO 偵測失敗，跳過。錯誤: {e}")
        continue

    if cfg.SKIP_IF_HUMAN_FACE and people_detected:
        print(f"  - 偵測到 {len(people_detected)} 個人物 -> 由於 SKIP_IF_HUMAN_FACE 啟用，跳過此檔案。")
        continue

    if not dogs_detected and cfg.SKIP_NO_DOG:
        continue

    img = cv2.imread(filepath_str)
    if img is None:
        print("  - 無法讀取圖像，跳過。")
        continue
        
    # 取得照片時間 (用於檔名命名，但不再用於路徑分類)
    photo_dt = get_photo_datetime(filepath)
    
    file_classified = False

    if not dogs_detected:
        # 沒有狗，但 SKIP_NO_DOG=False，歸類到 unknown
        target_dir = os.path.join(target_script_dir, "unknown") # 使用提前定義的 target_script_dir
        os.makedirs(target_dir, exist_ok=True)
        dest = os.path.join(target_dir, filename)
        if not os.path.exists(dest) or cfg.OVERWRITE_EXISTING: shutil.copy2(filepath_str, dest)
        print("  - 未偵測到狗 -> unknown")
        continue

    for idx, box in enumerate(dogs_detected, start=1):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # --- 裁剪框收縮 (10% 排除邊緣雜訊) ---
        w_box, h_box = x2 - x1, y2 - y1
        shrink_w, shrink_h = int(w_box * cfg.CROP_SHRINK_RATIO), int(h_box * cfg.CROP_SHRINK_RATIO)
        
        x1_crop = max(0, x1 + shrink_w)
        y1_crop = max(0, y1 + shrink_h)
        x2_crop = min(img.shape[1], x2 - shrink_w)
        y2_crop = min(img.shape[0], y2 - shrink_h)
        
        if y1_crop >= y2_crop or x1_crop >= x2_crop:
            print(f"  - 偵測框 {idx} 縮小後無效，跳過。")
            continue

        dog_crop = img[y1_crop:y2_crop, x1_crop:x2_crop]
        
        # 0. CLIP 驗證
        if not verify_is_dog_with_clip(dog_crop):
            print(f"  - 偵測框 {idx} 經 CLIP 驗證為非狗動物，跳過此偵測結果。")
            continue 
        
        # 1. 動作分類 (使用 CLIP)
        action_label_full = classify_action(dog_crop, cfg, filename) if cfg.ENABLE_ACTION_CLASSIFY else ""
        action_label_short = action_label_full.replace('動作_', '') if action_label_full else ""
        
        # 2. 狗狗 ID 初判 (根據亮度)
        brightness = smart_brightness(dog_crop)
        dog_id_initial = cfg.DOG_ID_BRIGHT_NAME if brightness > cfg.BRIGHTNESS_THRESHOLD else cfg.DOG_ID_DARK_NAME
        
        # 3. 輔助特徵判斷 (毛色/環境)
        color_label, diag_data = classify_color_with_clip(dog_crop, cfg, filename) if cfg.ENABLE_COLOR_CLASSIFY and CLIP_INITIALIZED else ("", None)
        
        # 環境判斷 (如果關閉，返回空字串)
        environment = classify_environment(img, box, cfg)
        
        # 【診斷數據輸出】
        if (diag_data or action_label_full.startswith("動作_")) and CLIP_INITIALIZED:
            with open(failure_report_path, 'a', encoding='utf-8') as f:
                f.write(f"檔案: {filename} | 亮度(Top20): {brightness:.1f} | ID初判: {dog_id_initial}\n")
                f.write(f"CLIP 動作判斷: {action_label_full}\n")
                if diag_data:
                    f.write(f"CLIP 毛色判斷: {color_label} | 依據: {diag_data.get('判斷依據', 'N/A')}\n")
                    f.write(f"毛色詳細分數: {diag_data.get('CLIP_Scores', 'N/A')}\n")
                f.write(f"\n")

            print(f"  CLIP 動作判斷: {action_label_full}")
            

        # 4. 雙重判斷邏輯修正
        dog_id_final = dog_id_initial
        if cfg.ENABLE_COLOR_CLASSIFY and CLIP_INITIALIZED:
            if color_label == "CLIP深色米克斯": 
                dog_id_final = cfg.DOG_ID_DARK_NAME
                print(f"  [修正]: 初判 ID {dog_id_initial} 修正為 {dog_id_final} (因 CLIP 偵測到「深色」特徵)")
            elif color_label == "CLIP淺色米克斯":
                dog_id_final = cfg.DOG_ID_BRIGHT_NAME
                print(f"  [修正]: 初判 ID {dog_id_initial} 修正為 {dog_id_final} (因 CLIP 偵測到「淺色」特徵)")
            
        # 5. 檔案重新命名與分類移動
        if cfg.RENAME_BY_DATETIME:
            # 檔名仍然使用照片本身的日期時間
            photo_dt_str_full = photo_dt.strftime("%Y%m%d_%H%M%S")
            
            # 處理檔名: {照片時間}_{動作標籤}_{偵測框序號}.{副檔名}
            base_name = f"{photo_dt_str_full}_{action_label_short}_{idx}" if action_label_short else f"{photo_dt_str_full}_{idx}"
            new_filename = f"{base_name}{filepath.suffix}"
        else:
            new_filename = filename # 保持原檔名
        
        # --- 路徑建構邏輯 (已簡化) ---
        # 結構: {target_script_dir}/{DOG_ID}/{Filename}

        # 1. 狗狗 ID (二季/四季)
        # target_script_dir 已經包含了 {OUTPUT_DIR}/{RUNTIME_DATETIME_STR}
        target_dir = os.path.join(target_script_dir, dog_id_final)
        
        # 確保目標目錄存在
        os.makedirs(target_dir, exist_ok=True)
        
        dest = os.path.join(target_dir, new_filename)
        
        # 檢查是否已存在且不覆寫
        if os.path.exists(dest) and not cfg.OVERWRITE_EXISTING:
            print(f"  - 目標檔案已存在且不覆寫，跳過偵測框 {idx}。目標路徑: {dest}")
            continue

        # 複製檔案
        shutil.copy2(filepath_str, dest)
        print(f"  - 成功分類 {dog_id_final} (亮度: {brightness:.1f})，動作: {action_label_short} -> 複製到: {target_dir}")
        file_classified = True

        # 如果有多隻狗，只處理第一個（假設主要目標）
        break 

# --- 結束處理 ---
print(f"\n--- 處理完成 ---")
print(f"結果輸出至: {cfg.OUTPUT_DIR}")

# 診斷報告檔案結尾
try:
    with open(failure_report_path, 'a', encoding='utf-8') as f:
        f.write(f"\n--- CLIP 診斷性數據列表 (結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
except Exception:
    pass