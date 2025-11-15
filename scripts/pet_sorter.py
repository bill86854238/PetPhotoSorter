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

# 為了展示程式碼完整性，這裡使用一個 MockConfig 模擬 cfg 檔案的內容。
class MockConfig:
    # 檔案路徑設定 (根據您的專案路徑設定)
    SOURCE_DIR = r"\\192.168.50.143\home\Photos\MobileBackup"
    OUTPUT_DIR = r"D:/Pet"
    
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
    ENABLE_DATE_CLASSIFY = False   
    RENAME_BY_DATETIME = True      # 檔名會加上動作標籤
    ENABLE_COLOR_CLASSIFY = True   # 毛色分類已啟用 (透過 CLIP 實現，用於輔助狗狗 ID 判斷)
    ENABLE_GEAR_CLASSIFY = False   
    
    # 【手動開關】: 若設為 False，則輸出路徑中不會有「戶外/室內」資料夾。
    ENABLE_ENV_CLASSIFY = False     # <-- 已設定為 False
    
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
    """
    【新版】使用 CLIP 模型判斷狗的動作 (站立, 坐下, 躺臥, 活動)。
    """
    if not CLIP_INITIALIZED or dog_crop is None or dog_crop.size == 0:
        return "動作_無效"

    try:
        dog_crop_rgb = cv2.cvtColor(dog_crop, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(dog_crop_rgb)
    except Exception:
        return "動作_圖像轉換失敗"

    # 定義動作描述 (用於 CLIP 語義判斷)
    action_descriptions = [
        "A dog is standing and looking around.",         # 站立 (Standing)
        "A dog is sitting with its front paws down.",   # 坐下 (Sitting)
        "A dog is lying down, resting or sleeping.",    # 躺臥 (Lying Down/Sleeping)
        "A dog is running, jumping, or moving quickly." # 活動 (Active/Moving)
    ]

    inputs = clip_processor(text=action_descriptions, images=image_pil, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clip_model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).squeeze().cpu().numpy()
    
    # 選擇最高機率的標籤索引
    max_index = np.argmax(probs)
    action_label = action_descriptions[max_index]

    # 根據最高機率的描述詞返回中文標籤
    if "standing" in action_label.lower():
        return "動作_站立"
    elif "sitting" in action_label.lower():
        return "動作_坐下"
    # 包含 resting (休息) 或 sleeping (睡覺) 的都歸類為躺臥
    elif "lying down" in action_label.lower() or "resting" in action_label.lower() or "sleeping" in action_label.lower():
        return "動作_躺臥"
    elif "running" in action_label.lower() or "moving" in action_label.lower():
        return "動作_活動"
    
    return "動作_中立" # 萬一都不是，歸類為中立


def verify_is_dog_with_clip(dog_crop):
    """
    使用 CLIP 模型驗證偵測到的物體是否真的是狗 (排除熊、小熊貓等誤判)。
    """
    if not CLIP_INITIALIZED: return True

    if dog_crop is None or dog_crop.size == 0: return True

    try:
        dog_crop_rgb = cv2.cvtColor(dog_crop, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(dog_crop_rgb)
    except Exception: return True

    # 驗證描述：狗 vs 幾種常見的誤判物種
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
    
    # 找到所有非狗類別中的最高機率
    max_non_dog_prob = max(probs[1:])
    
    # 判斷邏輯：如果非狗的最高機率比狗的機率高出 20% (因子 1.2)
    if max_non_dog_prob > dog_prob * 1.2:
        non_dog_index = np.argmax(probs[1:]) + 1
        non_dog_label = verification_descriptions[non_dog_index]
        print(f"  [CLIP 驗證警告]: 狗機率 {dog_prob:.2f}。最高非狗機率 {max_non_dog_prob:.2f} ({non_dog_label[:20]}...)。判定為誤判。")
        return False
    
    print(f"  [CLIP 驗證通過]: 狗機率 {dog_prob:.2f}, 非狗機率最高為 {max_non_dog_prob:.2f}。")
    return True


def classify_color_with_clip(dog_crop, cfg, filepath_name):
    """
    使用 CLIP 模型判斷狗的顏色特徵 (深色 vs 淺色)。
    返回的標籤用於輔助判斷狗的 ID。
    """
    if not CLIP_INITIALIZED or dog_crop is None or dog_crop.size == 0:
        return "未知顏色", None

    try:
        dog_crop_rgb = cv2.cvtColor(dog_crop, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(dog_crop_rgb)
    except Exception as e:
        return "圖像轉換失敗", {"file": filepath_name, "錯誤": f"圖像轉換失敗: {e}"}

    text_descriptions = [
        "A dog with predominantly dark, black, or deep brown fur.",      
        "A dog with predominantly light, yellow, tan, or white fur.",   
        "A multi-colored dog (brindle, black and tan, etc.)",           
        "A generic dog photograph"                                      
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
    if img is None or img.size == 0: return "環境判斷錯誤"
        
    try:
        h, w, _ = img.shape
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # 建立一個只包含背景的遮罩 (排除狗的偵測框)
        mask = np.ones((h, w), dtype=np.uint8) * 255
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
        
        background_pixels = np.sum(mask == 255)
        if background_pixels == 0: return "室內/特寫"

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 偵測綠色 (植被) 和藍色 (天空) 像素
        green_mask = cv2.inRange(hsv, cfg.GREEN_LOWER, cfg.GREEN_UPPER)
        blue_mask = cv2.inRange(hsv, cfg.BLUE_LOWER, cfg.BLUE_UPPER)
        
        outdoor_mask = cv2.bitwise_or(green_mask, blue_mask)
        # 只計算背景區域的綠色/藍色像素
        outdoor_mask = cv2.bitwise_and(outdoor_mask, outdoor_mask, mask=mask)
        
        outdoor_pixels = np.sum(outdoor_mask > 0)
        outdoor_ratio = outdoor_pixels / background_pixels
        
        if outdoor_ratio > cfg.ENV_OUTDOOR_PIXEL_RATIO:
            return "戶外"
        else:
            return "室內"
    except Exception: return "環境判斷錯誤"

# --- 預備階段：初始化診斷檔案 ---
failure_report_path = os.path.join(cfg.OUTPUT_DIR, "diagnostic_clip_analysis.txt")
try:
    with open(failure_report_path, 'w', encoding='utf-8') as f:
        f.write(f"--- CLIP 診斷性數據列表 (開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
        f.write("此報告用於追蹤 CLIP 驗證、毛色分類及動作分類的詳細分數。\n\n")
    print(f"已初始化診斷報告檔案: {failure_report_path}")
except Exception as e:
    print(f"錯誤：無法寫入診斷報告檔案 {failure_report_path}。錯誤: {e}")

# --- 主程式 ---
print(f"--- 開始處理目錄: {cfg.SOURCE_DIR} ---")

if not Path(cfg.SOURCE_DIR).exists():
    print(f"錯誤：來源目錄 {cfg.SOURCE_DIR} 不存在。請檢查您的網路路徑。")
    sys.exit(1)


for filepath in Path(cfg.SOURCE_DIR).rglob("*"):
    if not filepath.is_file() or filepath.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    filename = filepath.name
    filepath_str = str(filepath)
    print(f"\n處理檔案: {filename}") 

    # YOLO 偵測 (狗 Class ID 16, 人 Class ID 0)
    try:
        results = model(filepath_str, verbose=False)
        dogs_detected = [box for box in results[0].boxes if int(box.cls) == 16]
        people_detected = [box for box in results[0].boxes if int(box.cls) == 0] 
    except Exception as e:
        print(f"  - YOLO 偵測失敗，跳過。錯誤: {e}")
        continue

    if cfg.SKIP_IF_HUMAN_FACE and people_detected:
        print(f"  - 偵測到 {len(people_detected)} 個人物 -> 由於 SKIP_IF_HUMAN_FACE 啟用，跳過此檔案。")
        continue

    if not dogs_detected and cfg.SKIP_NO_DOG:
        continue

    img = cv2.imread(filepath_str)
    if img is None:
        print("  - 無法讀取圖像，跳過。")
        continue
        
    photo_dt = get_photo_datetime(filepath)
    
    file_classified = False

    if not dogs_detected:
        # 沒有狗，但 SKIP_NO_DOG=False，歸類到 unknown
        target_dir = os.path.join(cfg.OUTPUT_DIR, "unknown")
        os.makedirs(target_dir, exist_ok=True)
        dest = os.path.join(target_dir, filename)
        if not os.path.exists(dest) or cfg.OVERWRITE_EXISTING: shutil.copy2(filepath_str, dest)
        print("  - 未偵測到狗 -> unknown")
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
            print(f"  - 偵測框 {idx} 縮小後無效，跳過。")
            continue

        dog_crop = img[y1_crop:y2_crop, x1_crop:x2_crop]
        
        # 0. CLIP 驗證：排除熊、小熊貓等誤判
        if not verify_is_dog_with_clip(dog_crop):
            print(f"  - 偵測框 {idx} 經 CLIP 驗證為非狗動物，跳過此偵測結果。")
            continue 
        
        # 1. 動作分類 (使用 CLIP)
        action_label_full = classify_action(dog_crop, cfg, filename) if cfg.ENABLE_ACTION_CLASSIFY else ""
        action_label_short = action_label_full.replace('動作_', '') if action_label_full else ""
        
        # 2. 狗狗 ID 初判 (根據亮度)
        brightness = smart_brightness(dog_crop)
        # 亮度高或低對應到不同的狗狗 ID
        dog_id_initial = cfg.DOG_ID_BRIGHT_NAME if brightness > cfg.BRIGHTNESS_THRESHOLD else cfg.DOG_ID_DARK_NAME
        
        # 3. 輔助特徵判斷 (毛色/環境)
        color_label, diag_data = classify_color_with_clip(dog_crop, cfg, filename) if cfg.ENABLE_COLOR_CLASSIFY and CLIP_INITIALIZED else ("", None)
        
        # 環境判斷 (只有在 ENABLE_ENV_CLASSIFY 為 True 時才執行)
        environment = classify_environment(img, box, cfg) if cfg.ENABLE_ENV_CLASSIFY else ""
        
        # 【診斷數據輸出】
        if (diag_data or action_label_full.startswith("動作_")) and CLIP_INITIALIZED:
            with open(failure_report_path, 'a', encoding='utf-8') as f:
                f.write(f"檔案: {filename} | 亮度(Top20): {brightness:.1f} | ID初判: {dog_id_initial}\n")
                f.write(f"CLIP 動作判斷: {action_label_full}\n")
                if diag_data:
                    f.write(f"CLIP 毛色判斷: {color_label} | 依據: {diag_data.get('判斷依據', 'N/A')}\n")
                    f.write(f"毛色詳細分數: {diag_data.get('CLIP_Scores', 'N/A')}\n")
                f.write(f"\n")

            print(f"  CLIP 動作判斷: {action_label_full}")
            

        # 4. 雙重判斷邏輯修正 - 使用 CLIP 毛色標籤修正狗狗 ID 判斷
        dog_id_final = dog_id_initial
        if cfg.ENABLE_COLOR_CLASSIFY and CLIP_INITIALIZED:
            if color_label == "CLIP深色米克斯": 
                dog_id_final = cfg.DOG_ID_DARK_NAME
                print(f"  [修正]: 初判 ID {dog_id_initial} 修正為 {dog_id_final} (因 CLIP 偵測到「深色」特徵)")
            elif color_label == "CLIP淺色米克斯":
                dog_id_final = cfg.DOG_ID_BRIGHT_NAME
                print(f"  [修正]: 初判 ID {dog_id_initial} 修正為 {dog_id_final} (因 CLIP 偵測到「淺色」特徵)")
            
        # 5. 檔案重新命名與分類移動
        if cfg.RENAME_BY_DATETIME:
            dt_str = photo_dt.strftime("%Y%m%d_%H%M%S")
            # 確保檔名中不包含重複的動作標籤
            clean_filename = f"{dt_str}_{idx}{filepath.suffix.lower()}"
            if action_label_short:
                new_filename = f"{dt_str}_{action_label_short}_{idx}{filepath.suffix.lower()}"
            else:
                new_filename = clean_filename
        else:
            new_filename = filename # 保持原檔名
        
        
        # 決定目標目錄 (根據狗狗 ID 及環境設定)
        target_dir = Path(cfg.OUTPUT_DIR) / dog_id_final
        
        if cfg.ENABLE_ENV_CLASSIFY: # 由於這裡現在是 False，將不會執行
            target_dir = target_dir / environment 
            print(f"  環境分類已啟用: {environment}")
        else:
            print("  環境分類已禁用。") # 現在會打印這句話
            
        target_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = target_dir / new_filename
        
        # 6. 複製檔案
        if not dest_path.exists() or cfg.OVERWRITE_EXISTING:
            shutil.copy2(filepath_str, dest_path)
            print(f"  -> 成功分類至: {target_dir.relative_to(cfg.OUTPUT_DIR)} / {new_filename}")
            file_classified = True
        else:
            print(f"  -> 目的地檔案已存在且 OVERWRITE_EXISTING=False，跳過複製。")
            file_classified = True

    if not file_classified and dogs_detected:
        print("  - 檔案處理完成，但所有偵測結果皆因誤判過濾或無效框而跳過。")
    elif not file_classified and not dogs_detected and not cfg.SKIP_NO_DOG:
        # 處理未偵測到狗但仍需分類的情況 (已在主迴圈開頭處理，此處理論上不會觸發)
        pass

print(f"\n--- 處理完成。結果儲存於 {cfg.OUTPUT_DIR} ---")
print(f"--- 詳細診斷數據請參閱 {failure_report_path} ---")