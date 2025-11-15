"""
================================================================================
  【潛在誤判或修正】資料夾 檔名標籤邏輯說明
================================================================================
當照片被複製到「潛在誤判或修正」資料夾時，其檔名會被加上以下前綴，說明需要手動檢查的原因：

| 觸發條件 (Condition) | 標籤前綴範例 (Example Tag) | 說明 (Explanation) |
| :--- | :--- | :--- |
| CLIP 低信心 (Low Confidence) | [低信心] | CLIP 模型對「狗」的機率分數低於設定閾值 (Config.CLIP_MIN_DOG_PROBABILITY, 預設 0.6)。 |
| ID 修正 (ID Correction) | [二季被四季修正] | 亮度判斷 (初判 ID) 與 CLIP 毛色分析 (最終 ID) 不一致，ID 被修正。例如：亮度判斷為二季，但 CLIP 判斷為深色（四季）。 |
| 兩者皆有 (Both) | [二季被四季修正_低信心] | 圖片同時符合 ID 修正和低信心警告。 |

---
"""
from pathlib import Path
import sys
import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from PIL import Image
import torch
import logging

# !!! 修正：導入 CLIP 模型的必要類別 !!!
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    # 這是為了在沒有安裝 transformers 庫時提供友善提示
    print("警告：缺少 'transformers' 庫。請嘗試安裝：pip install transformers torch torchvision")
    # 如果缺少，將這些類別設為 None，讓程式碼能夠運行但禁用 CLIP 功能
    CLIPProcessor = None
    CLIPModel = None


# 配置日誌：設定為 INFO 級別，以顯示關鍵資訊和操作細節 
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ==============================================================================
# 設定區塊 (Config Block)
# ==============================================================================
class Config:
    # 檔案路徑設定
    SOURCE_DIR = Path(r"\\192.168.50.143\home\Photos\MobileBackup") 
    OUTPUT_DIR = Path(r"D:/Pet")                                   
    
    # --- 狗狗 ID 分類設定 ---
    BRIGHTNESS_THRESHOLD = 185 
    DOG_ID_BRIGHT_NAME = "二季"
    DOG_ID_DARK_NAME = "四季"
    
    # --- CLIP 驗證閾值 ---
    CLIP_VERIFY_MULTIPLIER = 1.2 # 非狗機率 > 狗機率 * 1.2 視為誤判 (Failure: 直接跳過)
    CLIP_MIN_DOG_PROBABILITY = 0.6 # 狗的絕對機率低於此值，視為低信心警告 (Warning: 複製到潛在誤判資料夾)
    
    # 功能開關
    SKIP_NO_DOG = True             
    OVERWRITE_EXISTING = False     
    ENABLE_ACTION_CLASSIFY = True  
    ENABLE_DATE_CLASSIFY = False   
    RENAME_BY_DATETIME = True      
    ENABLE_COLOR_CLASSIFY = True   
    ENABLE_GEAR_CLASSIFY = False   
    ENABLE_ENV_CLASSIFY = False    
    SKIP_IF_HUMAN_FACE = False 
    
    # --- 新增：不確定性處理資料夾 (Misclassified/Uncertainty) ---
    UNCERTAINTY_FOLDER_NAME = "潛在誤判或修正"
    
    # --- 其他配置 ---
    CROP_SHRINK_RATIO = 0.10 
    ENV_OUTDOOR_PIXEL_RATIO = 0.1 
    GREEN_LOWER = np.array([40, 40, 40])
    BLUE_LOWER = np.array([100, 40, 40])
    GREEN_UPPER = np.array([80, 255, 255])
    BLUE_UPPER = np.array([140, 255, 255])

# 初始化配置
cfg = Config() 
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_INITIALIZED = False
device = "cpu"
clip_model = None
clip_processor = None
yolo_model = None

# ==============================================================================
# 輔助函數 (Utility Functions)
# ==============================================================================

def initialize_models():
    """初始化 YOLO 和 CLIP 模型"""
    global CLIP_INITIALIZED, device, clip_model, clip_processor, yolo_model
    
    # --- 初始化 CLIP 模型 ---
    if CLIPModel is not None:
        try:
            # 確保使用 CUDA 或 CPU 
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"CLIP 運行設備: {device}")
            clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
            clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            CLIP_INITIALIZED = True
            logging.info("CLIP 模型載入成功。")
        except Exception as e:
            logging.error(f"錯誤：無法載入 CLIP 模型。動作/毛色分類將被禁用。錯誤: {e}")
            CLIP_INITIALIZED = False
    else:
        logging.warning("CLIP 類別未定義，請檢查 'transformers' 庫是否已安裝。動作/毛色分類將被禁用。")


    # --- 初始化 YOLO 模型 ---
    try:
        yolo_model = YOLO("yolov8n.pt")
        logging.info("YOLO 模型載入成功。")
    except Exception as e:
        logging.error(f"錯誤：無法載入 YOLO 模型。錯誤: {e}")
        sys.exit(1)

def smart_brightness(img_bgr):
    """計算圖像最亮 20% 像素的平均亮度"""
    if img_bgr is None or img_bgr.size == 0: return 0
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        flat = gray.flatten()
        flat.sort()
        top20_index = int(len(flat) * 0.8)
        top20 = flat[top20_index:]
        return np.mean(top20) if top20.size > 0 else 0
    except Exception: return 0


def classify_action(image_pil):
    """使用 CLIP 模型判斷狗的動作"""
    if not CLIP_INITIALIZED: return "動作_無效"

    action_descriptions = [
        "A dog is standing and looking around.",
        "A dog is sitting with its front paws down.",
        "A dog is lying down, resting or sleeping.",
        "A dog is running, jumping, or moving quickly."
    ]

    inputs = clip_processor(text=action_descriptions, images=image_pil, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clip_model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1).squeeze().cpu().numpy()
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


def verify_is_dog_with_clip(image_pil):
    """
    使用 CLIP 模型驗證偵測到的物體是否真的是狗。
    回傳 (is_dog: bool, is_low_confidence_warning: bool)
    is_dog=False 導致跳過。
    is_low_confidence_warning=True 導致檔案被額外複製到不確定資料夾。
    """
    if not CLIP_INITIALIZED: return True, False

    verification_descriptions = [
        "A photograph of a pet dog (Labrador, Husky, Poodle, etc.).",
        "A photograph of a bear (Black bear, brown bear, panda bear, etc.).",
        "A photograph of a wild animal (fox, raccoon, squirrel, etc.)."
    ]

    inputs = clip_processor(text=verification_descriptions, images=image_pil, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clip_model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1).squeeze().cpu().numpy()

    dog_prob = probs[0]
    max_non_dog_prob = max(probs[1:])
    
    # 1. 檢查是否符合倍數誤判 (Failure: 判定為非狗，必須跳過)
    if max_non_dog_prob > dog_prob * cfg.CLIP_VERIFY_MULTIPLIER:
        non_dog_index = np.argmax(probs[1:]) + 1
        non_dog_label = verification_descriptions[non_dog_index]
        logging.warning(f"  [CLIP 驗證失敗]: 非狗機率過高 ({max_non_dog_prob:.2f})。判定為誤判。")
        return False, False
    
    # 2. 檢查是否符合絕對低機率誤判 (Warning/Uncertainty: 仍視為狗，但標記為低信心)
    if dog_prob < cfg.CLIP_MIN_DOG_PROBABILITY:
        logging.warning(f"  [CLIP 潛在誤判警告]: 狗機率過低 ({dog_prob:.2f})。將標記為低信心。")
        return True, True 
    
    # 3. 驗證通過 (Success: 正常處理)
    logging.debug(f"  [CLIP 驗證通過]: 狗機率 {dog_prob:.2f}, 非狗機率最高為 {max_non_dog_prob:.2f}。")
    return True, False


def classify_color_with_clip(image_pil, filepath_name):
    """使用 CLIP 模型判斷狗的顏色特徵"""
    if not CLIP_INITIALIZED:
        return "未知顏色", None

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

    probs = outputs.logits_per_image.softmax(dim=1).squeeze().cpu().numpy()

    dark_mix_prob = probs[0]
    light_mix_prob = probs[1]

    color_label = "CLIP多色/通用"
    
    # 判斷依據：誰比誰高 10%
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


def get_photo_datetime(image_pil, filepath):
    """從 PIL Image 讀取 EXIF 拍照日期"""
    try:
        exif = image_pil.getexif()
        dt_str = exif.get(36867) 
        if dt_str:
            return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    
    try:
        # 如果 EXIF 失敗，退回到檔案修改時間 (使用 Path.stat())
        t = filepath.stat().st_mtime
        return datetime.fromtimestamp(t)
    except Exception:
        return datetime.now()
        

# ==============================================================================
# 核心處理函數 (Main Processing Function)
# ==============================================================================

def process_file(filepath, cfg, target_script_dir, failure_report_path):
    """處理單個圖片檔案的邏輯"""
    filename = filepath.name
    
    # --- 立即顯示正在處理的檔案名稱 ---
    logging.info(f"-> 開始處理: {filename}")
    
    is_low_confidence = False # CLIP 低信心警告旗標
    is_id_corrected = False   # ID 修正旗標
    dog_id_initial = ""       # 狗狗 ID 初判結果
    dog_id_final = ""         # 狗狗 ID 最終結果
    
    # 1. 載入圖像 (僅載入一次)
    try:
        image_pil = Image.open(filepath)
        img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"  - 無法讀取或轉換圖像，跳過 {filename}。錯誤: {e}")
        return
        
    photo_dt = get_photo_datetime(image_pil, filepath)

    # 2. YOLO 偵測
    try:
        results = yolo_model(str(filepath), verbose=False)
        # Class 16 is 'dog', Class 0 is 'person'
        dogs_detected = [box for box in results[0].boxes if int(box.cls) == 16]
        people_detected = [box for box in results[0].boxes if int(box.cls) == 0] 
    except Exception as e:
        logging.error(f"  - YOLO 偵測失敗，跳過 {filename}。錯誤: {e}")
        return

    if cfg.SKIP_IF_HUMAN_FACE and people_detected:
        logging.info(f"  - 偵測到 {len(people_detected)} 個人物 -> 跳過 {filename}。")
        return

    if not dogs_detected:
        if cfg.SKIP_NO_DOG:
            return
        # ... 處理無狗邏輯 ...
        return

    for idx, box in enumerate(dogs_detected, start=1):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # --- 裁剪框收縮 ---
        w_box, h_box = x2 - x1, y2 - y1
        shrink_w, shrink_h = int(w_box * cfg.CROP_SHRINK_RATIO), int(h_box * cfg.CROP_SHRINK_RATIO)
        
        x1_crop = max(0, x1 + shrink_w)
        y1_crop = max(0, y1 + shrink_h)
        x2_crop = min(img_bgr.shape[1], x2 - shrink_w)
        y2_crop = min(img_bgr.shape[0], y2 - shrink_h)
        
        if y1_crop >= y2_crop or x1_crop >= x2_crop:
            logging.warning(f"  - 偵測框 {idx} 縮小後無效，跳過。")
            continue

        dog_crop_bgr = img_bgr[y1_crop:y2_crop, x1_crop:x2_crop]
        
        # 3. 準備 CLIP 輸入 (RGB PIL Image)
        try:
            dog_crop_rgb = cv2.cvtColor(dog_crop_bgr, cv2.COLOR_BGR2RGB)
            dog_crop_pil = Image.fromarray(dog_crop_rgb)
        except Exception:
            logging.error(f"  - 偵測框 {idx} 圖像轉換失敗，跳過 CLIP 分析。")
            continue
        
        # 4. CLIP 驗證 (誤判過濾 + 低信心標記)
        is_dog_verified, is_low_confidence = verify_is_dog_with_clip(dog_crop_pil)
        if not is_dog_verified:
            # 這是硬性誤判 (非狗機率太高)，直接跳過不複製
            continue 
        
        # 5. 動作分類
        action_label_full = classify_action(dog_crop_pil) if cfg.ENABLE_ACTION_CLASSIFY else ""
        action_label_short = action_label_full.replace('動作_', '') if action_label_full else ""
        logging.info(f"  - 偵測框 {idx} | CLIP 動作判斷: {action_label_full}")
        
        # 6. 狗狗 ID 初判 (根據亮度)
        brightness = smart_brightness(dog_crop_bgr) # 使用 BGR 格式的裁剪圖
        dog_id_initial = cfg.DOG_ID_BRIGHT_NAME if brightness > cfg.BRIGHTNESS_THRESHOLD else cfg.DOG_ID_DARK_NAME
        dog_id_final = dog_id_initial # 初始設定最終 ID
        
        # 7. 輔助特徵判斷 (毛色/環境)
        color_label, diag_data = classify_color_with_clip(dog_crop_pil, filename) if cfg.ENABLE_COLOR_CLASSIFY and CLIP_INITIALIZED else ("", None)
        environment = "" 
        
        # 【診斷數據輸出】: 寫入報告
        if CLIP_INITIALIZED:
            with open(failure_report_path, 'a', encoding='utf-8') as f:
                f.write(f"檔案: {filename} | 偵測框: {idx} | 亮度(Top20): {brightness:.1f} | ID初判: {dog_id_initial}\n")
                f.write(f"CLIP 動作判斷: {action_label_full}\n")
                if diag_data:
                    f.write(f"CLIP 毛色判斷: {color_label} | 依據: {diag_data.get('判斷依據', 'N/A')}\n")
                    f.write(f"毛色詳細分數: {diag_data.get('CLIP_Scores', 'N/A')}\n")
                f.write(f"\n")
            
        # 8. 雙重判斷邏輯修正 (毛色修正 ID)
        if cfg.ENABLE_COLOR_CLASSIFY and CLIP_INITIALIZED:
            # 判斷是否從「淺色」修正為「深色」
            if color_label == "CLIP深色米克斯" and dog_id_initial != cfg.DOG_ID_DARK_NAME:
                dog_id_final = cfg.DOG_ID_DARK_NAME
                is_id_corrected = True
                logging.info(f"  [修正]: ID {dog_id_initial} 修正為 {dog_id_final} (CLIP深色)")
            # 判斷是否從「深色」修正為「淺色」
            elif color_label == "CLIP淺色米克斯" and dog_id_initial != cfg.DOG_ID_BRIGHT_NAME:
                dog_id_final = cfg.DOG_ID_BRIGHT_NAME
                is_id_corrected = True
                logging.info(f"  [修正]: ID {dog_id_initial} 修正為 {cfg.DOG_ID_BRIGHT_NAME} (CLIP淺色)")
            
        # 9. 檔案重新命名與移動 (主分類)
        if cfg.RENAME_BY_DATETIME:
            photo_dt_str_full = photo_dt.strftime("%Y%m%d_%H%M%S")
            # 確保動作標籤在檔名中不為空
            action_part = f"_{action_label_short}" if action_label_short and action_label_short != "無效" else ""
            base_name = f"{photo_dt_str_full}{action_part}_{idx}"
            new_filename = f"{base_name}{filepath.suffix}"
        else:
            new_filename = filename
        
        # --- 主要路徑建構：存入 RUN_DIR/ID_NAME ---
        target_dir = target_script_dir / dog_id_final
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / new_filename
        
        if dest.exists() and not cfg.OVERWRITE_EXISTING:
            logging.warning(f"  - 目標檔案已存在且不覆寫，跳過偵測框 {idx}。目標路徑: {dest}")
            continue

        shutil.copy2(filepath, dest)
        logging.info(f"  - 成功分類 {dog_id_final} (亮度: {brightness:.1f})，動作: {action_label_short} -> 複製到: {target_dir.name}/{new_filename}")
        
        # 10. 處理潛在誤判/修正 (額外複製到專門資料夾，並加上標籤前綴)
        if is_low_confidence or is_id_corrected:
            
            # 10a. 決定標籤 (TAG)
            uncertainty_tag = ""
            if is_low_confidence and is_id_corrected:
                # 兩者都是: 原始ID被最終ID修正 + 低信心
                uncertainty_tag = f"{dog_id_initial}被{dog_id_final}修正_低信心"
            elif is_id_corrected:
                # 只有 ID 修正: 原始ID被最終ID修正
                uncertainty_tag = f"{dog_id_initial}被{dog_id_final}修正"
            elif is_low_confidence:
                # 只有低信心
                uncertainty_tag = "低信心"
                
            # 將標籤放在檔名前，用中括號和底線隔開
            # 最終檔名範例: [二季被四季修正]_20240101_120000_躺臥_1.jpg
            uncertain_new_filename = f"[{uncertainty_tag}]_{new_filename}"
            
            # 路徑：RUN_DIR/UNCERTAINTY_FOLDER_NAME
            uncertain_target_dir = target_script_dir / cfg.UNCERTAINTY_FOLDER_NAME
            uncertain_target_dir.mkdir(parents=True, exist_ok=True)
            uncertain_dest = uncertain_target_dir / uncertain_new_filename
            
            # 如果主要檔案已存在且不覆寫，則不複製額外檔案，避免重複檢查
            if uncertain_dest.exists() and not cfg.OVERWRITE_EXISTING:
                 logging.debug(f"  [不確定性複製]: 目標已存在，跳過。")
            else:
                shutil.copy2(filepath, uncertain_dest)
                logging.warning(f"  [額外複製/不確定性]: 由於 {uncertainty_tag} -> 同時複製到 {cfg.UNCERTAINTY_FOLDER_NAME}/{uncertain_new_filename}")

        # 只處理第一個偵測到的狗
        break 


def main():
    """主程序入口點"""
    runtime_dt = datetime.now()
    RUNTIME_DATETIME_STR = runtime_dt.strftime("%Y%m%d_%H%M%S")
    
    # 預備階段：設定執行批次資料夾 (確保隔離性)
    target_script_dir = cfg.OUTPUT_DIR / f"PetSorter_Run_{RUNTIME_DATETIME_STR}" 
    target_script_dir.mkdir(parents=True, exist_ok=True)
    
    # 建立診斷報告檔案
    failure_report_path = target_script_dir / f"classification_diagnostics_report.txt"
    logging.info(f"分類結果將儲存至獨立資料夾: {target_script_dir}")
    
    # 1. 初始化模型
    initialize_models()

    # 2. 核心處理邏輯 (直接迭代，不預先列舉)
    logging.info(f"--- 開始處理網路路徑 '{cfg.SOURCE_DIR}' 下的圖片檔案 (串流模式)。 ---")
    
    processed_count = 0
    
    # 直接迭代 rglob 的結果，每找到一個檔案就立即處理
    try:
        for filepath in cfg.SOURCE_DIR.rglob("*"):
            if filepath.is_file() and filepath.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                process_file(filepath, cfg, target_script_dir, failure_report_path)
                processed_count += 1
    except Exception as e:
        logging.error(f"FATAL: 在串流處理過程中發生致命錯誤。請檢查網路權限。錯誤: {e}")
        
    logging.info(f"--- 處理完成。總共嘗試處理了 {processed_count} 個符合條件的檔案。 ---")


if __name__ == "__main__":
    main()