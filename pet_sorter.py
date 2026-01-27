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

================================================================================
  Ollama AI 描述與影片處理說明
================================================================================
本程式已整合本地 Ollama AI 服務：
1. 若啟用 ENABLE_OLLAMA_CAPTION，將嘗試連線至 http://localhost:11434。
2. 使用視覺模型 (預設 moondream) 為狗狗照片生成英文描述，存為同名 .txt 檔。
3. 支援影片 (.mp4, .mov) 抽樣檢查，若影片中出現狗狗，自動歸類至「影片」資料夾。
"""

from pathlib import Path
import sys
import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime, date
from PIL import Image, ExifTags
import torch
import logging
import time
import json
import base64
import urllib.request
import urllib.error
import io

# 修正：導入 CLIP 模型的必要類別
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("警告：缺少 'transformers' 庫。請嘗試安裝：pip install transformers torch torchvision")
    CLIPProcessor = None
    CLIPModel = None

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ==============================================================================
# 設定區塊 (Config Block)
# ==============================================================================
class Config:
    # 檔案路徑設定
    SOURCE_DIR = Path(r"\\192.168.50.143\home\Photos\MobileBackup") 
    OUTPUT_DIR = Path(r"\\192.168.50.143\photo\照片-Pet_分類") 
    
    # --- 性能優化設定 ---
    MAX_AI_INPUT_SIZE = 1024 
    
    # --- 狗狗 ID 分類設定 ---
    BRIGHTNESS_THRESHOLD = 185 
    DOG_ID_BRIGHT_NAME = "二季" 
    DOG_ID_DARK_NAME = "四季"   
    
    # --- 年齡分類設定 ---
    DOG_BIRTH_DATE = datetime(2022, 11, 23) 
    
    # --- CLIP 驗證閾值 ---
    CLIP_VERIFY_MULTIPLIER = 1.2 
    CLIP_MIN_DOG_PROBABILITY = 0.6 
    
    # --- 功能開關 ---
    SKIP_NO_DOG = True             
    OVERWRITE_EXISTING = False     
    ENABLE_ACTION_CLASSIFY = True  
    ENABLE_DATE_CLASSIFY = True   
    RENAME_BY_DATETIME = True      
    ENABLE_COLOR_CLASSIFY = True   
    ENABLE_GEAR_CLASSIFY = False   
    ENABLE_ENV_CLASSIFY = False    
    SKIP_IF_HUMAN_FACE = False 
    ENABLE_DUPLICATE_CHECK = True 

    # --- Ollama AI 設定 (新功能) ---
    ENABLE_OLLAMA_CAPTION = True  # 是否啟用 AI 生成描述
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "moondream"    # 推薦使用 moondream (速度快) 或 llava (精度高)
    OLLAMA_PROMPT = "Describe this image in one brief sentence, focusing on the dog's action and emotion."
    
    # --- 影片處理設定 (新功能) ---
    ENABLE_VIDEO_PROCESS = True   # 是否啟用影片處理
    VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv']
    VIDEO_SAMPLE_INTERVAL = 2.0   # 每隔幾秒抽取一幀進行檢測 (數值越大處理越快，但可能漏掉)
    VIDEO_OUTPUT_FOLDER = "影片_狗狗" # 影片存放的根目錄名稱

    # --- 不確定性處理資料夾 ---
    UNCERTAINTY_FOLDER_NAME = "潛在誤判或修正"
    
    # --- 其他配置 ---
    CROP_SHRINK_RATIO = 0.10 
    ENV_OUTDOOR_PIXEL_RATIO = 0.1 

# 初始化全局變數
cfg = Config() 
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_INITIALIZED = False
device = "cpu"
clip_model = None
clip_processor = None
yolo_model = None
known_hashes = set()

# ==============================================================================
# 輔助函數 (Utility Functions)
# ==============================================================================

def initialize_models():
    """初始化 YOLO 和 CLIP 模型"""
    global CLIP_INITIALIZED, device, clip_model, clip_processor, yolo_model
    
    # --- 初始化 CLIP ---
    if CLIPModel is not None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"CLIP 運行設備: {device}")
            clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
            clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            CLIP_INITIALIZED = True
            logging.info("CLIP 模型載入成功。" )
        except Exception as e:
            logging.error(f"錯誤：無法載入 CLIP 模型。{e}")
            CLIP_INITIALIZED = False
    else:
        logging.warning("CLIP 類別未定義，跳過 CLIP 初始化。" )

    # --- 初始化 YOLO ---
    try:
        yolo_model = YOLO("yolov8n.pt")
        logging.info("YOLO 模型載入成功。" )
    except Exception as e:
        logging.error(f"錯誤：無法載入 YOLO 模型。{e}")

def check_ollama_status():
    """檢查 Ollama 服務是否可用"""
    if not cfg.ENABLE_OLLAMA_CAPTION:
        return False
    try:
        with urllib.request.urlopen("http://localhost:11434/") as response:
            if response.status == 200:
                logging.info(f"Ollama 服務連線成功 (Model: {cfg.OLLAMA_MODEL})。將啟用 AI 描述生成。" )
                return True
    except Exception:
        logging.warning("警告：無法連線至 Ollama (localhost:11434)。AI 描述生成功能將被暫時停用。" )
        return False
    return False

OLLAMA_AVAILABLE = False # 將在 main 中更新

def calculate_aHash(img_bgr):
    if img_bgr is None or img_bgr.size == 0: return None
    try:
        img_resized = cv2.resize(img_bgr, (8, 8), interpolation=cv2.INTER_AREA)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        avg = np.mean(img_gray)
        hash_value = sum([1 << i for i, pixel in enumerate(img_gray.flatten()) if pixel > avg])
        return f'{hash_value:016x}'
    except Exception: return None

def resize_for_ai(image_pil):
    w, h = image_pil.size
    max_size = cfg.MAX_AI_INPUT_SIZE
    if max(w, h) > max_size:
        if w > h:
            new_w = max_size
            new_h = int(h * (max_size / w))
        else:
            new_h = max_size
            new_w = int(w * (max_size / h))
        return image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image_pil

def classify_age(photo_dt, birth_dt):
    age_years = photo_dt.year - birth_dt.year - ((photo_dt.month, photo_dt.day) < (birth_dt.month, birth_dt.day))
    if age_years < 1: return "1_幼犬 (0-1歲)"       
    elif age_years < 3: return "2_少年期 (1-3歲)"     
    elif age_years < 7: return "3_青壯年 (3-7歲)"     
    elif age_years < 12: return "4_中年期 (7-12歲)"     
    else: return "5_老年 (12歲以上)"       

def smart_brightness(img_bgr):
    if img_bgr is None or img_bgr.size == 0: return 0
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        flat = gray.flatten()
        flat.sort()
        top20 = flat[int(len(flat) * 0.8):]
        return np.mean(top20) if top20.size > 0 else 0
    except Exception: return 0

def classify_action(image_pil_resized):
    if not CLIP_INITIALIZED or not cfg.ENABLE_ACTION_CLASSIFY: return "動作_無效"
    action_descriptions = [
        "A dog is standing and looking around.",
        "A dog is sitting with its front paws down.",
        "A dog is lying down, resting or sleeping.",
        "A dog is running, jumping, or moving quickly."
    ]
    inputs = clip_processor(text=action_descriptions, images=image_pil_resized, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).squeeze().cpu().numpy()
    max_index = np.argmax(probs)
    
    if max_index == 0: return "動作_站立"
    elif max_index == 1: return "動作_坐下"
    elif max_index == 2: return "動作_躺臥"
    elif max_index == 3: return "動作_活動"
    return "動作_中立" 

def verify_is_dog_with_clip(image_pil_resized):
    if not CLIP_INITIALIZED: return True, False
    descriptions = [
        "A photograph of a pet dog, including mixed-breeds.", 
        "A photograph of a bear.",
        "A photograph of a wild animal."
    ]
    inputs = clip_processor(text=descriptions, images=image_pil_resized, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad(): outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).squeeze().cpu().numpy()
    
    dog_prob = probs[0]
    max_non_dog_prob = max(probs[1:])
    
    if max_non_dog_prob > dog_prob * cfg.CLIP_VERIFY_MULTIPLIER: return False, False
    if dog_prob < cfg.CLIP_MIN_DOG_PROBABILITY: return True, True
    return True, False

def classify_color_with_clip(image_pil_resized):
    if not CLIP_INITIALIZED or not cfg.ENABLE_COLOR_CLASSIFY: return "未知顏色_未啟用CLIP", None
    descriptions = [
        f"A dog with predominantly dark, black, or deep brown fur, matching {cfg.DOG_ID_DARK_NAME}.", 
        f"A dog with predominantly light, white, cream, or golden fur, matching {cfg.DOG_ID_BRIGHT_NAME}."
    ]
    inputs = clip_processor(text=descriptions, images=image_pil_resized, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad(): outputs = clip_model(**inputs)
    probs = torch.softmax(outputs.logits_per_image.squeeze().cpu(), dim=0).numpy()
    
    return (cfg.DOG_ID_DARK_NAME if np.argmax(probs) == 0 else cfg.DOG_ID_BRIGHT_NAME), probs

def extract_datetime(file_path):
    """通用時間提取 (支援圖片與影片)"""
    # 1. 嘗試從檔名提取
    filename_parts = file_path.name.split('_')
    if len(filename_parts) >= 2:
        try:
            date_str = filename_parts[0]
            time_str = filename_parts[1]
            if len(date_str) == 8 and date_str.isdigit() and len(time_str) == 6 and time_str.isdigit():
                return datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
        except ValueError: pass
    
    # 2. 如果是圖片，嘗試讀取 EXIF
    if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
        try:
            with Image.open(file_path) as img:
                exif = img._getexif()
                if exif:
                    for tag, value in exif.items():
                        if ExifTags.TAGS.get(tag) == 'DateTimeOriginal':
                            return datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
        except Exception: pass

    # 3. 最後使用檔案修改時間
    return datetime.fromtimestamp(os.path.getmtime(file_path))

# ==============================================================================
# Ollama AI 功能 (New)
# ==============================================================================

def generate_ollama_caption(image_path, output_txt_path):
    """呼叫本地 Ollama API 為圖片生成描述"""
    if not OLLAMA_AVAILABLE: return

    try:
        # 1. 將圖片轉為 Base64
        with Image.open(image_path) as img:
            img = resize_for_ai(img) 
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 2. 建構 Request payload
        payload = {
            "model": cfg.OLLAMA_MODEL,
            "prompt": cfg.OLLAMA_PROMPT,
            "stream": False,
            "images": [img_base64]
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(cfg.OLLAMA_API_URL, data=data, headers={'Content-Type': 'application/json'})

        # 3. 發送請求
        logging.info(f"  [AI 描述]: 正在請求 Ollama ({cfg.OLLAMA_MODEL})...")
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            caption = result.get('response', '').strip()
            
            if caption:
                # 4. 寫入檔案
                with open(output_txt_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
                logging.info(f"  [AI 描述完成]: {caption[:50]}...")
            else:
                logging.warning("  [AI 描述失敗]: Ollama 回傳空內容。" )

    except Exception as e:
        logging.error(f"  [AI 描述錯誤]: {e}")

# ==============================================================================
# 影片處理邏輯 (New)
# ==============================================================================

def analyze_video_content(video_path):
    """
    對影片進行抽樣分析：
    1. 是否包含狗
    2. 如果有，主要動作是什麼 (根據樣本中出現頻率最高的動作)
    """
    if yolo_model is None: return False, "YOLO未載入"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.warning(f"無法開啟影片: {video_path}")
        return False, "無法開啟"

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    sample_interval_frames = int(fps * cfg.VIDEO_SAMPLE_INTERVAL)
    if sample_interval_frames <= 0: sample_interval_frames = 30

    dog_detected_count = 0
    frames_checked = 0
    actions_detected = []

    current_frame = 0
    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret: break

        frames_checked += 1
        
        results = yolo_model(frame, conf=0.3, iou=0.5, classes=[16], verbose=False) 
        
        has_dog = False
        for r in results:
            if len(r.boxes) > 0:
                has_dog = True
                break
        
        if has_dog:
            dog_detected_count += 1
            if cfg.ENABLE_ACTION_CLASSIFY and CLIP_INITIALIZED and len(actions_detected) < 3: 
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_resized = resize_for_ai(img_pil)
                action = classify_action(img_resized)
                actions_detected.append(action)

        current_frame += sample_interval_frames

    cap.release()

    if frames_checked == 0: return False, "空白影片"
    
    dog_ratio = dog_detected_count / frames_checked
    is_dog_video = dog_ratio > 0.15 or dog_detected_count >= 2 

    final_action = "動作_一般"
    if actions_detected:
        final_action = max(set(actions_detected), key=actions_detected.count)

    logging.info(f"  - 影片分析: 抽樣 {frames_checked} 幀, 狗出現 {dog_detected_count} 次 ({dog_ratio:.0%}) -> {'是' if is_dog_video else '否'}")
    
    return is_dog_video, final_action

def process_video_file(file_path, base_output_dir):
    """處理單一影片檔案"""
    logging.info(f"-> 開始處理影片: {file_path.name}")
    
    base_dt = extract_datetime(file_path)
    
    is_dog_video, action_label = analyze_video_content(file_path)
    
    if not is_dog_video and cfg.SKIP_NO_DOG:
        logging.info("  [跳過/INFO]: 影片中未偵測到足夠的狗狗畫面。" )
        return

    video_root = base_output_dir / cfg.VIDEO_OUTPUT_FOLDER
    target_dir = video_root / action_label
    
    try: target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e: logging.error(f"建立影片資料夾失敗: {e}"); return

    if cfg.RENAME_BY_DATETIME:
        timestamp_str = base_dt.strftime('%Y%m%d_%H%M%S')
        action_part = action_label.replace('動作_', '')
        new_filename = f"{timestamp_str}_{action_part}_{file_path.stem}{file_path.suffix}"
    else:
        new_filename = file_path.name

    target_path = target_dir / new_filename

    try:
        if target_path.exists() and not cfg.OVERWRITE_EXISTING:
            logging.info(f"  [跳過]: 影片已存在 {target_path.name}")
            return
        
        logging.info(f"  [複製影片]: -> {target_path.relative_to(base_output_dir)}")
        shutil.copy2(str(file_path), str(target_path))
        
    except Exception as e:
        logging.error(f"複製影片失敗: {e}")


# ==============================================================================
# 主要處理邏輯
# ==============================================================================

def get_dog_bbox(image_bgr):
    """(保持原有的 YOLO 偵測邏輯)"""
    if yolo_model is None: return None, None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = yolo_model(image_rgb, conf=0.25, iou=0.7, classes=[16, 0], verbose=False)
    dog_bbox, person_bbox = None, None
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if cls_id == 16: dog_bbox = (x1, y1, x2, y2)
            elif cls_id == 0 and cfg.SKIP_IF_HUMAN_FACE: person_bbox = (x1, y1, x2, y2)
    
    if cfg.SKIP_IF_HUMAN_FACE and person_bbox is not None: return None, None 
    return dog_bbox, person_bbox

def crop_image(image_bgr, dog_bbox):
    """(保持原有的裁切邏輯)"""
    if dog_bbox is None: return image_bgr, Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    h, w, _ = image_bgr.shape
    x1, y1, x2, y2 = dog_bbox
    dx, dy = int((x2 - x1) * cfg.CROP_SHRINK_RATIO), int((y2 - y1) * cfg.CROP_SHRINK_RATIO)
    crop_x1, crop_y1 = max(0, x1 - dx), max(0, y1 - dy)
    crop_x2, crop_y2 = min(w, x2 + dx), min(h, y2 + dy)
    cropped_bgr = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    return cropped_bgr, Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))

def construct_path_and_copy(file_path, base_dt, dog_id_final, age_stage, action_label, low_confidence_warning, id_correction_tag, base_output_dir):
    """(保持原有的路徑建構邏輯，新增回傳 target_path 以供 Ollama 使用)"""
    category_list = [dog_id_final, age_stage, action_label]
    category_list = [c for c in category_list if c and c not in ["動作_無效", "未知顏色_未啟用CLIP", "動作_未啟用"]]

    uncertainty_tags = []
    if id_correction_tag: uncertainty_tags.append(id_correction_tag.strip('[]'))
    if low_confidence_warning: uncertainty_tags.append("低信心")

    if uncertainty_tags:
        tags_str = "_".join(uncertainty_tags)
        target_dir = base_output_dir / Config.UNCERTAINTY_FOLDER_NAME / dog_id_final / age_stage / action_label
        new_filename_prefix = f"[{tags_str}]_"
    else:
        target_dir = base_output_dir / Path(*category_list)
        new_filename_prefix = ""

    try: target_dir.mkdir(parents=True, exist_ok=True)
    except Exception: return None

    if cfg.RENAME_BY_DATETIME:
        timestamp_str = base_dt.strftime('%Y%m%d_%H%M%S')
        action_part = action_label.replace('動作_', '') 
        new_filename = f"{new_filename_prefix}{timestamp_str}_{dog_id_final}_{action_part}_{file_path.stem}{file_path.suffix}"
    else:
        new_filename = f"{new_filename_prefix}{file_path.name}"

    target_path = target_dir / new_filename

    try:
        if target_path.exists() and not cfg.OVERWRITE_EXISTING:
            logging.info(f"  [跳過/INFO]: 目標檔案已存在: {target_path.name}")
            return target_path 
        
        shutil.copy2(str(file_path), str(target_path))
        logging.info(f"  [成功複製]: -> {target_path.relative_to(base_output_dir)}")
        return target_path
    except Exception as e:
        logging.error(f"複製檔案失敗: {e}")
        return None

def process_file(file_path, base_output_dir):
    """(主流程更新: 加入 Ollama 呼叫)"""
    logging.info(f"-> 開始處理圖片: {file_path.name}")
    base_dt = extract_datetime(file_path)
    
    try:
        img_bgr = cv2.imdecode(np.fromfile(str(file_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None: return
    except Exception: return

    # aHash 重複檢查
    a_hash = calculate_aHash(img_bgr)
    if cfg.ENABLE_DUPLICATE_CHECK and a_hash in known_hashes and not cfg.OVERWRITE_EXISTING:
        logging.info(f"  [跳過]: 圖片內容重複 (aHash: {a_hash})")
        return
    if cfg.ENABLE_DUPLICATE_CHECK and a_hash: known_hashes.add(a_hash)
    
    # YOLO & CLIP
    dog_bbox, person_bbox = get_dog_bbox(img_bgr)
    if dog_bbox is None and cfg.SKIP_NO_DOG:
        logging.info("  [跳過]: 未偵測到狗。" )
        return
    
    cropped_bgr, cropped_pil = crop_image(img_bgr, dog_bbox)
    image_pil_resized = resize_for_ai(cropped_pil)
    
    is_dog, low_confidence_warning = verify_is_dog_with_clip(image_pil_resized)
    if not is_dog:
        logging.info("  [跳過]: CLIP 判定為非狗。" )
        return

    # ID & 顏色
    brightness = smart_brightness(cropped_bgr)
    dog_id_initial = cfg.DOG_ID_BRIGHT_NAME if brightness >= cfg.BRIGHTNESS_THRESHOLD else cfg.DOG_ID_DARK_NAME
    
    dog_id_final = dog_id_initial
    id_correction_tag = ""
    if cfg.ENABLE_COLOR_CLASSIFY and CLIP_INITIALIZED:
        clip_color_id, _ = classify_color_with_clip(image_pil_resized)
        if dog_id_initial != clip_color_id:
            id_correction_tag = f"[{dog_id_initial}被{clip_color_id}修正]"
            dog_id_final = clip_color_id
            
    # 年齡 & 動作
    age_stage = classify_age(base_dt, cfg.DOG_BIRTH_DATE) if cfg.ENABLE_DATE_CLASSIFY else "未分類"
    action_label = classify_action(image_pil_resized) if (cfg.ENABLE_ACTION_CLASSIFY and CLIP_INITIALIZED) else "動作_未啟用"
    
    logging.info(f"  - 分類: ID={dog_id_final}, 年齡={age_stage}, 動作={action_label}")

    # 複製檔案
    final_path = construct_path_and_copy(
        file_path, base_dt, dog_id_final, age_stage, action_label, 
        low_confidence_warning, id_correction_tag, base_output_dir
    )

    # --- 新增：Ollama 生成描述 ---
    if final_path and OLLAMA_AVAILABLE and cfg.ENABLE_OLLAMA_CAPTION:
        txt_path = final_path.with_suffix('.txt')
        if not txt_path.exists() or cfg.OVERWRITE_EXISTING:
            generate_ollama_caption(file_path, txt_path)

def main():
    if not Config.SOURCE_DIR.exists():
        logging.error(f"來源資料夾不存在: {Config.SOURCE_DIR}")
        sys.exit(1)
        
    if not Config.OUTPUT_DIR.exists():
        try: Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e: logging.error(f"無法建立輸出資料夾: {e}"); sys.exit(1)
            
    initialize_models()
    
    global OLLAMA_AVAILABLE
    OLLAMA_AVAILABLE = check_ollama_status()

    all_files = []
    if Config.SOURCE_DIR.is_dir():
        for p in Config.SOURCE_DIR.rglob('*'):
            if not p.is_file(): continue
            ext = p.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.webp']:
                all_files.append((p, 'image'))
            elif cfg.ENABLE_VIDEO_PROCESS and ext in cfg.VIDEO_EXTENSIONS:
                all_files.append((p, 'video'))

    total_files = len(all_files)
    if total_files == 0:
        logging.info("未找到任何可處理的檔案。" )
        return

    logging.info(f"共找到 {total_files} 個檔案 (包含影片)。開始處理...")
    
    start_time = time.time()
    for i, (file_path, f_type) in enumerate(all_files):
        logging.info(f"[{i+1}/{total_files}] ({f_type}) {file_path.name}")
        try:
            if f_type == 'image':
                process_file(file_path, Config.OUTPUT_DIR)
            else:
                process_video_file(file_path, Config.OUTPUT_DIR)
        except Exception as e:
            logging.error(f"處理失敗: {e}")
            
    logging.info(f"處理完成！總用時: {time.time() - start_time:.2f} 秒。" )

if __name__ == '__main__':
    main()
