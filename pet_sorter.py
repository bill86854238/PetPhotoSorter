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
import platform
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
import queue
from concurrent.futures import ThreadPoolExecutor

# 導入自訂模組
import mac_utils

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
    # --- 載入外部設定 (config.json) ---
    CONFIG_PATH = Path("config.json")
    _config_data = {}
    
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                _config_data = json.load(f)
        except Exception as e:
            logging.warning(f"載入 config.json 失敗: {e}，將使用預設值。")
            
    _nas_ip = _config_data.get("nas_ip", "192.168.1.100")
    
    # 檔案路徑設定
    # 自動判斷作業系統
    if platform.system() == "Darwin": # macOS
        # macOS 透過 Finder (Cmd+K) 掛載 SMB 時，預設路徑位於 /Volumes/
        _mac_cfg = _config_data.get("mac", {})
        SOURCE_DIR = Path(_mac_cfg.get("source_dir", "/Volumes/home/Photos/MobileBackup"))
        OUTPUT_DIR = Path(_mac_cfg.get("output_dir", "/Volumes/photo/照片-Pet_分類"))
    else: # Windows (Default)
        _win_cfg = _config_data.get("windows", {})
        _src_tmpl = _win_cfg.get("source_dir", r"\\{ip}\home\Photos\MobileBackup")
        _out_tmpl = _win_cfg.get("output_dir", r"\\{ip}\photo\照片-Pet_分類")
        
        # 替換 IP 變數
        SOURCE_DIR = Path(_src_tmpl.replace("{ip}", _nas_ip))
        OUTPUT_DIR = Path(_out_tmpl.replace("{ip}", _nas_ip))
    
    # --- 讀取其他設定 (可選) ---
    _settings = _config_data.get("settings", {})
    
    # --- 性能優化設定 ---
    MAX_AI_INPUT_SIZE = 1024 
    BATCH_SIZE = _settings.get("batch_size", 16)
    DEVICE = _settings.get("device", "mps" if platform.system() == "Darwin" else "cpu")
    
    # --- 美感評分設定 ---
    AESTHETIC_SCORE_MIN = _settings.get("aesthetic_score_min", 0.2)
    AESTHETIC_SCORE_HIGH = _settings.get("aesthetic_score_high", 0.6)
    FINDER_TAG_COLORS = _settings.get("finder_tag_colors", {})
    
    # --- 狗狗 ID 分類設定 ---
    BRIGHTNESS_THRESHOLD = _settings.get("brightness_threshold", 185) 
    DOG_ID_BRIGHT_NAME = "二季" 
    DOG_ID_DARK_NAME = "四季"   
    
    # --- 年齡分類設定 ---
    DOG_BIRTH_DATE = datetime(2022, 11, 23) 
    
    # --- CLIP 驗證閾值 ---
    CLIP_VERIFY_MULTIPLIER = 1.2 
    CLIP_MIN_DOG_PROBABILITY = 0.6 
    
    # --- 功能開關 ---
    SKIP_NO_DOG = True             
    OVERWRITE_EXISTING = _settings.get("overwrite_existing", False)     
    ENABLE_ACTION_CLASSIFY = True  
    ENABLE_DATE_CLASSIFY = True   
    RENAME_BY_DATETIME = _settings.get("rename_by_datetime", True)
    ENABLE_COLOR_CLASSIFY = True   
    ENABLE_GEAR_CLASSIFY = False   
    ENABLE_ENV_CLASSIFY = False    
    SKIP_IF_HUMAN_FACE = False 
    ENABLE_DUPLICATE_CHECK = True 

    # --- Ollama AI 設定 (新功能) ---
    ENABLE_OLLAMA_CAPTION = _settings.get("enable_ollama", True)  # 是否啟用 AI 生成描述
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = _settings.get("ollama_model", "moondream")    # 推薦使用 moondream (速度快) 或 llava (精度高)
    OLLAMA_PROMPT = _settings.get("ollama_prompt", "請用繁體中文簡短描述這張圖片，重點在狗狗的動作與情緒。")
    
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
    """初始化 YOLO 和 CLIP 模型 (M4 優化版)"""
    global CLIP_INITIALIZED, device, clip_model, clip_processor, yolo_model
    
    # 設定裝置 (優先使用 Config 指定的，例如 mps)
    device = cfg.DEVICE
    if device == "mps" and not torch.backends.mps.is_available():
        logging.warning("⚠️ 無法偵測到 MPS 加速，將退回 CPU 模式。")
        device = "cpu"
    
    logging.info(f"🚀 硬體加速裝置: {device.upper()}")

    # --- 初始化 CLIP ---
    if CLIPModel is not None:
        try:
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
        # 將 YOLO 移動到指定裝置 (MPS)
        yolo_model.to(device)
        logging.info(f"YOLO 模型載入成功 (Device: {yolo_model.device})。" )
    except Exception as e:
        logging.error(f"錯誤：無法載入 YOLO 模型。{e}")

def check_ollama_status():
    """檢查 Ollama 服務是否可用，並驗證模型是否存在"""
    if not cfg.ENABLE_OLLAMA_CAPTION:
        return False
        
    try:
        # 1. 檢查基本連線
        with urllib.request.urlopen("http://localhost:11434/") as response:
            if response.status != 200:
                raise ConnectionError("Ollama 服務未回應")
                
        # 2. 檢查指定模型是否存在
        model_name = cfg.OLLAMA_MODEL
        req = urllib.request.Request(
            "http://localhost:11434/api/show", 
            data=json.dumps({"name": model_name}).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        try:
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    logging.info(f"Ollama 服務連線成功且模型 '{model_name}' 已就緒。")
                    return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logging.warning(f"Ollama 連線成功，但找不到模型 '{model_name}'。請先執行 'ollama pull {model_name}'")
            else:
                logging.warning(f"Ollama 模型檢查失敗: {e}")

    except Exception as e:
        logging.warning(f"警告：無法連線至 Ollama (localhost:11434) 或發生錯誤: {e}")

    # 3. 詢問使用者是否繼續
    print("\n" + "="*60)
    print(f"⚠️  警告: Ollama AI 功能無法使用 (模型: {cfg.OLLAMA_MODEL})")
    print("   這表示無法為照片生成文字描述，但基本的分類功能仍然可以運作。")
    print(f"   建議指令: ollama pull {cfg.OLLAMA_MODEL}")
    print("="*60)
    
    while True:
        choice = input("請問是否要在「沒有 AI 描述」的情況下繼續執行？(y/n): ").strip().lower()
        if choice == 'y':
            logging.info("使用者選擇忽略 AI 錯誤，繼續執行分類任務。")
            return False
        elif choice == 'n':
            logging.info("使用者中止程式。")
            sys.exit(0)
    
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

def generate_ollama_caption(image_path, output_txt_path, dog_name="狗狗", date_context=""):
    """呼叫本地 Ollama API 為圖片生成描述"""
    if not OLLAMA_AVAILABLE: return

    try:
        # 0. 準備提示詞 (加入名字與日期情境)
        prompt_text = cfg.OLLAMA_PROMPT.replace("{name}", dog_name).replace("{date_context}", date_context)

        # 1. 將圖片轉為 Base64
        with Image.open(image_path) as img:
            img = resize_for_ai(img) 
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 2. 建構 Request payload
        payload = {
            "model": cfg.OLLAMA_MODEL,
            "prompt": prompt_text,
            "stream": False,
            "images": [img_base64]
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(cfg.OLLAMA_API_URL, data=data, headers={'Content-Type': 'application/json'})

        # 3. 發送請求 (加入重試機制)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logging.info(f"  [AI 描述]: 正在請求 Ollama ({cfg.OLLAMA_MODEL})...")
                with urllib.request.urlopen(req) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    caption = result.get('response', '').strip()
                    
                    if caption:
                        # 4. 寫入檔案
                        with open(output_txt_path, 'w', encoding='utf-8') as f:
                            f.write(caption)
                        logging.info(f"  [AI 描述完成]: {caption[:50]}...")
                        return # 成功則退出函數
                    else:
                        logging.warning("  [AI 描述失敗]: Ollama 回傳空內容。" )
                        return
            except urllib.error.HTTPError as e:
                if e.code == 500 and attempt < max_retries - 1:
                    logging.warning(f"  [AI 忙碌中]: 伺服器錯誤 (500)，2秒後重試 ({attempt+1}/{max_retries})...")
                    time.sleep(2)
                else:
                    raise e # 最後一次還是失敗則拋出

    except Exception as e:
        logging.error(f"  [AI 描述錯誤]: {e}")
def create_markdown_log(original_path, final_path, base_dt, dog_name, age, action, caption, base_output_dir, score=-1.0):
    """為照片建立 Obsidian 友善的 Markdown 筆記"""
    try:
        # 1. 準備目錄
        md_dir = base_output_dir / "Obsidian_Logs" / dog_name / base_dt.strftime('%Y')
        md_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. 準備檔名與路徑
        md_filename = final_path.stem + ".md"
        md_path = md_dir / md_filename
        
        # 3. 處理圖片連結 (使用相對路徑)
        try:
            rel_path = os.path.relpath(final_path, md_dir)
            rel_path = Path(rel_path).as_posix()
            img_link_std = f"![{final_path.name}]({rel_path})"
        except ValueError:
            img_link_std = f"![{final_path.name}]({final_path.resolve().as_uri()})"
        
        # 4. 準備標籤與星等
        tags = [dog_name, age, action]
        if "[顏色異常]" in caption:
            tags.append("顏色異常")
        
        stars = ""
        score_display = ""
        if score >= 0:
            # 簡單的 5 星制轉換
            star_count = int(score * 5) + 1 if score > 0 else 0
            if star_count > 5: star_count = 5
            stars = "★" * star_count + "☆" * (5 - star_count)
            score_display = f"Aesthetic Score: {score:.2f} {stars}"
            if score >= cfg.AESTHETIC_SCORE_HIGH:
                tags.append("精選")
                
        tags_str = ", ".join([t.replace(" ", "_") for t in tags if t])
        
        # 5. 生成內容
        content = f"""---
date: {base_dt.strftime('%Y-%m-%d %H:%M:%S')}
dog: {dog_name}
age: {age}
action: {action}
tags: [{tags_str}]
image_path: "{final_path.name}"
score: {score:.2f}
verified: false
source_device: "iPhone 17 Pro"
---

# {final_path.stem} {stars}

{img_link_std}

> **{score_display}**

### 📝 AI 觀察日記
> {caption}

---
*檔案位置: `{final_path}`*
"""
        # 6. 寫入檔案
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    except Exception as e:
        logging.error(f"建立 Markdown 筆記失敗: {e}")

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

def calculate_memorial_score(yolo_conf, clip_score, img_shape, dog_bbox):
    """
    計算「紀念書指數」 (0.0 ~ 1.0)
    綜合考量: YOLO信心度、CLIP美感、主體佔比、解析度
    """
    h, w = img_shape[:2]
    img_area = h * w
    if img_area == 0: return 0.0

    # 1. YOLO 信心度 (基礎分) - 權重 0.3
    # 預期範圍 0.5~0.95 -> 歸一化
    score_yolo = min(1.0, max(0, (yolo_conf - 0.5) * 2)) 
    
    # 2. CLIP 美感 (藝術分) - 權重 0.3
    score_clip = clip_score
    
    # 3. 主體佔比 (構圖分) - 權重 0.2
    # 最佳佔比假設為 20% ~ 70%
    if dog_bbox:
        x1, y1, x2, y2 = dog_bbox
        box_area = (x2 - x1) * (y2 - y1)
        ratio = box_area / img_area
        if 0.2 <= ratio <= 0.7:
            score_ratio = 1.0
        elif ratio < 0.2:
            score_ratio = ratio / 0.2 # 太小扣分
        else:
            score_ratio = max(0, 1.0 - (ratio - 0.7) * 3) # 太大(爆框)扣分
    else:
        score_ratio = 0.0

    # 4. 解析度 (硬體分) - 權重 0.2
    # 假設 1200萬畫素 (4032x3024) 為滿分
    target_pixels = 12000000
    score_res = min(1.0, img_area / target_pixels)

    # 加權總分
    final_score = (score_yolo * 0.3) + (score_clip * 0.3) + (score_ratio * 0.2) + (score_res * 0.2)
    
    return final_score

def calculate_clip_aesthetic_score(image_pil_resized):
    """
    使用 CLIP 模型進行美感評分
    原理: 比較圖片與「高品質」vs「低品質」描述的相似度
    回傳: 0.0 ~ 1.0 的分數
    """
    if not CLIP_INITIALIZED: return 0.5 # 預設中庸

    try:
        # 正負向提示詞
        prompts = [
            "a masterpiece, high quality, sharp focus, clear, professional photography",
            "low quality, blurry, out of focus, grainy, amateur, ugly"
        ]
        
        inputs = clip_processor(text=prompts, images=image_pil_resized, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
            
        # 計算機率 (softmax)
        probs = outputs.logits_per_image.softmax(dim=1).squeeze().cpu().numpy()
        
        # 取正向提示詞的機率作為分數
        positive_score = probs[0]
        return float(positive_score)
        
    except Exception as e:
        logging.warning(f"CLIP 美感評分錯誤: {e}")
        return 0.5

def process_image_result(file_path, img_bgr, yolo_result, base_output_dir, progress_str=""):
    """
    處理單張圖片的推論結果 (漏斗式篩選邏輯)
    """
    base_dt = extract_datetime(file_path)
    # ... (省略中間代碼) ...
    # 2. 美感評估 (Apple Vision) - 漏斗第二層
    # aesthetic_score = mac_utils.calculate_aesthetic_score(file_path)
    
    # 改用 CLIP 進行評分 (需先裁切或縮放)
    # 這裡我們先簡單縮放原圖來評分，或者用裁切後的圖
    # 為了效能，我們用裁切後的圖來評分 (因為已經知道要裁切了)
    
    # 取得狗的 bbox 用於裁切
    has_dog = False
    dog_bbox = None
    yolo_conf = 0.0
    for box in yolo_result.boxes:
        if int(box.cls[0].item()) == 16:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            dog_bbox = (x1, y1, x2, y2)
            yolo_conf = float(box.conf[0].item()) # 取得信心度
            has_dog = True
            break
            
    if not has_dog:
        if cfg.SKIP_NO_DOG: return
        else: dog_bbox = None

    cropped_bgr, cropped_pil = crop_image(img_bgr, dog_bbox)
    image_pil_resized = resize_for_ai(cropped_pil)

    # 計算 CLIP 美感分數
    clip_aesthetic_score = calculate_clip_aesthetic_score(image_pil_resized)
    
    # 計算最終「紀念書指數」
    memorial_score = calculate_memorial_score(yolo_conf, clip_aesthetic_score, img_bgr.shape, dog_bbox)
    
    # 判斷門檻
    is_low_quality = memorial_score < cfg.AESTHETIC_SCORE_MIN
    is_high_quality = memorial_score >= cfg.AESTHETIC_SCORE_HIGH
    
    # 使用 memorial_score 作為顯示用的分數
    aesthetic_score = memorial_score 
    
    if is_low_quality:
        # 低分照片：只做最基本的存檔，不做後續 AI
        target_dir = base_output_dir / "低分"
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(file_path), str(target_dir / file_path.name))
        return

    # --- 進入精細處理流程 ---
    
    # 裁切與 CLIP 驗證 (上面已經做過了，直接用)
    # is_dog_clip, low_confidence = verify_is_dog_with_clip(image_pil_resized)
    # 注意：verify_is_dog_with_clip 內部會再次呼叫 CLIP，
    # 為了省去一次推論，其實可以把「驗證狗」和「美感評分」和「顏色/動作」合併成一次 CLIP call，
    # 但為了程式結構清晰，我們先分開呼叫 (MPS 很快，沒關係)。
    
    is_dog_clip, low_confidence = verify_is_dog_with_clip(image_pil_resized)
    if not is_dog_clip: return 

    # ID 與 顏色
    brightness = smart_brightness(cropped_bgr)
    # ... (後續邏輯保持不變) ...
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

    # 生成星星字串以便顯示
    stars = ""
    if aesthetic_score >= 0:
        star_count = int(aesthetic_score * 5) + 1 if aesthetic_score > 0 else 0
        if star_count > 5: star_count = 5
        stars = "★" * star_count + "☆" * (5 - star_count)
    
    score_log = f" [Score: {aesthetic_score:.2f} {stars}]" if aesthetic_score >= 0 else ""
    date_str = base_dt.strftime('%Y-%m-%d')
    logging.info(f"{progress_str} - [{date_str}] 分類: ID={dog_id_final}, 年齡={age_stage}, 動作={action_label}{score_log}")

    # 複製檔案
    final_path = construct_path_and_copy(
        file_path, base_dt, dog_id_final, age_stage, action_label, 
        low_confidence, id_correction_tag, base_output_dir
    )

    if final_path:
        # 3. 寫入 Finder Metadata (原生 Mac 功能)
        finder_comment = f"ID: {dog_id_final} | Action: {action_label}"
        if aesthetic_score > 0:
            finder_comment = f"Score: {aesthetic_score:.2f} | {finder_comment}"
            
        tags = []
        # 根據 ID 加顏色
        if dog_id_final in cfg.FINDER_TAG_COLORS:
            tags.append(cfg.FINDER_TAG_COLORS[dog_id_final])
        # 高分加紅色
        if is_high_quality:
            tags.append(cfg.FINDER_TAG_COLORS.get("精選", "Red"))
            
        mac_utils.write_finder_metadata(final_path, comment=finder_comment, tags=tags)

        # 4. Ollama 生成描述 (僅「紀念書等級」的高分照片才執行)
        caption_text = ""
        # 門檻邏輯：必須開啟 AI 功能，且必須是高分照片
        if cfg.ENABLE_OLLAMA_CAPTION and is_high_quality:
            txt_path = final_path.with_suffix('.txt')
            # 檢查是否已存在描述，若有則直接讀取，若無則生成
            if txt_path.exists() and not cfg.OVERWRITE_EXISTING:
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        caption_text = f.read().strip()
                except: pass
            
            if not caption_text: # 如果沒讀到或需要重新生成
                # 計算日期情境 (僅在特殊節日加入提示，平時不干擾)
                date_ctx = ""
                if base_dt.month == 11 and base_dt.day == 23:
                    date_ctx = "今天是這隻狗狗的生日！請在日記中加上生日快樂的氛圍。"
                
                generate_ollama_caption(file_path, txt_path, dog_name=dog_id_final, date_context=date_ctx)
                if txt_path.exists():
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            caption_text = f.read().strip()
                    except: pass
        elif cfg.ENABLE_OLLAMA_CAPTION:
            logging.info(f"  [跳過 AI]: 分數 ({memorial_score:.2f}) 未達紀念書門檻，僅進行分類。")

        # 5. 建立 Markdown (無論分數高低都建立，但高分會有 AI 日記)
        create_markdown_log(
            original_path=file_path,
            final_path=final_path,
            base_dt=base_dt,
            dog_name=dog_id_final,
            age=age_stage,
            action=action_label,
            caption=caption_text if caption_text else "（本照片為一般紀錄，未生成 AI 描述）",
            base_output_dir=base_output_dir,
            score=aesthetic_score 
        )

def process_file(file_path, base_output_dir):
    """(舊函式保留作為 fallback，或可移除)"""
    pass # 實際上 batch 模式不會用到這個了


def update_markdown_index(base_output_dir):
    """
    掃描所有 Markdown 日記，生成總目錄 (index.md)
    """
    logs_dir = base_output_dir / "Obsidian_Logs"
    if not logs_dir.exists(): return

    logging.info("正在更新 Markdown 總目錄 (index.md)...")
    
    entries = []
    
    # 遍歷所有 .md 檔案
    for md_file in logs_dir.rglob("*.md"):
        if md_file.name == "index.md": continue
        
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 簡單解析 YAML
            date_str = ""
            dog = ""
            action = ""
            score = 0.0
            
            # 逐行找
            for line in content.split('\n'):
                if line.startswith("date:"): date_str = line.split(":", 1)[1].strip()
                elif line.startswith("dog:"): dog = line.split(":", 1)[1].strip()
                elif line.startswith("action:"): action = line.split(":", 1)[1].strip()
                elif line.startswith("score:"): 
                    try: score = float(line.split(":", 1)[1].strip())
                    except: pass
                elif line.startswith("---") and date_str: break # YAML 結束
            
            # 計算星星
            stars = ""
            if score > 0:
                cnt = int(score * 5) + 1 if score > 0 else 0
                if cnt > 5: cnt = 5
                stars = "★" * cnt
            
            # 解析摘要 (從 AI 觀察日記區塊提取)
            summary = ""
            if "### 📝 AI 觀察日記" in content:
                try:
                    # 抓取標題後面的內容
                    part = content.split("### 📝 AI 觀察日記")[1]
                    # 去除引用符號 > 並清理空白
                    summary = part.replace(">", "").strip().split("\n")[0]
                    if len(summary) > 30: summary = summary[:30] + "..."
                except: pass
            
            # 建立相對路徑連結
            try:
                rel_path = os.path.relpath(md_file, logs_dir)
                # Windows 路徑相容
                rel_path = Path(rel_path).as_posix() 
            except: continue
            
            entries.append({
                "date": date_str,
                "dog": dog,
                "action": action,
                "score": score,
                "stars": stars,
                "summary": summary,
                "link": rel_path
            })
            
        except Exception as e:
            logging.warning(f"解析 MD 失敗 {md_file.name}: {e}")

    # 排序：日期新到舊
    entries.sort(key=lambda x: x["date"], reverse=True)
    
    # 生成 index.md 內容
    md_content = ["# 🐶 二季與四季的成長目錄\n"]
    md_content.append(f"> 最後更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_content.append(f"> 總計篇數: {len(entries)}\n")
    md_content.append("| 日期 | 主角 | 動作 | 評分 | 摘要 | 連結 |")
    md_content.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for e in entries:
        # 簡化日期顯示
        d_show = e["date"].split(" ")[0]
        link_md = f"[{d_show}]({e['link']})"
        row = f"| {link_md} | {e['dog']} | {e['action']} | {e['stars']} | {e['summary']} | [查看日記]({e['link']}) |"
        md_content.append(row)
        
    # 寫入
    with open(logs_dir / "index.md", 'w', encoding='utf-8') as f:
        f.write("\n".join(md_content))
        
    logging.info(f"目錄更新完成！共 {len(entries)} 篇日記。")

def main():
    if not Config.SOURCE_DIR.exists():
        logging.error(f"來源資料夾不存在: {Config.SOURCE_DIR}")
        if platform.system() == "Darwin":
            logging.info("提示 (Mac): 請確認您已掛載網路磁碟 (Finder -> 前往 -> 連接伺服器 -> smb://<YOUR_NAS_IP>)")
        sys.exit(1)
        
    if not Config.OUTPUT_DIR.exists():
        try: Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e: logging.error(f"無法建立輸出資料夾: {e}"); sys.exit(1)
            
    initialize_models()
    
    global OLLAMA_AVAILABLE
    OLLAMA_AVAILABLE = check_ollama_status()

    # 收集待處理檔案 (優化掃描 + 快取機制)
    CACHE_PATH = Path("scan_cache.json")
    STATE_PATH = Path("processed_state.json")
    
    # --- 讀取處理狀態 (續傳功能) ---
    processed_files = set()
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, 'r', encoding='utf-8') as f:
                processed_files = set(json.load(f))
        except: pass
        
    if processed_files:
        print("\n" + "="*50)
        print(f"📦 發現上次的處理紀錄: 已完成 {len(processed_files)} 個檔案")
        choice = input("請問要「接續處理」(y) 還是「全部重新檢查」(n)？(y/n): ").strip().lower()
        if choice == 'n':
            processed_files = set()
            print("已清除紀錄，將重新檢查所有檔案。")
        else:
            print("將跳過已處理的檔案...")
        print("="*50 + "\n")

    all_files_gen = []
    loaded_from_cache = False

    # 1. 嘗試讀取快取
    if CACHE_PATH.exists():
        mtime = os.path.getmtime(CACHE_PATH)
        mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "="*50)
        print(f"📦 發現掃描快取: {CACHE_PATH}")
        print(f"   上次掃描時間: {mtime_str}")
        choice = input("請問是否直接使用快取清單 (跳過 NAS 掃描)？(y/n): ").strip().lower()
        if choice == 'y':
            try:
                with open(CACHE_PATH, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    # 還原成 (Path, type) 的格式
                    all_files_gen = [(Path(p), t) for p, t in cached_data]
                logging.info(f"已從快取載入 {len(all_files_gen)} 個檔案。")
                loaded_from_cache = True
            except Exception as e:
                logging.error(f"讀取快取失敗: {e}，將重新掃描。")
        else:
            logging.info("使用者選擇重新掃描。")
        print("="*50 + "\n")

    # 2. 如果沒有讀取快取，則執行 NAS 掃描
    if not loaded_from_cache:
        img_exts = {'.jpg', '.jpeg', '.png', '.webp'}
        vid_exts = set(cfg.VIDEO_EXTENSIONS) if cfg.ENABLE_VIDEO_PROCESS else set()
        
        logging.info(f"正在掃描來源資料夾: {Config.SOURCE_DIR}")
        
        count = 0
        if Config.SOURCE_DIR.is_dir():
            for p in Config.SOURCE_DIR.rglob('*'):
                if not p.is_file(): continue
                ext = p.suffix.lower()
                if ext in img_exts:
                    all_files_gen.append((p, 'image'))
                    count += 1
                elif ext in vid_exts:
                    all_files_gen.append((p, 'video'))
                    count += 1
                
                if count > 0 and count % 500 == 0:
                    logging.info(f"  已找到 {count} 個檔案...")
        
        # 3. 掃描完成後，儲存快取
        if len(all_files_gen) > 0:
            try:
                # 將 Path 物件轉為字串以便存入 JSON
                cache_data = [(str(p), t) for p, t in all_files_gen]
                with open(CACHE_PATH, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
                logging.info(f"已將掃描結果儲存至快取: {CACHE_PATH}")
            except Exception as e:
                logging.warning(f"無法儲存快取: {e}")

    total_files = len(all_files_gen)
    if total_files == 0:
        logging.info("未找到任何可處理的檔案。" )
        return

    logging.info(f"共找到 {total_files} 個檔案 (包含影片)。開始處理...")
    
    # 分離圖片與影片 (圖片批次處理，影片逐一處理)
    image_files = []
    video_files = []
    for fp, ft in all_files_gen:
        if ft == 'image': image_files.append(fp)
        else: video_files.append(fp)
        
    start_time = time.time()
    processed_count_session = 0
    
    # --- 1. 處理圖片 (M4 優化批次處理) ---
    total_imgs = len(image_files)
    if total_imgs > 0:
        logging.info(f"開始批次處理圖片 (Batch Size: {cfg.BATCH_SIZE}, Device: {cfg.DEVICE})...")
        
        # 定義讀圖函數 (IO Bound)
        def load_img_data(path):
            try:
                # 使用 numpy 讀取加速
                data = np.fromfile(str(path), dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                return path, img
            except: return path, None

        # 批次迭代
        batch_size = cfg.BATCH_SIZE
        # 建立 ThreadPool (根據 CPU 核數調整，通常 4-8 夠用)
        with ThreadPoolExecutor(max_workers=8) as executor:
            for i in range(0, total_imgs, batch_size):
                batch_paths = image_files[i : i + batch_size]
                
                # 過濾掉已處理的 (續傳)
                valid_batch_paths = [p for p in batch_paths if str(p) not in processed_files]
                if not valid_batch_paths: continue
                
                # A. 非同步平行讀圖
                futures = [executor.submit(load_img_data, p) for p in valid_batch_paths]
                
                # 收集讀取結果
                loaded_batch = []
                valid_paths_for_inference = []
                
                for future in futures:
                    p, img = future.result()
                    if img is not None:
                        loaded_batch.append(img)
                        valid_paths_for_inference.append(p)
                    else:
                        logging.warning(f"無法讀取圖片: {p.name}")

                if not loaded_batch: continue

                # B. YOLO 批次推論 (MPS 加速)
                try:
                    # verbose=False 減少 log
                    results = yolo_model(loaded_batch, conf=0.25, iou=0.7, classes=[16, 0], verbose=False)
                    
                    # C. 逐一處理結果 (CPU 邏輯)
                    for j, result in enumerate(results):
                        orig_path = valid_paths_for_inference[j]
                        orig_img = loaded_batch[j]
                        
                        # 計算當前總序號
                        current_idx = i + j + 1
                        prog_str = f"[{current_idx}/{total_imgs}]"
                        
                        try:
                            process_image_result(orig_path, orig_img, result, Config.OUTPUT_DIR, progress_str=prog_str)
                            
                            processed_files.add(str(orig_path))
                            processed_count_session += 1
                        except Exception as e:
                            logging.error(f"處理失敗 {orig_path.name}: {e}")

                    # 進度回報與存檔
                    current_progress = i + len(batch_paths)
                    if current_progress % 50 == 0:
                        logging.info(f"已處理 {current_progress}/{total_imgs} 張圖片...")
                        with open(STATE_PATH, 'w', encoding='utf-8') as f: json.dump(list(processed_files), f)
                        
                        # 定時更新 Markdown 目錄 (每 200 張才更新一次，避免 NAS I/O 過重)
                        if current_progress % 200 == 0:
                            try:
                                update_markdown_index(Config.OUTPUT_DIR)
                            except Exception as e:
                                logging.warning(f"更新目錄失敗: {e}")
                        
                except Exception as e:
                    logging.error(f"YOLO 批次推論失敗: {e}")

    # --- 2. 處理影片 (維持逐一處理) ---
    for i, file_path in enumerate(video_files):
        if str(file_path) in processed_files: continue
        logging.info(f"[Video {i+1}/{len(video_files)}] {file_path.name}")
        try:
            process_video_file(file_path, Config.OUTPUT_DIR)
            processed_files.add(str(file_path))
            processed_count_session += 1
            if processed_count_session % 5 == 0:
                with open(STATE_PATH, 'w', encoding='utf-8') as f: json.dump(list(processed_files), f)
        except Exception as e: logging.error(f"影片處理失敗: {e}")

    # 最後再存一次確保完整
    with open(STATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(list(processed_files), f)
    
    # 更新 Markdown 目錄
    update_markdown_index(Config.OUTPUT_DIR)
            
    logging.info(f"處理完成！本次處理: {processed_count_session} 張。總用時: {time.time() - start_time:.2f} 秒。" )

if __name__ == '__main__':
    main()
