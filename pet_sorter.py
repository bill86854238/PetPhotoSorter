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
from datetime import datetime, date
from PIL import Image, ExifTags
import torch
import logging
import time

# 修正：導入 CLIP 模型的必要類別
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    # 這是為了在沒有安裝 transformers 庫時提供友善提示
    print("警告：缺少 'transformers' 庫。請嘗試安裝：pip install transformers torch torchvision")
    CLIPProcessor = None
    CLIPModel = None


# 配置日誌：設定為 INFO 級別，以顯示關鍵資訊和操作細節 
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ==============================================================================
# 設定區塊 (Config Block)
# ==============================================================================
class Config:
    # 檔案路徑設定
    # 警告: UNC 網路路徑 '\\192.168.50.143\...' 可能因權限或環境沙箱限制而無法訪問。
    # 建議首次運行時，先使用本地測試路徑 (如 r"C:\TestSource" 和 r"C:\TestOutput") 進行測試。
    SOURCE_DIR = Path(r"\\192.168.50.143\home\Photos\MobileBackup") 
    OUTPUT_DIR = Path(r"\\192.168.50.143\photo\照片-Pet_分類") 
    
    # 【已移除】RUN_FOLDER_PREFIX: 採用穩定輸出結構，不再為每次運行建立時間戳資料夾

    # --- 性能優化設定 ---
    # 針對 AI 模型輸入進行預先縮放的最大尺寸（長邊）。YOLO/CLIP 效能優化的關鍵！
    MAX_AI_INPUT_SIZE = 1024 # 建議值 640 或 1024 像素
    
    # --- 狗狗 ID 分類設定 ---
    BRIGHTNESS_THRESHOLD = 185 # 亮度閾值 (用於區分淺色/深色外觀)
    DOG_ID_BRIGHT_NAME = "二季" # 代表淺色外觀的米克斯 (Bright Appearance)
    DOG_ID_DARK_NAME = "四季"   # 代表深色外觀的米克斯 (Dark Appearance)
    
    # --- 年齡分類設定 ---
    DOG_BIRTH_DATE = datetime(2022, 11, 23) 
    
    # --- CLIP 驗證閾值 ---
    CLIP_VERIFY_MULTIPLIER = 1.2 # 非狗機率 > 狗機率 * 1.2 視為誤判 (Failure: 直接跳過)
    CLIP_MIN_DOG_PROBABILITY = 0.6 # 狗的絕對機率低於此值，視為低信心警告 (Warning: 複製到潛在誤判資料夾)
    
    # 功能開關
    SKIP_NO_DOG = True             
    OVERWRITE_EXISTING = False     
    ENABLE_ACTION_CLASSIFY = True  
    ENABLE_DATE_CLASSIFY = True   # <--- 確保年齡分類已開啟
    RENAME_BY_DATETIME = True      
    ENABLE_COLOR_CLASSIFY = True   
    ENABLE_GEAR_CLASSIFY = False   
    ENABLE_ENV_CLASSIFY = False    
    SKIP_IF_HUMAN_FACE = False 
    ENABLE_DUPLICATE_CHECK = True # 啟用重複檔案檢查 (aHash 檢查速度極快)

    # --- 不確定性處理資料夾 ---
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
# 儲存已處理檔案的 aHash 值，用於重複檔案檢查
known_hashes = set()

# ==============================================================================
# 輔助函數 (Utility Functions)
# ==============================================================================

def initialize_models():
    """初始化 YOLO 和 CLIP 模型"""
    global CLIP_INITIALIZED, device, clip_model, clip_processor, yolo_model
    
    # --- 初始化 CLIP 模型 ---
    if CLIPModel is not None:
        try:
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
        # 不退出，確保可以進行路徑檢查和基礎操作
        pass

def calculate_aHash(img_bgr):
    """計算圖像的平均哈希 (aHash)，用於內容重複檢測"""
    if img_bgr is None or img_bgr.size == 0:
        return None
    try:
        # 1. 縮放為 8x8
        img_resized = cv2.resize(img_bgr, (8, 8), interpolation=cv2.INTER_AREA)
        # 2. 轉換為灰度圖
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        # 3. 計算平均灰度
        avg = np.mean(img_gray)
        # 4. 根據像素是否大於平均值生成二進制序列 (哈希)
        # 產生一個 64-bit 數字
        hash_value = sum([1 << i for i, pixel in enumerate(img_gray.flatten()) if pixel > avg])
        # 轉換為 16 位元的十六進制字串
        return f'{hash_value:016x}'
    except Exception as e:
        logging.debug(f"計算 aHash 失敗: {e}")
        return None

def resize_for_ai(image_pil):
    """
    將圖像等比例縮放到 MAX_AI_INPUT_SIZE，以加速 AI 推理。
    """
    w, h = image_pil.size
    max_size = cfg.MAX_AI_INPUT_SIZE
    
    if max(w, h) > max_size:
        if w > h:
            new_w = max_size
            new_h = int(h * (max_size / w))
        else:
            new_h = max_size
            new_w = int(w * (max_size / h))
            
        resized_pil = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        logging.debug(f"  - 圖片已從 {w}x{h} 縮放至 {new_w}x{new_h} 以加速 AI 推理。")
        return resized_pil
    
    return image_pil

def classify_age(photo_dt, birth_dt):
    """根據照片日期和出生日期，判斷狗狗的年齡階段。"""
    
    # 計算照片拍攝時的狗狗完整歲數
    age_years = photo_dt.year - birth_dt.year - ((photo_dt.month, photo_dt.day) < (birth_dt.month, birth_dt.day))
    
    # 新增數字前綴和年齡範圍
    if age_years < 1:
        return "1_幼犬 (0-1歲)"       
    elif age_years < 3:
        return "2_少年期 (1-3歲)"     
    elif age_years < 7:
        return "3_青壯年 (3-7歲)"     
    elif age_years < 12:
        return "4_中年期 (7-12歲)"     
    else:
        return "5_老年 (12歲以上)"       


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


def classify_action(image_pil_resized):
    """使用 CLIP 模型判斷狗的動作"""
    if not CLIP_INITIALIZED or not cfg.ENABLE_ACTION_CLASSIFY: return "動作_無效"

    action_descriptions = [
        "A dog is standing and looking around.",
        "A dog is sitting with its front paws down.",
        "A dog is lying down, resting or sleeping.",
        "A dog is running, jumping, or moving quickly."
    ]

    # 使用已經過預先縮放的 PIL 圖像
    inputs = clip_processor(text=action_descriptions, images=image_pil_resized, return_tensors="pt", padding=True)
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


def verify_is_dog_with_clip(image_pil_resized):
    """
    使用 CLIP 模型驗證偵測到的物體是否真的是狗。
    回傳 (is_dog: bool, is_low_confidence_warning: bool)
    """
    if not CLIP_INITIALIZED: return True, False # 如果 CLIP 沒初始化，則跳過驗證

    verification_descriptions = [
        # 0: Dog
        "A photograph of a pet dog, including mixed-breeds (Labrador, Husky, Poodle, etc.).", 
        # 1: Bear
        "A photograph of a bear (Black bear, brown bear, panda bear, etc.).",
        # 2: Other Wild Animal
        "A photograph of a wild animal (fox, raccoon, squirrel, etc.)."
    ]

    # 使用已經過預先縮放的 PIL 圖像
    inputs = clip_processor(text=verification_descriptions, images=image_pil_resized, return_tensors="pt", padding=True)
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
        logging.warning(f"  [CLIP 驗證失敗/SKIP]: 非狗機率過高 ({max_non_dog_prob:.2f})。判定為誤判。")
        return False, False
    
    # 2. 檢查是否符合絕對低機率誤判 (Warning/Uncertainty: 仍視為狗，但標記為低信心)
    if dog_prob < cfg.CLIP_MIN_DOG_PROBABILITY:
        logging.warning(f"  [CLIP 潛在誤判警告]: 狗機率低於閾值 ({dog_prob:.2f} < {cfg.CLIP_MIN_DOG_PROBABILITY})。將標記為低信心。")
        return True, True 
    
    # 3. 驗證通過 (Success: 正常處理)
    logging.info(f"  [CLIP 驗證通過]: 狗機率 {dog_prob:.2f}。")
    return True, False


def classify_color_with_clip(image_pil_resized):
    """
    使用 CLIP 模型判斷狗的顏色特徵，並直接映射到設定的 ID 名稱。
    回傳 (color_id_name: str, diag_data: dict)
    """
    if not CLIP_INITIALIZED or not cfg.ENABLE_COLOR_CLASSIFY:
        return "未知顏色_未啟用CLIP", None

    text_descriptions = [
        # 0: 深色 (對應四季)
        f"A dog with predominantly dark, black, or deep brown fur, matching {cfg.DOG_ID_DARK_NAME} characteristics.", 
        # 1: 淺色 (對應二季)
        f"A dog with predominantly light, white, cream, or golden fur, matching {cfg.DOG_ID_BRIGHT_NAME} characteristics."
    ]

    # 使用已經過預先縮放的 PIL 圖像
    inputs = clip_processor(text=text_descriptions, images=image_pil_resized, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clip_model(**inputs)

    # Logits per image
    logits = outputs.logits_per_image.squeeze().cpu()
    
    # 計算 Softmax 概率
    probs = torch.softmax(logits, dim=0).numpy()
    
    # 找出最高概率的索引
    max_index = np.argmax(probs)
    
    # 根據索引返回對應的 ID 名稱
    if max_index == 0:
        color_id_name = cfg.DOG_ID_DARK_NAME
    else: # max_index == 1
        color_id_name = cfg.DOG_ID_BRIGHT_NAME
        
    diag_data = {
        cfg.DOG_ID_DARK_NAME: probs[0],
        cfg.DOG_ID_BRIGHT_NAME: probs[1]
    }
    
    logging.info(f"  - CLIP 毛色分類結果: {color_id_name} (機率: {probs[max_index]:.2f})")
    
    return color_id_name, diag_data


def extract_datetime(image_path):
    """從 EXIF 或檔案名中提取拍攝時間"""
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data is not None:
                # 尋找 ExifTags.TAGS 字典中的日期時間標籤 (36867 = DateTimeOriginal)
                for tag, value in exif_data.items():
                    if ExifTags.TAGS.get(tag) == 'DateTimeOriginal':
                        # 格式: 'YYYY:MM:DD HH:MM:SS'
                        dt_str = value
                        return datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
            
    except Exception as e:
        # logging.debug(f"  - 無 EXIF: {e}")
        pass

    # 嘗試從檔名中提取 (格式: YYYYMMDD_HHMMSS)
    filename_parts = image_path.name.split('_')
    if len(filename_parts) >= 2:
        try:
            date_str = filename_parts[0]
            time_str = filename_parts[1]
            if len(date_str) == 8 and date_str.isdigit() and len(time_str) == 6 and time_str.isdigit():
                dt_str = date_str + time_str
                return datetime.strptime(dt_str, '%Y%m%d%H%M%S')
        except ValueError:
            pass
            
    # 最終使用檔案的修改時間
    timestamp = os.path.getmtime(image_path)
    return datetime.fromtimestamp(timestamp)


# ==============================================================================
# 主要處理邏輯 (Main Processing Logic)
# ==============================================================================

def preload_hashes():
    """
    預先載入 Config.OUTPUT_DIR 中所有已處理圖片的 aHash 值，用於全面重複檔案檢測。
    """
    if not cfg.ENABLE_DUPLICATE_CHECK:
        logging.info("重複檔案檢查 (aHash) 已禁用。")
        return

    base_output_dir = Config.OUTPUT_DIR # 檢查總目標資料夾
    if not base_output_dir.exists():
        logging.warning("總目標輸出資料夾不存在，跳過預載入哈希。")
        return

    logging.info("正在預載入總目標資料夾中 (所有歷史分類) 的檔案哈希值...")
    total_files = 0
    start_time = time.time()
    
    try:
        # 遍歷總輸出資料夾及其所有子資料夾
        for file_path in base_output_dir.rglob('*'):
            # 確保只處理圖片檔案，並跳過臨時/隱藏文件
            if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                total_files += 1
                try:
                    # 使用 cv2 載入圖片 (對網路路徑更友好)
                    img_bgr = cv2.imdecode(np.fromfile(str(file_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                    a_hash = calculate_aHash(img_bgr)
                    if a_hash:
                        known_hashes.add(a_hash)
                except Exception as e:
                    logging.debug(f"無法載入或計算哈希: {file_path.name} ({e})")
                    
    except Exception as e:
        logging.error(f"預載入哈希失敗 (可能為網路路徑權限或連線問題): {e}")

    elapsed_time = time.time() - start_time
    logging.info(f"總共找到 {total_files} 個歷史圖片檔案。哈希清單建立完成 (用時: {elapsed_time:.2f} 秒)。")


def get_dog_bbox(image_bgr):
    """
    使用 YOLO 模型偵測圖像中的物體，並返回狗的邊界框 (bbox)。
    """
    if yolo_model is None:
        logging.warning("YOLO 模型未初始化，無法進行物體偵測。")
        return None, None

    # 將 BGR 轉換為 RGB 供 YOLO 使用
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 運行 YOLO 模型
    # 限制輸入尺寸以加速推理
    results = yolo_model(image_rgb, conf=0.25, iou=0.7, classes=[16, 0]) # 16: dog, 0: person
    
    dog_bbox = None
    person_bbox = None
    
    # 處理結果
    for r in results:
        # 處理邊界框和類別
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Dog (類別 ID = 16)
            if cls_id == 16:
                dog_bbox = (x1, y1, x2, y2)
                logging.debug(f"  - YOLO 偵測到狗 (Dog BBOX: {dog_bbox})")
                
            # Person (類別 ID = 0)
            elif cls_id == 0 and cfg.SKIP_IF_HUMAN_FACE:
                person_bbox = (x1, y1, x2, y2)
                logging.debug(f"  - YOLO 偵測到人 (Person BBOX: {person_bbox})")
                
    
    if cfg.SKIP_IF_HUMAN_FACE and person_bbox is not None:
        # 如果設定為跳過含人臉的圖片，則視為未偵測到合格的狗
        logging.info("  - 偵測到人臉且設定為跳過 (SKIP_IF_HUMAN_FACE=True)。")
        return None, None 

    return dog_bbox, person_bbox # person_bbox 僅用於 crop_image

def crop_image(image_bgr, dog_bbox):
    """
    根據狗的邊界框裁剪並擴展圖像，返回裁剪後的 BGR 圖像和 PIL 圖像。
    如果沒有邊界框，則返回原圖。
    """
    if dog_bbox is None:
        # 如果沒有偵測到狗，則使用原圖進行 CLIP 驗證
        return image_bgr, Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    h, w, _ = image_bgr.shape
    x1, y1, x2, y2 = dog_bbox

    # 擴展邊界框
    dx = int((x2 - x1) * cfg.CROP_SHRINK_RATIO)
    dy = int((y2 - y1) * cfg.CROP_SHRINK_RATIO)
    
    # 確保邊界在圖像範圍內
    crop_x1 = max(0, x1 - dx)
    crop_y1 = max(0, y1 - dy)
    crop_x2 = min(w, x2 + dx)
    crop_y2 = min(h, y2 + dy)

    # 裁剪
    cropped_bgr = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # 轉換為 PIL 格式
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    cropped_pil = Image.fromarray(cropped_rgb)

    return cropped_bgr, cropped_pil


def get_image_paths(source_dir):
    """遞迴地從來源資料夾中獲取所有圖片檔案路徑。"""
    if not source_dir.is_dir():
        logging.error(f"錯誤：來源資料夾不存在或無法訪問: {source_dir}")
        return []
    
    # 過濾常見圖片副檔名，並遞迴搜尋
    image_paths = [p for p in source_dir.rglob('*') if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    return image_paths


def construct_path_and_copy(file_path, base_dt, dog_id_final, age_stage, action_label, low_confidence_warning, id_correction_tag, base_output_dir):
    """
    根據分類結果建構目標路徑，並將檔案複製到該路徑。
    使用傳入的 base_output_dir (即 Config.OUTPUT_DIR) 作為基準。
    """
    
    # 1. 建立分類結構 (相對於總輸出資料夾)
    category_list = [
        dog_id_final,              # 最終 ID/顏色分類 (必選)
        age_stage,                 # 年齡階段 (必選)
        action_label               # 動作分類 (如果啟用)
    ]
    
    # 移除「無效」或「未啟用」的標籤
    category_list = [c for c in category_list if c and c not in ["動作_無效", "未知顏色_未啟用CLIP", "動作_未啟用"]]

    # 2. 決定是否進入「潛在誤判或修正」資料夾
    uncertainty_tags = []
    
    if id_correction_tag:
        # 去掉修正標籤中的中括號 [ ]
        uncertainty_tags.append(id_correction_tag.strip('[]'))
    if low_confidence_warning:
        uncertainty_tags.append("低信心")

    if uncertainty_tags:
        # 進入不確定性資料夾
        tags_str = "_".join(uncertainty_tags)
        # 結構: [BASE_OUTPUT_DIR] / [UNCERTAINTY_FOLDER] / [ID] / [AGE] / [ACTION]
        target_dir = base_output_dir / Config.UNCERTAINTY_FOLDER_NAME / dog_id_final / age_stage / action_label
        new_filename_prefix = f"[{tags_str}]_"
    else:
        # 正常分類資料夾
        # 結構: [BASE_OUTPUT_DIR] / [ID] / [AGE] / [ACTION]
        target_dir = base_output_dir / Path(*category_list)
        new_filename_prefix = ""


    # 3. 建立目標資料夾
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"錯誤：無法建立目標資料夾 {target_dir}，可能為網路路徑權限問題。錯誤: {e}")
        return

    # 4. 處理檔名
    if cfg.RENAME_BY_DATETIME:
        # 檔名格式: [前綴]YYYYMMDD_HHMMSS_[最終ID]_[Action]_[OriginalName]
        timestamp_str = base_dt.strftime('%Y%m%d_%H%M%S')
        action_part = action_label.replace('動作_', '') # 檔名中只保留「站立」、「躺臥」等
        new_filename = f"{new_filename_prefix}{timestamp_str}_{dog_id_final}_{action_part}_{file_path.stem}{file_path.suffix}"
    else:
        # 檔名格式: [前綴][OriginalName]
        new_filename = f"{new_filename_prefix}{file_path.name}"

    target_path = target_dir / new_filename

    # 5. 複製檔案
    try:
        # 檢查目標檔案是否已經存在
        if target_path.exists() and not cfg.OVERWRITE_EXISTING:
            logging.info(f"  [跳過/INFO]: 目標檔案已存在且不允許覆蓋: {target_path.name}")
            return
        
        # 執行複製操作 (使用 shutil.copy2 保留更多元數據，如時間戳)
        shutil.copy2(str(file_path), str(target_path))
        # 記錄相對路徑，讓日誌更簡潔
        logging.info(f"  [成功複製]: -> {target_path.relative_to(base_output_dir)}")
        
    except Exception as e:
        logging.error(f"錯誤：複製檔案失敗。檔案: {file_path.name} -> {target_path}。錯誤: {e}")
        

def process_file(file_path, base_output_dir):
    """處理單一圖片檔案，進行所有分類並複製到目標位置。"""
    
    logging.info(f"-> 開始處理: {file_path.name}")

    # 1. 提取時間 (第一步，用於後續年齡計算和檔名)
    base_dt = extract_datetime(file_path)
    
    # 2. 載入圖像
    try:
        # 使用 numpy 和 cv2.imdecode 來處理中文路徑/網路路徑可能帶來的問題
        img_bgr = cv2.imdecode(np.fromfile(str(file_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            logging.warning(f"  [跳過/WARNING]: 無法載入圖片 (cv2.imdecode 返回 None)。檔案可能已損壞或路徑編碼錯誤。")
            return
    except Exception as e:
        logging.warning(f"  [跳過/WARNING]: 載入圖片發生異常。錯誤: {e}")
        return

    # 3. 重複檔案檢查 (aHash)
    a_hash = calculate_aHash(img_bgr)
    # **核心重複檢查邏輯**：如果哈希值已存在於歷史分類清單中，則跳過。
    if cfg.ENABLE_DUPLICATE_CHECK and a_hash in known_hashes and not cfg.OVERWRITE_EXISTING:
        logging.info(f"  [跳過/INFO]: 檔案內容與目標資料夾中已存在檔案重複 (aHash: {a_hash})。")
        return
    
    # 如果是新檔案，則將其哈希值加入清單，防止在本次運行中重複處理
    if cfg.ENABLE_DUPLICATE_CHECK and a_hash:
         known_hashes.add(a_hash)
    
    # 4. YOLO 偵測
    dog_bbox, person_bbox = get_dog_bbox(img_bgr)
    
    # 根據配置檢查是否跳過無狗圖片
    if dog_bbox is None and cfg.SKIP_NO_DOG:
        logging.info("  [跳過/INFO]: YOLO 未偵測到狗 (或偵測到人臉並跳過)。")
        return
    
    # 5. 裁剪圖片並準備用於 CLIP
    cropped_bgr, cropped_pil = crop_image(img_bgr, dog_bbox)
    
    # 6. 縮放圖像 (加速 CLIP 處理)
    image_pil_resized = resize_for_ai(cropped_pil)
    
    # 7. CLIP 驗證 (防止將非狗物體分類進去)
    is_dog, low_confidence_warning = verify_is_dog_with_clip(image_pil_resized)
    if not is_dog:
        logging.info("  [跳過/INFO]: CLIP 驗證判定為非狗。")
        return

    # 8. ID 初步判斷 (亮度)
    brightness = smart_brightness(cropped_bgr)
    if brightness >= cfg.BRIGHTNESS_THRESHOLD:
        dog_id_initial = cfg.DOG_ID_BRIGHT_NAME
    else:
        dog_id_initial = cfg.DOG_ID_DARK_NAME
    
    logging.info(f"  - 初步 ID (亮度 {brightness:.1f}) -> {dog_id_initial}")

    # 9. CLIP 毛色/ID 最終確認
    dog_id_final = dog_id_initial
    id_correction_tag = ""
    
    if cfg.ENABLE_COLOR_CLASSIFY and CLIP_INITIALIZED:
        clip_color_id, _ = classify_color_with_clip(image_pil_resized)
        
        # 進行 ID 修正檢查
        if dog_id_initial != clip_color_id:
            # 亮度判斷和 CLIP 毛色判斷不一致，以 CLIP 為準並發出修正警告
            id_correction_tag = f"[{dog_id_initial}被{clip_color_id}修正]"
            dog_id_final = clip_color_id
            logging.warning(f"  - 發生 ID 修正: 亮度判斷 ({dog_id_initial}) 被 CLIP 毛色 ({clip_color_id}) 修正。")
        else:
            dog_id_final = dog_id_initial # 兩者一致，保持原樣
            
    else:
        # 如果未啟用 CLIP 顏色分類，最終 ID 就是初步 ID
        dog_id_final = dog_id_initial
        
    # 10. 年齡分類
    if cfg.ENABLE_DATE_CLASSIFY:
        age_stage = classify_age(base_dt, cfg.DOG_BIRTH_DATE)
    else:
        age_stage = "未分類"

    # 11. 動作分類 (使用 CLIP)
    if cfg.ENABLE_ACTION_CLASSIFY and CLIP_INITIALIZED:
        action_label = classify_action(image_pil_resized)
    else:
        action_label = "動作_未啟用"
        
    logging.info(f"  - 最終分類結果: ID={dog_id_final}, 年齡={age_stage}, 動作={action_label}")

    # 12. 複製檔案
    construct_path_and_copy(
        file_path, 
        base_dt, 
        dog_id_final, 
        age_stage, 
        action_label, 
        low_confidence_warning, 
        id_correction_tag,
        base_output_dir # 傳入 Config.OUTPUT_DIR
    )

# --- Main Execution Block ---
def main():
    if not Config.SOURCE_DIR.exists():
        logging.error(f"來源資料夾不存在: {Config.SOURCE_DIR}")
        sys.exit(1)
        
    if not Config.OUTPUT_DIR.exists():
        logging.warning(f"目標輸出資料夾不存在: {Config.OUTPUT_DIR}。將嘗試建立。")
        try:
            Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"無法建立目標輸出資料夾: {e}")
            sys.exit(1)
            
    # 1. 初始化模型 (YOLO 和 CLIP)
    initialize_models()

    # 2. 預載入所有歷史分類的哈希值
    if cfg.ENABLE_DUPLICATE_CHECK:
        preload_hashes()

    # 3. 獲取所有圖片路徑
    image_paths = get_image_paths(Config.SOURCE_DIR)
    total_files = len(image_paths)
    if total_files == 0:
        logging.info(f"在來源資料夾 {Config.SOURCE_DIR} 中未找到任何圖片檔案。")
        return

    logging.info(f"共找到 {total_files} 個圖片檔案待處理。")

    # 4. 處理檔案
    processed_count = 0
    start_time = time.time()
    
    for i, file_path in enumerate(image_paths):
        logging.info(f"==================================================")
        logging.info(f"[{i+1}/{total_files}] 處理檔案: {file_path.name}")
        
        try:
            # 確保傳遞 Config.OUTPUT_DIR 作為基礎輸出路徑
            process_file(file_path, Config.OUTPUT_DIR)
            processed_count += 1
        except Exception as e:
            logging.error(f"處理 {file_path.name} 時發生未預期的錯誤: {e}")
            # 可以在此處添加更多錯誤處理邏輯
            
    end_time = time.time()
    elapsed_time = end_time - start_time

    logging.info("==================================================")
    logging.info("處理完成！")
    logging.info(f"總共處理了 {total_files} 個檔案。")
    logging.info(f"總運行時間: {elapsed_time:.2f} 秒。")

if __name__ == '__main__':
    main()