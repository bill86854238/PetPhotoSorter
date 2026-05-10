import os
import json
import time
import logging
import threading
import platform
import shutil
import cv2
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from PIL import Image, ExifTags
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import base64
import io

# 核心 AI 庫
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    CLIPProcessor = None
    CLIPModel = None

try:
    import mac_utils
except ImportError:
    mac_utils = None

class SorterEngine:
    def __init__(self, config_path="config.json", progress_callback=None, log_callback=None):
        self.config_path = Path(config_path)
        self.progress_callback = progress_callback  # (current, total, message)
        self.log_callback = log_callback           # (message, level)
        self.is_running = False
        self.stop_requested = False
        
        # 模型與狀態
        self.yolo_model = None
        self.clip_model = None
        self.clip_processor = None
        self.device = "cpu"
        self.known_hashes = set() # 用於重複檢查
        
        # 統計數據 (供 GUI 使用)
        self.stats = {
            "processed": 0,
            "total": 0,
            "high_score": 0,
            "dogs_found": 0,
            "videos": 0,
            "duplicates": 0,
            "dog_counts": {} # {"二季": 10, "四季": 5}
        }
        
        self.load_config()

    def log(self, msg, level=logging.INFO):
        clean_msg = str(msg).strip()
        logging.log(level, clean_msg)
        if self.log_callback:
            self.log_callback(clean_msg, level)

    def load_config(self):
        self._config_data = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config_data = json.load(f)
            except Exception as e:
                self.log(f"載入 config.json 失敗: {e}", logging.WARNING)
        
        settings = self._config_data.get("settings", {})
        self.nas_ip = self._config_data.get("nas_ip", "192.168.1.100")
        
        # 基礎 AI 設定
        self.batch_size = settings.get("batch_size", 16)
        self.device_pref = settings.get("device", "mps" if platform.system() == "Darwin" else "cpu")
        self.aesthetic_min = settings.get("aesthetic_score_min", 0.2)
        self.aesthetic_high = settings.get("aesthetic_score_high", 0.6)
        self.enable_ollama = settings.get("enable_ollama", True)
        self.ollama_url = settings.get("ollama_url", "http://localhost:11434")
        self.ollama_model = settings.get("ollama_model", "moondream")
        
        # 狗狗分類設定
        self.dog_id_bright = settings.get("dog_id_bright", "亮色狗狗")
        self.dog_id_dark = settings.get("dog_id_dark", "暗色狗狗")
        self.brightness_threshold = settings.get("brightness_threshold", 185)
        self.dog_birth_date = datetime(2022, 11, 23)
        
        # 功能開關
        self.enable_video = settings.get("enable_video", True)
        self.enable_duplicate_check = settings.get("enable_duplicate_check", True)
        self.enable_action_classify = settings.get("enable_action_classify", True)

        # 決定路徑 (GUI 啟動時會被覆蓋)
        if platform.system() == "Darwin":
            mac_cfg = self._config_data.get("mac", {})
            self.source_dir = Path(mac_cfg.get("source_dir", "/Volumes/home/Photos/MobileBackup"))
            self.output_dir = Path(mac_cfg.get("output_dir", "/Volumes/photo/照片-Pet_分類"))
        else:
            win_cfg = self._config_data.get("windows", {})
            src_tmpl = win_cfg.get("source_dir", r"\\{ip}\home\Photos\MobileBackup")
            out_tmpl = win_cfg.get("output_dir", r"\\{ip}\photo\照片-Pet_分類")
            self.source_dir = Path(src_tmpl.replace("{ip}", self.nas_ip))
            self.output_dir = Path(out_tmpl.replace("{ip}", self.nas_ip))

    def save_config_to_file(self):
        data = self._config_data
        if "settings" not in data: data["settings"] = {}
        data["settings"].update({
            "batch_size": self.batch_size,
            "aesthetic_score_min": self.aesthetic_min,
            "aesthetic_score_high": self.aesthetic_high,
            "enable_ollama": self.enable_ollama,
            "ollama_url": self.ollama_url,
            "ollama_model": self.ollama_model,
            "enable_video": self.enable_video,
            "enable_duplicate_check": self.enable_duplicate_check,
            "enable_action_classify": self.enable_action_classify
        })
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        except: return False

    def initialize_models(self):
        if YOLO is None: return False
        self.device = self.device_pref
        if self.device == "mps" and not torch.backends.mps.is_available(): self.device = "cpu"
        elif self.device == "cuda" and not torch.cuda.is_available(): self.device = "cpu"
        
        try:
            self.yolo_model = YOLO("yolov8n.pt").to(self.device)
            if CLIPModel is not None:
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            return True
        except Exception as e:
            self.log(f"模型初始化失敗: {e}", logging.ERROR)
            return False

    # --- 演算法工具 ---
    def calculate_ahash(self, img_bgr):
        """計算感知雜湊用於重複檢查"""
        try:
            resized = cv2.resize(img_bgr, (8, 8), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            avg = gray.mean()
            hash_val = sum([1 << i for i, v in enumerate(gray.flatten()) if v > avg])
            return f"{hash_val:016x}"
        except: return None

    def classify_action(self, pil_img):
        if not self.clip_model or not self.enable_action_classify: return "動作_一般"
        actions = ["standing", "sitting", "lying down", "running or jumping"]
        labels = ["站立", "坐下", "躺臥", "活動"]
        try:
            inputs = self.clip_processor(text=[f"A dog is {a}" for a in actions], images=pil_img, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                probs = self.clip_model(**inputs).logits_per_image.softmax(dim=1).cpu().numpy()[0]
            return f"動作_{labels[np.argmax(probs)]}"
        except: return "動作_一般"

    def classify_id_by_color(self, pil_img, initial_id):
        """利用 CLIP 修正因光線造成的 ID 誤判"""
        if not self.clip_model: return initial_id
        try:
            prompts = [f"A dog with dark fur like {self.dog_id_dark}", f"A dog with light fur like {self.dog_id_bright}"]
            inputs = self.clip_processor(text=prompts, images=pil_img, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                probs = self.clip_model(**inputs).logits_per_image.softmax(dim=1).cpu().numpy()[0]
            return self.dog_id_dark if probs[0] > probs[1] else self.dog_id_bright
        except: return initial_id

    # --- 影片處理 ---
    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): return False, "無法讀取"
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 每 2 秒抽一幀
        step = int(fps * 2) if fps > 0 else 30
        
        dog_detected = False
        for fno in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ret, frame = cap.read()
            if not ret: break
            
            res = self.yolo_model(frame, verbose=False)
            if len(res[0].boxes) > 0:
                dog_detected = True
                break
        cap.release()
        return dog_detected, "動作_一般"

    # --- 核心流程 ---
    def process_image(self, img_bgr, yolo_result, file_path):
        if len(yolo_result.boxes) == 0: return
        self.stats["dogs_found"] += 1
        
        # 重複檢查
        if self.enable_duplicate_check:
            h = self.calculate_ahash(img_bgr)
            if h in self.known_hashes:
                self.stats["duplicates"] += 1
                return
            self.known_hashes.add(h)

        # 分類與評分
        photo_dt = datetime.fromtimestamp(os.path.getmtime(file_path))
        age_str = self.classify_age(photo_dt)
        
        x1, y1, x2, y2 = map(int, yolo_result.boxes[0].xyxy[0].tolist())
        dog_crop = img_bgr[max(0, y1-20):y2+20, max(0, x1-20):x2+20]
        pil_crop = Image.fromarray(cv2.cvtColor(dog_crop if dog_crop.size > 0 else img_bgr, cv2.COLOR_BGR2RGB))
        
        score = self.calculate_aesthetic_score(pil_crop)
        action = self.classify_action(pil_crop)

        # ID 判斷 (純 CLIP 動態特徵)
        dog_name = self.classify_dog_identity(pil_crop)

        # 更新狗狗個別統計

        is_high = score >= self.aesthetic_high
        if is_high:
            self.stats["high_score"] += 1
            target_dir = self.output_dir / "精選照片" / dog_name / age_str / action
        elif score < self.aesthetic_min:
            self.stats["low_score"] += 1
            target_dir = self.output_dir / "低分存檔"
        else:
            target_dir = self.output_dir / dog_name / age_str / action
            
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / file_path.name
        shutil.copy2(file_path, target_path)
        
        # AI 與 Markdown
        caption = self.generate_ollama_caption(file_path, dog_name) if is_high else ""
        self.create_markdown_log(target_path, dog_name, age_str, score, caption)

    def _run_loop(self):
        self.stats = {k: 0 for k in self.stats}
        self.known_hashes = set()
        try:
            if not self.initialize_models(): return
            
            img_exts = {'.jpg', '.jpeg', '.png', '.webp'}
            vid_exts = {'.mp4', '.mov', '.avi'}
            
            all_paths = [p for p in self.source_dir.rglob('*') if p.is_file()]
            images = [p for p in all_paths if p.suffix.lower() in img_exts]
            videos = [p for p in all_paths if p.suffix.lower() in vid_exts]
            
            self.stats["total"] = len(images) + len(videos)
            self.log(f"🚀 開始任務: 圖片 {len(images)}, 影片 {len(videos)}")
            
            # 處理圖片 (批次)
            for i in range(0, len(images), self.batch_size):
                if self.stop_requested: break
                batch = images[i : i + self.batch_size]
                imgs_bgr, v次)
            for i in range(0, len(images), self.batch_size):
                if self.stop_requested: break
                batch = images[i : i + self.batch_size]
                imgs_bgr, valid_p = [], []
                for p in batch:
                    try:
                        data = np.fromfile(str(p), dtype=np.uint8)
                        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                        if img is not None:
                            imgs_bgr.append(img)
                            valid_p.append(p)
                    except: continue
                
                if imgs_bgr:
                    results = self.yolo_model(imgs_bgr, verbose=False)
                    for j, res in enumerate(results):
                        self.process_image(imgs_bgr[j], res, valid_p[j])
                        self.stats["processed"] += 1
                        if self.progress_callback:
                            self.progress_callback(self.stats["processed"], self.stats["total"], f"正在處理: {valid_p[j].name}")

            # 處理影片 (逐一)
            if self.enable_video:
                for v in videos:
                    if self.stop_requested: break
                    has_dog, _ = self.analyze_video(v)
                    if has_dog:
                        self.stats["videos"] += 1
                        target = self.output_dir / "影片_有狗狗"
                        target.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(v, target / v.name)
                    self.stats["processed"] += 1
                    if self.progress_callback:
                        self.progress_callback(self.stats["processed"], self.stats["total"], f"影片分析: {v.name}")

            self.log(f"✅ 任務完成！總數: {self.stats['processed']}, 發現狗狗: {self.stats['dogs_found']}, 精選: {self.stats['high_score']}")
        except Exception as e: self.log(f"❌ 錯誤: {e}", logging.ERROR)
        finally:
            self.is_running = False
            if self.progress_callback: self.progress_callback(self.stats["total"], self.stats["total"], "就緒")

    def start_processing(self):
        if self.is_running: return
        self.is_running = True
        self.stop_requested = False
        threading.Thread(target=self._run_loop, daemon=True).start()

    def stop_processing(self):
        self.stop_requested = True
        self.log("⏳ 正在請求停止...")

    def calculate_aesthetic_score(self, pil_img):
        if not self.clip_model: return 0.5
        try:
            inputs = self.clip_processor(text=["high quality", "low quality"], images=pil_img, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
            return float(outputs.logits_per_image.softmax(dim=1).squeeze().cpu().numpy()[0])
        except: return 0.5

    def classify_age(self, photo_dt):
        birth = self.dog_birth_date
        age_years = photo_dt.year - birth.year - ((photo_dt.month, photo_dt.day) < (birth.month, birth.day))
        if age_years < 1: return "1_幼犬"
        elif age_years < 3: return "2_少年期"
        else: return "3_成年期"

    def generate_ollama_caption(self, image_path, dog_name="狗狗"):
        if not self.enable_ollama: return ""
        try:
            with Image.open(image_path) as img:
                img.thumbnail((512, 512))
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            payload = {"model": self.ollama_model, "prompt": f"描述這張{dog_name}的照片，繁體中文。", "stream": False, "images": [img_base64]}
            req = urllib.request.Request(f"{self.ollama_url}/api/generate", data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode('utf-8')).get('response', '').strip()
        except: return ""

    def create_markdown_log(self, target_path, dog_name, age, score, caption):
        try:
            md_dir = self.output_dir / "Obsidian_Logs" / dog_name
            md_dir.mkdir(parents=True, exist_ok=True)
            stars = "★" * int(score * 5)
            content = f"---\ndog: {dog_name}\nscore: {score:.2f}\n---\n# {target_path.name}\n![[{target_path.name}]]\n\n{caption}"
            with open(md_dir / (target_path.stem + ".md"), 'w', encoding='utf-8') as f: f.write(content)
        except: pass
'w', encoding='utf-8') as f: f.write(content)
        except: pass
 except: pass
'w', encoding='utf-8') as f: f.write(content)
        except: pass
'utf-8') as f: f.write(content)
        except: pass
 except: pass
'w', encoding='utf-8') as f: f.write(content)
        except: pass
