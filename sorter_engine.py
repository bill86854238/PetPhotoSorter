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
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.is_running = False
        self.stop_requested = False
        
        self.yolo_model = None
        self.clip_model = None
        self.clip_processor = None
        self.device = "cpu"
        self.known_hashes = set()
        
        self.stats = {
            "processed": 0,
            "total": 0,
            "high_score": 0,
            "low_score": 0,
            "dogs_found": 0,
            "videos": 0,
            "duplicates": 0,
            "dog_counts": {},
            "daily_summary": {},
            "folder_summary": {},
            "hierarchy_summary": {}
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
        
        self.batch_size = settings.get("batch_size", 16)
        self.device_pref = settings.get("device", "mps" if platform.system() == "Darwin" else "cpu")
        self.aesthetic_min = settings.get("aesthetic_score_min", 0.2)
        self.aesthetic_high = settings.get("aesthetic_score_high", 0.6)
        self.enable_ollama = settings.get("enable_ollama", True)
        self.ollama_url = settings.get("ollama_url", "http://localhost:11434")
        self.ollama_model = settings.get("ollama_model", "moondream")
        
        self.dog1_name = settings.get("dog1_name", "暗色狗狗")
        self.dog1_feature = settings.get("dog1_feature", "dark fur")
        self.dog2_name = settings.get("dog2_name", "亮色狗狗")
        self.dog2_feature = settings.get("dog2_feature", "light fur")
        self.brightness_threshold = settings.get("brightness_threshold", 185)
        self.dog_birth_date = datetime(2022, 11, 23)
        
        # "both", "images", "videos"
        self.media_type = settings.get("media_type", "both")
        self.enable_duplicate_check = settings.get("enable_duplicate_check", True)
        self.enable_action_classify = settings.get("enable_action_classify", True)
        self.copy_files = settings.get("copy_files", False)

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
            "media_type": self.media_type,
            "enable_duplicate_check": self.enable_duplicate_check,
            "enable_action_classify": self.enable_action_classify
        })
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        except: return False

    def initialize_models(self):
        if self.yolo_model is not None: return True
        if YOLO is None: return False
        self.device = self.device_pref
        if self.device == "mps" and not torch.backends.mps.is_available(): self.device = "cpu"
        elif self.device == "cuda" and not torch.cuda.is_available(): self.device = "cpu"
        
        self.log(f"🧠 正在載入 AI 模型 (裝置: {self.device})...")
        try:
            self.yolo_model = YOLO("yolov8n.pt").to(self.device)
            if CLIPModel is not None:
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            return True
        except Exception as e:
            self.log(f"❌ 模型初始化失敗: {e}", logging.ERROR)
            return False

    def calculate_ahash(self, img_bgr):
        try:
            resized = cv2.resize(img_bgr, (8, 8), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            avg = gray.mean()
            hash_val = sum([1 << i for i, v in enumerate(gray.flatten()) if v > avg])
            return f"{hash_val:016x}"
        except: return None

    def classify_action(self, pil_img, target_type="dog"):
        if not self.clip_model or not self.enable_action_classify: return "動作_一般"
        
        if target_type == "dog":
            actions = ["sleeping", "playing with a toy", "lying down", "standing or walking", "sitting"]
            labels = ["睡覺", "玩玩具", "趴著", "站立/走動", "坐下"]
        else:
            actions = ["walking", "sitting", "interacting with a dog", "standing"]
            labels = ["走動", "坐下", "與狗互動", "站立"]
            
        try:
            prompt_prefix = "A dog is" if target_type == "dog" else "A person is"
            inputs = self.clip_processor(text=[f"{prompt_prefix} {a}" for a in actions], images=pil_img, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                probs = self.clip_model(**inputs).logits_per_image.softmax(dim=1).cpu().numpy()[0]
            return labels[np.argmax(probs)]
        except: return "一般"

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): return False, "無法讀取"
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

    def process_image(self, img_bgr, yolo_result, file_path):
        if len(yolo_result.boxes) == 0: return
        self.stats["dogs_found"] += 1

        if self.enable_duplicate_check:
            h = self.calculate_ahash(img_bgr)
            if h in self.known_hashes:
                self.stats["duplicates"] += 1
                return
            self.known_hashes.add(h)

        photo_dt = datetime.fromtimestamp(os.path.getmtime(file_path))
        age_str = self.classify_age(photo_dt)

        x1, y1, x2, y2 = map(int, yolo_result.boxes[0].xyxy[0].tolist())
        dog_crop = img_bgr[max(0, y1-20):y2+20, max(0, x1-20):x2+20]
        pil_crop = Image.fromarray(cv2.cvtColor(dog_crop if dog_crop.size > 0 else img_bgr, cv2.COLOR_BGR2RGB))

        score = self.calculate_aesthetic_score(pil_crop)
        action = self.classify_action(pil_crop)
        is_high = score >= self.aesthetic_high
        dog_name = self.classify_dog_identity(pil_crop)

        date_key = photo_dt.strftime('%Y-%m-%d')
        if date_key not in self.stats["daily_summary"]:
            self.stats["daily_summary"][date_key] = {"count": 0, "high": 0}
        self.stats["daily_summary"][date_key]["count"] += 1
        if is_high: self.stats["daily_summary"][date_key]["high"] += 1

        folder_key = action
        if folder_key not in self.stats["folder_summary"]:
            self.stats["folder_summary"][folder_key] = {"count": 0, "score_sum": 0.0}
        self.stats["folder_summary"][folder_key]["count"] += 1
        self.stats["folder_summary"][folder_key]["score_sum"] += score

        path_key = f"{dog_name}/{age_str}/{action}"
        if path_key not in self.stats["hierarchy_summary"]:
            self.stats["hierarchy_summary"][path_key] = {"count": 0, "high": 0, "score_sum": 0.0}
        h = self.stats["hierarchy_summary"][path_key]
        h["count"] += 1
        h["score_sum"] += score
        if is_high: h["high"] += 1

        self.stats["dog_counts"][dog_name] = self.stats["dog_counts"].get(dog_name, 0) + 1

        if is_high:
            self.stats["high_score"] += 1
        elif score < self.aesthetic_min:
            self.stats["low_score"] += 1

        if getattr(self, "test_mode", False) or not self.copy_files:
            self.log(f"🐶 {file_path.name} | {dog_name} | {action} | 分數: {score:.2f}{'  ★精選' if is_high else ''}")
            return

        if is_high:
            target_dir = self.output_dir / "精選照片" / dog_name / age_str / action
        elif score < self.aesthetic_min:
            target_dir = self.output_dir / "低分存檔"
        else:
            target_dir = self.output_dir / dog_name / age_str / action

        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / file_path.name
        shutil.move(str(file_path), target_path)

        caption = self.generate_ollama_caption(file_path, dog_name) if is_high else ""
        self.create_markdown_log(target_path, dog_name, age_str, score, caption)

    def classify_dog_identity(self, pil_img):
        if not self.clip_model: return "狗狗"
        try:
            prompts = [
                f"A dog named {self.dog1_name} which is {self.dog1_feature}",
                f"A dog named {self.dog2_name} which is {self.dog2_feature}"
            ]
            inputs = self.clip_processor(text=prompts, images=pil_img, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                probs = self.clip_model(**inputs).logits_per_image.softmax(dim=1).cpu().numpy()[0]
            return self.dog1_name if probs[0] > probs[1] else self.dog2_name
        except: return "狗狗"

    def generate_html_dashboard(self):
        try:
            output_file = self.output_dir / "dashboard.html"
            daily_data = json.dumps(self.stats["daily_summary"])
            hierarchy_data = json.dumps(self.stats["hierarchy_summary"])
            dog_data = json.dumps(self.stats["dog_counts"])
            
            html_template = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pet Photo Sorter - AI 數據儀表板</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; margin: 0; padding: 20px; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .card h3 {{ margin-top: 0; color: #2c3e50; border-left: 5px solid #3498db; padding-left: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #eee; }}
        th {{ background-color: #f8f9fa; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; background: #e1f5fe; color: #0288d1; }}
        .path-text {{ font-family: 'Consolas', monospace; color: #666; font-size: 13px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🐶 Pet Sorter AI Dashboard</h1>
            <p>生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </header>

        <div class="card-grid">
            <div class="card">
                <h3>📊 任務概況</h3>
                <div style="display: flex; justify-content: space-around; text-align: center;">
                    <div><p>總處理</p><p class="stat-value">{self.stats['processed']}</p></div>
                    <div><p>精選數</p><p class="stat-value" style="color: #f1c40f;">{self.stats['high_score']}</p></div>
                    <div><p>發現狗狗</p><p class="stat-value" style="color: #2ecc71;">{self.stats['dogs_found']}</p></div>
                </div>
            </div>
            <div class="card">
                <h3>🐕 狗狗比例</h3>
                <canvas id="dogChart" height="150"></canvas>
            </div>
        </div>

        <div class="card">
            <h3>📈 處理趨勢與活動亮點</h3>
            <canvas id="dailyChart" height="100"></canvas>
        </div>

        <div style="margin-top: 30px;">
            <div class="card">
                <h3>靈活資料夾摘要 (分類品質分析)</h3>
                <table id="hierarchyTable">
                    <thead><tr><th>歸類路徑 (狗狗/年齡/動作)</th><th>檔案總數</th><th>精選數</th><th>平均評分</th></tr></thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        const dailyData = {daily_data};
        const hierarchyData = {hierarchy_data};
        const dogData = {dog_data};

        new Chart(document.getElementById('dogChart'), {{
            type: 'doughnut',
            data: {{
                labels: Object.keys(dogData),
                datasets: [{{
                    data: Object.values(dogData),
                    backgroundColor: ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6']
                }}]
            }}
        }});

        const dates = Object.keys(dailyData).sort();
        new Chart(document.getElementById('dailyChart'), {{
            type: 'line',
            data: {{
                labels: dates,
                datasets: [
                    {{ label: '處理張數', data: dates.map(d => dailyData[d].count), borderColor: '#3498db', tension: 0.1 }},
                    {{ label: '精選張數', data: dates.map(d => dailyData[d].high), borderColor: '#f1c40f', tension: 0.1 }}
                ]
            }}
        }});

        const hierarchyBody = document.querySelector('#hierarchyTable tbody');
        for (const [path, info] of Object.entries(hierarchyData)) {{
            const avg = (info.score_sum / info.count).toFixed(2);
            hierarchyBody.innerHTML += `
                <tr>
                    <td class="path-text">${{path}}</td>
                    <td>${{info.count}}</td>
                    <td><span class="badge" style="background: #fff9c4; color: #f57f17;">✨ ${{info.high}}</span></td>
                    <td>⭐ ${{avg}}</td>
                </tr>`;
        }}
    </script>
</body>
</html>
            """
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_template)
            self.log(f"📊 靈活儀表板已生成: {output_file.name}")
        except Exception as e:
            self.log(f"生成儀表板失敗: {e}", logging.ERROR)

    def _run_loop(self):
        self.stats = {
            "processed": 0, "total": 0, "high_score": 0, "low_score": 0,
            "dogs_found": 0, "videos": 0, "duplicates": 0,
            "dog_counts": {}, "daily_summary": {}, "folder_summary": {}, "hierarchy_summary": {}
        }
        self.known_hashes = set()
        
        try:
            src_str = str(self.source_dir)
            test_mode = getattr(self, "test_mode", False)
            media_type = getattr(self, "media_type", "both")

            if test_mode:
                self.log(f"👁️ 啟動執行 [預覽模式 - 僅分析不移動檔案]，來源: {src_str}")
            else:
                self.log(f"🔍 啟動執行，來源: {src_str}")
            
            if not self.initialize_models(): return
            
            try:
                source_exists = self.source_dir.exists()
            except Exception as e:
                self.log(f"❌ 無法存取來源路徑: {src_str}\n這通常是網路磁碟未連線或路徑無效。", logging.ERROR)
                return

            if not source_exists:
                self.log(f"❌ 找不到來源目錄: {src_str}\n請確認路徑設定是否正確。", logging.ERROR)
                return

            self.log("🔍 正在掃描檔案...")
            img_exts = {'.jpg', '.jpeg', '.png', '.webp'}
            vid_exts = {'.mp4', '.mov', '.avi'}
            
            try:
                all_paths = [p for p in self.source_dir.rglob('*') if p.is_file()]
            except Exception as e:
                self.log(f"❌ 掃描過程出錯 (網路可能不穩定): {e}", logging.ERROR)
                return

            images = [p for p in all_paths if p.suffix.lower() in img_exts]
            videos = [p for p in all_paths if p.suffix.lower() in vid_exts]

            process_images = media_type in ("images", "both")
            process_videos = media_type in ("videos", "both") and not test_mode

            self.stats["total"] = (len(images) if process_images else 0) + (len(videos) if process_videos else 0)
            if self.stats["total"] == 0:
                self.log("⚠️ 找不到可處理的檔案。", logging.WARNING)
                return

            log_parts = []
            if process_images: log_parts.append(f"{len(images)} 張圖片")
            if process_videos: log_parts.append(f"{len(videos)} 個影片")
            self.log(f"🚀 開始任務: {', '.join(log_parts)}")

            # 處理圖片
            if process_images:
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
                                self.progress_callback(self.stats["processed"], self.stats["total"], f"分析中: {valid_p[j].name}")

            # 處理影片
            if process_videos:
                for v in videos:
                    if self.stop_requested: break

                    has_dog, _ = self.analyze_video(v)
                    if has_dog:
                        self.stats["videos"] += 1
                        age_str = self.classify_age(datetime.fromtimestamp(v.stat().st_mtime))

                        # 推測身份與動作（僅記錄，不影響資料夾路徑）
                        identity, action = "狗狗", "一般"
                        if self.clip_model:
                            try:
                                cap = cv2.VideoCapture(str(v))
                                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                cap.set(cv2.CAP_PROP_POS_FRAMES, total_f // 2)
                                ret, frame = cap.read()
                                cap.release()
                                if ret:
                                    res = self.yolo_model(frame, verbose=False)
                                    if len(res[0].boxes) > 0:
                                        box = res[0].boxes[0]
                                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                        crop = frame[max(0, y1-10):y2+10, max(0, x1-10):x2+10]
                                        if crop.size > 0:
                                            pil_c = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                            identity = self.classify_dog_identity(pil_c)
                                            action = self.classify_action(pil_c)
                            except Exception: pass

                        self.log(f"📹 {v.name} | {age_str} | 推測: {identity} | 動作: {action}")

                        if self.copy_files:
                            try:
                                target = self.output_dir / "影片" / age_str
                                target.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(v, target / v.name)
                            except Exception as e:
                                self.log(f"⚠️ 影片複製失敗: {e}", logging.WARNING)

                    self.stats["processed"] += 1
                    if self.progress_callback:
                        self.progress_callback(self.stats["processed"], self.stats["total"], f"影片分析: {v.name}")

            self.log(f"✅ 任務完成！處理總數: {self.stats['processed']}")
            
            if not test_mode:
                try: self.generate_html_dashboard()
                except: pass
        except Exception as e:
            try: self.log(f"❌ 執行中斷: {e}", logging.ERROR)
            except: print(f"CRITICAL ERROR: {e}")
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
            content = f"---\ndog: {dog_name}\nscore: {score:.2f}\nstars: {stars}\n---\n# {target_path.name}\n![[{target_path.name}]]\n\n{caption}"
            with open(md_dir / (target_path.stem + ".md"), 'w', encoding='utf-8') as f: f.write(content)
        except: pass
