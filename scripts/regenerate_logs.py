import os
import shutil
import torch
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
# from tqdm import tqdm # 移除 tqdm 依賴
import urllib.parse

# 嘗試載入 CLIP (用於美感評分)
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

# =========================
# 設定
# =========================
BASE_DIR = Path(r"E:\Project\PetPhotoSorter")
SOURCE_ROOT = BASE_DIR / "分類"
LOGS_DIR = SOURCE_ROOT / "Obsidian_Logs"

# 評分設定
MIN_SCORE_THRESHOLD = 0.60  # 提高閾值，只選比較好的照片
USE_CLIP_SCORING = True

# Obsidian 設定
VAULT_ROOT_IS_SOURCE = True 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================
# 模型初始化
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
processor = None

def init_clip():
    global model, processor
    if not CLIP_AVAILABLE:
        logging.warning("transformers 套件未安裝，使用預設分數。")
        return False
    
    try:
        logging.info(f"正在載入 CLIP 模型 ({device})...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return True
    except Exception as e:
        logging.error(f"模型載入失敗: {e}")
        return False

# =========================
# 核心功能
# =========================

def calculate_aesthetic_score(image_path):
    if not model or not processor:
        return 1.0

    try:
        image = Image.open(image_path)
        labels = ["a high quality, aesthetic, beautiful photo", "a blurry, dark, low quality, noisy photo"]
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        score = probs[0][0].item() 
        return score
    except Exception as e:
        logging.error(f"評分失敗 {image_path}: {e}")
        return 0.5

def generate_markdown(file_info, output_md_path):
    img_path = file_info['path']
    rel_img_path = ""
    
    try:
        rel_img_path = os.path.relpath(img_path, output_md_path.parent)
        rel_img_path = rel_img_path.replace("\\", "/")
    except ValueError:
        rel_img_path = str(img_path)

    name = file_info['name']
    date_str = file_info['date_str']
    dog_name = file_info['dog']
    action = file_info['action']
    score = file_info['score']
    stars = "★" * int(score * 5)
    
    content = f"""---
date: {date_str}
dog: {dog_name}
action: {action}
score: {score:.2f}
tags: [{dog_name}, {action}]
---

# {name} {stars}

![{name}]({rel_img_path})

> **AI 美感評分: {score:.2f}**

### 📝 摘要
（此處可整合 Ollama 生成的描述）

---
*原始路徑: `{img_path}`*
"""
    
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(content)

def main():
    if USE_CLIP_SCORING:
        if not init_clip():
            print("無法啟用 AI 評分。")

    print(f"開始掃描資料夾: {SOURCE_ROOT}")
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    processed_count = 0
    skipped_count = 0
    
    if not LOGS_DIR.exists():
        LOGS_DIR.mkdir(parents=True)

    targets = ["二季", "四季"]
    entries = []

    for target in targets:
        target_dir = SOURCE_ROOT / target
        if not target_dir.exists(): continue
        
        # 取得所有檔案列表
        all_imgs = list(target_dir.rglob("*"))
        total_imgs = len(all_imgs)
        print(f"找到 {target} 照片共 {total_imgs} 張，開始處理...")

        for idx, img_path in enumerate(all_imgs):
            if img_path.suffix.lower() not in valid_extensions:
                continue
            
            if "Obsidian_Logs" in img_path.parts:
                continue

            # 簡單進度顯示
            if idx % 50 == 0:
                print(f"[{target}] 處理進度: {idx}/{total_imgs}", end="\r")

            if USE_CLIP_SCORING and model:
                score = calculate_aesthetic_score(img_path)
            else:
                score = 1.0 
            
            if score < MIN_SCORE_THRESHOLD:
                skipped_count += 1
                continue

            action = "未分類"
            for part in img_path.parts:
                if "動作_" in part:
                    action = part
                    break
            
            try:
                date_str = img_path.name.split("_")[0]
                dt = datetime.strptime(date_str, "%Y%m%d")
                display_date = dt.strftime("%Y-%m-%d")
                year = dt.strftime("%Y")
            except:
                display_date = "Unknown"
                year = "Unknown"

            entries.append({
                "path": img_path,
                "name": img_path.name,
                "dog": target,
                "action": action,
                "score": score,
                "date_str": display_date,
                "year": year
            })
            processed_count += 1
        print("") # 換行

    print(f"正在生成 {len(entries)} 篇日記...")
    
    entries.sort(key=lambda x: x['name'], reverse=True)

    # 產生 index.md 的內容
    index_content = "# 🐶 成長全紀錄索引\n\n| 日期 | 主角 | 動作 | 評分 | 連結 |\n|---|---|---|---|---|\n"
    
    for i, entry in enumerate(entries):
        if i % 100 == 0:
             print(f"寫入進度: {i}/{len(entries)}", end="\r")

        md_dir = LOGS_DIR / entry['dog'] / entry['year']
        md_dir.mkdir(parents=True, exist_ok=True)
        
        md_filename = Path(entry['name']).with_suffix('.md')
        md_path = md_dir / md_filename
        
        generate_markdown(entry, md_path)
        
        rel_link = f"{entry['dog']}/{entry['year']}/{md_filename}"
        stars = "★" * int(entry['score'] * 5)
        index_content += f"| {entry['date_str']} | {entry['dog']} | {entry['action']} | {entry['score']:.2f} {stars} | [{entry['name']}]({rel_link}) |\n"

    with open(LOGS_DIR / "index.md", "w", encoding="utf-8") as f:
        f.write(index_content)

    print(f"\n處理完成！")
    print(f"- 已生成: {processed_count} 篇")
    print(f"- 因低分略過: {skipped_count} 篇")
    print(f"- 索引檔: {LOGS_DIR / 'index.md'}")
    print(f"\n【重要提示】")
    print(f"請在 Obsidian 中開啟 '{SOURCE_ROOT}' 資料夾作為 Vault。")

if __name__ == "__main__":
    main()
