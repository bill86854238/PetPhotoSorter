import logging
import platform
from pathlib import Path

# 僅在 macOS 上導入相關庫
if platform.system() == "Darwin":
    try:
        import Vision
        import Quartz
        from Cocoa import NSURL
        from osxmetadata import OSXMetaData
        MAC_DEPENDENCIES_OK = True
    except ImportError:
        MAC_DEPENDENCIES_OK = False
else:
    MAC_DEPENDENCIES_OK = False

def calculate_aesthetic_score(image_path):
    """
    使用 macOS 原生 Vision Framework 計算圖片美感分數 (0.0 - 1.0)
    回傳: float 分數 (若失敗則回傳 -1.0)
    """
    if not MAC_DEPENDENCIES_OK:
        return -1.0

    try:
        from pathlib import Path
        from Cocoa import NSURL
        import Vision

        # 1. 確保路徑為絕對路徑 (使用者要求)
        abs_path = str(Path(image_path).resolve())
        file_url = NSURL.fileURLWithPath_(abs_path)

        # 2. 建立請求
        request = Vision.VNGenerateImageAestheticsScoresRequest.alloc().init()
        
        # 3. 建立 Handler 並執行 (使用 URL)
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(file_url, None)
        
        # 4. 捕捉詳細錯誤 (使用者要求)
        success, error = handler.performRequests_error_([request], None)

        if success:
            results = request.results()
            if results:
                score = results[0].overallScore()
                return float(score)
        else:
            if error:
                # 取得詳細的本地化描述
                error_msg = error.localizedDescription()
                logging.warning(f"Vision 執行失敗 ({image_path.name}): {error_msg}")
            else:
                logging.warning(f"Vision 執行失敗 ({image_path.name}): 未知錯誤")
                
    except Exception as e:
        logging.warning(f"美感評分系統異常: {e}")
    
    return -1.0

def write_finder_metadata(file_path, comment=None, tags=None, clear_previous=False):
    """
    寫入 macOS Finder 的註解與顏色標籤
    """
    if not MAC_DEPENDENCIES_OK:
        return

    try:
        md = OSXMetaData(str(file_path))
        
        if clear_previous:
            md.tags = []
            md.finder_comment = ""
            
        if comment:
            # 保留原有註解，或是覆蓋
            current_comment = md.finder_comment or ""
            if comment not in current_comment:
                md.finder_comment = f"{comment} | {current_comment}" if current_comment else comment
                
        if tags:
            # tags 是一個 list，例如 ["Blue", "Red"]
            # osxmetadata 會自動處理顏色名稱對應
            existing_tags = set(md.tags)
            for t in tags:
                existing_tags.add(t)
            md.tags = list(existing_tags)
            
    except Exception as e:
        msg = str(e)
        if "Invalid attribute" in msg or "Operation not supported" in msg:
            # 這是 NAS 不支援 xattr 的常見錯誤，不需要一直報錯
            pass 
        else:
            logging.error(f"寫入 Finder Metadata 失敗: {e}")
