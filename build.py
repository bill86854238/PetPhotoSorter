import os
import sys
import subprocess
from pathlib import Path

def build():
    print("🚀 開始封裝 Pet Photo Sorter Pro...")
    
    # 確保安裝了 pyinstaller
    try:
        import PyInstaller
    except ImportError:
        print("📥 正在安裝 PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        
    project_dir = Path(__file__).parent
    gui_script = project_dir / "gui.py"
    
    # PyInstaller 參數
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onedir",
        "--windowed", # 不顯示終端機黑框
        "--name", "PetPhotoSorter",
        "--add-data", f"{project_dir / 'locales'}{os.pathsep}locales",
        "--hidden-import", "customtkinter",
        "--hidden-import", "PIL",
        "--hidden-import", "cv2",
        "--hidden-import", "ultralytics",
        "--hidden-import", "transformers",
        str(gui_script)
    ]
    
    # 如果未來有 ICON，可以在這裡加上 "--icon=app.ico"
    # cmd.extend(["--icon", str(project_dir / "app.ico")])
    
    print(f"📦 執行指令: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    
    print("✅ 封裝完成！執行檔位於 'dist/PetPhotoSorter' 資料夾中。")

if __name__ == "__main__":
    build()
