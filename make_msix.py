import os
import shutil
import subprocess
from pathlib import Path

def make_msix():
    project_dir = Path(__file__).parent
    dist_dir = project_dir / "dist" / "PetPhotoSorter"
    build_msix_dir = project_dir / "msix_build"
    
    print("🚀 開始 MSIX 封裝流程...")
    
    # 1. 檢查 PyInstaller 產物
    if not dist_dir.exists():
        print("❌ 錯誤：找不到 dist/PetPhotoSorter。請先執行 python build.py")
        return

    # 2. 清理並建立 build 目錄
    if build_msix_dir.exists():
        shutil.rmtree(build_msix_dir)
    build_msix_dir.mkdir()
    
    # 3. 複製 exe 產物到 build 目錄
    print("📂 正在準備檔案...")
    # 注意：MSIX 要求所有檔案都在根目錄或子目錄
    shutil.copytree(dist_dir, build_msix_dir, dirs_exist_ok=True)
    
    # 4. 建立 Assets 資料夾並放入占位圖
    assets_dir = build_msix_dir / "Assets"
    assets_dir.mkdir(exist_ok=True)
    # (這裡未來會放置真正的圖示，現在先略過或建立空白檔)
    for img in ["StoreLogo.png", "Square150x150Logo.png", "Square44x44Logo.png", "Wide310x150Logo.png"]:
        (assets_dir / img).touch() 

    # 5. 生成 AppxManifest.xml
    print("📝 生成 Manifest...")
    with open(project_dir / "AppxManifest.xml.template", "r", encoding="utf-8") as f:
        manifest_content = f.read()
    
    # 您可以在這裡做字串替換，例如替換版本號
    with open(build_msix_dir / "AppxManifest.xml", "w", encoding="utf-8") as f:
        f.write(manifest_content)

    # 6. 尋找 MakeAppx.exe (從 Windows SDK 路徑)
    # 常見路徑：C:\Program Files (x86)\Windows Kits\10\bin\<version>\x64\makeappx.exe
    sdk_bin_root = Path(r"C:\Program Files (x86)\Windows Kits\10\bin")
    makeappx_exe = None
    if sdk_bin_root.exists():
        versions = sorted([d.name for d in sdk_bin_root.iterdir() if d.is_dir()], reverse=True)
        for v in versions:
            potential_path = sdk_bin_root / v / "x64" / "makeappx.exe"
            if potential_path.exists():
                makeappx_exe = potential_path
                break
    
    if not makeappx_exe:
        print("⚠️ 找不到 MakeAppx.exe！請確保已安裝 Windows SDK。")
        print("您也可以手動使用 MSIX Packaging Tool 分裝 msix_build 資料夾。")
        return

    # 7. 執行打包
    output_msix = project_dir / "PetPhotoSorter.msix"
    print(f"📦 正在執行 MakeAppx 打包成 {output_msix}...")
    
    try:
        cmd = [
            str(makeappx_exe),
            "pack",
            "/d", str(build_msix_dir),
            "/p", str(output_msix),
            "/o" # 覆蓋現有檔案
        ]
        subprocess.check_call(cmd)
        print(f"✅ MSIX 打包完成：{output_msix}")
        print("💡 提示：提交到微軟商店時直接上傳此檔案即可。")
    except Exception as e:
        print(f"❌ 打包失敗: {e}")

if __name__ == "__main__":
    make_msix()
