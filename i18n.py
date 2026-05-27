import json
import locale
import os
from pathlib import Path

class I18n:
    def __init__(self, lang=None):
        self.locales_dir = Path(__file__).parent / "locales"
        self.locales_dir.mkdir(exist_ok=True)
        
        if not lang:
            sys_lang, _ = locale.getdefaultlocale()
            lang = "zh_TW" if not sys_lang or sys_lang == "C" or "zh" in sys_lang.lower() else "en"
            
        self.lang = lang
        self.translations = {}
        self.load_translations()

    def load_translations(self):
        file_path = self.locales_dir / f"{self.lang}.json"
        if not file_path.exists():
            # Fallback to zh_TW
            file_path = self.locales_dir / "zh_TW.json"
            
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                self.translations = json.load(f)

    def t(self, key, default=""):
        return self.translations.get(key, default or key)

# 全域單例
_i18n_instance = I18n()

def t(key, default=""):
    return _i18n_instance.t(key, default)
