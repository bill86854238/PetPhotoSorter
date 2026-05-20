import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import logging
from sorter_engine import SorterEngine
import os
from pathlib import Path
import urllib.request
import json
import platform
from i18n import t

# 設定外觀
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class PetPhotoSorterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(t("app_title"))
        self.geometry("1100x800")

        # 初始化引擎
        self.engine = SorterEngine(
            progress_callback=self.update_progress,
            log_callback=self.append_log
        )

        # 建立 UI 佈局
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- 側邊欄 (Sidebar) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="🐶 Pet Sorter AI", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.start_button = ctk.CTkButton(self.sidebar_frame, text=t("btn_start"), command=self.start_worker, height=45, font=ctk.CTkFont(size=16, weight="bold"))
        self.start_button.grid(row=1, column=0, padx=20, pady=10)
        
        self.stop_button = ctk.CTkButton(self.sidebar_frame, text=t("btn_stop"), command=self.stop_worker, fg_color="#E74C3C", hover_color="#C0392B")
        self.stop_button.grid(row=2, column=0, padx=20, pady=10)

        self.save_button = ctk.CTkButton(self.sidebar_frame, text=t("btn_save"), command=self.save_settings, fg_color="#2ECC71", hover_color="#27AE60")
        self.save_button.grid(row=3, column=0, padx=20, pady=10)

        self.sep = ctk.CTkLabel(self.sidebar_frame, text="—" * 15, text_color="gray")
        self.sep.grid(row=4, column=0, pady=5)

        # AI 快速開關
        self.surveillance_switch = ctk.CTkSwitch(self.sidebar_frame, text=t("lbl_mode_surveillance"))
        self.surveillance_switch.grid(row=5, column=0, padx=20, pady=10, sticky="w")
        
        self.test_mode_switch = ctk.CTkSwitch(self.sidebar_frame, text=t("lbl_mode_test"))
        self.test_mode_switch.grid(row=6, column=0, padx=20, pady=10, sticky="w")

        self.ollama_switch = ctk.CTkSwitch(self.sidebar_frame, text="Ollama AI", command=self.toggle_ollama_ui)
        self.ollama_switch.grid(row=7, column=0, padx=20, pady=10, sticky="w")
        if self.engine.enable_ollama: self.ollama_switch.select()

        self.action_switch = ctk.CTkSwitch(self.sidebar_frame, text="CLIP Action")
        self.action_switch.grid(row=8, column=0, padx=20, pady=10, sticky="w")
        self.action_switch.select()

        # 主題切換
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"], command=self.change_appearance_mode)
        self.appearance_mode_optionemenu.grid(row=11, column=0, padx=20, pady=(10, 20))
        self.appearance_mode_optionemenu.set("System")

        # --- 主面板 (Main Panel) ---
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.tabview.add(t("tab_status"))
        self.tabview.add(t("tab_ai"))
        self.tabview.add(t("tab_path"))

        # --- 頁籤 1: 執行狀態 ---
        self.status_tab = self.tabview.tab(t("tab_status"))
        self.status_tab.grid_columnconfigure(0, weight=1)

        self.progress_label = ctk.CTkLabel(self.status_tab, text=t("lbl_status_ready"), font=ctk.CTkFont(size=15))
        self.progress_label.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.progressbar = ctk.CTkProgressBar(self.status_tab)
        self.progressbar.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="ew")
        self.progressbar.set(0)

        # 統計儀表板
        self.stats_frame = ctk.CTkFrame(self.status_tab)
        self.stats_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.stats_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.stat_high = ctk.CTkLabel(self.stats_frame, text=f"{t('lbl_stats_high')} 0", font=ctk.CTkFont(size=13))
        self.stat_high.grid(row=0, column=0, padx=10, pady=10)
        self.stat_dogs = ctk.CTkLabel(self.stats_frame, text=f"{t('lbl_stats_dogs')} 0", font=ctk.CTkFont(size=13))
        self.stat_dogs.grid(row=0, column=1, padx=10, pady=10)
        self.stat_vids = ctk.CTkLabel(self.stats_frame, text=f"{t('lbl_stats_vids')} 0", font=ctk.CTkFont(size=13))
        self.stat_vids.grid(row=0, column=2, padx=10, pady=10)
        self.stat_dups = ctk.CTkLabel(self.stats_frame, text=f"{t('lbl_stats_dups')} 0", font=ctk.CTkFont(size=13))
        self.stat_dups.grid(row=0, column=3, padx=10, pady=10)

        # 比例視覺化
        self.ratio_frame = ctk.CTkFrame(self.status_tab, height=30)
        self.ratio_frame.grid(row=3, column=0, padx=20, pady=5, sticky="ew")
        self.ratio_label = ctk.CTkLabel(self.ratio_frame, text=t("lbl_ratio"), font=ctk.CTkFont(size=12))
        self.ratio_label.pack(pady=5)

        # 結果操作按鈕
        self.results_action_frame = ctk.CTkFrame(self.status_tab, fg_color="transparent")
        self.results_action_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        self.results_action_frame.grid_columnconfigure((0, 1), weight=1)

        self.open_out_btn = ctk.CTkButton(self.results_action_frame, text=t("btn_open_out"), command=self.open_output_folder, state="disabled", fg_color="gray")
        self.open_out_btn.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        self.open_obsidian_btn = ctk.CTkButton(self.results_action_frame, text=t("btn_open_obs"), command=self.open_obsidian_logs, state="disabled", fg_color="gray")
        self.open_obsidian_btn.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        self.log_textbox = ctk.CTkTextbox(self.status_tab, font=ctk.CTkFont(family="Consolas", size=13))
        self.log_textbox.grid(row=5, column=0, padx=20, pady=10, sticky="nsew")
        self.status_tab.grid_rowconfigure(5, weight=1)

        # --- 頁籤 2: AI 設定 ---
        self.ai_tab = self.tabview.tab(t("tab_ai"))
        
        self.ollama_frame = ctk.CTkFrame(self.ai_tab)
        self.ollama_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(self.ollama_frame, text=t("lbl_ollama_setting"), font=ctk.CTkFont(weight="bold", size=16)).pack(pady=10)
        ctk.CTkLabel(self.ollama_frame, text=t("lbl_ollama_desc"), justify="left", text_color="gray").pack(padx=20, pady=5)
        
        url_row = ctk.CTkFrame(self.ollama_frame, fg_color="transparent")
        url_row.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(url_row, text=t("lbl_api_url")).pack(side="left", padx=5)
        self.ollama_url_entry = ctk.CTkEntry(url_row, width=250)
        self.ollama_url_entry.pack(side="left", padx=5)
        self.ollama_url_entry.insert(0, self.engine.ollama_url)

        model_row = ctk.CTkFrame(self.ollama_frame, fg_color="transparent")
        model_row.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(model_row, text=t("lbl_model_name")).pack(side="left", padx=5)
        self.ollama_model_entry = ctk.CTkEntry(model_row, width=200)
        self.ollama_model_entry.pack(side="left", padx=5)
        self.ollama_model_entry.insert(0, self.engine.ollama_model)
        
        self.test_ollama_btn = ctk.CTkButton(model_row, text=t("btn_test_conn"), width=100, command=self.test_ollama_connection)
        self.test_ollama_btn.pack(side="left", padx=10)

        self.score_frame = ctk.CTkFrame(self.ai_tab)
        self.score_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(self.score_frame, text=t("lbl_score_setting"), font=ctk.CTkFont(weight="bold", size=16)).pack(pady=10)
        
        self.high_score_label = ctk.CTkLabel(self.score_frame, text=f"{t('lbl_score_thresh')} {self.engine.aesthetic_high:.2f}")
        self.high_score_label.pack()
        self.high_score_slider = ctk.CTkSlider(self.score_frame, from_=0, to=1, command=self.update_score_labels)
        self.high_score_slider.set(self.engine.aesthetic_high)
        self.high_score_slider.pack(padx=40, pady=10, fill="x")

        # 狗狗名稱設定 (新增特徵輸入)
        self.dog_label_frame = ctk.CTkFrame(self.ai_tab)
        self.dog_label_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(self.dog_label_frame, text=t("lbl_dog_setting"), font=ctk.CTkFont(weight="bold", size=16)).pack(pady=10)
        
        dog_row1 = ctk.CTkFrame(self.dog_label_frame, fg_color="transparent")
        dog_row1.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(dog_row1, text=t("lbl_dog1_name")).pack(side="left", padx=5)
        self.dog1_name_entry = ctk.CTkEntry(dog_row1, width=100)
        self.dog1_name_entry.pack(side="left", padx=5)
        self.dog1_name_entry.insert(0, self.engine.dog1_name)

        ctk.CTkLabel(dog_row1, text=t("lbl_dog1_feat")).pack(side="left", padx=15)
        self.dog1_feat_entry = ctk.CTkEntry(dog_row1, width=200)
        self.dog1_feat_entry.pack(side="left", padx=5)
        self.dog1_feat_entry.insert(0, self.engine.dog1_feature)

        dog_row2 = ctk.CTkFrame(self.dog_label_frame, fg_color="transparent")
        dog_row2.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(dog_row2, text=t("lbl_dog2_name")).pack(side="left", padx=5)
        self.dog2_name_entry = ctk.CTkEntry(dog_row2, width=100)
        self.dog2_name_entry.pack(side="left", padx=5)
        self.dog2_name_entry.insert(0, self.engine.dog2_name)

        ctk.CTkLabel(dog_row2, text=t("lbl_dog2_feat")).pack(side="left", padx=15)
        self.dog2_feat_entry = ctk.CTkEntry(dog_row2, width=200)
        self.dog2_feat_entry.pack(side="left", padx=5)
        self.dog2_feat_entry.insert(0, self.engine.dog2_feature)

        # --- 頁籤 3: 路徑設定 ---
        self.path_tab = self.tabview.tab(t("tab_path"))
        self.path_container = ctk.CTkFrame(self.path_tab)
        self.path_container.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(self.path_container, text=t("lbl_src_dir")).pack(anchor="w", padx=20, pady=(20, 5))
        self.src_entry = ctk.CTkEntry(self.path_container)
        self.src_entry.pack(fill="x", padx=20, pady=5)
        self.src_entry.insert(0, str(self.engine.source_dir))
        ctk.CTkButton(self.path_container, text=t("btn_browse"), command=lambda: self.browse_folder(self.src_entry)).pack(padx=20, pady=5)

        ctk.CTkLabel(self.path_container, text=t("lbl_out_dir")).pack(anchor="w", padx=20, pady=(20, 5))
        self.out_entry = ctk.CTkEntry(self.path_container)
        self.out_entry.pack(fill="x", padx=20, pady=5)
        self.out_entry.insert(0, str(self.engine.output_dir))
        ctk.CTkButton(self.path_container, text=t("btn_browse"), command=lambda: self.browse_folder(self.out_entry)).pack(padx=20, pady=5)

        ctk.CTkLabel(self.path_container, text=t("lbl_perf_setting")).pack(anchor="w", padx=20, pady=(20, 5))
        batch_row = ctk.CTkFrame(self.path_container, fg_color="transparent")
        batch_row.pack(fill="x", padx=20)
        ctk.CTkLabel(batch_row, text=t("lbl_batch_size")).pack(side="left")
        self.batch_entry = ctk.CTkEntry(batch_row, width=80)
        self.batch_entry.pack(side="left", padx=10)
        self.batch_entry.insert(0, str(self.engine.batch_size))

    def test_ollama_connection(self):
        url = self.ollama_url_entry.get().strip("/")
        model = self.ollama_model_entry.get()
        self.test_ollama_btn.configure(state="disabled", text="Testing...")
        def run_test():
            try:
                with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        data = json.loads(response.read().decode())
                        models = [m['name'] for m in data.get('models', [])]
                        if any(model in m for m in models):
                            self.append_log(f"✅ Connection Success ({model})", logging.INFO)
                            self.after(0, lambda: messagebox.showinfo("OK", f"Model Found: {model}"))
                        else:
                            self.append_log(f"⚠️ Model not found: {model}", logging.WARNING)
            except Exception as e:
                self.append_log(f"❌ Connection Failed: {e}", logging.ERROR)
            finally:
                self.after(0, lambda: self.test_ollama_btn.configure(state="normal", text=t("btn_test_conn")))
        threading.Thread(target=run_test, daemon=True).start()

    def save_settings(self):
        self.engine.source_dir = Path(self.src_entry.get())
        self.engine.output_dir = Path(self.out_entry.get())
        self.engine.enable_ollama = self.ollama_switch.get()
        self.engine.ollama_url = self.ollama_url_entry.get().strip("/")
        self.engine.ollama_model = self.ollama_model_entry.get()
        
        # 儲存特徵
        self.engine.dog1_name = self.dog1_name_entry.get()
        self.engine.dog1_feature = self.dog1_feat_entry.get()
        self.engine.dog2_name = self.dog2_name_entry.get()
        self.engine.dog2_feature = self.dog2_feat_entry.get()
        
        self.engine.aesthetic_high = self.high_score_slider.get()
        try: self.engine.batch_size = int(self.batch_entry.get())
        except: pass
        if self.engine.save_config_to_file():
            messagebox.showinfo("OK", t("msg_save_success"))
        else:
            messagebox.showerror("Error", t("msg_save_fail"))

    def toggle_ollama_ui(self):
        self.ollama_frame.configure(border_width=2 if self.ollama_switch.get() else 0)

    def browse_folder(self, entry_widget):
        path = filedialog.askdirectory()
        if path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, path)

    def update_score_labels(self, value):
        self.high_score_label.configure(text=f"{t('lbl_score_thresh')} {value:.2f}")

    def change_appearance_mode(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def start_worker(self):
        self.engine.source_dir = Path(self.src_entry.get())
        self.engine.output_dir = Path(self.out_entry.get())
        self.engine.surveillance_mode = self.surveillance_switch.get()
        self.engine.test_mode = self.test_mode_switch.get() # 同步測試模式狀態
        self.open_out_btn.configure(state="disabled", fg_color="gray")
        self.open_obsidian_btn.configure(state="disabled", fg_color="gray")
        self.start_button.configure(state="disabled")
        self.log_textbox.delete("1.0", tk.END)
        self.engine.start_processing()

    def stop_worker(self):
        self.engine.stop_processing()

    def update_progress(self, current, total, message):
        self.after(0, self._update_progress_ui, current, total, message)

    def _update_progress_ui(self, current, total, message):
        if total > 0: self.progressbar.set(current / total)
        self.progress_label.configure(text=f"Status: {message} ({current}/{total})")
        
        s = self.engine.stats
        self.stat_high.configure(text=f"{t('lbl_stats_high')} {s['high_score']}")
        self.stat_dogs.configure(text=f"{t('lbl_stats_dogs')} {s['dogs_found']}")
        self.stat_vids.configure(text=f"{t('lbl_stats_vids')} {s['videos']}")
        self.stat_dups.configure(text=f"{t('lbl_stats_dups')} {s['duplicates']}")

        counts = s['dog_counts']
        if counts:
            total_dogs = sum(counts.values())
            ratio_text = t("lbl_ratio").split(":")[0] + ": " + " | ".join([f"{n}: {c} ({c/total_dogs:.1%})" for n, c in counts.items()])
            self.ratio_label.configure(text=ratio_text)

        if current >= total and not self.engine.is_running:
            self.start_button.configure(state="normal")
            self.open_out_btn.configure(state="normal", fg_color="#3498DB")
            self.open_obsidian_btn.configure(state="normal", fg_color="#9B59B6")

    def open_output_folder(self):
        path = self.engine.output_dir
        if path.exists():
            if platform.system() == "Windows": os.startfile(path)
            else: os.system(f'open "{path}"')
        else: messagebox.showwarning("Warning", "Directory not found")

    def open_obsidian_logs(self):
        path = self.engine.output_dir / "Obsidian_Logs"
        if path.exists():
            if platform.system() == "Windows": os.startfile(path)
            else: os.system(f'open "{path}"')
        else: messagebox.showwarning("Warning", "Notes not generated yet")

    def append_log(self, message, level):
        self.after(0, self._append_log_ui, message)

    def _append_log_ui(self, message):
        self.log_textbox.insert(tk.END, f"{message}\n")
        self.log_textbox.see(tk.END)

if __name__ == "__main__":
    app = PetPhotoSorterApp()
    app.mainloop()
