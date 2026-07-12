# -----------------------------------------------------------
# NICEC – NIoi Corneal Entropy Calculator
# Unified GUI for Collector, Predictor & Trainer
#
# Copyright © Matteo Nioi, 2026
# All Rights Reserved.
# -----------------------------------------------------------

import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# -----------------------------------------------------------
#  Costanti
# -----------------------------------------------------------

APP_TITLE = "NICEC – NIoi Corneal Entropy Calculator"
APP_SUBTITLE = "by Matteo Nioi"
COPYRIGHT = "© 2026 Matteo Nioi – All Rights Reserved"
VERSION = "2.0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLASH_IMAGE = os.path.join(BASE_DIR, "COPERTINA.png")
COVER_IMAGE = SPLASH_IMAGE   # stessa immagine per copertina
COLLECTOR_SCRIPT = os.path.join(BASE_DIR, "EYE_MEP_Collector.py")
PREDICTOR_SCRIPT = os.path.join(BASE_DIR, "EYE_MEP_Predictor.py")
TRAINER_SCRIPT = os.path.join(BASE_DIR, "EYE_MEP_Trainer.py")
DATASET_CSV = os.path.join(BASE_DIR, "oea_pmi_dataset.csv")
REPORT_TXT = os.path.join(BASE_DIR, "eye_mep_training_report.txt")


# -----------------------------------------------------------
#  Splash Screen
# -----------------------------------------------------------

class SplashScreen(tk.Toplevel):
    def __init__(self, master, image_path, timeout=2000):
        super().__init__(master)
        self.master = master
        self.overrideredirect(True)          # senza bordi
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")
        self.configure(bg="#0A192F")

        # Carica immagine
        self.img = None
        if os.path.exists(image_path) and HAS_PIL:
            try:
                pil_img = Image.open(image_path)
                # Ridimensiona mantenendo proporzioni, massimo 80% dello schermo
                screen_w = self.winfo_screenwidth()
                screen_h = self.winfo_screenheight()
                max_w = int(screen_w * 0.9)
                max_h = int(screen_h * 0.9)
                pil_img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                self.img = ImageTk.PhotoImage(pil_img)
            except Exception:
                self.img = None

        if self.img:
            lbl = tk.Label(self, image=self.img, bg="#0A192F")
            lbl.pack(expand=True)
        else:
            # Fallback testuale
            lbl = tk.Label(self, text="NICEC\nNIoi Corneal Entropy Calculator\nby Matteo Nioi",
                           font=("Segoe UI", 36, "bold"), fg="#F1F5FB", bg="#0A192F",
                           justify="center")
            lbl.pack(expand=True)

        # Pulsante Skip
        skip_btn = tk.Button(self, text="Skip →", command=self.close,
                             font=("Segoe UI", 12), bg="#1F6FEB", fg="white",
                             relief="flat", padx=20, pady=8)
        skip_btn.place(relx=0.95, rely=0.95, anchor="se")

        # Timer per chiusura automatica
        self.after(timeout, self.close)

    def close(self):
        self.destroy()
        self.master.deiconify()          # mostra finestra principale
        self.master.focus_force()


# -----------------------------------------------------------
#  Classe principale (GUI)
# -----------------------------------------------------------

class MainApp(tk.Tk):
    def __init__(self):
        # Nascondi la finestra principale finché lo splash non finisce
        super().__init__()
        self.withdraw()
        self.title(f"{APP_TITLE} – {APP_SUBTITLE}")
        self.geometry("920x820")
        self.configure(bg="#0A192F")
        self.minsize(800, 700)

        self.cover_image = None
        self.progress_active = False

        self._build_style()
        self._build_layout()

        # Mostra splash screen (2 secondi)
        self.splash = SplashScreen(self, SPLASH_IMAGE, timeout=2000)

        # Dopo lo splash, carica la copertina nella GUI
        self.after(2100, self._load_cover)   # si avvia dopo la chiusura dello splash

        self.log("NICEC Suite loaded successfully.")
        self.log(f"{COPYRIGHT}")

    # ---------------------- UI Stile -------------------------

    def _build_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#0A192F")
        style.configure("TLabel", background="#0A192F",
                        foreground="#E6EDF7", font=("Segoe UI", 10))
        style.configure("Title.TLabel",
                        font=("Segoe UI", 20, "bold"),
                        foreground="#F1F5FB")
        style.configure("Subtitle.TLabel",
                        font=("Segoe UI", 12, "italic"),
                        foreground="#8899AA")
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.map("TButton",
                  background=[("active", "#1F6FEB")],
                  foreground=[("active", "white")])
        style.configure("TProgressbar", background="#1F6FEB",
                        troughcolor="#020817")

    # ---------------------- Layout -------------------------

    def _build_layout(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Cover Frame (conterrà l'immagine)
        self.cover_frame = ttk.Frame(main_frame)
        self.cover_frame.pack(fill="x", pady=(0, 15))

        # Separator
        ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=8)

        # Pulsanti
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=10)

        btn_collector = ttk.Button(btn_frame, text="Launch Collector",
                                   command=self.launch_collector, width=25)
        btn_collector.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        btn_predictor = ttk.Button(btn_frame, text="Run Predictor (Stats)",
                                   command=self.run_predictor, width=25)
        btn_predictor.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        btn_trainer = ttk.Button(btn_frame, text="Run Trainer (Model)",
                                 command=self.run_trainer, width=25)
        btn_trainer.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        btn_view_report = ttk.Button(btn_frame, text="View Training Report",
                                     command=self.view_report, width=25)
        btn_view_report.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        btn_open_dataset = ttk.Button(btn_frame, text="Open Dataset CSV",
                                      command=self.open_dataset, width=25)
        btn_open_dataset.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        btn_exit = ttk.Button(btn_frame, text="Exit Suite",
                              command=self.quit, width=25)
        btn_exit.grid(row=1, column=2, padx=5, pady=5, sticky="w")

        # Progress bar
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill="x", pady=(8, 5))

        self.progress_label = ttk.Label(progress_frame, text="",
                                        foreground="#E6EDF7")
        self.progress_label.pack(side="left", padx=(0, 10))
        self.progress_label.pack_forget()

        self.progress_bar = ttk.Progressbar(progress_frame,
                                            mode="indeterminate",
                                            length=400,
                                            style="TProgressbar")
        self.progress_bar.pack(side="left", fill="x", expand=True)
        self.progress_bar.pack_forget()

        # Log
        log_frame = ttk.LabelFrame(main_frame, text="Execution Log",
                                   labelanchor="n")
        log_frame.pack(fill="both", expand=True, pady=(10, 0))

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            bg="#020817",
            fg="#E6EDF7",
            insertbackground="white",
            font=("Consolas", 9),
            wrap="word",
            height=12
        )
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

        # Status
        self.status_var = tk.StringVar(value="Ready.")
        status_lbl = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status_lbl.pack(side="bottom", fill="x", padx=10, pady=5)

    # ---------------------- Copertina persistente -------------

    def _load_cover(self):
        """Carica l'immagine COPERTINA.png come header della GUI."""
        for widget in self.cover_frame.winfo_children():
            widget.destroy()

        if os.path.exists(COVER_IMAGE) and HAS_PIL:
            try:
                img = Image.open(COVER_IMAGE)
                max_width = 860
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_size = (max_width, int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                self.cover_image = ImageTk.PhotoImage(img)
                lbl = ttk.Label(self.cover_frame, image=self.cover_image,
                                background="#0A192F")
                lbl.pack()
                self.log("Copertina caricata: COPERTINA.png")
                return
            except Exception as e:
                self.log(f"Errore caricamento copertina: {e}")

        # Fallback testuale
        self._build_text_cover()

    def _build_text_cover(self):
        title_lbl = ttk.Label(self.cover_frame, text=APP_TITLE,
                              style="Title.TLabel")
        title_lbl.pack()

        subtitle_lbl = ttk.Label(self.cover_frame, text=APP_SUBTITLE,
                                 style="Subtitle.TLabel")
        subtitle_lbl.pack(pady=(2, 0))

        copyright_lbl = ttk.Label(self.cover_frame, text=COPYRIGHT,
                                  font=("Segoe UI", 9, "italic"),
                                  foreground="#8899AA")
        copyright_lbl.pack(pady=(4, 0))
        self.log("Copertina testuale (nessuna immagine)")

    # ---------------------- Progress bar ---------------------

    def start_progress(self, label_text="Processing..."):
        if self.progress_active:
            return
        self.progress_active = True
        self.progress_label.config(text=label_text)
        self.progress_label.pack(side="left", padx=(0, 10))
        self.progress_bar.pack(side="left", fill="x", expand=True)
        self.progress_bar.start(10)
        self.update_idletasks()

    def stop_progress(self):
        if not self.progress_active:
            return
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.progress_label.pack_forget()
        self.progress_active = False
        self.update_idletasks()

    # ---------------------- Log e status ---------------------

    def log(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.update_idletasks()

    def set_status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    # ---------------------- Lanciatori moduli ----------------

    def launch_collector(self):
        if not os.path.exists(COLLECTOR_SCRIPT):
            messagebox.showerror("Error", f"Collector not found:\n{COLLECTOR_SCRIPT}")
            return
        self.log("Launching Collector...")
        self.set_status("Collector running...")
        self.start_progress("Collector is running...")
        try:
            subprocess.Popen([sys.executable, COLLECTOR_SCRIPT])
            self.log("Collector started in a new window.")
            self.set_status("Collector launched.")
            self.stop_progress()
        except Exception as e:
            self.log(f"ERROR: {e}")
            self.set_status("Error launching Collector.")
            self.stop_progress()
            messagebox.showerror("Launch Error", str(e))

    def run_script(self, script_path, label):
        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"Script not found:\n{script_path}")
            return

        def target():
            self.after(0, self.start_progress, f"{label} is running...")
            self.log(f"Starting {label}...")
            self.set_status(f"{label} running...")
            try:
                process = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=BASE_DIR
                )
                for line in process.stdout:
                    self.log(f"[{label}] {line.rstrip()}")
                process.wait()
                if process.returncode == 0:
                    self.log(f"{label} finished successfully.")
                    self.set_status(f"{label} completed.")
                else:
                    self.log(f"{label} finished with code {process.returncode}.")
                    self.set_status(f"{label} completed with errors.")
            except Exception as e:
                self.log(f"ERROR: {e}")
                self.set_status(f"Error running {label}.")
                messagebox.showerror("Execution Error", str(e))
            finally:
                self.after(0, self.stop_progress)

        threading.Thread(target=target, daemon=True).start()

    def run_predictor(self):
        self.run_script(PREDICTOR_SCRIPT, "Predictor")

    def run_trainer(self):
        self.run_script(TRAINER_SCRIPT, "Trainer")

    # ---------------------- Utility --------------------------

    def view_report(self):
        if not os.path.exists(REPORT_TXT):
            messagebox.showinfo("Not found",
                                f"Report not found:\n{REPORT_TXT}\nRun Trainer first.")
            return
        try:
            if sys.platform.startswith('win'):
                os.startfile(REPORT_TXT)
            elif sys.platform.startswith('darwin'):
                subprocess.run(['open', REPORT_TXT])
            else:
                subprocess.run(['xdg-open', REPORT_TXT])
            self.log(f"Opened report: {REPORT_TXT}")
        except Exception as e:
            self.log(f"Error opening report: {e}")
            self._show_text_file(REPORT_TXT, "Training Report")

    def open_dataset(self):
        if not os.path.exists(DATASET_CSV):
            messagebox.showinfo("Not found",
                                f"Dataset not found:\n{DATASET_CSV}\nRun Collector first.")
            return
        try:
            if sys.platform.startswith('win'):
                os.startfile(DATASET_CSV)
            elif sys.platform.startswith('darwin'):
                subprocess.run(['open', DATASET_CSV])
            else:
                subprocess.run(['xdg-open', DATASET_CSV])
            self.log(f"Opened dataset: {DATASET_CSV}")
        except Exception as e:
            self.log(f"Error opening dataset: {e}")

    def _show_text_file(self, filepath, title):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file:\n{e}")
            return
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("800x600")
        win.configure(bg="#0A192F")
        txt = scrolledtext.ScrolledText(
            win,
            bg="#020817",
            fg="#E6EDF7",
            insertbackground="white",
            font=("Consolas", 9),
            wrap="word"
        )
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        txt.insert("1.0", content)
        txt.configure(state="disabled")


# -----------------------------------------------------------
#  Avvio
# -----------------------------------------------------------

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()