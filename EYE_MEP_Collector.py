# -----------------------------------------------------------
#  OptoVue Entropy Analyzer (OEA) – PMI Collector v3.0
#  Cadaveric OptoVue Entropy Dataset Builder
#
#  Copyright © Matteo Nioi, 2025
#  All Rights Reserved.
#
#  Questo software raccoglie in modo strutturato le scansioni
#  OCT/pachimetria OptoVue (cadaveriche) e calcola un set di
#  entropie "grezze" (multi-dominio) su MAPPA e CORNEA,
#  salvando i dati in un CSV unico: "oea_pmi_dataset.csv".
#
#  v3.0 = estensione entropica:
#   - crop automatico MAPPA e CORNEA
#   - 4 entropie per MAPPA e 4 per CORNEA:
#       Shannon, GLCM, Laplacian, High-frequency
#   - somma grezza delle 4 entropie (map/cornea)
#   - grafico grezzo Σ entropie vs PMI per il batch
#
#  Mantiene:
#   - Dataset Editor (delete/update/undo)
#   - estetica dark, layout elegante v2.1
# -----------------------------------------------------------

import os
import csv
import time
import math
import datetime as dt
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

APP_TITLE = "OEA – PMI Collector v3.0 (Cadaveric OptoVue Entropy Dataset)"
DATASET_FILENAME = "oea_pmi_dataset.csv"


# -----------------------------------------------------------
#  Utility per entropia e filtri immagini
# -----------------------------------------------------------

def is_white_background(img_gray: Image.Image,
                        threshold_mean=245,
                        frac_white_thresh=0.35) -> bool:
    """
    Ritorna True se l'immagine ha sfondo quasi completamente bianco
    (tipico dei report con OCT su sfondo bianco che vogliamo escludere).
    """
    arr = np.array(img_gray, dtype=np.uint8)
    if arr.size == 0:
        return True
    mean_val = float(arr.mean())
    frac_white = float((arr > 250).sum()) / arr.size

    if mean_val > threshold_mean and frac_white > frac_white_thresh:
        return True
    return False


def compute_entropy_gray(img_gray: Image.Image) -> float:
    """
    Entropia di Shannon (in nats) di un'immagine in scala di grigi.
    """
    arr = np.array(img_gray, dtype=np.uint8).ravel()
    if arr.size == 0:
        return 0.0
    hist, _ = np.histogram(arr, bins=256, range=(0, 255))
    p = hist.astype(np.float64)
    s = p.sum()
    if s == 0:
        return 0.0
    p /= s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def compute_glcm_entropy(img_gray: Image.Image, levels: int = 32) -> float:
    """
    Entropia (Shannon, nats) della GLCM orizzontale.
    L'immagine viene quantizzata a 'levels' livelli.
    """
    arr = np.array(img_gray, dtype=np.uint8)
    if arr.size == 0:
        return 0.0

    # quantizzazione 0..255 -> 0..levels-1
    arr_q = (arr.astype(np.uint16) * levels // 256).astype(np.uint8)

    glcm = np.zeros((levels, levels), dtype=np.uint64)

    # co-occorrenza orizzontale (dx=1, dy=0)
    a = arr_q[:, :-1].ravel()
    b = arr_q[:, 1:].ravel()
    for x, y in zip(a, b):
        glcm[x, y] += 1

    total = glcm.sum()
    if total == 0:
        return 0.0

    p = glcm.astype(np.float64) / float(total)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def compute_laplacian_entropy(img_gray: Image.Image) -> float:
    """
    Entropia di Shannon del modulo del Laplaciano (bordi).
    """
    arr = np.array(img_gray, dtype=np.float32)
    if arr.size == 0:
        return 0.0

    # Laplaciano discreto 3x3 (convoluzione semplice)
    up    = np.pad(arr, ((1, 1), (0, 0)), mode="edge")[0:-2, :]
    down  = np.pad(arr, ((1, 1), (0, 0)), mode="edge")[2:, :]
    left  = np.pad(arr, ((0, 0), (1, 1)), mode="edge")[:, 0:-2]
    right = np.pad(arr, ((0, 0), (1, 1)), mode="edge")[:, 2:]

    lap = (up + down + left + right - 4.0 * arr)
    lap_abs = np.abs(lap).astype(np.float32)
    if lap_abs.size == 0:
        return 0.0

    m = lap_abs.max()
    if m <= 0:
        return 0.0

    lap_norm = (lap_abs * (255.0 / m)).astype(np.uint8).ravel()
    hist, _ = np.histogram(lap_norm, bins=256, range=(0, 255))
    p = hist.astype(np.float64)
    s = p.sum()
    if s == 0:
        return 0.0
    p /= s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def compute_highfreq_entropy(img_gray: Image.Image,
                             frac_high: float = 0.25) -> float:
    """
    Entropia (Shannon, nats) della potenza nelle alte frequenze (FFT2).
    Usa un anello esterno (ultimi 'frac_high' del raggio) come "high-freq".
    """
    arr = np.array(img_gray, dtype=np.float32)
    if arr.size == 0:
        return 0.0

    F = np.fft.fftshift(np.fft.fft2(arr))
    power = np.abs(F) ** 2

    h, w = power.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    r_max = dist.max()
    r_cut = (1.0 - frac_high) * r_max

    mask = dist >= r_cut
    hf = power[mask].ravel()
    if hf.size == 0:
        return 0.0

    hf = hf.astype(np.float64)
    s = hf.sum()
    if s <= 0:
        return 0.0
    p = hf / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# -----------------------------------------------------------
#  Crop MAPPA e CORNEA
# -----------------------------------------------------------

CORNEA_FRAC_W = 0.50  # 50% larghezza centrale
CORNEA_FRAC_H = 0.35  # 35% altezza centrale

def crop_cornea_center(img_gray: Image.Image) -> Image.Image:
    """
    Ritaglia la regione centrale della cornea (rettangolo centrale).
    Parametri CORNEA_FRAC_W/H regolano la dimensione del crop.
    """
    w, h = img_gray.size
    cw = int(w * CORNEA_FRAC_W)
    ch = int(h * CORNEA_FRAC_H)
    left = (w - cw) // 2
    top = (h - ch) // 2
    right = left + cw
    bottom = top + ch
    return img_gray.crop((left, top, right, bottom))


def crop_map_region(img_gray: Image.Image) -> Image.Image:
    """
    Ritaglia la 'mappa' come quasi tutta l'immagine,
    togliendo un margine per eliminare eventuali bordi GUI.
    """
    w, h = img_gray.size
    margin_w = int(w * 0.05)
    margin_h = int(h * 0.05)
    left = margin_w
    top = margin_h
    right = w - margin_w
    bottom = h - margin_h
    return img_gray.crop((left, top, right, bottom))


# -----------------------------------------------------------
#  Dataset CSV helpers
# -----------------------------------------------------------

def ensure_dataset_header(csv_path: Path, fieldnames):
    """
    Crea il CSV con header se non esiste o è vuoto.
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def append_rows_to_dataset(csv_path: Path, fieldnames, rows):
    ensure_dataset_header(csv_path, fieldnames)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for r in rows:
            writer.writerow(r)


def load_dataset_rows(csv_path: Path):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return [], []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return rows, fieldnames


def write_dataset_rows(csv_path: Path, fieldnames, rows):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# -----------------------------------------------------------
#  GUI principale
# -----------------------------------------------------------

class OEAPMICollectorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1400x800")
        self.configure(bg="#0A192F")

        # dove salvare il CSV (stessa cartella dello script)
        self.base_dir = Path(os.path.abspath(os.path.dirname(__file__)))
        self.dataset_path = self.base_dir / DATASET_FILENAME

        # campi di base per il CSV (v3: estesi con entropie multi-dominio)
        self.fieldnames = [
            "id",               # cadaver-eye-idx
            "cadaver_id",       # C01, C02, ...
            "eye_side",         # OD / OS
            "eye_state",        # open / closed
            "pmi_hours",        # float
            "image_path",       # path assoluto

            # ENTROPIE MAPPA
            "H_total",          # Shannon map (nats)
            "E_total",          # H_total normalizzata 0–1
            "H_glcm_map",       # GLCM entropy map (nats)
            "H_laplacian_map",  # Laplacian entropy map (nats)
            "H_highfreq_map",   # High-frequency entropy map (nats)
            "E_sum_map",        # Somma grezza 4 entropie map (nats)

            # ENTROPIE CORNEA
            "H_total_cornea",        # Shannon cornea (nats)
            "E_total_cornea",        # normalizzata 0–1
            "H_glcm_cornea",         # GLCM entropy cornea
            "H_laplacian_cornea",    # Laplacian entropy cornea
            "H_highfreq_cornea",     # High-frequency entropy cornea
            "E_sum_cornea",          # Somma grezza 4 entropie cornea

            "map_saturated",    # 0/1 (pachimetria bianca)
            "bg_white",         # 0/1 (sfondo bianco, esclusa)
            "batch_id",         # int (timestamp)
            "added_at",         # stringa timestamp
        ]

        # buffer per grafico grezzo (batch corrente)
        self.batch_pmi = []
        self.batch_Esum_map = []
        self.batch_Esum_cornea = []

        self._build_style()
        self._build_layout()

        # refresh summary iniziale (se esiste già un CSV)
        self.refresh_dataset_summary()

    # ---------------------- UI setup ------------------------

    def _build_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#0A192F")
        style.configure("TLabel", background="#0A192F",
                        foreground="#E6EDF7", font=("Segoe UI", 10))
        style.configure("Heading.TLabel",
                        font=("Segoe UI", 11, "bold"),
                        foreground="#F1F5FB")
        style.configure("Title.TLabel",
                        font=("Segoe UI", 14, "bold"),
                        foreground="#F1F5FB")
        style.configure("TButton", font=("Segoe UI", 9), padding=4)
        style.map("TButton",
                  background=[("active", "#1F6FEB")],
                  foreground=[("active", "white")])

    def _build_layout(self):
        # top bar
        top_frame = ttk.Frame(self)
        top_frame.pack(side="top", fill="x", padx=8, pady=4)

        title_lbl = ttk.Label(top_frame, text=APP_TITLE, style="Title.TLabel")
        title_lbl.pack(side="left")

        copyright_lbl = ttk.Label(
            top_frame,
            text="© 2025 Matteo Nioi – All Rights Reserved",
            font=("Segoe UI", 9),
        )
        copyright_lbl.pack(side="right")

        # main horiz split
        main_frame = ttk.Frame(self)
        main_frame.pack(side="top", fill="both", expand=True, padx=8, pady=4)

        # left control panel
        left = ttk.Frame(main_frame)
        left.pack(side="left", fill="y", padx=(0, 8))

        # right stacked (log + summary)
        right = ttk.Frame(main_frame)
        right.pack(side="left", fill="both", expand=True)

        # --------- LEFT: controls ----------

        # Cadaver & Eye box
        cad_frame = ttk.LabelFrame(left, text="Cadaver & Eye", labelanchor="n")
        cad_frame.pack(fill="x", pady=4)

        ttk.Label(cad_frame, text="Cadaver ID (C01–C24 or other):").grid(
            row=0, column=0, sticky="w"
        )
        self.cadaver_var = tk.StringVar(value="C01")
        cad_entry = ttk.Entry(cad_frame, textvariable=self.cadaver_var, width=10)
        cad_entry.grid(row=0, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(cad_frame, text="Eye:").grid(row=1, column=0, sticky="w")
        self.eye_side_var = tk.StringVar(value="OD")
        ttk.Radiobutton(cad_frame, text="OD (right)",
                        variable=self.eye_side_var, value="OD").grid(
            row=1, column=1, sticky="w"
        )
        ttk.Radiobutton(cad_frame, text="OS (left)",
                        variable=self.eye_side_var, value="OS").grid(
            row=1, column=2, sticky="w"
        )

        ttk.Label(cad_frame, text="Lids:").grid(row=2, column=0, sticky="w")
        self.eye_state_var = tk.StringVar(value="closed")
        ttk.Radiobutton(cad_frame, text="Closed",
                        variable=self.eye_state_var, value="closed").grid(
            row=2, column=1, sticky="w"
        )
        ttk.Radiobutton(cad_frame, text="Open",
                        variable=self.eye_state_var, value="open").grid(
            row=2, column=2, sticky="w"
        )

        # PMI box
        pmi_frame = ttk.LabelFrame(left, text="Postmortem Interval", labelanchor="n")
        pmi_frame.pack(fill="x", pady=4)

        ttk.Label(pmi_frame, text="PMI (hours):").grid(row=0, column=0, sticky="w")
        self.pmi_var = tk.StringVar(value="6")
        pmi_entry = ttk.Entry(pmi_frame, textvariable=self.pmi_var, width=8)
        pmi_entry.grid(row=0, column=1, sticky="w", padx=4, pady=2)

        hint = "(es. 3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 17...)"
        ttk.Label(pmi_frame, text=hint).grid(row=1, column=0, columnspan=3, sticky="w")

        btn_add_imgs = ttk.Button(
            pmi_frame,
            text="Add IMAGES (files)…",
            command=self.add_images_files,
        )
        btn_add_imgs.grid(row=2, column=0, columnspan=3, sticky="we", pady=(6, 2))

        btn_add_folder = ttk.Button(
            pmi_frame,
            text="Add FOLDER of scans…",
            command=self.add_images_folder,
        )
        btn_add_folder.grid(row=3, column=0, columnspan=3, sticky="we", pady=2)

        # Dataset buttons
        ds_frame = ttk.LabelFrame(left, text="Dataset tools", labelanchor="n")
        ds_frame.pack(fill="x", pady=4)

        ttk.Button(ds_frame,
                   text="Refresh dataset summary",
                   command=self.refresh_dataset_summary).pack(
            fill="x", pady=2
        )

        ttk.Button(ds_frame,
                   text="Open Dataset Editor…",
                   command=self.open_dataset_editor).pack(
            fill="x", pady=(6, 2)
        )

        # Note box
        note_frame = ttk.LabelFrame(left, text="Notes", labelanchor="n")
        note_frame.pack(fill="both", expand=True, pady=4)

        note_txt = (
            "Le immagini con sfondo completamente bianco vengono escluse.\n"
            "Le entropie sono calcolate su due regioni:\n"
            " - MAPPA: quasi tutta l'immagine (tolti i bordi GUI)\n"
            " - CORNEA: rettangolo centrale (parametrico)\n\n"
            "Per ogni immagine:\n"
            " - 4 entropie sulla MAPPA (Shannon, GLCM, Laplacian, High-freq)\n"
            " - 4 entropie sulla CORNEA\n"
            " - somma grezza delle 4 entropie in ciascuna regione (nats)\n\n"
            "Suggerimento: usa UNA cartella per ogni blocco, es.\n"
            "C12_OD_closed_6h_operator1\n"
            "C12_OD_closed_6h_operator2\n"
            "Puoi chiamare Add FOLDER più volte per lo stesso cadavere/occhio/PMI."
        )
        lbl_note = ttk.Label(note_frame, text=note_txt,
                             justify="left", wraplength=260)
        lbl_note.pack(fill="both", expand=True)

        # --------- RIGHT: log + summary ----------

        # Log
        log_frame = ttk.LabelFrame(right, text="Log of last operations", labelanchor="n")
        log_frame.pack(fill="both", expand=True, pady=(0, 4))

        self.log_text = tk.Text(
            log_frame,
            height=14,
            bg="#020817",
            fg="#E6EDF7",
            insertbackground="white",
            font=("Consolas", 9),
            wrap="word",
        )
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

        # Summary
        sum_frame = ttk.LabelFrame(right, text="Dataset summary (rows per cadaver/eye)", labelanchor="n")
        sum_frame.pack(fill="both", expand=True)

        self.summary_text = tk.Text(
            sum_frame,
            height=8,
            bg="#020817",
            fg="#E6EDF7",
            insertbackground="white",
            font=("Consolas", 9),
            wrap="none",
        )
        self.summary_text.pack(fill="both", expand=True)
        self.summary_text.configure(state="disabled")

        # Status bar
        self.status_var = tk.StringVar(value="Ready.")
        status_lbl = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status_lbl.pack(side="bottom", fill="x")

    # ---------------------- Log & summary -------------------

    def log(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def set_status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    def refresh_dataset_summary(self):
        rows, _ = load_dataset_rows(self.dataset_path)
        counts = {}
        for r in rows:
            key = f"{r.get('cadaver_id','?')}_{r.get('eye_side','?')}"
            counts[key] = counts.get(key, 0) + 1

        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        if not counts:
            self.summary_text.insert("end", "No data yet.\n")
        else:
            self.summary_text.insert("end", "Cadaver_Eye   #scans (rows in CSV)\n")
            self.summary_text.insert("end", "--------------------------------\n")
            for key in sorted(counts.keys()):
                self.summary_text.insert("end", f"{key:10s} : {counts[key]:4d}\n")
        self.summary_text.configure(state="disabled")

    # ---------------------- Image import --------------------

    def _get_current_context(self):
        cad = self.cadaver_var.get().strip()
        eye = self.eye_side_var.get().strip()
        state = self.eye_state_var.get().strip()
        pmi_str = self.pmi_var.get().strip().replace(",", ".")
        try:
            pmi = float(pmi_str)
        except ValueError:
            raise ValueError("PMI deve essere un numero (ore).")

        if not cad:
            raise ValueError("Cadaver ID mancante.")
        if eye not in ("OD", "OS"):
            raise ValueError("Eye deve essere OD o OS.")
        if state not in ("open", "closed"):
            raise ValueError("State deve essere open/closed.")

        return cad, eye, state, pmi

    def add_images_files(self):
        try:
            cad, eye, state, pmi = self._get_current_context()
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e), parent=self)
            return

        filepaths = filedialog.askopenfilenames(
            parent=self,
            title="Select OptoVue OCT/pachymetry images",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not filepaths:
            return

        self._process_image_list(filepaths, cad, eye, state, pmi)

    def add_images_folder(self):
        try:
            cad, eye, state, pmi = self._get_current_context()
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e), parent=self)
            return

        folder = filedialog.askdirectory(parent=self, title="Select folder with OptoVue scans")
        if not folder:
            return

        exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
        filepaths = [
            str(Path(folder) / name)
            for name in sorted(os.listdir(folder))
            if name.lower().endswith(exts)
        ]
        if not filepaths:
            messagebox.showwarning("No images", "Nessuna immagine trovata nella cartella.", parent=self)
            return

        self._process_image_list(filepaths, cad, eye, state, pmi)

    def _process_image_list(self, paths, cad, eye, state, pmi):
        """
        Elabora una lista di immagini, calcola entropie multi-dominio
        su MAPPA e CORNEA, aggiorna CSV e mostra grafico grezzo.
        """
        batch_id = int(time.time())
        timestamp = dt.datetime.now().isoformat(timespec="seconds")

        n_total = 0
        n_added = 0
        n_excluded_white = 0
        n_errors = 0

        rows_to_add = []

        # reset buffer per grafico batch
        self.batch_pmi = []
        self.batch_Esum_map = []
        self.batch_Esum_cornea = []

        for img_path in paths:
            n_total += 1
            try:
                img = Image.open(img_path)
                img_gray = img.convert("L")

                # escludi sfondo completamente bianco (report inadatti)
                if is_white_background(img_gray):
                    n_excluded_white += 1
                    self.log(f"[SKIP white bg] {img_path}")
                    continue

                # --- MAPPA (quasi tutta l'immagine) ---
                img_map = crop_map_region(img_gray)

                H_total_map = compute_entropy_gray(img_map)
                H_max = math.log(256.0)
                E_total_map = H_total_map / H_max if H_max > 0 else 0.0

                H_glcm_map = compute_glcm_entropy(img_map)
                H_laplacian_map = compute_laplacian_entropy(img_map)
                H_highfreq_map = compute_highfreq_entropy(img_map)

                E_sum_map = (
                    H_total_map
                    + H_glcm_map
                    + H_laplacian_map
                    + H_highfreq_map
                )

                # --- CORNEA (crop centrale) ---
                img_cornea = crop_cornea_center(img_gray)

                H_total_cornea = compute_entropy_gray(img_cornea)
                E_total_cornea = H_total_cornea / H_max if H_max > 0 else 0.0

                H_glcm_cornea = compute_glcm_entropy(img_cornea)
                H_laplacian_cornea = compute_laplacian_entropy(img_cornea)
                H_highfreq_cornea = compute_highfreq_entropy(img_cornea)

                E_sum_cornea = (
                    H_total_cornea
                    + H_glcm_cornea
                    + H_laplacian_cornea
                    + H_highfreq_cornea
                )

                row = {
                    "id": "",
                    "cadaver_id": cad,
                    "eye_side": eye,
                    "eye_state": state,
                    "pmi_hours": f"{pmi:.3f}",
                    "image_path": os.path.abspath(img_path),

                    # MAPPA
                    "H_total": f"{H_total_map:.6f}",
                    "E_total": f"{E_total_map:.6f}",
                    "H_glcm_map": f"{H_glcm_map:.6f}",
                    "H_laplacian_map": f"{H_laplacian_map:.6f}",
                    "H_highfreq_map": f"{H_highfreq_map:.6f}",
                    "E_sum_map": f"{E_sum_map:.6f}",

                    # CORNEA
                    "H_total_cornea": f"{H_total_cornea:.6f}",
                    "E_total_cornea": f"{E_total_cornea:.6f}",
                    "H_glcm_cornea": f"{H_glcm_cornea:.6f}",
                    "H_laplacian_cornea": f"{H_laplacian_cornea:.6f}",
                    "H_highfreq_cornea": f"{H_highfreq_cornea:.6f}",
                    "E_sum_cornea": f"{E_sum_cornea:.6f}",

                    "map_saturated": "0",  # placeholder per futura logica
                    "bg_white": "0",
                    "batch_id": str(batch_id),
                    "added_at": timestamp,
                }
                rows_to_add.append(row)
                n_added += 1

                # Per il grafico grezzo (batch corrente)
                self.batch_pmi.append(pmi)
                self.batch_Esum_map.append(E_sum_map)
                self.batch_Esum_cornea.append(E_sum_cornea)

            except Exception as e:
                n_errors += 1
                self.log(f"[ERROR] {img_path} → {e}")

        # assegna id progressivo cadaver-eye
        if rows_to_add:
            existing_rows, existing_fns = load_dataset_rows(self.dataset_path)
            if existing_rows:
                # ultima id numerica per stesso cadaver/eye
                max_idx_per_key = {}
                for r in existing_rows:
                    key = (r.get("cadaver_id", ""), r.get("eye_side", ""))
                    try:
                        idx = int((r.get("id") or "0").split("_")[-1])
                    except ValueError:
                        idx = 0
                    max_idx_per_key[key] = max(max_idx_per_key.get(key, 0), idx)

                for r in rows_to_add:
                    key = (r["cadaver_id"], r["eye_side"])
                    max_idx_per_key[key] = max(max_idx_per_key.get(key, 0), 1)
                    idx = max_idx_per_key[key]
                    r["id"] = f"{r['cadaver_id']}_{r['eye_side']}_{idx:05d}"
                    max_idx_per_key[key] = idx + 1
            else:
                # primo inserimento
                counters = {}
                for r in rows_to_add:
                    key = (r["cadaver_id"], r["eye_side"])
                    counters[key] = counters.get(key, 0) + 1
                    r["id"] = f"{r['cadaver_id']}_{r['eye_side']}_{counters[key]:05d}"

            append_rows_to_dataset(self.dataset_path, self.fieldnames, rows_to_add)

        msg = (
            f"Processed {len(paths)} file(s) for {cad} {eye} ({state}), "
            f"PMI={pmi:.1f} h → {n_added} added, "
            f"{n_excluded_white} excluded (white bg), {n_errors} errors."
        )
        self.log(msg)
        self.set_status(msg)
        self.refresh_dataset_summary()

        # grafico grezzo Σ entropie vs PMI
        try:
            self.plot_raw_entropy_batch()
        except Exception as e:
            self.log(f"[PLOT ERROR] {e}")

    def plot_raw_entropy_batch(self):
        """
        Grafico grezzo: somma delle entropie vs PMI per il batch corrente,
        sia per MAPPA sia per CORNEA.
        """
        if not self.batch_pmi:
            self.log("No data for raw entropy plot.")
            return

        data = sorted(
            zip(self.batch_pmi, self.batch_Esum_map, self.batch_Esum_cornea),
            key=lambda x: x[0],
        )
        pmi_vals = [d[0] for d in data]
        Esum_map_vals = [d[1] for d in data]
        Esum_cornea_vals = [d[2] for d in data]

        plt.figure()
        plt.plot(pmi_vals, Esum_map_vals, marker="o", label="Σ entropie MAPPA")
        plt.plot(pmi_vals, Esum_cornea_vals, marker="s", label="Σ entropie CORNEA")
        plt.xlabel("PMI (hours)")
        plt.ylabel("Sum of entropies (nats)")
        plt.title("Batch: raw entropy sum vs PMI")
        plt.grid(True)
        plt.legend()
        plt.show()

    # ---------------------- Dataset Editor ------------------

    def open_dataset_editor(self):
        DatasetEditor(self)


# -----------------------------------------------------------
#  Dataset Editor window (come v2.1, funziona anche con nuovi campi)
# -----------------------------------------------------------

class DatasetEditor(tk.Toplevel):
    def __init__(self, master: OEAPMICollectorApp):
        super().__init__(master)
        self.master_app = master
        self.title("OEA – Dataset Editor")
        self.geometry("900x500")
        self.configure(bg="#0A192F")

        self._build_ui()
        self.refresh_preview()

    def _build_ui(self):
        style = ttk.Style()
        style.configure("Editor.TFrame", background="#0A192F")
        style.configure("Editor.TLabel",
                        background="#0A192F",
                        foreground="#E6EDF7",
                        font=("Segoe UI", 10))

        main = ttk.Frame(self, style="Editor.TFrame")
        main.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(main, style="Editor.TFrame")
        left.pack(side="left", fill="y", padx=(0, 8))

        right = ttk.Frame(main, style="Editor.TFrame")
        right.pack(side="left", fill="both", expand=True)

        # --- LEFT controls ---
        # Delete block
        del_frame = ttk.LabelFrame(left, text="Delete rows (Cadaver / Eye / PMI)", labelanchor="n")
        del_frame.pack(fill="x", pady=4)

        ttk.Label(del_frame, text="Cadaver ID:", style="Editor.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.del_cad_var = tk.StringVar()
        ttk.Entry(del_frame, textvariable=self.del_cad_var, width=8).grid(
            row=0, column=1, padx=4, pady=2
        )

        ttk.Label(del_frame, text="Eye:", style="Editor.TLabel").grid(
            row=1, column=0, sticky="w"
        )
        self.del_eye_var = tk.StringVar(value="OD")
        ttk.Combobox(del_frame, textvariable=self.del_eye_var,
                     values=["OD", "OS"], width=5).grid(
            row=1, column=1, padx=4, pady=2
        )

        ttk.Label(del_frame, text="PMI (h):", style="Editor.TLabel").grid(
            row=2, column=0, sticky="w"
        )
        self.del_pmi_var = tk.StringVar()
        ttk.Entry(del_frame, textvariable=self.del_pmi_var, width=8).grid(
            row=2, column=1, padx=4, pady=2
        )

        ttk.Button(del_frame,
                   text="DELETE matching rows",
                   command=self.delete_rows).grid(
            row=3, column=0, columnspan=2, sticky="we", pady=(6, 2)
        )

        # Update PMI block
        upd_frame = ttk.LabelFrame(left, text="Update PMI (correct hour)", labelanchor="n")
        upd_frame.pack(fill="x", pady=8)

        ttk.Label(upd_frame, text="Cadaver ID:", style="Editor.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.upd_cad_var = tk.StringVar()
        ttk.Entry(upd_frame, textvariable=self.upd_cad_var, width=8).grid(
            row=0, column=1, padx=4, pady=2
        )

        ttk.Label(upd_frame, text="Eye:", style="Editor.TLabel").grid(
            row=1, column=0, sticky="w"
        )
        self.upd_eye_var = tk.StringVar(value="OD")
        ttk.Combobox(upd_frame, textvariable=self.upd_eye_var,
                     values=["OD", "OS"], width=5).grid(
            row=1, column=1, padx=4, pady=2
        )

        ttk.Label(upd_frame, text="PMI wrong (h):", style="Editor.TLabel").grid(
            row=2, column=0, sticky="w"
        )
        self.upd_pmi_old_var = tk.StringVar()
        ttk.Entry(upd_frame, textvariable=self.upd_pmi_old_var, width=8).grid(
            row=2, column=1, padx=4, pady=2
        )

        ttk.Label(upd_frame, text="PMI correct (h):", style="Editor.TLabel").grid(
            row=3, column=0, sticky="w"
        )
        self.upd_pmi_new_var = tk.StringVar()
        ttk.Entry(upd_frame, textvariable=self.upd_pmi_new_var, width=8).grid(
            row=3, column=1, padx=4, pady=2
        )

        ttk.Button(upd_frame,
                   text="UPDATE PMI",
                   command=self.update_pmi).grid(
            row=4, column=0, columnspan=2, sticky="we", pady=(6, 2)
        )

        # Undo block
        undo_frame = ttk.LabelFrame(left, text="Undo last import", labelanchor="n")
        undo_frame.pack(fill="x", pady=4)

        ttk.Button(undo_frame,
                   text="UNDO last batch_id",
                   command=self.undo_last_batch).pack(
            fill="x", pady=(4, 2)
        )

        ttk.Label(
            undo_frame,
            text=(
                "Usa solo se hai appena importato un blocco\n"
                "di cartelle errato (batch più recente)."
            ),
            style="Editor.TLabel",
        ).pack(fill="x", pady=(2, 2))

        # --- RIGHT preview ---
        prev_frame = ttk.LabelFrame(right, text="Dataset preview (first 200 rows)", labelanchor="n")
        prev_frame.pack(fill="both", expand=True)

        self.prev_text = tk.Text(
            prev_frame,
            bg="#020817",
            fg="#E6EDF7",
            insertbackground="white",
            font=("Consolas", 9),
            wrap="none",
        )
        self.prev_text.pack(fill="both", expand=True)

        # scrollbar orizzontale
        xscroll = tk.Scrollbar(prev_frame, orient="horizontal",
                               command=self.prev_text.xview)
        xscroll.pack(side="bottom", fill="x")
        self.prev_text.configure(xscrollcommand=xscroll.set)

    def refresh_preview(self):
        rows, fns = load_dataset_rows(self.master_app.dataset_path)
        self.prev_text.configure(state="normal")
        self.prev_text.delete("1.0", "end")
        if not rows:
            self.prev_text.insert("end", "Dataset vuoto.\n")
        else:
            self.prev_text.insert("end", ",".join(fns) + "\n")
            self.prev_text.insert("end", "-" * 120 + "\n")
            for r in rows[:200]:
                vals = [r.get(fn, "") for fn in fns]
                self.prev_text.insert("end", ",".join(vals) + "\n")
            if len(rows) > 200:
                self.prev_text.insert(
                    "end",
                    f"... ({len(rows) - 200} righe ulteriori non mostrate)\n",
                )
        self.prev_text.configure(state="disabled")

    # --- Editor actions ---

    def delete_rows(self):
        cad = self.del_cad_var.get().strip()
        eye = self.del_eye_var.get().strip()
        pmi_str = self.del_pmi_var.get().strip().replace(",", ".")
        if not cad or not eye or not pmi_str:
            messagebox.showerror("Missing data",
                                 "Compila Cadaver, Eye e PMI (h).", parent=self)
            return
        try:
            pmi = float(pmi_str)
        except ValueError:
            messagebox.showerror("Invalid PMI", "PMI deve essere numerico.",
                                 parent=self)
            return

        rows, fns = load_dataset_rows(self.master_app.dataset_path)
        if not rows:
            messagebox.showinfo("Empty", "Dataset vuoto.", parent=self)
            return

        pmi_formatted = f"{pmi:.3f}"
        before = len(rows)
        new_rows = [
            r for r in rows
            if not (
                r.get("cadaver_id") == cad
                and r.get("eye_side") == eye
                and r.get("pmi_hours") == pmi_formatted
            )
        ]
        removed = before - len(new_rows)
        if removed == 0:
            messagebox.showinfo("No match",
                                "Nessuna riga corrispondente trovata.",
                                parent=self)
            return

        if not messagebox.askyesno(
            "Confirm delete",
            (
                f"Verranno cancellate {removed} righe per {cad} {eye}, "
                f"PMI={pmi_formatted} h.\nProcedere?"
            ),
            parent=self,
        ):
            return

        write_dataset_rows(self.master_app.dataset_path, fns, new_rows)
        self.refresh_preview()
        self.master_app.refresh_dataset_summary()
        self.master_app.log(
            f"[EDITOR] Deleted {removed} rows for {cad} {eye}, PMI={pmi_formatted} h."
        )
        self.master_app.set_status(
            f"Deleted {removed} rows for {cad} {eye}, PMI={pmi_formatted} h."
        )

    def update_pmi(self):
        cad = self.upd_cad_var.get().strip()
        eye = self.upd_eye_var.get().strip()
        old_str = self.upd_pmi_old_var.get().strip().replace(",", ".")
        new_str = self.upd_pmi_new_var.get().strip().replace(",", ".")
        if not cad or not eye or not old_str or not new_str:
            messagebox.showerror(
                "Missing data",
                "Compila tutti i campi (Cadaver, Eye, PMI old/new).",
                parent=self,
            )
            return
        try:
            pmi_old = float(old_str)
            pmi_new = float(new_str)
        except ValueError:
            messagebox.showerror("Invalid PMI",
                                 "PMI devono essere numerici.", parent=self)
            return

        rows, fns = load_dataset_rows(self.master_app.dataset_path)
        if not rows:
            messagebox.showinfo("Empty", "Dataset vuoto.", parent=self)
            return

        old_fmt = f"{pmi_old:.3f}"
        new_fmt = f"{pmi_new:.3f}"
        changed = 0
        for r in rows:
            if (
                r.get("cadaver_id") == cad
                and r.get("eye_side") == eye
                and r.get("pmi_hours") == old_fmt
            ):
                r["pmi_hours"] = new_fmt
                changed += 1

        if changed == 0:
            messagebox.showinfo("No match",
                                "Nessuna riga corrispondente trovata.",
                                parent=self)
            return

        if not messagebox.askyesno(
            "Confirm update",
            (
                f"Verranno aggiornate {changed} righe per {cad} {eye}, "
                f"PMI {old_fmt} h → {new_fmt} h.\nProcedere?"
            ),
            parent=self,
        ):
            return

        write_dataset_rows(self.master_app.dataset_path, fns, rows)
        self.refresh_preview()
        self.master_app.refresh_dataset_summary()
        self.master_app.log(
            f"[EDITOR] Updated PMI for {changed} rows ({cad} {eye}): {old_fmt} → {new_fmt} h."
        )
        self.master_app.set_status(
            f"Updated PMI for {changed} rows ({cad} {eye}): {old_fmt} → {new_fmt} h."
        )

    def undo_last_batch(self):
        rows, fns = load_dataset_rows(self.master_app.dataset_path)
        if not rows:
            messagebox.showinfo("Empty", "Dataset vuoto.", parent=self)
            return

        # trova batch_id massimo
        batch_ids = [int(r.get("batch_id", "0") or 0) for r in rows]
        if not batch_ids:
            messagebox.showwarning(
                "No batch_id",
                "Nessun batch_id disponibile nel dataset.\n"
                "Funzione UNDO non utilizzabile.",
                parent=self,
            )
            return
        last_batch = max(batch_ids)

        before = len(rows)
        new_rows = [
            r for r in rows
            if int(r.get("batch_id", "0") or 0) != last_batch
        ]
        removed = before - len(new_rows)
        if removed == 0:
            messagebox.showinfo("No match",
                                "Nessuna riga con batch_id più recente trovata.",
                                parent=self)
            return

        if not messagebox.askyesno(
            "Confirm UNDO",
            (
                f"Verranno cancellate {removed} righe dell'ultimo import "
                f"(batch_id={last_batch}).\nProcedere?"
            ),
            parent=self,
        ):
            return

        write_dataset_rows(self.master_app.dataset_path, fns, new_rows)
        self.refresh_preview()
        self.master_app.refresh_dataset_summary()
        self.master_app.log(
            f"[EDITOR] UNDO last import: removed {removed} rows (batch_id={last_batch})."
        )
        self.master_app.set_status(
            f"UNDO last import: removed {removed} rows (batch_id={last_batch})."
        )


# -----------------------------------------------------------
#  Main
# -----------------------------------------------------------

if __name__ == "__main__":
    app = OEAPMICollectorApp()
    app.mainloop()
