# -----------------------------------------------------------
#  EYE-MEP – Entropy Stats & Model Trainer
#
#  Legge "oea_pmi_dataset.csv" (o altro CSV compatibile),
#  calcola z-score, entropia multivariata (MAPPA e CORNEA),
#  esegue regressioni PMI vs entropia, controlla le scale
#  delle entropie e salva:
#
#   - eye_mep_model_cornea.json   (modello lineare multientropico)
#   - eye_mep_stats_summary.txt   (riassunto leggibile)
#
#  Copyright © Matteo Nioi, 2025
#  All Rights Reserved.
# -----------------------------------------------------------

from dataclasses import dataclass
from typing import Dict, Tuple
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Nome di default del dataset prodotto dal Collector
DEFAULT_DATASET = "oea_pmi_dataset.csv"

MODEL_JSON = "eye_mep_model_cornea.json"
STATS_TXT = "eye_mep_stats_summary.txt"


# -----------------------------------------------------------
#  Struttura per salvare i risultati delle regressioni
# -----------------------------------------------------------

@dataclass
class EntropyStats:
    slope: float
    intercept: float
    r2: float
    mae: float
    rmse: float


# -----------------------------------------------------------
#  Loader
# -----------------------------------------------------------

def load_dataset(csv_path: str = DEFAULT_DATASET) -> pd.DataFrame:
    """
    Carica il dataset dal CSV e forza i tipi numerici per le colonne
    utilizzate nelle analisi.
    """
    df = pd.read_csv(csv_path)

    # PMI in ore (float)
    df["pmi_hours"] = df["pmi_hours"].astype(float)

    # Colonne di entropia che ci interessano (se presenti)
    entropy_cols = [
        "H_total",
        "H_glcm_map",
        "H_laplacian_map",
        "H_highfreq_map",
        "H_total_cornea",
        "H_glcm_cornea",
        "H_laplacian_cornea",
        "H_highfreq_cornea",
    ]
    for col in entropy_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


# -----------------------------------------------------------
#  Z-score + multivariata
# -----------------------------------------------------------

def compute_z_and_multi(df: pd.DataFrame, region: str = "map") -> pd.DataFrame:
    """
    region: 'map' oppure 'cornea'
    Aggiunge al df:
      - z_H_region, z_G_region, z_L_region, z_F_region
      - E_multi_map o E_multi_cornea = media dei 4 z-score
    """
    if region == "map":
        cols = {
            "H": "H_total",
            "G": "H_glcm_map",
            "L": "H_laplacian_map",
            "F": "H_highfreq_map",
        }
        multi_col = "E_multi_map"
    else:
        cols = {
            "H": "H_total_cornea",
            "G": "H_glcm_cornea",
            "L": "H_laplacian_cornea",
            "F": "H_highfreq_cornea",
        }
        multi_col = "E_multi_cornea"

    for key, col in cols.items():
        if col not in df.columns:
            df[f"z_{key}_{region}"] = 0.0
            continue

        mu = df[col].mean()
        sigma = df[col].std(ddof=0)
        if sigma <= 0:
            df[f"z_{key}_{region}"] = 0.0
        else:
            df[f"z_{key}_{region}"] = (df[col] - mu) / sigma

    zcols = [f"z_{k}_{region}" for k in ["H", "G", "L", "F"]]
    df[multi_col] = df[zcols].mean(axis=1)

    return df


# -----------------------------------------------------------
#  Controllo scale delle entropie
# -----------------------------------------------------------

def check_entropy_scales(df: pd.DataFrame, region: str = "map") -> str:
    """
    Restituisce una stringa con media e deviazione standard delle 4 entropie
    per verificare se la somma grezza è dominata da una di esse.
    region: 'map' oppure 'cornea'
    """
    if region == "map":
        cols = {
            "Shannon_map": "H_total",
            "GLCM_map": "H_glcm_map",
            "Laplacian_map": "H_laplacian_map",
            "HighFreq_map": "H_highfreq_map",
        }
    else:
        cols = {
            "Shannon_cornea": "H_total_cornea",
            "GLCM_cornea": "H_glcm_cornea",
            "Laplacian_cornea": "H_laplacian_cornea",
            "HighFreq_cornea": "H_highfreq_cornea",
        }

    lines = [f"=== Entropy scales ({region}) ==="]
    for name, col in cols.items():
        if col not in df.columns:
            lines.append(f"{name}: (colonna '{col}' non trovata)")
            continue
        vals = df[col].to_numpy(dtype=float)
        mean = np.mean(vals)
        std = np.std(vals)
        lines.append(f"{name}: mean={mean:.3f} ± {std:.3f} nats")
    return "\n".join(lines)


# -----------------------------------------------------------
#  Regressione lineare semplice
# -----------------------------------------------------------

def regression_pmi_vs(df: pd.DataFrame,
                      x_col: str,
                      pmi_col: str = "pmi_hours") -> EntropyStats:
    """
    Esegue una regressione lineare:
        PMI = slope * X + intercept
    e calcola R², MAE e RMSE.
    """
    df_clean = df.dropna(subset=[pmi_col, x_col])
    if df_clean.empty:
        return EntropyStats(0.0, 0.0, 0.0, 0.0, 0.0)

    x = df_clean[x_col].to_numpy(dtype=float)
    y = df_clean[pmi_col].to_numpy(dtype=float)

    A = np.vstack([x, np.ones_like(x)]).T
    beta, alpha = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = beta * x + alpha

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    mae = float(np.mean(np.abs(y - y_pred)))
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    return EntropyStats(
        slope=float(beta),
        intercept=float(alpha),
        r2=float(r2),
        mae=mae,
        rmse=rmse,
    )


# -----------------------------------------------------------
#  Grafici (facoltativi, per ispezione visiva)
# -----------------------------------------------------------

def plot_entropy_vs_pmi(df: pd.DataFrame, col: str, title: str):
    df_clean = df.dropna(subset=["pmi_hours", col])
    if df_clean.empty:
        print(f"Nessun dato valido per il grafico {col}.")
        return

    x = df_clean["pmi_hours"].to_numpy(dtype=float)
    y = df_clean[col].to_numpy(dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    plt.figure()
    plt.plot(x, y, "o-")
    plt.xlabel("PMI (hours)")
    plt.ylabel(col)
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_multi_curves(df: pd.DataFrame):
    if "E_multi_map" in df.columns:
        plot_entropy_vs_pmi(df, "E_multi_map", "PMI vs E_multi_map")
    if "E_multi_cornea" in df.columns:
        plot_entropy_vs_pmi(df, "E_multi_cornea", "PMI vs E_multi_cornea")


# -----------------------------------------------------------
#  Analisi completa
# -----------------------------------------------------------

def compute_full_stats(csv_path: str = DEFAULT_DATASET) -> Tuple[pd.DataFrame, Dict[str, EntropyStats]]:
    """
    Legge il dataset, calcola E_multi (map + cornea) e le regressioni base.
    Ritorna:
      - DataFrame arricchito
      - dizionario di EntropyStats
    """
    df = load_dataset(csv_path)

    df = compute_z_and_multi(df, region="map")
    df = compute_z_and_multi(df, region="cornea")

    stats: Dict[str, EntropyStats] = {}
    stats["multi_map"] = regression_pmi_vs(df, "E_multi_map")
    stats["multi_cornea"] = regression_pmi_vs(df, "E_multi_cornea")
    stats["H_map"] = regression_pmi_vs(df, "H_total")
    stats["H_cornea"] = regression_pmi_vs(df, "H_total_cornea")

    return df, stats


# -----------------------------------------------------------
#  Salvataggio modello multientropico CORNEA
# -----------------------------------------------------------

def save_cornea_model(df: pd.DataFrame, stats: EntropyStats, out_json: str = MODEL_JSON):
    """
    Calcola medie e deviazioni standard delle 4 entropie corneali e
    salva un file JSON con:
      - mu e sigma delle 4 entropie
      - parametri del modello lineare su E_multi_cornea
    """
    cols = [
        "H_total_cornea",
        "H_glcm_cornea",
        "H_laplacian_cornea",
        "H_highfreq_cornea",
    ]
    mu = {}
    sigma = {}
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"Colonna mancante per il modello corneale: {c}")
        arr = df[c].to_numpy(dtype=float)
        m = float(arr.mean())
        s = float(arr.std(ddof=0))
        if s <= 0:
            s = 1.0
        mu[c] = m
        sigma[c] = s

    model_dict = {
        "model_type": "linear",
        "region": "cornea",
        "x_feature": "E_multi_cornea",
        "slope": stats.slope,
        "intercept": stats.intercept,
        "r2": stats.r2,
        "mae": stats.mae,
        "rmse": stats.rmse,
        "entropy_means": mu,
        "entropy_stds": sigma,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(model_dict, f, indent=4)
    print(f"[OK] Modello corneale salvato in: {out_json}")


# -----------------------------------------------------------
#  Salvataggio riassunto testuale
# -----------------------------------------------------------

def save_stats_summary(stats: Dict[str, EntropyStats],
                       scale_map: str,
                       scale_cornea: str,
                       out_txt: str = STATS_TXT):
    """
    Salva un riassunto leggibile dei risultati in un file di testo.
    """
    lines = []
    lines.append("EYE-MEP – Entropy Stats & Model Trainer\n")
    lines.append("=== Regressioni PMI vs entropia ===\n")
    for name, st in stats.items():
        lines.append(
            f"{name:12s} | "
            f"slope={st.slope:.4f}, intercept={st.intercept:.4f}, "
            f"R2={st.r2:.4f}, MAE={st.mae:.3f} h, RMSE={st.rmse:.3f} h"
        )
    lines.append("\n")
    lines.append(scale_map)
    lines.append("\n")
    lines.append(scale_cornea)
    lines.append("\n")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Riassunto statistico salvato in: {out_txt}")


# -----------------------------------------------------------
#  Entry point (uso da console)
# -----------------------------------------------------------

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, DEFAULT_DATASET)

    df, stats = compute_full_stats(csv_path)

    print("=== ENTROPY STATS (EYE-MEP Trainer) ===")
    for name, st in stats.items():
        print(
            f"{name:12s} | "
            f"slope={st.slope:.4f}, intercept={st.intercept:.4f}, "
            f"R2={st.r2:.4f}, MAE={st.mae:.3f} h, RMSE={st.rmse:.3f} h"
        )

    # scale entropiche
    scale_map = check_entropy_scales(df, region="map")
    scale_cornea = check_entropy_scales(df, region="cornea")
    print()
    print(scale_map)
    print()
    print(scale_cornea)

    # salvataggio modello cornea (multientropia)
    save_cornea_model(df, stats["multi_cornea"], out_json=os.path.join(base_dir, MODEL_JSON))

    # salvataggio riassunto testuale
    save_stats_summary(
        stats,
        scale_map,
        scale_cornea,
        out_txt=os.path.join(base_dir, STATS_TXT),
    )

    # se vuoi vedere le curve:
    # plot_multi_curves(df)
