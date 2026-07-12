# -----------------------------------------------------------
#  EYE_MEP_Trainer.py
#
#  - Pulisce automaticamente oea_pmi_dataset.csv
#  - Calcola entropia grezza e normalizzata (map + cornea)
#  - Allena regressione lineare e Random Forest
#  - Sceglie il modello migliore (R2 >= 0.75 → lineare, altrimenti RF)
#  - Salva:
#       * oea_pmi_dataset_cleaned.csv
#       * eye_mep_entropy_curves.csv
#       * eye_mep_training_report.txt
#       * eye_mep_model_global.json / closed / open
# -----------------------------------------------------------

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold

DEFAULT_DATASET = "oea_pmi_dataset.csv"
CLEANED_DATASET = "oea_pmi_dataset_cleaned.csv"
CURVES_CSV = "eye_mep_entropy_curves.csv"
REPORT_TXT = "eye_mep_training_report.txt"


# ------------------------ Data classes ---------------------

@dataclass
class ModelMetrics:
    r2: float
    rmse: float
    mae: float


@dataclass
class ModelResult:
    label: str            # "Global", "Closed", "Open"
    eye_state: str        # "mixed", "closed", "open"
    features: List[str]
    linear: ModelMetrics
    rf_holdout: ModelMetrics
    rf_cv_mean_r2: float
    rf_cv_std_r2: float
    feature_importances: Dict[str, float]
    chosen_model: str     # "linear" oppure "rf"
    chosen_metrics: ModelMetrics


# --------------- Entropia grezza e normalizzata ------------

def add_raw_and_norm_entropy(df: pd.DataFrame) -> pd.DataFrame:
    # MAPPA
    if "E_sum_map" in df.columns:
        raw_map = df["E_sum_map"].astype(float)
    else:
        needed = ["H_total", "H_glcm_map", "H_laplacian_map", "H_highfreq_map"]
        raw_map = (
            df["H_total"].astype(float)
            + df["H_glcm_map"].astype(float)
            + df["H_laplacian_map"].astype(float)
            + df["H_highfreq_map"].astype(float)
        ) if all(c in df.columns for c in needed) else None

    if raw_map is not None:
        df["E_raw_map"] = raw_map
        mu = raw_map.mean()
        sigma = raw_map.std(ddof=0)
        df["E_norm_map"] = (raw_map - mu) / sigma if sigma > 0 else 0.0

    # CORNEA
    if "E_sum_cornea" in df.columns:
        raw_cornea = df["E_sum_cornea"].astype(float)
    else:
        needed_c = [
            "H_total_cornea",
            "H_glcm_cornea",
            "H_laplacian_cornea",
            "H_highfreq_cornea",
        ]
        raw_cornea = (
            df["H_total_cornea"].astype(float)
            + df["H_glcm_cornea"].astype(float)
            + df["H_laplacian_cornea"].astype(float)
            + df["H_highfreq_cornea"].astype(float)
        ) if all(c in df.columns for c in needed_c) else None

    if raw_cornea is not None:
        df["E_raw_cornea"] = raw_cornea
        mu_c = raw_cornea.mean()
        sigma_c = raw_cornea.std(ddof=0)
        df["E_norm_cornea"] = (raw_cornea - mu_c) / sigma_c if sigma_c > 0 else 0.0

    return df


# ------------------- Loader + cleaning ---------------------

def load_and_clean(csv_path: str = DEFAULT_DATASET) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)

    num_cols = ["pmi_hours"] + [
        c for c in df.columns if c.startswith("H_") or c.startswith("E_")
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["pmi_hours"])

    dup_cols = ["cadaver_id", "eye_side", "eye_state", "pmi_hours"]
    dup_cols = [c for c in dup_cols if c in df.columns]
    if dup_cols:
        df = df.drop_duplicates(subset=dup_cols, keep="first")

    grp_cols = ["cadaver_id", "pmi_hours", "eye_state"]
    grp_cols = [c for c in grp_cols if c in df.columns]
    if grp_cols:
        df = df.groupby(grp_cols, as_index=False).head(2)

    df = add_raw_and_norm_entropy(df)
    df.to_csv(CLEANED_DATASET, index=False)
    return df, CLEANED_DATASET


# --------- Curve entropia vs PMI per i grafici -------------

def save_entropy_curves(df: pd.DataFrame, out_path: str = CURVES_CSV) -> None:
    cols_to_agg = []
    for c in ["E_raw_map", "E_norm_map", "E_raw_cornea", "E_norm_cornea"]:
        if c in df.columns:
            cols_to_agg.append(c)
    if not cols_to_agg:
        return

    group_cols = ["eye_state", "pmi_hours"]
    group_cols = [c for c in group_cols if c in df.columns]

    grouped = (
        df[group_cols + cols_to_agg]
        .groupby(group_cols)
        .agg(["mean", "std", "count"])
    )
    grouped.columns = [
        f"{var}_{stat}" for var, stat in grouped.columns.to_flat_index()
    ]
    grouped.reset_index(inplace=True)
    grouped.to_csv(out_path, index=False)


# ---------------- Training su subset -----------------------

def train_on_subset(label: str,
                    df: pd.DataFrame,
                    features: List[str]) -> Optional[ModelResult]:
    if len(df) < 10:
        return None

    X = df[features].copy()
    y = df["pmi_hours"].astype(float)
    mask = ~X.isna().any(axis=1) & ~y.isna()
    X, y = X[mask], y[mask]
    if len(X) < 10:
        return None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # --- Regressione lineare
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    y_pred_lr = lr.predict(X_te)
    linear_metrics = ModelMetrics(
        r2=float(r2_score(y_te, y_pred_lr)),
        rmse=float(np.sqrt(mean_squared_error(y_te, y_pred_lr))),
        mae=float(mean_absolute_error(y_te, y_pred_lr)),
    )

    # --- Random Forest
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    y_pred_rf = rf.predict(X_te)
    rf_holdout = ModelMetrics(
        r2=float(r2_score(y_te, y_pred_rf)),
        rmse=float(np.sqrt(mean_squared_error(y_te, y_pred_rf))),
        mae=float(mean_absolute_error(y_te, y_pred_rf)),
    )

    # --- RF cross-validation
    n_splits = min(5, len(X))
    if n_splits < 3:
        cv_mean, cv_std = float("nan"), float("nan")
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in kf.split(X):
            X_tr_cv, X_te_cv = X.iloc[train_idx], X.iloc[test_idx]
            y_tr_cv, y_te_cv = y.iloc[train_idx], y.iloc[test_idx]
            rf_cv = RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            )
            rf_cv.fit(X_tr_cv, y_tr_cv)
            y_pred_cv = rf_cv.predict(X_te_cv)
            scores.append(r2_score(y_te_cv, y_pred_cv))
        scores = np.asarray(scores, dtype=float)
        cv_mean, cv_std = float(scores.mean()), float(scores.std())

    importances = rf.feature_importances_
    feat_imp = {f: float(imp) for f, imp in zip(features, importances)}

    eye_state = str(df["eye_state"].iloc[0]) if "eye_state" in df.columns else "mixed"

    # --- Scelta automatica modello
    if linear_metrics.r2 >= 0.75:
        chosen_model = "linear"
        chosen_metrics = linear_metrics
    else:
        chosen_model = "rf"
        chosen_metrics = rf_holdout

    print(f"\n[{label}] Modello scelto: {chosen_model.upper()}")
    print(f"  R2={chosen_metrics.r2:.3f}, "
          f"RMSE={chosen_metrics.rmse:.2f} h, MAE={chosen_metrics.mae:.2f} h")

    return ModelResult(
        label=label,
        eye_state=eye_state,
        features=list(features),
        linear=linear_metrics,
        rf_holdout=rf_holdout,
        rf_cv_mean_r2=cv_mean,
        rf_cv_std_r2=cv_std,
        feature_importances=feat_imp,
        chosen_model=chosen_model,
        chosen_metrics=chosen_metrics,
    )


# -------------- Salvataggio modello scelto -----------------

def save_chosen_model_json(filename: str,
                           df: pd.DataFrame,
                           features: List[str],
                           result: ModelResult) -> None:
    X = df[features].copy()
    y = df["pmi_hours"].astype(float)
    mask = ~X.isna().any(axis=1) & ~y.isna()
    X, y = X[mask], y[mask]
    if len(X) < 5:
        return

    if result.chosen_model == "linear":
        model_type = "linear_regression"
        final = LinearRegression()
        final.fit(X, y)
        params = {
            "coefficients": {
                feat: float(coef) for feat, coef in zip(features, final.coef_)
            },
            "intercept": float(final.intercept_),
        }
    else:
        model_type = "random_forest"
        final = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )
        final.fit(X, y)
        params = {
            "feature_importances": {
                feat: float(imp)
                for feat, imp in zip(features, final.feature_importances_)
            }
        }

    stats = {}
    for f in features:
        col = X[f].astype(float)
        stats[f] = {
            "mean": float(col.mean()),
            "std": float(col.std(ddof=0)),
            "min": float(col.min()),
            "max": float(col.max()),
        }

    model_dict = {
        "model_type": model_type,
        "eye_state": result.eye_state,
        "features": features,
        "chosen_model": result.chosen_model,
        "chosen_metrics": asdict(result.chosen_metrics),
        "linear_metrics": asdict(result.linear),
        "rf_holdout_metrics": asdict(result.rf_holdout),
        "rf_cv": {
            "mean_r2": result.rf_cv_mean_r2,
            "std_r2": result.rf_cv_std_r2,
        },
        "feature_importances": result.feature_importances,
        "feature_statistics": stats,
        "training_info": {
            "n_samples": int(len(X)),
            "pmi_range": [float(y.min()), float(y.max())],
        },
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(model_dict, f, indent=2)


# ---------------- Report testuale --------------------------

def write_report(df: pd.DataFrame,
                 results: List[ModelResult],
                 report_path: str = REPORT_TXT) -> None:
    lines: List[str] = []

    lines.append("=== EYE_MEP Trainer – Report Statistico ===\n")
    lines.append(f"Righe dataset pulito: {len(df)}")
    if "cadaver_id" in df.columns:
        lines.append(f"Cadaveri unici: {df['cadaver_id'].nunique()}")
    if "eye_state" in df.columns:
        lines.append("Distribuzione eye_state:")
        lines.append(str(df["eye_state"].value_counts().to_dict()))
    lines.append("Distribuzione PMI:")
    lines.append(str(df["pmi_hours"].value_counts().sort_index().to_dict()))
    lines.append("")

    for res in results:
        lines.append(f"\n--- {res.label} (eye_state={res.eye_state}) ---")
        lines.append(f"Features: {', '.join(res.features)}")
        lines.append(
            f"Linear   → R2={res.linear.r2:.3f}, RMSE={res.linear.rmse:.2f} h, "
            f"MAE={res.linear.mae:.2f} h"
        )
        lines.append(
            f"RF hold-out → R2={res.rf_holdout.r2:.3f}, "
            f"RMSE={res.rf_holdout.rmse:.2f} h, MAE={res.rf_holdout.mae:.2f} h"
        )
        lines.append(
            f"RF CV (R2) → mean={res.rf_cv_mean_r2:.3f}, "
            f"std={res.rf_cv_std_r2:.3f}"
        )
        lines.append(
            f"Modello scelto: {res.chosen_model.upper()} "
            f"(R2={res.chosen_metrics.r2:.3f}, "
            f"RMSE={res.chosen_metrics.rmse:.2f} h, "
            f"MAE={res.chosen_metrics.mae:.2f} h)"
        )
        lines.append("Importanza variabili (RF):")
        for f, imp in sorted(res.feature_importances.items(),
                             key=lambda x: x[1], reverse=True):
            lines.append(f"  - {f}: {imp:.3f}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -------------------------- MAIN ---------------------------

def main(csv_path: str = DEFAULT_DATASET) -> None:
    print("=== EYE_MEP Trainer – cleaning + entropia grezza + modelli ===")

    df, cleaned_path = load_and_clean(csv_path)
    print(f"[OK] Dataset pulito salvato in: {cleaned_path}")
    print(f"Righe dopo cleaning: {len(df)}")

    save_entropy_curves(df, CURVES_CSV)
    print(f"[OK] Curve di entropia salvate in: {CURVES_CSV}")

    base_features = [
        "H_total",
        "H_glcm_map",
        "H_laplacian_map",
        "H_highfreq_map",
        "E_sum_map",
    ]
    features = [f for f in base_features if f in df.columns]
    if len(features) < 2:
        raise ValueError("Mancano colonne di entropia sufficienti per il training.")

    results: List[ModelResult] = []

    # Global
    res_global = train_on_subset("Global", df, features)
    if res_global:
        results.append(res_global)

    # Closed / Open
    if "eye_state" in df.columns:
        closed = df[df["eye_state"] == "closed"]
        open_ = df[df["eye_state"] == "open"]

        res_closed = train_on_subset("Closed", closed, features)
        if res_closed:
            results.append(res_closed)

        res_open = train_on_subset("Open", open_, features)
        if res_open:
            results.append(res_open)

    # Salva modelli
    if "eye_state" in df.columns:
        if res_global:
            save_chosen_model_json(
                "eye_mep_model_global.json", df, features, res_global
            )
            print("[OK] Modello GLOBAL salvato in eye_mep_model_global.json")

        if "eye_state" in df.columns and res_closed:
            save_chosen_model_json(
                "eye_mep_model_closed.json",
                df[df["eye_state"] == "closed"],
                features,
                res_closed,
            )
            print("[OK] Modello CLOSED salvato in eye_mep_model_closed.json")

        if "eye_state" in df.columns and res_open:
            save_chosen_model_json(
                "eye_mep_model_open.json",
                df[df["eye_state"] == "open"],
                features,
                res_open,
            )
            print("[OK] Modello OPEN salvato in eye_mep_model_open.json")

    write_report(df, results, REPORT_TXT)
    print(f"[OK] Report statistico salvato in: {REPORT_TXT}")


if __name__ == "__main__":
    main()
