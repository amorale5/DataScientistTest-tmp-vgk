"""
Politica de aprobacion basada en la economia del producto, usando el modelo
de regresion logistica de la Ronda 3 (SMOTE + correccion de calibracion,
sin tasa_interes_anual). Misma logica que code/round1/approval_policy.py.
"""
import json
import numpy as np
import pandas as pd
import joblib
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from custom_transformers import GroupMedianImputer, ConstantFillImputer, PriorCorrectedClassifier  # noqa: F401
from train_models import RAW_FEATURE_COLS

PROJECT = "/Users/onlive/Library/CloudStorage/GoogleDrive-amatias.morales25@gmail.com/Mi unidad/respaldo-matias-agm/Trabajos Personales/postulacion-bice/DataScientistTest"
DATA_DIR = f"{PROJECT}/data"
MODELS_DIR = f"{PROJECT}/models/round3"
REPORTS_DIR = f"{PROJECT}/reports"

TASA_ANUAL = 0.12
LGD = 0.55


def add_economics(df):
    df = df.copy()
    df["ganancia_si_paga"] = df["monto_solicitado"] * TASA_ANUAL * (df["plazo_meses"] / 12) * 0.5
    df["perdida_si_default"] = df["monto_solicitado"] * LGD
    df["p_umbral"] = df["ganancia_si_paga"] / (df["ganancia_si_paga"] + df["perdida_si_default"])
    return df


def gain_curve(df, p_col, thresholds):
    out = []
    for t in thresholds:
        aprobar = df[p_col] < t
        valor = np.where(aprobar,
                          np.where(df["default_12m"] == 1, -df["perdida_si_default"], df["ganancia_si_paga"]),
                          0.0)
        out.append({"threshold": float(t), "gain": float(valor.sum()), "pct_approved": float(aprobar.mean())})
    return out


def main():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    train["fecha_solicitud"] = pd.to_datetime(train["fecha_solicitud"])
    va = train[train["fecha_solicitud"] >= "2025-01-01"].copy()
    va = add_economics(va)

    model_oot = joblib.load(f"{MODELS_DIR}/logit_oot_train_window.joblib")
    va["p_default"] = model_oot.predict_proba(va[RAW_FEATURE_COLS])[:, 1]
    va["aprobar"] = va["p_default"] < va["p_umbral"]

    va["valor_politica"] = np.where(va["aprobar"],
                                     np.where(va["default_12m"] == 1, -va["perdida_si_default"], va["ganancia_si_paga"]),
                                     0.0)
    va["valor_aprobar_todo"] = np.where(va["default_12m"] == 1, -va["perdida_si_default"], va["ganancia_si_paga"])

    n = len(va)
    ganancia_politica = float(va["valor_politica"].sum())
    ganancia_aprobar_todo = float(va["valor_aprobar_todo"].sum())
    n_aprob = int(va["aprobar"].sum())

    mask_ap = va["aprobar"]
    desglose = {
        "perdida_evitada": float(va.loc[(~mask_ap) & (va["default_12m"] == 1), "perdida_si_default"].sum()),
        "n_perdida_evitada": int(((~mask_ap) & (va["default_12m"] == 1)).sum()),
        "costo_oportunidad": float(va.loc[(~mask_ap) & (va["default_12m"] == 0), "ganancia_si_paga"].sum()),
        "n_costo_oportunidad": int(((~mask_ap) & (va["default_12m"] == 0)).sum()),
        "perdida_no_detectada": float(va.loc[mask_ap & (va["default_12m"] == 1), "perdida_si_default"].sum()),
        "n_perdida_no_detectada": int((mask_ap & (va["default_12m"] == 1)).sum()),
        "ganancia_capturada": float(va.loc[mask_ap & (va["default_12m"] == 0), "ganancia_si_paga"].sum()),
        "n_ganancia_capturada": int((mask_ap & (va["default_12m"] == 0)).sum()),
    }

    thresholds = np.arange(0.02, 0.55, 0.02)
    curve_fixed = gain_curve(va, "p_default", thresholds)

    plazos = sorted(va["plazo_meses"].unique().tolist())
    tabla_umbral = []
    for pl in plazos:
        g = 1_000_000 * TASA_ANUAL * (pl / 12) * 0.5
        l = 1_000_000 * LGD
        tabla_umbral.append({"plazo_meses": int(pl), "p_umbral": float(g / (g + l))})

    hist_counts, hist_edges = np.histogram(va["p_default"], bins=30, range=(0, min(1.0, va["p_default"].quantile(0.995))))

    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    test = add_economics(test)
    model_final = joblib.load(f"{MODELS_DIR}/logit_final_full_train.joblib")
    test["p_default"] = model_final.predict_proba(test[RAW_FEATURE_COLS])[:, 1]
    test["aprobar"] = test["p_default"] < test["p_umbral"]
    test["ev_aprobar"] = (1 - test["p_default"]) * test["ganancia_si_paga"] - test["p_default"] * test["perdida_si_default"]

    ganancia_esperada_politica_test = float(np.where(test["aprobar"], test["ev_aprobar"], 0.0).sum())
    ganancia_esperada_aprobar_todo_test = float(test["ev_aprobar"].sum())
    n_test = len(test)
    n_aprob_test = int(test["aprobar"].sum())

    hist_counts_test, hist_edges_test = np.histogram(test["p_default"], bins=30, range=(0, min(1.0, test["p_default"].quantile(0.995))))

    out_cols = ["id_solicitud", "p_default", "p_umbral", "aprobar", "monto_solicitado", "plazo_meses"]
    test_out = test[out_cols].rename(columns={"aprobar": "recomendacion_aprobar"})
    test_out.to_csv(f"{REPORTS_DIR}/round3_politica_aprobacion_test.csv", index=False)

    out = {
        "n_valid": n, "n_aprob_valid": n_aprob,
        "ganancia_politica_valid": ganancia_politica,
        "ganancia_aprobar_todo_valid": ganancia_aprobar_todo,
        "desglose": desglose,
        "curve_fixed_threshold": curve_fixed,
        "tabla_umbral_por_plazo": tabla_umbral,
        "hist_p_valid": {"counts": hist_counts.tolist(), "edges": hist_edges.tolist()},
        "n_test": n_test, "n_aprob_test": n_aprob_test,
        "ganancia_esperada_politica_test": ganancia_esperada_politica_test,
        "ganancia_esperada_aprobar_todo_test": ganancia_esperada_aprobar_todo_test,
        "hist_p_test": {"counts": hist_counts_test.tolist(), "edges": hist_edges_test.tolist()},
    }
    with open(f"{REPORTS_DIR}/policy_report_data_round3.json", "w") as f:
        json.dump(out, f, indent=2, allow_nan=True)

    print(f"Validacion: politica=${ganancia_politica:,.0f}  aprobar_todo=${ganancia_aprobar_todo:,.0f}  aprobadas={n_aprob}/{n} ({n_aprob/n*100:.1f}%)")
    print(f"Test.csv (esperado): politica=${ganancia_esperada_politica_test:,.0f}  aprobar_todo=${ganancia_esperada_aprobar_todo_test:,.0f}  aprobadas={n_aprob_test}/{n_test} ({n_aprob_test/n_test*100:.1f}%)")


if __name__ == "__main__":
    main()
