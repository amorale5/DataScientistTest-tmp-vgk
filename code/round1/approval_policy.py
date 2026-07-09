"""
Politica de aprobacion basada en la economia del producto, usando el modelo
de regresion logistica de la Ronda 1.

Economia del producto (simplificada, segun el enunciado):
- Si paga: ganancia = monto_solicitado * 12% anual * (plazo_meses/12) * 0.5
- Si cae en default: perdida = monto_solicitado * 55% (LGD)
- Si se rechaza: 0

Valor esperado de aprobar una solicitud con probabilidad de default p:
EV(aprobar) = (1-p)*ganancia_si_paga - p*perdida_si_default
Se aprueba si EV(aprobar) > 0, lo que equivale a: p < p* = G/(G+L)
Como G = monto*0.06*(plazo/12) y L = monto*0.55, el monto se cancela: el
umbral optimo depende SOLO del plazo, no del monto.

Se valida con datos reales (default_12m observado) en la ventana de
validacion de la Ronda 1 (2025, nunca vista en el entrenamiento) usando el
modelo OOT (entrenado solo con 2024). Para test.csv (sin resultado
observado) se usa el modelo FINAL (entrenado con el 100% de train.csv) y se
reporta la ganancia ESPERADA (no realizada, ya que no hay resultado real).
"""
import json
import numpy as np
import pandas as pd
import joblib
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from custom_transformers import GroupMedianImputer, ConstantFillImputer  # noqa: F401 (necesario para unpickle)

PROJECT = "/Users/onlive/Library/CloudStorage/GoogleDrive-amatias.morales25@gmail.com/Mi unidad/respaldo-matias-agm/Trabajos Personales/postulacion-bice/DataScientistTest"
DATA_DIR = f"{PROJECT}/data"
MODELS_DIR = f"{PROJECT}/models/round1"
REPORTS_DIR = f"{PROJECT}/reports"

TASA_ANUAL = 0.12
LGD = 0.55

RAW_FEATURE_COLS = ["ingreso_declarado", "tipo_empleo", "antiguedad_laboral_meses",
                     "deuda_sistema", "monto_solicitado", "edad", "antiguedad_cliente_meses",
                     "score_buro", "num_creditos_vigentes", "peor_morosidad_12m",
                     "num_consultas_buro_3m", "uso_linea_credito_pct", "plazo_meses",
                     "tasa_interes_anual", "region", "canal"]


def add_economics(df):
    df = df.copy()
    df["ganancia_si_paga"] = df["monto_solicitado"] * TASA_ANUAL * (df["plazo_meses"] / 12) * 0.5
    df["perdida_si_default"] = df["monto_solicitado"] * LGD
    df["p_umbral"] = df["ganancia_si_paga"] / (df["ganancia_si_paga"] + df["perdida_si_default"])
    return df


def gain_curve(df, p_col, thresholds):
    """Ganancia total realizada si se usa un umbral FIJO (global) en vez del
    umbral optimo por plazo -- sirve para mostrar que la politica variable
    por plazo es al menos tan buena como cualquier umbral fijo razonable."""
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

    # umbral por plazo (tabla de referencia)
    plazos = sorted(va["plazo_meses"].unique().tolist())
    tabla_umbral = []
    for pl in plazos:
        g = 1_000_000 * TASA_ANUAL * (pl / 12) * 0.5
        l = 1_000_000 * LGD
        tabla_umbral.append({"plazo_meses": int(pl), "p_umbral": float(g / (g + l))})

    # distribucion de p_default en validacion (para el grafico)
    hist_counts, hist_edges = np.histogram(va["p_default"], bins=30, range=(0, min(1.0, va["p_default"].quantile(0.995))))

    # ================= aplicar a test.csv (modelo FINAL, 100% train.csv) =================
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

    # guardar csv de salida con la recomendacion (no es el predictions.csv final del desafio)
    out_cols = ["id_solicitud", "p_default", "p_umbral", "aprobar", "monto_solicitado", "plazo_meses"]
    test_out = test[out_cols].rename(columns={"aprobar": "recomendacion_aprobar"})
    test_out.to_csv(f"{REPORTS_DIR}/round1_politica_aprobacion_test.csv", index=False)

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
    with open(f"{REPORTS_DIR}/policy_report_data.json", "w") as f:
        json.dump(out, f, indent=2, allow_nan=True)

    print(f"Validacion: politica=${ganancia_politica:,.0f}  aprobar_todo=${ganancia_aprobar_todo:,.0f}  aprobadas={n_aprob}/{n} ({n_aprob/n*100:.1f}%)")
    print(f"Test.csv (esperado): politica=${ganancia_esperada_politica_test:,.0f}  aprobar_todo=${ganancia_esperada_aprobar_todo_test:,.0f}  aprobadas={n_aprob_test}/{n_test} ({n_aprob_test/n_test*100:.1f}%)")
    print("OK ->", f"{REPORTS_DIR}/policy_report_data.json")
    print("OK ->", f"{REPORTS_DIR}/round1_politica_aprobacion_test.csv")


if __name__ == "__main__":
    main()
