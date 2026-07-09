"""
Politica de aprobacion para la Ronda 5 (descartada, calculada por
completitud). OJO: el modelo OOT de esta ronda se valida contra jul-dic 2024
(n=20.426), NO contra ene-feb 2025 como las demas rondas -- por lo tanto
esta "ganancia real con politica" NO es comparable directamente contra la
de las Rondas 1/3/4/6 (poblacion, periodo y tasa de default distintos).
"""
import json
import numpy as np
import pandas as pd
import joblib
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from custom_transformers import GroupMedianImputer, ConstantFillImputer  # noqa: F401
from train_models import RAW_FEATURE_COLS, clean_inconsistencies, edad_median_valida, TRAIN_START, TRAIN_END, VALID_END

PROJECT = "/Users/onlive/Library/CloudStorage/GoogleDrive-amatias.morales25@gmail.com/Mi unidad/respaldo-matias-agm/Trabajos Personales/postulacion-bice/DataScientistTest"
DATA_DIR = f"{PROJECT}/data"
MODELS_DIR = f"{PROJECT}/models/round5"
REPORTS_DIR = f"{PROJECT}/reports"

TASA_ANUAL = 0.12
LGD = 0.55


def add_economics(df):
    df = df.copy()
    df["ganancia_si_paga"] = df["monto_solicitado"] * TASA_ANUAL * (df["plazo_meses"] / 12) * 0.5
    df["perdida_si_default"] = df["monto_solicitado"] * LGD
    df["p_umbral"] = df["ganancia_si_paga"] / (df["ganancia_si_paga"] + df["perdida_si_default"])
    return df


def main():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    train["fecha_solicitud"] = pd.to_datetime(train["fecha_solicitud"])
    tr_win_raw = train[(train["fecha_solicitud"] >= TRAIN_START) & (train["fecha_solicitud"] < TRAIN_END)]
    va_raw = train[(train["fecha_solicitud"] >= TRAIN_END) & (train["fecha_solicitud"] < VALID_END)].copy()  # jul-dic 2024 en esta ronda

    edad_fill_trwin = edad_median_valida(tr_win_raw)
    va = clean_inconsistencies(va_raw, edad_fill_trwin)
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

    test_raw = pd.read_csv(f"{DATA_DIR}/test.csv")
    test_raw["fecha_solicitud"] = pd.to_datetime(test_raw["fecha_solicitud"])
    edad_fill_full = edad_median_valida(train)
    test = clean_inconsistencies(test_raw, edad_fill_full)
    test = add_economics(test)

    model_final = joblib.load(f"{MODELS_DIR}/logit_final_full_train.joblib")
    test["p_default"] = model_final.predict_proba(test[RAW_FEATURE_COLS])[:, 1]
    test["aprobar"] = test["p_default"] < test["p_umbral"]
    test["ev_aprobar"] = (1 - test["p_default"]) * test["ganancia_si_paga"] - test["p_default"] * test["perdida_si_default"]

    ganancia_esperada_politica_test = float(np.where(test["aprobar"], test["ev_aprobar"], 0.0).sum())
    ganancia_esperada_aprobar_todo_test = float(test["ev_aprobar"].sum())
    n_test = len(test)
    n_aprob_test = int(test["aprobar"].sum())

    out = {
        "ronda": 5,
        "nota": "Validacion sobre jul-dic 2024 (n=20426), NO ene-feb 2025 -- no comparable directamente con las otras rondas",
        "n_valid": n, "n_aprob_valid": n_aprob,
        "ganancia_politica_valid": ganancia_politica,
        "ganancia_aprobar_todo_valid": ganancia_aprobar_todo,
        "desglose": desglose,
        "n_test": n_test, "n_aprob_test": n_aprob_test,
        "ganancia_esperada_politica_test": ganancia_esperada_politica_test,
        "ganancia_esperada_aprobar_todo_test": ganancia_esperada_aprobar_todo_test,
    }
    with open(f"{REPORTS_DIR}/policy_summary_round5.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"RONDA 5 | Validacion (jul-dic 2024): politica=${ganancia_politica:,.0f}  aprobar_todo=${ganancia_aprobar_todo:,.0f}  aprobadas={n_aprob}/{n} ({n_aprob/n*100:.1f}%)  perdida_evitada=${desglose['perdida_evitada']:,.0f}")
    print(f"RONDA 5 | Test.csv (esperado): politica=${ganancia_esperada_politica_test:,.0f}  aprobar_todo=${ganancia_esperada_aprobar_todo_test:,.0f}  aprobadas={n_aprob_test}/{n_test} ({n_aprob_test/n_test*100:.1f}%)")


if __name__ == "__main__":
    main()
