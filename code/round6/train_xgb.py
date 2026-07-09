"""
XGBoost para la Ronda 6 (descartada, calculada por completitud). Entrena
solo jul-dic 2024 (6 meses), valida ene-feb 2025 -- misma ventana de
validacion que Rondas 1/3/4, por lo tanto SI comparable en pesos.
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from train_models import (RAW_FEATURE_COLS, build_preprocessor, load_data, TARGET,
                           clean_inconsistencies, edad_median_valida, TRAIN_START, TRAIN_END)

PROJECT = "/Users/onlive/Library/CloudStorage/GoogleDrive-amatias.morales25@gmail.com/Mi unidad/respaldo-matias-agm/Trabajos Personales/postulacion-bice/DataScientistTest"
MODELS_DIR = f"{PROJECT}/models/round6"
REPORTS_DIR = f"{PROJECT}/reports"
TASA_ANUAL = 0.12
LGD = 0.55


def ks_stat(y_true, y_score):
    df = pd.DataFrame({"y": y_true, "s": y_score}).sort_values("s")
    df["cum_bad"] = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    df["cum_good"] = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()
    return float((df["cum_bad"] - df["cum_good"]).abs().max())


def eval_metrics(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    return {"auc": float(auc), "gini": float(2*auc-1), "ks": ks_stat(y_true, y_score),
            "pr_auc": float(average_precision_score(y_true, y_score)),
            "brier": float(brier_score_loss(y_true, y_score)),
            "n": int(len(y_true)), "n_pos": int(np.sum(y_true)), "default_rate": float(np.mean(y_true))}


def add_economics(df):
    df = df.copy()
    df["ganancia_si_paga"] = df["monto_solicitado"] * TASA_ANUAL * (df["plazo_meses"] / 12) * 0.5
    df["perdida_si_default"] = df["monto_solicitado"] * LGD
    df["p_umbral"] = df["ganancia_si_paga"] / (df["ganancia_si_paga"] + df["perdida_si_default"])
    return df


def policy_eval(df, p_col):
    mask_ap = df[p_col] < df["p_umbral"]
    valor = np.where(mask_ap, np.where(df[TARGET] == 1, -df["perdida_si_default"], df["ganancia_si_paga"]), 0.0)
    valor_todo = np.where(df[TARGET] == 1, -df["perdida_si_default"], df["ganancia_si_paga"])
    perdida_evitada = float(df.loc[(~mask_ap) & (df[TARGET] == 1), "perdida_si_default"].sum())
    return {"n": len(df), "n_aprob": int(mask_ap.sum()), "pct_aprob": float(mask_ap.mean()),
            "ganancia_politica": float(valor.sum()), "ganancia_aprobar_todo": float(valor_todo.sum()),
            "perdida_evitada": perdida_evitada}


def main():
    train, test = load_data()
    tr_win_raw = train[(train["fecha_solicitud"] >= TRAIN_START) & (train["fecha_solicitud"] < TRAIN_END)]
    va_win_raw = train[train["fecha_solicitud"] >= TRAIN_END]
    edad_fill_trwin = edad_median_valida(tr_win_raw)
    edad_fill_full = edad_median_valida(train)
    tr_win = clean_inconsistencies(tr_win_raw, edad_fill_trwin)
    va_win = clean_inconsistencies(va_win_raw, edad_fill_trwin)
    full = clean_inconsistencies(train, edad_fill_full)
    test_c = clean_inconsistencies(test, edad_fill_full)

    Xtr, ytr = tr_win[RAW_FEATURE_COLS], tr_win[TARGET].values
    Xva, yva = va_win[RAW_FEATURE_COLS], va_win[TARGET].values
    Xfull, yfull = full[RAW_FEATURE_COLS], full[TARGET].values
    Xtest = test_c[RAW_FEATURE_COLS]

    n_pos = ytr.sum(); n_neg = len(ytr) - n_pos
    spw_balanced = float(n_neg / n_pos)
    candidates = [
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300, "min_child_weight": 5, "scale_pos_weight": 1.0},
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300, "min_child_weight": 5, "scale_pos_weight": spw_balanced},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300, "min_child_weight": 5, "scale_pos_weight": 1.0},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 500, "min_child_weight": 10, "scale_pos_weight": 1.0},
        {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 200, "min_child_weight": 10, "scale_pos_weight": 1.0},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 300, "min_child_weight": 3, "scale_pos_weight": 1.0},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300, "min_child_weight": 20, "scale_pos_weight": 1.0},
        {"max_depth": 4, "learning_rate": 0.02, "n_estimators": 800, "min_child_weight": 10, "scale_pos_weight": 1.0},
    ]
    search = []
    for cand in candidates:
        pipe = Pipeline([("pre", build_preprocessor()),
                          ("clf", XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                                 random_state=42, tree_method="hist", **cand))])
        pipe.fit(Xtr, ytr)
        p_va = pipe.predict_proba(Xva)[:, 1]
        m = eval_metrics(yva, p_va)
        search.append({**cand, **m})
        print("XGB", cand, "AUC=", round(m["auc"], 5), "KS=", round(m["ks"], 5), "Brier=", round(m["brier"], 5))

    calib_safe = [r for r in search if r["scale_pos_weight"] == 1.0]
    best_cfg = max(calib_safe, key=lambda r: r["auc"])
    print("Mejor XGB (calibration-safe):", best_cfg)
    cfg_clean = {k: v for k, v in best_cfg.items() if k in ["max_depth", "learning_rate", "n_estimators", "min_child_weight", "scale_pos_weight"]}

    xgb_oot = Pipeline([("pre", build_preprocessor()),
                         ("clf", XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, tree_method="hist", **cfg_clean))])
    xgb_oot.fit(Xtr, ytr)
    p_va = xgb_oot.predict_proba(Xva)[:, 1]
    metrics_oot = eval_metrics(yva, p_va)

    xgb_final = Pipeline([("pre", build_preprocessor()),
                           ("clf", XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, tree_method="hist", **cfg_clean))])
    xgb_final.fit(Xfull, yfull)
    p_test = xgb_final.predict_proba(Xtest)[:, 1]

    joblib.dump(xgb_oot, f"{MODELS_DIR}/xgb_oot_train_window.joblib")
    joblib.dump(xgb_final, f"{MODELS_DIR}/xgb_final_full_train.joblib")

    va_econ = add_economics(va_win.copy())
    va_econ["p_default"] = p_va
    pol_valid = policy_eval(va_econ, "p_default")

    test_econ = add_economics(test_c.copy())
    test_econ["p_default"] = p_test
    ev = (1 - test_econ["p_default"]) * test_econ["ganancia_si_paga"] - test_econ["p_default"] * test_econ["perdida_si_default"]
    aprob_test = test_econ["p_default"] < test_econ["p_umbral"]
    ganancia_esperada_test = float(np.where(aprob_test, ev, 0.0).sum())

    out = {"round": 6, "model": "xgboost", "best_cfg": best_cfg, "metrics_oot": metrics_oot,
           "search": search, "policy_valid": pol_valid,
           "ganancia_esperada_test": ganancia_esperada_test, "pct_aprob_test": float(aprob_test.mean())}
    with open(f"{REPORTS_DIR}/xgb_summary_round6.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== RESULTADO XGB RONDA 6 ===")
    print("Metrics OOT:", metrics_oot)
    print("Policy valid:", pol_valid)
    print("Ganancia esperada test.csv:", ganancia_esperada_test, "% aprob test:", aprob_test.mean())


if __name__ == "__main__":
    main()
