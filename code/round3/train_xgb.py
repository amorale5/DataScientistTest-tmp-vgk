"""
XGBoost para la Ronda 3, con el MISMO criterio de esta ronda: grid search
(arquitectura x nivel de SMOTE) + correccion bayesiana de probabilidad,
seleccion por AUC entre candidatos cuyo Brier no empeore mas de 10% frente
al candidato sin SMOTE de la misma arquitectura. Sin tasa_interes_anual.
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from train_models import RAW_FEATURE_COLS, load_data, build_pipeline, SPLIT_DATE, TARGET

PROJECT = "/Users/onlive/Library/CloudStorage/GoogleDrive-amatias.morales25@gmail.com/Mi unidad/respaldo-matias-agm/Trabajos Personales/postulacion-bice/DataScientistTest"
MODELS_DIR = f"{PROJECT}/models/round3"
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
    tr_win = train[train["fecha_solicitud"] < SPLIT_DATE]
    va_win = train[train["fecha_solicitud"] >= SPLIT_DATE]
    Xtr, ytr = tr_win[RAW_FEATURE_COLS], tr_win[TARGET].values
    Xva, yva = va_win[RAW_FEATURE_COLS], va_win[TARGET].values
    Xfull, yfull = train[RAW_FEATURE_COLS], train[TARGET].values
    Xtest = test[RAW_FEATURE_COLS]
    pi_true_tr = float(ytr.mean())
    pi_true_full = float(yfull.mean())

    base_cfgs = [
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300, "min_child_weight": 5},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300, "min_child_weight": 5},
    ]
    SAMPLING_STRATEGIES = [None, 0.3, 0.5, 0.7, 1.0]

    search = []
    for base_cfg in base_cfgs:
        for ss in SAMPLING_STRATEGIES:
            clf = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, tree_method="hist", **base_cfg)
            pipe = build_pipeline(clf, ss, pi_true_tr, scale=False)
            pipe.fit(Xtr, ytr)
            p_va = pipe.predict_proba(Xva)[:, 1]
            m = eval_metrics(yva, p_va)
            cand = {**base_cfg, "sampling_strategy": ss, **m}
            search.append(cand)
            print("XGB", {**base_cfg, "ss": ss}, "AUC=", round(m["auc"], 5), "KS=", round(m["ks"], 5), "Brier=", round(m["brier"], 5))

    by_arch = {}
    for r in search:
        key = (r["max_depth"], r["learning_rate"], r["n_estimators"], r["min_child_weight"])
        by_arch.setdefault(key, {})[r["sampling_strategy"]] = r
    calib_safe = []
    for key, by_ss in by_arch.items():
        baseline_brier = by_ss[None]["brier"]
        for ss, r in by_ss.items():
            if r["brier"] <= baseline_brier * 1.10:
                calib_safe.append(r)
    best_cfg = max(calib_safe, key=lambda r: r["auc"])
    print("Mejor XGB (calibration-safe):", best_cfg)
    cfg_clean = {k: v for k, v in best_cfg.items() if k in ["max_depth", "learning_rate", "n_estimators", "min_child_weight"]}
    ss = best_cfg["sampling_strategy"]

    xgb_oot = build_pipeline(XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, tree_method="hist", **cfg_clean), ss, pi_true_tr, scale=False)
    xgb_oot.fit(Xtr, ytr)
    p_va = xgb_oot.predict_proba(Xva)[:, 1]
    metrics_oot = eval_metrics(yva, p_va)

    xgb_final = build_pipeline(XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, tree_method="hist", **cfg_clean), ss, pi_true_full, scale=False)
    xgb_final.fit(Xfull, yfull)
    p_test = xgb_final.predict_proba(Xtest)[:, 1]

    joblib.dump(xgb_oot, f"{MODELS_DIR}/xgb_oot_train_window.joblib")
    joblib.dump(xgb_final, f"{MODELS_DIR}/xgb_final_full_train.joblib")

    va_econ = add_economics(va_win.copy())
    va_econ["p_default"] = p_va
    pol_valid = policy_eval(va_econ, "p_default")

    test_econ = add_economics(test.copy())
    test_econ["p_default"] = p_test
    ev = (1 - test_econ["p_default"]) * test_econ["ganancia_si_paga"] - test_econ["p_default"] * test_econ["perdida_si_default"]
    aprob_test = test_econ["p_default"] < test_econ["p_umbral"]
    ganancia_esperada_test = float(np.where(aprob_test, ev, 0.0).sum())

    out = {"round": 3, "model": "xgboost", "best_cfg": best_cfg, "metrics_oot": metrics_oot,
           "search": search, "policy_valid": pol_valid,
           "ganancia_esperada_test": ganancia_esperada_test, "pct_aprob_test": float(aprob_test.mean())}
    with open(f"{REPORTS_DIR}/xgb_summary_round3.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== RESULTADO XGB RONDA 3 ===")
    print("Metrics OOT:", metrics_oot)
    print("Policy valid:", pol_valid)
    print("Ganancia esperada test.csv:", ganancia_esperada_test, "% aprob test:", aprob_test.mean())


if __name__ == "__main__":
    main()
