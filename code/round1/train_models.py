"""
Ronda 1 de modelamiento — score de probabilidad de default a 12 meses.

Reglas de esta ronda (indicadas por el usuario):
- Se usan TODOS los datos de train.csv tal cual vienen, SIN corregir ninguna
  inconsistencia (edad fuera de [18,100], ingreso_declarado < 200.000,
  tasa_interes_anual sobre la TMC, etc.). Esas correcciones se aplican recien
  en la Ronda 3.
- antiguedad_laboral_meses y antiguedad_cliente_meses se consideran variables
  SIN inconsistencia (no se tocan ademas de la imputacion de nulos).
- Los unicos campos con nulos son ingreso_declarado (18,11%) y
  antiguedad_laboral_meses (16,11%). La estrategia de imputacion se decidio
  en imputation_bakeoff.py: las 3 recetas probadas dan AUC practicamente
  identico (diferencia ~0,00006, dentro del ruido), asi que se elige
  'group_median_tipo_empleo' por ser la mas defendible en negocio, no por
  ganar en performance.
- num_contactos_ult_trimestre se excluye: fuga de informacion confirmada en
  el EDA/multivariado (IV=4,52, Cramer's V=0,80, correlacion 0,73 con el
  target — 9x el umbral de sospecha de fuga).
- dia_semana_solicitud se excluye: 85,7% no coincide con el dia real de
  fecha_solicitud y no tiene asociacion estadistica con el target.
- Validacion temporal: ventana de entrenamiento para seleccion de
  hiperparametros = fecha_solicitud < 2025-01-01; ventana de validacion =
  fecha_solicitud >= 2025-01-01 (resto de train.csv). Una vez elegidos los
  hiperparametros, el modelo final que se guarda y se usa para scorear
  test.csv se re-entrena con el 100% de train.csv (practica estandar).
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import MissingIndicator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from lightgbm import LGBMClassifier

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from custom_transformers import GroupMedianImputer, ConstantFillImputer

PROJECT = "/Users/onlive/Library/CloudStorage/GoogleDrive-amatias.morales25@gmail.com/Mi unidad/respaldo-matias-agm/Trabajos Personales/postulacion-bice/DataScientistTest"
DATA_DIR = f"{PROJECT}/data"
MODELS_DIR = f"{PROJECT}/models/round1"
REPORT_DATA_PATH = f"{PROJECT}/reports/round1_report_data.json"  # tmp, se copia despues

TARGET = "default_12m"
SPLIT_DATE = pd.Timestamp("2025-01-01")

CAT_VARS = ["tipo_empleo", "region", "canal"]
NUM_LOG_PLAIN = ["deuda_sistema", "monto_solicitado"]          # log1p, sin nulos
NUM_PLAIN = ["edad", "antiguedad_cliente_meses", "score_buro", "num_creditos_vigentes",
             "peor_morosidad_12m", "num_consultas_buro_3m", "uso_linea_credito_pct",
             "plazo_meses", "tasa_interes_anual"]
# ingreso_declarado y antiguedad_laboral_meses se tratan aparte (imputacion + log)

RAW_FEATURE_COLS = (["ingreso_declarado", "tipo_empleo", "antiguedad_laboral_meses"]
                     + NUM_LOG_PLAIN + NUM_PLAIN + CAT_VARS)
RAW_FEATURE_COLS = list(dict.fromkeys(RAW_FEATURE_COLS))  # dedup preservando orden


def load_data():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    train["fecha_solicitud"] = pd.to_datetime(train["fecha_solicitud"])
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    test["fecha_solicitud"] = pd.to_datetime(test["fecha_solicitud"])
    return train, test


def build_preprocessor():
    ingreso_branch = Pipeline([
        ("impute", GroupMedianImputer()),
        ("log1p", FunctionTransformer(np.log1p)),
    ])
    antig_branch = Pipeline([
        ("impute", ConstantFillImputer(fill_value=0.0)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("flags", MissingIndicator(features="all"), ["ingreso_declarado", "antiguedad_laboral_meses"]),
            ("ingreso", ingreso_branch, ["ingreso_declarado", "tipo_empleo"]),
            ("antig", antig_branch, ["antiguedad_laboral_meses"]),
            ("log_others", FunctionTransformer(np.log1p), NUM_LOG_PLAIN),
            ("plain", "passthrough", NUM_PLAIN),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_VARS),
        ],
        remainder="drop",
    )
    return pre


def feature_names(pre: ColumnTransformer):
    names = []
    names += ["flag_ingreso_declarado_nulo", "flag_antiguedad_laboral_nulo"]
    names += ["log_ingreso_declarado"]
    names += ["antiguedad_laboral_meses"]
    names += [f"log_{c}" for c in NUM_LOG_PLAIN]
    names += NUM_PLAIN
    ohe = pre.named_transformers_["cat"]
    names += list(ohe.get_feature_names_out(CAT_VARS))
    return names


def ks_stat(y_true, y_score):
    df = pd.DataFrame({"y": y_true, "s": y_score}).sort_values("s")
    df["cum_bad"] = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    df["cum_good"] = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()
    return float((df["cum_bad"] - df["cum_good"]).abs().max())


def eval_metrics(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_stat(y_true, y_score),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "n": int(len(y_true)),
        "n_pos": int(np.sum(y_true)),
        "default_rate": float(np.mean(y_true)),
    }


def main():
    train, test = load_data()
    tr_win = train[train["fecha_solicitud"] < SPLIT_DATE]
    va_win = train[train["fecha_solicitud"] >= SPLIT_DATE]
    print(f"train window: {len(tr_win)}  valid window: {len(va_win)}  test.csv: {len(test)}")

    Xtr, ytr = tr_win[RAW_FEATURE_COLS], tr_win[TARGET].values
    Xva, yva = va_win[RAW_FEATURE_COLS], va_win[TARGET].values
    Xtest = test[RAW_FEATURE_COLS]
    Xfull, yfull = train[RAW_FEATURE_COLS], train[TARGET].values

    # ============== LOGISTIC REGRESSION: busqueda de hiperparametros ==============
    logit_candidates = []
    for C in [0.01, 0.1, 1.0, 10.0]:
        for class_weight in [None, "balanced"]:
            logit_candidates.append({"C": C, "class_weight": class_weight})

    logit_search = []
    for cand in logit_candidates:
        pre = build_preprocessor()
        pipe = Pipeline([
            ("pre", pre),
            ("scale", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=3000, C=cand["C"], class_weight=cand["class_weight"])),
        ])
        pipe.fit(Xtr, ytr)
        p_va = pipe.predict_proba(Xva)[:, 1]
        m = eval_metrics(yva, p_va)
        logit_search.append({**cand, **m})
        print("LOGIT", cand, "AUC=", round(m["auc"], 5), "KS=", round(m["ks"], 5))

    # Seleccion: entre los candidatos con class_weight=None (o balanced) el AUC/KS
    # es practicamente identico, pero "balanced" degrada el Brier score (calibracion)
    # de ~0.09 a ~0.17 -- las probabilidades dejan de ser interpretables como
    # probabilidad real, lo que rompe el calculo de ganancia esperada de la politica
    # de aprobacion. Por eso se selecciona el mejor AUC SOLO entre los candidatos
    # sin reponderar clases (class_weight=None), y se deja el resto en la tabla de
    # busqueda para mostrar el trade-off.
    logit_calibration_safe = [r for r in logit_search if r["class_weight"] is None]
    best_logit_cfg = max(logit_calibration_safe, key=lambda r: r["auc"])
    print("Mejor logistica (calibration-safe):", best_logit_cfg)

    # ============== LIGHTGBM: busqueda de hiperparametros ==============
    n_pos = ytr.sum(); n_neg = len(ytr) - n_pos
    spw_balanced = float(n_neg / n_pos)

    lgbm_candidates = [
        {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 300, "min_child_samples": 30, "scale_pos_weight": 1.0},
        {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 300, "min_child_samples": 30, "scale_pos_weight": spw_balanced},
        {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300, "min_child_samples": 30, "scale_pos_weight": 1.0},
        {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 500, "min_child_samples": 50, "scale_pos_weight": 1.0},
        {"num_leaves": 31, "learning_rate": 0.1,  "n_estimators": 200, "min_child_samples": 50, "scale_pos_weight": 1.0},
        {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 300, "min_child_samples": 20, "scale_pos_weight": 1.0},
        {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300, "min_child_samples": 100, "scale_pos_weight": 1.0},
        {"num_leaves": 31, "learning_rate": 0.02, "n_estimators": 800, "min_child_samples": 50, "scale_pos_weight": 1.0},
    ]
    lgbm_search = []
    for cand in lgbm_candidates:
        pre = build_preprocessor()
        clf = LGBMClassifier(objective="binary", random_state=42, verbosity=-1, **cand)
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        p_va = pipe.predict_proba(Xva)[:, 1]
        m = eval_metrics(yva, p_va)
        lgbm_search.append({**cand, **m})
        print("LGBM", cand, "AUC=", round(m["auc"], 5), "KS=", round(m["ks"], 5))

    # Mismo criterio que en logistica: scale_pos_weight != 1 mejora marginalmente
    # (o ni eso) el AUC/KS pero dispara el Brier score de ~0.09 a ~0.16 -- se
    # descarta por calibracion, no por ranking.
    lgbm_calibration_safe = [r for r in lgbm_search if r["scale_pos_weight"] == 1.0]
    best_lgbm_cfg = max(lgbm_calibration_safe, key=lambda r: r["auc"])
    print("Mejor LightGBM (calibration-safe):", best_lgbm_cfg)

    # ============== Modelos OOT (solo train window) para metricas honestas ==============
    pre_oot_logit = build_preprocessor()
    logit_oot = Pipeline([
        ("pre", pre_oot_logit), ("scale", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=3000, C=best_logit_cfg["C"], class_weight=best_logit_cfg["class_weight"])),
    ])
    logit_oot.fit(Xtr, ytr)
    p_va_logit = logit_oot.predict_proba(Xva)[:, 1]
    metrics_logit_oot = eval_metrics(yva, p_va_logit)

    pre_oot_lgbm = build_preprocessor()
    lgbm_cfg_clean = {k: v for k, v in best_lgbm_cfg.items() if k in
                       ["num_leaves", "learning_rate", "n_estimators", "min_child_samples", "scale_pos_weight"]}
    lgbm_oot = Pipeline([("pre", pre_oot_lgbm),
                          ("clf", LGBMClassifier(objective="binary", random_state=42, verbosity=-1, **lgbm_cfg_clean))])
    lgbm_oot.fit(Xtr, ytr)
    p_va_lgbm = lgbm_oot.predict_proba(Xva)[:, 1]
    metrics_lgbm_oot = eval_metrics(yva, p_va_lgbm)

    # ============== Modelos FINALES (100% train.csv) para deploy + test.csv ==============
    pre_final_logit = build_preprocessor()
    logit_final = Pipeline([
        ("pre", pre_final_logit), ("scale", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=3000, C=best_logit_cfg["C"], class_weight=best_logit_cfg["class_weight"])),
    ])
    logit_final.fit(Xfull, yfull)

    pre_final_lgbm = build_preprocessor()
    lgbm_final = Pipeline([("pre", pre_final_lgbm),
                            ("clf", LGBMClassifier(objective="binary", random_state=42, verbosity=-1, **lgbm_cfg_clean))])
    lgbm_final.fit(Xfull, yfull)

    p_test_logit = logit_final.predict_proba(Xtest)[:, 1]
    p_test_lgbm = lgbm_final.predict_proba(Xtest)[:, 1]

    # feature importance / coeficientes (del modelo FINAL, 100% de train.csv)
    feat_names_logit = feature_names(logit_final.named_steps["pre"])
    coefs = logit_final.named_steps["clf"].coef_[0]
    logit_coefs = sorted(
        [{"feature": f, "coef": float(c)} for f, c in zip(feat_names_logit, coefs)],
        key=lambda x: -abs(x["coef"]),
    )

    feat_names_lgbm = feature_names(lgbm_final.named_steps["pre"])
    importances = lgbm_final.named_steps["clf"].feature_importances_
    lgbm_importance = sorted(
        [{"feature": f, "importance": int(v)} for f, v in zip(feat_names_lgbm, importances)],
        key=lambda x: -x["importance"],
    )

    # ============== Guardar artefactos ==============
    joblib.dump(logit_oot, f"{MODELS_DIR}/logit_oot_train_window.joblib")
    joblib.dump(lgbm_oot, f"{MODELS_DIR}/lgbm_oot_train_window.joblib")
    joblib.dump(logit_final, f"{MODELS_DIR}/logit_final_full_train.joblib")
    joblib.dump(lgbm_final, f"{MODELS_DIR}/lgbm_final_full_train.joblib")
    with open(f"{MODELS_DIR}/feature_columns.json", "w") as f:
        json.dump({"raw_feature_cols": RAW_FEATURE_COLS, "target": TARGET,
                    "engineered_feature_names_logit": feat_names_logit,
                    "engineered_feature_names_lgbm": feat_names_lgbm}, f, indent=2)

    # ============== Datos para el reporte HTML ==============
    out = {
        "split_date": str(SPLIT_DATE.date()),
        "n_train_window": len(tr_win), "n_valid_window": len(va_win), "n_test": len(test),
        "n_full_train": len(train),
        "default_rate_train_window": float(ytr.mean()),
        "default_rate_valid_window": float(yva.mean()),
        "imputation_recipe": "group_median_tipo_empleo",
        "excluded_features": ["num_contactos_ult_trimestre (fuga confirmada)", "dia_semana_solicitud (no confiable)"],
        "logit_search": logit_search,
        "lgbm_search": lgbm_search,
        "best_logit_cfg": best_logit_cfg,
        "best_lgbm_cfg": best_lgbm_cfg,
        "metrics_logit_oot": metrics_logit_oot,
        "metrics_lgbm_oot": metrics_lgbm_oot,
        "logit_coefs": logit_coefs,
        "lgbm_importance": lgbm_importance,
        "score_valid_logit": {"y": yva.tolist(), "score": p_va_logit.tolist()},
        "score_valid_lgbm": {"y": yva.tolist(), "score": p_va_lgbm.tolist()},
        "score_test_logit": p_test_logit.tolist(),
        "score_test_lgbm": p_test_lgbm.tolist(),
    }
    with open(REPORT_DATA_PATH, "w") as f:
        json.dump(out, f, indent=2, allow_nan=True)
    print("\nOK -> ", REPORT_DATA_PATH)
    print("Logit OOT:", metrics_logit_oot)
    print("LGBM  OOT:", metrics_lgbm_oot)


if __name__ == "__main__":
    main()
