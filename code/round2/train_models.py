"""
Ronda 2 de modelamiento — score de probabilidad de default a 12 meses.

Diferencia vs. Ronda 1: se agrega `mes_solicitud` (mes calendario, 1-12,
tratado como variable CATEGORICA/nominal, no ordinal) extraido de
fecha_solicitud. Esta variable esta disponible tanto en train.csv como en
test.csv (no es fuga), y los 12 meses del año quedan representados en 2024,
por lo que no hay categorias "nunca vistas" al aplicar el modelo a test.csv
(que cubre feb-jun 2025).

Regla explicita del usuario para esta ronda: el modelo NO debe entrenarse
con ningun dato de 2025 -- a diferencia de la Ronda 1 (donde el modelo final
se reentrenaba con el 100% de train.csv, incluyendo enero-febrero 2025 para
el deploy), aqui hay un UNICO modelo por algoritmo, entrenado solo con 2024
(fecha_solicitud < 2025-01-01), validado con el resto de train.csv (2025) y
ese mismo modelo (sin reentrenar) es el que se aplica a test.csv. Es una
eleccion mas conservadora que la Ronda 1: usa ~10% menos datos para el
modelo final, pero garantiza cero contacto con datos de 2025 durante el
entrenamiento, tal como se pidio.

Todo lo demas se mantiene igual que en la Ronda 1: mismos datos crudos (sin
tratar inconsistencias), misma receta de imputacion (mediana por tipo_empleo
para ingreso_declarado, 0 para antiguedad_laboral_meses + flags de nulidad),
misma exclusion de num_contactos_ult_trimestre (fuga) y dia_semana_solicitud
(no confiable), mismas transformaciones log1p, mismo criterio de seleccion
de hiperparametros "calibration-safe" (se descartan configuraciones que
reponderan clases porque arruinan el Brier score).
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
MODELS_DIR = f"{PROJECT}/models/round2"
REPORT_DATA_PATH = f"{PROJECT}/reports/round2_report_data.json"

TARGET = "default_12m"
SPLIT_DATE = pd.Timestamp("2025-01-01")

MESES_ES = {1:"01-enero",2:"02-febrero",3:"03-marzo",4:"04-abril",5:"05-mayo",6:"06-junio",
            7:"07-julio",8:"08-agosto",9:"09-septiembre",10:"10-octubre",11:"11-noviembre",12:"12-diciembre"}

CAT_VARS = ["tipo_empleo", "region", "canal", "mes_solicitud"]
NUM_LOG_PLAIN = ["deuda_sistema", "monto_solicitado"]
NUM_PLAIN = ["edad", "antiguedad_cliente_meses", "score_buro", "num_creditos_vigentes",
             "peor_morosidad_12m", "num_consultas_buro_3m", "uso_linea_credito_pct",
             "plazo_meses", "tasa_interes_anual"]

RAW_FEATURE_COLS = (["ingreso_declarado", "tipo_empleo", "antiguedad_laboral_meses"]
                     + NUM_LOG_PLAIN + NUM_PLAIN + CAT_VARS)
RAW_FEATURE_COLS = list(dict.fromkeys(RAW_FEATURE_COLS))


def add_mes_solicitud(df):
    df = df.copy()
    df["mes_solicitud"] = df["fecha_solicitud"].dt.month.map(MESES_ES)
    return df


def load_data():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    train["fecha_solicitud"] = pd.to_datetime(train["fecha_solicitud"])
    train = add_mes_solicitud(train)
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    test["fecha_solicitud"] = pd.to_datetime(test["fecha_solicitud"])
    test = add_mes_solicitud(test)
    return train, test


def build_preprocessor():
    ingreso_branch = Pipeline([
        ("impute", GroupMedianImputer()),
        ("log1p", FunctionTransformer(np.log1p)),
    ])
    antig_branch = Pipeline([("impute", ConstantFillImputer(fill_value=0.0))])
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
        "auc": float(auc), "gini": float(2 * auc - 1), "ks": ks_stat(y_true, y_score),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "n": int(len(y_true)), "n_pos": int(np.sum(y_true)), "default_rate": float(np.mean(y_true)),
    }


def main():
    train, test = load_data()
    tr_win = train[train["fecha_solicitud"] < SPLIT_DATE]
    va_win = train[train["fecha_solicitud"] >= SPLIT_DATE]
    assert tr_win["fecha_solicitud"].dt.year.max() == 2024, "Se coló data de 2025 en el set de entrenamiento"
    print(f"train (solo 2024): {len(tr_win)}  valid (2025): {len(va_win)}  test.csv: {len(test)}")
    print("Meses en train:", sorted(tr_win['mes_solicitud'].unique()))
    print("Meses en test:", sorted(test['mes_solicitud'].unique()))

    Xtr, ytr = tr_win[RAW_FEATURE_COLS], tr_win[TARGET].values
    Xva, yva = va_win[RAW_FEATURE_COLS], va_win[TARGET].values
    Xtest = test[RAW_FEATURE_COLS]

    # ============== LOGISTIC REGRESSION ==============
    logit_candidates = [{"C": C, "class_weight": cw} for C in [0.01, 0.1, 1.0, 10.0] for cw in [None, "balanced"]]
    logit_search = []
    for cand in logit_candidates:
        pre = build_preprocessor()
        pipe = Pipeline([("pre", pre), ("scale", StandardScaler(with_mean=False)),
                          ("clf", LogisticRegression(max_iter=3000, C=cand["C"], class_weight=cand["class_weight"]))])
        pipe.fit(Xtr, ytr)
        p_va = pipe.predict_proba(Xva)[:, 1]
        m = eval_metrics(yva, p_va)
        logit_search.append({**cand, **m})
        print("LOGIT", cand, "AUC=", round(m["auc"], 5), "KS=", round(m["ks"], 5))

    logit_calibration_safe = [r for r in logit_search if r["class_weight"] is None]
    best_logit_cfg = max(logit_calibration_safe, key=lambda r: r["auc"])
    print("Mejor logistica (calibration-safe):", best_logit_cfg)

    # ============== LIGHTGBM ==============
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

    lgbm_calibration_safe = [r for r in lgbm_search if r["scale_pos_weight"] == 1.0]
    best_lgbm_cfg = max(lgbm_calibration_safe, key=lambda r: r["auc"])
    print("Mejor LightGBM (calibration-safe):", best_lgbm_cfg)

    # ============== Modelos FINALES: entrenados SOLO con 2024 ==============
    pre_logit = build_preprocessor()
    logit_model = Pipeline([("pre", pre_logit), ("scale", StandardScaler(with_mean=False)),
                             ("clf", LogisticRegression(max_iter=3000, C=best_logit_cfg["C"], class_weight=best_logit_cfg["class_weight"]))])
    logit_model.fit(Xtr, ytr)
    p_va_logit = logit_model.predict_proba(Xva)[:, 1]
    metrics_logit = eval_metrics(yva, p_va_logit)

    lgbm_cfg_clean = {k: v for k, v in best_lgbm_cfg.items() if k in
                       ["num_leaves", "learning_rate", "n_estimators", "min_child_samples", "scale_pos_weight"]}
    pre_lgbm = build_preprocessor()
    lgbm_model = Pipeline([("pre", pre_lgbm), ("clf", LGBMClassifier(objective="binary", random_state=42, verbosity=-1, **lgbm_cfg_clean))])
    lgbm_model.fit(Xtr, ytr)
    p_va_lgbm = lgbm_model.predict_proba(Xva)[:, 1]
    metrics_lgbm = eval_metrics(yva, p_va_lgbm)

    p_test_logit = logit_model.predict_proba(Xtest)[:, 1]
    p_test_lgbm = lgbm_model.predict_proba(Xtest)[:, 1]

    feat_names_logit = feature_names(logit_model.named_steps["pre"])
    coefs = logit_model.named_steps["clf"].coef_[0]
    logit_coefs = sorted([{"feature": f, "coef": float(c)} for f, c in zip(feat_names_logit, coefs)], key=lambda x: -abs(x["coef"]))

    feat_names_lgbm = feature_names(lgbm_model.named_steps["pre"])
    importances = lgbm_model.named_steps["clf"].feature_importances_
    lgbm_importance = sorted([{"feature": f, "importance": int(v)} for f, v in zip(feat_names_lgbm, importances)], key=lambda x: -x["importance"])

    # ============== Guardar artefactos (un solo modelo por algoritmo) ==============
    joblib.dump(logit_model, f"{MODELS_DIR}/logit_2024_only.joblib")
    joblib.dump(lgbm_model, f"{MODELS_DIR}/lgbm_2024_only.joblib")
    with open(f"{MODELS_DIR}/feature_columns.json", "w") as f:
        json.dump({"raw_feature_cols": RAW_FEATURE_COLS, "target": TARGET,
                    "engineered_feature_names_logit": feat_names_logit,
                    "engineered_feature_names_lgbm": feat_names_lgbm,
                    "meses_categoria": list(MESES_ES.values())}, f, indent=2)

    out = {
        "split_date": str(SPLIT_DATE.date()),
        "n_train_window": len(tr_win), "n_valid_window": len(va_win), "n_test": len(test),
        "n_full_train": len(train),
        "default_rate_train_window": float(ytr.mean()),
        "default_rate_valid_window": float(yva.mean()),
        "imputation_recipe": "group_median_tipo_empleo",
        "excluded_features": ["num_contactos_ult_trimestre (fuga confirmada)", "dia_semana_solicitud (no confiable)"],
        "new_feature": "mes_solicitud (categorica, extraida de fecha_solicitud)",
        "meses_train": sorted(tr_win["mes_solicitud"].unique().tolist()),
        "meses_test": sorted(test["mes_solicitud"].unique().tolist()),
        "logit_search": logit_search, "lgbm_search": lgbm_search,
        "best_logit_cfg": best_logit_cfg, "best_lgbm_cfg": best_lgbm_cfg,
        "metrics_logit_oot": metrics_logit, "metrics_lgbm_oot": metrics_lgbm,
        "logit_coefs": logit_coefs, "lgbm_importance": lgbm_importance,
        "score_valid_logit": {"y": yva.tolist(), "score": p_va_logit.tolist()},
        "score_valid_lgbm": {"y": yva.tolist(), "score": p_va_lgbm.tolist()},
        "score_test_logit": p_test_logit.tolist(), "score_test_lgbm": p_test_lgbm.tolist(),
    }
    with open(REPORT_DATA_PATH, "w") as f:
        json.dump(out, f, indent=2, allow_nan=True)
    print("\nOK -> ", REPORT_DATA_PATH)
    print("Logit:", metrics_logit)
    print("LGBM :", metrics_lgbm)


if __name__ == "__main__":
    main()
