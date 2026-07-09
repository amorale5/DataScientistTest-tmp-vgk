"""
Ronda 3 de modelamiento — score de probabilidad de default a 12 meses.

Diferencias vs. Ronda 1:
1. Se elimina tasa_interes_anual: la tasa la fija la propia empresa segun su
   politica de pricing, por lo que probablemente ya incorpora una evaluacion
   de riesgo previa (circularidad) -- se prueba el modelo sin ella.
2. Se trata el desbalanceo de clases (~9,5% de default) con SMOTE
   (generacion de datos sinteticos de la clase minoritaria por interpolacion
   entre vecinos), en vez de dejarlo como esta (Ronda 1) o de reponderar
   clases (que ya se probo y se descarto en la Ronda 1 por daniar la
   calibracion).

Punto tecnico importante: SMOTE tiene el MISMO problema de fondo que
class_weight='balanced' -- cambia la prevalencia con la que se entrena el
modelo, asi que sus probabilidades de salida quedan descalibradas respecto
a la tasa de default real. A diferencia de la Ronda 1 (donde se descartaron
las variantes reponderadas), aqui se aplica una correccion de calibracion
posterior (correccion bayesiana de prior, ver custom_transformers.py) que
devuelve las probabilidades a la escala real. Esta correccion es una
transformacion monotona: no cambia AUC/KS, solo corrige el Brier score.

Todo lo demas igual que la Ronda 1: datos crudos (sin tratar
inconsistencias), misma receta de imputacion, mismas exclusiones
(num_contactos_ult_trimestre por fuga, dia_semana_solicitud por no
confiable), particion temporal train<2025 / validacion 2025, modelo OOT
(solo 2024) para metricas honestas + modelo final (100% train.csv) para
test.csv.
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import MissingIndicator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from custom_transformers import GroupMedianImputer, ConstantFillImputer, PriorCorrectedClassifier

PROJECT = "/Users/onlive/Library/CloudStorage/GoogleDrive-amatias.morales25@gmail.com/Mi unidad/respaldo-matias-agm/Trabajos Personales/postulacion-bice/DataScientistTest"
DATA_DIR = f"{PROJECT}/data"
MODELS_DIR = f"{PROJECT}/models/round3"
REPORT_DATA_PATH = f"{PROJECT}/reports/round3_report_data.json"

TARGET = "default_12m"
SPLIT_DATE = pd.Timestamp("2025-01-01")

CAT_VARS = ["tipo_empleo", "region", "canal"]
NUM_LOG_PLAIN = ["deuda_sistema", "monto_solicitado"]
# tasa_interes_anual eliminada respecto a la Ronda 1
NUM_PLAIN = ["edad", "antiguedad_cliente_meses", "score_buro", "num_creditos_vigentes",
             "peor_morosidad_12m", "num_consultas_buro_3m", "uso_linea_credito_pct", "plazo_meses"]

RAW_FEATURE_COLS = (["ingreso_declarado", "tipo_empleo", "antiguedad_laboral_meses"]
                     + NUM_LOG_PLAIN + NUM_PLAIN + CAT_VARS)
RAW_FEATURE_COLS = list(dict.fromkeys(RAW_FEATURE_COLS))


def load_data():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    train["fecha_solicitud"] = pd.to_datetime(train["fecha_solicitud"])
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    test["fecha_solicitud"] = pd.to_datetime(test["fecha_solicitud"])
    return train, test


def build_preprocessor():
    ingreso_branch = ImbPipeline([
        ("impute", GroupMedianImputer()),
        ("log1p", FunctionTransformer(np.log1p)),
    ])
    antig_branch = ImbPipeline([("impute", ConstantFillImputer(fill_value=0.0))])
    pre = ColumnTransformer(
        transformers=[
            ("flags", MissingIndicator(features="all"), ["ingreso_declarado", "antiguedad_laboral_meses"]),
            ("ingreso", ingreso_branch, ["ingreso_declarado", "tipo_empleo"]),
            ("antig", antig_branch, ["antiguedad_laboral_meses"]),
            ("log_others", FunctionTransformer(np.log1p), NUM_LOG_PLAIN),
            ("plain", "passthrough", NUM_PLAIN),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_VARS),
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


def build_pipeline(base_clf, sampling_strategy, pi_true, scale=False):
    steps = [("pre", build_preprocessor())]
    if sampling_strategy is not None:
        steps.append(("smote", SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5)))
    if scale:
        steps.append(("scale", StandardScaler(with_mean=False)))
    steps.append(("clf", PriorCorrectedClassifier(base_estimator=base_clf, pi_true=pi_true)))
    return ImbPipeline(steps)


def main():
    train, test = load_data()
    tr_win = train[train["fecha_solicitud"] < SPLIT_DATE]
    va_win = train[train["fecha_solicitud"] >= SPLIT_DATE]
    assert tr_win["fecha_solicitud"].dt.year.max() == 2024

    Xtr, ytr = tr_win[RAW_FEATURE_COLS], tr_win[TARGET].values
    Xva, yva = va_win[RAW_FEATURE_COLS], va_win[TARGET].values
    Xtest = test[RAW_FEATURE_COLS]
    Xfull, yfull = train[RAW_FEATURE_COLS], train[TARGET].values

    pi_true_tr = float(ytr.mean())
    pi_true_full = float(yfull.mean())

    SAMPLING_STRATEGIES = [None, 0.3, 0.5, 0.7, 1.0]

    # ============== LOGISTIC REGRESSION ==============
    logit_search = []
    for C in [0.01, 0.1, 1.0]:
        for ss in SAMPLING_STRATEGIES:
            pipe = build_pipeline(LogisticRegression(max_iter=3000, C=C), ss, pi_true_tr, scale=True)
            pipe.fit(Xtr, ytr)
            p_va = pipe.predict_proba(Xva)[:, 1]
            m = eval_metrics(yva, p_va)
            cand = {"C": C, "sampling_strategy": ss, **m}
            logit_search.append(cand)
            print("LOGIT", {"C": C, "ss": ss}, "AUC=", round(m["auc"], 5), "KS=", round(m["ks"], 5), "Brier=", round(m["brier"], 5))

    best_logit_cfg = max(logit_search, key=lambda r: r["auc"])
    print("Mejor logistica:", best_logit_cfg)

    # ============== LIGHTGBM ==============
    lgbm_base_cfgs = [
        {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 300, "min_child_samples": 30},
        {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300, "min_child_samples": 30},
    ]
    lgbm_search = []
    for base_cfg in lgbm_base_cfgs:
        for ss in SAMPLING_STRATEGIES:
            clf = LGBMClassifier(objective="binary", random_state=42, verbosity=-1, **base_cfg)
            pipe = build_pipeline(clf, ss, pi_true_tr, scale=False)
            pipe.fit(Xtr, ytr)
            p_va = pipe.predict_proba(Xva)[:, 1]
            m = eval_metrics(yva, p_va)
            cand = {**base_cfg, "sampling_strategy": ss, **m}
            lgbm_search.append(cand)
            print("LGBM", {**base_cfg, "ss": ss}, "AUC=", round(m["auc"], 5), "KS=", round(m["ks"], 5), "Brier=", round(m["brier"], 5))

    # A diferencia de la logistica (donde la correccion de prior deja el Brier
    # practicamente intacto en todos los sampling_strategy), en LightGBM el
    # Brier empeora fuertemente con SMOTE incluso despues de corregir (de
    # ~0.090 a 0.10-0.12) -- la correccion bayesiana asume que el oversampling
    # no cambia la distribucion de las variables dentro de cada clase, y esa
    # aproximacion se rompe con arboles (los splits pueden explotar la
    # geometria de los puntos sinteticos de SMOTE de forma no lineal). Por
    # eso, para LightGBM se exige ademas que el Brier no empeore mas de 10%
    # respecto al candidato sin SMOTE de la misma arquitectura.
    lgbm_by_arch = {}
    for r in lgbm_search:
        key = (r["num_leaves"], r["learning_rate"], r["n_estimators"], r["min_child_samples"])
        lgbm_by_arch.setdefault(key, {})[r["sampling_strategy"]] = r
    lgbm_calibration_safe = []
    for key, by_ss in lgbm_by_arch.items():
        baseline_brier = by_ss[None]["brier"]
        for ss, r in by_ss.items():
            if r["brier"] <= baseline_brier * 1.10:
                lgbm_calibration_safe.append(r)
    best_lgbm_cfg = max(lgbm_calibration_safe, key=lambda r: r["auc"])
    print("Mejor LightGBM (calibration-safe):", best_lgbm_cfg)

    # ============== Modelos OOT (solo 2024) para metricas honestas ==============
    logit_oot = build_pipeline(LogisticRegression(max_iter=3000, C=best_logit_cfg["C"]),
                                best_logit_cfg["sampling_strategy"], pi_true_tr, scale=True)
    logit_oot.fit(Xtr, ytr)
    p_va_logit = logit_oot.predict_proba(Xva)[:, 1]
    metrics_logit_oot = eval_metrics(yva, p_va_logit)

    lgbm_cfg_clean = {k: v for k, v in best_lgbm_cfg.items() if k in ["num_leaves", "learning_rate", "n_estimators", "min_child_samples"]}
    lgbm_oot = build_pipeline(LGBMClassifier(objective="binary", random_state=42, verbosity=-1, **lgbm_cfg_clean),
                               best_lgbm_cfg["sampling_strategy"], pi_true_tr, scale=False)
    lgbm_oot.fit(Xtr, ytr)
    p_va_lgbm = lgbm_oot.predict_proba(Xva)[:, 1]
    metrics_lgbm_oot = eval_metrics(yva, p_va_lgbm)

    # ============== Modelos FINALES (100% train.csv) ==============
    logit_final = build_pipeline(LogisticRegression(max_iter=3000, C=best_logit_cfg["C"]),
                                  best_logit_cfg["sampling_strategy"], pi_true_full, scale=True)
    logit_final.fit(Xfull, yfull)

    lgbm_final = build_pipeline(LGBMClassifier(objective="binary", random_state=42, verbosity=-1, **lgbm_cfg_clean),
                                 best_lgbm_cfg["sampling_strategy"], pi_true_full, scale=False)
    lgbm_final.fit(Xfull, yfull)

    p_test_logit = logit_final.predict_proba(Xtest)[:, 1]
    p_test_lgbm = lgbm_final.predict_proba(Xtest)[:, 1]

    feat_names_logit = feature_names(logit_final.named_steps["pre"])
    coefs = logit_final.named_steps["clf"].base_estimator_.coef_[0]
    logit_coefs = sorted([{"feature": f, "coef": float(c)} for f, c in zip(feat_names_logit, coefs)], key=lambda x: -abs(x["coef"]))

    feat_names_lgbm = feature_names(lgbm_final.named_steps["pre"])
    importances = lgbm_final.named_steps["clf"].base_estimator_.feature_importances_
    lgbm_importance = sorted([{"feature": f, "importance": int(v)} for f, v in zip(feat_names_lgbm, importances)], key=lambda x: -x["importance"])

    # ============== Guardar artefactos ==============
    joblib.dump(logit_oot, f"{MODELS_DIR}/logit_oot_train_window.joblib")
    joblib.dump(lgbm_oot, f"{MODELS_DIR}/lgbm_oot_train_window.joblib")
    joblib.dump(logit_final, f"{MODELS_DIR}/logit_final_full_train.joblib")
    joblib.dump(lgbm_final, f"{MODELS_DIR}/lgbm_final_full_train.joblib")
    with open(f"{MODELS_DIR}/feature_columns.json", "w") as f:
        json.dump({"raw_feature_cols": RAW_FEATURE_COLS, "target": TARGET,
                    "engineered_feature_names_logit": feat_names_logit,
                    "engineered_feature_names_lgbm": feat_names_lgbm}, f, indent=2)

    out = {
        "split_date": str(SPLIT_DATE.date()),
        "n_train_window": len(tr_win), "n_valid_window": len(va_win), "n_test": len(test),
        "n_full_train": len(train),
        "default_rate_train_window": float(ytr.mean()),
        "default_rate_valid_window": float(yva.mean()),
        "imputation_recipe": "group_median_tipo_empleo",
        "excluded_features": ["num_contactos_ult_trimestre (fuga confirmada)", "dia_semana_solicitud (no confiable)",
                               "tasa_interes_anual (posible circularidad con pricing basado en riesgo)"],
        "logit_search": logit_search, "lgbm_search": lgbm_search,
        "best_logit_cfg": best_logit_cfg, "best_lgbm_cfg": best_lgbm_cfg,
        "metrics_logit_oot": metrics_logit_oot, "metrics_lgbm_oot": metrics_lgbm_oot,
        "logit_coefs": logit_coefs, "lgbm_importance": lgbm_importance,
        "score_valid_logit": {"y": yva.tolist(), "score": p_va_logit.tolist()},
        "score_valid_lgbm": {"y": yva.tolist(), "score": p_va_lgbm.tolist()},
        "score_test_logit": p_test_logit.tolist(), "score_test_lgbm": p_test_lgbm.tolist(),
    }
    with open(REPORT_DATA_PATH, "w") as f:
        json.dump(out, f, indent=2, allow_nan=True)
    print("\nOK -> ", REPORT_DATA_PATH)
    print("Logit OOT:", metrics_logit_oot)
    print("LGBM  OOT:", metrics_lgbm_oot)


if __name__ == "__main__":
    main()
