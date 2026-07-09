"""
Ronda 5 de modelamiento — score de probabilidad de default a 12 meses.

Diferencia vs. Ronda 4: cambia la PARTICION TEMPORAL usada para elegir
hiperparametros y medir metricas honestas. En vez de train<2025-01-01 /
validacion=resto de train.csv (14 meses de train, ~1,5 meses de validacion,
justo antes del periodo de test.csv), aqui se usa:

  - Entrenamiento: 2024-01-01 a 2024-06-30 (6 meses, incluye junio completo)
  - Validacion:    2024-07-01 a 2024-12-31 (los 6 meses siguientes)

Es un backtest de ventana rodante (6 meses entrena / 6 meses valida) en vez
del esquema "todo el pasado disponible / ultimo tramo antes de produccion"
de las rondas anteriores. Sirve para responder una pregunta distinta: ¿el
modelo generaliza igual de bien con menos historia y una brecha temporal
mas corta entre entrenar y validar, en un tramo intermedio del año en vez
del tramo mas cercano a cuando se usaria en produccion? Enero-febrero 2025
queda fuera de esta ronda (ni entrena ni valida en esos meses).

Se mantienen las mismas 3 correcciones de inconsistencia de la Ronda 4
(edad, ingreso_declarado, tasa_interes_anual vs. TMC) y el mismo esquema de
modelo OOT (metricas honestas) + modelo final reentrenado con el 100% de
train.csv (para scorear test.csv), igual que en Rondas 1, 3 y 4.
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
MODELS_DIR = f"{PROJECT}/models/round5"
REPORT_DATA_PATH = f"{PROJECT}/reports/round5_report_data.json"

TARGET = "default_12m"
TRAIN_START = pd.Timestamp("2024-01-01")
TRAIN_END = pd.Timestamp("2024-07-01")     # exclusivo: entrena ene-jun 2024
VALID_END = pd.Timestamp("2025-01-01")     # exclusivo: valida jul-dic 2024
SPLIT_DATE = TRAIN_END  # se mantiene el nombre por compatibilidad con el resto del script

CAT_VARS = ["tipo_empleo", "region", "canal"]
NUM_LOG_PLAIN = ["deuda_sistema", "monto_solicitado"]
NUM_PLAIN = ["edad", "antiguedad_cliente_meses", "score_buro", "num_creditos_vigentes",
             "peor_morosidad_12m", "num_consultas_buro_3m", "uso_linea_credito_pct",
             "plazo_meses", "tasa_interes_anual",
             "flag_edad_inconsistente", "flag_ingreso_inconsistente", "flag_tasa_corregida"]

RAW_FEATURE_COLS = (["ingreso_declarado", "tipo_empleo", "antiguedad_laboral_meses"]
                     + NUM_LOG_PLAIN + NUM_PLAIN + CAT_VARS)
RAW_FEATURE_COLS = list(dict.fromkeys(RAW_FEATURE_COLS))

# ================= CMF: Tasa Maxima Convencional (TMC) =================
# Mismos 12 certificados usados en exploracion_variables (tasas.cmfchile.cl).
_TMC_DATES = pd.to_datetime([
    "2023-12-15", "2024-01-15", "2024-02-15", "2024-04-15", "2024-06-15",
    "2024-07-15", "2024-08-14", "2024-09-15", "2024-10-15", "2024-11-15", "2024-12-14", "2025-01-15",
])
_TMC_50 = np.array([41.68, 41.88, 41.06, 39.40, 39.08, 38.58, 38.58, 38.16, 38.10, 38.20, 38.56, 39.08])
_TMC_200 = np.array([34.68, 34.88, 34.06, 32.40, 32.08, 31.58, 31.58, 31.16, 31.10, 31.20, 31.56, 32.08])
_TMC_5000 = np.array([31.02, 31.32, 30.09, 27.60, 27.12, 26.37, 26.37, 25.74, 25.65, 25.80, 26.34, 27.12])
_ORD_TMC = _TMC_DATES.map(pd.Timestamp.toordinal).to_numpy()
_LAST_TMC_DATE = _TMC_DATES.max()

_UF_DATES = pd.to_datetime([
    "2024-01-01","2024-01-15","2024-02-01","2024-02-15","2024-03-01","2024-03-15",
    "2024-04-01","2024-04-15","2024-05-01","2024-05-15","2024-06-01","2024-06-15",
    "2024-07-01","2024-07-15","2024-08-01","2024-08-15","2024-09-01","2024-09-15",
    "2024-10-01","2024-10-15","2024-11-01","2024-11-15","2024-12-01","2024-12-15",
    "2025-01-01","2025-01-15","2025-02-01","2025-02-15",
])
_UF_VALS = np.array([
    36797.64,36828.19, 36727.10,36732.60, 36865.37,36979.17,
    37100.68,37187.68, 37266.94,37342.66, 37444.94,37515.63,
    37575.61,37598.36, 37577.74,37618.79, 37762.97,37853.68,
    37914.20,37951.84, 37972.65,38058.10, 38260.61,38377.10,
    38419.17,38424.09, 38381.93,38452.14,
])
_ORD_UF = _UF_DATES.map(pd.Timestamp.toordinal).to_numpy()
_LAST_UF_DATE = _UF_DATES.max()


def _tmc_aplicable(fecha_solicitud, monto_solicitado):
    # despues del ultimo certificado/valor UF conocido, se sostiene plano
    # (aproximacion documentada; afecta a muy pocas filas de test.csv)
    fecha_clip_tmc = fecha_solicitud.clip(upper=_LAST_TMC_DATE)
    fecha_clip_uf = fecha_solicitud.clip(upper=_LAST_UF_DATE)
    ords_tmc = fecha_clip_tmc.map(pd.Timestamp.toordinal).to_numpy()
    ords_uf = fecha_clip_uf.map(pd.Timestamp.toordinal).to_numpy()
    uf_val = np.interp(ords_uf, _ORD_UF, _UF_VALS)
    tmc50 = np.interp(ords_tmc, _ORD_TMC, _TMC_50)
    tmc200 = np.interp(ords_tmc, _ORD_TMC, _TMC_200)
    tmc5000 = np.interp(ords_tmc, _ORD_TMC, _TMC_5000)
    monto_uf = monto_solicitado / uf_val
    return np.select([monto_uf <= 50, monto_uf <= 200], [tmc50, tmc200], default=tmc5000)


def clean_inconsistencies(df, edad_fill_value):
    """Aplica las 3 correcciones de inconsistencia. edad_fill_value debe
    calcularse SOLO sobre el set de entrenamiento correspondiente (ventana
    2024 para el modelo OOT, 100% de train.csv para el modelo final) para
    no filtrar informacion de validacion/test."""
    df = df.copy()

    edad_mask = (df["edad"] < 18) | (df["edad"] > 100)
    df["flag_edad_inconsistente"] = edad_mask.astype(int)
    df.loc[edad_mask, "edad"] = edad_fill_value

    ingreso_mask = df["ingreso_declarado"].notnull() & (df["ingreso_declarado"] <= 200000)
    df["flag_ingreso_inconsistente"] = ingreso_mask.astype(int)
    df.loc[ingreso_mask, "ingreso_declarado"] = np.nan

    tmc_ap = _tmc_aplicable(df["fecha_solicitud"], df["monto_solicitado"])
    tasa_mask = df["tasa_interes_anual"] > tmc_ap
    df["flag_tasa_corregida"] = tasa_mask.astype(int)
    df.loc[tasa_mask, "tasa_interes_anual"] = tmc_ap[tasa_mask.values]

    return df


def edad_median_valida(df):
    valid = df.loc[(df["edad"] >= 18) & (df["edad"] <= 100), "edad"]
    return float(valid.median())


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
    names += ["flag_ingreso_declarado_nulo_original", "flag_antiguedad_laboral_nulo"]
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
    tr_win_raw = train[(train["fecha_solicitud"] >= TRAIN_START) & (train["fecha_solicitud"] < TRAIN_END)]
    va_win_raw = train[(train["fecha_solicitud"] >= TRAIN_END) & (train["fecha_solicitud"] < VALID_END)]
    print(f"Ventana train: {tr_win_raw['fecha_solicitud'].min().date()} a {tr_win_raw['fecha_solicitud'].max().date()}  ({len(tr_win_raw)} filas)")
    print(f"Ventana valid: {va_win_raw['fecha_solicitud'].min().date()} a {va_win_raw['fecha_solicitud'].max().date()}  ({len(va_win_raw)} filas)")

    # limpieza fit-on-train: la mediana de edad se calcula por separado para
    # el contexto OOT (solo 2024) y el contexto final (100% train.csv)
    edad_fill_trwin = edad_median_valida(tr_win_raw)
    edad_fill_full = edad_median_valida(train)

    tr_win = clean_inconsistencies(tr_win_raw, edad_fill_trwin)
    va_win = clean_inconsistencies(va_win_raw, edad_fill_trwin)
    full = clean_inconsistencies(train, edad_fill_full)
    test_c = clean_inconsistencies(test, edad_fill_full)

    print(f"train window: {len(tr_win)}  valid window: {len(va_win)}  test.csv: {len(test_c)}")
    print("Inconsistencias corregidas (train window):",
          "edad=", int(tr_win["flag_edad_inconsistente"].sum()),
          "ingreso=", int(tr_win["flag_ingreso_inconsistente"].sum()),
          "tasa=", int(tr_win["flag_tasa_corregida"].sum()))
    print("Inconsistencias corregidas (test.csv):",
          "edad=", int(test_c["flag_edad_inconsistente"].sum()),
          "ingreso=", int(test_c["flag_ingreso_inconsistente"].sum()),
          "tasa=", int(test_c["flag_tasa_corregida"].sum()))

    Xtr, ytr = tr_win[RAW_FEATURE_COLS], tr_win[TARGET].values
    Xva, yva = va_win[RAW_FEATURE_COLS], va_win[TARGET].values
    Xtest = test_c[RAW_FEATURE_COLS]
    Xfull, yfull = full[RAW_FEATURE_COLS], full[TARGET].values

    # ============== LOGISTIC REGRESSION ==============
    logit_search = []
    for C in [0.01, 0.1, 1.0, 10.0]:
        for class_weight in [None, "balanced"]:
            pre = build_preprocessor()
            pipe = Pipeline([("pre", pre), ("scale", StandardScaler(with_mean=False)),
                              ("clf", LogisticRegression(max_iter=3000, C=C, class_weight=class_weight))])
            pipe.fit(Xtr, ytr)
            p_va = pipe.predict_proba(Xva)[:, 1]
            m = eval_metrics(yva, p_va)
            cand = {"C": C, "class_weight": class_weight}
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

    # ============== Modelos OOT (solo 2024) ==============
    logit_oot = Pipeline([("pre", build_preprocessor()), ("scale", StandardScaler(with_mean=False)),
                           ("clf", LogisticRegression(max_iter=3000, C=best_logit_cfg["C"], class_weight=best_logit_cfg["class_weight"]))])
    logit_oot.fit(Xtr, ytr)
    p_va_logit = logit_oot.predict_proba(Xva)[:, 1]
    metrics_logit_oot = eval_metrics(yva, p_va_logit)

    lgbm_cfg_clean = {k: v for k, v in best_lgbm_cfg.items() if k in
                       ["num_leaves", "learning_rate", "n_estimators", "min_child_samples", "scale_pos_weight"]}
    lgbm_oot = Pipeline([("pre", build_preprocessor()),
                          ("clf", LGBMClassifier(objective="binary", random_state=42, verbosity=-1, **lgbm_cfg_clean))])
    lgbm_oot.fit(Xtr, ytr)
    p_va_lgbm = lgbm_oot.predict_proba(Xva)[:, 1]
    metrics_lgbm_oot = eval_metrics(yva, p_va_lgbm)

    # ============== Modelos FINALES (100% train.csv) ==============
    logit_final = Pipeline([("pre", build_preprocessor()), ("scale", StandardScaler(with_mean=False)),
                             ("clf", LogisticRegression(max_iter=3000, C=best_logit_cfg["C"], class_weight=best_logit_cfg["class_weight"]))])
    logit_final.fit(Xfull, yfull)

    lgbm_final = Pipeline([("pre", build_preprocessor()),
                            ("clf", LGBMClassifier(objective="binary", random_state=42, verbosity=-1, **lgbm_cfg_clean))])
    lgbm_final.fit(Xfull, yfull)

    p_test_logit = logit_final.predict_proba(Xtest)[:, 1]
    p_test_lgbm = lgbm_final.predict_proba(Xtest)[:, 1]

    feat_names_logit = feature_names(logit_final.named_steps["pre"])
    coefs = logit_final.named_steps["clf"].coef_[0]
    logit_coefs = sorted([{"feature": f, "coef": float(c)} for f, c in zip(feat_names_logit, coefs)], key=lambda x: -abs(x["coef"]))

    feat_names_lgbm = feature_names(lgbm_final.named_steps["pre"])
    importances = lgbm_final.named_steps["clf"].feature_importances_
    lgbm_importance = sorted([{"feature": f, "importance": int(v)} for f, v in zip(feat_names_lgbm, importances)], key=lambda x: -x["importance"])

    # ============== Guardar artefactos ==============
    joblib.dump(logit_oot, f"{MODELS_DIR}/logit_oot_train_window.joblib")
    joblib.dump(lgbm_oot, f"{MODELS_DIR}/lgbm_oot_train_window.joblib")
    joblib.dump(logit_final, f"{MODELS_DIR}/logit_final_full_train.joblib")
    joblib.dump(lgbm_final, f"{MODELS_DIR}/lgbm_final_full_train.joblib")
    with open(f"{MODELS_DIR}/feature_columns.json", "w") as f:
        json.dump({"raw_feature_cols": RAW_FEATURE_COLS, "target": TARGET,
                    "edad_fill_value_full_train": edad_fill_full,
                    "engineered_feature_names_logit": feat_names_logit,
                    "engineered_feature_names_lgbm": feat_names_lgbm,
                    "nota": "Aplicar clean_inconsistencies() de este modulo a los datos nuevos ANTES de llamar a predict_proba."},
                  f, indent=2)

    out = {
        "train_start": str(TRAIN_START.date()), "train_end": str(TRAIN_END.date()), "valid_end": str(VALID_END.date()),
        "split_date": str(SPLIT_DATE.date()),
        "n_train_window": len(tr_win), "n_valid_window": len(va_win), "n_test": len(test_c),
        "n_full_train": len(train),
        "default_rate_train_window": float(ytr.mean()),
        "default_rate_valid_window": float(yva.mean()),
        "imputation_recipe": "group_median_tipo_empleo",
        "excluded_features": ["num_contactos_ult_trimestre (fuga confirmada)", "dia_semana_solicitud (no confiable)"],
        "inconsistencias": {
            "edad_train": int(tr_win["flag_edad_inconsistente"].sum()),
            "edad_test": int(test_c["flag_edad_inconsistente"].sum()),
            "ingreso_train": int(tr_win["flag_ingreso_inconsistente"].sum()),
            "ingreso_test": int(test_c["flag_ingreso_inconsistente"].sum()),
            "tasa_train": int(tr_win["flag_tasa_corregida"].sum()),
            "tasa_test": int(test_c["flag_tasa_corregida"].sum()),
            "edad_fill_value": edad_fill_full,
        },
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
