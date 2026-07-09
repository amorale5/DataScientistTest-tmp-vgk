"""
Ronda 1 - Bake-off de imputacion para ingreso_declarado y antiguedad_laboral_meses.

Comparamos 3 recetas combinadas de imputacion, midiendo el impacto real en
performance: entrenamos una regresion logistica simple con el set completo de
features (mismo para las 3 recetas, solo cambia como se llenan los nulos) y
evaluamos AUC en una validacion temporal (train: fechas < 2025-01-01,
validacion: fechas >= 2025-01-01), que es la misma partición que se usará
para todo el modelamiento de esta ronda.

No se trata ninguna inconsistencia (edad, ingreso, tasa) en este paso: se
usan los valores crudos, tal como pidió el usuario para la Ronda 1.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer

DATA_DIR = "/Users/onlive/Library/CloudStorage/GoogleDrive-amatias.morales25@gmail.com/Mi unidad/respaldo-matias-agm/Trabajos Personales/postulacion-bice/DataScientistTest/data"
TARGET = "default_12m"
SPLIT_DATE = pd.Timestamp("2025-01-01")

CAT_VARS = ["tipo_empleo", "region", "canal"]
NUM_LOG = ["ingreso_declarado", "deuda_sistema", "monto_solicitado"]
NUM_PLAIN = ["edad", "antiguedad_laboral_meses", "antiguedad_cliente_meses", "score_buro",
             "num_creditos_vigentes", "peor_morosidad_12m", "num_consultas_buro_3m",
             "uso_linea_credito_pct", "plazo_meses", "tasa_interes_anual"]
# num_contactos_ult_trimestre excluido (fuga confirmada en el EDA/multivariado)
# dia_semana_solicitud excluido (no confiable, no asociado al target)

ALL_NUM = NUM_LOG + NUM_PLAIN


def load():
    df = pd.read_csv(f"{DATA_DIR}/train.csv")
    df["fecha_solicitud"] = pd.to_datetime(df["fecha_solicitud"])
    return df


def apply_recipe(df, recipe):
    """Devuelve una copia de df con ingreso_declarado y antiguedad_laboral_meses
    imputados segun la receta, mas las columnas de flag de nulidad."""
    d = df.copy()
    d["ingreso_declarado_isnull"] = d["ingreso_declarado"].isnull().astype(int)
    d["antiguedad_laboral_meses_isnull"] = d["antiguedad_laboral_meses"].isnull().astype(int)

    if recipe == "median_global":
        d["ingreso_declarado"] = d["ingreso_declarado"].fillna(d["ingreso_declarado"].median())
        d["antiguedad_laboral_meses"] = d["antiguedad_laboral_meses"].fillna(d["antiguedad_laboral_meses"].median())

    elif recipe == "group_median_tipo_empleo":
        grp_med_ingreso = d.groupby("tipo_empleo")["ingreso_declarado"].transform("median")
        d["ingreso_declarado"] = d["ingreso_declarado"].fillna(grp_med_ingreso)
        d["ingreso_declarado"] = d["ingreso_declarado"].fillna(d["ingreso_declarado"].median())
        # antiguedad_laboral_meses: TODOS los nulos son informal/jubilado (sin dato
        # real dentro del grupo para promediar) -> se imputa con 0 (sin antiguedad
        # laboral formal que reportar), que es la interpretacion de negocio del nulo.
        d["antiguedad_laboral_meses"] = d["antiguedad_laboral_meses"].fillna(0.0)

    elif recipe == "knn":
        knn_cols = ["ingreso_declarado", "antiguedad_laboral_meses", "edad", "score_buro",
                    "monto_solicitado", "antiguedad_cliente_meses"]
        imputer = KNNImputer(n_neighbors=10)
        knn_out = imputer.fit_transform(d[knn_cols])
        d["ingreso_declarado"] = knn_out[:, knn_cols.index("ingreso_declarado")]
        d["antiguedad_laboral_meses"] = knn_out[:, knn_cols.index("antiguedad_laboral_meses")]
    else:
        raise ValueError(recipe)
    return d


def build_pipeline():
    numeric_log_tf = Pipeline([("log1p", __import__("sklearn.preprocessing", fromlist=["FunctionTransformer"]).FunctionTransformer(np.log1p))])
    pre = ColumnTransformer([
        ("log", numeric_log_tf, NUM_LOG),
        ("plain", "passthrough", NUM_PLAIN + ["ingreso_declarado_isnull", "antiguedad_laboral_meses_isnull"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_VARS),
    ])
    pipe = Pipeline([
        ("pre", pre),
        ("scale", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])
    return pipe


def run():
    df = load()
    results = []
    for recipe in ["median_global", "group_median_tipo_empleo", "knn"]:
        d = apply_recipe(df, recipe)
        train = d[d["fecha_solicitud"] < SPLIT_DATE]
        valid = d[d["fecha_solicitud"] >= SPLIT_DATE]

        feat_cols = ALL_NUM + ["ingreso_declarado_isnull", "antiguedad_laboral_meses_isnull"] + CAT_VARS
        Xtr, ytr = train[feat_cols], train[TARGET]
        Xva, yva = valid[feat_cols], valid[TARGET]

        pipe = build_pipeline()
        pipe.fit(Xtr, ytr)
        p_va = pipe.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(yva, p_va)
        results.append({"recipe": recipe, "auc_valid": auc, "n_train": len(train), "n_valid": len(valid)})
        print(f"{recipe:<28} AUC valid={auc:.5f}  n_train={len(train)} n_valid={len(valid)}")

    best = max(results, key=lambda r: r["auc_valid"])
    print("\nGanador:", best["recipe"], f"AUC={best['auc_valid']:.5f}")
    return results, best


if __name__ == "__main__":
    run()
