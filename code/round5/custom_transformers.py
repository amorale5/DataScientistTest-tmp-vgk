"""
Transformers custom, reutilizables para entrenar y para scorear datos nuevos
(se guardan con joblib como parte del pipeline, por lo que este modulo debe
poder importarse tanto al entrenar como al cargar el modelo despues).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Imputa la primera columna usando la mediana calculada por grupo (segunda
    columna) sobre los datos de entrenamiento. Si el grupo no se vio en fit
    (o su mediana es NaN), usa la mediana global de entrenamiento.

    Input esperado: array/DataFrame de 2 columnas [valor, grupo].
    Output: array de 1 columna con el valor imputado.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(np.asarray(X, dtype=object), columns=["val", "grp"])
        val = pd.to_numeric(X["val"])
        self.global_median_ = float(val.median())
        self.group_medians_ = val.groupby(X["grp"]).median().to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(np.asarray(X, dtype=object), columns=["val", "grp"]).copy()
        val = pd.to_numeric(X["val"])
        grp = X["grp"]
        filled = val.copy()
        mask = filled.isnull()
        fallback = grp[mask].map(self.group_medians_).fillna(self.global_median_)
        filled.loc[mask] = fallback.values
        return filled.to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["ingreso_declarado_imputado"])


class ConstantFillImputer(BaseEstimator, TransformerMixin):
    """Imputa nulos con un valor constante (por defecto 0). Se usa para
    antiguedad_laboral_meses: sus nulos son 100% estructurales (informal/
    jubilado no tienen un empleador formal actual), asi que 0 representa
    la semantica real del campo, no un supuesto estadistico."""

    def __init__(self, fill_value=0.0):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        self.fill_value_ = self.fill_value  # marca de "fitted" para sklearn
        return self

    def transform(self, X):
        X = pd.DataFrame(np.asarray(X, dtype=float))
        return X.fillna(self.fill_value_).to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.array(["antiguedad_laboral_meses_imputada"])
