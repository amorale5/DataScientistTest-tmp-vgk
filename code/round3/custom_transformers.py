"""
Transformers custom, reutilizables para entrenar y para scorear datos nuevos
(se guardan con joblib como parte del pipeline, por lo que este modulo debe
poder importarse tanto al entrenar como al cargar el modelo despues).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone


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


class PriorCorrectedClassifier(BaseEstimator, ClassifierMixin):
    """Envuelve un clasificador entrenado sobre datos con oversampling (SMOTE)
    y corrige las probabilidades de vuelta a la prevalencia real, usando la
    correccion bayesiana de prior (Dal Pozzolo et al., 2015 / King & Zeng,
    2001): asume que el oversampling no cambia la distribucion de las
    variables DENTRO de cada clase, solo la proporcion entre clases.

    p_real = (p' * pi * (1-pi')) / (p' * pi * (1-pi') + (1-p') * pi' * (1-pi))

    donde pi = prevalencia real (antes de SMOTE), pi' = prevalencia con la
    que efectivamente se entreno el clasificador (despues de SMOTE).

    Es una transformacion monotona de p' -> p_real, por lo que NO cambia
    AUC/KS/ranking, solo corrige la calibracion (Brier / probabilidades).
    Si no hubo oversampling (pi' == pi), la correccion es la identidad.
    """

    def __init__(self, base_estimator, pi_true=None):
        self.base_estimator = base_estimator
        self.pi_true = pi_true

    def fit(self, X, y):
        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_.fit(X, y)
        self.pi_train_ = float(np.mean(y))
        self.classes_ = self.base_estimator_.classes_
        return self

    def predict_proba(self, X):
        p_prime = self.base_estimator_.predict_proba(X)[:, 1]
        pi = self.pi_true if self.pi_true is not None else self.pi_train_
        pt = self.pi_train_
        num = p_prime * pi * (1 - pt)
        den = num + (1 - p_prime) * pt * (1 - pi)
        den = np.where(den <= 0, 1e-12, den)
        p = num / den
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
