# src/features.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def _clip(X):
    return X.clip(-3, 3)

def build_numeric_preprocess():
    """
    Prétraitement minimal (baseline) :
    - imputation médiane
    - standardisation
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])