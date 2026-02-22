"""
Preprocessing module.
Builds a scikit-learn ColumnTransformer for categorical and numerical features.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def build_preprocessor():
    """
    Build and return a ColumnTransformer that:
    - Categorical: impute with most frequent -> OneHotEncode
    - Numerical: impute with median -> StandardScale
    """
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor
