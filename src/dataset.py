"""
Dataset loading and validation module.
Loads the Sri Lanka flood risk dataset and validates schema integrity.
"""

import pandas as pd
import numpy as np

from src.config import (
    DATA_PATH,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
    ALL_FEATURES,
)


def load_dataset(path=None):
    """Load the CSV dataset from the configured path."""
    path = path or DATA_PATH
    df = pd.read_csv(path)
    return df


def validate_dataset(df):
    """
    Validate the dataset against the expected schema.

    Checks:
    - All required columns exist.
    - Target column is binary (0/1).
    - Data types are correct (or castable).

    Returns:
        dict with validation summary.
    """
    required_columns = ALL_FEATURES + [TARGET]
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Verify target is binary
    unique_targets = set(df[TARGET].dropna().unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Target column must be binary (0/1). Found: {unique_targets}"
        )

    # Check data types
    for col in NUMERICAL_FEATURES:
        if not np.issubdtype(df[col].dtype, np.number):
            raise TypeError(f"Column '{col}' should be numeric, got {df[col].dtype}")

    # Missing value report
    missing = df[required_columns].isnull().sum()
    missing_report = missing[missing > 0].to_dict()

    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": missing_report,
        "target_distribution": df[TARGET].value_counts().to_dict(),
        "class_balance_ratio": round(
            df[TARGET].value_counts(normalize=True).to_dict().get(1, 0), 4
        ),
    }
    return summary


def get_features_and_target(df):
    """Split the dataframe into feature matrix X and target vector y."""
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y
