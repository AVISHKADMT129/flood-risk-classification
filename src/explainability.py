"""
Global explainability module.
Computes feature importance and Partial Dependence Plot data for the API.
"""

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH, MODEL_PATH, ALL_FEATURES, TARGET,
    TEST_SIZE, RANDOM_STATE, FEATURE_IMPORTANCE_PATH,
)

# Display names for transformed features
FEATURE_DISPLAY_NAMES = {
    "river_level_m": "River Level (m)",
    "soil_saturation_percent": "Soil Saturation (%)",
    "rainfall_mm": "Rainfall (mm)",
    "climate_zone_Wet": "Climate Zone (Wet)",
    "climate_zone_Dry": "Climate Zone (Dry)",
    "climate_zone_Intermediate": "Climate Zone (Intermediate)",
    "district_flood_prone": "Flood-Prone District",
    "year": "Year",
    "month": "Month",
    "drainage_quality_Good": "Good Drainage",
    "drainage_quality_Moderate": "Moderate Drainage",
    "drainage_quality_Poor": "Poor Drainage",
}

PDP_FEATURES = ["rainfall_mm", "river_level_m", "soil_saturation_percent"]
PDP_LABELS = {
    "rainfall_mm": {"name": "Rainfall", "unit": "mm"},
    "river_level_m": {"name": "River Level", "unit": "m"},
    "soil_saturation_percent": {"name": "Soil Saturation", "unit": "%"},
}

# Cache
_cached_explainability = None


def _load_test_data():
    """Load and split dataset, return X_test."""
    df = pd.read_csv(DATA_PATH)
    X = df[ALL_FEATURES]
    y = df[TARGET]
    _, X_test, _, _ = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return X_test


def _compute_feature_importance():
    """Read feature importance from the CSV output."""
    try:
        df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        # Take top 12 features
        top = df.head(12)
        results = []
        for _, row in top.iterrows():
            fname = row["feature"]
            display = FEATURE_DISPLAY_NAMES.get(
                fname, fname.replace("_", " ").title()
            )
            results.append({
                "feature": fname,
                "display_name": display,
                "importance": round(float(row["importance"]), 4),
            })
        return results
    except Exception:
        return []


def _compute_pdp_data(model, X_test):
    """Compute PDP curves for key numerical features."""
    pdp_data = {}
    for feature in PDP_FEATURES:
        if feature not in X_test.columns:
            continue
        try:
            result = partial_dependence(
                model, X_test, features=[feature], grid_resolution=40
            )
            grid_values = result["grid_values"][0]
            avg_predictions = result["average"][0]

            pdp_data[feature] = {
                "label": PDP_LABELS[feature]["name"],
                "unit": PDP_LABELS[feature]["unit"],
                "values": [round(float(v), 2) for v in grid_values],
                "predictions": [round(float(p), 4) for p in avg_predictions],
            }
        except Exception:
            continue
    return pdp_data


def get_global_explainability(model):
    """
    Compute and cache global explainability data:
    - Feature importance (top 12)
    - PDP curves for key features
    """
    global _cached_explainability
    if _cached_explainability is not None:
        return _cached_explainability

    X_test = _load_test_data()
    feature_importance = _compute_feature_importance()
    pdp_data = _compute_pdp_data(model, X_test)

    _cached_explainability = {
        "feature_importance": feature_importance,
        "pdp_data": pdp_data,
    }
    return _cached_explainability
