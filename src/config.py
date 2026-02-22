"""
Configuration module for the Flood Risk Classification project.
Centralizes all paths, feature definitions, and hyperparameters.
"""

import os
from pathlib import Path

# ──────────────────────────── Paths ────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "sri_lanka_flood_risk_10000.csv"
MODEL_PATH = BASE_DIR / "models" / "flood_model.pkl"
OUTPUTS_DIR = BASE_DIR / "outputs"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = OUTPUTS_DIR / "roc_curve.png"
PR_CURVE_PATH = OUTPUTS_DIR / "pr_curve.png"
CALIBRATION_CURVE_PATH = OUTPUTS_DIR / "calibration_curve.png"
FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_PLOT_PATH = OUTPUTS_DIR / "feature_importance.png"
MODEL_COMPARISON_PATH = OUTPUTS_DIR / "model_comparison.csv"
PDP_DIR = OUTPUTS_DIR
LEARNING_CURVE_PATH = OUTPUTS_DIR / "learning_curve.png"
DISTRICT_ANALYSIS_PATH = OUTPUTS_DIR / "district_analysis.csv"
DISTRICT_ANALYSIS_PLOT_PATH = OUTPUTS_DIR / "district_analysis.png"

# ──────────────────────────── Features ─────────────────────────
TARGET = "flood_occurred"

CATEGORICAL_FEATURES = ["district", "division", "climate_zone", "drainage_quality"]
NUMERICAL_FEATURES = [
    "year",
    "month",
    "rainfall_mm",
    "river_level_m",
    "soil_saturation_percent",
    "district_flood_prone",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# ──────────────────────────── Training ─────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
SCORING_METRIC = "f1"

# ──────────────────────────── Risk Thresholds ──────────────────
RISK_LOW_THRESHOLD = 0.35
RISK_HIGH_THRESHOLD = 0.65
