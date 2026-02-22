"""
Model loader for the FastAPI backend.
Loads the trained model pipeline once at startup.
"""

import joblib
from src.config import MODEL_PATH


_model = None


def get_model():
    """Load and cache the trained model pipeline."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model
