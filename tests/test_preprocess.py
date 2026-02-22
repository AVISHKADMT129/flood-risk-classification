"""
Unit tests for the preprocessing pipeline.
"""

import pytest
import pandas as pd
import numpy as np

from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, ALL_FEATURES, TARGET
from src.dataset import load_dataset, validate_dataset, get_features_and_target
from src.preprocess import build_preprocessor


@pytest.fixture
def sample_df():
    """Create a small sample dataframe for testing."""
    return pd.DataFrame({
        "district": ["Colombo", "Kandy", "Galle"],
        "division": ["Colombo-DS1", "Kandy-DS2", "Galle-DS3"],
        "climate_zone": ["Wet", "Wet", "Wet"],
        "drainage_quality": ["Good", "Moderate", "Poor"],
        "year": [2020, 2021, 2022],
        "month": [5, 11, 3],
        "rainfall_mm": [300.0, 250.0, 180.0],
        "river_level_m": [3.5, 2.8, 2.1],
        "soil_saturation_percent": [60.0, 45.0, 30.0],
        "district_flood_prone": [1, 1, 0],
        "flood_occurred": [1, 0, 0],
    })


def test_load_dataset():
    """Test that the dataset loads correctly."""
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_validate_dataset():
    """Test dataset validation passes on real data."""
    df = load_dataset()
    summary = validate_dataset(df)
    assert "total_rows" in summary
    assert summary["total_rows"] == len(df)
    assert "target_distribution" in summary


def test_validate_missing_column():
    """Test that validation raises error for missing columns."""
    df = pd.DataFrame({"district": ["Colombo"], "year": [2020]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataset(df)


def test_get_features_and_target(sample_df):
    """Test feature/target splitting."""
    X, y = get_features_and_target(sample_df)
    assert list(X.columns) == ALL_FEATURES
    assert y.name == TARGET
    assert len(X) == len(y) == 3


def test_preprocessor_fits(sample_df):
    """Test that the preprocessor fits and transforms data."""
    preprocessor = build_preprocessor()
    X, _ = get_features_and_target(sample_df)
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == 3
    assert transformed.shape[1] > len(ALL_FEATURES)  # OneHot expands columns


def test_preprocessor_handles_missing():
    """Test that the preprocessor handles missing values."""
    df = pd.DataFrame({
        "district": ["Colombo", np.nan, "Galle"],
        "division": ["Colombo-DS1", "Kandy-DS2", np.nan],
        "climate_zone": ["Wet", "Wet", "Wet"],
        "drainage_quality": ["Good", np.nan, "Poor"],
        "year": [2020, 2021, np.nan],
        "month": [5, np.nan, 3],
        "rainfall_mm": [300.0, 250.0, 180.0],
        "river_level_m": [3.5, 2.8, 2.1],
        "soil_saturation_percent": [60.0, 45.0, 30.0],
        "district_flood_prone": [1, 1, 0],
        "flood_occurred": [1, 0, 0],
    })
    preprocessor = build_preprocessor()
    X, _ = get_features_and_target(df)
    transformed = preprocessor.fit_transform(X)
    assert not np.isnan(transformed).any()
