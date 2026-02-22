"""
Tests for the training pipeline.
Validates the full training flow including model configs, data splitting, and model output.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import (
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, ALL_FEATURES, TARGET,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
)
from src.dataset import load_dataset, get_features_and_target
from src.preprocess import build_preprocessor
from src.train import get_model_configs


class TestModelConfigs:
    """Test that model configs match the specification."""

    def test_three_models_defined(self):
        configs = get_model_configs()
        assert len(configs) >= 3, "At least 3 models must be defined"
        assert "LogisticRegression" in configs
        assert "RandomForest" in configs
        assert "GradientBoosting" in configs

    def test_logistic_regression_grid(self):
        configs = get_model_configs()
        params = configs["LogisticRegression"]["params"]
        assert params["model__C"] == [0.1, 1, 10]
        assert params["model__penalty"] == ["l2"]
        assert params["model__solver"] == ["lbfgs"]
        assert params["model__class_weight"] == [None, "balanced"]

    def test_random_forest_grid(self):
        configs = get_model_configs()
        params = configs["RandomForest"]["params"]
        assert params["model__n_estimators"] == [200, 400]
        assert params["model__max_depth"] == [None, 8, 16]
        assert params["model__min_samples_split"] == [2, 5]
        assert params["model__min_samples_leaf"] == [1, 2]
        assert params["model__class_weight"] == [None, "balanced"]

    def test_gradient_boosting_grid(self):
        configs = get_model_configs()
        params = configs["GradientBoosting"]["params"]
        assert params["model__n_estimators"] == [150, 300]
        assert params["model__learning_rate"] == [0.05, 0.1]
        assert params["model__max_depth"] == [2, 3]


class TestDataSplit:
    """Test that data splitting follows the specification."""

    def test_stratified_split_ratio(self):
        df = load_dataset()
        X, y = get_features_and_target(df)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )
        expected_test = int(len(df) * TEST_SIZE)
        assert abs(len(X_test) - expected_test) <= 1

    def test_stratified_kfold_config(self):
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        assert cv.n_splits == 5
        assert cv.shuffle is True
        assert cv.random_state == RANDOM_STATE


class TestPipelineIntegration:
    """Test that preprocessing + model pipeline works end-to-end."""

    def test_pipeline_builds(self):
        preprocessor = build_preprocessor()
        configs = get_model_configs()
        for name, cfg in configs.items():
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", cfg["model"]),
            ])
            assert pipeline is not None

    def test_pipeline_fit_predict(self):
        df = load_dataset()
        X, y = get_features_and_target(df)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )
        preprocessor = build_preprocessor()
        configs = get_model_configs()
        # Quick test with LogReg (fastest)
        cfg = configs["LogisticRegression"]
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", cfg["model"]),
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        assert len(preds) == len(y_test)
        assert set(preds).issubset({0, 1})
