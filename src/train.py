"""
Training module.
Trains multiple ML models with GridSearchCV, selects the best, and saves the pipeline.
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

from src.config import (
    TEST_SIZE,
    RANDOM_STATE,
    CV_FOLDS,
    SCORING_METRIC,
    MODEL_PATH,
    OUTPUTS_DIR,
    MODEL_COMPARISON_PATH,
)
from src.dataset import load_dataset, validate_dataset, get_features_and_target
from src.preprocess import build_preprocessor
from src.evaluate import (
    evaluate_model, plot_confusion_matrix, plot_roc_curve,
    plot_pr_curve, plot_calibration_curve,
    save_feature_importance, plot_feature_importance,
    plot_learning_curve, analyze_per_district,
)

warnings.filterwarnings("ignore")


# ──────────────────────────── Model Definitions ────────────────
def get_model_configs():
    """Return model definitions with hyperparameter grids for GridSearchCV."""
    return {
        "LogisticRegression": {
            "model": LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "model__C": [0.1, 1, 10],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs"],
                "model__class_weight": [None, "balanced"],
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                random_state=RANDOM_STATE,
            ),
            "params": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 8, 16],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
                "model__class_weight": [None, "balanced"],
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),
            "params": {
                "model__n_estimators": [150, 300],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        },
    }


# ──────────────────────────── Training Logic ───────────────────
def train_and_select_best():
    """
    Full training pipeline:
    1. Load and validate data
    2. Split train/test
    3. Train all models via GridSearchCV with StratifiedKFold
    4. Select the best model
    5. Evaluate and save outputs
    """
    # 1. Load and validate
    print("=" * 60)
    print("FLOOD RISK CLASSIFICATION - MODEL TRAINING")
    print("=" * 60)

    df = load_dataset()
    summary = validate_dataset(df)
    print(f"\nDataset: {summary['total_rows']} rows, {summary['total_columns']} cols")
    print(f"Target distribution: {summary['target_distribution']}")
    print(f"Class balance (flood=1): {summary['class_balance_ratio']:.2%}")
    if summary["missing_values"]:
        print(f"Missing values: {summary['missing_values']}")
    else:
        print("No missing values detected.")

    # 2. Split
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    # 3. Train models with StratifiedKFold CV
    preprocessor = build_preprocessor()
    model_configs = get_model_configs()
    results = {}

    cv_strategy = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    for name, cfg in model_configs.items():
        print(f"\n{'-' * 40}")
        print(f"Training: {name}")
        print(f"{'-' * 40}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", cfg["model"]),
            ]
        )

        grid_search = GridSearchCV(
            pipeline,
            param_grid=cfg["params"],
            cv=cv_strategy,
            scoring=SCORING_METRIC,
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)

        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        print(f"  Best CV F1: {best_score:.4f}")
        print(f"  Best params: {best_params}")

        # Evaluate on test set for comparison table
        best_est = grid_search.best_estimator_
        y_pred = best_est.predict(X_test)
        y_proba = best_est.predict_proba(X_test)[:, 1]

        results[name] = {
            "grid_search": grid_search,
            "best_score": best_score,
            "best_params": best_params,
            "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
            "test_precision": round(precision_score(y_test, y_pred), 4),
            "test_recall": round(recall_score(y_test, y_pred), 4),
            "test_f1": round(f1_score(y_test, y_pred), 4),
            "test_roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "test_pr_auc": round(average_precision_score(y_test, y_proba), 4),
        }

    # 4. Select best model
    best_name = max(results, key=lambda k: results[k]["best_score"])
    best_pipeline = results[best_name]["grid_search"].best_estimator_
    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_name} (CV F1={results[best_name]['best_score']:.4f})")
    print(f"{'=' * 60}")

    # 5. Evaluate
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_model(best_pipeline, X_test, y_test, best_name)
    plot_confusion_matrix(best_pipeline, X_test, y_test)
    plot_roc_curve(best_pipeline, X_test, y_test)
    plot_pr_curve(best_pipeline, X_test, y_test)
    plot_calibration_curve(best_pipeline, X_test, y_test)
    save_feature_importance(best_pipeline, preprocessor)
    plot_feature_importance(best_pipeline, preprocessor)

    # Save model_comparison.csv
    comparison_rows = []
    for name, res in results.items():
        row = {
            "model": name,
            "best_params": str({
                k.replace("model__", ""): v for k, v in res["best_params"].items()
            }),
            "accuracy": res["test_accuracy"],
            "precision": res["test_precision"],
            "recall": res["test_recall"],
            "f1": res["test_f1"],
            "roc_auc": res["test_roc_auc"],
            "pr_auc": res["test_pr_auc"],
        }
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    print(f"\nModel comparison saved to {MODEL_COMPARISON_PATH}")

    # Save comparison summary in metrics too
    comparison = {}
    for name, res in results.items():
        comparison[name] = {
            "cv_f1_score": round(res["best_score"], 4),
            "best_params": {
                k.replace("model__", ""): v for k, v in res["best_params"].items()
            },
            "test_accuracy": res["test_accuracy"],
            "test_precision": res["test_precision"],
            "test_recall": res["test_recall"],
            "test_f1": res["test_f1"],
            "test_roc_auc": res["test_roc_auc"],
            "test_pr_auc": res["test_pr_auc"],
        }
    metrics["model_comparison"] = comparison

    # Save metrics
    from src.config import METRICS_PATH

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")

    # Save model
    import joblib

    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Run explainability
    from src.explain import generate_pdp_plots
    generate_pdp_plots(best_pipeline, X_test)

    # Learning curve
    plot_learning_curve(best_pipeline, X_train, y_train)

    # Per-district bias analysis
    analyze_per_district(best_pipeline, X_test, y_test)

    return best_pipeline, metrics


# ──────────────────────────── Entry Point ──────────────────────
if __name__ == "__main__":
    train_and_select_best()
