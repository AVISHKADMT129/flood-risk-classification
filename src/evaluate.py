"""
Evaluation module.
Computes metrics, generates plots, and extracts feature importance.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
)
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import learning_curve, StratifiedKFold

from src.config import (
    CONFUSION_MATRIX_PATH,
    ROC_CURVE_PATH,
    PR_CURVE_PATH,
    CALIBRATION_CURVE_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_IMPORTANCE_PLOT_PATH,
    LEARNING_CURVE_PATH,
    DISTRICT_ANALYSIS_PATH,
    DISTRICT_ANALYSIS_PLOT_PATH,
    OUTPUTS_DIR,
    RANDOM_STATE,
)


def evaluate_model(pipeline, X_test, y_test, model_name="Best Model"):
    """
    Evaluate the trained pipeline on the test set.

    Returns:
        dict of evaluation metrics.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
    }

    # Confusion matrix breakdown
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics["confusion_matrix"] = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    print(f"\n--- Evaluation Results ({model_name}) ---")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"\n  Confusion Matrix: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Flood', 'Flood'])}")

    return metrics


def plot_confusion_matrix(pipeline, X_test, y_test):
    """Generate and save the confusion matrix plot."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test,
        display_labels=["No Flood", "Flood"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")


def plot_roc_curve(pipeline, X_test, y_test):
    """Generate and save the ROC curve plot."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROC_CURVE_PATH, dpi=150)
    plt.close(fig)
    print(f"ROC curve saved to {ROC_CURVE_PATH}")


def plot_pr_curve(pipeline, X_test, y_test):
    """Generate and save the Precision-Recall curve plot."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(PR_CURVE_PATH, dpi=150)
    plt.close(fig)
    print(f"PR curve saved to {PR_CURVE_PATH}")


def plot_calibration_curve(pipeline, X_test, y_test):
    """Generate and save the calibration curve plot."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    CalibrationDisplay.from_estimator(
        pipeline, X_test, y_test, n_bins=10, ax=ax
    )
    ax.set_title("Calibration Curve")
    fig.tight_layout()
    fig.savefig(CALIBRATION_CURVE_PATH, dpi=150)
    plt.close(fig)
    print(f"Calibration curve saved to {CALIBRATION_CURVE_PATH}")


def save_feature_importance(pipeline, preprocessor):
    """
    Extract and save feature importance from the trained model.
    Supports tree-based models (feature_importances_) and linear models (coef_).
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    model = pipeline.named_steps["model"]

    # Get feature names from the fitted preprocessor
    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    try:
        feature_names = fitted_preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [f"feature_{i}" for i in range(100)]

    # Clean up feature names
    feature_names = [
        name.replace("cat__", "").replace("num__", "")
        for name in feature_names
    ]

    # Extract importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        print("Model does not support feature importance extraction.")
        return

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    print(f"Feature importance saved to {FEATURE_IMPORTANCE_PATH}")

    # Print top 10
    print("\nTop 10 Features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:40s} {row['importance']:.4f}")


def plot_feature_importance(pipeline, preprocessor):
    """Generate and save a bar chart of top 20 feature importances."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    model = pipeline.named_steps["model"]

    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    try:
        feature_names = fitted_preprocessor.get_feature_names_out()
    except AttributeError:
        return

    feature_names = [
        name.replace("cat__", "").replace("num__", "")
        for name in feature_names
    ]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    importance_df = importance_df.sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        importance_df["feature"][::-1],
        importance_df["importance"][::-1],
        color="#0284c7",
    )
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Feature Importances")
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Feature importance plot saved to {FEATURE_IMPORTANCE_PLOT_PATH}")


def plot_learning_curve(pipeline, X_train, y_train):
    """
    Generate and save a learning curve showing training vs validation performance
    at different training set sizes. Helps diagnose bias/variance.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.15, color="#2563eb")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.15, color="#dc2626")
    ax.plot(train_sizes, train_mean, "o-", color="#2563eb", label="Training F1", linewidth=2)
    ax.plot(train_sizes, val_mean, "o-", color="#dc2626", label="Validation F1", linewidth=2)
    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Learning Curve", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(LEARNING_CURVE_PATH, dpi=150)
    plt.close(fig)
    print(f"Learning curve saved to {LEARNING_CURVE_PATH}")


def analyze_per_district(pipeline, X_test, y_test):
    """
    Analyze model performance per district to identify potential bias.
    Computes accuracy, F1, and sample count for each district in the test set.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    y_pred = pipeline.predict(X_test)
    results_df = X_test.copy()
    results_df = results_df.assign(y_true=y_test.values, y_pred=y_pred)

    district_stats = []
    for district, group in results_df.groupby("district"):
        n = len(group)
        if n < 5:
            continue
        acc = accuracy_score(group["y_true"], group["y_pred"])
        flood_rate = group["y_true"].mean()
        f1 = f1_score(group["y_true"], group["y_pred"], zero_division=0)
        district_stats.append({
            "district": district,
            "samples": n,
            "flood_rate": round(flood_rate, 4),
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
        })

    stats_df = pd.DataFrame(district_stats).sort_values("f1_score", ascending=True)
    stats_df.to_csv(DISTRICT_ANALYSIS_PATH, index=False)
    print(f"Per-district analysis saved to {DISTRICT_ANALYSIS_PATH}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(stats_df) * 0.35)))

    # F1 Score by district
    colors = ["#dc2626" if f < 0.5 else "#d97706" if f < 0.65 else "#059669"
              for f in stats_df["f1_score"]]
    axes[0].barh(stats_df["district"], stats_df["f1_score"], color=colors)
    axes[0].set_xlabel("F1 Score", fontsize=11)
    axes[0].set_title("F1 Score by District", fontsize=13)
    axes[0].axvline(x=stats_df["f1_score"].mean(), color="#64748b",
                     linestyle="--", label=f"Mean: {stats_df['f1_score'].mean():.3f}")
    axes[0].legend(fontsize=10)
    axes[0].set_xlim(0, 1)

    # Flood rate by district
    axes[1].barh(stats_df["district"], stats_df["flood_rate"], color="#3b82f6")
    axes[1].set_xlabel("Flood Rate", fontsize=11)
    axes[1].set_title("Actual Flood Rate by District", fontsize=13)
    axes[1].axvline(x=stats_df["flood_rate"].mean(), color="#64748b",
                     linestyle="--", label=f"Mean: {stats_df['flood_rate'].mean():.3f}")
    axes[1].legend(fontsize=10)
    axes[1].set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(DISTRICT_ANALYSIS_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Per-district plot saved to {DISTRICT_ANALYSIS_PLOT_PATH}")

    return stats_df
