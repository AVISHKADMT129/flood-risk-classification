"""
Inference module.
Loads the trained model and makes predictions on new data.
"""

import json
import argparse
import joblib
import numpy as np
import pandas as pd
import shap

from src.config import MODEL_PATH, ALL_FEATURES, RISK_LOW_THRESHOLD, RISK_HIGH_THRESHOLD

# Human-readable display names for features
FEATURE_DISPLAY_NAMES = {
    "river_level_m": "River Level",
    "soil_saturation_percent": "Soil Saturation",
    "rainfall_mm": "Rainfall",
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

# Cached dataset statistics (loaded once)
_cached_stats = None


def _get_stats():
    """Load and cache feature statistics."""
    global _cached_stats
    if _cached_stats is None:
        from src.utils import get_feature_statistics
        _cached_stats = get_feature_statistics()
    return _cached_stats


def load_model(path=None):
    """Load the saved model pipeline from disk."""
    path = path or MODEL_PATH
    return joblib.load(path)


def classify_risk(probability):
    """
    Map flood probability to a risk level.
    <0.35 -> Low, 0.35-0.65 -> Medium, >0.65 -> High
    """
    if probability < RISK_LOW_THRESHOLD:
        return "Low"
    elif probability <= RISK_HIGH_THRESHOLD:
        return "Medium"
    else:
        return "High"


def _get_feature_contributions(model, top_n=5):
    """Extract top N feature importances with display names."""
    fitted_preprocessor = model.named_steps["preprocessor"]
    clf = model.named_steps["model"]

    try:
        feature_names = fitted_preprocessor.get_feature_names_out()
    except AttributeError:
        return []

    feature_names = [
        name.replace("cat__", "").replace("num__", "")
        for name in feature_names
    ]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return []

    sorted_indices = np.argsort(importances)[::-1][:top_n]

    contributions = []
    for rank, idx in enumerate(sorted_indices, 1):
        fname = feature_names[idx]
        display = FEATURE_DISPLAY_NAMES.get(fname, fname.replace("_", " ").title())
        contributions.append({
            "feature": fname,
            "display_name": display,
            "importance": round(float(importances[idx]), 4),
            "rank": rank,
        })

    return contributions


def _get_input_context(input_data):
    """Compare user inputs against dataset statistics."""
    stats = _get_stats()
    context = {}
    for col in ["rainfall_mm", "river_level_m", "soil_saturation_percent"]:
        val = float(input_data.get(col, 0))
        s = stats.get(col, {})
        if not s:
            continue
        if val < s["p25"]:
            level = "Low"
        elif val > s["p75"]:
            level = "High"
        else:
            level = "Average"
        context[col] = {
            "value": round(val, 2),
            "mean": s["mean"],
            "p25": s["p25"],
            "p75": s["p75"],
            "level": level,
        }
    return context


def _generate_explanation(risk_level, probability, input_context, contributions):
    """Generate a human-readable explanation of the prediction."""
    pct = round(probability * 100, 1)

    # Build drivers text from top 2 contributions
    top_features = [c["display_name"].lower() for c in contributions[:2]]
    drivers = " and ".join(top_features) if top_features else "environmental factors"

    # Build context details
    details = []
    labels = {
        "rainfall_mm": ("rainfall", "mm"),
        "river_level_m": ("river level", "m"),
        "soil_saturation_percent": ("soil saturation", "%"),
    }
    for col, (name, unit) in labels.items():
        ctx = input_context.get(col)
        if ctx and ctx["level"] != "Average":
            qualifier = "above" if ctx["level"] == "High" else "below"
            details.append(
                f"{name} of {ctx['value']}{unit} is {qualifier} the average ({ctx['mean']}{unit})"
            )

    if risk_level == "Low":
        base = f"Flood risk is LOW ({pct}% probability)."
    elif risk_level == "Medium":
        base = f"Flood risk is MEDIUM ({pct}% probability)."
    else:
        base = f"Flood risk is HIGH ({pct}% probability)."

    explanation = f"{base} The primary drivers are {drivers}."
    if details:
        explanation += " " + "; ".join(details) + "."

    return explanation


def _get_shap_values(model, df, top_n=7):
    """
    Compute SHAP values for a single prediction using TreeExplainer.

    Returns list of {feature, display_name, shap_value} sorted by |shap_value|.
    """
    preprocessor = model.named_steps["preprocessor"]
    clf = model.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        return []

    feature_names = [
        name.replace("cat__", "").replace("num__", "")
        for name in feature_names
    ]

    # Transform input through the preprocessor
    X_transformed = preprocessor.transform(df)

    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_transformed)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Old format: list of arrays per class
            vals = shap_values[1][0]
        elif shap_values.ndim == 3:
            # New format: (n_samples, n_features, n_classes)
            vals = shap_values[0, :, 1]
        else:
            vals = shap_values[0]

        # Sort by absolute value, take top N
        sorted_indices = np.argsort(np.abs(vals))[::-1][:top_n]

        shap_results = []
        for idx in sorted_indices:
            fname = feature_names[idx]
            display = FEATURE_DISPLAY_NAMES.get(fname, fname.replace("_", " ").title())
            shap_results.append({
                "feature": fname,
                "display_name": display,
                "shap_value": round(float(vals[idx]), 4),
            })
        return shap_results
    except Exception:
        return []


def predict(model, input_data: dict):
    """
    Run prediction on a single input sample.

    Args:
        model: Trained sklearn pipeline.
        input_data: dict with keys matching ALL_FEATURES.

    Returns:
        dict with prediction, probability, risk_level, and explanation data.
    """
    df = pd.DataFrame([input_data])[ALL_FEATURES]
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    risk_level = classify_risk(probability)

    feature_contributions = _get_feature_contributions(model)
    input_context = _get_input_context(input_data)
    explanation = _generate_explanation(
        risk_level, probability, input_context, feature_contributions
    )
    shap_values = _get_shap_values(model, df)

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_level": risk_level,
        "feature_contributions": feature_contributions,
        "input_context": input_context,
        "explanation": explanation,
        "shap_values": shap_values,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flood Risk Inference")
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="JSON string with input features",
    )
    args = parser.parse_args()

    input_data = json.loads(args.json)
    model = load_model()
    result = predict(model, input_data)
    print(json.dumps(result, indent=2))
