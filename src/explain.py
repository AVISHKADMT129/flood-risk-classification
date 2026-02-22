"""
Explainability module.
Generates Partial Dependence Plots (PDP) for key numerical features.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

from src.config import PDP_DIR, OUTPUTS_DIR


PDP_FEATURES = ["rainfall_mm", "river_level_m", "soil_saturation_percent"]


def generate_pdp_plots(pipeline, X_data):
    """
    Generate and save Partial Dependence Plots for the key numerical features.

    Args:
        pipeline: Trained sklearn pipeline.
        X_data: DataFrame of features (test or train set).
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    for feature_name in PDP_FEATURES:
        if feature_name not in X_data.columns:
            print(f"Warning: {feature_name} not found in data, skipping PDP.")
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        PartialDependenceDisplay.from_estimator(
            pipeline,
            X_data,
            features=[feature_name],
            ax=ax,
            grid_resolution=50,
        )
        ax.set_title(f"Partial Dependence Plot - {feature_name}")
        fig.tight_layout()

        save_path = PDP_DIR / f"pdp_{feature_name}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"PDP saved to {save_path}")


if __name__ == "__main__":
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from src.config import MODEL_PATH, DATA_PATH, TEST_SIZE, RANDOM_STATE, TARGET, ALL_FEATURES

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df[ALL_FEATURES]
    y = df[TARGET]
    _, X_test, _, _ = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    generate_pdp_plots(model, X_test)
