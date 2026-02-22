"""
Utility functions for the Flood Risk Classification project.
"""

import json
import pandas as pd
from src.config import DATA_PATH, CATEGORICAL_FEATURES


def get_categorical_metadata(path=None):
    """
    Extract unique values for each categorical feature from the dataset.
    Used by the API /metadata endpoint.
    """
    path = path or DATA_PATH
    df = pd.read_csv(path)
    metadata = {}
    for col in CATEGORICAL_FEATURES:
        metadata[col] = sorted(df[col].dropna().unique().tolist())
    return metadata


def get_district_mappings(path=None):
    """
    Build per-district mappings for auto-fill fields.

    Returns a dict keyed by district name, each containing:
    - divisions: list of divisions for that district
    - climate_zone: the single climate zone for that district
    - district_flood_prone: 0 or 1 for that district
    - drainage_qualities: list of drainage quality values seen
    """
    path = path or DATA_PATH
    df = pd.read_csv(path)

    mappings = {}
    for district, group in df.groupby("district"):
        mappings[district] = {
            "divisions": sorted(group["division"].dropna().unique().tolist()),
            "climate_zone": group["climate_zone"].mode().iloc[0],
            "district_flood_prone": int(group["district_flood_prone"].mode().iloc[0]),
            "drainage_qualities": sorted(
                group["drainage_quality"].dropna().unique().tolist()
            ),
            "drainage_default": group["drainage_quality"].mode().iloc[0],
        }

    return mappings


def get_feature_statistics(path=None):
    """
    Compute statistical baselines for key numeric input features.
    Used by the frontend to show 'Above avg' / 'Below avg' context.
    """
    path = path or DATA_PATH
    df = pd.read_csv(path)
    stats = {}
    for col in ["rainfall_mm", "river_level_m", "soil_saturation_percent"]:
        stats[col] = {
            "mean": round(float(df[col].mean()), 2),
            "p25": round(float(df[col].quantile(0.25)), 2),
            "p75": round(float(df[col].quantile(0.75)), 2),
        }
    return stats


def format_metrics(metrics_dict):
    """Pretty-print metrics dictionary."""
    return json.dumps(metrics_dict, indent=2)
