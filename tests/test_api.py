"""
Unit tests for the FastAPI backend.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test the /health endpoint returns ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metadata_endpoint():
    """Test the /metadata endpoint returns categorical values and district mappings."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "categorical" in data
    assert "district_mappings" in data
    cat = data["categorical"]
    assert "district" in cat
    assert isinstance(cat["district"], list)
    assert len(cat["district"]) > 0
    # Verify district mappings
    mappings = data["district_mappings"]
    assert "Colombo" in mappings
    colombo = mappings["Colombo"]
    assert "divisions" in colombo
    assert "climate_zone" in colombo
    assert "district_flood_prone" in colombo
    assert colombo["climate_zone"] == "Wet"


def test_predict_endpoint():
    """Test the /predict endpoint with valid input."""
    payload = {
        "district": "Colombo",
        "division": "Colombo-DS3",
        "climate_zone": "Wet",
        "drainage_quality": "Poor",
        "year": 2023,
        "month": 5,
        "rainfall_mm": 350.0,
        "river_level_m": 3.5,
        "soil_saturation_percent": 65.0,
        "district_flood_prone": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "risk_level" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1
    assert data["risk_level"] in ["Low", "Medium", "High"]


def test_predict_invalid_input():
    """Test the /predict endpoint with missing fields."""
    payload = {"district": "Colombo"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error
