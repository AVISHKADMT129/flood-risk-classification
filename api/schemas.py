"""
Pydantic schemas for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    district: str = Field(..., json_schema_extra={"example": "Colombo"})
    division: str = Field(..., json_schema_extra={"example": "Colombo-DS3"})
    climate_zone: str = Field(..., json_schema_extra={"example": "Wet"})
    drainage_quality: str = Field(..., json_schema_extra={"example": "Moderate"})
    year: int = Field(..., json_schema_extra={"example": 2023})
    month: int = Field(..., ge=1, le=12, json_schema_extra={"example": 5})
    rainfall_mm: float = Field(..., ge=0, json_schema_extra={"example": 250.0})
    river_level_m: float = Field(..., ge=0, json_schema_extra={"example": 3.2})
    soil_saturation_percent: float = Field(..., ge=0, le=100, json_schema_extra={"example": 55.0})
    district_flood_prone: int = Field(..., ge=0, le=1, json_schema_extra={"example": 1})


class FeatureContribution(BaseModel):
    feature: str
    display_name: str
    importance: float
    rank: int


class InputContext(BaseModel):
    value: float
    mean: float
    p25: float
    p75: float
    level: str


class ShapValue(BaseModel):
    feature: str
    display_name: str
    shap_value: float


class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    feature_contributions: list[FeatureContribution] = []
    input_context: dict[str, InputContext] = {}
    explanation: str = ""
    shap_values: list[ShapValue] = []


class HealthResponse(BaseModel):
    status: str
