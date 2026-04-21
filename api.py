import io
import csv
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from src.config import MODELS_DIR, REGRESSION_TARGETS
from src.model_trainer import load_model_artifacts
from src.predictor import predict_weather
from src.summary_generator import (
    generate_monthly_overview,
    generate_weekly_summary,
    generate_air_quality_advisory,
)

artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    required_files = [
        MODELS_DIR / "weather_model.keras",
        MODELS_DIR / "feature_scaler.pkl",
        MODELS_DIR / "target_scaler.pkl",
        MODELS_DIR / "label_encoder.pkl",
        MODELS_DIR / "location_index.pkl",
        MODELS_DIR / "historical_agg.pkl",
    ]
    if not all(f.exists() for f in required_files):
        from src.data_loader import load_raw_data, build_location_index, merge_location_ids
        from src.preprocessing import preprocess_pipeline
        from src.feature_engineering import (
            add_calendar_features, add_location_features,
            compute_historical_aggregates, build_feature_matrix,
        )
        from src.model_trainer import train_model, save_data_artifacts

        df = load_raw_data()
        location_index = build_location_index(df)
        df = merge_location_ids(df, location_index)
        df = preprocess_pipeline(df)
        df = add_calendar_features(df)
        df = add_location_features(df)
        historical_agg = compute_historical_aggregates(df)
        df = build_feature_matrix(df, historical_agg)
        model, feature_scaler, target_scaler, label_encoder = train_model(df)
        save_data_artifacts(location_index, historical_agg)
        artifacts.update({
            "model": model,
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "label_encoder": label_encoder,
            "location_index": location_index,
            "historical_agg": historical_agg,
        })
    else:
        artifacts.update(load_model_artifacts())
    yield
    artifacts.clear()


app = FastAPI(
    title="Weather Prediction API",
    description="30-day weather and air quality forecasts for global locations.",
    lifespan=lifespan,
)


# -- Request / response schemas --

class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    days: int = Field(default=30, ge=1, le=90)


class DayForecast(BaseModel):
    date: str
    temperature_celsius: float
    humidity: float
    precip_mm: float
    wind_kph: float
    condition_group: str
    condition_probability: float
    air_quality_PM2_5: float
    air_quality_PM10: float
    air_quality_Ozone: float
    air_quality_Carbon_Monoxide: float
    air_quality_Nitrogen_dioxide: float
    air_quality_Sulphur_dioxide: float


class MatchInfo(BaseModel):
    location_id: int
    location_name: str
    country: str
    latitude: float
    longitude: float
    distance_km: float
    confidence: str


class PredictResponse(BaseModel):
    match_info: MatchInfo
    overview: str
    air_quality_advisory: str
    daily_forecasts: list[DayForecast]


class BatchLocationResult(BaseModel):
    latitude: float
    longitude: float
    match_info: MatchInfo
    daily_forecasts: list[DayForecast]
    error: str | None = None


class BatchResponse(BaseModel):
    results: list[BatchLocationResult]


# -- Helpers --

def _build_day_forecast(row: pd.Series) -> DayForecast:
    return DayForecast(
        date=str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
        temperature_celsius=round(float(row["temperature_celsius"]), 2),
        humidity=round(float(row["humidity"]), 2),
        precip_mm=round(float(row["precip_mm"]), 2),
        wind_kph=round(float(row["wind_kph"]), 2),
        condition_group=str(row["condition_group"]),
        condition_probability=round(float(row["condition_probability"]), 4),
        air_quality_PM2_5=round(float(row["air_quality_PM2.5"]), 2),
        air_quality_PM10=round(float(row["air_quality_PM10"]), 2),
        air_quality_Ozone=round(float(row["air_quality_Ozone"]), 2),
        air_quality_Carbon_Monoxide=round(float(row["air_quality_Carbon_Monoxide"]), 2),
        air_quality_Nitrogen_dioxide=round(float(row["air_quality_Nitrogen_dioxide"]), 2),
        air_quality_Sulphur_dioxide=round(float(row["air_quality_Sulphur_dioxide"]), 2),
    )


def _predict_single(lat: float, lon: float, days: int) -> tuple[list[DayForecast], dict, str, str]:
    predictions, match_info = predict_weather(lat, lon, artifacts, n_days=days)
    daily = [_build_day_forecast(predictions.iloc[i]) for i in range(len(predictions))]
    overview = generate_monthly_overview(predictions, match_info["location_name"], match_info["country"])
    advisory = generate_air_quality_advisory(predictions)
    return daily, match_info, overview, advisory


# -- Endpoints --

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        daily, match_info, overview, advisory = _predict_single(req.latitude, req.longitude, req.days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(
        match_info=MatchInfo(**match_info),
        overview=overview,
        air_quality_advisory=advisory,
        daily_forecasts=daily,
    )


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(file: UploadFile = File(...)):
    """
    Accept a CSV with columns `latitude` and `longitude` (and optionally `days`).
    Returns forecasts for each row.
    """
    if file.content_type and file.content_type not in ("text/csv", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="File must be a CSV.")

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded CSV.")

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)

    if not rows:
        raise HTTPException(status_code=400, detail="CSV is empty.")

    if "latitude" not in rows[0] or "longitude" not in rows[0]:
        raise HTTPException(
            status_code=400,
            detail="CSV must have 'latitude' and 'longitude' columns.",
        )

    MAX_LOCATIONS = 50
    if len(rows) > MAX_LOCATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Batch limited to {MAX_LOCATIONS} locations per request.",
        )

    results: list[BatchLocationResult] = []
    for row in rows:
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            days = int(row.get("days", 30))
        except (ValueError, TypeError) as e:
            results.append(BatchLocationResult(
                latitude=0, longitude=0,
                match_info=MatchInfo(location_id=0, location_name="", country="",
                                     latitude=0, longitude=0, distance_km=0, confidence=""),
                daily_forecasts=[],
                error=f"Invalid row values: {e}",
            ))
            continue

        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            results.append(BatchLocationResult(
                latitude=lat, longitude=lon,
                match_info=MatchInfo(location_id=0, location_name="", country="",
                                     latitude=0, longitude=0, distance_km=0, confidence=""),
                daily_forecasts=[],
                error="Latitude must be -90..90 and longitude -180..180.",
            ))
            continue

        days = max(1, min(days, 90))

        try:
            daily, match_info, _, _ = _predict_single(lat, lon, days)
            results.append(BatchLocationResult(
                latitude=lat,
                longitude=lon,
                match_info=MatchInfo(**match_info),
                daily_forecasts=daily,
            ))
        except Exception as e:
            results.append(BatchLocationResult(
                latitude=lat, longitude=lon,
                match_info=MatchInfo(location_id=0, location_name="", country="",
                                     latitude=0, longitude=0, distance_km=0, confidence=""),
                daily_forecasts=[],
                error=str(e),
            ))

    return BatchResponse(results=results)
