import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from .config import REGRESSION_TARGETS
from .feature_engineering import get_feature_columns
from .location_finder import get_prediction_location


def generate_future_dates(start_date, n_days=30):
    """Generate a DataFrame of future dates with calendar features."""
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    df = pd.DataFrame({"date": dates})
    df["last_updated"] = pd.to_datetime(df["date"])

    doy = df["last_updated"].dt.dayofyear
    month = df["last_updated"].dt.month
    dow = df["last_updated"].dt.dayofweek

    df["day_of_year"] = doy
    df["month"] = month
    df["day_of_week"] = dow

    df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)

    return df


def build_prediction_features(future_dates_df, matched_location, historical_agg):
    """Build the 30-column feature matrix for prediction."""
    df = future_dates_df.copy()

    # Location features
    df["latitude"] = matched_location["latitude"]
    df["longitude"] = matched_location["longitude"]
    df["lat_abs"] = abs(matched_location["latitude"])
    df["lat_hemisphere"] = int(matched_location["latitude"] >= 0)

    # Historical aggregates
    loc_id = matched_location["location_id"]
    loc_agg = historical_agg[historical_agg["location_id"] == loc_id]

    agg_cols = [c for c in historical_agg.columns if c.endswith(("_month_mean", "_month_std"))]

    df = df.merge(
        loc_agg[["month"] + agg_cols],
        on="month",
        how="left",
    )

    # Fill any missing months with global averages
    for col in agg_cols:
        if df[col].isna().any():
            global_mean = historical_agg[col].mean()
            df[col] = df[col].fillna(global_mean)

    feature_cols = get_feature_columns()
    return df, feature_cols


def predict_weather(lat, lon, artifacts, n_days=30):
    """
    End-to-end prediction: find nearest location, build features,
    run model, postprocess.
    """
    model = artifacts["model"]
    feature_scaler = artifacts["feature_scaler"]
    target_scaler = artifacts["target_scaler"]
    label_encoder = artifacts["label_encoder"]
    location_index = artifacts["location_index"]
    historical_agg = artifacts["historical_agg"]

    # Find nearest location
    match_info = get_prediction_location(lat, lon, location_index)

    # Generate future dates starting from tomorrow
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    future_df = generate_future_dates(start_date, n_days)

    # Build features
    future_df, feature_cols = build_prediction_features(
        future_df, match_info, historical_agg
    )

    X = future_df[feature_cols].values.astype(np.float32)
    X_scaled = feature_scaler.transform(X)

    # Predict
    reg_pred_scaled, cls_pred = model.predict(X_scaled, verbose=0)

    # Inverse-transform regression
    reg_pred = target_scaler.inverse_transform(reg_pred_scaled)

    # Decode classification
    cls_indices = np.argmax(cls_pred, axis=1)
    cls_labels = label_encoder.inverse_transform(cls_indices)
    cls_probs = np.max(cls_pred, axis=1)

    # Build results DataFrame
    results = pd.DataFrame({
        "date": future_df["date"].values,
    })
    for i, target in enumerate(REGRESSION_TARGETS):
        results[target] = reg_pred[:, i]

    results["condition_group"] = cls_labels
    results["condition_probability"] = cls_probs

    results = postprocess_predictions(results)

    # Attach match info
    results.attrs["match_info"] = match_info

    return results, match_info


def postprocess_predictions(df):
    """Apply physical constraints to predictions."""
    df = df.copy()
    df["humidity"] = df["humidity"].clip(0, 100)
    df["precip_mm"] = df["precip_mm"].clip(lower=0)
    df["wind_kph"] = df["wind_kph"].clip(lower=0)
    for col in df.columns:
        if col.startswith("air_quality_"):
            df[col] = df[col].clip(lower=0)
    return df
