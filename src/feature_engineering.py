import numpy as np
import pandas as pd
from .config import REGRESSION_TARGETS


def add_calendar_features(df):
    """Add cyclical sin/cos calendar features from last_updated."""
    df = df.copy()
    dt = df["last_updated"]
    doy = dt.dt.dayofyear
    month = dt.dt.month
    dow = dt.dt.dayofweek

    df["day_of_year"] = doy
    df["month"] = month
    df["day_of_week"] = dow
    df["year"] = dt.dt.year

    df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


def add_location_features(df):
    """Add derived location features."""
    df = df.copy()
    df["lat_abs"] = df["latitude"].abs()
    df["lat_hemisphere"] = (df["latitude"] >= 0).astype(int)
    return df


def compute_historical_aggregates(df):
    """
    Compute per-(location_id, month) mean and std for each regression target.
    Returns a DataFrame keyed by (location_id, month).
    """
    agg_dict = {}
    for t in REGRESSION_TARGETS:
        agg_dict[f"{t}_month_mean"] = (t, "mean")
        agg_dict[f"{t}_month_std"] = (t, "std")

    hist_agg = df.groupby(["location_id", "month"]).agg(**agg_dict).reset_index()
    # Fill NaN std (groups with 1 row) with 0
    std_cols = [c for c in hist_agg.columns if c.endswith("_month_std")]
    hist_agg[std_cols] = hist_agg[std_cols].fillna(0)
    return hist_agg


def get_feature_columns():
    """Return the complete list of feature column names."""
    base = [
        "day_of_year_sin", "day_of_year_cos",
        "month_sin", "month_cos",
        "day_of_week_sin", "day_of_week_cos",
        "latitude", "longitude", "lat_abs", "lat_hemisphere",
    ]
    for t in REGRESSION_TARGETS:
        base.append(f"{t}_month_mean")
        base.append(f"{t}_month_std")
    return base


def build_feature_matrix(df, historical_agg):
    """Merge historical aggregates onto df and return with feature columns populated."""
    df = df.copy()
    merge_cols = ["location_id", "month"]
    agg_cols = [c for c in historical_agg.columns if c not in merge_cols]

    # Drop any existing aggregate columns to avoid conflicts
    for col in agg_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.merge(historical_agg, on=merge_cols, how="left")

    # Fill any missing aggregates with column-level means
    for col in agg_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

    return df
