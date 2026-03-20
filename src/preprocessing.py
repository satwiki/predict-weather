import numpy as np
import pandas as pd
from .config import AQ_COLUMNS, WIND_KPH_CAP, CONDITION_GROUP_MAP


def replace_sentinels(df):
    """Replace sentinel values (-9999) and negative AQ values with NaN, then fill."""
    df = df.copy()
    for col in AQ_COLUMNS:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
    aq_present = [c for c in AQ_COLUMNS if c in df.columns]
    df[aq_present] = df.groupby("location_id")[aq_present].transform(
        lambda g: g.ffill().bfill()
    )
    # If any NaN remain (e.g. entire location is NaN), fill with column median
    for col in aq_present:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def cap_outliers(df):
    """Cap wind speeds at physical limits."""
    df = df.copy()
    df["wind_kph"] = df["wind_kph"].clip(upper=WIND_KPH_CAP)
    df["wind_mph"] = df["wind_mph"].clip(upper=WIND_KPH_CAP / 1.60934)
    if "gust_kph" in df.columns:
        df["gust_kph"] = df["gust_kph"].clip(upper=WIND_KPH_CAP * 1.5)
    if "gust_mph" in df.columns:
        df["gust_mph"] = df["gust_mph"].clip(upper=WIND_KPH_CAP * 1.5 / 1.60934)
    return df


def normalize_condition_text(df):
    """Map condition_text to broader condition_group categories."""
    df = df.copy()
    df["condition_text"] = df["condition_text"].str.strip()
    df["condition_group"] = df["condition_text"].map(CONDITION_GROUP_MAP)
    # Fallback for any unmapped conditions
    df["condition_group"] = df["condition_group"].fillna("Cloudy/Overcast")
    return df


def preprocess_pipeline(df):
    """Run full preprocessing: sentinels → outlier cap → condition normalization."""
    df = replace_sentinels(df)
    df = cap_outliers(df)
    df = normalize_condition_text(df)
    return df
