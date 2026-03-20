import pandas as pd
import numpy as np
from .config import DATA_PATH, MIN_ROWS_PER_LOCATION


def load_raw_data(path=None):
    path = path or DATA_PATH
    df = pd.read_csv(path)
    df["last_updated"] = pd.to_datetime(df["last_updated"])
    return df


def build_location_index(df):
    """Build a deduplicated reference table of locations with canonical coordinates."""
    location_stats = (
        df.groupby("location_name")
        .agg(
            country=("country", lambda x: x.mode().iloc[0]),
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            n_rows=("location_name", "size"),
        )
        .reset_index()
    )
    location_stats = location_stats[
        location_stats["n_rows"] >= MIN_ROWS_PER_LOCATION
    ].copy()
    location_stats = location_stats.reset_index(drop=True)
    location_stats["location_id"] = location_stats.index
    return location_stats


def merge_location_ids(df, location_index):
    """Merge location_id onto the main DataFrame, dropping unmatched rows."""
    loc_map = location_index[["location_name", "location_id"]].copy()
    merged = df.merge(loc_map, on="location_name", how="inner")
    return merged
