import numpy as np
import pandas as pd
from .config import CLOSE_DISTANCE_KM, MAX_DISTANCE_KM


def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute haversine distance in km. Supports vectorized (array) inputs."""
    R = 6371.0
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def find_nearest_locations(lat, lon, location_index, k=3):
    """Find the k nearest locations to (lat, lon) from the location index."""
    distances = haversine_distance(
        lat, lon,
        location_index["latitude"].values,
        location_index["longitude"].values,
    )
    result = location_index.copy()
    result["distance_km"] = distances
    result = result.sort_values("distance_km").head(k)
    return result


def get_prediction_location(lat, lon, location_index):
    """
    Determine the best location to use for prediction.

    Returns dict with: location_id, location_name, country, latitude,
    longitude, distance_km, confidence.
    """
    nearest = find_nearest_locations(lat, lon, location_index, k=1)
    row = nearest.iloc[0]
    dist = row["distance_km"]

    if dist < CLOSE_DISTANCE_KM:
        confidence = "exact"
    elif dist < MAX_DISTANCE_KM:
        confidence = "approximate"
    else:
        confidence = "distant"

    return {
        "location_id": int(row["location_id"]),
        "location_name": row["location_name"],
        "country": row["country"],
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "distance_km": round(dist, 1),
        "confidence": confidence,
    }
