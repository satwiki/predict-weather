import pandas as pd
import numpy as np


def _describe_temp(min_t, max_t, mean_t):
    """Return a natural-language temperature phrase."""
    if mean_t < 0:
        feel = "freezing"
    elif mean_t < 10:
        feel = "cold"
    elif mean_t < 20:
        feel = "mild"
    elif mean_t < 30:
        feel = "warm"
    else:
        feel = "hot"
    return f"{feel} with temperatures between {min_t:.0f}°C and {max_t:.0f}°C"


def _describe_wind(avg_wind):
    if avg_wind < 10:
        return "calm winds"
    elif avg_wind < 25:
        return f"light winds around {avg_wind:.0f} km/h"
    elif avg_wind < 40:
        return f"moderate winds around {avg_wind:.0f} km/h"
    else:
        return f"strong winds around {avg_wind:.0f} km/h"


def _describe_precip(total_precip):
    if total_precip < 1:
        return "No significant rainfall expected"
    elif total_precip < 10:
        return f"Light rainfall totaling about {total_precip:.0f}mm"
    elif total_precip < 30:
        return f"Moderate rainfall expected, totaling around {total_precip:.0f}mm"
    else:
        return f"Heavy rainfall expected, with a total of about {total_precip:.0f}mm"


def _dominant_condition(conditions):
    """Return the most frequent condition group."""
    return conditions.mode().iloc[0] if len(conditions) > 0 else "Cloudy/Overcast"


def _condition_phrase(dominant):
    """Convert condition group to a natural-language sky phrase."""
    phrases = {
        "Clear/Sunny": "clear and sunny skies",
        "Partly Cloudy": "partly cloudy skies",
        "Cloudy/Overcast": "overcast skies",
        "Mist/Fog": "misty or foggy conditions",
        "Light Rain": "light rain showers",
        "Moderate Rain": "periods of moderate rain",
        "Heavy Rain/Storm": "heavy rainfall and possible storms",
        "Thunderstorm": "thunderstorm activity",
        "Light Snow": "light snowfall",
        "Moderate/Heavy Snow": "significant snowfall",
        "Sleet": "sleet and wintry mix",
    }
    return phrases.get(dominant, "variable conditions")


def generate_monthly_overview(predictions_df, location_name, country):
    """Generate a one-paragraph 30-day overview in natural language."""
    temp = predictions_df["temperature_celsius"]
    precip = predictions_df["precip_mm"]
    humidity = predictions_df["humidity"]
    wind = predictions_df["wind_kph"]
    dominant = _dominant_condition(predictions_df["condition_group"])

    overview = (
        f"**30-Day Outlook for {location_name}, {country}:** "
        f"The coming month is expected to be predominantly "
        f"{_condition_phrase(dominant)}, "
        f"{_describe_temp(temp.min(), temp.max(), temp.mean())}. "
        f"Average humidity will be around {humidity.mean():.0f}%. "
        f"{_describe_precip(precip.sum())}. "
        f"Expect {_describe_wind(wind.mean())} throughout the period."
    )
    return overview


def generate_weekly_summary(predictions_df, week_num):
    """
    Generate a natural-language summary for one week.
    week_num: 0-indexed (0 = first 7 days, 1 = next 7, etc.)
    """
    start_idx = week_num * 7
    end_idx = min(start_idx + 7, len(predictions_df))
    if start_idx >= len(predictions_df):
        return ""

    week = predictions_df.iloc[start_idx:end_idx]
    dates = pd.to_datetime(week["date"])
    start_str = dates.iloc[0].strftime("%b %d")
    end_str = dates.iloc[-1].strftime("%b %d")

    temp = week["temperature_celsius"]
    precip = week["precip_mm"]
    humidity = week["humidity"]
    wind = week["wind_kph"]
    dominant = _dominant_condition(week["condition_group"])

    summary = (
        f"**Week {week_num + 1} ({start_str} - {end_str}):** "
        f"Expect {_condition_phrase(dominant)}, "
        f"{_describe_temp(temp.min(), temp.max(), temp.mean())}. "
        f"Humidity around {humidity.mean():.0f}%. "
        f"{_describe_precip(precip.sum())}. "
        f"{_describe_wind(wind.mean()).capitalize()}."
    )
    return summary


def _aq_level(pm25_avg):
    """Classify air quality based on average PM2.5."""
    if pm25_avg <= 12:
        return "good"
    elif pm25_avg <= 35:
        return "moderate"
    elif pm25_avg <= 55:
        return "unhealthy for sensitive groups"
    elif pm25_avg <= 150:
        return "unhealthy"
    else:
        return "very unhealthy"


def generate_air_quality_advisory(predictions_df):
    """Generate a plain-language air quality advisory."""
    pm25 = predictions_df["air_quality_PM2.5"]
    pm10 = predictions_df["air_quality_PM10"]
    ozone = predictions_df["air_quality_Ozone"]
    co = predictions_df["air_quality_Carbon_Monoxide"]
    no2 = predictions_df["air_quality_Nitrogen_dioxide"]
    so2 = predictions_df["air_quality_Sulphur_dioxide"]

    pm25_avg = pm25.mean()
    level = _aq_level(pm25_avg)

    parts = [f"**Air Quality:** Overall air quality is expected to be **{level}**."]
    parts.append(
        f"PM2.5 levels will average around {pm25_avg:.1f} ug/m3 "
        f"(range: {pm25.min():.1f} - {pm25.max():.1f})."
    )

    if pm10.mean() > 50:
        parts.append(
            f"PM10 levels are elevated at an average of {pm10.mean():.1f} ug/m3."
        )

    ozone_avg = ozone.mean()
    if ozone_avg > 100:
        parts.append(
            f"Ozone levels are moderately high, averaging {ozone_avg:.1f} ug/m3."
        )
    else:
        parts.append(
            f"Ozone levels are within normal range at {ozone_avg:.1f} ug/m3."
        )

    co_avg = co.mean()
    if co_avg > 10000:
        parts.append(
            f"Carbon monoxide is elevated at {co_avg:.0f} ug/m3."
        )

    return " ".join(parts)
