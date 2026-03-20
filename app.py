import streamlit as st
import pandas as pd
import numpy as np
import math
import os

st.set_page_config(
    page_title="Weather Predictor",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading data and training model (first run only)...")
def initialize():
    """Load data, preprocess, engineer features, train or load model."""
    from src.config import MODELS_DIR
    from src.model_trainer import load_model_artifacts

    # Try loading pre-trained artifacts first
    required_files = [
        MODELS_DIR / "weather_model.keras",
        MODELS_DIR / "feature_scaler.pkl",
        MODELS_DIR / "target_scaler.pkl",
        MODELS_DIR / "label_encoder.pkl",
        MODELS_DIR / "location_index.pkl",
        MODELS_DIR / "historical_agg.pkl",
    ]
    if all(f.exists() for f in required_files):
        return load_model_artifacts()

    # Train from scratch
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

    return {
        "model": model,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "label_encoder": label_encoder,
        "location_index": location_index,
        "historical_agg": historical_agg,
    }


artifacts = initialize()
location_index = artifacts["location_index"]

# ── Sidebar: Location Input ──────────────────────────────────────────────

st.sidebar.header("Select Location")

input_method = st.sidebar.radio(
    "How would you like to specify a location?",
    ["Pick on Map", "Search by City Name", "Enter Coordinates", "Paste Google Maps URL"],
)

lat, lon = None, None
location_display_name = None

if input_method == "Search by City Name":
    city_query = st.sidebar.text_input("Enter city or location name")
    if city_query:
        matches = location_index[
            location_index["location_name"].str.lower().str.contains(
                city_query.lower(), na=False, regex=False
            )
        ]
        if len(matches) > 0:
            selected = st.sidebar.selectbox(
                "Select from matches:",
                matches["location_name"].tolist(),
            )
            row = matches[matches["location_name"] == selected].iloc[0]
            lat, lon = row["latitude"], row["longitude"]
            location_display_name = f"{selected}, {row['country']}"
        else:
            st.sidebar.warning(
                "No matching location found in dataset. "
                "Try entering coordinates or picking on the map instead."
            )

elif input_method == "Enter Coordinates":
    lat = st.sidebar.number_input(
        "Latitude", min_value=-90.0, max_value=90.0, value=0.0, step=0.01, format="%.4f"
    )
    lon = st.sidebar.number_input(
        "Longitude", min_value=-180.0, max_value=180.0, value=0.0, step=0.01, format="%.4f"
    )
    location_display_name = f"({lat:.4f}, {lon:.4f})"

elif input_method == "Paste Google Maps URL":
    maps_url = st.sidebar.text_input("Paste Google Maps URL")
    if maps_url:
        from src.google_maps_parser import extract_coords_from_google_maps_url
        coords = extract_coords_from_google_maps_url(maps_url)
        if coords:
            lat, lon = coords
            location_display_name = f"({lat:.4f}, {lon:.4f})"
        else:
            st.sidebar.error(
                "Could not extract coordinates from URL. "
                "Try a different URL format or enter coordinates manually."
            )

elif input_method == "Pick on Map":
    st.sidebar.write("Click on the map to select a location")
    try:
        import folium
        from streamlit_folium import st_folium

        m = folium.Map(location=[20, 0], zoom_start=2)
        map_data = st_folium(m, height=600, use_container_width=True, key="location_picker")
        st.sidebar.caption("After selecting a point and generating forecast, scroll down to view results.")
        if map_data and map_data.get("last_clicked"):
            lat = map_data["last_clicked"]["lat"]
            lon = map_data["last_clicked"]["lng"]
            location_display_name = f"({lat:.4f}, {lon:.4f})"
            st.sidebar.success(f"Selected: {location_display_name}")
    except ImportError:
        st.sidebar.error(
            "Map widget requires streamlit-folium and folium packages. "
            "Install with: pip install streamlit-folium folium"
        )

# ── Main Content ─────────────────────────────────────────────────────────

st.title("30-Day Weather Forecast")

if lat is not None and lon is not None:
    if st.sidebar.button("Generate Forecast", type="primary"):
        with st.spinner("Generating 30-day forecast..."):
            from src.predictor import predict_weather
            from src.summary_generator import (
                generate_monthly_overview,
                generate_weekly_summary,
                generate_air_quality_advisory,
            )

            predictions, match_info = predict_weather(lat, lon, artifacts, n_days=30)

        # ── Location match header ──
        conf = match_info["confidence"]
        if conf == "exact":
            conf_color = "green"
        elif conf == "approximate":
            conf_color = "orange"
        else:
            conf_color = "red"

        st.markdown(
            f"### {match_info['location_name']}, {match_info['country']} "
            f"&nbsp; <span style='color:{conf_color}; font-size:0.8em;'>"
            f"({conf} match, {match_info['distance_km']} km away)</span>",
            unsafe_allow_html=True,
        )

        if conf == "distant":
            st.warning(
                "The nearest location in the dataset is far from the selected point. "
                "Predictions may be less accurate."
            )

        st.divider()

        # ── 30-Day Overview ──
        overview = generate_monthly_overview(
            predictions, match_info["location_name"], match_info["country"]
        )
        st.markdown(overview)

        st.divider()

        # ── Weekly Breakdown ──
        st.subheader("Weekly Breakdown")
        n_weeks = math.ceil(len(predictions) / 7)

        for week_num in range(n_weeks):
            summary = generate_weekly_summary(predictions, week_num)
            if summary:
                st.markdown(summary)

                start_idx = week_num * 7
                end_idx = min(start_idx + 7, len(predictions))
                week_data = predictions.iloc[start_idx:end_idx]

                with st.expander(f"View Week {week_num + 1} charts"):
                    import plotly.express as px

                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.line(
                            week_data, x="date", y="temperature_celsius",
                            title="Temperature (°C)",
                            labels={"temperature_celsius": "°C", "date": ""},
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig = px.line(
                            week_data, x="date", y="humidity",
                            title="Humidity (%)",
                            labels={"humidity": "%", "date": ""},
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    col3, col4 = st.columns(2)
                    with col3:
                        fig = px.bar(
                            week_data, x="date", y="precip_mm",
                            title="Precipitation (mm)",
                            labels={"precip_mm": "mm", "date": ""},
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    with col4:
                        fig = px.line(
                            week_data, x="date", y="wind_kph",
                            title="Wind Speed (km/h)",
                            labels={"wind_kph": "km/h", "date": ""},
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Condition timeline
                    fig = px.bar(
                        week_data, x="date", y="condition_probability",
                        color="condition_group",
                        title="Weather Conditions",
                        labels={"condition_probability": "Confidence", "date": ""},
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── Air Quality Advisory ──
        st.subheader("Air Quality Advisory")
        aq_advisory = generate_air_quality_advisory(predictions)
        st.markdown(aq_advisory)

        with st.expander("View air quality charts"):
            import plotly.express as px

            aq_cols = {
                "air_quality_PM2.5": "PM2.5 (ug/m3)",
                "air_quality_PM10": "PM10 (ug/m3)",
                "air_quality_Ozone": "Ozone (ug/m3)",
                "air_quality_Carbon_Monoxide": "CO (ug/m3)",
                "air_quality_Nitrogen_dioxide": "NO2 (ug/m3)",
                "air_quality_Sulphur_dioxide": "SO2 (ug/m3)",
            }
            for col, label in aq_cols.items():
                fig = px.line(
                    predictions, x="date", y=col,
                    title=label,
                    labels={col: label, "date": ""},
                )
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── Full Data Table ──
        with st.expander("View full data table & download"):
            display_cols = ["date", "temperature_celsius", "humidity", "precip_mm",
                           "wind_kph", "condition_group", "condition_probability",
                           "air_quality_PM2.5", "air_quality_PM10", "air_quality_Ozone",
                           "air_quality_Carbon_Monoxide", "air_quality_Nitrogen_dioxide",
                           "air_quality_Sulphur_dioxide"]
            st.dataframe(predictions[display_cols], use_container_width=True)
            csv_data = predictions[display_cols].to_csv(index=False)
            st.download_button(
                "Download Forecast CSV",
                csv_data,
                file_name="weather_forecast_30d.csv",
                mime="text/csv",
            )

else:
    st.info("Please select a location using the sidebar to generate a forecast.")

# ── Footer ──
st.markdown("---")
st.caption(
    "Data source: GlobalWeatherRepository.csv | "
    "Model: Multi-output TensorFlow neural network | "
    "Predictions are approximations based on historical patterns."
)
