# Predice-Weather app

A web app that predicts weather and air quality for the coming 30 days at available locations worldwide. Powered by a TensorFlow neural network trained on 23 months of global weather observations from 205 cities worldwide. User can navigate the app by selecting a location on the map, entering co-ords or google map url, or searching by city name. If selected location is not available in the model's training dataset, the app will show the user nearest available location's weather prediction.

## Dataset

The model is trained on the **GlobalWeatherRepository** dataset (from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository?resource=download)) containing **128,345 daily weather observations** across **205 cities** in **186 countries**, spanning from May 2024 to March 2026. Each observation includes 41 attributes covering temperature, humidity, precipitation, wind, visibility, UV index, cloud cover, and six air quality metrics (PM2.5, PM10, Ozone, CO, NO2, SO2).

Data cleaning and pre-processing notes:
- Locations with fewer than 50 data points were filtered out (removed 52 sparse or malformed entries, e.g. single-observation cities).
- Four sentinel values (-9999) in air quality columns were replaced via forward/backward fill within each location group.
- Four erroneous wind speed readings (up to 2963 km/h) were capped at 200 km/h.
- The 49 raw weather condition strings (with case inconsistencies like "Partly Cloudy" vs "Partly cloudy") were normalized and grouped into 11 broader categories: Clear/Sunny, Partly Cloudy, Cloudy/Overcast, Mist/Fog, Light Rain, Moderate Rain, Heavy Rain/Storm, Thunderstorm, Light Snow, Moderate/Heavy Snow, and Sleet.

## Feature Engineering

The model uses **30 engineered features** derived from date and location. No lag features are used — this avoids compounding prediction errors across the 30-day forecast horizon.

| Category | Features | Count |
|---|---|---|
| **Cyclical calendar** | day_of_year (sin/cos), month (sin/cos), day_of_week (sin/cos) — encoded with sine/cosine to capture periodicity (e.g. Dec 31 is close to Jan 1) | 6 |
| **Location** | latitude, longitude, absolute latitude (climate zone proxy), hemisphere flag (northern=1, southern=0) | 4 |
| **Historical aggregates** | For each of the 10 regression targets: the historical monthly mean and standard deviation at that location. These capture "what does this city typically look like in March?" without requiring recurrent connections or lag values. | 20 |

The historical aggregates are the key to making the model location-aware — they encode each city's seasonal climate fingerprint. For a query location not in the dataset, the system finds the nearest known city by haversine distance and uses its historical statistics.

## Prediction Model

### Architecture

A single **multi-output neural network** built with TensorFlow/Keras using the Functional API. The network has a shared backbone of fully connected layers that branches into two specialized output heads:

```
Input (30 features)
  → BatchNormalization
  → Dense(128, ReLU) → Dropout(0.2)
  → Dense(64, ReLU)  → Dropout(0.2)
  → Dense(32, ReLU)  → Dropout(0.2)
  ├── Regression head:      Dense(10, linear)   → temperature, humidity, precipitation,
  │                                                wind, PM2.5, PM10, Ozone, CO, NO2, SO2
  └── Classification head:  Dense(11, softmax)  → weather condition probabilities
```

Total parameters: **15,057 trainable** (58 KB). The model is deliberately compact — with ~128K training rows and 30 features, a larger network would overfit.

### Why a single model instead of separate models per target

The shared hidden layers learn cross-target correlations that physically exist in weather (high humidity correlates with precipitation, which correlates with cloud cover and rain conditions). A single forward pass produces all 11 predictions simultaneously, which is faster than running 11 independent models.

### Training

- **Loss function**: Weighted sum of MSE (regression head, weight 1.0) and categorical cross-entropy (classification head, weight 0.3). Regression is weighted higher because it covers 10 continuous targets.
- **Optimizer**: Adam with learning rate 0.001, reduced by half on plateau (patience 5).
- **Validation**: Last 20% of data chronologically (not shuffled) to respect temporal ordering.
- **Early stopping**: Patience 10 on validation loss, restoring best weights. Typically converges in ~20 epochs.
- **Feature scaling**: StandardScaler on all 30 input features. Regression targets are also scaled during training and inverse-transformed at prediction time.
- **Condition encoding**: The 11 weather condition groups are label-encoded then one-hot encoded for the softmax classification head.

### How prediction works

1. **Location matching**: The user specifies a location by city name, coordinates, Google Maps URL, or map click. The system finds the nearest city in the dataset using haversine distance and reports match confidence (exact <50km, approximate 50-500km, distant >500km).
2. **Feature construction**: For each of the next 30 days, calendar features are computed from the date, location features from the matched city's coordinates, and historical aggregates are looked up from the matched city's per-month statistics.
3. **Model inference**: The 30-day feature matrix is scaled and passed through the network in a single `model.predict()` call. The regression head outputs are inverse-scaled to original units. The classification head's argmax gives the predicted weather condition.
4. **Post-processing**: Physical constraints are enforced (humidity clipped to 0-100%, precipitation and wind clipped to non-negative, air quality clipped to non-negative).
5. **Natural language generation**: Rule-based templates convert the raw predictions into a monthly overview paragraph, weekly breakdown summaries, and an air quality advisory. Detailed charts and the full data table are available in expandable sections.

## Running the Project

### Prerequisites

- Python 3.12+
- pip

### Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. On first launch, the model trains automatically on the local dataset. Subsequent launches will load the cached model from the `models/` directory instantly.

### Docker

```bash
docker-compose up --build
```

The app is accessible at `http://localhost:8501`. The `models/` directory is mounted as a volume, so the trained model persists across container restarts.