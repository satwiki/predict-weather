from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "GlobalWeatherRepository.csv"
MODELS_DIR = BASE_DIR / "models"

# --- Regression targets ---
REGRESSION_TARGETS = [
    "temperature_celsius",
    "humidity",
    "precip_mm",
    "wind_kph",
    "air_quality_PM2.5",
    "air_quality_PM10",
    "air_quality_Ozone",
    "air_quality_Carbon_Monoxide",
    "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide",
]

# --- Condition group mapping (title-cased raw → broad group) ---
CONDITION_GROUPS = [
    "Clear/Sunny",
    "Partly Cloudy",
    "Cloudy/Overcast",
    "Mist/Fog",
    "Light Rain",
    "Moderate Rain",
    "Heavy Rain/Storm",
    "Thunderstorm",
    "Light Snow",
    "Moderate/Heavy Snow",
    "Sleet",
]

CONDITION_GROUP_MAP = {
    "Sunny": "Clear/Sunny",
    "Clear": "Clear/Sunny",
    "Partly Cloudy": "Partly Cloudy",
    "Partly cloudy": "Partly Cloudy",
    "Cloudy": "Cloudy/Overcast",
    "Overcast": "Cloudy/Overcast",
    "Mist": "Mist/Fog",
    "Fog": "Mist/Fog",
    "Freezing Fog": "Mist/Fog",
    "Freezing fog": "Mist/Fog",
    "Light Rain": "Light Rain",
    "Light rain": "Light Rain",
    "Light Rain Shower": "Light Rain",
    "Light rain shower": "Light Rain",
    "Light Drizzle": "Light Rain",
    "Light drizzle": "Light Rain",
    "Patchy Light Drizzle": "Light Rain",
    "Patchy light drizzle": "Light Rain",
    "Patchy Light Rain": "Light Rain",
    "Patchy light rain": "Light Rain",
    "Patchy Rain Nearby": "Light Rain",
    "Patchy rain nearby": "Light Rain",
    "Patchy Rain Possible": "Light Rain",
    "Patchy rain possible": "Light Rain",
    "Freezing Drizzle": "Light Rain",
    "Freezing drizzle": "Light Rain",
    "Light Freezing Rain": "Light Rain",
    "Light freezing rain": "Light Rain",
    "Moderate Rain": "Moderate Rain",
    "Moderate rain": "Moderate Rain",
    "Moderate Rain At Times": "Moderate Rain",
    "Moderate rain at times": "Moderate Rain",
    "Moderate Or Heavy Rain Shower": "Moderate Rain",
    "Moderate or heavy rain shower": "Moderate Rain",
    "Heavy Rain": "Heavy Rain/Storm",
    "Heavy rain": "Heavy Rain/Storm",
    "Heavy Rain At Times": "Heavy Rain/Storm",
    "Heavy rain at times": "Heavy Rain/Storm",
    "Torrential Rain Shower": "Heavy Rain/Storm",
    "Torrential rain shower": "Heavy Rain/Storm",
    "Heavy Freezing Drizzle": "Heavy Rain/Storm",
    "Heavy freezing drizzle": "Heavy Rain/Storm",
    "Moderate Or Heavy Freezing Rain": "Heavy Rain/Storm",
    "Moderate or heavy freezing rain": "Heavy Rain/Storm",
    "Moderate Or Heavy Rain With Thunder": "Thunderstorm",
    "Moderate or heavy rain with thunder": "Thunderstorm",
    "Moderate Or Heavy Rain In Area With Thunder": "Thunderstorm",
    "Moderate or heavy rain in area with thunder": "Thunderstorm",
    "Patchy Light Rain With Thunder": "Thunderstorm",
    "Patchy light rain with thunder": "Thunderstorm",
    "Patchy Light Rain In Area With Thunder": "Thunderstorm",
    "Patchy light rain in area with thunder": "Thunderstorm",
    "Thundery Outbreaks In Nearby": "Thunderstorm",
    "Thundery outbreaks in nearby": "Thunderstorm",
    "Thundery Outbreaks Possible": "Thunderstorm",
    "Thundery outbreaks possible": "Thunderstorm",
    "Light Snow": "Light Snow",
    "Light snow": "Light Snow",
    "Light Snow Showers": "Light Snow",
    "Light snow showers": "Light Snow",
    "Patchy Light Snow": "Light Snow",
    "Patchy light snow": "Light Snow",
    "Patchy Snow Nearby": "Light Snow",
    "Patchy snow nearby": "Light Snow",
    "Patchy Snow Possible": "Light Snow",
    "Patchy snow possible": "Light Snow",
    "Patchy Light Snow In Area With Thunder": "Light Snow",
    "Patchy light snow in area with thunder": "Light Snow",
    "Moderate Snow": "Moderate/Heavy Snow",
    "Moderate snow": "Moderate/Heavy Snow",
    "Heavy Snow": "Moderate/Heavy Snow",
    "Heavy snow": "Moderate/Heavy Snow",
    "Patchy Heavy Snow": "Moderate/Heavy Snow",
    "Patchy heavy snow": "Moderate/Heavy Snow",
    "Patchy Moderate Snow": "Moderate/Heavy Snow",
    "Patchy moderate snow": "Moderate/Heavy Snow",
    "Moderate Or Heavy Snow Showers": "Moderate/Heavy Snow",
    "Moderate or heavy snow showers": "Moderate/Heavy Snow",
    "Moderate Or Heavy Snow In Area With Thunder": "Moderate/Heavy Snow",
    "Moderate or heavy snow in area with thunder": "Moderate/Heavy Snow",
    "Blizzard": "Moderate/Heavy Snow",
    "Blowing Snow": "Moderate/Heavy Snow",
    "Blowing snow": "Moderate/Heavy Snow",
    "Light Sleet": "Sleet",
    "Light sleet": "Sleet",
    "Light Sleet Showers": "Sleet",
    "Light sleet showers": "Sleet",
    "Moderate Or Heavy Sleet": "Sleet",
    "Moderate or heavy sleet": "Sleet",
    "Moderate Or Heavy Sleet Showers": "Sleet",
    "Moderate or heavy sleet showers": "Sleet",
    "Ice Pellets": "Sleet",
    "Ice pellets": "Sleet",
    "Light Showers Of Ice Pellets": "Sleet",
    "Light showers of ice pellets": "Sleet",
    "Moderate Or Heavy Showers Of Ice Pellets": "Sleet",
    "Moderate or heavy showers of ice pellets": "Sleet",
}

# --- TensorFlow model hyperparameters ---
HIDDEN_LAYERS = [128, 64, 32]
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 256
EARLY_STOPPING_PATIENCE = 10
REGRESSION_LOSS_WEIGHT = 1.0
CLASSIFICATION_LOSS_WEIGHT = 0.3
VALIDATION_SPLIT_RATIO = 0.2

# --- Location thresholds ---
MIN_ROWS_PER_LOCATION = 50
CLOSE_DISTANCE_KM = 50
MAX_DISTANCE_KM = 500

# --- Air quality columns ---
AQ_COLUMNS = [
    "air_quality_Carbon_Monoxide",
    "air_quality_Ozone",
    "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide",
    "air_quality_PM2.5",
    "air_quality_PM10",
]

# --- Outlier caps ---
WIND_KPH_CAP = 200.0
