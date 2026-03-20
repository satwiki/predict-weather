import os
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .config import (
    MODELS_DIR, REGRESSION_TARGETS, CONDITION_GROUPS,
    HIDDEN_LAYERS, DROPOUT_RATE, LEARNING_RATE, EPOCHS, BATCH_SIZE,
    EARLY_STOPPING_PATIENCE, REGRESSION_LOSS_WEIGHT,
    CLASSIFICATION_LOSS_WEIGHT, VALIDATION_SPLIT_RATIO,
)
from .feature_engineering import get_feature_columns


def build_model(n_features, n_regression_targets, n_classes):
    """Build a multi-output Keras model with shared backbone and two heads."""
    inputs = keras.Input(shape=(n_features,), name="features")

    x = keras.layers.BatchNormalization()(inputs)
    for units in HIDDEN_LAYERS:
        x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.Dropout(DROPOUT_RATE)(x)

    regression_output = keras.layers.Dense(
        n_regression_targets, activation="linear", name="regression"
    )(x)

    classification_output = keras.layers.Dense(
        n_classes, activation="softmax", name="classification"
    )(x)

    model = keras.Model(inputs=inputs, outputs=[regression_output, classification_output])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            "regression": "mse",
            "classification": "categorical_crossentropy",
        },
        loss_weights={
            "regression": REGRESSION_LOSS_WEIGHT,
            "classification": CLASSIFICATION_LOSS_WEIGHT,
        },
        metrics={
            "regression": ["mae"],
            "classification": ["accuracy"],
        },
    )
    return model


def train_model(df):
    """
    Full training pipeline:
    1. Extract features and targets
    2. Scale features and regression targets
    3. Encode classification target
    4. Split chronologically
    5. Train with early stopping
    6. Save all artifacts
    """
    import tensorflow as tf
    from tensorflow import keras

    os.makedirs(MODELS_DIR, exist_ok=True)

    feature_cols = get_feature_columns()
    X = df[feature_cols].values.astype(np.float32)
    y_reg = df[REGRESSION_TARGETS].values.astype(np.float32)

    le = LabelEncoder()
    le.fit(CONDITION_GROUPS)  # Fit on canonical list so ordering is consistent
    y_cls_int = le.transform(df["condition_group"].values)
    n_classes = len(le.classes_)
    y_cls_onehot = np.eye(n_classes, dtype=np.float32)[y_cls_int]

    # Scale features
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)

    # Scale regression targets
    target_scaler = StandardScaler()
    y_reg_scaled = target_scaler.fit_transform(y_reg)

    # Chronological split
    split_idx = int(len(X_scaled) * (1 - VALIDATION_SPLIT_RATIO))
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_reg_train, y_reg_val = y_reg_scaled[:split_idx], y_reg_scaled[split_idx:]
    y_cls_train, y_cls_val = y_cls_onehot[:split_idx], y_cls_onehot[split_idx:]

    model = build_model(
        n_features=X_scaled.shape[1],
        n_regression_targets=y_reg_scaled.shape[1],
        n_classes=n_classes,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=5,
            factor=0.5,
            min_lr=1e-6,
        ),
    ]

    model.fit(
        X_train,
        {"regression": y_reg_train, "classification": y_cls_train},
        validation_data=(
            X_val,
            {"regression": y_reg_val, "classification": y_cls_val},
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Save all artifacts
    model.save(str(MODELS_DIR / "weather_model.keras"))

    with open(MODELS_DIR / "feature_scaler.pkl", "wb") as f:
        pickle.dump(feature_scaler, f)
    with open(MODELS_DIR / "target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)
    with open(MODELS_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    return model, feature_scaler, target_scaler, le


def save_data_artifacts(location_index, historical_agg):
    """Save location index and historical aggregates for prediction time."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODELS_DIR / "location_index.pkl", "wb") as f:
        pickle.dump(location_index, f)
    with open(MODELS_DIR / "historical_agg.pkl", "wb") as f:
        pickle.dump(historical_agg, f)


def load_model_artifacts():
    """Load all saved artifacts from models/."""
    import tensorflow as tf
    from tensorflow import keras

    model = keras.models.load_model(str(MODELS_DIR / "weather_model.keras"))

    with open(MODELS_DIR / "feature_scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)
    with open(MODELS_DIR / "target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)
    with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open(MODELS_DIR / "location_index.pkl", "rb") as f:
        location_index = pickle.load(f)
    with open(MODELS_DIR / "historical_agg.pkl", "rb") as f:
        historical_agg = pickle.load(f)

    return {
        "model": model,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "label_encoder": label_encoder,
        "location_index": location_index,
        "historical_agg": historical_agg,
    }
