"""
Hybrid CNN + LSTM model for spatio-temporal calamity prediction.

Architecture:
    Input(30, 6)
        -> Conv1D(64, k=3, relu) -> MaxPooling1D(2) -> Conv1D(32, k=3, relu)
        -> LSTM(128, return_sequences=True) -> LSTM(64)
        -> Dense(32, relu) -> Dropout(0.3) -> Dense(2, sigmoid)

Output: [flood_probability, cyclone_probability] (independent, multi-label).

Built from scratch (no AutoML, no pre-trained weights).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    LSTM,
    MaxPooling1D,
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


INPUT_SHAPE = (30, 5)
NUM_OUTPUTS = 2


def build_model(input_shape: tuple[int, int] = INPUT_SHAPE) -> Model:
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=3, activation="relu")(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(32, kernel_size=3, activation="relu")(x)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_OUTPUTS, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs, name="hybrid_cnn_lstm")
    model.compile(
        optimizer=Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_local(X, y, epochs: int = 5, batch_size: int = 32):
    model = build_model()
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model.get_weights()


def evaluate_model(X_test, y_test) -> dict:
    model = build_model()
    y_prob = model.predict(X_test, verbose=0)
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = np.asarray(y_test).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def get_initial_weights(seed: int = 42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    return build_model().get_weights()
