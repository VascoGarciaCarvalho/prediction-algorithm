"""
Neural Network Model
Conv1D → LSTM → Monte Carlo Dropout → Binary classification
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Monte Carlo Dropout (dropout active at inference time)
# ---------------------------------------------------------------------------

class MCDropout(layers.Layer):
    """Dropout that stays active during inference for uncertainty estimation."""

    def __init__(self, rate: float, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        # Always apply dropout, regardless of training flag
        return tf.nn.dropout(inputs, rate=self.rate)

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def build_model(
    input_shape: tuple[int, int],
    conv_filters: int = 64,
    conv_kernel_size: int = 3,
    lstm_units: int = 64,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Build the Conv1D → LSTM → MC Dropout model.

    Args:
        input_shape: (window_size, n_features)
        conv_filters: Number of Conv1D filters.
        conv_kernel_size: Kernel size for Conv1D.
        lstm_units: Number of LSTM hidden units.
        dropout_rate: Dropout rate (applied with MC Dropout).
        learning_rate: Adam learning rate.

    Returns:
        Compiled Keras model.
    """
    inputs = keras.Input(shape=input_shape, name="input")

    # --- Layer 1: 1D Convolution ---
    # Captures short-term trends / local dependencies in the time series
    x = layers.Conv1D(
        filters=conv_filters,
        kernel_size=conv_kernel_size,
        padding="causal",          # no future leakage
        activation="relu",
        name="conv1d",
    )(inputs)
    x = layers.BatchNormalization(name="bn_conv")(x)
    x = MCDropout(rate=dropout_rate, name="mc_dropout_conv")(x)

    # --- Layer 2: LSTM ---
    # Captures long-range sequential dependencies
    x = layers.LSTM(
        units=lstm_units,
        return_sequences=False,
        name="lstm",
    )(x)
    x = layers.BatchNormalization(name="bn_lstm")(x)
    x = MCDropout(rate=dropout_rate, name="mc_dropout_lstm")(x)

    # --- Output: binary classification ---
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="conv_lstm_predictor")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Monte Carlo Inference
# ---------------------------------------------------------------------------

def mc_predict(
    model: keras.Model,
    X: np.ndarray,
    n_passes: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run multiple forward passes with dropout active to estimate uncertainty.

    Args:
        model: Trained Keras model with MCDropout layers.
        X: Input array of shape (n_samples, window_size, n_features).
        n_passes: Number of stochastic forward passes.

    Returns:
        mean_probs: Mean predicted probability per sample.
        std_probs: Standard deviation (uncertainty) per sample.
    """
    preds = np.stack(
        [model(X, training=True).numpy().squeeze() for _ in range(n_passes)],
        axis=0,
    )  # shape: (n_passes, n_samples)
    mean_probs = preds.mean(axis=0)
    std_probs = preds.std(axis=0)
    return mean_probs, std_probs


# ---------------------------------------------------------------------------
# Model Summary Utility
# ---------------------------------------------------------------------------

def print_model_summary(model: keras.Model) -> None:
    model.summary()
