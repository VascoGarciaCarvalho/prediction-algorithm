"""
Neural Network Model
Shared Conv1D → LSTM backbone with 4 independent sigmoid output heads,
one per time horizon (1d, 5d, 21d, 126d).
Monte Carlo Dropout active at inference time for uncertainty estimation.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from data_engineering import HORIZONS


# ---------------------------------------------------------------------------
# Monte Carlo Dropout (dropout active at inference time)
# ---------------------------------------------------------------------------

class MCDropout(layers.Layer):
    """Dropout that stays active during inference for uncertainty estimation."""

    def __init__(self, rate: float, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
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
    Build the shared Conv1D → LSTM backbone with 4 independent output heads.

    Output layer names: 'out_1d', 'out_5d', 'out_21d', 'out_126d'
    Each head predicts P(price up) for its horizon.

    Args:
        input_shape: (window_size, n_features)
    """
    inputs = keras.Input(shape=input_shape, name="input")

    # --- Shared backbone ---
    x = layers.Conv1D(
        filters=conv_filters,
        kernel_size=conv_kernel_size,
        padding="causal",
        activation="relu",
        name="conv1d",
    )(inputs)
    x = layers.BatchNormalization(name="bn_conv")(x)
    x = MCDropout(rate=dropout_rate, name="mc_dropout_conv")(x)

    x = layers.LSTM(units=lstm_units, return_sequences=False, name="lstm")(x)
    x = layers.BatchNormalization(name="bn_lstm")(x)
    x = MCDropout(rate=dropout_rate, name="mc_dropout_lstm")(x)

    # --- One sigmoid head per horizon ---
    outputs = {
        f"out_{h}d": layers.Dense(1, activation="sigmoid", name=f"out_{h}d")(x)
        for h in HORIZONS
    }

    model = keras.Model(inputs=inputs, outputs=outputs, name="conv_lstm_multi_horizon")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={f"out_{h}d": "binary_crossentropy" for h in HORIZONS},
        loss_weights={
            "out_1d":   1.00,   # ~252 independent samples/year
            "out_5d":   1.00,   # ~50 independent samples/year
            "out_21d":  0.50,   # ~12 independent samples/year
            "out_126d": 0.25,   # ~2 independent samples/year
        },
        metrics={f"out_{h}d": ["accuracy"] for h in HORIZONS},
    )
    return model


# ---------------------------------------------------------------------------
# Monte Carlo Inference
# ---------------------------------------------------------------------------

def mc_predict(
    model: keras.Model,
    X: np.ndarray,
    n_passes: int = 50,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Run multiple stochastic forward passes (dropout active) for all 4 heads.

    Args:
        model: Trained multi-output Keras model with MCDropout layers.
        X: Input array shape (n_samples, window_size, n_features).
        n_passes: Number of stochastic forward passes.

    Returns:
        Dict mapping each horizon key to (mean_probs, std_probs):
        {
          'out_1d':   (mean (n_samples,), std (n_samples,)),
          'out_5d':   (...),
          'out_21d':  (...),
          'out_126d': (...),
        }
    """
    horizon_keys = [f"out_{h}d" for h in HORIZONS]
    all_passes   = {k: [] for k in horizon_keys}

    for _ in range(n_passes):
        raw = model(X, training=True)   # dict of tensors, each (n_samples, 1)
        for k in horizon_keys:
            all_passes[k].append(raw[k].numpy().squeeze())

    result = {}
    for k, passes in all_passes.items():
        stacked     = np.stack(passes, axis=0)   # (n_passes, n_samples)
        result[k]   = (stacked.mean(axis=0), stacked.std(axis=0))

    return result


# ---------------------------------------------------------------------------
# Model Summary Utility
# ---------------------------------------------------------------------------

def print_model_summary(model: keras.Model) -> None:
    model.summary()
