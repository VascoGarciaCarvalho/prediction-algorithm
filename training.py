"""
3-Phase Training Workflow
Phase 1: Direct target leak check (1 epoch, should ~100% accuracy)
Phase 2: Indirect check via next-day close (50 epochs, should be very high)
Phase 3: Legitimate training (200 epochs, no future data)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from data_engineering import build_dataset, FEATURE_COLS
from model import build_model, print_model_summary


# ---------------------------------------------------------------------------
# Shared Training Helper
# ---------------------------------------------------------------------------

def _train(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int = 64,
    use_early_stopping: bool = False,
    verbose: int = 1,
) -> keras.callbacks.History:
    callbacks = []
    if use_early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=10,
                restore_best_weights=True,
            )
        )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=verbose,
    )
    return history


def _evaluate(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray, phase: int) -> None:
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[Phase {phase}] Test Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")
    if phase == 1 and acc < 0.90:
        print("[Phase 1 WARNING] Accuracy should be near 1.0 with target leaked. Check pipeline.")
    elif phase == 2 and acc < 0.70:
        print("[Phase 2 WARNING] Accuracy lower than expected. Indirect learning may have failed.")
    else:
        print(f"[Phase {phase}] Sanity check passed.")


# ---------------------------------------------------------------------------
# Phase 1: Direct Target Leak
# ---------------------------------------------------------------------------

def run_phase1(
    tickers: list[str],
    start: str,
    end: str,
    window_size: int = 20,
) -> None:
    """
    Leave Target inside input features. Train 1 epoch.
    Accuracy must be very high to prove pipeline plumbing works.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Direct Target Check (sanity — 1 epoch)")
    print("=" * 60)

    dataset = build_dataset(tickers, start, end, window_size=window_size, phase=1)
    X_train, X_test = dataset["X_train"], dataset["X_test"]
    y_train, y_test = dataset["y_train"], dataset["y_test"]

    n_features = X_train.shape[2]
    model = build_model(input_shape=(window_size, n_features))
    print_model_summary(model)

    _train(model, X_train, y_train, X_test, y_test, epochs=1)
    _evaluate(model, X_test, y_test, phase=1)
    return model


# ---------------------------------------------------------------------------
# Phase 2: Indirect Target (Next-Day Close)
# ---------------------------------------------------------------------------

def run_phase2(
    tickers: list[str],
    start: str,
    end: str,
    window_size: int = 20,
    epochs: int = 50,
) -> None:
    """
    Remove Target from features but add shifted Next_Close.
    Train 50 epochs. Model must infer direction from next-day close price.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Indirect Target Check (Next_Close, 50 epochs)")
    print("=" * 60)

    dataset = build_dataset(tickers, start, end, window_size=window_size, phase=2)
    X_train, X_test = dataset["X_train"], dataset["X_test"]
    y_train, y_test = dataset["y_train"], dataset["y_test"]

    n_features = X_train.shape[2]
    model = build_model(input_shape=(window_size, n_features))

    _train(model, X_train, y_train, X_test, y_test, epochs=epochs)
    _evaluate(model, X_test, y_test, phase=2)
    return model


# ---------------------------------------------------------------------------
# Phase 3: Real Training
# ---------------------------------------------------------------------------

def run_phase3(
    tickers: list[str],
    start: str,
    end: str,
    window_size: int = 20,
    epochs: int = 200,
    batch_size: int = 64,
    save_path: str = "model_phase3.keras",
) -> tuple[keras.Model, dict]:
    """
    Legitimate training: no future data, no callbacks, 200 epochs.
    Saves model to disk and returns (model, dataset).
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Real Training (200 epochs, clean features)")
    print("=" * 60)

    dataset = build_dataset(tickers, start, end, window_size=window_size, phase=3)
    X_train, X_test = dataset["X_train"], dataset["X_test"]
    y_train, y_test = dataset["y_train"], dataset["y_test"]

    n_features = X_train.shape[2]
    model = build_model(input_shape=(window_size, n_features))
    print_model_summary(model)

    # No early stopping per spec
    history = _train(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=epochs,
        batch_size=batch_size,
        use_early_stopping=False,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[Phase 3] Final Test Loss: {loss:.4f} | Final Test Accuracy: {acc:.4f}")

    model.save(save_path)
    print(f"[Phase 3] Model saved to {save_path}")

    return model, dataset, history


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LSTM Stock Predictor — Training")
    parser.add_argument("--tickers", nargs="+", default=[
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "JPM", "JNJ", "V",
    ])
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save", default="model_phase3.keras")
    args = parser.parse_args()

    if args.phase == 1:
        run_phase1(args.tickers, args.start, args.end, args.window)
    elif args.phase == 2:
        epochs = args.epochs or 50
        run_phase2(args.tickers, args.start, args.end, args.window, epochs)
    else:
        epochs = args.epochs or 200
        run_phase3(args.tickers, args.start, args.end, args.window, epochs, save_path=args.save)
