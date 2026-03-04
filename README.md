# LSTM Stock Predictor

A deep learning pipeline that predicts stock price direction across multiple time horizons and evaluates whether those predictions can generate a statistically meaningful trading edge.

---

## What It Does

The system downloads historical market data for a set of stocks, engineers a rich set of features from price action and alternative data sources, and trains a neural network to predict whether each stock will be higher after 1 day, 1 week, 1 month, or 6 months.

Once trained, the model drives a simple trading strategy — buying the most confident, sector-diversified picks each day — and measures whether that strategy meaningfully outperforms random chance and the S&P 500.

---

## How It Works

**Data & Features**
Price data is downloaded from Yahoo Finance. On top of raw OHLCV data, the pipeline computes technical indicators (moving averages, volatility, volume trends), calendar features, insider transaction activity, and news sentiment scored locally using NLP. No paid API is required.

**Model**
A Conv1D layer first extracts short-term patterns from the 20-day input window, then an LSTM captures longer sequential dependencies. The model outputs four independent predictions — one per time horizon — each as a probability that the price will be higher by that date.

Monte Carlo Dropout keeps dropout active at inference time, running multiple forward passes to produce both a prediction and an uncertainty estimate for each output. Uncertain signals are penalised in the trading strategy.

**Training**
Training follows a three-phase sanity-check workflow before committing to real training.

- **Phase 1 — 1 epoch, answer leaked into input.** The correct labels are injected directly as input features. The model is given the answer. If it can't reach near-perfect accuracy in a single epoch, the data pipeline is broken and there's no point going further.

- **Phase 2 — 50 epochs, indirect hint.** The labels are removed but the future closing prices are included instead. The model must infer direction from those prices. This confirms the architecture is capable of learning a meaningful signal before any real training begins.

- **Phase 3 — 200 epochs, clean features only.** No future data of any kind. This is the real training run, learning purely from historical patterns. The model is saved to disk and used for backtesting.

**Backtesting**
The saved model is applied to the held-out test period. Each day, the strategy selects up to three stocks with the highest risk-adjusted confidence scores, constrained to one pick per market sector. Returns are benchmarked against a buy-and-hold S&P 500 position and 1,000 random stock-picking simulations. A 3-sigma statistical test determines whether any outperformance is meaningful or just noise.

---

## Project Structure

```
prediction-algorithm/
├── main.py               # CLI entry point
├── data_engineering.py   # Data download, feature engineering, windowing
├── model.py              # Neural network architecture and inference
├── training.py           # 3-phase training workflow
├── backtesting.py        # Strategy evaluation and statistical testing
└── requirements.txt      # Dependencies
```

---

## Installation

```bash
pip3 install -r requirements.txt
```

Python 3.9+ required. No API keys needed.

---

## Usage

```bash
# Run the full pipeline end-to-end
python3 main.py --all

# Or run individual steps
python3 main.py --phase 1
python3 main.py --phase 2
python3 main.py --phase 3
python3 main.py --backtest --model model_phase3_multihorizon.keras

# Customise tickers and date range
python3 main.py --all --tickers AAPL MSFT NVDA --start 2018-01-01 --end 2024-01-01
```

Run `python3 main.py --help` for all available flags.
