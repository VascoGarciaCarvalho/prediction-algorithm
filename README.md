# LSTM Stock Predictor

A deep learning pipeline that predicts daily stock price direction using a Conv1D → LSTM → Monte Carlo Dropout architecture, then evaluates the strategy against S&P 500 and a random baseline with a 3-sigma statistical test.

---

## Architecture

```
OHLCV + Insider Data + News Sentiment
              ↓
     Feature Engineering
   (SMA, EMA, returns, volume,
    time, insider, VADER NLP)
              ↓
   Sliding Windows (20 days)
    Chronological 80/20 Split
              ↓
  ┌─────────────────────────┐
  │  Conv1D (causal)        │  ← short-term pattern detection
  │  LSTM                   │  ← sequential dependencies
  │  Monte Carlo Dropout    │  ← uncertainty estimation
  │  Dense (sigmoid)        │  ← P(price goes up tomorrow)
  └─────────────────────────┘
              ↓
     Top-3 Daily Strategy
              ↓
   vs S&P 500 + Random Baseline
          (3-sigma test)
```

---

## Features

### Technical Indicators
| Feature | Description |
|---|---|
| SMA 10/20/30 | Simple moving averages |
| EMA 10/20/30 | Exponential moving averages |
| Price-to-SMA | How far price sits above/below each average |
| Daily & Log Return | Raw and log-scale price change |
| Volatility (10-day) | Rolling standard deviation of returns |
| Volume Ratio | Today's volume vs. 10-day average |
| H-L Range | Daily high-low spread normalised by close |
| Day / Week / Month | Calendar seasonality features |

### Alternative Data (free, no API key)
| Feature | Source |
|---|---|
| Insider Net Shares | `yfinance` insider transactions |
| Insider Buy Flag | 1 if insiders are net buying |
| News Sentiment | `yfinance` news + VADER NLP (runs locally) |
| News Count | Number of articles published that day |

### Target Variable
Binary classification: `1` if next day's return > 0, else `0`.

---

## Project Structure

```
prediction-algorithm/
├── main.py               # CLI orchestrator
├── data_engineering.py   # Feature pipeline, windows, train/test split
├── model.py              # Conv1D → LSTM → MC Dropout model
├── training.py           # 3-phase training workflow
├── backtesting.py        # Strategy evaluation and statistical testing
└── requirements.txt      # Dependencies
```

---

## Installation

```bash
cd prediction-algorithm
pip3 install -r requirements.txt
```

**Requirements:** Python 3.9+, no API keys needed.

---

## Usage

### Full pipeline
```bash
python3 main.py --all
```
Runs Phase 1 → Phase 2 → Phase 3 → Backtest in sequence.

### Step by step
```bash
# Phase 1: sanity check (1 epoch, target leaked into features)
python3 main.py --phase 1

# Phase 2: indirect check (50 epochs, next-day close in features)
python3 main.py --phase 2

# Phase 3: real training (200 epochs, saves model_phase3.keras)
python3 main.py --phase 3

# Backtest a saved model
python3 main.py --backtest --model model_phase3.keras
```

### Custom tickers and dates
```bash
python3 main.py --all \
  --tickers AAPL MSFT NVDA TSLA AMZN \
  --start 2018-01-01 \
  --end 2024-01-01 \
  --threshold 0.7 \
  --sims 1000
```

### All flags
| Flag | Default | Description |
|---|---|---|
| `--all` | — | Run full pipeline |
| `--phase [1\|2\|3]` | — | Run a single phase |
| `--backtest` | — | Run backtest only |
| `--tickers` | 15 large-caps | Space-separated ticker list |
| `--start` | `2015-01-01` | Training start date |
| `--end` | `2024-01-01` | Training end date |
| `--window` | `20` | Sliding window size (days) |
| `--threshold` | `0.7` | Minimum probability to trigger a buy signal |
| `--sims` | `1000` | Random baseline simulations |
| `--model` | `model_phase3.keras` | Path to save/load model |
| `--no-plot` | — | Skip chart output |

---

## 3-Phase Training Workflow

The training follows a deliberate sanity-check sequence before running real training. Each phase must pass before trusting the next.

### Phase 1 — Direct target leak (1 epoch)
The binary target (`1`/`0`) is injected into the input features. The model is given the answer. Expected accuracy: ~100%. If it fails here, the data pipeline itself is broken.

### Phase 2 — Indirect target leak (50 epochs)
The target is removed but tomorrow's closing price is included. The model must compute the direction from that price. Expected accuracy: high. If it fails here, the model architecture is too weak to learn.

### Phase 3 — Real training (200 epochs)
No future data. No shortcuts. No early stopping. The model learns from legitimate historical patterns only. The saved model is used for backtesting.

---

## Backtesting & Evaluation

### Top-3 Strategy
Each trading day, buy equal-weight positions in the 3 stocks with the highest predicted `P(up)`. Sell the next day. Repeat.

### Baselines
- **S&P 500**: Buy SPY on the first test day, hold to the end.
- **Random**: Pick 3 random stocks every day, 1,000 times. Creates a distribution of results achievable by chance.

### 3-Sigma Test
The strategy's total return is compared against the random distribution:

```
Z = (strategy_return - random_mean) / random_std
```

| Z-score | Interpretation |
|---|---|
| < 1σ | Likely random noise |
| 1–2σ | Weak signal |
| 2–3σ | Promising, needs more validation |
| ≥ 3σ | Statistically significant edge |

### Output
After backtesting, `backtest_results.png` is saved with two charts:
- **Cumulative returns** — Top-3 strategy vs S&P 500 vs the grey cloud of random simulations
- **Return distribution** — Histogram of random outcomes with the strategy's result and 3σ threshold marked

---

## Monte Carlo Dropout

Standard dropout is disabled at inference time. Here it stays **always on**. Running 50 forward passes on the same input produces a distribution of predictions:

- **Mean** → the probability forecast used for trading decisions
- **Std** → the model's uncertainty about that forecast

High uncertainty signals can be filtered out to improve precision at the cost of fewer trades.

---

## Notes

- **News sentiment** via `yfinance` only returns recent articles (~20 per ticker), so `News_Sentiment` will be 0 for most historical training rows. It is most useful for live/recent inference.
- **Insider data** depth varies by ticker and yfinance's available history.
- **No lookahead bias**: the train/test split is strictly chronological. The scaler is fit only on training data and applied to test data.
- **CPU training** on 15 tickers from 2015–2024 takes roughly 15–30 minutes for Phase 3. Reduce tickers or shorten the date range to iterate faster.
