"""
Backtesting & Strategy Evaluation

Strategies:
  1. Top-3 daily picks (buy the 3 stocks with highest predicted P(up))
  2. Buy-and-hold S&P 500
  3. Random stock baseline (Monte Carlo simulation)

Evaluation: 3-sigma test against random distribution.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats
from tensorflow import keras
from model import mc_predict
from data_engineering import build_dataset, FEATURE_COLS


# ---------------------------------------------------------------------------
# 1. Generate Per-Ticker Predictions
# ---------------------------------------------------------------------------

def generate_predictions(
    model: keras.Model,
    dataset: dict,
    buy_threshold: float = 0.7,
    mc_passes: int = 50,
) -> pd.DataFrame:
    """
    Run MC inference on the test set and return a DataFrame with columns:
      date, ticker, prob_up, uncertainty, signal (buy/hold), actual
    """
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]
    dates = dataset["dates_test"]
    tickers = dataset["tickers_test"]

    mean_probs, std_probs = mc_predict(model, X_test, n_passes=mc_passes)

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "ticker": tickers,
        "prob_up": mean_probs,
        "uncertainty": std_probs,
        "actual": y_test.astype(int),
    })
    df["signal"] = (df["prob_up"] >= buy_threshold).map({True: "buy", False: "hold"})
    df = df.sort_values(["date", "prob_up"], ascending=[True, False]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Top-3 Strategy Returns
# ---------------------------------------------------------------------------

def _daily_return_for(ticker: str, date: pd.Timestamp, price_cache: dict) -> float:
    """Look up the actual next-day return from the price cache."""
    prices = price_cache.get(ticker)
    if prices is None:
        return 0.0
    loc = prices.index.get_indexer([date], method="nearest")[0]
    if loc + 1 >= len(prices):
        return 0.0
    return float(prices.iloc[loc + 1] / prices.iloc[loc] - 1)


def compute_top3_returns(
    preds_df: pd.DataFrame,
    price_cache: dict,
) -> pd.Series:
    """
    Each day: buy equal-weight position in the top-3 stocks by prob_up.
    Return a daily return series.
    """
    daily_returns = {}
    for date, group in preds_df.groupby("date"):
        top3 = group.nlargest(3, "prob_up")
        day_returns = [
            _daily_return_for(row["ticker"], date, price_cache)
            for _, row in top3.iterrows()
        ]
        daily_returns[date] = np.mean(day_returns) if day_returns else 0.0
    return pd.Series(daily_returns).sort_index()


# ---------------------------------------------------------------------------
# 3. S&P 500 Buy-and-Hold Baseline
# ---------------------------------------------------------------------------

def compute_sp500_returns(start: str, end: str) -> pd.Series:
    """Download SPY and compute daily returns."""
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    return spy["Close"].pct_change().dropna()


# ---------------------------------------------------------------------------
# 4. Random Baseline (Monte Carlo simulation)
# ---------------------------------------------------------------------------

def compute_random_baseline(
    preds_df: pd.DataFrame,
    price_cache: dict,
    n_simulations: int = 1000,
    top_k: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate buying `top_k` random stocks each day, `n_simulations` times.
    Returns a DataFrame of shape (n_dates, n_simulations) with daily returns.
    """
    rng = np.random.default_rng(seed)
    dates = sorted(preds_df["date"].unique())
    sim_results = []

    for _ in range(n_simulations):
        sim_returns = {}
        for date in dates:
            group = preds_df[preds_df["date"] == date]
            if len(group) == 0:
                sim_returns[date] = 0.0
                continue
            sample = group.sample(n=min(top_k, len(group)), random_state=rng.integers(0, 2**31))
            day_returns = [
                _daily_return_for(row["ticker"], date, price_cache)
                for _, row in sample.iterrows()
            ]
            sim_returns[date] = np.mean(day_returns) if day_returns else 0.0
        sim_results.append(sim_returns)

    return pd.DataFrame(sim_results, columns=dates).T  # (n_dates, n_simulations)


# ---------------------------------------------------------------------------
# 5. Cumulative Return Conversion
# ---------------------------------------------------------------------------

def cumulative_return(daily_returns: pd.Series) -> pd.Series:
    """Convert daily return series to cumulative growth (starting at 1.0)."""
    return (1 + daily_returns).cumprod()


# ---------------------------------------------------------------------------
# 6. 3-Sigma Statistical Test
# ---------------------------------------------------------------------------

def three_sigma_test(
    strategy_total_return: float,
    random_returns: np.ndarray,  # total returns of each simulation
) -> dict:
    """
    Test whether strategy_total_return is significantly above the random distribution.

    Returns dict with mean, std, z_score, p_value, passes_3sigma.
    """
    mu = random_returns.mean()
    sigma = random_returns.std()
    z_score = (strategy_total_return - mu) / sigma if sigma > 0 else 0.0
    p_value = 1 - stats.norm.cdf(z_score)
    passes = z_score >= 3.0

    return {
        "strategy_return": strategy_total_return,
        "random_mean": mu,
        "random_std": sigma,
        "z_score": z_score,
        "p_value": p_value,
        "passes_3sigma": passes,
    }


# ---------------------------------------------------------------------------
# 7. Master Backtesting Runner
# ---------------------------------------------------------------------------

def run_backtest(
    model: keras.Model,
    dataset: dict,
    tickers: list[str],
    start: str,
    end: str,
    buy_threshold: float = 0.7,
    n_random_sims: int = 1000,
    mc_passes: int = 50,
    plot: bool = True,
    save_plot: str = "backtest_results.png",
) -> dict:
    """
    Full backtest pipeline. Returns a results dict with all metrics.
    """
    print("\n[Backtest] Generating predictions...")
    preds_df = generate_predictions(model, dataset, buy_threshold, mc_passes)

    # Build price cache for all tickers
    print("[Backtest] Loading price data for return calculations...")
    raw_prices = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if not df.empty:
                raw_prices[ticker] = df["Close"]
        except Exception:
            pass

    # --- Strategy Returns ---
    print("[Backtest] Computing Top-3 strategy returns...")
    top3_daily = compute_top3_returns(preds_df, raw_prices)

    # Align dates to test period
    test_start = preds_df["date"].min()
    test_end = preds_df["date"].max()

    # --- S&P 500 ---
    print("[Backtest] Computing S&P 500 baseline...")
    sp500_daily = compute_sp500_returns(str(test_start.date()), str(test_end.date()))
    sp500_daily = sp500_daily[sp500_daily.index >= test_start]

    # --- Random Baseline ---
    print(f"[Backtest] Running {n_random_sims} random simulations...")
    random_daily = compute_random_baseline(preds_df, raw_prices, n_simulations=n_random_sims)

    # --- Cumulative Returns ---
    top3_cum = cumulative_return(top3_daily)
    sp500_cum = cumulative_return(sp500_daily)
    random_cum = random_daily.apply(lambda col: (1 + col).cumprod())  # (n_dates, n_sims)

    # --- Total Returns ---
    top3_total = float(top3_cum.iloc[-1]) - 1
    sp500_total = float(sp500_cum.iloc[-1]) - 1
    random_totals = random_cum.iloc[-1].values - 1  # array of n_sims

    # --- 3-Sigma Test ---
    sigma_result = three_sigma_test(top3_total, random_totals)

    # --- Print Summary ---
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Strategy (Top-3) Total Return : {top3_total:+.2%}")
    print(f"  S&P 500 Total Return          : {sp500_total:+.2%}")
    print(f"  Random Baseline Mean Return   : {sigma_result['random_mean']:+.2%}")
    print(f"  Random Baseline Std           : {sigma_result['random_std']:.4f}")
    print(f"  Z-Score vs Random             : {sigma_result['z_score']:.2f}σ")
    print(f"  P-Value                       : {sigma_result['p_value']:.4f}")
    print(f"  Passes 3-Sigma Test           : {'YES ✓' if sigma_result['passes_3sigma'] else 'NO ✗'}")
    print("=" * 60)

    # --- Plot ---
    if plot:
        _plot_results(top3_cum, sp500_cum, random_cum, sigma_result, save_path=save_plot)

    return {
        "predictions": preds_df,
        "top3_daily": top3_daily,
        "top3_cumulative": top3_cum,
        "sp500_daily": sp500_daily,
        "sp500_cumulative": sp500_cum,
        "random_daily": random_daily,
        "random_cumulative": random_cum,
        "top3_total_return": top3_total,
        "sp500_total_return": sp500_total,
        "random_totals": random_totals,
        "sigma_test": sigma_result,
    }


# ---------------------------------------------------------------------------
# 8. Plot
# ---------------------------------------------------------------------------

def _plot_results(
    top3_cum: pd.Series,
    sp500_cum: pd.Series,
    random_cum: pd.DataFrame,
    sigma_result: dict,
    save_path: str = "backtest_results.png",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: Cumulative Returns ---
    ax = axes[0]
    # Random simulations (light grey)
    for col in random_cum.columns:
        ax.plot(random_cum.index, random_cum[col], color="grey", alpha=0.05, linewidth=0.5)
    # Random mean
    ax.plot(random_cum.index, random_cum.mean(axis=1), color="grey",
            linewidth=1.5, linestyle="--", label="Random baseline (mean)")
    # S&P 500
    sp500_aligned = sp500_cum.reindex(random_cum.index, method="ffill")
    ax.plot(sp500_aligned.index, sp500_aligned.values, color="blue",
            linewidth=2, label="S&P 500 (buy & hold)")
    # Top-3 strategy
    top3_aligned = top3_cum.reindex(random_cum.index, method="ffill")
    ax.plot(top3_aligned.index, top3_aligned.values, color="green",
            linewidth=2.5, label="Top-3 Strategy")

    ax.set_title("Cumulative Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (starting at 1.0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Right: Distribution of random total returns ---
    ax2 = axes[1]
    ax2.hist(sigma_result.get("_random_totals", random_cum.iloc[-1].values - 1),
             bins=50, color="grey", alpha=0.7, edgecolor="white", label="Random simulations")
    strategy_ret = sigma_result["strategy_return"]
    ax2.axvline(strategy_ret, color="green", linewidth=2.5, label=f"Top-3 ({strategy_ret:+.2%})")
    ax2.axvline(sigma_result["random_mean"], color="grey", linewidth=1.5,
                linestyle="--", label=f"Random mean ({sigma_result['random_mean']:+.2%})")
    # 3-sigma line
    threshold = sigma_result["random_mean"] + 3 * sigma_result["random_std"]
    ax2.axvline(threshold, color="red", linewidth=1.5, linestyle=":",
                label=f"3σ threshold ({threshold:+.2%})")
    ax2.set_title(f"Strategy vs Random Distribution\n(Z={sigma_result['z_score']:.2f}σ, "
                  f"p={sigma_result['p_value']:.4f})")
    ax2.set_xlabel("Total Return")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Backtest] Plot saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from training import run_phase3

    parser = argparse.ArgumentParser(description="LSTM Stock Predictor — Backtest")
    parser.add_argument("--tickers", nargs="+", default=[
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "JPM", "JNJ", "V",
    ])
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--model", default=None,
                        help="Path to saved .keras model. If not given, trains from scratch.")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.model:
        print(f"[Main] Loading model from {args.model}")
        from tensorflow import keras as _keras
        model = _keras.models.load_model(args.model, custom_objects={"MCDropout": __import__("model").MCDropout})
        dataset = build_dataset(args.tickers, args.start, args.end, window_size=args.window, phase=3)
    else:
        print("[Main] No model path given — running Phase 3 training first.")
        model, dataset, _ = run_phase3(
            args.tickers, args.start, args.end,
            window_size=args.window, epochs=200,
        )

    run_backtest(
        model=model,
        dataset=dataset,
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        buy_threshold=args.threshold,
        n_random_sims=args.sims,
        plot=not args.no_plot,
    )
