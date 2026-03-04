"""
Backtesting & Strategy Evaluation

Strategies:
  1. Top-3 daily picks — sector-diversified, ranked by risk-adjusted score
  2. Buy-and-hold S&P 500
  3. Random stock baseline (Monte Carlo simulation)

Risk-adjusted score per horizon h:
  score = (P(up) / h_days) - lambda_penalty * (uncertainty / h_days)

Sector diversification: at most 1 stock per sector in the Top-3.

Evaluation: 3-sigma test against random distribution.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats
from tensorflow import keras
from model import mc_predict
from data_engineering import build_dataset, HORIZONS

# Map each horizon output key to its number of trading days
HORIZON_DAYS: dict[str, int] = {f"out_{h}d": h for h in HORIZONS}

# Sector classifications for the default 15 tickers (no API needed)
SECTOR_MAP: dict[str, str] = {
    "AAPL":  "Technology",
    "MSFT":  "Technology",
    "GOOGL": "Technology",
    "NVDA":  "Technology",
    "AMZN":  "ConsumerDiscretionary",
    "TSLA":  "ConsumerDiscretionary",
    "HD":    "ConsumerDiscretionary",
    "META":  "CommunicationServices",
    "DIS":   "CommunicationServices",
    "JPM":   "Financials",
    "V":     "Financials",
    "MA":    "Financials",
    "JNJ":   "Healthcare",
    "UNH":   "Healthcare",
    "PG":    "ConsumerStaples",
}


# ---------------------------------------------------------------------------
# 1. Generate Per-Ticker Predictions with Risk-Adjusted Scores
# ---------------------------------------------------------------------------

def generate_predictions(
    model: keras.Model,
    dataset: dict,
    buy_threshold: float = 0.7,
    lambda_penalty: float = 1.0,
    mc_passes: int = 50,
) -> pd.DataFrame:
    """
    Run MC inference on the test set and compute risk-adjusted scores for all horizons.

    For each horizon h:
      score_{h} = (prob_up / h_days) - lambda_penalty * (uncertainty / h_days)

    The best horizon per row is the one with the highest score.
    Buy signal fires when: best_score > 0 AND best_prob_up >= buy_threshold.

    Returns DataFrame with columns:
      date, ticker,
      prob_up_{h}, uncertainty_{h}, score_{h}  (x4 horizons),
      best_horizon, best_prob_up, best_uncertainty, best_score,
      signal, actual_1d
    """
    X_test   = dataset["X_test"]
    y_test   = dataset["y_test"]
    dates    = dataset["dates_test"]
    tickers  = dataset["tickers_test"]

    mc_results = mc_predict(model, X_test, n_passes=mc_passes)

    rows: dict = {
        "date":   pd.to_datetime(dates),
        "ticker": tickers,
    }

    for key, h in HORIZON_DAYS.items():
        mean_p, std_p = mc_results[key]
        score  = (mean_p / h) - lambda_penalty * (std_p / h)
        label  = key.replace("out_", "")          # e.g. '1d'
        rows[f"prob_up_{label}"]     = mean_p
        rows[f"uncertainty_{label}"] = std_p
        rows[f"score_{label}"]       = score

    df = pd.DataFrame(rows)

    # Select best horizon per row
    score_cols   = [f"score_{key.replace('out_', '')}" for key in HORIZON_DAYS]
    horizon_keys = list(HORIZON_DAYS.keys())
    best_idx     = df[score_cols].values.argmax(axis=1)

    df["best_horizon"] = [horizon_keys[i] for i in best_idx]
    df["best_prob_up"] = df.apply(
        lambda r: r[f"prob_up_{r['best_horizon'].replace('out_', '')}"], axis=1
    )
    df["best_uncertainty"] = df.apply(
        lambda r: r[f"uncertainty_{r['best_horizon'].replace('out_', '')}"], axis=1
    )
    df["best_score"] = df[score_cols].max(axis=1)

    # Buy criterion: risk-adjusted score is positive AND confidence above threshold
    df["signal"] = (
        (df["best_score"] > 0) & (df["best_prob_up"] >= buy_threshold)
    ).map({True: "buy", False: "hold"})

    df["actual_1d"] = y_test["out_1d"].astype(int)

    df = df.sort_values(["date", "best_score"], ascending=[True, False]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Horizon-Aware Return Lookup
# ---------------------------------------------------------------------------

def _hday_return_for(
    ticker: str,
    date: pd.Timestamp,
    horizon_days: int,
    price_cache: dict,
) -> float:
    """
    Return the h-day return for ticker starting at date, normalized to daily-equivalent.

    normalized_return = (price[loc + h] / price[loc] - 1) / h_days

    This makes 1-day and 126-day positions directly comparable in daily return terms.
    """
    prices = price_cache.get(ticker)
    if prices is None:
        return 0.0
    loc = prices.index.get_indexer([date], method="nearest")[0]
    target_loc = loc + horizon_days
    if target_loc >= len(prices):
        return 0.0
    raw_return = float(prices.iloc[target_loc] / prices.iloc[loc] - 1)
    return raw_return / horizon_days


# ---------------------------------------------------------------------------
# 3. Sector-Diversified Top-3 Strategy
# ---------------------------------------------------------------------------

def compute_top3_returns(
    preds_df: pd.DataFrame,
    price_cache: dict,
    sector_map: dict[str, str] = None,
) -> tuple:
    """
    Each day: select up to 3 stocks with sector diversification.

    Algorithm:
      1. Filter to buy-signal rows.
      2. Per sector: keep only the stock with the highest best_score.
      3. Pick the top-3 sectors by their candidate's score.
      4. Fallback: if < 3 sectors qualify, fill slots from any remaining buy signals.
      5. Compute mean normalized daily-equivalent return across selected stocks.

    Returns:
      (pd.Series of daily returns, pd.DataFrame of selections log)
    """
    if sector_map is None:
        sector_map = SECTOR_MAP

    daily_returns = {}
    selections = []

    for date, group in preds_df.groupby("date"):
        group = group.copy()
        group["sector"] = group["ticker"].map(lambda t: sector_map.get(t, t))

        buy_candidates = group[group["signal"] == "buy"].copy()
        selected = []

        if not buy_candidates.empty:
            # Best stock per sector
            sector_best = (
                buy_candidates
                .sort_values("best_score", ascending=False)
                .groupby("sector")
                .first()
                .reset_index()
            )
            # Top-3 sectors
            top_sectors = sector_best.nlargest(3, "best_score")
            selected    = top_sectors.to_dict("records")

            # Fill remaining slots from any-sector candidates
            if len(selected) < 3:
                selected_tickers = {r["ticker"] for r in selected}
                remaining = buy_candidates[
                    ~buy_candidates["ticker"].isin(selected_tickers)
                ].sort_values("best_score", ascending=False)
                for _, row in remaining.iterrows():
                    if len(selected) >= 3:
                        break
                    selected.append(row.to_dict())

        if not selected:
            daily_returns[date] = 0.0
            continue

        day_ret = []
        for rec in selected:
            h_key  = rec.get("best_horizon", "out_1d")
            h_days = HORIZON_DAYS.get(h_key, 1)
            ret    = _hday_return_for(rec["ticker"], date, h_days, price_cache)
            day_ret.append(ret)
            selections.append({
                "date":         date,
                "ticker":       rec["ticker"],
                "sector":       rec.get("sector", sector_map.get(rec["ticker"], rec["ticker"])),
                "best_horizon": rec.get("best_horizon", "out_1d"),
                "best_score":   rec.get("best_score", 0.0),
                "best_prob_up": rec.get("best_prob_up", 0.0),
            })

        daily_returns[date] = np.mean(day_ret) if day_ret else 0.0

    return pd.Series(daily_returns).sort_index(), pd.DataFrame(selections)


# ---------------------------------------------------------------------------
# 4. S&P 500 Buy-and-Hold Baseline
# ---------------------------------------------------------------------------

def compute_sp500_returns(start: str, end: str) -> pd.Series:
    """Download SPY and compute daily returns."""
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    return spy["Close"].pct_change().dropna()


# ---------------------------------------------------------------------------
# 5. Random Baseline (Monte Carlo simulation)
# ---------------------------------------------------------------------------

def compute_random_baseline(
    preds_df: pd.DataFrame,
    price_cache: dict,
    n_simulations: int = 1000,
    top_k: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate buying top_k random stocks each day, n_simulations times.
    Uses 1-day returns (normalized) for fair comparison with random chance.
    Returns DataFrame of shape (n_dates, n_simulations).
    """
    rng   = np.random.default_rng(seed)
    dates = sorted(preds_df["date"].unique())
    sims  = []

    for _ in range(n_simulations):
        sim_returns = {}
        for date in dates:
            group = preds_df[preds_df["date"] == date]
            if group.empty:
                sim_returns[date] = 0.0
                continue
            sample = group.sample(
                n=min(top_k, len(group)),
                random_state=int(rng.integers(0, 2**31))
            )
            day_ret = [
                _hday_return_for(row["ticker"], date, 1, price_cache)
                for _, row in sample.iterrows()
            ]
            sim_returns[date] = np.mean(day_ret) if day_ret else 0.0
        sims.append(sim_returns)

    return pd.DataFrame(sims, columns=dates).T   # (n_dates, n_simulations)


# ---------------------------------------------------------------------------
# 6. Cumulative Return Conversion
# ---------------------------------------------------------------------------

def cumulative_return(daily_returns: pd.Series) -> pd.Series:
    return (1 + daily_returns).cumprod()


# ---------------------------------------------------------------------------
# 7. 3-Sigma Statistical Test
# ---------------------------------------------------------------------------

def three_sigma_test(
    strategy_total_return: float,
    random_returns: np.ndarray,
) -> dict:
    """
    Z-score of strategy vs. random distribution.
    passes_3sigma is True if z_score >= 3.0 (99.7% confidence).
    """
    mu      = random_returns.mean()
    sigma   = random_returns.std()
    z_score = (strategy_total_return - mu) / sigma if sigma > 0 else 0.0
    p_value = 1 - stats.norm.cdf(z_score)

    return {
        "strategy_return": strategy_total_return,
        "random_mean":     mu,
        "random_std":      sigma,
        "z_score":         z_score,
        "p_value":         p_value,
        "passes_3sigma":   z_score >= 3.0,
    }


# ---------------------------------------------------------------------------
# 8. Master Backtesting Runner
# ---------------------------------------------------------------------------

def run_backtest(
    model: keras.Model,
    dataset: dict,
    tickers: list[str],
    start: str,
    end: str,
    buy_threshold: float = 0.7,
    lambda_penalty: float = 1.0,
    sector_map: dict[str, str] = None,
    n_random_sims: int = 1000,
    mc_passes: int = 50,
    plot: bool = True,
    save_plot: str = "backtest_results.png",
    save_decision_log: str = "decision_log.png",
) -> dict:
    """Full backtest pipeline. Returns a results dict with all metrics."""
    if sector_map is None:
        sector_map = SECTOR_MAP

    print("\n[Backtest] Generating predictions...")
    preds_df = generate_predictions(model, dataset, buy_threshold, lambda_penalty, mc_passes)

    # Horizon distribution summary
    horizon_counts = preds_df[preds_df["signal"] == "buy"]["best_horizon"].value_counts()
    print("[Backtest] Buy signal horizon distribution:")
    for h, count in horizon_counts.items():
        print(f"  {h}: {count} signals")

    print("\n[Backtest] Loading price data for return calculations...")
    price_cache = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                price_cache[ticker] = df["Close"]
        except Exception:
            pass

    test_start = preds_df["date"].min()
    test_end   = preds_df["date"].max()

    print("[Backtest] Computing Top-3 (sector-diversified) strategy returns...")
    top3_daily, selections_df = compute_top3_returns(preds_df, price_cache, sector_map)

    print("[Backtest] Computing S&P 500 baseline...")
    sp500_daily = compute_sp500_returns(str(test_start.date()), str(test_end.date()))
    sp500_daily = sp500_daily[sp500_daily.index >= test_start]

    print(f"[Backtest] Running {n_random_sims} random simulations...")
    random_daily = compute_random_baseline(preds_df, price_cache, n_simulations=n_random_sims)

    # Cumulative returns
    top3_cum   = cumulative_return(top3_daily)
    sp500_cum  = cumulative_return(sp500_daily)
    random_cum = random_daily.apply(lambda col: (1 + col).cumprod())

    # Total returns
    top3_total    = float(top3_cum.iloc[-1]) - 1
    sp500_total   = float(sp500_cum.iloc[-1]) - 1
    random_totals = random_cum.iloc[-1].values - 1

    sigma_result = three_sigma_test(top3_total, random_totals)
    sigma_result["_random_totals"] = random_totals   # stored for plotting

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

    if plot:
        _plot_results(top3_cum, sp500_cum, random_cum, sigma_result, save_path=save_plot)
        _plot_decision_log(selections_df, tickers, sector_map, save_path=save_decision_log)

    return {
        "predictions":       preds_df,
        "selections":        selections_df,
        "top3_daily":        top3_daily,
        "top3_cumulative":   top3_cum,
        "sp500_daily":       sp500_daily,
        "sp500_cumulative":  sp500_cum,
        "random_daily":      random_daily,
        "random_cumulative": random_cum,
        "top3_total_return": top3_total,
        "sp500_total_return": sp500_total,
        "random_totals":     random_totals,
        "sigma_test":        sigma_result,
    }


# ---------------------------------------------------------------------------
# 9. Plot
# ---------------------------------------------------------------------------

def _plot_results(
    top3_cum: pd.Series,
    sp500_cum: pd.Series,
    random_cum: pd.DataFrame,
    sigma_result: dict,
    save_path: str = "backtest_results.png",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for col in random_cum.columns:
        ax.plot(random_cum.index, random_cum[col], color="grey", alpha=0.05, linewidth=0.5)
    ax.plot(random_cum.index, random_cum.mean(axis=1), color="grey",
            linewidth=1.5, linestyle="--", label="Random baseline (mean)")
    sp500_aligned = sp500_cum.reindex(random_cum.index, method="ffill")
    ax.plot(sp500_aligned.index, sp500_aligned.values, color="blue",
            linewidth=2, label="S&P 500 (buy & hold)")
    top3_aligned = top3_cum.reindex(random_cum.index, method="ffill")
    ax.plot(top3_aligned.index, top3_aligned.values, color="green",
            linewidth=2.5, label="Top-3 Strategy (sector-diversified)")
    ax.set_title("Cumulative Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (starting at 1.0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    random_totals = sigma_result.get("_random_totals", random_cum.iloc[-1].values - 1)
    ax2.hist(random_totals, bins=50, color="grey", alpha=0.7, edgecolor="white",
             label="Random simulations")
    strategy_ret = sigma_result["strategy_return"]
    ax2.axvline(strategy_ret, color="green", linewidth=2.5,
                label=f"Top-3 ({strategy_ret:+.2%})")
    ax2.axvline(sigma_result["random_mean"], color="grey", linewidth=1.5, linestyle="--",
                label=f"Random mean ({sigma_result['random_mean']:+.2%})")
    threshold = sigma_result["random_mean"] + 3 * sigma_result["random_std"]
    ax2.axvline(threshold, color="red", linewidth=1.5, linestyle=":",
                label=f"3σ threshold ({threshold:+.2%})")
    ax2.set_title(
        f"Strategy vs Random Distribution\n"
        f"(Z={sigma_result['z_score']:.2f}σ, p={sigma_result['p_value']:.4f})"
    )
    ax2.set_xlabel("Total Return")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Backtest] Plot saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 10. Decision Log Chart
# ---------------------------------------------------------------------------

def _plot_decision_log(
    selections_df: pd.DataFrame,
    all_tickers: list[str],
    sector_map: dict[str, str],
    save_path: str = "decision_log.png",
) -> None:
    """3-panel figure showing which stocks were picked, horizons used, and sector allocation."""
    import matplotlib.colors as mcolors

    if selections_df.empty:
        print("[Backtest] No selections to plot in decision log.")
        return

    # Horizon encoding
    horizon_order = ["out_1d", "out_5d", "out_21d", "out_126d"]
    horizon_code  = {h: i + 1 for i, h in enumerate(horizon_order)}
    horizon_label = {"out_1d": "1d", "out_5d": "5d", "out_21d": "21d", "out_126d": "126d"}

    # Sort tickers by sector for heatmap columns
    tickers_sorted = sorted(all_tickers, key=lambda t: (sector_map.get(t, t), t))

    # Unique dates
    all_dates = sorted(selections_df["date"].unique())
    n_dates   = len(all_dates)
    date_idx  = {d: i for i, d in enumerate(all_dates)}

    # Build heatmap matrix
    matrix = np.zeros((n_dates, len(tickers_sorted)), dtype=float)
    for _, row in selections_df.iterrows():
        r = date_idx.get(row["date"])
        c = tickers_sorted.index(row["ticker"]) if row["ticker"] in tickers_sorted else None
        if r is not None and c is not None:
            matrix[r, c] = horizon_code.get(row["best_horizon"], 1)

    fig, axes = plt.subplots(3, 1, figsize=(18, 14))

    # --- Panel 1: Heatmap ---
    ax1 = axes[0]
    cmap = mcolors.ListedColormap(["white", "#084c61", "#4c9be8", "#a8d5f5", "#f5c842"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)

    # Sample y-axis labels for readability
    y_labels = [""] * n_dates
    step = max(1, n_dates // 30)
    for i in range(0, n_dates, step):
        y_labels[i] = str(all_dates[i].date()) if hasattr(all_dates[i], "date") else str(all_dates[i])

    im = ax1.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax1.set_xticks(range(len(tickers_sorted)))
    ax1.set_xticklabels(tickers_sorted, rotation=45, ha="right", fontsize=8)
    ax1.set_yticks(range(n_dates))
    ax1.set_yticklabels(y_labels, fontsize=7)
    ax1.set_title("Daily Stock Selections by Horizon")

    # Vertical dashed lines between sectors
    prev_sector = None
    for c_idx, ticker in enumerate(tickers_sorted):
        sec = sector_map.get(ticker, ticker)
        if prev_sector is not None and sec != prev_sector:
            ax1.axvline(x=c_idx - 0.5, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
        prev_sector = sec

    cbar = fig.colorbar(im, ax=ax1, orientation="vertical", pad=0.01)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(["not picked", "1d", "5d", "21d", "126d"])

    # --- Panel 2: Horizon usage over time (weekly stacked bar) ---
    ax2 = axes[1]
    horizon_colors = {"out_1d": "#084c61", "out_5d": "#4c9be8", "out_21d": "#a8d5f5", "out_126d": "#f5c842"}

    sel = selections_df.copy()
    sel["date"] = pd.to_datetime(sel["date"])
    for h in horizon_order:
        sel[h] = (sel["best_horizon"] == h).astype(int)

    weekly_h = sel.set_index("date")[horizon_order].resample("W").sum()

    bottom = np.zeros(len(weekly_h))
    for h in horizon_order:
        ax2.bar(weekly_h.index, weekly_h[h], bottom=bottom,
                label=horizon_label[h], color=horizon_colors[h], width=5)
        bottom += weekly_h[h].values

    ax2.set_title("Horizon Usage Over Time")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Pick Count")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Sector allocation over time (weekly stacked bar) ---
    ax3 = axes[2]
    sectors_all = sorted(selections_df["sector"].unique())
    sector_colors = plt.cm.tab10(np.linspace(0, 1, len(sectors_all)))

    for col in sectors_all:
        sel[col] = (sel["sector"] == col).astype(int)

    weekly_s = sel.set_index("date")[sectors_all].resample("W").sum()

    bottom = np.zeros(len(weekly_s))
    for i, sec in enumerate(sectors_all):
        ax3.bar(weekly_s.index, weekly_s[sec], bottom=bottom,
                label=sec, color=sector_colors[i], width=5)
        bottom += weekly_s[sec].values

    ax3.set_title("Sector Allocation Over Time")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Pick Count")
    ax3.legend(loc="upper right", fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Backtest] Decision log saved to {save_path}")
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
        "UNH", "HD", "PG", "MA", "DIS",
    ])
    parser.add_argument("--start",          default="2015-01-01")
    parser.add_argument("--end",            default="2024-01-01")
    parser.add_argument("--window",         type=int,   default=20)
    parser.add_argument("--threshold",      type=float, default=0.7)
    parser.add_argument("--lambda-penalty", type=float, default=1.0)
    parser.add_argument("--sims",           type=int,   default=1000)
    parser.add_argument("--model",          default=None)
    parser.add_argument("--no-plot",        action="store_true")
    args = parser.parse_args()

    if args.model:
        import tensorflow as _tf
        from model import MCDropout as _MCDropout
        model = _tf.keras.models.load_model(
            args.model, custom_objects={"MCDropout": _MCDropout}
        )
        dataset = build_dataset(
            args.tickers, args.start, args.end, window_size=args.window, phase=3
        )
    else:
        model, dataset, _ = run_phase3(
            args.tickers, args.start, args.end, window_size=args.window, epochs=200,
        )

    run_backtest(
        model=model,
        dataset=dataset,
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        buy_threshold=args.threshold,
        lambda_penalty=args.lambda_penalty,
        n_random_sims=args.sims,
        plot=not args.no_plot,
    )
