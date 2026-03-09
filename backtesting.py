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
        score  = mean_p - lambda_penalty * std_p
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
                "pick_return":  ret,
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
# 6a. Risk / Performance Metrics
# ---------------------------------------------------------------------------

def compute_risk_metrics(daily_returns: pd.Series) -> dict:
    """
    Compute standard risk/performance metrics from a daily returns Series.

    Returns:
      total_return, annualised_return, sharpe_ratio, max_drawdown,
      calmar_ratio, win_rate, profit_factor, n_trading_days
    """
    dr = daily_returns.dropna()
    if len(dr) == 0:
        return {}

    total_return      = float((1 + dr).prod() - 1)
    n_days            = len(dr)
    annualised_return = float((1 + total_return) ** (252 / n_days) - 1)

    mean_daily = float(dr.mean())
    std_daily  = float(dr.std())
    sharpe     = float((mean_daily / std_daily) * np.sqrt(252)) if std_daily > 0 else 0.0

    cum         = (1 + dr).cumprod()
    drawdowns   = cum / cum.cummax() - 1
    max_drawdown = float(drawdowns.min())
    calmar      = float(annualised_return / abs(max_drawdown)) if max_drawdown < 0 else float("nan")

    win_rate     = float((dr > 0).mean())
    gains        = dr[dr > 0].sum()
    losses       = dr[dr < 0].sum()
    profit_factor = float(gains / abs(losses)) if losses < 0 else float("nan")

    return {
        "total_return":      total_return,
        "annualised_return": annualised_return,
        "sharpe_ratio":      sharpe,
        "max_drawdown":      max_drawdown,
        "calmar_ratio":      calmar,
        "win_rate":          win_rate,
        "profit_factor":     profit_factor,
        "n_trading_days":    n_days,
    }


# ---------------------------------------------------------------------------
# 6b. Per-Ticker Performance Breakdown
# ---------------------------------------------------------------------------

def compute_ticker_breakdown(
    selections_df: pd.DataFrame,
    price_cache: dict,
) -> pd.DataFrame:
    """
    Compute per-ticker performance stats from the selections log.

    Returns DataFrame: ticker | times_picked | win_rate | mean_return | contribution_pct
    """
    if selections_df.empty:
        return pd.DataFrame()

    rows = []
    for ticker, grp in selections_df.groupby("ticker"):
        returns = []
        for _, row in grp.iterrows():
            h_key  = row.get("best_horizon", "out_1d")
            h_days = HORIZON_DAYS.get(h_key, 1)
            ret    = _hday_return_for(ticker, row["date"], h_days, price_cache)
            returns.append(ret)
        returns = np.array(returns)
        rows.append({
            "ticker":       ticker,
            "times_picked": len(grp),
            "win_rate":     float((returns > 0).mean()) if len(returns) else 0.0,
            "mean_return":  float(returns.mean())       if len(returns) else 0.0,
            "total_contrib": float(returns.sum()),
        })

    df = pd.DataFrame(rows).sort_values("total_contrib", ascending=False).reset_index(drop=True)
    total_abs = df["total_contrib"].abs().sum()
    df["contribution_pct"] = df["total_contrib"] / total_abs * 100 if total_abs > 0 else 0.0
    return df


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
    export: bool = True,
    save_plot: str = "backtest_results.html",
    save_decision_log: str = "decision_log.html",
) -> dict:
    """Full backtest pipeline. Returns a results dict with all metrics."""
    import json

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

    # Risk metrics
    risk = compute_risk_metrics(top3_daily)
    sp500_risk = compute_risk_metrics(sp500_daily)

    # Per-ticker breakdown
    ticker_breakdown = compute_ticker_breakdown(selections_df, price_cache)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Strategy (Top-3) Total Return : {top3_total:+.2%}")
    print(f"  S&P 500 Total Return          : {sp500_total:+.2%}")
    print(f"  Random Baseline Mean Return   : {sigma_result['random_mean']:+.2%}")
    print(f"  Z-Score vs Random             : {sigma_result['z_score']:.2f}σ")
    print(f"  P-Value                       : {sigma_result['p_value']:.4f}")
    print(f"  Passes 3-Sigma Test           : {'YES ✓' if sigma_result['passes_3sigma'] else 'NO ✗'}")
    print("-" * 60)
    print("RISK METRICS (Strategy vs S&P 500)")
    print(f"  Annualised Return  : {risk.get('annualised_return', 0):+.2%}  vs  {sp500_risk.get('annualised_return', 0):+.2%}")
    print(f"  Sharpe Ratio       : {risk.get('sharpe_ratio', 0):.3f}  vs  {sp500_risk.get('sharpe_ratio', 0):.3f}")
    print(f"  Max Drawdown       : {risk.get('max_drawdown', 0):+.2%}  vs  {sp500_risk.get('max_drawdown', 0):+.2%}")
    print(f"  Calmar Ratio       : {risk.get('calmar_ratio', float('nan')):.3f}")
    print(f"  Win Rate           : {risk.get('win_rate', 0):.1%}")
    print(f"  Profit Factor      : {risk.get('profit_factor', float('nan')):.3f}")
    print("-" * 60)
    if not ticker_breakdown.empty:
        print("PER-TICKER BREAKDOWN (top contributors)")
        print(f"  {'Ticker':<8} {'Picked':>6} {'WinRate':>8} {'MeanRet':>9} {'Contrib%':>9}")
        for _, row in ticker_breakdown.head(10).iterrows():
            print(f"  {row['ticker']:<8} {int(row['times_picked']):>6} "
                  f"{row['win_rate']:>8.1%} {row['mean_return']:>+9.4%} "
                  f"{row['contribution_pct']:>+8.1f}%")
    print("=" * 60)

    if plot:
        _plot_results(top3_cum, sp500_cum, random_cum, sigma_result, risk_metrics=risk, save_path=save_plot)
        _plot_decision_log(selections_df, tickers, sector_map, save_path=save_decision_log)

    if export:
        # selections CSV
        sel_out = selections_df.copy()
        sel_out["date"] = sel_out["date"].astype(str)
        sel_out.to_csv("backtest_selections.csv", index=False)
        print("[Backtest] Selections saved to backtest_selections.csv")

        # daily returns CSV
        returns_df = pd.DataFrame({
            "date": top3_daily.index.astype(str),
            "strategy_daily_return": top3_daily.values,
        })
        sp500_aligned = sp500_daily.reindex(top3_daily.index, method="nearest")
        returns_df["sp500_daily_return"] = sp500_aligned.values
        returns_df.to_csv("backtest_daily_returns.csv", index=False)
        print("[Backtest] Daily returns saved to backtest_daily_returns.csv")

        # summary JSON
        def _safe(v):
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return None
            if isinstance(v, (np.integer, np.floating)):
                return float(v)
            return v

        summary = {
            "total_return":      _safe(top3_total),
            "sp500_total_return": _safe(sp500_total),
            "sigma_test": {k: _safe(v) for k, v in sigma_result.items() if k != "_random_totals"},
            "risk_metrics": {k: _safe(v) for k, v in risk.items()},
            "sp500_risk_metrics": {k: _safe(v) for k, v in sp500_risk.items()},
            "ticker_breakdown": ticker_breakdown.to_dict("records") if not ticker_breakdown.empty else [],
        }
        with open("backtest_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("[Backtest] Summary saved to backtest_summary.json")

    return {
        "predictions":        preds_df,
        "selections":         selections_df,
        "top3_daily":         top3_daily,
        "top3_cumulative":    top3_cum,
        "sp500_daily":        sp500_daily,
        "sp500_cumulative":   sp500_cum,
        "random_daily":       random_daily,
        "random_cumulative":  random_cum,
        "top3_total_return":  top3_total,
        "sp500_total_return": sp500_total,
        "random_totals":      random_totals,
        "sigma_test":         sigma_result,
        "risk_metrics":       risk,
        "sp500_risk_metrics": sp500_risk,
        "ticker_breakdown":   ticker_breakdown,
    }


# ---------------------------------------------------------------------------
# 9. Plot
# ---------------------------------------------------------------------------

def _plot_results(
    top3_cum: pd.Series,
    sp500_cum: pd.Series,
    random_cum: pd.DataFrame,
    sigma_result: dict,
    risk_metrics: "dict | None" = None,
    save_path: str = "backtest_results.html",
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import webbrowser, os

    idx           = random_cum.index
    sp500_aligned = sp500_cum.reindex(idx, method="ffill")
    top3_aligned  = top3_cum.reindex(idx, method="ffill")
    rand_mean     = random_cum.mean(axis=1)

    def _dd(s): return (s - s.cummax()) / s.cummax()

    dd_top3  = _dd(top3_aligned)
    dd_sp500 = _dd(sp500_aligned)

    rm           = risk_metrics or {}
    z            = sigma_result["z_score"]
    pv           = sigma_result["p_value"]
    strategy_ret = sigma_result["strategy_return"]
    rand_mu      = sigma_result["random_mean"]
    rand_sd      = sigma_result["random_std"]
    threshold    = rand_mu + 3 * rand_sd
    random_totals = sigma_result.get("_random_totals", random_cum.iloc[-1].values - 1)

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=(
            "Cumulative Returns",
            "Drawdown",
            f"3σ Test  —  Z = {z:.2f}  p = {pv:.4f}",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # ── Panel 1: Cumulative returns ──────────────────────────────────────────
    # Random sims (thin, low opacity)
    for col in random_cum.columns[:200]:   # cap at 200 traces for perf
        fig.add_trace(go.Scatter(
            x=idx, y=random_cum[col],
            mode="lines", line=dict(color="rgba(160,160,160,0.06)", width=0.5),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=idx, y=rand_mean,
        mode="lines", name="Random baseline (mean)",
        line=dict(color="#aaaaaa", width=1.5, dash="dash"),
        hovertemplate="%{x|%b %d %Y}<br>%{y:.3f}<extra>Random mean</extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=idx, y=sp500_aligned,
        mode="lines", name="S&P 500 (buy & hold)",
        line=dict(color="#4c9be8", width=2),
        hovertemplate="%{x|%b %d %Y}<br>%{y:.3f}<extra>S&P 500</extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=idx, y=top3_aligned,
        mode="lines", name="Top-3 Strategy",
        line=dict(color="#4cdd80", width=2.5),
        hovertemplate="%{x|%b %d %Y}<br>%{y:.3f}<extra>Top-3 Strategy</extra>",
    ), row=1, col=1)

    # Fill between strategy and S&P 500
    fig.add_trace(go.Scatter(
        x=list(idx) + list(idx[::-1]),
        y=list(top3_aligned) + list(sp500_aligned[::-1]),
        fill="toself",
        fillcolor="rgba(76,221,128,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    # Metrics annotation
    sharpe_str = f"{rm.get('sharpe_ratio', float('nan')):.2f}" if rm else "—"
    maxdd_str  = f"{rm.get('max_drawdown', float('nan')):.1%}"  if rm else "—"
    fig.add_annotation(
        xref="x domain", yref="y domain", x=0.01, y=0.97,
        text=f"<b>Sharpe: {sharpe_str}   Max DD: {maxdd_str}</b>",
        showarrow=False, font=dict(size=11, color="#cccccc"),
        bgcolor="rgba(30,30,30,0.75)", bordercolor="#555", borderwidth=1,
        row=1, col=1,
    )

    # ── Panel 2: Drawdown ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=idx, y=dd_sp500 * 100,
        mode="lines", name="S&P 500 DD",
        line=dict(color="#4c9be8", width=1.5),
        fill="tozeroy", fillcolor="rgba(76,155,232,0.1)",
        hovertemplate="%{x|%b %d %Y}<br>%{y:.2f}%<extra>S&P 500</extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=idx, y=dd_top3 * 100,
        mode="lines", name="Strategy DD",
        line=dict(color="#4cdd80", width=1.5),
        fill="tozeroy", fillcolor="rgba(76,221,128,0.1)",
        hovertemplate="%{x|%b %d %Y}<br>%{y:.2f}%<extra>Strategy</extra>",
    ), row=2, col=1)

    # ── Panel 3: Z-Score distribution ───────────────────────────────────────
    fig.add_trace(go.Histogram(
        x=random_totals * 100,
        nbinsx=50,
        name="Random simulations",
        marker_color="#4c9be8",
        opacity=0.6,
        hovertemplate="Return: %{x:.1f}%<br>Count: %{y}<extra></extra>",
    ), row=2, col=2)

    for val, color, label in [
        (strategy_ret * 100, "#4cdd80", f"Strategy ({strategy_ret:+.2%})"),
        (rand_mu * 100,      "#aaaaaa", f"Random mean ({rand_mu:+.2%})"),
        (threshold * 100,    "#e87c4c", f"3σ threshold ({threshold:+.2%})"),
    ]:
        fig.add_vline(x=val, line_color=color, line_width=2,
                      annotation_text=label, annotation_font_color=color,
                      annotation_position="top right",
                      row=2, col=2)

    # ── Layout ───────────────────────────────────────────────────────────────
    final_top3  = top3_aligned.iloc[-1]
    final_sp500 = sp500_aligned.iloc[-1]
    passes = "✓ PASSES" if sigma_result.get("passes_3sigma") else "✗ FAILS"

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"Backtest Results — Strategy {final_top3-1:+.1%} vs S&P 500 {final_sp500-1:+.1%} — 3σ Test: {passes}",
            font=dict(size=16),
        ),
        height=750,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, system-ui, sans-serif"),
        paper_bgcolor="#111",
        plot_bgcolor="#111",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)", zeroline=False)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    fig.update_yaxes(title_text="Portfolio Value (1.0 = start)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)",  row=2, col=1)
    fig.update_yaxes(title_text="Frequency",     row=2, col=2)
    fig.update_xaxes(title_text="Total Return (%)", row=2, col=2)

    save_path = save_path.replace(".png", ".html")
    fig.write_html(save_path, include_plotlyjs="cdn")
    print(f"[Backtest] Interactive plot saved to {save_path}")
    webbrowser.open(f"file://{os.path.abspath(save_path)}")


# ---------------------------------------------------------------------------
# 10. Decision Log Chart
# ---------------------------------------------------------------------------

def _plot_decision_log(
    selections_df: pd.DataFrame,
    all_tickers: list[str],
    sector_map: dict[str, str],
    save_path: str = "decision_log.html",
) -> None:
    """3-panel interactive chart: stock heatmap, horizon usage, sector allocation."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import webbrowser, os

    if selections_df.empty:
        print("[Backtest] No selections to plot in decision log.")
        return

    horizon_order  = [f"out_{h}d" for h in HORIZONS]
    horizon_label  = {f"out_{h}d": f"{h}d" for h in HORIZONS}
    horizon_palette = ["#084c61", "#4c9be8", "#a8d5f5", "#f5c842", "#e87c4c"]
    horizon_colors  = {h: horizon_palette[i] for i, h in enumerate(horizon_order)}

    tickers_sorted = sorted(all_tickers, key=lambda t: (sector_map.get(t, t), t))

    sel = selections_df.copy()
    sel["date"] = pd.to_datetime(sel["date"])

    # ── Panel 1: Heatmap (date × ticker, coloured by horizon) ───────────────
    all_dates = sorted(sel["date"].unique())
    date_strs = [str(pd.Timestamp(d).date()) for d in all_dates]
    date_idx  = {d: i for i, d in enumerate(all_dates)}
    ticker_idx = {t: i for i, t in enumerate(tickers_sorted)}

    n_h = len(horizon_order)
    # 0 = not picked; 1…n_h = horizon index+1
    z_matrix    = np.zeros((len(all_dates), len(tickers_sorted)), dtype=int)
    hover_matrix = [[""] * len(tickers_sorted) for _ in range(len(all_dates))]

    for _, row in sel.iterrows():
        r = date_idx.get(row["date"])
        c = ticker_idx.get(row["ticker"])
        if r is not None and c is not None:
            code = horizon_order.index(row["best_horizon"]) + 1 if row["best_horizon"] in horizon_order else 1
            z_matrix[r, c] = code
            ret_str = f"  ret={row['pick_return']:+.2%}" if "pick_return" in row and pd.notna(row.get("pick_return")) else ""
            hover_matrix[r][c] = (
                f"<b>{row['ticker']}</b><br>"
                f"Date: {str(pd.Timestamp(row['date']).date())}<br>"
                f"Horizon: {horizon_label.get(row['best_horizon'], row['best_horizon'])}<br>"
                f"P(up): {row['best_prob_up']:.1%}<br>"
                f"Score: {row['best_score']:.4f}{ret_str}"
            )

    colorscale = [[i / n_h, c] for i, c in enumerate(["#1e1e1e"] + horizon_palette[:n_h])]
    colorscale[-1][0] = 1.0

    heatmap = go.Heatmap(
        z=z_matrix,
        x=tickers_sorted,
        y=date_strs,
        colorscale=colorscale,
        zmin=0, zmax=n_h,
        text=hover_matrix,
        hovertemplate="%{text}<extra></extra>",
        showscale=True,
        colorbar=dict(
            tickvals=list(range(n_h + 1)),
            ticktext=["—"] + [horizon_label[h] for h in horizon_order],
            title="Horizon",
            thickness=12,
        ),
    )

    # ── Panel 2: Weekly horizon usage (stacked bar) ──────────────────────────
    for h in horizon_order:
        sel[h] = (sel["best_horizon"] == h).astype(int)
    weekly_h = sel.set_index("date")[horizon_order].resample("W").sum()

    horizon_bars = []
    for h in horizon_order:
        horizon_bars.append(go.Bar(
            x=weekly_h.index, y=weekly_h[h],
            name=horizon_label[h],
            marker_color=horizon_colors[h],
            hovertemplate=f"Week %{{x|%b %d '%y}}<br>{horizon_label[h]}: %{{y}} picks<extra></extra>",
        ))

    # ── Panel 3: Weekly sector allocation (stacked bar) ──────────────────────
    sector_totals = {sec: int((sel["sector"] == sec).sum()) for sec in sel["sector"].unique()}
    sectors_all   = sorted(sector_totals, key=lambda s: -sector_totals[s])
    tab20 = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    ]
    for col in sectors_all:
        sel[col] = (sel["sector"] == col).astype(int)
    weekly_s = sel.set_index("date")[sectors_all].resample("W").sum()

    sector_bars = []
    for i, sec in enumerate(sectors_all):
        sector_bars.append(go.Bar(
            x=weekly_s.index, y=weekly_s[sec],
            name=f"{sec} ({sector_totals[sec]})",
            marker_color=tab20[i % len(tab20)],
            hovertemplate=f"Week %{{x|%b %d '%y}}<br>{sec}: %{{y}} picks<extra></extra>",
        ))

    # ── Assemble figure ──────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Daily Stock Selections by Horizon", "Horizon Usage Over Time", "Sector Allocation Over Time"),
        row_heights=[0.55, 0.225, 0.225],
        vertical_spacing=0.07,
    )

    fig.add_trace(heatmap, row=1, col=1)
    for bar in horizon_bars:
        bar.showlegend = True
        fig.add_trace(bar, row=2, col=1)
    for bar in sector_bars:
        bar.showlegend = True
        fig.add_trace(bar, row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        title="Decision Log — Daily Selections, Horizon & Sector Breakdown",
        height=1100,
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        font=dict(family="Inter, system-ui, sans-serif"),
        paper_bgcolor="#111",
        plot_bgcolor="#111",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(title_text="Pick Count", row=2, col=1)
    fig.update_yaxes(title_text="Pick Count", row=3, col=1)

    save_path = save_path.replace(".png", ".html")
    fig.write_html(save_path, include_plotlyjs="cdn")
    print(f"[Backtest] Interactive decision log saved to {save_path}")
    webbrowser.open(f"file://{os.path.abspath(save_path)}")


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
    parser.add_argument("--no-export",      action="store_true")
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
        export=not args.no_export,
    )
