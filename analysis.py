"""
Post-Backtest Analysis
======================
Loads backtest_selections.csv and backtest_daily_returns.csv and produces:

  1. Horizon distribution         — pick frequency per horizon
  2. Per-ticker win rate          — win rate, mean return, times picked
  3. Per-sector analysis          — sector-level win rate and return
  4. Score calibration            — does higher score predict better returns?
  5. Drawdown analysis            — max drawdown, longest drawdown period
  6. Rolling Sharpe ratio         — 60-day rolling Sharpe vs S&P 500
  7. Regime breakdown             — bull vs bear market performance
  8. Score threshold sweep        — cumulative return at different prob_up thresholds

Usage:
  python3 analysis.py
  python3 analysis.py --selections my_selections.csv --returns my_returns.csv --no-plot
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Loaders
# ---------------------------------------------------------------------------

def load_data(
    selections_path: str = "backtest_selections.csv",
    returns_path: str = "backtest_daily_returns.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sel = pd.read_csv(selections_path, parse_dates=["date"])
    ret = pd.read_csv(returns_path, parse_dates=["date"])
    ret = ret.set_index("date").sort_index()
    return sel, ret


# ---------------------------------------------------------------------------
# 2. Horizon Distribution
# ---------------------------------------------------------------------------

def horizon_distribution(sel: pd.DataFrame) -> pd.DataFrame:
    counts = sel["best_horizon"].value_counts().reset_index()
    counts.columns = ["horizon", "count"]
    counts["pct"] = 100 * counts["count"] / counts["count"].sum()
    counts = counts.sort_values("horizon")
    return counts


# ---------------------------------------------------------------------------
# 3. Per-Ticker Win Rate
# ---------------------------------------------------------------------------

def ticker_win_rate(sel: pd.DataFrame) -> pd.DataFrame:
    if "pick_return" not in sel.columns:
        print("[Analysis] 'pick_return' column missing — re-run backtest to get per-pick returns.")
        return pd.DataFrame()
    rows = []
    for ticker, grp in sel.groupby("ticker"):
        rets = grp["pick_return"].dropna().values
        rows.append({
            "ticker":       ticker,
            "times_picked": len(rets),
            "win_rate":     float(np.mean(rets > 0)) if len(rets) else 0.0,
            "mean_return":  float(np.mean(rets)) if len(rets) else 0.0,
            "total_return": float(np.sum(rets)) if len(rets) else 0.0,
        })
    df = pd.DataFrame(rows).sort_values("win_rate", ascending=False)
    return df


# ---------------------------------------------------------------------------
# 4. Per-Sector Analysis
# ---------------------------------------------------------------------------

def sector_analysis(sel: pd.DataFrame) -> pd.DataFrame:
    if "pick_return" not in sel.columns:
        print("[Analysis] 'pick_return' column missing — re-run backtest to get per-pick returns.")
        return pd.DataFrame()
    rows = []
    for sector, grp in sel.groupby("sector"):
        rets = grp["pick_return"].dropna().values
        rows.append({
            "sector":       sector,
            "times_picked": len(rets),
            "win_rate":     float(np.mean(rets > 0)) if len(rets) else 0.0,
            "mean_return":  float(np.mean(rets)) if len(rets) else 0.0,
            "total_return": float(np.sum(rets)) if len(rets) else 0.0,
        })
    df = pd.DataFrame(rows).sort_values("mean_return", ascending=False)
    return df


# ---------------------------------------------------------------------------
# 5. Score Calibration
# ---------------------------------------------------------------------------

def score_calibration(sel: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    """Bin picks by best_score quartile, show mean return per bin."""
    if "pick_return" not in sel.columns:
        print("[Analysis] 'pick_return' column missing — re-run backtest to get per-pick returns.")
        return pd.DataFrame()
    df = sel[["best_score", "best_prob_up", "pick_return"]].dropna().copy()
    df["score_bin"] = pd.qcut(df["best_score"], q=n_bins, labels=[f"Q{i+1}" for i in range(n_bins)])
    result = df.groupby("score_bin", observed=True).agg(
        count=("pick_return", "count"),
        mean_score=("best_score", "mean"),
        mean_prob_up=("best_prob_up", "mean"),
        mean_return=("pick_return", "mean"),
        win_rate=("pick_return", lambda x: (x > 0).mean()),
    ).reset_index()
    return result


# ---------------------------------------------------------------------------
# 6. Drawdown Analysis
# ---------------------------------------------------------------------------

def drawdown_analysis(ret: pd.DataFrame) -> dict:
    equity = (1 + ret["strategy_daily_return"]).cumprod()
    peak   = equity.cummax()
    dd     = (equity - peak) / peak

    max_dd   = float(dd.min())
    max_dd_date = dd.idxmin()

    # Longest drawdown duration
    in_dd = dd < 0
    longest = 0
    current = 0
    for v in in_dd:
        if v:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return {
        "max_drawdown":      max_dd,
        "max_drawdown_date": max_dd_date,
        "longest_dd_days":   longest,
        "drawdown_series":   dd,
        "equity_series":     equity,
    }


# ---------------------------------------------------------------------------
# 7. Rolling Sharpe
# ---------------------------------------------------------------------------

def rolling_sharpe(ret: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    strat = ret["strategy_daily_return"]
    sp500 = ret["sp500_daily_return"]

    def _sharpe(s: pd.Series) -> pd.Series:
        mu = s.rolling(window).mean()
        sd = s.rolling(window).std()
        return (mu / sd) * np.sqrt(252)

    return pd.DataFrame({
        "strategy_sharpe": _sharpe(strat),
        "sp500_sharpe":    _sharpe(sp500),
    }, index=ret.index)


# ---------------------------------------------------------------------------
# 8. Regime Breakdown
# ---------------------------------------------------------------------------

def regime_breakdown(ret: pd.DataFrame, window: int = 20) -> dict:
    """Split performance by bull/bear regime using SP500 rolling return sign."""
    sp500_roll = ret["sp500_daily_return"].rolling(window).mean()
    bull_mask  = sp500_roll >= 0

    def _stats(series: pd.Series) -> dict:
        cum = float((1 + series).prod() - 1)
        sr  = float((series.mean() / series.std()) * np.sqrt(252)) if series.std() > 0 else 0.0
        return {"total_return": cum, "sharpe": sr, "n_days": len(series)}

    results = {}
    for label, mask in [("bull", bull_mask), ("bear", ~bull_mask)]:
        results[label] = {
            "strategy": _stats(ret.loc[mask, "strategy_daily_return"].dropna()),
            "sp500":    _stats(ret.loc[mask, "sp500_daily_return"].dropna()),
        }
    return results


# ---------------------------------------------------------------------------
# 9. Score Threshold Sweep
# ---------------------------------------------------------------------------

def threshold_sweep(
    sel: pd.DataFrame,
    ret: pd.DataFrame,
    thresholds: list[float] = None,
) -> pd.DataFrame:
    """
    For each prob_up threshold, simulate the strategy using only picks above
    that threshold, compute cumulative return.
    """
    if "pick_return" not in sel.columns:
        print("[Analysis] 'pick_return' column missing — re-run backtest to get per-pick returns.")
        return pd.DataFrame()

    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    rows = []
    for thresh in thresholds:
        filtered = sel[sel["best_prob_up"] >= thresh]
        if filtered.empty:
            rows.append({"threshold": thresh, "n_picks": 0, "win_rate": 0.0, "mean_return": 0.0, "total_return": 0.0})
            continue
        rets  = filtered["pick_return"].dropna()
        rows.append({
            "threshold":    thresh,
            "n_picks":      len(rets),
            "win_rate":     float((rets > 0).mean()),
            "mean_return":  float(rets.mean()),
            "total_return": float(rets.sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 10. Print Reports
# ---------------------------------------------------------------------------

def _pct(x):
    return f"{x*100:+.2f}%"

def print_report(sel, ret, horizon_dist, ticker_wr, sector_an, score_cal, dd, regime, thresh_df):
    w = 65
    print("\n" + "=" * w)
    print("POST-BACKTEST ANALYSIS REPORT")
    print("=" * w)

    # Horizon distribution
    print("\n── 1. HORIZON DISTRIBUTION ──────────────────────────────────")
    for _, r in horizon_dist.iterrows():
        bar = "█" * int(r["pct"] / 2)
        print(f"  {r['horizon']:<12}  {int(r['count']):>4} picks  ({r['pct']:5.1f}%)  {bar}")

    # Per-ticker win rate
    if not ticker_wr.empty:
        print("\n── 2. PER-TICKER WIN RATE ────────────────────────────────────")
        print(f"  {'Ticker':<8} {'Picks':>5} {'Win%':>7} {'Mean Ret':>10} {'Total Ret':>11}")
        print("  " + "-" * 44)
        for _, r in ticker_wr.iterrows():
            print(f"  {r['ticker']:<8} {int(r['times_picked']):>5} {r['win_rate']*100:>6.1f}%"
                  f" {_pct(r['mean_return']):>10} {_pct(r['total_return']):>11}")

    # Per-sector
    if not sector_an.empty:
        print("\n── 3. PER-SECTOR ANALYSIS ────────────────────────────────────")
        print(f"  {'Sector':<28} {'Picks':>5} {'Win%':>7} {'Mean Ret':>10}")
        print("  " + "-" * 53)
        for _, r in sector_an.iterrows():
            print(f"  {r['sector']:<28} {int(r['times_picked']):>5} {r['win_rate']*100:>6.1f}%"
                  f" {_pct(r['mean_return']):>10}")

    # Score calibration
    if not score_cal.empty:
        print("\n── 4. SCORE CALIBRATION (quartiles) ─────────────────────────")
        print(f"  {'Bin':<6} {'N':>5} {'Avg Score':>10} {'Avg P(up)':>10} {'Mean Ret':>10} {'Win%':>7}")
        print("  " + "-" * 51)
        for _, r in score_cal.iterrows():
            print(f"  {r['score_bin']:<6} {int(r['count']):>5} {r['mean_score']:>10.4f}"
                  f" {r['mean_prob_up']:>10.3f} {_pct(r['mean_return']):>10} {r['win_rate']*100:>6.1f}%")

    # Drawdown
    print("\n── 5. DRAWDOWN ANALYSIS ──────────────────────────────────────")
    print(f"  Max Drawdown          : {_pct(dd['max_drawdown'])}  (on {dd['max_drawdown_date'].date()})")
    print(f"  Longest DD Period     : {dd['longest_dd_days']} trading days")

    # Regime
    print("\n── 7. REGIME BREAKDOWN ───────────────────────────────────────")
    for regime_name, data in regime.items():
        s, m = data["strategy"], data["sp500"]
        print(f"  {regime_name.upper()} market ({s['n_days']} days):")
        print(f"    Strategy  total={_pct(s['total_return'])}  Sharpe={s['sharpe']:.2f}")
        print(f"    S&P 500   total={_pct(m['total_return'])}  Sharpe={m['sharpe']:.2f}")

    # Threshold sweep
    if not thresh_df.empty:
        print("\n── 8. SCORE THRESHOLD SWEEP ──────────────────────────────────")
        print(f"  {'P(up) ≥':<10} {'Picks':>6} {'Win%':>7} {'Mean Ret':>10} {'Total Ret':>11}")
        print("  " + "-" * 47)
        for _, r in thresh_df.iterrows():
            print(f"  {r['threshold']:.2f}       {int(r['n_picks']):>6} {r['win_rate']*100:>6.1f}%"
                  f" {_pct(r['mean_return']):>10} {_pct(r['total_return']):>11}")

    print("\n" + "=" * w)


# ---------------------------------------------------------------------------
# 11. Plots
# ---------------------------------------------------------------------------

def plot_analysis(
    sel, ret, horizon_dist, ticker_wr, sector_an, score_cal, dd, sharpe_df, regime, thresh_df,
    save_path: str = "analysis_report.html",
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import webbrowser, os

    has_returns = "pick_return" in sel.columns
    palette = ["#084c61", "#4c9be8", "#a8d5f5", "#f5c842", "#e87c4c"]

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "1. Horizon Distribution",
            "2. Per-Ticker Win Rate",
            "3. Per-Sector Mean Return",
            "4. Score Calibration (by Quartile)",
            f"5. Drawdown  (max {dd['max_drawdown']*100:.1f}%)",
            "6. Rolling 60-Day Sharpe",
            "7. Regime Breakdown",
            "8. P(up) Threshold Sweep",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "xy"},  {"type": "xy"}],
            [{"type": "bar"}, {"type": "xy"}],
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    # ── 1. Horizon distribution (bar) ────────────────────────────────────────
    h_labels = horizon_dist["horizon"].str.replace("out_", "")
    fig.add_trace(go.Bar(
        x=h_labels, y=horizon_dist["count"],
        marker_color=palette[:len(horizon_dist)],
        text=[f"{p:.1f}%" for p in horizon_dist["pct"]],
        textposition="outside",
        customdata=horizon_dist["pct"],
        hovertemplate="%{x}: %{y} picks (%{customdata:.1f}%)<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # ── 2. Per-ticker win rate ────────────────────────────────────────────────
    if has_returns and not ticker_wr.empty:
        colors = ["#2ecc71" if w >= 0.5 else "#e74c3c" for w in ticker_wr["win_rate"]]
        fig.add_trace(go.Bar(
            x=ticker_wr["ticker"], y=ticker_wr["win_rate"] * 100,
            marker_color=colors,
            customdata=np.stack([ticker_wr["times_picked"], ticker_wr["mean_return"] * 100], axis=-1),
            hovertemplate="<b>%{x}</b><br>Win Rate: %{y:.1f}%<br>Picks: %{customdata[0]}<br>Mean Ret: %{customdata[1]:+.2f}%<extra></extra>",
            showlegend=False,
        ), row=1, col=2)
        fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=1, col=2)

    # ── 3. Per-sector mean return (horizontal bar) ────────────────────────────
    if has_returns and not sector_an.empty:
        colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in sector_an["mean_return"]]
        fig.add_trace(go.Bar(
            x=sector_an["mean_return"] * 100, y=sector_an["sector"],
            orientation="h",
            marker_color=colors,
            customdata=np.stack([sector_an["times_picked"], sector_an["win_rate"] * 100], axis=-1),
            hovertemplate="<b>%{y}</b><br>Mean Return: %{x:+.2f}%<br>Picks: %{customdata[0]}<br>Win Rate: %{customdata[1]:.1f}%<extra></extra>",
            showlegend=False,
        ), row=2, col=1)
        fig.add_vline(x=0, line_color="rgba(255,255,255,0.3)", row=2, col=1)

    # ── 4. Score calibration ──────────────────────────────────────────────────
    if has_returns and not score_cal.empty:
        fig.add_trace(go.Bar(
            x=score_cal["score_bin"].astype(str), y=score_cal["mean_return"] * 100,
            marker_color=palette[:len(score_cal)],
            customdata=np.stack([score_cal["count"], score_cal["win_rate"] * 100, score_cal["mean_prob_up"]], axis=-1),
            hovertemplate="<b>%{x}</b><br>Mean Return: %{y:+.2f}%<br>Picks: %{customdata[0]}<br>Win Rate: %{customdata[1]:.1f}%<br>Avg P(up): %{customdata[2]:.1%}<extra></extra>",
            showlegend=False,
        ), row=2, col=2)
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", row=2, col=2)

    # ── 5. Drawdown ───────────────────────────────────────────────────────────
    dd_s = dd["drawdown_series"] * 100
    fig.add_trace(go.Scatter(
        x=dd_s.index, y=dd_s,
        mode="lines", name="Drawdown",
        line=dict(color="#e74c3c", width=1.5),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
        hovertemplate="%{x|%b %d %Y}<br>DD: %{y:.2f}%<extra></extra>",
        showlegend=False,
    ), row=3, col=1)

    # ── 6. Rolling Sharpe ─────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=sharpe_df.index, y=sharpe_df["sp500_sharpe"],
        mode="lines", name="S&P 500 Sharpe",
        line=dict(color="#aaaaaa", width=1.5, dash="dash"),
        hovertemplate="%{x|%b %d %Y}<br>S&P 500: %{y:.2f}<extra></extra>",
    ), row=3, col=2)
    fig.add_trace(go.Scatter(
        x=sharpe_df.index, y=sharpe_df["strategy_sharpe"],
        mode="lines", name="Strategy Sharpe",
        line=dict(color="#4c9be8", width=1.5),
        hovertemplate="%{x|%b %d %Y}<br>Strategy: %{y:.2f}<extra></extra>",
    ), row=3, col=2)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", row=3, col=2)

    # ── 7. Regime breakdown ───────────────────────────────────────────────────
    regime_labels = list(regime.keys())
    strat_rets = [regime[r]["strategy"]["total_return"] * 100 for r in regime_labels]
    sp500_rets = [regime[r]["sp500"]["total_return"] * 100 for r in regime_labels]
    strat_sh   = [regime[r]["strategy"]["sharpe"] for r in regime_labels]
    sp500_sh   = [regime[r]["sp500"]["sharpe"]    for r in regime_labels]

    fig.add_trace(go.Bar(
        x=[r.title() for r in regime_labels], y=strat_rets,
        name="Strategy", marker_color="#084c61",
        customdata=[[s] for s in strat_sh],
        hovertemplate="<b>%{x}</b><br>Return: %{y:+.1f}%<br>Sharpe: %{customdata[0]:.2f}<extra>Strategy</extra>",
    ), row=4, col=1)
    fig.add_trace(go.Bar(
        x=[r.title() for r in regime_labels], y=sp500_rets,
        name="S&P 500", marker_color="#aaaaaa",
        customdata=[[s] for s in sp500_sh],
        hovertemplate="<b>%{x}</b><br>Return: %{y:+.1f}%<br>Sharpe: %{customdata[0]:.2f}<extra>S&P 500</extra>",
    ), row=4, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", row=4, col=1)

    # ── 8. Threshold sweep ────────────────────────────────────────────────────
    if has_returns and not thresh_df.empty and thresh_df["n_picks"].sum() > 0:
        fig.add_trace(go.Scatter(
            x=thresh_df["threshold"], y=thresh_df["win_rate"] * 100,
            mode="lines+markers", name="Win Rate",
            line=dict(color="#4c9be8", width=2),
            marker=dict(size=7),
            hovertemplate="Threshold ≥ %{x:.0%}<br>Win Rate: %{y:.1f}%<extra></extra>",
        ), row=4, col=2)
        fig.add_trace(go.Scatter(
            x=thresh_df["threshold"], y=thresh_df["n_picks"],
            mode="lines+markers", name="# Picks",
            line=dict(color="#e87c4c", width=2, dash="dash"),
            marker=dict(size=7),
            hovertemplate="Threshold ≥ %{x:.0%}<br># Picks: %{y}<extra></extra>",
        ), row=4, col=2)
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=4, col=2)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        title=dict(text="Post-Backtest Analysis Report", font=dict(size=18)),
        height=1400,
        barmode="group",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        font=dict(family="Inter, system-ui, sans-serif"),
        paper_bgcolor="#111",
        plot_bgcolor="#111",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(ticksuffix="%", row=3, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
    fig.update_yaxes(title_text="Sharpe", row=3, col=2)
    fig.update_yaxes(title_text="Win Rate (%)", row=4, col=2)
    fig.update_xaxes(title_text="P(up) Threshold", row=4, col=2)

    save_path = save_path.replace(".png", ".html")
    fig.write_html(save_path, include_plotlyjs="cdn")
    print(f"[Analysis] Interactive report saved to {save_path}")
    webbrowser.open(f"file://{os.path.abspath(save_path)}")


# ---------------------------------------------------------------------------
# 12. Main
# ---------------------------------------------------------------------------

def run_analysis(
    selections_path: str = "backtest_selections.csv",
    returns_path:    str = "backtest_daily_returns.csv",
    plot:            bool = True,
    save_path:       str = "analysis_report.html",
):
    print(f"[Analysis] Loading {selections_path} and {returns_path}...")
    sel, ret = load_data(selections_path, returns_path)

    horizon_dist = horizon_distribution(sel)
    ticker_wr    = ticker_win_rate(sel)
    sector_an    = sector_analysis(sel)
    score_cal    = score_calibration(sel)
    dd           = drawdown_analysis(ret)
    sharpe_df    = rolling_sharpe(ret)
    regime       = regime_breakdown(ret)
    thresh_df    = threshold_sweep(sel, ret)

    print_report(sel, ret, horizon_dist, ticker_wr, sector_an, score_cal, dd, regime, thresh_df)

    if plot:
        plot_analysis(
            sel, ret, horizon_dist, ticker_wr, sector_an,
            score_cal, dd, sharpe_df, regime, thresh_df,
            save_path=save_path,
        )

    return {
        "horizon_distribution": horizon_dist,
        "ticker_win_rate":      ticker_wr,
        "sector_analysis":      sector_an,
        "score_calibration":    score_cal,
        "drawdown":             dd,
        "rolling_sharpe":       sharpe_df,
        "regime_breakdown":     regime,
        "threshold_sweep":      thresh_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-backtest analysis")
    parser.add_argument("--selections", default="backtest_selections.csv")
    parser.add_argument("--returns",    default="backtest_daily_returns.csv")
    parser.add_argument("--no-plot",    action="store_true")
    parser.add_argument("--save",       default="analysis_report.html")
    args = parser.parse_args()

    run_analysis(
        selections_path=args.selections,
        returns_path=args.returns,
        plot=not args.no_plot,
        save_path=args.save,
    )
