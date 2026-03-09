"""
Walk-Forward Validation
=======================

Re-trains on a rolling window of history, tests on the next block, then steps
forward. Averages key metrics across all folds to give a more robust estimate
of out-of-sample performance than a single 80/20 split.

Algorithm
---------
  Given total date range [start, end]:
    fold 0 : train on [start, start + train_years], test on the next test_months
    fold 1 : train on [start + step, ...], test on the next test_months
    ...
  Step size == test_months (non-overlapping test windows).

Each fold:
  1. Call run_phase3() to train a fresh model on the fold's training data.
  2. Call run_backtest() on the fold's test data (no random simulation, no plots,
     no export — to keep runtime manageable).
  3. Record total_return, annualised_return, sharpe_ratio, max_drawdown.

Final output: per-fold table + mean ± std across folds.
"""

import json
from datetime import date
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_str(d) -> str:
    return d.strftime("%Y-%m-%d")


def _build_folds(
    start: str,
    end: str,
    train_years: int,
    test_months: int,
) -> list[dict]:
    """
    Build non-overlapping walk-forward fold definitions.

    Returns list of dicts: {train_start, train_end, test_start, test_end}
    """
    start_d = date.fromisoformat(start)
    end_d   = date.fromisoformat(end)

    folds = []
    fold_start = start_d
    while True:
        train_end  = fold_start + relativedelta(years=train_years)
        test_start = train_end
        test_end   = test_start + relativedelta(months=test_months)

        if test_end > end_d:
            break

        folds.append({
            "train_start": _date_str(fold_start),
            "train_end":   _date_str(train_end),
            "test_start":  _date_str(test_start),
            "test_end":    _date_str(test_end),
        })
        fold_start = fold_start + relativedelta(months=test_months)

    return folds


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_walkforward(
    tickers: list[str],
    start: str,
    end: str,
    window_size: int = 20,
    train_years: int = 5,
    test_months: int = 6,
    buy_threshold: float = 0.6,
    lambda_penalty: float = 0.5,
    sector_map: dict = None,
    model_path: str = "model_wf_fold.keras",
    plot: bool = False,
    export: bool = True,
) -> list[dict]:
    """
    Run walk-forward validation.

    Parameters
    ----------
    tickers, start, end        : same as build_dataset()
    train_years                : number of years in each training window
    test_months                : number of months in each test window (= step size)
    buy_threshold, lambda_penalty, sector_map : passed to run_backtest()
    model_path                 : base path — fold index is appended automatically
    plot                       : whether to generate plots per fold (slow)
    export                     : whether to write a final walk-forward summary JSON

    Returns
    -------
    List of per-fold result dicts.
    """
    from training import run_phase3
    from backtesting import run_backtest, SECTOR_MAP
    from data_engineering import build_dataset

    if sector_map is None:
        sector_map = SECTOR_MAP

    folds = _build_folds(start, end, train_years, test_months)
    if not folds:
        print(f"[WalkForward] No folds generated for {start} → {end} "
              f"with train_years={train_years}, test_months={test_months}.")
        return []

    print(f"\n[WalkForward] {len(folds)} fold(s) detected.")
    for i, f in enumerate(folds):
        print(f"  Fold {i}: train {f['train_start']} → {f['train_end']}  |  "
              f"test {f['test_start']} → {f['test_end']}")

    fold_results = []

    for fold_idx, fold in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"[WalkForward] Fold {fold_idx} / {len(folds) - 1}")
        print(f"  Training : {fold['train_start']} → {fold['train_end']}")
        print(f"  Testing  : {fold['test_start']} → {fold['test_end']}")
        print("=" * 60)

        fold_model_path = model_path.replace(".keras", f"_fold{fold_idx}.keras")

        try:
            # Train on fold training window
            model, _, _ = run_phase3(
                tickers=tickers,
                start=fold["train_start"],
                end=fold["train_end"],
                window_size=window_size,
                epochs=200,
                save_path=fold_model_path,
            )

            # Build test dataset (full pipeline for the test window)
            test_dataset = build_dataset(
                tickers,
                start=fold["test_start"],
                end=fold["test_end"],
                window_size=window_size,
                phase=3,
            )

            # Run backtest on test window (skip random sims + plots for speed)
            bt_result = run_backtest(
                model=model,
                dataset=test_dataset,
                tickers=tickers,
                start=fold["test_start"],
                end=fold["test_end"],
                buy_threshold=buy_threshold,
                lambda_penalty=lambda_penalty,
                sector_map=sector_map,
                n_random_sims=0,   # skip random sims per fold
                mc_passes=50,
                plot=plot,
                export=False,      # aggregate export at end
            )

            risk = bt_result.get("risk_metrics", {})
            fold_results.append({
                "fold":              fold_idx,
                "train_start":       fold["train_start"],
                "train_end":         fold["train_end"],
                "test_start":        fold["test_start"],
                "test_end":          fold["test_end"],
                "total_return":      risk.get("total_return",      float("nan")),
                "annualised_return": risk.get("annualised_return", float("nan")),
                "sharpe_ratio":      risk.get("sharpe_ratio",      float("nan")),
                "max_drawdown":      risk.get("max_drawdown",      float("nan")),
                "calmar_ratio":      risk.get("calmar_ratio",      float("nan")),
                "win_rate":          risk.get("win_rate",          float("nan")),
            })

        except Exception as e:
            print(f"[WalkForward] Fold {fold_idx} failed: {e}")
            fold_results.append({"fold": fold_idx, "error": str(e)})

    # Summary table
    valid = [r for r in fold_results if "error" not in r]
    print("\n" + "=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)
    print(f"  {'Fold':>4}  {'Test Period':<24} {'TotalRet':>9} {'AnnRet':>8} {'Sharpe':>7} {'MaxDD':>8}")
    print("  " + "-" * 64)
    for r in fold_results:
        if "error" in r:
            print(f"  {r['fold']:>4}  ERROR: {r['error']}")
            continue
        print(
            f"  {r['fold']:>4}  {r['test_start']} → {r['test_end']}  "
            f"{r['total_return']:>+9.2%}  {r['annualised_return']:>+7.2%}  "
            f"{r['sharpe_ratio']:>7.3f}  {r['max_drawdown']:>+7.2%}"
        )

    if valid:
        metrics = ["total_return", "annualised_return", "sharpe_ratio", "max_drawdown", "win_rate"]
        print("  " + "-" * 64)
        means = {m: np.nanmean([r[m] for r in valid]) for m in metrics}
        stds  = {m: np.nanstd( [r[m] for r in valid]) for m in metrics}
        print(
            f"  {'Mean':>4}  {'':24}  "
            f"{means['total_return']:>+9.2%}  {means['annualised_return']:>+7.2%}  "
            f"{means['sharpe_ratio']:>7.3f}  {means['max_drawdown']:>+7.2%}"
        )
        print(
            f"  {'Std':>4}  {'':24}  "
            f"{stds['total_return']:>9.4f}  {stds['annualised_return']:>7.4f}  "
            f"{stds['sharpe_ratio']:>7.4f}  {stds['max_drawdown']:>7.4f}"
        )
        print(f"\n  Mean Win Rate : {means['win_rate']:.1%}")
    print("=" * 70)

    if export and fold_results:
        def _safe(v):
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return None
            return v

        out = {
            "folds": [{k: _safe(v) for k, v in r.items()} for r in fold_results],
            "summary": {m: {"mean": _safe(means.get(m)), "std": _safe(stds.get(m))} for m in metrics} if valid else {},
        }
        with open("walkforward_results.json", "w") as f:
            json.dump(out, f, indent=2)
        print("[WalkForward] Results saved to walkforward_results.json")

    return fold_results
