"""
Main Orchestrator — LSTM Stock Predictor

Usage examples:

  # Full pipeline (Phase 1 → 2 → 3 → backtest)
  python3 main.py --all

  # Individual phases
  python3 main.py --phase 1
  python3 main.py --phase 2
  python3 main.py --phase 3
  python3 main.py --backtest --model model_phase3_multihorizon.keras

  # Live signals for today (requires a trained model)
  python3 main.py --live --model model_phase3_multihorizon.keras

  # Custom tickers / dates / penalty
  python3 main.py --all --tickers AAPL MSFT NVDA --start 2018-01-01 --end 2024-01-01
  python3 main.py --backtest --lambda-penalty 0.5

  # Load defaults from a YAML config file (CLI flags still override)
  python3 main.py --config my_config.yaml --backtest
"""

import argparse
import os


DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "JNJ", "V",
    "UNH", "HD", "PG", "MA", "DIS",
]
DEFAULT_START      = "2015-01-01"
DEFAULT_END        = "2024-01-01"
DEFAULT_WINDOW     = 20
DEFAULT_MODEL_PATH = "model_phase3_multihorizon.keras"


def _load_config(path: str) -> dict:
    """Load YAML config file. Returns empty dict if file absent or PyYAML not installed."""
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        print(f"[Config] Loaded settings from {path}")
        return data
    except ImportError:
        print("[WARN] PyYAML not installed — ignoring config file. Run: pip install pyyaml")
        return {}


def parse_args(config: dict = None):
    cfg = config or {}
    parser = argparse.ArgumentParser(description="LSTM Stock Predictor")

    # Actions
    parser.add_argument("--all",      action="store_true",
                        help="Run Phase 1 → 2 → 3 → backtest in sequence")
    parser.add_argument("--phase",    type=int, choices=[1, 2, 3],
                        help="Run a specific training phase only")
    parser.add_argument("--backtest", action="store_true",
                        help="Run backtest (requires trained model)")
    parser.add_argument("--live",     action="store_true",
                        help="Generate live signals for today (requires trained model + scalers)")
    parser.add_argument("--walkforward", action="store_true",
                        help="Run walk-forward validation")
    parser.add_argument("--analyze", action="store_true",
                        help="Run post-backtest analysis on saved CSVs")

    # Data / model
    parser.add_argument("--tickers",  nargs="+",  default=cfg.get("tickers", DEFAULT_TICKERS))
    parser.add_argument("--start",    default=cfg.get("start",  DEFAULT_START))
    parser.add_argument("--end",      default=cfg.get("end",    DEFAULT_END))
    parser.add_argument("--window",   type=int,   default=cfg.get("window",  DEFAULT_WINDOW))
    parser.add_argument("--model",    default=cfg.get("model",  DEFAULT_MODEL_PATH),
                        help="Path to save/load .keras model")

    # Strategy
    parser.add_argument("--threshold",      type=float, default=cfg.get("threshold",      0.6),
                        help="Minimum P(up) to trigger a buy signal (default 0.6)")
    parser.add_argument("--lambda-penalty", type=float, default=cfg.get("lambda_penalty", 0.5),
                        help="Uncertainty penalty weight in risk-adjusted score (default 0.5)")
    parser.add_argument("--sims",           type=int,   default=cfg.get("sims",           1000),
                        help="Number of random baseline simulations")

    # Walk-forward
    parser.add_argument("--wf-train-years",  type=int, default=cfg.get("wf_train_years",  5))
    parser.add_argument("--wf-test-months",  type=int, default=cfg.get("wf_test_months",  6))

    # Output
    parser.add_argument("--no-plot",   action="store_true")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip CSV/JSON export after backtest (export is on by default)")
    parser.add_argument("--config",    default=None,
                        help="Path to YAML config file (values can be overridden by CLI flags)")

    return parser.parse_args()


def _run_live(args):
    """Generate live buy/hold signals for today using a trained model."""
    import numpy as np
    import tensorflow as tf
    from model import MCDropout, mc_predict
    from data_engineering import build_live_windows, build_dataset, HORIZONS
    from backtesting import SECTOR_MAP, HORIZON_DAYS

    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        print("  Train first: python3 main.py --phase 3")
        return

    print(f"\n[Live] Loading model from {args.model}...")
    model = tf.keras.models.load_model(args.model, custom_objects={"MCDropout": MCDropout})

    # Try to recover scalers from a recent dataset; fall back to no scaling
    scalers = None
    print("[Live] Building training dataset to recover feature scalers...")
    try:
        dataset = build_dataset(
            args.tickers, args.start, args.end, window_size=args.window, phase=3
        )
        scalers = dataset.get("scalers")
    except Exception as e:
        print(f"[WARN] Could not recover scalers: {e}. Predictions will be unscaled.")

    print("[Live] Downloading today's market data and building windows...")
    X_live, live_tickers = build_live_windows(
        args.tickers, window_size=args.window, scalers=scalers
    )

    print(f"[Live] Running MC inference on {len(live_tickers)} tickers...")
    mc_results = mc_predict(model, X_live, n_passes=50)

    horizon_keys = [f"out_{h}d" for h in HORIZONS]
    rows = []
    for i, ticker in enumerate(live_tickers):
        best_score = -np.inf
        best_h     = horizon_keys[0]
        best_prob  = 0.0
        best_unc   = 0.0

        for key in horizon_keys:
            h_days    = HORIZON_DAYS[key]
            mean_p, std_p = mc_results[key]
            prob = float(mean_p[i])
            unc  = float(std_p[i])
            score = (prob / h_days) - args.lambda_penalty * (unc / h_days)
            if score > best_score:
                best_score = score
                best_h     = key
                best_prob  = prob
                best_unc   = unc

        signal = (
            "BUY"
            if best_score > 0 and best_prob >= args.threshold
            else "hold"
        )
        rows.append({
            "ticker":       ticker,
            "signal":       signal,
            "p_up":         best_prob,
            "uncertainty":  best_unc,
            "best_horizon": best_h.replace("out_", ""),
            "score":        best_score,
            "sector":       SECTOR_MAP.get(ticker, "Unknown"),
        })

    rows.sort(key=lambda r: r["score"], reverse=True)

    print("\n" + "=" * 70)
    print("LIVE SIGNALS — TODAY")
    print("=" * 70)
    print(f"  {'Ticker':<8} {'Signal':<6} {'P(up)':>6} {'Uncert':>8} {'Horizon':>8} {'Score':>8}  Sector")
    print("  " + "-" * 66)
    for r in rows:
        flag = "  <---" if r["signal"] == "BUY" else ""
        print(
            f"  {r['ticker']:<8} {r['signal']:<6} {r['p_up']:>6.1%} "
            f"{r['uncertainty']:>8.4f} {r['best_horizon']:>8} {r['score']:>8.4f}  "
            f"{r['sector']}{flag}"
        )
    print("=" * 70)
    buys = [r for r in rows if r["signal"] == "BUY"]
    print(f"\n  {len(buys)} BUY signal(s) out of {len(rows)} tickers.")


def main():
    # Two-pass parse: first grab --config, then load YAML and re-parse
    import sys
    config_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break
    cfg  = _load_config(config_path)
    args = parse_args(cfg)

    from training import run_phase1, run_phase2, run_phase3
    from backtesting import run_backtest, SECTOR_MAP
    from data_engineering import build_dataset

    kwargs = dict(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        window_size=args.window,
    )

    backtest_kwargs = dict(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        buy_threshold=args.threshold,
        lambda_penalty=args.lambda_penalty,
        sector_map=SECTOR_MAP,
        n_random_sims=args.sims,
        plot=not args.no_plot,
        export=not args.no_export,
    )

    if args.live:
        _run_live(args)

    elif args.all:
        print("\n>>> Running Phase 1 (direct target sanity check)...")
        run_phase1(**kwargs)

        print("\n>>> Running Phase 2 (indirect target check)...")
        run_phase2(**kwargs, epochs=50)

        print("\n>>> Running Phase 3 (real training)...")
        model, dataset, _ = run_phase3(**kwargs, epochs=500, save_path=args.model)

        print("\n>>> Running backtest...")
        run_backtest(model=model, dataset=dataset, **backtest_kwargs)

        print("\n>>> Running post-backtest analysis...")
        from analysis import run_analysis
        run_analysis(plot=not args.no_plot)

    elif args.phase == 1:
        run_phase1(**kwargs)

    elif args.phase == 2:
        run_phase2(**kwargs, epochs=50)

    elif args.phase == 3:
        run_phase3(**kwargs, epochs=500, save_path=args.model)

    elif args.backtest:
        if not os.path.exists(args.model):
            print(f"[ERROR] Model file not found: {args.model}")
            print("  Run Phase 3 first: python3 main.py --phase 3")
            return

        import tensorflow as tf
        from model import MCDropout
        model   = tf.keras.models.load_model(
            args.model, custom_objects={"MCDropout": MCDropout}
        )
        dataset = build_dataset(
            args.tickers, args.start, args.end, window_size=args.window, phase=3
        )
        run_backtest(model=model, dataset=dataset, **backtest_kwargs)

        print("\n>>> Running post-backtest analysis...")
        from analysis import run_analysis
        run_analysis(plot=not args.no_plot)

    elif args.walkforward:
        from walkforward import run_walkforward
        run_walkforward(
            tickers=args.tickers,
            start=args.start,
            end=args.end,
            window_size=args.window,
            train_years=args.wf_train_years,
            test_months=args.wf_test_months,
            buy_threshold=args.threshold,
            lambda_penalty=args.lambda_penalty,
            sector_map=SECTOR_MAP,
            model_path=args.model,
            plot=not args.no_plot,
            export=not args.no_export,
        )

    elif args.analyze:
        from analysis import run_analysis
        run_analysis(plot=not args.no_plot)

    else:
        print("No action specified. Use --all, --phase [1|2|3], --backtest, --live, --walkforward, or --analyze.")
        print("Run: python3 main.py --help")


if __name__ == "__main__":
    main()
