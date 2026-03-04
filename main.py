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

  # Custom tickers / dates / penalty
  python3 main.py --all --tickers AAPL MSFT NVDA --start 2018-01-01 --end 2024-01-01
  python3 main.py --backtest --lambda-penalty 0.5
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


def parse_args():
    parser = argparse.ArgumentParser(description="LSTM Stock Predictor")
    parser.add_argument("--all",     action="store_true",
                        help="Run Phase 1 → 2 → 3 → backtest in sequence")
    parser.add_argument("--phase",   type=int, choices=[1, 2, 3],
                        help="Run a specific training phase only")
    parser.add_argument("--backtest", action="store_true",
                        help="Run backtest (requires trained model)")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start",   default=DEFAULT_START)
    parser.add_argument("--end",     default=DEFAULT_END)
    parser.add_argument("--window",  type=int,   default=DEFAULT_WINDOW)
    parser.add_argument("--threshold",      type=float, default=0.7,
                        help="Minimum P(up) to trigger a buy signal (default 0.7)")
    parser.add_argument("--lambda-penalty", type=float, default=1.0,
                        help="Uncertainty penalty weight in risk-adjusted score (default 1.0)")
    parser.add_argument("--sims",    type=int,   default=1000,
                        help="Number of random baseline simulations")
    parser.add_argument("--model",   default=DEFAULT_MODEL_PATH,
                        help="Path to save/load .keras model")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

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
    )

    if args.all:
        print("\n>>> Running Phase 1 (direct target sanity check)...")
        run_phase1(**kwargs)

        print("\n>>> Running Phase 2 (indirect target check)...")
        run_phase2(**kwargs, epochs=50)

        print("\n>>> Running Phase 3 (real training)...")
        model, dataset, _ = run_phase3(**kwargs, epochs=200, save_path=args.model)

        print("\n>>> Running backtest...")
        run_backtest(model=model, dataset=dataset, **backtest_kwargs)

    elif args.phase == 1:
        run_phase1(**kwargs)

    elif args.phase == 2:
        run_phase2(**kwargs, epochs=50)

    elif args.phase == 3:
        run_phase3(**kwargs, epochs=200, save_path=args.model)

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

    else:
        print("No action specified. Use --all, --phase [1|2|3], or --backtest.")
        print("Run: python3 main.py --help")


if __name__ == "__main__":
    main()
