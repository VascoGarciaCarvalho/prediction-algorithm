"""
Main Orchestrator — LSTM Stock Predictor

Usage examples:

  # Full pipeline (Phase 1 → 2 → 3 → backtest)
  python main.py --all

  # Individual phases
  python main.py --phase 1
  python main.py --phase 2
  python main.py --phase 3
  python main.py --backtest --model model_phase3.keras

  # Custom tickers / dates
  python main.py --all --tickers AAPL MSFT NVDA --start 2018-01-01 --end 2024-01-01
"""

import argparse
import os


DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "JNJ", "V",
    "UNH", "HD", "PG", "MA", "DIS",
]
DEFAULT_START = "2015-01-01"
DEFAULT_END   = "2024-01-01"
DEFAULT_WINDOW = 20
DEFAULT_MODEL_PATH = "model_phase3.keras"


def parse_args():
    parser = argparse.ArgumentParser(description="LSTM Stock Predictor")
    parser.add_argument("--all", action="store_true",
                        help="Run Phase 1 → 2 → 3 → backtest in sequence")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3],
                        help="Run a specific training phase only")
    parser.add_argument("--backtest", action="store_true",
                        help="Run backtest (requires trained model)")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Buy signal threshold (default 0.7)")
    parser.add_argument("--sims", type=int, default=1000,
                        help="Number of random baseline simulations")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help="Path to save/load .keras model")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    from training import run_phase1, run_phase2, run_phase3
    from backtesting import run_backtest
    from data_engineering import build_dataset

    kwargs = dict(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        window_size=args.window,
    )

    if args.all:
        # --- Phase 1 ---
        print("\n>>> Running Phase 1 (direct target sanity check)...")
        run_phase1(**kwargs)

        # --- Phase 2 ---
        print("\n>>> Running Phase 2 (indirect target check)...")
        run_phase2(**kwargs, epochs=50)

        # --- Phase 3 ---
        print("\n>>> Running Phase 3 (real training)...")
        model, dataset, _ = run_phase3(**kwargs, epochs=200, save_path=args.model)

        # --- Backtest ---
        print("\n>>> Running backtest...")
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

    elif args.phase == 1:
        run_phase1(**kwargs)

    elif args.phase == 2:
        run_phase2(**kwargs, epochs=50)

    elif args.phase == 3:
        run_phase3(**kwargs, epochs=200, save_path=args.model)

    elif args.backtest:
        if not os.path.exists(args.model):
            print(f"[ERROR] Model file not found: {args.model}")
            print("  Run Phase 3 first: python main.py --phase 3")
            return

        import tensorflow as tf
        from model import MCDropout
        model = tf.keras.models.load_model(args.model, custom_objects={"MCDropout": MCDropout})
        dataset = build_dataset(args.tickers, args.start, args.end, window_size=args.window, phase=3)
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

    else:
        print("No action specified. Use --all, --phase [1|2|3], or --backtest.")
        print("Run: python main.py --help")


if __name__ == "__main__":
    main()
