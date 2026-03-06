"""
Data Engineering Module
Handles feature engineering, alternative data, preprocessing, and data splitting.

Alternative data sources (all free, no API key required):
  - Insider transactions: yfinance Ticker.insider_transactions
  - News sentiment:       yfinance Ticker.news + VADER (vaderSentiment, local NLP)

Time horizons: 1d, 5d, 21d, 126d (1 day, 1 week, 1 month, 6 months in trading days)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings("ignore")

_vader = SentimentIntensityAnalyzer()

# Holding horizons in trading days — imported by model.py and backtesting.py
HORIZONS = [1, 5, 21, 126]

TARGET_COLS     = [f"Target_{h}d"        for h in HORIZONS]
NEXT_CLOSE_COLS = [f"Next_Close_{h}d"    for h in HORIZONS]
RETURN_COLS     = [f"Future_Return_{h}d" for h in HORIZONS]


# ---------------------------------------------------------------------------
# 1. Market Data Download
# ---------------------------------------------------------------------------

def download_market_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download OHLCV data for a list of tickers from Yahoo Finance."""
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            # Newer yfinance returns MultiIndex columns like ('Close', 'AAPL') — flatten to 'Close'
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 60:
                data[ticker] = df
        except Exception as e:
            print(f"[WARN] Could not download {ticker}: {e}")
    return data


# ---------------------------------------------------------------------------
# 2. Technical Indicators
# ---------------------------------------------------------------------------

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA 10/20/30 and EMA 10/20/30."""
    close = df["Close"]
    df["SMA_10"] = close.rolling(10).mean()
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_30"] = close.rolling(30).mean()
    df["EMA_10"] = close.ewm(span=10, adjust=False).mean()
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA_30"] = close.ewm(span=30, adjust=False).mean()
    df["Price_to_SMA10"] = close / df["SMA_10"] - 1
    df["Price_to_SMA20"] = close / df["SMA_20"] - 1
    df["Price_to_SMA30"] = close / df["SMA_30"] - 1
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclically-encoded calendar features (sin/cos) to preserve circular topology."""
    dow   = df.index.dayofweek   # 0=Mon … 4=Fri
    month = df.index.month       # 1=Jan … 12=Dec
    df["DoW_sin"]   = np.sin(2 * np.pi * dow   / 5)
    df["DoW_cos"]   = np.cos(2 * np.pi * dow   / 5)
    df["Month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["Month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add return, volatility, and volume features."""
    close = df["Close"]
    df["Daily_Return"]  = close.pct_change()
    df["Log_Return"]    = np.log(close / close.shift(1))
    df["Volatility_10"] = df["Daily_Return"].rolling(10).std()
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_MA10"]   = df["Volume"].rolling(10).mean()
    df["Volume_Ratio"]  = df["Volume"] / df["Volume_MA10"]
    df["HL_Range"]      = (df["High"] - df["Low"]) / close
    return df


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI(14) and MACD(12, 26, 9) momentum indicators."""
    close = df["Close"]
    # RSI(14) — Wilder smoothing via ewm(com=13)
    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rs            = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"]  = (100 - (100 / (1 + rs))).fillna(50)  # 50 = neutral fill
    # MACD(12, 26, 9)
    ema12              = close.ewm(span=12, adjust=False).mean()
    ema26              = close.ewm(span=26, adjust=False).mean()
    df["MACD"]         = ema12 - ema26
    df["MACD_Signal"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]    = df["MACD"] - df["MACD_Signal"]
    return df


def add_market_context(
    df: pd.DataFrame,
    spy_returns: pd.Series,
    vix_levels: pd.Series,
) -> pd.DataFrame:
    """Merge market-wide SPY daily return and VIX closing level into the ticker DataFrame."""
    df["SPY_Return"] = spy_returns.reindex(df.index).ffill().fillna(0)
    df["VIX_Level"]  = vix_levels.reindex(df.index).ffill().fillna(20)  # 20 = historical median
    return df


# ---------------------------------------------------------------------------
# 3. Alternative Data (Insider & News Sentiment)
# ---------------------------------------------------------------------------

def fetch_insider_data(ticker: str) -> pd.DataFrame:
    """
    Fetch insider transaction data via yfinance (free, no API key).
    Returns daily Insider_Net_Shares and Insider_Buy_Flag.
    """
    try:
        raw = yf.Ticker(ticker).insider_transactions
        if raw is None or raw.empty:
            return pd.DataFrame()

        raw = raw.copy()
        if not isinstance(raw.index, pd.DatetimeIndex):
            date_col = next(
                (c for c in raw.columns if "date" in c.lower() or "start" in c.lower()), None
            )
            if date_col is None:
                return pd.DataFrame()
            raw.index = pd.to_datetime(raw[date_col])

        raw.index = pd.to_datetime(raw.index).normalize()
        shares_col = next((c for c in raw.columns if "shares" in c.lower()), None)
        if shares_col is None:
            return pd.DataFrame()

        raw[shares_col] = pd.to_numeric(raw[shares_col], errors="coerce").fillna(0)
        daily    = raw.groupby(raw.index)[shares_col].sum().rename("Insider_Net_Shares")
        buy_flag = (daily > 0).astype(int).rename("Insider_Buy_Flag")
        return pd.concat([daily, buy_flag], axis=1)

    except Exception as e:
        print(f"[WARN] Insider data failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_news_sentiment(ticker: str) -> pd.DataFrame:
    """
    Fetch recent news via yfinance and score headlines with VADER (local NLP, no API key).
    Returns daily News_Sentiment (mean compound) and News_Count.
    """
    try:
        news = yf.Ticker(ticker).news
        if not news:
            return pd.DataFrame()

        rows = []
        for item in news:
            publish_ts = item.get("providerPublishTime") or item.get("publishTime")
            title      = item.get("title") or item.get("headline", "")
            if publish_ts is None or not title:
                continue
            date  = pd.Timestamp(publish_ts, unit="s").normalize()
            score = _vader.polarity_scores(title)["compound"]
            rows.append({"date": date, "sentiment": score})

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("date")
        daily_sentiment = df.groupby(df.index)["sentiment"].mean().rename("News_Sentiment")
        daily_count     = df.groupby(df.index)["sentiment"].count().rename("News_Count")
        return pd.concat([daily_sentiment, daily_count], axis=1)

    except Exception as e:
        print(f"[WARN] News sentiment failed for {ticker}: {e}")
        return pd.DataFrame()


def add_alternative_data(df: pd.DataFrame, ticker: str, start: str = "", end: str = "") -> pd.DataFrame:
    """Merge insider and news sentiment into the main DataFrame, filling gaps with 0."""
    insider = fetch_insider_data(ticker)
    if not insider.empty:
        df = df.join(insider, how="left")
    else:
        df["Insider_Net_Shares"] = 0.0
        df["Insider_Buy_Flag"]   = 0

    news = fetch_news_sentiment(ticker)
    if not news.empty:
        df = df.join(news, how="left")
    else:
        df["News_Sentiment"] = 0.0
        df["News_Count"]     = 0

    df["Insider_Net_Shares"] = df["Insider_Net_Shares"].fillna(0)
    df["Insider_Buy_Flag"]   = df["Insider_Buy_Flag"].fillna(0)
    df["News_Sentiment"]     = df["News_Sentiment"].fillna(0)
    df["News_Count"]         = df["News_Count"].fillna(0)
    return df


# ---------------------------------------------------------------------------
# 4. Target Variables (all four horizons)
# ---------------------------------------------------------------------------

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute binary up/down targets, raw future returns, and next-close values
    for all four horizons.

    For horizon h (trading days):
      Future_Return_{h}d = (Close.shift(-h) / Close) - 1   raw h-day return
      Target_{h}d        = 1 if Future_Return > 0 else 0
      Next_Close_{h}d    = Close.shift(-h)                  used in Phase 2 only
    """
    close = df["Close"]
    for h in HORIZONS:
        label = f"{h}d"
        df[f"Future_Return_{label}"] = close.shift(-h) / close - 1
        df[f"Target_{label}"]        = (df[f"Future_Return_{label}"] > 0).astype(int)
        df[f"Next_Close_{label}"]    = close.shift(-h)
    return df


# ---------------------------------------------------------------------------
# 5. Full Feature Engineering Pipeline per Ticker
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "Close", "Volume",
    "SMA_10", "SMA_20", "SMA_30",
    "EMA_10", "EMA_20", "EMA_30",
    "Price_to_SMA10", "Price_to_SMA20", "Price_to_SMA30",
    "Daily_Return", "Log_Return", "Volatility_10",
    "Volume_Change", "Volume_Ratio", "HL_Range",
    "DoW_sin", "DoW_cos", "Month_sin", "Month_cos",
    "SPY_Return", "VIX_Level",
    "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
    "Insider_Net_Shares", "Insider_Buy_Flag",
    "News_Sentiment", "News_Count",
]


def build_features(
    df: pd.DataFrame,
    ticker: str,
    start: str = "",
    end: str = "",
    spy_returns: pd.Series | None = None,
    vix_levels: pd.Series | None = None,
) -> pd.DataFrame:
    """Apply the full feature engineering pipeline to a single ticker DataFrame."""
    df = df.copy()
    df = add_moving_averages(df)
    df = add_time_features(df)
    df = add_price_features(df)
    df = add_momentum_indicators(df)
    if spy_returns is not None and vix_levels is not None:
        df = add_market_context(df, spy_returns, vix_levels)
    df = add_alternative_data(df, ticker, start, end)
    df = add_target(df)
    df = df.dropna(subset=FEATURE_COLS)
    df = df.dropna(subset=TARGET_COLS)   # removes last 126 rows (worst-case horizon)
    return df


# ---------------------------------------------------------------------------
# 6. Sliding Window Construction
# ---------------------------------------------------------------------------

def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 20,
    include_target_in_features: bool = False,
    include_next_close: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Build sliding window arrays for all four horizons.

    Args:
        df: Feature DataFrame with Target_{h}d and Next_Close_{h}d columns.
        feature_cols: Base feature columns (never includes targets for Phase 3).
        window_size: Number of past days per sample.
        include_target_in_features: Phase 1 — leak all 4 Target columns into X.
        include_next_close: Phase 2 — leak all 4 Next_Close columns into X.

    Returns:
        X: shape (n_samples, window_size, n_features)
        y: dict {'out_1d': ..., 'out_5d': ..., 'out_21d': ..., 'out_126d': ...}
           each value shape (n_samples,)
    """
    cols = list(feature_cols)
    if include_target_in_features:
        cols = cols + TARGET_COLS
    if include_next_close:
        cols = cols + NEXT_CLOSE_COLS

    data    = df[cols].values
    targets = {f"out_{h}d": df[f"Target_{h}d"].values for h in HORIZONS}

    X_list  = []
    y_lists = {k: [] for k in targets}

    for i in range(window_size, len(data)):
        if include_target_in_features or include_next_close:
            # Shift window forward to include current row (so leaked column is visible)
            X_list.append(data[i - window_size + 1 : i + 1])
        else:
            X_list.append(data[i - window_size : i])
        for k, arr in targets.items():
            y_lists[k].append(arr[i])

    X = np.array(X_list, dtype=np.float32)
    y = {k: np.array(v, dtype=np.float32) for k, v in y_lists.items()}
    return X, y


# ---------------------------------------------------------------------------
# 7. Chronological Train / Test Split
# ---------------------------------------------------------------------------

def chronological_split(
    X: np.ndarray,
    y: dict[str, np.ndarray],
    train_ratio: float = 0.8,
    purge_horizon: int = 126,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Split arrays chronologically. Purges the last `purge_horizon` training samples
    to prevent label leakage: targets for those samples reference prices in the test period.
    The test set is unaffected.
    """
    split     = int(len(X) * train_ratio)
    purge_cut = max(0, split - purge_horizon)
    X_tr  = X[:purge_cut]
    X_te  = X[split:]
    y_tr  = {k: v[:purge_cut] for k, v in y.items()}
    y_te  = {k: v[split:]     for k, v in y.items()}
    return X_tr, X_te, y_tr, y_te


# ---------------------------------------------------------------------------
# 8. Multi-Ticker Dataset Builder
# ---------------------------------------------------------------------------

def build_dataset(
    tickers: list[str],
    start: str,
    end: str,
    window_size: int = 20,
    train_ratio: float = 0.8,
    phase: int = 3,
) -> dict:
    """
    Download data for all tickers, engineer features, build sliding windows,
    and return a consolidated train/test split.

    phase:
        1 — include all 4 Target columns in features (direct leak check)
        2 — include all 4 Next_Close columns in features (indirect check)
        3 — clean features only (real training)

    Returns dict with keys:
        X_train, X_test          : np.ndarray
        y_train, y_test          : dict[str, np.ndarray]  (4 horizon outputs)
        future_returns_test      : dict[str, np.ndarray]  (4 raw return arrays)
        scalers                  : dict[str, StandardScaler]
        dates_test               : list
        tickers_test             : list
    """
    all_X_train, all_X_test = [], []
    all_y_train = {f"out_{h}d": [] for h in HORIZONS}
    all_y_test  = {f"out_{h}d": [] for h in HORIZONS}
    all_fr_test = {f"Future_Return_{h}d": [] for h in HORIZONS}
    all_dates_test, all_tickers_test = [], []
    scalers: dict[str, StandardScaler] = {}

    raw_data = download_market_data(tickers, start, end)

    _spy = yf.download("SPY",  start=start, end=end, auto_adjust=True, progress=False)
    _vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(_spy.columns, pd.MultiIndex):
        _spy.columns = _spy.columns.get_level_values(0)
    if isinstance(_vix.columns, pd.MultiIndex):
        _vix.columns = _vix.columns.get_level_values(0)
    spy_returns = _spy["Close"].pct_change().rename("SPY_Return")
    vix_levels  = _vix["Close"].rename("VIX_Level")

    for ticker, raw_df in raw_data.items():
        try:
            df = build_features(raw_df, ticker, start, end, spy_returns=spy_returns, vix_levels=vix_levels)
            # Need enough rows after the 126-day horizon drops the tail
            if len(df) < window_size + 126 + 10:
                continue

            X, y = make_windows(
                df,
                FEATURE_COLS,
                window_size=window_size,
                include_target_in_features=(phase == 1),
                include_next_close=(phase == 2),
            )

            n = len(X)

            # Future returns aligned to window indices (row i → target date i)
            future_returns = {
                f"Future_Return_{h}d": df[f"Future_Return_{h}d"].values[window_size : window_size + n]
                for h in HORIZONS
            }

            X_tr, X_te, y_tr, y_te = chronological_split(X, y, train_ratio)
            split = int(n * train_ratio)
            fr_te = {k: v[split:] for k, v in future_returns.items()}

            # Scale features — only X, never targets
            n_features  = X_tr.shape[2]
            scaler      = StandardScaler()
            X_tr_flat   = X_tr.reshape(-1, n_features)
            scaler.fit(X_tr_flat)
            X_tr = scaler.transform(X_tr_flat).reshape(X_tr.shape)
            X_te = scaler.transform(X_te.reshape(-1, n_features)).reshape(X_te.shape)
            scalers[ticker] = scaler

            # Dates for test windows
            ticker_dates = df.index[window_size : window_size + n][split:]
            ticker_dates = ticker_dates[: len(y_te["out_1d"])]

            all_X_train.append(X_tr)
            all_X_test.append(X_te)
            for k in all_y_train:
                all_y_train[k].append(y_tr[k])
                all_y_test[k].append(y_te[k])
            for k in all_fr_test:
                all_fr_test[k].append(fr_te[k])
            all_dates_test.extend(ticker_dates)
            all_tickers_test.extend([ticker] * len(y_te["out_1d"]))

        except Exception as e:
            print(f"[WARN] Skipping {ticker}: {e}")
            continue

    if not all_X_train:
        raise RuntimeError("No usable tickers after feature engineering.")

    return {
        "X_train":            np.concatenate(all_X_train, axis=0),
        "X_test":             np.concatenate(all_X_test,  axis=0),
        "y_train":            {k: np.concatenate(v) for k, v in all_y_train.items()},
        "y_test":             {k: np.concatenate(v) for k, v in all_y_test.items()},
        "future_returns_test": {k: np.concatenate(v) for k, v in all_fr_test.items()},
        "scalers":            scalers,
        "dates_test":         all_dates_test,
        "tickers_test":       all_tickers_test,
    }
