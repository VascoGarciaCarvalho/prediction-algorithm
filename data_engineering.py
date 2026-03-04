"""
Data Engineering Module
Handles feature engineering, alternative data, preprocessing, and data splitting.

Alternative data sources (all free, no API key required):
  - Insider transactions: yfinance Ticker.insider_transactions
  - News sentiment:       yfinance Ticker.news + VADER (vaderSentiment, local NLP)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings("ignore")

_vader = SentimentIntensityAnalyzer()


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
    # Price relative to moving averages (normalised signals)
    df["Price_to_SMA10"] = close / df["SMA_10"] - 1
    df["Price_to_SMA20"] = close / df["SMA_20"] - 1
    df["Price_to_SMA30"] = close / df["SMA_30"] - 1
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar / time-based features."""
    df["Day_of_Month"] = df.index.day
    df["Day_of_Week"] = df.index.dayofweek
    df["Month"] = df.index.month
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add return, volatility, and volume features."""
    close = df["Close"]
    df["Daily_Return"] = close.pct_change()
    df["Log_Return"] = np.log(close / close.shift(1))
    df["Volatility_10"] = df["Daily_Return"].rolling(10).std()
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_MA10"] = df["Volume"].rolling(10).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA10"]
    # High-Low range
    df["HL_Range"] = (df["High"] - df["Low"]) / close
    return df


# ---------------------------------------------------------------------------
# 3. Alternative Data (Insider & News Sentiment)
# ---------------------------------------------------------------------------

def fetch_insider_data(ticker: str) -> pd.DataFrame:
    """
    Fetch insider transaction data via yfinance (free, no API key).

    yfinance returns a DataFrame with columns including 'Shares', 'Transaction',
    and a date index. We aggregate net shares per day and flag net-buying days.
    """
    try:
        tkr = yf.Ticker(ticker)
        raw = tkr.insider_transactions
        if raw is None or raw.empty:
            return pd.DataFrame()

        raw = raw.copy()
        # Normalise the date index
        if not isinstance(raw.index, pd.DatetimeIndex):
            # Older yfinance versions store date in a column called 'Start Date' or 'Date'
            date_col = next(
                (c for c in raw.columns if "date" in c.lower() or "start" in c.lower()),
                None,
            )
            if date_col is None:
                return pd.DataFrame()
            raw.index = pd.to_datetime(raw[date_col])

        raw.index = pd.to_datetime(raw.index).normalize()

        # Net shares: positive values = purchase, negative = sale
        shares_col = next((c for c in raw.columns if "shares" in c.lower()), None)
        if shares_col is None:
            return pd.DataFrame()

        raw[shares_col] = pd.to_numeric(raw[shares_col], errors="coerce").fillna(0)
        daily = raw.groupby(raw.index)[shares_col].sum().rename("Insider_Net_Shares")
        buy_flag = (daily > 0).astype(int).rename("Insider_Buy_Flag")
        return pd.concat([daily, buy_flag], axis=1)

    except Exception as e:
        print(f"[WARN] Insider data failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_news_sentiment(ticker: str) -> pd.DataFrame:
    """
    Fetch recent news via yfinance and score each headline with VADER (local NLP).

    yfinance returns up to ~20 recent news items per ticker.
    VADER compound score: +1 = most positive, -1 = most negative.
    Returns daily: News_Sentiment (mean compound), News_Count (article count).
    """
    try:
        tkr = yf.Ticker(ticker)
        news = tkr.news
        if not news:
            return pd.DataFrame()

        rows = []
        for item in news:
            # yfinance news dict keys vary by version; handle both
            publish_ts = item.get("providerPublishTime") or item.get("publishTime")
            title = item.get("title") or item.get("headline", "")
            if publish_ts is None or not title:
                continue
            date = pd.Timestamp(publish_ts, unit="s").normalize()
            score = _vader.polarity_scores(title)["compound"]
            rows.append({"date": date, "sentiment": score})

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("date")
        daily_sentiment = df.groupby(df.index)["sentiment"].mean().rename("News_Sentiment")
        daily_count = df.groupby(df.index)["sentiment"].count().rename("News_Count")
        return pd.concat([daily_sentiment, daily_count], axis=1)

    except Exception as e:
        print(f"[WARN] News sentiment failed for {ticker}: {e}")
        return pd.DataFrame()


def add_alternative_data(df: pd.DataFrame, ticker: str, start: str, end: str) -> pd.DataFrame:
    """Merge insider and news sentiment into the main DataFrame, filling gaps with 0."""
    insider = fetch_insider_data(ticker)
    if not insider.empty:
        df = df.join(insider, how="left")
    else:
        df["Insider_Net_Shares"] = 0.0
        df["Insider_Buy_Flag"] = 0

    news = fetch_news_sentiment(ticker)
    if not news.empty:
        df = df.join(news, how="left")
    else:
        df["News_Sentiment"] = 0.0
        df["News_Count"] = 0

    df["Insider_Net_Shares"] = df["Insider_Net_Shares"].fillna(0)
    df["Insider_Buy_Flag"] = df["Insider_Buy_Flag"].fillna(0)
    df["News_Sentiment"] = df["News_Sentiment"].fillna(0)
    df["News_Count"] = df["News_Count"].fillna(0)
    return df


# ---------------------------------------------------------------------------
# 4. Target Variable
# ---------------------------------------------------------------------------

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary target: 1 if next day's return > 0, else 0.
    Also stores the next-day close (for Phase 2 indirect check).
    """
    df["Next_Return"] = df["Close"].pct_change().shift(-1)
    df["Target"] = (df["Next_Return"] > 0).astype(int)
    df["Next_Close"] = df["Close"].shift(-1)  # used only in Phase 2
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
    "Day_of_Month", "Day_of_Week", "Month",
    "Insider_Net_Shares", "Insider_Buy_Flag",
    "News_Sentiment", "News_Count",
]


def build_features(df: pd.DataFrame, ticker: str, start: str = "", end: str = "") -> pd.DataFrame:
    """Apply the full feature engineering pipeline to a single ticker DataFrame."""
    df = df.copy()
    df = add_moving_averages(df)
    df = add_time_features(df)
    df = add_price_features(df)
    df = add_alternative_data(df, ticker, start, end)
    df = add_target(df)
    df = df.dropna(subset=FEATURE_COLS)
    df = df.dropna(subset=["Target"])
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) sliding window arrays.

    Args:
        df: Feature DataFrame with 'Target' and 'Next_Close' columns.
        feature_cols: Base feature columns to include.
        window_size: Number of past days per sample.
        include_target_in_features: Phase 1 sanity check — leak target into X.
        include_next_close: Phase 2 sanity check — include next day's close in X.

    Returns:
        X: shape (n_samples, window_size, n_features)
        y: shape (n_samples,)
    """
    cols = list(feature_cols)
    if include_target_in_features:
        cols = cols + ["Target"]
    if include_next_close:
        cols = cols + ["Next_Close"]

    data = df[cols].values
    targets = df["Target"].values

    X, y = [], []
    for i in range(window_size, len(data)):
        if include_target_in_features or include_next_close:
            # Shift window to include row i so the leaked column is visible to the model
            X.append(data[i - window_size + 1 : i + 1])
        else:
            X.append(data[i - window_size : i])
        y.append(targets[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# 7. Chronological Train / Test Split
# ---------------------------------------------------------------------------

def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays chronologically (no shuffling)."""
    split = int(len(X) * train_ratio)
    return X[:split], X[split:], y[:split], y[split:]


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
        1 — include Target in features (direct leak check)
        2 — include Next_Close in features (indirect check)
        3 — clean features only (real training)

    Returns a dict with keys: X_train, X_test, y_train, y_test, scalers, dates_test, tickers_test
    """
    all_X_train, all_y_train = [], []
    all_X_test, all_y_test = [], []
    all_dates_test, all_tickers_test = [], []
    scalers: dict[str, StandardScaler] = {}

    raw_data = download_market_data(tickers, start, end)

    for ticker, raw_df in raw_data.items():
        try:
            df = build_features(raw_df, ticker, start, end)
            if len(df) < window_size + 10:
                continue

            include_target = (phase == 1)
            include_next_close = (phase == 2)

            X, y = make_windows(
                df,
                FEATURE_COLS,
                window_size=window_size,
                include_target_in_features=include_target,
                include_next_close=include_next_close,
            )

            X_tr, X_te, y_tr, y_te = chronological_split(X, y, train_ratio)

            # Fit scaler on training data, apply to both splits
            n_features = X_tr.shape[2]
            scaler = StandardScaler()
            X_tr_flat = X_tr.reshape(-1, n_features)
            scaler.fit(X_tr_flat)
            X_tr = scaler.transform(X_tr_flat).reshape(X_tr.shape)
            X_te = scaler.transform(X_te.reshape(-1, n_features)).reshape(X_te.shape)
            scalers[ticker] = scaler

            # Dates corresponding to test windows (the target date, i.e. row after window)
            test_start_idx = int(len(df) * train_ratio) + window_size
            # Safe slice — align with actual y_te length
            ticker_dates = df.index[window_size:][int(len(df[window_size:]) * train_ratio):]
            ticker_dates = ticker_dates[: len(y_te)]

            all_X_train.append(X_tr)
            all_y_train.append(y_tr)
            all_X_test.append(X_te)
            all_y_test.append(y_te)
            all_dates_test.extend(ticker_dates)
            all_tickers_test.extend([ticker] * len(y_te))

        except Exception as e:
            print(f"[WARN] Skipping {ticker}: {e}")
            continue

    if not all_X_train:
        raise RuntimeError("No usable tickers after feature engineering.")

    return {
        "X_train": np.concatenate(all_X_train, axis=0),
        "y_train": np.concatenate(all_y_train, axis=0),
        "X_test": np.concatenate(all_X_test, axis=0),
        "y_test": np.concatenate(all_y_test, axis=0),
        "scalers": scalers,
        "dates_test": all_dates_test,
        "tickers_test": all_tickers_test,
    }
