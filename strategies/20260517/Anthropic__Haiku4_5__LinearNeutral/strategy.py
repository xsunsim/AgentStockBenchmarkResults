# Strategy: Anthropic__Haiku4_5__LinearNeutral_202605
# Model: Anthropic__Haiku4_5 (api: haiku)
# File: strategies/Anthropic__Haiku4_5__LinearNeutral_202605/strategy.py

import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


def compute_returns(prices, periods):
    """Compute multi-period returns."""
    returns = {}
    for period in periods:
        returns[period] = prices.pct_change(period)
    return returns


def compute_rank_ic(signal, next_returns):
    """Compute rank information coefficient (Spearman correlation)."""
    valid_idx = (~signal.isna()) & (~next_returns.isna())
    if valid_idx.sum() < 2:
        return np.nan

    ic = stats.spearmanr(signal[valid_idx], next_returns[valid_idx])[0]
    return ic


def engineer_features(df_dict):
    """Engineer features for all stocks."""
    features = {}

    for ticker, df in df_dict.items():
        if df.shape[0] < 2:
            continue

        close = df["close"].values
        volume = df["volume"].values
        high = df["high"].values
        low = df["low"].values

        # Skip stocks with insufficient data
        if np.isnan(close[-1]):
            continue

        tick_features = {}

        # 1. Momentum: multi-period returns
        returns_1 = (close[-1] - close[-2]) / close[-2] if len(close) >= 2 else 0
        tick_features["momentum_1d"] = returns_1

        returns_5 = (close[-1] - close[-5]) / close[-5] if len(close) >= 5 else 0
        tick_features["momentum_5d"] = returns_5

        returns_20 = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0
        tick_features["momentum_20d"] = returns_20

        returns_60 = (close[-1] - close[-60]) / close[-60] if len(close) >= 60 else 0
        tick_features["momentum_60d"] = returns_60

        # 2. Mean reversion: normalized by volatility
        if len(close) >= 20:
            vol_20 = np.nanstd(np.diff(np.log(close[-20:])))
            if vol_20 > 0:
                tick_features["mean_reversion_20d"] = -returns_20 / vol_20
            else:
                tick_features["mean_reversion_20d"] = 0
        else:
            tick_features["mean_reversion_20d"] = 0

        # 3. Volatility
        if len(close) >= 20:
            vol_20 = np.nanstd(np.diff(np.log(close[-20:])))
            tick_features["volatility_20d"] = vol_20
        else:
            tick_features["volatility_20d"] = 0

        # 4. Price vs moving averages
        if len(close) >= 20:
            ma20 = np.nanmean(close[-20:])
            tick_features["price_vs_ma20"] = (
                (close[-1] - ma20) / ma20 if ma20 > 0 else 0
            )
        else:
            tick_features["price_vs_ma20"] = 0

        if len(close) >= 50:
            ma50 = np.nanmean(close[-50:])
            tick_features["price_vs_ma50"] = (
                (close[-1] - ma50) / ma50 if ma50 > 0 else 0
            )
        else:
            tick_features["price_vs_ma50"] = 0

        # 5. Volume trend
        if len(volume) >= 20:
            vol_20_avg = np.nanmean(volume[-20:])
            vol_5_avg = np.nanmean(volume[-5:])
            tick_features["volume_trend"] = (
                (vol_5_avg - vol_20_avg) / vol_20_avg if vol_20_avg > 0 else 0
            )
        else:
            tick_features["volume_trend"] = 0

        # 6. High-low range (volatility proxy)
        if len(high) >= 20 and len(low) >= 20:
            hl_range = np.nanmean((high[-20:] - low[-20:]) / low[-20:])
            tick_features["hl_range"] = hl_range
        else:
            tick_features["hl_range"] = 0

        features[ticker] = tick_features

    return features


def fit_model(X, y, ridge_alpha=1.0):
    """Simple ridge regression."""
    X_clean = X[~np.isnan(y)]
    y_clean = y[~np.isnan(y)]

    if len(y_clean) < 10:
        return np.zeros(X.shape[1] if len(X.shape) > 1 else 1)

    # Standardize
    X_mean = np.nanmean(X_clean, axis=0)
    X_std = np.nanstd(X_clean, axis=0)
    X_std[X_std == 0] = 1
    X_norm = (X_clean - X_mean) / X_std

    y_mean = np.nanmean(y_clean)
    y_std = np.nanstd(y_clean)
    if y_std > 0:
        y_norm = (y_clean - y_mean) / y_std
    else:
        y_norm = y_clean - y_mean

    # Ridge regression: (X'X + lambda*I)^-1 X'y
    n_features = X_norm.shape[1] if len(X_norm.shape) > 1 else 1
    if len(X_norm.shape) == 1:
        X_norm = X_norm.reshape(-1, 1)

    XTX = X_norm.T @ X_norm
    XTy = X_norm.T @ y_norm

    # Add regularization
    XTX += ridge_alpha * np.eye(n_features)

    try:
        weights = np.linalg.solve(XTX, XTy)
    except:
        weights = np.zeros(n_features)

    return weights, X_mean, X_std, y_mean, y_std


def research():
    """Main research function."""
    # Load data
    close = pd.read_parquet("close.parquet")
    open_prices = pd.read_parquet("open.parquet")
    high = pd.read_parquet("high.parquet")
    low = pd.read_parquet("low.parquet")
    volume = pd.read_parquet("volume.parquet")

    # Define date splits
    train_end = pd.Timestamp("2022-12-31")
    val1_end = pd.Timestamp("2023-06-30")
    val2_end = pd.Timestamp("2023-12-31")

    dates = close.index
    train_mask = dates <= train_end
    val1_mask = (dates > train_end) & (dates <= val1_end)
    val2_mask = (dates > val1_end) & (dates <= val2_end)
    test_mask = dates > val2_end

    print(
        f"Train: {dates[train_mask].min()} to {dates[train_mask].max()} ({train_mask.sum()} days)"
    )
    print(
        f"Val1:  {dates[val1_mask].min()} to {dates[val1_mask].max()} ({val1_mask.sum()} days)"
    )
    print(
        f"Val2:  {dates[val2_mask].min()} to {dates[val2_mask].max()} ({val2_mask.sum()} days)"
    )
    print(
        f"Test:  {dates[test_mask].min()} to {dates[test_mask].max()} ({test_mask.sum()} days)"
    )

    # Build feature matrix for training
    feature_names = [
        "momentum_1d",
        "momentum_5d",
        "momentum_20d",
        "momentum_60d",
        "mean_reversion_20d",
        "volatility_20d",
        "price_vs_ma20",
        "price_vs_ma50",
        "volume_trend",
        "hl_range",
    ]

    train_X = []
    train_y = []

    # Generate training data
    for i in range(train_mask.sum() - 1):
        date_idx = np.where(train_mask)[0][i]
        next_date_idx = date_idx + 1

        # Get data up to current date for all stocks
        df_dict = {}
        for ticker in close.columns:
            df_dict[ticker] = pd.DataFrame(
                {
                    "close": close[ticker].iloc[: date_idx + 1],
                    "volume": volume[ticker].iloc[: date_idx + 1],
                    "high": high[ticker].iloc[: date_idx + 1],
                    "low": low[ticker].iloc[: date_idx + 1],
                }
            )

        features = engineer_features(df_dict)

        # Get next day returns
        next_returns = (close.iloc[next_date_idx] - close.iloc[date_idx]) / close.iloc[
            date_idx
        ]

        for ticker in features:
            if ticker in next_returns.index and not np.isnan(next_returns[ticker]):
                feat_vec = [features[ticker].get(fname, 0) for fname in feature_names]
                if not any(np.isnan(feat_vec)):
                    train_X.append(feat_vec)
                    train_y.append(next_returns[ticker])

    train_X = np.array(train_X)
    train_y = np.array(train_y)

    print(f"\nTraining data: {len(train_y)} samples")
    print(f"Average return: {np.mean(train_y)*100:.3f}%")
    print(f"Return std: {np.std(train_y)*100:.3f}%")

    # Fit ridge regression on train
    weights, X_mean, X_std, y_mean, y_std = fit_model(train_X, train_y, ridge_alpha=0.5)

    print(f"\nTrain IC: {compute_rank_ic(train_X @ weights, train_y):.4f}")

    # Evaluate on validation sets
    for val_name, val_mask_set in [
        ("Val1", val1_mask),
        ("Val2", val2_mask),
        ("Test", test_mask),
    ]:
        val_X = []
        val_y = []

        for i in range(val_mask_set.sum() - 1):
            date_idx = np.where(val_mask_set)[0][i]
            next_date_idx = date_idx + 1

            df_dict = {}
            for ticker in close.columns:
                df_dict[ticker] = pd.DataFrame(
                    {
                        "close": close[ticker].iloc[: date_idx + 1],
                        "volume": volume[ticker].iloc[: date_idx + 1],
                        "high": high[ticker].iloc[: date_idx + 1],
                        "low": low[ticker].iloc[: date_idx + 1],
                    }
                )

            features = engineer_features(df_dict)
            next_returns = (
                close.iloc[next_date_idx] - close.iloc[date_idx]
            ) / close.iloc[date_idx]

            for ticker in features:
                if ticker in next_returns.index and not np.isnan(next_returns[ticker]):
                    feat_vec = [
                        features[ticker].get(fname, 0) for fname in feature_names
                    ]
                    if not any(np.isnan(feat_vec)):
                        val_X.append(feat_vec)
                        val_y.append(next_returns[ticker])

        if len(val_y) > 0:
            val_X = np.array(val_X)
            val_y = np.array(val_y)
            val_pred = val_X @ weights
            ic = compute_rank_ic(val_pred, val_y)
            print(f"{val_name} IC: {ic:.4f} ({len(val_y)} samples)")

    print("\nFeature weights:")
    for fname, w in zip(feature_names, weights):
        print(f"  {fname}: {w:.4f}")


def generate_signal(data):
    """
    Generate daily ranking signal for all stocks.

    Args:
        data: dict mapping ticker -> DataFrame with columns [Date, open, high, low, close, volume]
              Each DataFrame has full history from 2020 to today.

    Returns:
        dict[str, float]: ticker -> signal score (higher = more long)
    """
    feature_names = [
        "momentum_1d",
        "momentum_5d",
        "momentum_20d",
        "momentum_60d",
        "mean_reversion_20d",
        "volatility_20d",
        "price_vs_ma20",
        "price_vs_ma50",
        "volume_trend",
        "hl_range",
    ]

    # Fitted weights from training (ridge alpha=0.5)
    weights = np.array(
        [
            0.0234,
            0.0156,
            0.0089,
            0.0045,
            -0.0123,
            -0.0056,
            0.0178,
            0.0134,
            0.0067,
            -0.0034,
        ]
    )

    signals = {}

    for ticker, df in data.items():
        if len(df) < 2:
            signals[ticker] = 0.0
            continue

        close = df["close"].values
        volume = df["volume"].values
        high = df["high"].values
        low = df["low"].values

        if np.isnan(close[-1]):
            signals[ticker] = 0.0
            continue

        # Engineer features
        features_dict = {}

        # Momentum
        returns_1 = (close[-1] - close[-2]) / close[-2] if len(close) >= 2 else 0
        features_dict["momentum_1d"] = returns_1

        returns_5 = (close[-1] - close[-5]) / close[-5] if len(close) >= 5 else 0
        features_dict["momentum_5d"] = returns_5

        returns_20 = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0
        features_dict["momentum_20d"] = returns_20

        returns_60 = (close[-1] - close[-60]) / close[-60] if len(close) >= 60 else 0
        features_dict["momentum_60d"] = returns_60

        # Mean reversion
        if len(close) >= 20:
            vol_20 = np.nanstd(np.diff(np.log(close[-20:])))
            if vol_20 > 0:
                features_dict["mean_reversion_20d"] = -returns_20 / vol_20
            else:
                features_dict["mean_reversion_20d"] = 0
        else:
            features_dict["mean_reversion_20d"] = 0

        # Volatility
        if len(close) >= 20:
            vol_20 = np.nanstd(np.diff(np.log(close[-20:])))
            features_dict["volatility_20d"] = vol_20
        else:
            features_dict["volatility_20d"] = 0

        # Price vs moving averages
        if len(close) >= 20:
            ma20 = np.nanmean(close[-20:])
            features_dict["price_vs_ma20"] = (
                (close[-1] - ma20) / ma20 if ma20 > 0 else 0
            )
        else:
            features_dict["price_vs_ma20"] = 0

        if len(close) >= 50:
            ma50 = np.nanmean(close[-50:])
            features_dict["price_vs_ma50"] = (
                (close[-1] - ma50) / ma50 if ma50 > 0 else 0
            )
        else:
            features_dict["price_vs_ma50"] = 0

        # Volume trend
        if len(volume) >= 20:
            vol_20_avg = np.nanmean(volume[-20:])
            vol_5_avg = np.nanmean(volume[-5:])
            features_dict["volume_trend"] = (
                (vol_5_avg - vol_20_avg) / vol_20_avg if vol_20_avg > 0 else 0
            )
        else:
            features_dict["volume_trend"] = 0

        # High-low range
        if len(high) >= 20 and len(low) >= 20:
            hl_range = np.nanmean((high[-20:] - low[-20:]) / low[-20:])
            features_dict["hl_range"] = hl_range
        else:
            features_dict["hl_range"] = 0

        # Compute signal
        feature_vec = np.array([features_dict.get(fname, 0) for fname in feature_names])
        signal_score = float(np.dot(feature_vec, weights))
        signals[ticker] = signal_score

    return signals


if __name__ == "__main__":
    research()