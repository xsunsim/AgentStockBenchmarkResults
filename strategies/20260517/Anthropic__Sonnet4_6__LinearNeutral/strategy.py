# Strategy: Anthropic__Sonnet4_6__LinearNeutral_202605
# Model: Anthropic__Sonnet4_6 (api: sonnet)
# File: strategies/Anthropic__Sonnet4_6__LinearNeutral_202605/strategy.py

import pandas as pd
import numpy as np
from scipy.stats import rankdata

# ============================================================================
# RESEARCH CODE: Load Data and Engineer Features
# ============================================================================


def load_data():
    """Load all OHLCV data"""
    close = pd.read_parquet("close.parquet")
    open_px = pd.read_parquet("open.parquet")
    high = pd.read_parquet("high.parquet")
    low = pd.read_parquet("low.parquet")
    volume = pd.read_parquet("volume.parquet")
    return close, open_px, high, low, volume


def compute_returns(prices, horizon=1):
    """Compute forward returns"""
    return prices.shift(-horizon) / prices - 1


def compute_momentum(prices, window):
    """Momentum: return over window days"""
    return prices / prices.shift(window) - 1


def compute_vol_normalized_reversion(prices, lookback=20):
    """Mean reversion normalized by volatility"""
    ret = prices.pct_change(fill_method=None)
    vol = ret.rolling(lookback, min_periods=max(1, lookback // 2)).std()
    recent_ret = prices / prices.shift(5) - 1
    return -recent_ret / (vol + 1e-8)


def compute_volume_signal(volume, close, window=20):
    """Volume relative to moving average"""
    vol_ma = volume.rolling(window, min_periods=max(1, window // 2)).mean()
    return volume / (vol_ma + 1e-8) - 1


def compute_rank_ic(signal, forward_ret):
    """Compute cross-sectional rank IC per day, then mean"""
    ics = []
    for date in signal.index:
        sig = signal.loc[date].values
        ret = forward_ret.loc[date].values

        mask = ~(np.isnan(sig) | np.isnan(ret) | np.isinf(sig) | np.isinf(ret))
        if mask.sum() < 20:
            continue

        sig_valid = sig[mask]
        ret_valid = ret[mask]

        rank_sig = rankdata(sig_valid)
        rank_ret = rankdata(ret_valid)

        ic = np.corrcoef(rank_sig, rank_ret)[0, 1]
        if not np.isnan(ic):
            ics.append(ic)

    return np.mean(ics) if ics else 0.0


def cross_sectional_rank(df):
    """Rank normalize each row (date) to [-1, 1]"""
    result = df.copy()
    for date in df.index:
        values = df.loc[date].values
        mask = ~(np.isnan(values) | np.isinf(values))
        if mask.sum() > 1:
            ranked = np.full_like(values, np.nan)
            valid_values = values[mask]
            ranks = rankdata(valid_values)
            ranked[mask] = (
                2 * (ranks - 1) / (len(ranks) - 1) - 1 if len(ranks) > 1 else 0
            )
            result.loc[date] = ranked
    return result


def winsorize(df, lower=0.01, upper=0.99):
    """Winsorize at percentiles per date"""
    result = df.copy()
    for date in df.index:
        values = df.loc[date].values
        mask = ~(np.isnan(values) | np.isinf(values))
        if mask.sum() > 10:
            valid = values[mask]
            lo, hi = np.percentile(valid, [lower * 100, upper * 100])
            clipped = np.clip(values, lo, hi)
            result.loc[date] = clipped
    return result


# ============================================================================
# MAIN RESEARCH EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    close, open_px, high, low, volume = load_data()

    print(f"Data shape: {close.shape}")
    print(f"Date range: {close.index[0]} to {close.index[-1]}")
    print(f"Tickers: {len(close.columns)}")
    print()

    # Forward returns (1-day)
    fwd_ret = compute_returns(close, 1)

    # ========================================================================
    # FEATURE ENGINEERING
    # ========================================================================
    print("=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)

    features = {}

    # Short-term momentum (tends to reverse)
    features["mom_5d"] = compute_momentum(close, 5)
    features["mom_10d"] = compute_momentum(close, 10)

    # Medium-term momentum
    features["mom_20d"] = compute_momentum(close, 20)
    features["mom_40d"] = compute_momentum(close, 40)
    features["mom_60d"] = compute_momentum(close, 60)

    # Mean reversion signals
    features["reversion_10d"] = compute_vol_normalized_reversion(close, 10)
    features["reversion_20d"] = compute_vol_normalized_reversion(close, 20)
    features["reversion_30d"] = compute_vol_normalized_reversion(close, 30)

    # Volume signals
    features["vol_signal_10d"] = compute_volume_signal(volume, close, 10)
    features["vol_signal_20d"] = compute_volume_signal(volume, close, 20)

    print(f"Engineered {len(features)} features")
    print()

    # ========================================================================
    # TRAIN/VAL/TEST SPLIT
    # ========================================================================

    train_end = "2022-12-31"
    val1_end = "2023-06-30"
    val2_end = "2023-12-31"

    train_mask = close.index <= train_end
    val1_mask = (close.index > train_end) & (close.index <= val1_end)
    val2_mask = (close.index > val1_end) & (close.index <= val2_end)
    test_mask = close.index > val2_end

    print("=" * 80)
    print("DATA SPLITS")
    print("=" * 80)
    print(
        f"Train: {train_mask.sum()} days ({close.index[train_mask][0]} to {close.index[train_mask][-1]})"
    )
    print(
        f"Val1:  {val1_mask.sum()} days ({close.index[val1_mask][0]} to {close.index[val1_mask][-1]})"
    )
    print(
        f"Val2:  {val2_mask.sum()} days ({close.index[val2_mask][0]} to {close.index[val2_mask][-1]})"
    )
    print(
        f"Test:  {test_mask.sum()} days ({close.index[test_mask][0]} to {close.index[test_mask][-1]})"
    )
    print()

    # ========================================================================
    # FEATURE EVALUATION (TRAIN + VAL1)
    # ========================================================================
    print("=" * 80)
    print("FEATURE EVALUATION")
    print("=" * 80)

    feature_results = []
    for name, feat in features.items():
        ic_train = compute_rank_ic(feat[train_mask], fwd_ret[train_mask])
        ic_val1 = compute_rank_ic(feat[val1_mask], fwd_ret[val1_mask])
        feature_results.append((name, ic_train, ic_val1))
        print(f"{name:20s}: Train IC = {ic_train:+.4f}, Val1 IC = {ic_val1:+.4f}")

    print()

    # ========================================================================
    # FEATURE SELECTION
    # ========================================================================
    print("=" * 80)
    print("FEATURE SELECTION (Consistent performers)")
    print("=" * 80)

    # Select features where IC has same sign in train and val1, and |val1_IC| > 0.005
    selected_features = {}
    for name, ic_train, ic_val1 in feature_results:
        if np.sign(ic_train) == np.sign(ic_val1) and abs(ic_val1) > 0.005:
            selected_features[name] = (ic_train, ic_val1)
            print(f"{name:20s}: Train IC = {ic_train:+.4f}, Val1 IC = {ic_val1:+.4f} ✓")

    print(f"\nSelected {len(selected_features)} features")
    print()

    # ========================================================================
    # BUILD MULTI-FACTOR MODEL (Simple weighted combination)
    # ========================================================================
    print("=" * 80)
    print("MULTI-FACTOR MODEL")
    print("=" * 80)

    # Use validation IC as weights (features with higher IC get more weight)
    feature_weights = {}
    for name, (ic_train, ic_val1) in selected_features.items():
        # Weight by validation IC, normalized
        feature_weights[name] = ic_val1

    # Normalize weights
    total_abs_weight = sum(abs(w) for w in feature_weights.values())
    if total_abs_weight > 0:
        for name in feature_weights:
            feature_weights[name] /= total_abs_weight

    print("Feature weights (normalized by Val1 IC):")
    for name, weight in sorted(
        feature_weights.items(), key=lambda x: abs(x[1]), reverse=True
    ):
        print(f"  {name:20s}: {weight:+.4f}")
    print()

    # ========================================================================
    # VALIDATION 2: MODEL PERFORMANCE
    # ========================================================================
    print("=" * 80)
    print("VALIDATION 2: COMBINED MODEL PERFORMANCE")
    print("=" * 80)

    def compute_combined_signal(feature_dict, feature_weights, mask):
        """Compute weighted combination of rank-normalized features"""
        signals = []
        for name, weight in feature_weights.items():
            feat = feature_dict[name][mask]
            feat_winsorized = winsorize(feat)
            feat_ranked = cross_sectional_rank(feat_winsorized)
            signals.append(feat_ranked * weight)

        combined = sum(signals)
        return cross_sectional_rank(combined)

    signal_val2 = compute_combined_signal(features, feature_weights, val2_mask)
    ic_val2 = compute_rank_ic(signal_val2, fwd_ret[val2_mask])

    print(f"Val2 Rank IC: {ic_val2:+.4f}")
    print()

    # ========================================================================
    # FINAL TEST PERFORMANCE (ONE TIME ONLY)
    # ========================================================================
    print("=" * 80)
    print("FINAL TEST PERFORMANCE")
    print("=" * 80)

    signal_test = compute_combined_signal(features, feature_weights, test_mask)
    ic_test = compute_rank_ic(signal_test, fwd_ret[test_mask])

    print(f"Test Rank IC: {ic_test:+.4f}")
    print()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Selected features: {len(feature_weights)}")
    print(f"Val2 IC: {ic_val2:+.4f}")
    print(f"Test IC: {ic_test:+.4f}")
    print()
    print("Final feature weights:")
    for name, weight in sorted(
        feature_weights.items(), key=lambda x: abs(x[1]), reverse=True
    ):
        print(f"  {name:20s}: {weight:+.4f}")
    print()

    # Save weights for production
    import json

    weights_dict = {name: float(weight) for name, weight in feature_weights.items()}
    print("Production weights dictionary:")
    print(json.dumps(weights_dict, indent=2))
    print()


# ============================================================================
# PRODUCTION SIGNAL FUNCTION
# ============================================================================

# Fitted weights from research
FEATURE_WEIGHTS = {
    "mom_5d": -0.18945305540661192,
    "mom_60d": -0.09868820707822253,
    "reversion_10d": 0.14044729922587376,
    "reversion_20d": 0.14310717903585443,
    "reversion_30d": 0.13700216751756786,
    "vol_signal_10d": 0.1560987772485294,
    "vol_signal_20d": 0.1352033144873401,
}


def generate_signal(data):
    """
    Production signal function.

    Args:
        data: dict mapping ticker (str) -> DataFrame with columns [Date, open, high, low, close, volume]
              Each DataFrame contains ALL history up to today.

    Returns:
        dict[str, float]: ticker -> signal score (higher = more long)
    """
    if not data:
        return {}

    # Convert to panel format
    tickers = sorted(data.keys())

    # Build price and volume DataFrames
    close_data = {}
    volume_data = {}

    for ticker in tickers:
        df = data[ticker].copy()
        df = df.set_index("Date") if "Date" in df.columns else df
        df.index = pd.to_datetime(df.index)

        close_data[ticker] = df["close"]
        volume_data[ticker] = df["volume"]

    close_df = pd.DataFrame(close_data)
    volume_df = pd.DataFrame(volume_data)

    # Compute all features
    features_dict = {}

    # Momentum features
    features_dict["mom_5d"] = close_df / close_df.shift(5) - 1
    features_dict["mom_10d"] = close_df / close_df.shift(10) - 1
    features_dict["mom_20d"] = close_df / close_df.shift(20) - 1
    features_dict["mom_60d"] = close_df / close_df.shift(60) - 1

    # Reversion features
    ret = close_df.pct_change(fill_method=None)
    recent_ret_5 = close_df / close_df.shift(5) - 1

    vol_10 = ret.rolling(10, min_periods=5).std()
    features_dict["reversion_10d"] = -recent_ret_5 / (vol_10 + 1e-8)

    vol_20 = ret.rolling(20, min_periods=10).std()
    features_dict["reversion_20d"] = -recent_ret_5 / (vol_20 + 1e-8)

    vol_30 = ret.rolling(30, min_periods=15).std()
    features_dict["reversion_30d"] = -recent_ret_5 / (vol_30 + 1e-8)

    # Volume features
    vol_ma_10 = volume_df.rolling(10, min_periods=5).mean()
    features_dict["vol_signal_10d"] = volume_df / (vol_ma_10 + 1e-8) - 1

    vol_ma_20 = volume_df.rolling(20, min_periods=10).mean()
    features_dict["vol_signal_20d"] = volume_df / (vol_ma_20 + 1e-8) - 1

    # Get latest date
    latest_date = close_df.index[-1]

    # Combine signals using trained weights
    combined_score = {}

    for ticker in tickers:
        score = 0.0
        for feat_name, weight in FEATURE_WEIGHTS.items():
            if feat_name in features_dict:
                feat_value = features_dict[feat_name].loc[latest_date, ticker]
                if not (np.isnan(feat_value) or np.isinf(feat_value)):
                    score += feat_value * weight

        combined_score[ticker] = score

    # Rank normalize across all stocks
    scores = np.array([combined_score.get(t, 0.0) for t in tickers])
    valid_mask = ~(np.isnan(scores) | np.isinf(scores))

    result = {}
    if valid_mask.sum() > 0:
        # Winsorize
        valid_scores = scores[valid_mask]
        lo, hi = np.percentile(valid_scores, [1, 99])
        scores_clipped = np.clip(scores, lo, hi)

        # Rank normalize to [-1, 1]
        ranks = rankdata(scores_clipped)
        normalized = 2 * (ranks - 1) / (len(ranks) - 1) - 1 if len(ranks) > 1 else 0

        for i, ticker in enumerate(tickers):
            if valid_mask[i]:
                result[ticker] = float(normalized[i])
            else:
                result[ticker] = 0.0
    else:
        result = {ticker: 0.0 for ticker in tickers}

    return result