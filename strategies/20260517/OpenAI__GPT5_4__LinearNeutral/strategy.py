from pathlib import Path

import numpy as np
import pandas as pd


CROSS_SECTION_CLIP = 4.0
MAX_LOOKBACK = 21
FACTOR_WEIGHTS = {
    "rev1_vol": 1.0,
    "vol_shock20": 1.0,
}


def _cross_sectional_zscore(values, clip=CROSS_SECTION_CLIP):
    series = pd.Series(values, dtype=float)
    if series.empty:
        return series
    series = series.replace([np.inf, -np.inf], np.nan)
    valid = series.notna()
    if valid.sum() < 2:
        out = pd.Series(0.0, index=series.index, dtype=float)
        out[~valid] = np.nan
        return out
    std = series[valid].std()
    if not np.isfinite(std) or std <= 0:
        out = pd.Series(0.0, index=series.index, dtype=float)
        out[~valid] = np.nan
        return out
    out = (series - series[valid].mean()) / std
    out = out.clip(-clip, clip)
    out[~valid] = np.nan
    return out


def _cross_sectional_zscore_frame(frame, clip=CROSS_SECTION_CLIP):
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0.0, np.nan)
    zscore = frame.sub(mean, axis=0).div(std, axis=0)
    return zscore.clip(-clip, clip)


def _latest_raw_features(history):
    close = pd.to_numeric(history["close"], errors="coerce")
    volume = pd.to_numeric(history["volume"], errors="coerce").where(lambda x: x > 0)

    features = {}

    if len(close) >= 2:
        ret1 = close.pct_change(fill_method=None)
        latest_ret = ret1.iloc[-1]
        if pd.notna(latest_ret):
            vol20 = ret1.rolling(20).std().iloc[-1]
            if pd.notna(vol20) and vol20 > 0:
                features["rev1_vol"] = float(-latest_ret / vol20)
            else:
                features["rev1_vol"] = float(-latest_ret)

    if len(close) >= 1:
        latest_volume = volume.iloc[-1]
        adv20 = volume.rolling(20).mean().iloc[-1]
        if (
            pd.notna(latest_volume)
            and latest_volume > 0
            and pd.notna(adv20)
            and adv20 > 0
        ):
            features["vol_shock20"] = float(np.log(latest_volume / adv20))

    return features


def generate_signal(data):
    if not data:
        return {}

    raw_features = {name: {} for name in FACTOR_WEIGHTS}
    tickers = list(data.keys())

    for ticker, frame in data.items():
        if frame is None or frame.empty:
            continue

        required = [col for col in ["close", "volume"] if col in frame.columns]
        if len(required) < 2:
            continue

        history = frame
        if "Date" in history.columns and not history["Date"].is_monotonic_increasing:
            history = history.sort_values("Date")
        cols = [col for col in ["Date", "close", "volume"] if col in history.columns]
        history = history.loc[:, cols].tail(MAX_LOOKBACK)

        features = _latest_raw_features(history)
        for name, value in features.items():
            if name in raw_features and np.isfinite(value):
                raw_features[name][ticker] = float(value)

    scores = pd.Series(0.0, index=tickers, dtype=float)
    used_weight = pd.Series(0.0, index=tickers, dtype=float)

    for name, weight in FACTOR_WEIGHTS.items():
        zscore = _cross_sectional_zscore(raw_features[name])
        if zscore.empty:
            continue
        valid = zscore.notna()
        scores.loc[valid.index[valid]] += weight * zscore.loc[valid.index[valid]]
        used_weight.loc[valid.index[valid]] += abs(weight)

    valid = used_weight > 0
    scores.loc[valid] = scores.loc[valid] / used_weight.loc[valid]
    scores.loc[~valid] = 0.0
    scores = scores.fillna(0.0)
    return {ticker: float(scores.loc[ticker]) for ticker in tickers}


def _build_research_signal(close, volume):
    ret1 = close.pct_change(fill_method=None)
    vol20 = ret1.rolling(20).std()
    safe_volume = volume.where(volume > 0)

    raw_features = {
        "rev1_vol": (-ret1.div(vol20)).combine_first(-ret1),
        "vol_shock20": np.log(safe_volume / safe_volume.rolling(20).mean()),
    }

    signal = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    weight_sum = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    for name, weight in FACTOR_WEIGHTS.items():
        zscore = _cross_sectional_zscore_frame(raw_features[name])
        valid = zscore.notna().astype(float)
        signal = signal + weight * zscore.fillna(0.0)
        weight_sum = weight_sum + abs(weight) * valid

    signal = signal.div(weight_sum.where(weight_sum > 0))
    return raw_features, signal


def _daily_rank_ic(signal, future_returns):
    values = []
    for dt in signal.index.intersection(future_returns.index):
        x = signal.loc[dt]
        y = future_returns.loc[dt]
        mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 30:
            continue
        ic = x[mask].rank().corr(y[mask].rank())
        if pd.notna(ic):
            values.append(float(ic))
    return np.asarray(values, dtype=float)


def _portfolio_returns(signal, future_returns):
    returns = []
    for dt in signal.index.intersection(future_returns.index):
        x = signal.loc[dt]
        y = future_returns.loc[dt]
        mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        n_names = int(mask.sum())
        if n_names < 30:
            continue
        ranks = x[mask].rank(method="first")
        weights = ranks - (n_names + 1.0) / 2.0
        gross = weights.abs().sum()
        if gross <= 0:
            continue
        returns.append(float((weights / gross * y[mask]).sum()))
    return np.asarray(returns, dtype=float)


def _summarize(signal, future_returns):
    daily_returns = _portfolio_returns(signal, future_returns)
    daily_ic = _daily_rank_ic(signal, future_returns)

    summary = {
        "days": int(len(daily_returns)),
        "mean_bps": np.nan,
        "sharpe": np.nan,
        "hit_rate": np.nan,
        "mean_ic": np.nan,
        "ic_ir": np.nan,
    }

    if len(daily_returns):
        summary["mean_bps"] = float(daily_returns.mean() * 10000.0)
        summary["hit_rate"] = float((daily_returns > 0).mean())
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            summary["sharpe"] = float(
                daily_returns.mean() / daily_returns.std() * np.sqrt(252.0)
            )

    if len(daily_ic):
        summary["mean_ic"] = float(daily_ic.mean())
        if len(daily_ic) > 1 and daily_ic.std() > 0:
            summary["ic_ir"] = float(daily_ic.mean() / daily_ic.std())

    return summary


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    close = pd.read_parquet(base_dir / "close.parquet")
    volume = pd.read_parquet(base_dir / "volume.parquet")

    future_returns = close.shift(-1) / close - 1.0
    raw_features, signal = _build_research_signal(close, volume)

    splits = {
        "train": close.index[close.index < "2023-01-01"],
        "val1": close.index[(close.index >= "2023-01-01") & (close.index < "2023-07-01")],
        "val2": close.index[(close.index >= "2023-07-01") & (close.index < "2024-01-01")],
        "test": close.index[close.index >= "2024-01-01"],
    }

    print("selected_weights", FACTOR_WEIGHTS)
    print("component_metrics")
    for name, feature in raw_features.items():
        zscore_feature = _cross_sectional_zscore_frame(feature)
        metrics = {
            split_name: _summarize(zscore_feature.loc[dates], future_returns.loc[dates])
            for split_name, dates in splits.items()
        }
        print(name, metrics)

    print("final_signal_metrics")
    for split_name, dates in splits.items():
        print(split_name, _summarize(signal.loc[dates], future_returns.loc[dates]))

    print("yearly_metrics")
    for year in sorted(close.index.year.unique()):
        dates = close.index[close.index.year == year]
        print(int(year), _summarize(signal.loc[dates], future_returns.loc[dates]))