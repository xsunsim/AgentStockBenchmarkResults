"""Daily cross-sectional S&P 500 ranking signal."""

import numpy as np
import pandas as pd


def _panel(data, field):
    """Convert the ticker-keyed production input into a date x ticker panel."""
    series = {}
    for ticker in sorted(data):
        frame = data[ticker]
        if field not in frame.columns or "Date" not in frame.columns:
            continue
        dates = pd.to_datetime(frame["Date"], errors="coerce")
        values = pd.to_numeric(frame[field], errors="coerce")
        item = pd.Series(values.to_numpy(), index=dates)
        item = item.loc[item.index.notna()]
        item = item[~item.index.duplicated(keep="last")].sort_index()
        series[ticker] = item
    return pd.DataFrame(series).sort_index()


def _cross_sectional_rank(values):
    """Percentile rank, with unavailable observations assigned a neutral score."""
    ranked = values.replace([np.inf, -np.inf], np.nan).rank(pct=True)
    return ranked.fillna(0.5)


def generate_signal(data):
    """Return the latest equal-weight rank blend; higher values are more long."""
    tickers = sorted(data)
    if not tickers:
        return {}

    close = _panel(data, "close").reindex(columns=tickers)
    open_ = _panel(data, "open").reindex(columns=tickers)
    if close.empty:
        return {ticker: 0.0 for ticker in tickers}

    # Each component was selected on 2020-2023 only.  The blend balances
    # distinct horizons and uses ranks so price/volatility scales do not matter.
    overnight_reversal = -(open_ / close.shift(1) - 1.0)
    prior_day_reversal = -close.pct_change(fill_method=None).shift(1)
    deviation_5 = -(close / close.rolling(5, min_periods=5).mean() - 1.0)
    momentum_120 = close / close.shift(120) - 1.0

    latest = close.index[-1]
    components = (
        overnight_reversal.loc[latest],
        prior_day_reversal.loc[latest],
        deviation_5.loc[latest],
        momentum_120.loc[latest],
    )
    score = sum(_cross_sectional_rank(x) for x in components) / len(components)

    # Break exact ties deterministically without affecting any non-tied ordering.
    tie_break = pd.Series(np.arange(len(tickers)), index=tickers, dtype=float)
    score = score.reindex(tickers).fillna(0.5) + tie_break * 1e-12
    return {ticker: float(score[ticker]) for ticker in tickers}


def _rank_ic(feature, forward_return):
    return feature.rank(axis=1, pct=True).corrwith(
        forward_return.rank(axis=1, pct=True), axis=1
    )


def _report(name, feature, forward_return, periods):
    ic = _rank_ic(feature, forward_return)
    fields = []
    for label, start, end in periods:
        sample = ic.loc[start:end].dropna()
        mean = sample.mean()
        ir = mean / sample.std() if sample.std() else np.nan
        fields.append(f"{label}: IC={mean:+.4f}, IR={ir:+.3f}, n={len(sample)}")
    print(f"{name:22s} " + " | ".join(fields))


if __name__ == "__main__":
    # Research-only I/O. Nothing in this block runs when the module is imported.
    close = pd.read_parquet("close.parquet")
    open_ = pd.read_parquet("open.parquet")
    high = pd.read_parquet("high.parquet")
    low = pd.read_parquet("low.parquet")
    volume = pd.read_parquet("volume.parquet")

    ret1 = close.pct_change(fill_method=None)
    overnight = open_ / close.shift(1) - 1.0
    intraday = close / open_ - 1.0
    true_range = (high - low) / close.shift(1)
    close_location = (2.0 * close - high - low) / (high - low).replace(0, np.nan)
    log_volume = np.log1p(volume)
    volume_shock = log_volume - log_volume.rolling(20).mean()
    forward_return = close.shift(-2) / close.shift(-1) - 1.0

    candidates = {
        "reversal_1": -ret1,
        "reversal_2": -(close / close.shift(2) - 1.0),
        "reversal_5": -(close / close.shift(5) - 1.0),
        "prior_day_reversal": -ret1.shift(1),
        "overnight_reversal": -overnight,
        "intraday_reversal": -intraday,
        "deviation_5": -(close / close.rolling(5).mean() - 1.0),
        "deviation_20": -(close / close.rolling(20).mean() - 1.0),
        "momentum_60": close / close.shift(60) - 1.0,
        "momentum_120": close / close.shift(120) - 1.0,
        "low_volatility_20": -ret1.rolling(20).std(),
        "close_location": close_location,
        "range_reversal": -true_range,
        "volume_shock_reversal": -volume_shock,
        "scaled_reversal": -ret1 / ret1.rolling(20).std(),
    }

    discovery_periods = (
        ("train", "2020-01-01", "2022-12-31"),
        ("validation", "2023-01-01", "2023-12-31"),
    )
    print(f"panel: {close.shape[0]} dates x {close.shape[1]} tickers")
    print("candidate screen (test excluded):")
    for name, feature in candidates.items():
        _report(name, feature, forward_return, discovery_periods)

    selected = (
        candidates["overnight_reversal"],
        candidates["prior_day_reversal"],
        candidates["deviation_5"],
        candidates["momentum_120"],
    )
    combined = sum(x.rank(axis=1, pct=True) for x in selected) / len(selected)
    print("locked final blend:")
    _report("equal_weight_blend", combined, forward_return, discovery_periods)
    print("one-time held-out test:")
    _report(
        "equal_weight_blend",
        combined,
        forward_return,
        (("test", "2024-01-01", "2024-12-31"),),
    )