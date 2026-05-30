# Strategy: OpenAI__O3__LinearNeutral_202605
# Model: OpenAI__O3 (api: o3)
# File: strategies/OpenAI__O3__LinearNeutral_202605/strategy.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_FIELDS = ("open", "high", "low", "close", "volume")
FINAL_FEATURES = ("short_meanrev", "range", "volshock20")


def winsorize_cross_section(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    row_lower = df.quantile(lower, axis=1)
    row_upper = df.quantile(upper, axis=1)
    return df.clip(lower=row_lower, upper=row_upper, axis=0)


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    clipped = winsorize_cross_section(df)
    row_mean = clipped.mean(axis=1)
    row_std = clipped.std(axis=1)
    return clipped.sub(row_mean, axis=0).div(row_std.replace(0.0, np.nan), axis=0)


def rowwise_corr(left: pd.DataFrame, right: pd.DataFrame) -> pd.Series:
    mask = left.notna() & right.notna()
    left_masked = left.where(mask)
    right_masked = right.where(mask)
    left_centered = left_masked.sub(left_masked.mean(axis=1), axis=0)
    right_centered = right_masked.sub(right_masked.mean(axis=1), axis=0)
    covariance = (left_centered * right_centered).sum(axis=1)
    left_var = (left_centered * left_centered).sum(axis=1)
    right_var = (right_centered * right_centered).sum(axis=1)
    return covariance / np.sqrt(left_var * right_var)


def rank_ic(signal: pd.DataFrame, future_returns: pd.DataFrame) -> pd.Series:
    return rowwise_corr(
        signal.rank(axis=1, method="average", na_option="keep"),
        future_returns.rank(axis=1, method="average", na_option="keep"),
    )


def rank_portfolio_returns(signal: pd.DataFrame, future_returns: pd.DataFrame) -> pd.Series:
    mask = signal.notna() & future_returns.notna()
    counts = mask.sum(axis=1)
    ranks = signal.where(mask).rank(axis=1, method="first", ascending=False)
    weights = ((counts + 1.0) / 2.0).to_numpy()[:, None] - ranks.to_numpy()
    weights[~mask.to_numpy()] = np.nan
    gross = np.nansum(np.abs(weights), axis=1)
    pnl = np.nansum((weights / gross[:, None]) * future_returns.to_numpy(), axis=1)
    pnl[gross == 0.0] = np.nan
    return pd.Series(pnl, index=signal.index)


def annualized_sharpe(returns: pd.Series) -> float:
    std = returns.std(ddof=1)
    if not np.isfinite(std) or std == 0.0:
        return np.nan
    return float(np.sqrt(252.0) * returns.mean() / std)


def load_research_panels(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    close = pd.read_parquet(base_dir / "close.parquet").sort_index()
    open_ = pd.read_parquet(base_dir / "open.parquet").reindex_like(close)
    high = pd.read_parquet(base_dir / "high.parquet").reindex_like(close)
    low = pd.read_parquet(base_dir / "low.parquet").reindex_like(close)
    volume = pd.read_parquet(base_dir / "volume.parquet").reindex_like(close)
    return open_, high, low, close, volume


def build_feature_library(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    close_safe = close.replace(0.0, np.nan)
    volume_safe = volume.replace(0.0, np.nan)

    ret1 = close_safe.pct_change(fill_method=None)
    vol20 = ret1.rolling(20, min_periods=20).std()
    ma5 = close_safe.rolling(5, min_periods=5).mean()
    ma20 = close_safe.rolling(20, min_periods=20).mean()
    adv20 = volume_safe.rolling(20, min_periods=20).mean()

    clv = (2.0 * close_safe - high - low) / (high - low).replace(0.0, np.nan)

    raw_features = {
        "short_meanrev": -((close_safe / ma5) - 1.0) / vol20.replace(0.0, np.nan),
        "range": (high - low) / close_safe,
        "volshock20": np.log1p(volume_safe / adv20.replace(0.0, np.nan)),
        "clv_rev": -clv,
        "med_meanrev": -((close_safe / ma20) - 1.0) / vol20.replace(0.0, np.nan),
    }
    feature_z = {name: cross_sectional_zscore(panel) for name, panel in raw_features.items()}
    feature_z["combo_eq3"] = (
        feature_z["short_meanrev"] + feature_z["range"] + feature_z["volshock20"]
    ) / 3.0
    feature_z["combo_eq4_clv"] = (
        feature_z["short_meanrev"] + feature_z["range"] + feature_z["volshock20"] + feature_z["clv_rev"]
    ) / 4.0
    feature_z["combo_eq5"] = (
        feature_z["short_meanrev"]
        + feature_z["range"]
        + feature_z["volshock20"]
        + feature_z["clv_rev"]
        + feature_z["med_meanrev"]
    ) / 5.0
    return feature_z


def build_split_masks(index: pd.Index) -> dict[str, np.ndarray]:
    years = index.year
    months = index.month
    return {
        "train": years <= 2022,
        "val1": (years == 2023) & (months <= 6),
        "val2": (years == 2023) & (months >= 7),
        "test": years == 2024,
    }


def summarize_signals(
    signals: dict[str, pd.DataFrame],
    future_returns: pd.DataFrame,
    split_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for name, signal in signals.items():
        ic_series = rank_ic(signal, future_returns)
        pnl_series = rank_portfolio_returns(signal, future_returns)
        row = {"signal": name}
        for split_name, split_mask in split_masks.items():
            split_ic = ic_series[split_mask]
            split_pnl = pnl_series[split_mask]
            row[f"{split_name}_ic_bps"] = 1.0e4 * split_ic.mean()
            row[f"{split_name}_sharpe"] = annualized_sharpe(split_pnl)
        rows.append(row)
    summary = pd.DataFrame(rows)
    order = [
        "signal",
        "train_ic_bps",
        "train_sharpe",
        "val1_ic_bps",
        "val1_sharpe",
        "val2_ic_bps",
        "val2_sharpe",
        "test_ic_bps",
        "test_sharpe",
    ]
    return summary[order]


def yearly_signal_metrics(signal: pd.DataFrame, future_returns: pd.DataFrame) -> pd.DataFrame:
    ic_series = rank_ic(signal, future_returns)
    pnl_series = rank_portfolio_returns(signal, future_returns)
    rows = []
    for year in sorted(signal.index.year.unique()):
        mask = signal.index.year == year
        ic_year = ic_series[mask]
        pnl_year = pnl_series[mask]
        rows.append(
            {
                "year": int(year),
                "days": int(ic_year.count()),
                "ic_bps": 1.0e4 * ic_year.mean(),
                "sharpe": annualized_sharpe(pnl_year),
                "mean_daily_bps": 1.0e4 * pnl_year.mean(),
            }
        )
    return pd.DataFrame(rows)


def coerce_history_frame(frame: pd.DataFrame) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None

    working = frame.copy()
    if "Date" in working.columns:
        dates = pd.to_datetime(working["Date"], errors="coerce")
    else:
        dates = pd.to_datetime(working.index, errors="coerce")

    valid_dates = dates.notna()
    if valid_dates.sum() == 0:
        return None

    missing_fields = [field for field in REQUIRED_FIELDS if field not in working.columns]
    if missing_fields:
        return None

    history = working.loc[valid_dates, list(REQUIRED_FIELDS)].copy()
    history.insert(0, "Date", dates[valid_dates].to_numpy())
    history = history.drop_duplicates("Date", keep="last").sort_values("Date")
    for field in REQUIRED_FIELDS:
        history[field] = pd.to_numeric(history[field], errors="coerce")
    return history


def build_live_panels(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    series_map = {field: {} for field in REQUIRED_FIELDS}
    for ticker, frame in data.items():
        history = coerce_history_frame(frame)
        if history is None:
            continue
        dates = pd.Index(history["Date"], name="Date")
        for field in REQUIRED_FIELDS:
            values = history[field].to_numpy(dtype=float, copy=False)
            series_map[field][ticker] = pd.Series(values, index=dates, name=ticker)

    if not series_map["close"]:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty

    open_ = pd.concat(series_map["open"], axis=1).sort_index()
    high = pd.concat(series_map["high"], axis=1).reindex_like(open_)
    low = pd.concat(series_map["low"], axis=1).reindex_like(open_)
    close = pd.concat(series_map["close"], axis=1).reindex_like(open_)
    volume = pd.concat(series_map["volume"], axis=1).reindex_like(open_)
    return open_, high, low, close, volume


def final_signal_panel(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
) -> pd.DataFrame:
    features = build_feature_library(open_, high, low, close, volume)
    return features["combo_eq3"]


def run_research() -> None:
    base_dir = Path(__file__).resolve().parent
    open_, high, low, close, volume = load_research_panels(base_dir)
    future_returns = close.shift(-1) / close - 1.0
    feature_library = build_feature_library(open_, high, low, close, volume)
    split_masks = build_split_masks(close.index)

    candidate_signals = {
        "short_meanrev": feature_library["short_meanrev"],
        "range": feature_library["range"],
        "volshock20": feature_library["volshock20"],
        "combo_eq3": feature_library["combo_eq3"],
        "combo_eq4_clv": feature_library["combo_eq4_clv"],
        "combo_eq5": feature_library["combo_eq5"],
    }

    print(f"Data shape: {close.shape[0]} trading days x {close.shape[1]} tickers")
    print(f"Date range: {close.index.min().date()} -> {close.index.max().date()}")
    print()
    summary = summarize_signals(candidate_signals, future_returns, split_masks)
    print("Split metrics (rank IC in bps, rank portfolio Sharpe):")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:8.2f}"))
    print()
    print("Yearly metrics for final signal (combo_eq3):")
    yearly = yearly_signal_metrics(feature_library["combo_eq3"], future_returns)
    print(yearly.to_string(index=False, float_format=lambda value: f"{value:8.2f}"))


if __name__ == "__main__":
    run_research()


def generate_signal(data: dict[str, pd.DataFrame]) -> dict[str, float]:
    scores = {ticker: 0.0 for ticker in data}
    open_, high, low, close, volume = build_live_panels(data)
    if close.empty:
        return scores

    signal = final_signal_panel(open_, high, low, close, volume)
    latest = signal.iloc[-1].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for ticker in latest.index:
        scores[ticker] = float(latest.loc[ticker])
    return scores