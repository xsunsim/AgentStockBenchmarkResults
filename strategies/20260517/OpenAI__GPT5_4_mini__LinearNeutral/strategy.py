# Strategy: OpenAI__GPT5_4_mini__LinearNeutral_202605
# Model: OpenAI__GPT5_4_mini (api: gpt-5.4-mini)
# File: strategies/OpenAI__GPT5_4_mini__LinearNeutral_202605/strategy.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FINAL_CLUSTER_WEIGHT = 0.5
FINAL_VOLUME_WEIGHT = 0.5
MIN_NAMES_PER_DAY = 20


def load_panel(base_dir: Path) -> dict[str, pd.DataFrame]:
    panel = {}
    for field in ("open", "high", "low", "close", "volume"):
        panel[field] = pd.read_parquet(base_dir / f"{field}.parquet").sort_index()
    common_index = panel["close"].index
    common_columns = panel["close"].columns
    for field, frame in panel.items():
        panel[field] = frame.reindex(index=common_index, columns=common_columns)
    return panel


def cross_sectional_rank(values: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    if isinstance(values, pd.Series):
        output = pd.Series(np.nan, index=values.index, dtype=float)
        mask = values.notna()
        count = int(mask.sum())
        if count == 0:
            return output
        ranks = values[mask].rank(method="average")
        output.loc[mask] = (ranks - (count + 1.0) / 2.0) / count
        return output

    counts = values.notna().sum(axis=1).replace(0, np.nan).astype(float)
    ranks = values.rank(axis=1, method="average")
    centered = ranks.sub((counts + 1.0) / 2.0, axis=0)
    return centered.div(counts, axis=0)


def combine_weighted(
    parts: list[pd.DataFrame | pd.Series],
    weights: list[float],
) -> pd.DataFrame | pd.Series:
    if len(parts) != len(weights):
        raise ValueError("parts and weights must have the same length")

    numerator = None
    denominator = None
    for part, weight in zip(parts, weights):
        weighted_part = part * weight
        present_weight = part.notna().astype(float) * weight
        numerator = weighted_part if numerator is None else numerator.add(weighted_part, fill_value=0.0)
        denominator = present_weight if denominator is None else denominator.add(present_weight, fill_value=0.0)
    return numerator / denominator.replace(0.0, np.nan)


def compute_feature_panel(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    open_ = open_.astype(float)
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    volume = volume.astype(float)

    ret1 = close.pct_change(1, fill_method=None)
    ret3 = close.pct_change(3, fill_method=None)
    vol20 = ret1.rolling(20, min_periods=10).std()
    avgvol20 = volume.rolling(20, min_periods=10).mean()

    open_safe = open_.replace(0.0, np.nan)
    range_safe = (high - low).replace(0.0, np.nan)

    raw_features = {
        "rev1_vol": -(ret1 / (vol20 + 1e-12)),
        "rev3_vol": -(ret3 / (vol20 + 1e-12)),
        "intraday_rev": -((close / open_safe) - 1.0),
        "range_pos_rev": -((2.0 * close - high - low) / range_safe),
        "vol_spike": np.log(volume.replace(0.0, np.nan) / avgvol20.replace(0.0, np.nan)),
    }

    ranked_features = {name: cross_sectional_rank(frame) for name, frame in raw_features.items()}
    ranked_features["reversion_cluster"] = combine_weighted(
        [
            ranked_features["rev1_vol"],
            ranked_features["rev3_vol"],
            ranked_features["intraday_rev"],
            ranked_features["range_pos_rev"],
        ],
        [0.25, 0.25, 0.25, 0.25],
    )
    ranked_features["ranked_cluster"] = cross_sectional_rank(ranked_features["reversion_cluster"])
    ranked_features["final_score"] = combine_weighted(
        [ranked_features["reversion_cluster"], ranked_features["vol_spike"]],
        [FINAL_CLUSTER_WEIGHT, FINAL_VOLUME_WEIGHT],
    )
    return ranked_features


def future_close_returns(close: pd.DataFrame) -> pd.DataFrame:
    return close.shift(-1) / close - 1.0


def linear_rank_weights(signal: pd.Series) -> pd.Series:
    ranks = signal.rank(method="first", ascending=False)
    weights = (len(ranks) + 1.0) / 2.0 - ranks
    return weights / weights.abs().sum()


def evaluate_period(
    score: pd.DataFrame,
    future_returns: pd.DataFrame,
    mask: pd.Series,
) -> dict[str, float]:
    daily_pnl = []
    daily_ic = []
    universe_sizes = []

    for date in score.index[mask]:
        signal = score.loc[date].replace([np.inf, -np.inf], np.nan)
        returns = future_returns.loc[date].replace([np.inf, -np.inf], np.nan)
        valid = signal.notna() & returns.notna()
        valid_count = int(valid.sum())
        if valid_count < MIN_NAMES_PER_DAY:
            continue

        signal_valid = signal[valid]
        returns_valid = returns[valid]
        weights = linear_rank_weights(signal_valid)
        pnl = float((weights * returns_valid).sum())

        ranked_signal = cross_sectional_rank(signal_valid)
        ranked_return = cross_sectional_rank(returns_valid)
        ic = float(ranked_signal.corr(ranked_return))

        if np.isfinite(ic):
            daily_ic.append(ic)
        daily_pnl.append(pnl)
        universe_sizes.append(valid_count)

    if not daily_pnl:
        return {
            "days": 0,
            "avg_universe": np.nan,
            "ic_mean": np.nan,
            "ic_ir": np.nan,
            "mean_bps": np.nan,
            "sharpe": np.nan,
        }

    pnl_array = np.asarray(daily_pnl, dtype=float)
    ic_array = np.asarray(daily_ic, dtype=float)

    sharpe = np.nan
    if len(pnl_array) > 1 and np.nanstd(pnl_array, ddof=1) > 0:
        sharpe = float(np.nanmean(pnl_array) / np.nanstd(pnl_array, ddof=1) * np.sqrt(252.0))

    ic_ir = np.nan
    if len(ic_array) > 1 and np.nanstd(ic_array, ddof=1) > 0:
        ic_ir = float(np.nanmean(ic_array) / np.nanstd(ic_array, ddof=1) * np.sqrt(252.0))

    return {
        "days": float(len(pnl_array)),
        "avg_universe": float(np.nanmean(universe_sizes)),
        "ic_mean": float(np.nanmean(ic_array)) if len(ic_array) else np.nan,
        "ic_ir": ic_ir,
        "mean_bps": float(np.nanmean(pnl_array) * 1e4),
        "sharpe": sharpe,
    }


def make_period_masks(index: pd.Index) -> dict[str, pd.Series]:
    timestamp_index = pd.Index(pd.to_datetime(index))
    return {
        "train": timestamp_index < pd.Timestamp("2023-01-01"),
        "val1": (timestamp_index >= pd.Timestamp("2023-01-01")) & (timestamp_index < pd.Timestamp("2023-07-01")),
        "val2": (timestamp_index >= pd.Timestamp("2023-07-01")) & (timestamp_index < pd.Timestamp("2024-01-01")),
        "test": timestamp_index >= pd.Timestamp("2024-01-01"),
    }


def evaluate_candidate_table(
    candidates: dict[str, pd.DataFrame],
    future_returns: pd.DataFrame,
    masks: dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []
    for name, score in candidates.items():
        row = {"candidate": name}
        for period_name, mask in masks.items():
            stats = evaluate_period(score, future_returns, mask)
            row[f"{period_name}_ic"] = stats["ic_mean"]
            row[f"{period_name}_sharpe"] = stats["sharpe"]
            row[f"{period_name}_bps"] = stats["mean_bps"]
        row["val_avg_ic"] = np.nanmean([row["val1_ic"], row["val2_ic"]])
        row["val_avg_sharpe"] = np.nanmean([row["val1_sharpe"], row["val2_sharpe"]])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["val_avg_sharpe", "val_avg_ic"], ascending=False)


def evaluate_feature_table(
    features: dict[str, pd.DataFrame],
    future_returns: pd.DataFrame,
    masks: dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []
    for name in ("rev1_vol", "rev3_vol", "intraday_rev", "range_pos_rev", "vol_spike"):
        row = {"feature": name}
        for period_name, mask in masks.items():
            stats = evaluate_period(features[name], future_returns, mask)
            row[f"{period_name}_ic"] = stats["ic_mean"]
            row[f"{period_name}_sharpe"] = stats["sharpe"]
            row[f"{period_name}_bps"] = stats["mean_bps"]
        row["val_avg_ic"] = np.nanmean([row["val1_ic"], row["val2_ic"]])
        row["val_avg_sharpe"] = np.nanmean([row["val1_sharpe"], row["val2_sharpe"]])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["val_avg_ic", "val_avg_sharpe"], ascending=False)


def print_table(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(frame.to_string(index=False, float_format=lambda x: f"{x: .4f}"))


def build_wide_from_live_data(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if not data:
        raise ValueError("data is empty")

    field_values: dict[str, dict[str, pd.Series]] = {field: {} for field in ("open", "high", "low", "close", "volume")}
    tickers = sorted(data)

    for ticker in tickers:
        frame = data[ticker]
        if frame is None or len(frame) == 0:
            continue

        local = frame.copy()
        lower_map = {str(column).lower(): column for column in local.columns}
        if "date" in lower_map:
            date_values = pd.to_datetime(local[lower_map["date"]], errors="coerce")
            local = local.loc[date_values.notna()].copy()
            local.index = pd.Index(date_values[date_values.notna()], name="Date")
        elif isinstance(local.index, pd.DatetimeIndex):
            local.index = pd.to_datetime(local.index, errors="coerce")
            local = local.loc[local.index.notna()].copy()
        else:
            raise ValueError(f"{ticker} is missing a Date column or DatetimeIndex")

        local = local[~local.index.duplicated(keep="last")].sort_index()

        for field in field_values:
            if field not in lower_map:
                raise ValueError(f"{ticker} is missing required column: {field}")
            field_values[field][ticker] = pd.to_numeric(local[lower_map[field]], errors="coerce")

    wide = {field: pd.DataFrame(columns=tickers) for field in field_values}
    union_index = pd.Index([])
    for field, series_map in field_values.items():
        frame = pd.DataFrame(series_map)
        union_index = union_index.union(frame.index)
        wide[field] = frame

    union_index = pd.Index(pd.to_datetime(union_index)).sort_values()
    for field, frame in wide.items():
        wide[field] = frame.reindex(index=union_index, columns=tickers).sort_index()
    return wide


def generate_signal(data: dict[str, pd.DataFrame]) -> dict[str, float]:
    if not data:
        return {}

    panel = build_wide_from_live_data(data)
    features = compute_feature_panel(
        panel["open"],
        panel["high"],
        panel["low"],
        panel["close"],
        panel["volume"],
    )

    latest_score = features["final_score"].iloc[-1].replace([np.inf, -np.inf], np.nan)
    latest_score = latest_score.fillna(0.0)
    return {ticker: float(latest_score.get(ticker, 0.0)) for ticker in sorted(data)}


def run_research(base_dir: Path | None = None) -> None:
    base_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    panel = load_panel(base_dir)

    close = panel["close"]
    summary_rows = []
    for field, frame in panel.items():
        summary_rows.append(
            {
                "field": field,
                "rows": frame.shape[0],
                "cols": frame.shape[1],
                "start": frame.index.min(),
                "end": frame.index.max(),
                "nan_count": int(frame.isna().sum().sum()),
            }
        )
    print_table("Panel Summary", pd.DataFrame(summary_rows))

    features = compute_feature_panel(
        panel["open"],
        panel["high"],
        panel["low"],
        panel["close"],
        panel["volume"],
    )
    future_returns = future_close_returns(close)
    masks = make_period_masks(close.index)

    feature_table = evaluate_feature_table(features, future_returns, masks)
    print_table("Single-Feature Results", feature_table)

    candidate_scores = {
        "cluster_only": features["reversion_cluster"],
        "vol_only": features["vol_spike"],
        "cluster_60_vol_40": combine_weighted(
            [features["reversion_cluster"], features["vol_spike"]],
            [0.6, 0.4],
        ),
        "cluster_50_vol_50": features["final_score"],
        "cluster_40_vol_60": combine_weighted(
            [features["reversion_cluster"], features["vol_spike"]],
            [0.4, 0.6],
        ),
        "ranked_cluster_50_vol_50": combine_weighted(
            [features["ranked_cluster"], features["vol_spike"]],
            [0.5, 0.5],
        ),
    }
    candidate_table = evaluate_candidate_table(candidate_scores, future_returns, masks)
    print_table("Composite Results", candidate_table)

    final_stats_rows = []
    for period_name, mask in masks.items():
        stats = evaluate_period(features["final_score"], future_returns, mask)
        final_stats_rows.append({"period": period_name, **stats})
    print_table("Final Model Detail", pd.DataFrame(final_stats_rows))

    latest_signal = features["final_score"].iloc[-1].dropna().sort_values(ascending=False)
    preview = pd.DataFrame(
        {
            "ticker": latest_signal.head(10).index,
            "score": latest_signal.head(10).values,
        }
    )
    print_table("Latest Top 10", preview)


if __name__ == "__main__":
    run_research()