# Strategy: OpenAI__O4_mini__LinearNeutral_202605
# Model: OpenAI__O4_mini (api: o4-mini)
# File: strategies/OpenAI__O4_mini__LinearNeutral_202605/strategy.py

import os

import numpy as np
import pandas as pd


SPLITS = {
    "train": ("2020-01-02", "2022-12-30"),
    "val1": ("2023-01-03", "2023-06-30"),
    "val2": ("2023-07-03", "2023-12-29"),
    "test": ("2024-01-02", "2024-12-31"),
}

FINAL_MODEL_NAME = "reversal4_m126_rank_equal"
FINAL_MODEL_FEATURES = (
    "rev_1_voladj",
    "intraday_reversal",
    "vol_shock_20",
    "mom_126_21",
)
FINAL_MODEL_WEIGHTS = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)


def _cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True).sub(0.5)


def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0.0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0).clip(-5.0, 5.0)


def _daily_rank_ic(signal: pd.DataFrame, next_returns: pd.DataFrame) -> pd.Series:
    signal_rank = signal.rank(axis=1, pct=True)
    return_rank = next_returns.rank(axis=1, pct=True)
    signal_centered = signal_rank.sub(signal_rank.mean(axis=1), axis=0)
    return_centered = return_rank.sub(return_rank.mean(axis=1), axis=0)
    numerator = (signal_centered * return_centered).mean(axis=1)
    denominator = signal_rank.std(axis=1) * return_rank.std(axis=1)
    return numerator.div(denominator.replace(0.0, np.nan))


def _daily_rank_portfolio_return(signal: pd.DataFrame, next_returns: pd.DataFrame) -> pd.Series:
    ranks = signal.rank(axis=1, method="first", na_option="keep", ascending=False)
    counts = signal.notna().sum(axis=1)
    centered = ranks.mul(-1.0).add((counts + 1.0) / 2.0, axis=0)
    gross = centered.abs().sum(axis=1).replace(0.0, np.nan)
    weights = centered.div(gross, axis=0)
    return (weights * next_returns).sum(axis=1, min_count=1)


def _series_tstat(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) < 2:
        return np.nan
    std = values.std(ddof=1)
    if not np.isfinite(std) or std == 0.0:
        return np.nan
    return values.mean() / (std / np.sqrt(len(values)))


def _series_sharpe(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) < 2:
        return np.nan
    std = values.std(ddof=1)
    if not np.isfinite(std) or std == 0.0:
        return np.nan
    return values.mean() / std * np.sqrt(252.0)


def _slice_by_split(obj, split_name: str):
    start, end = SPLITS[split_name]
    return obj.loc[start:end]


def _load_research_panels(base_path: str) -> dict:
    panels = {}
    for name in ("open", "high", "low", "close", "volume"):
        path = os.path.join(base_path, f"{name}.parquet")
        panels[name] = pd.read_parquet(path).sort_index()
    return panels


def _build_feature_library(panels: dict) -> tuple:
    close = panels["close"]
    open_ = panels["open"].reindex_like(close)
    high = panels["high"].reindex_like(close)
    low = panels["low"].reindex_like(close)
    volume = panels["volume"].reindex_like(close)

    ret1 = close.pct_change(fill_method=None)
    vol20 = ret1.rolling(20, min_periods=20).std()
    log_volume = np.log(volume.replace(0.0, np.nan))
    intraday = close.div(open_).sub(1.0)
    close_in_range = close.sub(low).div(high.sub(low).replace(0.0, np.nan)).sub(0.5)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high.sub(low),
            high.sub(prev_close).abs(),
            low.sub(prev_close).abs(),
        ],
        axis=0,
    ).groupby(level=0).max()

    vol_shock_20 = log_volume.sub(log_volume.rolling(20, min_periods=20).mean())

    features = {
        "rev_1": -ret1,
        "rev_3": -close.pct_change(3, fill_method=None),
        "rev_1_voladj": -ret1.div(vol20.add(1e-12)),
        "intraday_reversal": -intraday,
        "close_weakness": -close_in_range,
        "vol_shock_20": vol_shock_20,
        "shock_rev_1": (-ret1).mul(vol_shock_20),
        "mom_126_21": close.shift(21).div(close.shift(126)).sub(1.0),
        "mom_252_21": close.shift(21).div(close.shift(252)).sub(1.0),
        "range_expansion": high.sub(low).div(close),
        "atrp_20_neg": -true_range.rolling(20, min_periods=20).mean().div(close),
        "dist_low_252": close.div(low.rolling(252, min_periods=252).min()).sub(1.0),
    }

    next_returns = close.shift(-1).div(close).sub(1.0)
    return features, next_returns


def _evaluate_features(features: dict, next_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, panel in features.items():
        ic = _daily_rank_ic(panel, next_returns)
        row = {"feature": name}
        for split_name in ("train", "val1", "val2"):
            split_ic = _slice_by_split(ic, split_name).dropna()
            row[f"{split_name}_ic"] = split_ic.mean()
            row[f"{split_name}_t"] = _series_tstat(split_ic)
        row["avg_val_ic"] = np.nanmean([row["val1_ic"], row["val2_ic"]])
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(["avg_val_ic", "train_ic"], ascending=False)


def _fit_pooled_ridge_weights(
    transformed_features: dict,
    next_returns: pd.DataFrame,
    feature_names: tuple,
    alpha: float,
) -> np.ndarray:
    target = _cross_sectional_zscore(next_returns)
    start, end = SPLITS["train"]
    dates = target.loc[start:end].index
    design = []
    response = []
    for date in dates:
        x_block = np.column_stack([transformed_features[name].loc[date].values for name in feature_names])
        y_block = target.loc[date].values
        mask = np.isfinite(x_block).all(axis=1) & np.isfinite(y_block)
        if mask.sum() < len(feature_names) + 5:
            continue
        design.append(x_block[mask])
        response.append(y_block[mask])
    if not design:
        return np.full(len(feature_names), 1.0 / len(feature_names))
    x_matrix = np.vstack(design)
    y_vector = np.concatenate(response)
    xtx = x_matrix.T @ x_matrix
    xty = x_matrix.T @ y_vector
    weights = np.linalg.solve(xtx + alpha * np.eye(len(feature_names)), xty)
    gross = np.sum(np.abs(weights))
    if not np.isfinite(gross) or gross == 0.0:
        return np.full(len(feature_names), 1.0 / len(feature_names))
    return weights / gross


def _build_signal(transformed_features: dict, feature_names: tuple, weights: np.ndarray) -> pd.DataFrame:
    signal = 0.0
    for weight, name in zip(weights, feature_names):
        signal = signal + transformed_features[name].fillna(0.0) * weight
    return signal


def _evaluate_signal(signal: pd.DataFrame, next_returns: pd.DataFrame) -> dict:
    metrics = {}
    daily_ic = _daily_rank_ic(signal, next_returns)
    daily_port = _daily_rank_portfolio_return(signal, next_returns)
    for split_name in ("train", "val1", "val2", "test"):
        split_ic = _slice_by_split(daily_ic, split_name).dropna()
        split_port = _slice_by_split(daily_port, split_name).dropna()
        metrics[f"{split_name}_ic"] = split_ic.mean()
        metrics[f"{split_name}_ic_t"] = _series_tstat(split_ic)
        metrics[f"{split_name}_hit"] = (split_ic > 0.0).mean() if len(split_ic) else np.nan
        metrics[f"{split_name}_sharpe"] = _series_sharpe(split_port)
        metrics[f"{split_name}_cumret"] = (1.0 + split_port).prod() - 1.0 if len(split_port) else np.nan
    return metrics


def run_research(base_path: str) -> None:
    panels = _load_research_panels(base_path)
    features, next_returns = _build_feature_library(panels)

    feature_summary = _evaluate_features(features, next_returns)

    rank_features = {name: _cross_sectional_rank(panel) for name, panel in features.items()}
    zscore_features = {name: _cross_sectional_zscore(panel) for name, panel in features.items()}

    candidate_specs = [
        {
            "model": "reversal3_rank_equal",
            "feature_names": ("rev_1_voladj", "intraday_reversal", "vol_shock_20"),
            "transform": "rank",
            "weights": np.array([1.0 / 3.0] * 3, dtype=float),
        },
        {
            "model": "reversal4_rank_equal",
            "feature_names": ("rev_1_voladj", "intraday_reversal", "vol_shock_20", "shock_rev_1"),
            "transform": "rank",
            "weights": np.array([0.25, 0.25, 0.25, 0.25], dtype=float),
        },
        {
            "model": "reversal4_close_rank_equal",
            "feature_names": ("rev_1_voladj", "intraday_reversal", "vol_shock_20", "close_weakness"),
            "transform": "rank",
            "weights": np.array([0.25, 0.25, 0.25, 0.25], dtype=float),
        },
        {
            "model": "reversal4_m126_rank_equal",
            "feature_names": FINAL_MODEL_FEATURES,
            "transform": "rank",
            "weights": FINAL_MODEL_WEIGHTS.copy(),
        },
        {
            "model": "reversal4_m252_rank_equal",
            "feature_names": ("rev_1_voladj", "intraday_reversal", "vol_shock_20", "mom_252_21"),
            "transform": "rank",
            "weights": np.array([0.25, 0.25, 0.25, 0.25], dtype=float),
        },
        {
            "model": "reversal5_m126_rank_equal",
            "feature_names": ("rev_1_voladj", "intraday_reversal", "vol_shock_20", "shock_rev_1", "mom_126_21"),
            "transform": "rank",
            "weights": np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=float),
        },
        {
            "model": "reversal4_m126_z_ridge10",
            "feature_names": FINAL_MODEL_FEATURES,
            "transform": "zscore",
            "weights": _fit_pooled_ridge_weights(zscore_features, next_returns, FINAL_MODEL_FEATURES, alpha=10.0),
        },
    ]

    model_rows = []
    for spec in candidate_specs:
        transformed = rank_features if spec["transform"] == "rank" else zscore_features
        signal = _build_signal(transformed, spec["feature_names"], spec["weights"])
        metrics = _evaluate_signal(signal, next_returns)
        row = {
            "model": spec["model"],
            "transform": spec["transform"],
            "weights": dict(zip(spec["feature_names"], np.round(spec["weights"], 4))),
        }
        for split_name in ("train", "val1", "val2"):
            row[f"{split_name}_ic"] = metrics[f"{split_name}_ic"]
            row[f"{split_name}_sharpe"] = metrics[f"{split_name}_sharpe"]
        row["avg_val_ic"] = np.nanmean([row["val1_ic"], row["val2_ic"]])
        row["avg_val_sharpe"] = np.nanmean([row["val1_sharpe"], row["val2_sharpe"]])
        model_rows.append(row)

    model_summary = pd.DataFrame(model_rows).sort_values(["avg_val_ic", "train_ic"], ascending=False)

    final_signal = _build_signal(rank_features, FINAL_MODEL_FEATURES, FINAL_MODEL_WEIGHTS)
    final_metrics = _evaluate_signal(final_signal, next_returns)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", "{:.6f}".format)

    print("Feature IC summary (train/validation only):")
    print(
        feature_summary[
            ["feature", "train_ic", "train_t", "val1_ic", "val1_t", "val2_ic", "val2_t", "avg_val_ic"]
        ]
        .head(12)
        .to_string(index=False)
    )
    print()
    print("Candidate model summary (selection uses train, val1, val2 only):")
    print(
        model_summary[
            [
                "model",
                "transform",
                "train_ic",
                "val1_ic",
                "val2_ic",
                "avg_val_ic",
                "train_sharpe",
                "val1_sharpe",
                "val2_sharpe",
                "avg_val_sharpe",
                "weights",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Selected final model: {FINAL_MODEL_NAME}")
    print(f"Selected weights: {dict(zip(FINAL_MODEL_FEATURES, FINAL_MODEL_WEIGHTS))}")
    print()
    print("Selected model performance:")
    final_table = pd.DataFrame(
        [
            {
                "split": split_name,
                "rank_ic": final_metrics[f"{split_name}_ic"],
                "ic_tstat": final_metrics[f"{split_name}_ic_t"],
                "ic_hit_rate": final_metrics[f"{split_name}_hit"],
                "portfolio_sharpe": final_metrics[f"{split_name}_sharpe"],
                "cum_return": final_metrics[f"{split_name}_cumret"],
            }
            for split_name in ("train", "val1", "val2", "test")
        ]
    )
    print(final_table.to_string(index=False))


def _history_dict_to_panels(data: dict) -> dict:
    open_dict = {}
    close_dict = {}
    volume_dict = {}

    for ticker, frame in data.items():
        if frame is None or len(frame) == 0:
            continue

        if "Date" in frame.columns:
            ordered = frame.sort_values("Date")
            index = pd.to_datetime(ordered["Date"])
        else:
            ordered = frame.sort_index()
            index = pd.to_datetime(ordered.index)

        open_series = pd.Series(pd.to_numeric(ordered["open"], errors="coerce").to_numpy(), index=index, name=ticker)
        close_series = pd.Series(pd.to_numeric(ordered["close"], errors="coerce").to_numpy(), index=index, name=ticker)
        volume_series = pd.Series(pd.to_numeric(ordered["volume"], errors="coerce").to_numpy(), index=index, name=ticker)

        open_series = open_series[~open_series.index.duplicated(keep="last")]
        close_series = close_series[~close_series.index.duplicated(keep="last")]
        volume_series = volume_series[~volume_series.index.duplicated(keep="last")]

        open_dict[ticker] = open_series
        close_dict[ticker] = close_series
        volume_dict[ticker] = volume_series

    close = pd.DataFrame(close_dict).sort_index()
    open_ = pd.DataFrame(open_dict).reindex_like(close)
    volume = pd.DataFrame(volume_dict).reindex_like(close)
    return {"open": open_, "close": close, "volume": volume}


def generate_signal(data: dict) -> dict[str, float]:
    if not data:
        return {}

    panels = _history_dict_to_panels(data)
    close = panels["close"]
    open_ = panels["open"]
    volume = panels["volume"]

    if close.empty:
        return {ticker: 0.0 for ticker in data}

    ret1 = close.pct_change(fill_method=None)
    vol20 = ret1.rolling(20, min_periods=20).std()
    log_volume = np.log(volume.replace(0.0, np.nan))
    intraday = close.div(open_).sub(1.0)

    latest_features = pd.DataFrame(
        {
            "rev_1_voladj": (-ret1.div(vol20.add(1e-12))).iloc[-1],
            "intraday_reversal": (-intraday).iloc[-1],
            "vol_shock_20": log_volume.sub(log_volume.rolling(20, min_periods=20).mean()).iloc[-1],
            "mom_126_21": close.shift(21).div(close.shift(126)).sub(1.0).iloc[-1],
        }
    )

    ranked = latest_features.rank(pct=True).sub(0.5)
    scores = ranked.fillna(0.0).mul(FINAL_MODEL_WEIGHTS, axis=1).sum(axis=1)
    scores = scores.reindex(list(data.keys())).fillna(0.0)
    return {ticker: float(score) for ticker, score in scores.items()}


if __name__ == "__main__":
    run_research(os.path.dirname(os.path.abspath(__file__)))