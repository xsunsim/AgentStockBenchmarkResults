# Strategy: OpenAI__GPT5_3_Codex__LinearNeutral_202605
# Model: OpenAI__GPT5_3_Codex (api: gpt-5.3-codex)
# File: strategies/OpenAI__GPT5_3_Codex__LinearNeutral_202605/strategy.py

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats


FIELDS = ("open", "high", "low", "close", "volume")
FINAL_MODEL_NAME = "smoothed_family_mean"


def align_panels(panels: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    aligned: dict[str, pd.DataFrame] = {}
    common_cols = sorted(set().union(*(set(panel.columns) for panel in panels.values())))
    common_index = pd.Index(sorted(set().union(*(panel.index for panel in panels.values()))))
    common_index = pd.to_datetime(common_index)
    for field, panel in panels.items():
        clean = panel.copy()
        clean.index = pd.to_datetime(clean.index)
        clean = clean.sort_index()
        clean = clean.reindex(index=common_index, columns=common_cols)
        aligned[field] = clean.astype(float)
    return aligned


def load_panel_data(base_dir: Path) -> dict[str, pd.DataFrame]:
    panels = {
        field: pd.read_parquet(base_dir / f"{field}.parquet")
        for field in FIELDS
    }
    return align_panels(panels)


def build_panels_from_data(data: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    field_maps: dict[str, dict[str, pd.Series]] = {field: {} for field in FIELDS}
    for ticker in sorted(data):
        frame = data[ticker]
        if frame is None or len(frame) == 0:
            continue
        if "Date" in frame.columns:
            local = frame.loc[:, ["Date", *FIELDS]].copy()
            local["Date"] = pd.to_datetime(local["Date"])
            local = local.sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date")
        else:
            local = frame.loc[:, list(FIELDS)].copy()
            local.index = pd.to_datetime(local.index)
            local = local.sort_index()
        for field in FIELDS:
            series = pd.to_numeric(local[field], errors="coerce")
            series.name = ticker
            field_maps[field][ticker] = series
    panels = {field: pd.DataFrame(mapping).sort_index() for field, mapping in field_maps.items()}
    if not panels["close"].empty:
        panels = align_panels(panels)
    return panels


def cross_sectional_zscore_panel(panel: pd.DataFrame, clip: float = 5.0) -> pd.DataFrame:
    mean = panel.mean(axis=1)
    std = panel.std(axis=1).replace(0.0, np.nan)
    z = panel.sub(mean, axis=0).div(std, axis=0)
    return z.clip(-clip, clip)


def cross_sectional_zscore_series(values: pd.Series, clip: float = 5.0) -> pd.Series:
    values = values.astype(float)
    std = values.std()
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(np.nan, index=values.index)
    z = (values - values.mean()) / std
    return z.clip(-clip, clip)


def average_panels(panels: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not panels:
        raise ValueError("panels must be non-empty")
    arrays = np.stack([panel.to_numpy(dtype=float) for panel in panels], axis=0)
    valid = np.isfinite(arrays).astype(float)
    numer = np.nansum(arrays, axis=0)
    denom = valid.sum(axis=0)
    avg = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)
    return pd.DataFrame(avg, index=panels[0].index, columns=panels[0].columns)


def weighted_average_panels(panels: Sequence[pd.DataFrame], weights: Sequence[float]) -> pd.DataFrame:
    if not panels:
        raise ValueError("panels must be non-empty")
    arr = np.stack([panel.to_numpy(dtype=float) for panel in panels], axis=0)
    w = np.asarray(weights, dtype=float).reshape(-1, 1, 1)
    valid = np.isfinite(arr).astype(float)
    numer = np.nansum(arr * w, axis=0)
    denom = np.sum(valid * w, axis=0)
    out = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)
    return pd.DataFrame(out, index=panels[0].index, columns=panels[0].columns)


def compute_feature_library(panels: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    close = panels["close"]
    open_ = panels["open"]
    high = panels["high"]
    low = panels["low"]
    volume = panels["volume"]

    ret1 = close.pct_change(fill_method=None)
    log_volume = np.log(volume.where(volume > 0))
    range_pct = (high - low) / close.shift(1)

    return {
        "rev1": -ret1,
        "rev3": -close.pct_change(3, fill_method=None),
        "intraday_rev": -(close / open_ - 1.0),
        "vol20": -ret1.rolling(20, min_periods=10).std(),
        "vol63": -ret1.rolling(63, min_periods=20).std(),
        "atr20": -range_pct.rolling(20, min_periods=10).mean(),
        "vol_shock20": log_volume - log_volume.rolling(20, min_periods=10).mean(),
        "vol_shock63": log_volume - log_volume.rolling(63, min_periods=20).mean(),
        "low_break20": close / low.rolling(20, min_periods=10).min() - 1.0,
        "low_break63": close / low.rolling(63, min_periods=20).min() - 1.0,
    }


def compute_family_components(
    features: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    z_features = {name: cross_sectional_zscore_panel(panel) for name, panel in features.items()}
    family_components = {
        "rev1": z_features["rev1"],
        "intraday_rev": z_features["intraday_rev"],
        "risk_avg": average_panels([z_features["vol20"], z_features["atr20"]]),
        "volu_avg": average_panels([z_features["vol_shock20"], z_features["vol_shock63"]]),
        "pos_avg": average_panels([z_features["low_break20"], z_features["low_break63"]]),
        "pos63": z_features["low_break63"],
    }
    return z_features, family_components


def make_research_splits(index: pd.Index) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(index)
    explicit_bounds = {
        "train": ("2020-01-02", "2022-12-30"),
        "val1": ("2023-01-03", "2023-06-30"),
        "val2": ("2023-07-03", "2023-12-29"),
        "test": ("2024-01-02", "2024-12-30"),
    }
    if dates.min() <= pd.Timestamp("2020-01-02") and dates.max() >= pd.Timestamp("2024-12-30"):
        return {
            name: (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
            for name, (start, end) in explicit_bounds.items()
        }
    n = len(dates)
    a = int(n * 0.60)
    b = int(n * 0.70)
    c = int(n * 0.80)
    positions = np.arange(n)
    return {
        "train": positions < a,
        "val1": (positions >= a) & (positions < b),
        "val2": (positions >= b) & (positions < c),
        "test": positions >= c,
    }


def annualized_sharpe(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    std = returns.std()
    if not np.isfinite(std) or std == 0.0:
        return np.nan
    return float(np.sqrt(252.0) * returns.mean() / std)


def daily_rank_ic(signal: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    values: list[float] = []
    for date in signal.index:
        x = signal.loc[date]
        y = target.loc[date]
        mask = x.notna() & y.notna()
        if mask.sum() < 20:
            values.append(np.nan)
            continue
        values.append(float(stats.spearmanr(x[mask], y[mask])[0]))
    return pd.Series(values, index=signal.index)


def rank_portfolio_returns(signal: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    ranks = signal.rank(axis=1, method="average", na_option="keep")
    counts = ranks.notna().sum(axis=1)
    centered = ranks.sub((counts + 1.0) / 2.0, axis=0)
    denom = centered.abs().sum(axis=1).replace(0.0, np.nan)
    weights = centered.div(denom, axis=0)
    return (weights * target).sum(axis=1)


def evaluate_signal(
    signal: pd.DataFrame,
    target: pd.DataFrame,
    splits: Mapping[str, np.ndarray],
) -> dict[str, float]:
    pearson_ic = signal.corrwith(target, axis=1)
    rank_ic = daily_rank_ic(signal, target)
    portfolio_returns = rank_portfolio_returns(signal, target)
    metrics: dict[str, float] = {}
    for split_name, mask in splits.items():
        metrics[f"{split_name}_pearson_ic"] = float(pearson_ic[mask].dropna().mean())
        metrics[f"{split_name}_rank_ic"] = float(rank_ic[mask].dropna().mean())
        metrics[f"{split_name}_sharpe"] = annualized_sharpe(portfolio_returns[mask])
    return metrics


def fit_nnls_family_weights(
    components: Mapping[str, pd.DataFrame],
    target: pd.DataFrame,
    train_mask: np.ndarray,
) -> pd.Series:
    feature_names = list(components)
    target_z = cross_sectional_zscore_panel(target)
    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    for date in target.index[train_mask]:
        frame = pd.concat(
            [components[name].loc[date] for name in feature_names] + [target_z.loc[date].rename("target")],
            axis=1,
        )
        frame.columns = [*feature_names, "target"]
        frame = frame.dropna()
        if len(frame) < len(feature_names) + 5:
            continue
        x_blocks.append(frame[feature_names].to_numpy())
        y_blocks.append(frame["target"].to_numpy())
    if not x_blocks:
        weights = np.ones(len(feature_names), dtype=float) / len(feature_names)
        return pd.Series(weights, index=feature_names)
    x = np.vstack(x_blocks)
    y = np.concatenate(y_blocks)
    weights, _ = optimize.nnls(x, y)
    if weights.sum() <= 0.0:
        weights = np.ones(len(feature_names), dtype=float) / len(feature_names)
    else:
        weights = weights / weights.sum()
    return pd.Series(weights, index=feature_names)


def build_candidate_signals(
    z_features: Mapping[str, pd.DataFrame],
    family_components: Mapping[str, pd.DataFrame],
    nnls_weights: pd.Series,
) -> dict[str, pd.DataFrame]:
    candidates = {
        "focused_lowbreak20": average_panels(
            [
                z_features["rev1"],
                z_features["intraday_rev"],
                z_features["vol20"],
                z_features["vol_shock63"],
                z_features["low_break20"],
            ]
        ),
        "simple_pos63": average_panels(
            [
                z_features["rev1"],
                z_features["intraday_rev"],
                z_features["vol20"],
                z_features["vol_shock20"],
                z_features["low_break63"],
            ]
        ),
        "smoothed_pos63": average_panels(
            [
                family_components["rev1"],
                family_components["intraday_rev"],
                family_components["risk_avg"],
                family_components["volu_avg"],
                family_components["pos63"],
            ]
        ),
        "smoothed_family_mean": average_panels(
            [
                family_components["rev1"],
                family_components["intraday_rev"],
                family_components["risk_avg"],
                family_components["volu_avg"],
                family_components["pos_avg"],
            ]
        ),
        "nnls_family_baseline": weighted_average_panels(
            [
                family_components["rev1"],
                family_components["intraday_rev"],
                family_components["risk_avg"],
                family_components["volu_avg"],
                family_components["pos_avg"],
            ],
            nnls_weights.reindex(
                ["rev1", "intraday_rev", "risk_avg", "volu_avg", "pos_avg"]
            ).to_numpy(),
        ),
    }
    return candidates


def format_metrics_table(frame: pd.DataFrame) -> str:
    ordered_cols = [
        "model",
        "train_rank_ic",
        "val1_rank_ic",
        "val2_rank_ic",
        "test_rank_ic",
        "train_sharpe",
        "val1_sharpe",
        "val2_sharpe",
        "test_sharpe",
    ]
    available_cols = [col for col in ordered_cols if col in frame.columns]
    with pd.option_context("display.max_rows", 200, "display.width", 200, "display.float_format", "{:.4f}".format):
        return frame.loc[:, available_cols].to_string(index=False)


def panels_to_production_input(panels: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    dates = pd.to_datetime(panels["close"].index)
    for ticker in panels["close"].columns:
        frame = pd.DataFrame(
            {
                "Date": dates,
                "open": panels["open"][ticker].to_numpy(),
                "high": panels["high"][ticker].to_numpy(),
                "low": panels["low"][ticker].to_numpy(),
                "close": panels["close"][ticker].to_numpy(),
                "volume": panels["volume"][ticker].to_numpy(),
            }
        )
        frame = frame.dropna(how="all", subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
        data[ticker] = frame
    return data


def latest_final_signal_from_panels(panels: Mapping[str, pd.DataFrame]) -> pd.Series:
    features = compute_feature_library(panels)
    z_features, family_components = compute_family_components(features)
    final_signal = average_panels(
        [
            family_components["rev1"],
            family_components["intraday_rev"],
            family_components["risk_avg"],
            family_components["volu_avg"],
            family_components["pos_avg"],
        ]
    )
    latest = final_signal.iloc[-1].replace([np.inf, -np.inf], np.nan)
    return latest.fillna(0.0).sort_index()


def run_research(base_dir: Path | None = None) -> None:
    base_dir = Path(__file__).resolve().parent if base_dir is None else Path(base_dir)
    panels = load_panel_data(base_dir)
    features = compute_feature_library(panels)
    z_features, family_components = compute_family_components(features)
    target = panels["close"].shift(-1) / panels["close"] - 1.0
    splits = make_research_splits(panels["close"].index)

    print(f"rows={panels['close'].shape[0]} cols={panels['close'].shape[1]} start={panels['close'].index.min().date()} end={panels['close'].index.max().date()}")

    screen_names = [
        "rev1",
        "intraday_rev",
        "rev3",
        "vol20",
        "vol63",
        "atr20",
        "vol_shock20",
        "vol_shock63",
        "low_break20",
        "low_break63",
    ]
    single_feature_rows: list[dict[str, float | str]] = []
    for name in screen_names:
        metrics = evaluate_signal(z_features[name], target, splits)
        metrics["model"] = name
        single_feature_rows.append(metrics)
    single_feature_frame = pd.DataFrame(single_feature_rows)
    single_feature_frame["validation_score"] = single_feature_frame["val1_rank_ic"] + single_feature_frame["val2_rank_ic"]
    single_feature_frame = single_feature_frame.sort_values(
        ["validation_score", "test_rank_ic", "train_rank_ic"],
        ascending=False,
    )

    print("\nTop single features:")
    print(format_metrics_table(single_feature_frame.head(8)))

    nnls_weights = fit_nnls_family_weights(
        {
            "rev1": family_components["rev1"],
            "intraday_rev": family_components["intraday_rev"],
            "risk_avg": family_components["risk_avg"],
            "volu_avg": family_components["volu_avg"],
            "pos_avg": family_components["pos_avg"],
        },
        target,
        splits["train"],
    )
    print("\nTrain-fit NNLS family weights:")
    print(nnls_weights.round(4).to_string())

    candidate_signals = build_candidate_signals(z_features, family_components, nnls_weights)
    candidate_rows: list[dict[str, float | str]] = []
    for name, signal in candidate_signals.items():
        metrics = evaluate_signal(signal, target, splits)
        metrics["model"] = name
        candidate_rows.append(metrics)
    candidate_frame = pd.DataFrame(candidate_rows).sort_values(
        ["val2_rank_ic", "val1_rank_ic", "test_rank_ic"],
        ascending=False,
    )

    print("\nCandidate models:")
    print(format_metrics_table(candidate_frame))
    print(f"\nSelected model: {FINAL_MODEL_NAME}")

    production_input = panels_to_production_input(panels)
    production_signal = pd.Series(generate_signal(production_input)).sort_values(ascending=False)
    print("\nProduction interface smoke test:")
    print(f"scores={len(production_signal)} top3={production_signal.head(3).round(4).to_dict()} bottom3={production_signal.tail(3).round(4).to_dict()}")


def generate_signal(data: Mapping[str, pd.DataFrame]) -> dict[str, float]:
    all_tickers = sorted(data)
    panels = build_panels_from_data(data)
    if not panels or panels["close"].empty:
        return {ticker: 0.0 for ticker in all_tickers}
    latest_scores = latest_final_signal_from_panels(panels)
    output = {ticker: 0.0 for ticker in all_tickers}
    output.update({ticker: float(score) for ticker, score in latest_scores.items()})
    return output


if __name__ == "__main__":
    run_research()