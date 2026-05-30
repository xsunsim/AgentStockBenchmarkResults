# Strategy: OpenAI__GPT4o__LinearNeutral_202605
# Model: OpenAI__GPT4o (api: gpt-4o)
# File: strategies/OpenAI__GPT4o__LinearNeutral_202605/strategy.py

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("/tmp/tmpfnrapo0z")
SPLITS = {
    "train": ("2020-01-02", "2022-12-30"),
    "val1": ("2023-01-03", "2023-06-30"),
    "val2": ("2023-07-03", "2023-12-29"),
    "test": ("2024-01-02", "2024-12-31"),
}
MODEL_SPECS = {
    "eq_reversal_volume": ["ret1_over_range", "vol_spike", "range_x_vol"],
    "eq_core4": ["ret1_over_range", "vol_spike", "range_x_vol", "mom126"],
    "eq_core5": [
        "ret1_over_range",
        "intraday_rev_range",
        "vol_spike",
        "range_x_vol",
        "mom126",
    ],
    "eq_alt": ["ret1_over_vol", "vol_spike", "mom126"],
}
SELECTED_MODEL = "eq_core5"
FINAL_FEATURES = MODEL_SPECS[SELECTED_MODEL]


def _safe_log_ratio(numerator, denominator):
    ratio = numerator / denominator
    ratio = ratio.where(ratio > 0)
    return np.log(ratio)


def _safe_log1p_ratio(numerator, denominator):
    ratio = numerator / denominator
    ratio = ratio.where(ratio >= 0)
    return np.log1p(ratio)


def _nanmean(values):
    if len(values) == 0:
        return np.nan
    return float(np.nanmean(values))


def _sharpe(values, annualize=False):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan
    std = np.nanstd(values, ddof=1)
    if not np.isfinite(std) or std == 0:
        return np.nan
    ratio = np.nanmean(values) / std
    if annualize:
        ratio *= np.sqrt(252.0)
    return float(ratio)


def load_panel(data_dir=DATA_DIR):
    panel = {
        name: pd.read_parquet(data_dir / f"{name}.parquet").sort_index()
        for name in ["open", "high", "low", "close", "volume"]
    }
    valid_columns = panel["close"].columns[panel["close"].notna().any(axis=0)]
    common_index = panel["close"].index
    for name, frame in panel.items():
        frame = frame.loc[common_index, valid_columns]
        frame.index = pd.to_datetime(frame.index)
        panel[name] = frame.astype(float)
    return panel


def summarize_panel(panel):
    close = panel["close"]
    coverage = close.notna().sum(axis=1)
    return {
        "dates": len(close.index),
        "tickers": len(close.columns),
        "start": str(close.index.min().date()),
        "end": str(close.index.max().date()),
        "min_coverage": int(coverage.min()),
        "median_coverage": float(coverage.median()),
        "max_coverage": int(coverage.max()),
    }


def compute_panel_features(panel):
    open_ = panel["open"]
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    volume = panel["volume"]

    ret1 = close / close.shift(1) - 1.0
    intraday = close / open_ - 1.0
    logret = _safe_log_ratio(close, close.shift(1))
    range_raw = (high - low) / close.replace(0.0, np.nan)
    avg_range20 = range_raw.rolling(20, min_periods=10).mean()
    vol_ma20 = volume.rolling(20, min_periods=10).mean()
    vol20 = logret.rolling(20, min_periods=10).std()
    close_pos_denom = (high - low).replace(0.0, np.nan)

    features = {
        "ret1_over_range": -ret1 / avg_range20,
        "ret1_over_vol": -ret1 / vol20,
        "intraday_rev_range": -intraday / avg_range20,
        "vol_spike": _safe_log_ratio(volume, vol_ma20),
        "range_x_vol": range_raw * _safe_log1p_ratio(volume, vol_ma20),
        "mom126": close / close.shift(126) - 1.0,
        "mom252": close / close.shift(252) - 1.0,
        "neg_close_pos": -((close - low) / close_pos_denom - 0.5),
    }
    return {
        name: frame.replace([np.inf, -np.inf], np.nan) for name, frame in features.items()
    }


def cross_section_rank_scale(frame):
    ranked = frame.rank(axis=1, pct=True, method="average")
    return ((ranked - 0.5) * 2.0).where(frame.notna())


def combine_ranked_features(ranked_features, feature_names):
    score = None
    counts = None
    for name in feature_names:
        feature = ranked_features[name]
        if score is None:
            score = feature.fillna(0.0).copy()
            counts = feature.notna().astype(float)
        else:
            score = score.add(feature.fillna(0.0), fill_value=0.0)
            counts = counts.add(feature.notna().astype(float), fill_value=0.0)
    return score.div(counts.where(counts > 0.0))


def forward_returns(close):
    return close.shift(-1) / close - 1.0


def evaluate_signal(scores, target, start, end):
    dates = scores.loc[start:end].index.intersection(target.loc[start:end].index)
    daily_ic = []
    daily_pnl = []

    for dt in dates:
        s = scores.loc[dt].replace([np.inf, -np.inf], np.nan)
        y = target.loc[dt].replace([np.inf, -np.inf], np.nan)
        mask = s.notna() & y.notna()
        if int(mask.sum()) < 20:
            continue
        s = s[mask]
        y = y[mask]
        if s.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
            continue
        ic = s.rank(method="average").corr(y.rank(method="average"))
        if pd.notna(ic):
            daily_ic.append(float(ic))
        ranks_desc = s.rank(method="first", ascending=False)
        weights = (len(s) + 1.0) / 2.0 - ranks_desc
        pnl = float((weights * y).sum() / weights.abs().sum())
        daily_pnl.append(pnl)

    return {
        "days": len(daily_pnl),
        "ic": _nanmean(daily_ic),
        "ic_ir": _sharpe(daily_ic, annualize=False),
        "pnl_mean_bps": _nanmean(daily_pnl) * 10000.0,
        "pnl_sharpe": _sharpe(daily_pnl, annualize=True),
        "hit_rate": float(np.mean(np.asarray(daily_pnl) > 0.0)) if daily_pnl else np.nan,
    }


def fit_pooled_ridge(ranked_features, ranked_target, feature_names, start, end, ridge_lambda):
    x_rows = []
    y_rows = []

    for dt in ranked_target.loc[start:end].index:
        parts = [ranked_features[name].loc[dt] for name in feature_names]
        frame = pd.concat(parts + [ranked_target.loc[dt]], axis=1)
        frame.columns = feature_names + ["target"]
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < len(feature_names) + 5:
            continue
        x_rows.append(frame[feature_names].to_numpy())
        y_rows.append(frame["target"].to_numpy())

    x_mat = np.vstack(x_rows)
    y_vec = np.concatenate(y_rows)
    xtx = x_mat.T @ x_mat
    beta = np.linalg.solve(
        xtx + ridge_lambda * np.eye(len(feature_names)),
        x_mat.T @ y_vec,
    )
    return pd.Series(beta, index=feature_names)


def linear_signal_from_weights(ranked_features, weights):
    signal = None
    for name, weight in weights.items():
        component = ranked_features[name].fillna(0.0) * float(weight)
        if signal is None:
            signal = component.copy()
        else:
            signal = signal.add(component, fill_value=0.0)
    return signal


def feature_screen_table(features, target):
    rows = []
    ranked_features = {name: cross_section_rank_scale(frame) for name, frame in features.items()}
    for name in [
        "ret1_over_range",
        "ret1_over_vol",
        "intraday_rev_range",
        "vol_spike",
        "range_x_vol",
        "mom126",
        "mom252",
        "neg_close_pos",
    ]:
        row = {"feature": name}
        for split in ["train", "val1", "val2"]:
            metrics = evaluate_signal(ranked_features[name], target, *SPLITS[split])
            row[f"{split}_ic"] = metrics["ic"]
        rows.append(row)
    result = pd.DataFrame(rows)
    result["validation_sum"] = result["val1_ic"].fillna(0.0) + result["val2_ic"].fillna(0.0)
    return result.sort_values(
        ["validation_sum", "train_ic"],
        ascending=False,
    ).drop(columns=["validation_sum"])


def model_selection_table(ranked_features, target):
    rows = []
    for model_name, feature_names in MODEL_SPECS.items():
        score = combine_ranked_features(ranked_features, feature_names)
        row = {"model": model_name, "features": ",".join(feature_names)}
        for split in ["train", "val1", "val2"]:
            metrics = evaluate_signal(score, target, *SPLITS[split])
            row[f"{split}_ic"] = metrics["ic"]
            row[f"{split}_sharpe"] = metrics["pnl_sharpe"]
        rows.append(row)

    ridge_names = MODEL_SPECS[SELECTED_MODEL]
    ranked_target = cross_section_rank_scale(target)
    ridge_weights = fit_pooled_ridge(
        ranked_features,
        ranked_target,
        ridge_names,
        *SPLITS["train"],
        ridge_lambda=1000.0,
    )
    ridge_score = linear_signal_from_weights(ranked_features, ridge_weights)
    ridge_row = {
        "model": "ridge_core5_l1000",
        "features": ",".join(f"{name}:{ridge_weights[name]:.4f}" for name in ridge_names),
    }
    for split in ["train", "val1", "val2"]:
        metrics = evaluate_signal(ridge_score, target, *SPLITS[split])
        ridge_row[f"{split}_ic"] = metrics["ic"]
        ridge_row[f"{split}_sharpe"] = metrics["pnl_sharpe"]
    rows.append(ridge_row)

    return pd.DataFrame(rows).sort_values(["val2_ic", "val1_ic"], ascending=False)


def yearly_breakdown(scores, target):
    rows = []
    years = sorted({int(dt.year) for dt in target.index})
    for year in years:
        metrics = evaluate_signal(scores, target, f"{year}-01-01", f"{year}-12-31")
        rows.append(
            {
                "year": year,
                "ic": metrics["ic"],
                "pnl_sharpe": metrics["pnl_sharpe"],
                "pnl_mean_bps": metrics["pnl_mean_bps"],
                "hit_rate": metrics["hit_rate"],
            }
        )
    return pd.DataFrame(rows)


def _format_frame(frame):
    return frame.to_string(
        index=False,
        float_format=lambda x: f"{x:0.4f}",
    )


def run_research(data_dir=DATA_DIR):
    panel = load_panel(data_dir)
    summary = summarize_panel(panel)
    features = compute_panel_features(panel)
    target = forward_returns(panel["close"])
    ranked_features = {name: cross_section_rank_scale(frame) for name, frame in features.items()}

    print("PANEL_SUMMARY")
    print(pd.Series(summary).to_string())
    print()

    print("FEATURE_SCREEN_TRAIN_VAL")
    print(_format_frame(feature_screen_table(features, target)))
    print()

    print("MODEL_SELECTION_TRAIN_VAL")
    print(_format_frame(model_selection_table(ranked_features, target)))
    print()

    selected_scores = combine_ranked_features(ranked_features, FINAL_FEATURES)
    test_metrics = evaluate_signal(selected_scores, target, *SPLITS["test"])
    print("SELECTED_MODEL")
    print(SELECTED_MODEL)
    print(",".join(FINAL_FEATURES))
    print()

    print("HELD_OUT_TEST_2024")
    print(pd.Series(test_metrics).to_string(float_format=lambda x: f"{x:0.4f}"))
    print()

    print("YEARLY_BREAKDOWN_SELECTED_MODEL")
    print(_format_frame(yearly_breakdown(selected_scores, target)))


def _last_feature_values(history):
    history = history.sort_values("Date").copy()
    close = pd.to_numeric(history["close"], errors="coerce").astype(float)
    open_ = pd.to_numeric(history["open"], errors="coerce").astype(float)
    high = pd.to_numeric(history["high"], errors="coerce").astype(float)
    low = pd.to_numeric(history["low"], errors="coerce").astype(float)
    volume = pd.to_numeric(history["volume"], errors="coerce").astype(float)

    ret1 = close / close.shift(1) - 1.0
    intraday = close / open_ - 1.0
    range_raw = (high - low) / close.replace(0.0, np.nan)
    avg_range20 = range_raw.rolling(20, min_periods=10).mean()
    vol_ma20 = volume.rolling(20, min_periods=10).mean()

    vol_ratio = volume / vol_ma20
    vol_ratio = vol_ratio.where(vol_ratio > 0.0)
    vol_ratio_plus = (volume / vol_ma20).where((volume / vol_ma20) >= 0.0)

    features = {
        "ret1_over_range": (-ret1 / avg_range20).iloc[-1],
        "intraday_rev_range": (-intraday / avg_range20).iloc[-1],
        "vol_spike": np.log(vol_ratio).iloc[-1] if len(vol_ratio) else np.nan,
        "range_x_vol": (range_raw * np.log1p(vol_ratio_plus)).iloc[-1],
        "mom126": (close / close.shift(126) - 1.0).iloc[-1],
    }
    return {name: float(value) if pd.notna(value) else np.nan for name, value in features.items()}

def generate_signal(data):
    feature_rows = {}
    for ticker, history in data.items():
        if history is None or len(history) == 0:
            feature_rows[ticker] = {name: np.nan for name in FINAL_FEATURES}
            continue
        feature_rows[ticker] = _last_feature_values(history)

    feature_frame = pd.DataFrame.from_dict(feature_rows, orient="index", columns=FINAL_FEATURES)
    scores = pd.Series(0.0, index=feature_frame.index, dtype=float)
    counts = pd.Series(0.0, index=feature_frame.index, dtype=float)

    for name in FINAL_FEATURES:
        values = feature_frame[name].replace([np.inf, -np.inf], np.nan)
        ranked = (values.rank(pct=True, method="average") - 0.5) * 2.0
        scores = scores.add(ranked.fillna(0.0), fill_value=0.0)
        counts = counts.add(values.notna().astype(float), fill_value=0.0)

    scores = scores.div(counts.where(counts > 0.0)).fillna(0.0)
    return {ticker: float(score) for ticker, score in scores.items()}


if __name__ == "__main__":
    run_research()