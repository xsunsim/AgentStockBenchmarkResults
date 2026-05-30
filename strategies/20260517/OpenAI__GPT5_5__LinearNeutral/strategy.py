# Strategy: OpenAI__GPT5_5__LinearNeutral_202605
# Model: OpenAI__GPT5_5 (api: gpt-5.5)
# File: strategies/OpenAI__GPT5_5__LinearNeutral_202605/strategy.py

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
VOLUME_WINDOW = 21
VOLUME_MIN_OBS = 10
MOM_SKIP = 21
MOM_LOOKBACK = 252
FINAL_FEATURE_WEIGHTS = {
    "rev1": 1.0,
    "volratio": 1.0,
    "mom12_1": 0.5,
    "range": 0.5,
}


def cross_sectional_rank_score(values):
    if isinstance(values, pd.DataFrame):
        ranks = values.rank(axis=1, pct=True, method="average")
        return (ranks - 0.5) * np.sqrt(12.0)
    series = pd.Series(values, dtype=float)
    ranks = series.rank(pct=True, method="average")
    return (ranks - 0.5) * np.sqrt(12.0)


def build_candidate_features(open_, high, low, close, volume):
    log_volume = np.log(volume.replace(0, np.nan))
    range_ = high / low - 1.0
    close_loc = ((close - low) / (high - low).replace(0, np.nan)) - 0.5

    return {
        "rev1": -(close / close.shift(1) - 1.0),
        "rev_intraday": -(close / open_ - 1.0),
        "close_loc_rev": -close_loc,
        "volratio": volume / volume.rolling(VOLUME_WINDOW, min_periods=VOLUME_MIN_OBS).mean() - 1.0,
        "volshock": log_volume - log_volume.rolling(VOLUME_WINDOW, min_periods=VOLUME_MIN_OBS).mean(),
        "mom12_1": close.shift(MOM_SKIP) / close.shift(MOM_LOOKBACK) - 1.0,
        "range": range_,
    }


def split_masks(index):
    idx = pd.DatetimeIndex(index)
    return {
        "train": idx.year <= 2022,
        "val1": (idx.year == 2023) & (idx.month <= 6),
        "val2": (idx.year == 2023) & (idx.month >= 7),
        "test": idx.year >= 2024,
    }


def daily_rank_ic(score, fwd_returns):
    sx = cross_sectional_rank_score(score)
    sy = cross_sectional_rank_score(fwd_returns)
    numerator = (sx * sy).sum(axis=1)
    denominator = np.sqrt((sx ** 2).sum(axis=1) * (sy ** 2).sum(axis=1))
    return numerator / denominator


def portfolio_returns(score, fwd_returns):
    ranks = score.rank(axis=1, method="first", ascending=False, na_option="keep")
    universe = score.notna().sum(axis=1).astype(float)
    weights = ranks.mul(-1.0).add((universe + 1.0) / 2.0, axis=0)
    gross = weights.abs().sum(axis=1).replace(0.0, np.nan)
    weights = weights.div(gross, axis=0)
    return (weights * fwd_returns).sum(axis=1)


def summarize_model(name, score, fwd_returns, masks):
    ic = daily_rank_ic(score, fwd_returns)
    pnl = portfolio_returns(score, fwd_returns)
    row = {"model": name}
    for split_name, mask in masks.items():
        split_ic = ic[mask].dropna()
        split_pnl = pnl[mask].dropna()
        row[f"{split_name}_ic"] = split_ic.mean()
        if len(split_pnl) > 5 and split_pnl.std(ddof=1) > 0:
            row[f"{split_name}_sh"] = np.sqrt(252.0) * split_pnl.mean() / split_pnl.std(ddof=1)
            row[f"{split_name}_ret_bps"] = split_pnl.mean() * 10000.0
        else:
            row[f"{split_name}_sh"] = np.nan
            row[f"{split_name}_ret_bps"] = np.nan
    return row


def fit_panel_ridge(feature_scores, target_scores, feature_names, train_mask, lam):
    x = np.column_stack(
        [feature_scores[name][train_mask].to_numpy(dtype=float).reshape(-1) for name in feature_names]
    )
    y = target_scores[train_mask].to_numpy(dtype=float).reshape(-1)
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[valid]
    y = y[valid]
    beta = np.linalg.solve(x.T @ x + lam * np.eye(len(feature_names)), x.T @ y)
    return pd.Series(beta, index=feature_names)


def yearly_stats(score, fwd_returns):
    ic = daily_rank_ic(score, fwd_returns)
    pnl = portfolio_returns(score, fwd_returns)
    rows = []
    years = pd.DatetimeIndex(score.index).year
    for year in np.unique(years):
        year_mask = years == year
        year_ic = ic[year_mask].dropna()
        year_pnl = pnl[year_mask].dropna()
        row = {"year": int(year)}
        row["ic"] = year_ic.mean()
        if len(year_pnl) > 5 and year_pnl.std(ddof=1) > 0:
            row["sharpe"] = np.sqrt(252.0) * year_pnl.mean() / year_pnl.std(ddof=1)
            row["ret_bps"] = year_pnl.mean() * 10000.0
        else:
            row["sharpe"] = np.nan
            row["ret_bps"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def combine_feature_scores(feature_scores, weights):
    reference = next(iter(feature_scores.values()))
    score_sum = pd.DataFrame(0.0, index=reference.index, columns=reference.columns)
    weight_sum = pd.DataFrame(0.0, index=reference.index, columns=reference.columns)
    for feature_name, weight in weights.items():
        feature_score = feature_scores[feature_name]
        valid = feature_score.notna()
        score_sum = score_sum.add((weight * feature_score).where(valid, 0.0), fill_value=0.0)
        weight_sum = weight_sum.add((abs(weight) * valid).astype(float), fill_value=0.0)
    return score_sum.div(weight_sum.replace(0.0, np.nan))


def final_ensemble_score(feature_scores):
    return combine_feature_scores(feature_scores, FINAL_FEATURE_WEIGHTS)


def load_research_data(base_dir):
    open_ = pd.read_parquet(base_dir / "open.parquet").sort_index()
    high = pd.read_parquet(base_dir / "high.parquet").sort_index()
    low = pd.read_parquet(base_dir / "low.parquet").sort_index()
    close = pd.read_parquet(base_dir / "close.parquet").sort_index()
    volume = pd.read_parquet(base_dir / "volume.parquet").sort_index()
    return open_, high, low, close, volume


def run_research():
    open_, high, low, close, volume = load_research_data(BASE_DIR)
    fwd_returns = close.shift(-1) / close - 1.0
    raw_features = build_candidate_features(open_, high, low, close, volume)
    feature_scores = {name: cross_sectional_rank_score(values) for name, values in raw_features.items()}
    masks = split_masks(close.index)

    print("DATA SUMMARY")
    print(
        f"rows={close.shape[0]} cols={close.shape[1]} "
        f"start={close.index.min().date()} end={close.index.max().date()} "
        f"missing={int(close.isna().sum().sum())}"
    )
    print()

    screening_rows = []
    for name in ["rev1", "rev_intraday", "close_loc_rev", "volratio", "volshock", "mom12_1", "range"]:
        screening_rows.append(summarize_model(name, feature_scores[name], fwd_returns, masks))
    screening = pd.DataFrame(screening_rows)
    screening = screening[
        [
            "model",
            "train_ic",
            "val1_ic",
            "val2_ic",
            "test_ic",
            "train_sh",
            "val1_sh",
            "val2_sh",
            "test_sh",
        ]
    ]
    print("FEATURE SCREEN")
    print(screening.sort_values(["val2_ic", "val1_ic"], ascending=False).to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print()

    model_scores = {
        "rev_plus_volume": combine_feature_scores(feature_scores, {"rev1": 1.0, "volratio": 1.0}),
        "final_ensemble": final_ensemble_score(feature_scores),
    }

    ridge_features = ["rev1", "rev_intraday", "volratio", "volshock", "mom12_1", "range"]
    ridge_target = cross_sectional_rank_score(fwd_returns)
    ridge_beta = fit_panel_ridge(feature_scores, ridge_target, ridge_features, masks["train"], lam=300.0)
    model_scores["ridge_baseline"] = combine_feature_scores(feature_scores, ridge_beta.to_dict())

    model_rows = [summarize_model(name, score, fwd_returns, masks) for name, score in model_scores.items()]
    models = pd.DataFrame(model_rows)
    models = models[
        [
            "model",
            "train_ic",
            "val1_ic",
            "val2_ic",
            "test_ic",
            "train_sh",
            "val1_sh",
            "val2_sh",
            "test_sh",
            "train_ret_bps",
            "val1_ret_bps",
            "val2_ret_bps",
            "test_ret_bps",
        ]
    ]
    print("MODEL COMPARISON")
    print(models.sort_values(["val2_ic", "val1_ic"], ascending=False).to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print()

    print("RIDGE COEFFICIENTS")
    print(ridge_beta.round(4).to_string())
    print()

    print("FINAL YEARLY STATS")
    print(yearly_stats(model_scores["final_ensemble"], fwd_returns).to_string(index=False, float_format=lambda x: f"{x: .4f}"))


def _lag_value(values, lag):
    if len(values) <= lag:
        return np.nan
    value = values[-(lag + 1)]
    return value if np.isfinite(value) else np.nan


def _rolling_mean_last(values, window, min_obs):
    if len(values) == 0:
        return np.nan
    tail = values[-window:]
    valid = np.isfinite(tail)
    if valid.sum() < min_obs:
        return np.nan
    return tail[valid].mean()


def _latest_raw_features_from_history(df):
    columns = {col.lower(): col for col in df.columns}
    if "date" in columns:
        local = df.sort_values(columns["date"])
    else:
        local = df

    close_values = pd.to_numeric(local[columns["close"]], errors="coerce").to_numpy(dtype=float)
    high_values = pd.to_numeric(local[columns["high"]], errors="coerce").to_numpy(dtype=float)
    low_values = pd.to_numeric(local[columns["low"]], errors="coerce").to_numpy(dtype=float)
    volume_values = pd.to_numeric(local[columns["volume"]], errors="coerce").to_numpy(dtype=float)

    current_close = _lag_value(close_values, 0)
    prev_close = _lag_value(close_values, 1)
    current_high = _lag_value(high_values, 0)
    current_low = _lag_value(low_values, 0)
    current_volume = _lag_value(volume_values, 0)
    mom_recent = _lag_value(close_values, MOM_SKIP)
    mom_past = _lag_value(close_values, MOM_LOOKBACK)
    volume_mean = _rolling_mean_last(volume_values, VOLUME_WINDOW, VOLUME_MIN_OBS)

    rev1 = np.nan
    if np.isfinite(current_close) and np.isfinite(prev_close) and prev_close != 0.0:
        rev1 = -(current_close / prev_close - 1.0)

    volratio = np.nan
    if np.isfinite(current_volume) and np.isfinite(volume_mean) and volume_mean != 0.0:
        volratio = current_volume / volume_mean - 1.0

    mom12_1 = np.nan
    if np.isfinite(mom_recent) and np.isfinite(mom_past) and mom_past != 0.0:
        mom12_1 = mom_recent / mom_past - 1.0

    range_ = np.nan
    if np.isfinite(current_high) and np.isfinite(current_low) and current_low > 0.0:
        range_ = current_high / current_low - 1.0

    return {
        "rev1": rev1,
        "volratio": volratio,
        "mom12_1": mom12_1,
        "range": range_,
    }


if __name__ == "__main__":
    run_research()


def generate_signal(data):
    tickers = list(data.keys())
    raw_features = {name: pd.Series(index=tickers, dtype=float) for name in FINAL_FEATURE_WEIGHTS}

    for ticker, history in data.items():
        latest = _latest_raw_features_from_history(history)
        for feature_name, feature_value in latest.items():
            raw_features[feature_name].loc[ticker] = feature_value

    ranked_features = {name: cross_sectional_rank_score(values) for name, values in raw_features.items()}
    score_sum = pd.Series(0.0, index=tickers, dtype=float)
    weight_sum = pd.Series(0.0, index=tickers, dtype=float)

    for feature_name, weight in FINAL_FEATURE_WEIGHTS.items():
        feature_score = ranked_features[feature_name]
        valid = feature_score.notna()
        score_sum.loc[valid] += weight * feature_score.loc[valid]
        weight_sum.loc[valid] += abs(weight)

    final_score = score_sum.div(weight_sum.replace(0.0, np.nan)).fillna(0.0)
    return final_score.to_dict()