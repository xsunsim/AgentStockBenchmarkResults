# Strategy: Anthropic__Opus4_6__LinearNeutral_202605
# Model: Anthropic__Opus4_6 (api: claude-opus-4-6)
# File: strategies/Anthropic__Opus4_6__LinearNeutral_202605/strategy.py

import numpy as np
import pandas as pd


def generate_signal(data):
    """
    Multi-factor mean-reversion signal with momentum and volume overlays.
    Combines 11 features organized into 2 correlated clusters (averaged first)
    plus 5 independent signals, all cross-sectionally ranked.

    Higher score = more long.
    """
    tickers = list(data.keys())
    if not tickers:
        return {}

    sample_df = data[tickers[0]]
    dates = sample_df["Date"].values

    n_dates = len(dates)
    n_tickers = len(tickers)

    close = np.full((n_dates, n_tickers), np.nan)
    open_ = np.full((n_dates, n_tickers), np.nan)
    high = np.full((n_dates, n_tickers), np.nan)
    low = np.full((n_dates, n_tickers), np.nan)
    volume = np.full((n_dates, n_tickers), np.nan)

    for j, ticker in enumerate(tickers):
        df = data[ticker]
        close[:, j] = df["close"].values
        open_[:, j] = df["open"].values
        high[:, j] = df["high"].values
        low[:, j] = df["low"].values
        volume[:, j] = df["volume"].values

    if n_dates < 130:
        return {t: 0.0 for t in tickers}

    def _safe_div(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = a / b
            r[~np.isfinite(r)] = np.nan
        return r

    def _rolling_mean(arr, w):
        out = np.full_like(arr, np.nan)
        if arr.shape[0] < w:
            return out
        arr = arr.astype(np.float64)
        cs = np.nancumsum(arr, axis=0)
        cnt = np.nancumsum((~np.isnan(arr)).astype(np.float64), axis=0)
        out[w - 1 :] = (
            cs[w - 1 :] - np.concatenate([np.zeros((1, arr.shape[1])), cs[:-w]], axis=0)
        ) / np.maximum(
            cnt[w - 1 :]
            - np.concatenate([np.zeros((1, arr.shape[1])), cnt[:-w]], axis=0),
            1,
        )
        return out

    def _rolling_std(arr, w):
        mean = _rolling_mean(arr, w)
        mean_sq = _rolling_mean(arr**2, w)
        with np.errstate(invalid="ignore"):
            var = mean_sq - mean**2
            var = np.maximum(var, 0)
            return np.sqrt(var)

    def _cs_rank_last(arr_2d):
        """Cross-sectional rank of last row, returned as pct rank centered at 0."""
        row = arr_2d[-1, :]
        valid = ~np.isnan(row)
        ranks = np.full(len(row), np.nan)
        if valid.sum() < 10:
            return ranks
        from scipy.stats import rankdata

        ranks[valid] = rankdata(row[valid]) / valid.sum() - 0.5
        return ranks

    def _pct_change(arr, periods=1):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = arr[periods:] / arr[:-periods] - 1
            r[~np.isfinite(r)] = np.nan
        pad = np.full((periods, arr.shape[1]), np.nan)
        return np.concatenate([pad, r], axis=0)

    ret = _pct_change(close, 1)
    vol_20d = _rolling_std(ret, 20)
    vol_20d_avg = _rolling_mean(volume, 20)

    # ---- FEATURES (last row only for efficiency) ----
    scores = np.zeros(n_tickers)
    count = np.zeros(n_tickers)

    def add_feature(raw_signal, weight=1.0):
        nonlocal scores, count
        valid = ~np.isnan(raw_signal)
        if valid.sum() < 10:
            return
        from scipy.stats import rankdata

        ranked = np.full(len(raw_signal), np.nan)
        ranked[valid] = rankdata(raw_signal[valid]) / valid.sum() - 0.5
        mask = ~np.isnan(ranked)
        scores[mask] += ranked[mask] * weight
        count[mask] += abs(weight)

    # --- Cluster 1: Short-term reversal (averaged) ---
    # rev_1d: -1d return
    c1_1 = -ret[-1, :]

    # intra_rev: -(close/open - 1) today
    c1_2 = -(close[-1, :] / open_[-1, :] - 1)

    # close_pos_rev: -(close-low)/(high-low)
    rng = high[-1, :] - low[-1, :]
    c1_3 = -(close[-1, :] - low[-1, :]) / (rng + 1e-10)

    # Average cluster 1 via rank
    from scipy.stats import rankdata

    def rank_pct(arr):
        valid = ~np.isnan(arr)
        out = np.full(len(arr), np.nan)
        if valid.sum() < 10:
            return out
        out[valid] = rankdata(arr[valid]) / valid.sum() - 0.5
        return out

    c1 = np.nanmean([rank_pct(c1_1), rank_pct(c1_2), rank_pct(c1_3)], axis=0)

    # --- Cluster 2: Medium-term reversal (averaged) ---
    # rev_3d
    ret_3d = _pct_change(close, 3)
    c2_1 = -ret_3d[-1, :]

    # rev_5d_vn
    ret_5d = _pct_change(close, 5)
    c2_2 = _safe_div(-ret_5d[-1, :], vol_20d[-1, :])

    # dist_ma5_vn
    ma5 = _rolling_mean(close, 5)
    c2_3 = _safe_div(-(close[-1, :] / ma5[-1, :] - 1), vol_20d[-1, :])

    c2 = np.nanmean([rank_pct(c2_1), rank_pct(c2_2), rank_pct(c2_3)], axis=0)

    # --- Independent signals ---

    # vol_ratio: volume / 20d avg volume
    ind_1 = _safe_div(volume[-1, :], vol_20d_avg[-1, :])

    # vp_div_neg: -(price_mom_5d_rank - vol_mom_5d_rank)
    vol_mom_5d = (
        _safe_div(volume[-1, :], volume[-6, :]) - 1
        if n_dates > 6
        else np.full(n_tickers, np.nan)
    )
    price_mom_5d = ret_5d[-1, :]
    ind_2 = -(rank_pct(price_mom_5d) - rank_pct(vol_mom_5d))

    # skew_20d
    if n_dates >= 20:
        ret_window = ret[-20:, :]
        with np.errstate(invalid="ignore"):
            m = np.nanmean(ret_window, axis=0)
            s = np.nanstd(ret_window, axis=0)
            n_valid = np.sum(~np.isnan(ret_window), axis=0).astype(float)
            skew_vals = np.where(
                (s > 1e-10) & (n_valid > 5),
                np.nanmean(((ret_window - m) / (s + 1e-10)) ** 3, axis=0),
                np.nan,
            )
        ind_3 = skew_vals
    else:
        ind_3 = np.full(n_tickers, np.nan)

    # mom_120d
    if n_dates > 120:
        ind_4 = close[-1, :] / close[-121, :] - 1
    else:
        ind_4 = np.full(n_tickers, np.nan)

    # drawdown_rev: -(close / 20d max - 1)  [negate so oversold stocks score high]
    if n_dates >= 20:
        rolling_max = np.nanmax(close[-20:, :], axis=0)
        ind_5 = -(close[-1, :] / rolling_max - 1)
    else:
        ind_5 = np.full(n_tickers, np.nan)

    # Combine: cluster averages get weight 1, each independent gets weight 1
    # Total 7 components
    components = [
        (c1, 1.0),  # cluster 1 avg
        (c2, 1.0),  # cluster 2 avg
        (ind_1, 1.0),  # vol_ratio
        (ind_2, 1.0),  # vp_div_neg (already rank-based)
        (ind_3, 1.0),  # skew_20d
        (ind_4, 1.0),  # mom_120d
        (ind_5, 1.0),  # drawdown_rev
    ]

    scores = np.zeros(n_tickers)
    count = np.zeros(n_tickers)

    for raw, w in components:
        ranked = (
            rank_pct(raw) if raw is not ind_2 else raw
        )  # ind_2 is already rank-based
        valid = ~np.isnan(ranked)
        scores[valid] += np.nan_to_num(ranked[valid]) * w
        count[valid] += w

    final = np.where(count > 0, scores / count, 0.0)

    return {tickers[j]: float(final[j]) for j in range(n_tickers)}


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    close = pd.read_parquet("close.parquet")
    open_ = pd.read_parquet("open.parquet")
    high = pd.read_parquet("high.parquet")
    low = pd.read_parquet("low.parquet")
    volume = pd.read_parquet("volume.parquet")

    ret = close.pct_change(fill_method=None)
    fwd_ret = ret.shift(-1)
    vol_20d = ret.rolling(20).std()

    def cs_rank(df):
        return df.rank(axis=1, pct=True) - 0.5

    vol_20d_avg = volume.rolling(20).mean()

    feats = {}
    feats["rev_1d"] = cs_rank(-ret)
    feats["rev_3d"] = cs_rank(-(close / close.shift(3) - 1))
    feats["rev_5d_vn"] = cs_rank(-(close / close.shift(5) - 1) / vol_20d)
    feats["intra_rev"] = cs_rank(-(close / open_ - 1))
    feats["close_pos_rev"] = cs_rank(-(close - low) / (high - low + 1e-10))
    feats["vol_ratio"] = cs_rank(volume / vol_20d_avg)
    feats["dist_ma5_vn"] = cs_rank(-(close / close.rolling(5).mean() - 1) / vol_20d)
    vol_mom_5d = volume / volume.shift(5) - 1
    price_mom_5d = close / close.shift(5) - 1
    feats["vp_div_neg"] = cs_rank(
        -(price_mom_5d.rank(axis=1, pct=True) - vol_mom_5d.rank(axis=1, pct=True))
    )
    feats["skew_20d"] = cs_rank(ret.rolling(20).skew())
    feats["mom_120d"] = cs_rank(close / close.shift(120) - 1)
    rolling_max_20 = close.rolling(20).max()
    feats["drawdown_rev"] = cs_rank(-(close / rolling_max_20 - 1))

    cluster1 = (
        feats["rev_1d"].fillna(0)
        + feats["intra_rev"].fillna(0)
        + feats["close_pos_rev"].fillna(0)
    ) / 3
    cluster2 = (
        feats["rev_3d"].fillna(0)
        + feats["rev_5d_vn"].fillna(0)
        + feats["dist_ma5_vn"].fillna(0)
    ) / 3
    indep_sum = (
        feats["vol_ratio"].fillna(0)
        + feats["vp_div_neg"].fillna(0)
        + feats["skew_20d"].fillna(0)
        + feats["mom_120d"].fillna(0)
        + feats["drawdown_rev"].fillna(0)
    )
    combo = (cluster1 + cluster2 + indep_sum) / 7

    def simulate_portfolio(signal, fwd_returns, mask):
        dates = signal.index[mask]
        pnls, ics = [], []
        for date in dates:
            sig = signal.loc[date].dropna()
            fwd = fwd_returns.loc[date].dropna()
            common = sig.index.intersection(fwd.index)
            if len(common) < 50:
                continue
            sig_c, fwd_c = sig[common], fwd[common]
            ic = sig_c.rank().corr(fwd_c.rank())
            ics.append(ic)
            n = len(common)
            weights = sig_c.rank() - (n + 1) / 2
            weights = weights / weights.abs().sum()
            pnls.append((weights * fwd_c).sum())
        pnls, ics = np.array(pnls), np.array(ics)
        ann = np.sqrt(252)
        sharpe = np.mean(pnls) / np.std(pnls) * ann if np.std(pnls) > 0 else 0
        return {
            "sharpe": sharpe,
            "mean_ic": np.nanmean(ics),
            "ic_ir": (
                np.nanmean(ics) / np.nanstd(ics) * ann if np.nanstd(ics) > 0 else 0
            ),
            "cum_pnl": np.sum(pnls),
        }

    print("=" * 60)
    print("STRATEGY BACKTEST RESULTS (V3_no_vwt)")
    print("=" * 60)
    for name, mask in [
        ("Train (2020-2022)", close.index <= "2022-12-31"),
        (
            "Val1 (2023 H1)",
            (close.index > "2022-12-31") & (close.index <= "2023-06-30"),
        ),
        (
            "Val2 (2023 H2)",
            (close.index > "2023-06-30") & (close.index <= "2023-12-31"),
        ),
        ("Test (2024)", close.index > "2023-12-31"),
    ]:
        r = simulate_portfolio(combo, fwd_ret, mask)
        print(f"\n{name}:")
        print(f"  Sharpe:  {r['sharpe']:.2f}")
        print(f"  Mean IC: {r['mean_ic']:.4f}")
        print(f"  IC IR:   {r['ic_ir']:.2f}")
        print(f"  Cum PnL: {r['cum_pnl']:.6f}")

    # Validate generate_signal against panel backtest
    print("\n" + "=" * 60)
    print("VALIDATING generate_signal() OUTPUT")
    print("=" * 60)
    data_dict = {}
    for ticker in close.columns:
        df = pd.DataFrame(
            {
                "Date": close.index,
                "open": open_[ticker].values,
                "high": high[ticker].values,
                "low": low[ticker].values,
                "close": close[ticker].values,
                "volume": volume[ticker].values,
            }
        )
        data_dict[ticker] = df

    signal = generate_signal(data_dict)
    panel_signal = combo.iloc[-1].dropna()

    common_tickers = sorted(set(signal.keys()) & set(panel_signal.index))
    func_vals = np.array([signal[t] for t in common_tickers])
    panel_vals = np.array([panel_signal[t] for t in common_tickers])

    from scipy.stats import spearmanr

    corr, _ = spearmanr(func_vals, panel_vals)
    print(f"Rank correlation between generate_signal and panel: {corr:.4f}")
    print(f"Number of tickers: {len(common_tickers)}")
    print(f"Signal range: [{np.min(func_vals):.4f}, {np.max(func_vals):.4f}]")