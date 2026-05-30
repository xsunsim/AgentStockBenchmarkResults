# Strategy: Anthropic__Opus4_7__LinearNeutral_202605
# Model: Anthropic__Opus4_7 (api: opus)
# File: strategies/Anthropic__Opus4_7__LinearNeutral_202605/strategy.py

import pandas as pd
import numpy as np
from scipy import stats


# =============================================================================
# RESEARCH SECTION
# =============================================================================
def run_research():
    close = pd.read_parquet("close.parquet")
    high = pd.read_parquet("high.parquet")
    low = pd.read_parquet("low.parquet")
    opn = pd.read_parquet("open.parquet")
    volume = pd.read_parquet("volume.parquet")

    fwd_ret = close.shift(-1) / close - 1
    ret_daily = close / close.shift(1) - 1
    vol20 = ret_daily.rolling(20).std()

    def cross_sectional_rank(df):
        return df.rank(axis=1, pct=True) * 2 - 1

    def rank_ic_series_fast(signal_df, fwd_ret_df, start, end):
        mask = (signal_df.index >= start) & (signal_df.index <= end)
        sig_sub = signal_df[mask]
        fwd_sub = fwd_ret_df.reindex(sig_sub.index)
        ics = []
        for i in range(len(sig_sub)):
            s = sig_sub.iloc[i].dropna()
            r = fwd_sub.iloc[i].dropna()
            common = s.index.intersection(r.index)
            if len(common) < 50:
                continue
            ic, _ = stats.spearmanr(s[common], r[common])
            ics.append(ic)
        if not ics:
            return None
        arr = np.array(ics)
        return {
            "mean_ic": round(arr.mean(), 5),
            "ic_ir": round(arr.mean() / arr.std(), 4),
            "hit": round((arr > 0).mean(), 4),
            "n": len(arr),
        }

    # ---- Build individual ranked factors ----
    rev_vnrev1 = cross_sectional_rank(-(ret_daily / vol20))
    rev_rev5 = cross_sectional_rank(-(close / close.shift(5) - 1))
    rev_intraday = cross_sectional_rank(-(close / opn - 1))
    mom_252 = cross_sectional_rank(close.shift(21) / close.shift(252) - 1)
    mom_120 = cross_sectional_rank(close.shift(20) / close.shift(120) - 1)
    vol_r = cross_sectional_rank(volume / volume.rolling(20).mean())

    # ---- Final composite: decorrelated weighted ----
    composite = (
        1.0 * rev_vnrev1
        + 0.5 * rev_rev5
        + 1.0 * rev_intraday
        + 1.0 * mom_252
        + 0.5 * mom_120
        + 0.5 * vol_r
    ) / 4.5

    print("=" * 60)
    print("FINAL COMPOSITE: Decorrelated Weighted 6-Factor")
    print("  Factors: vnrev1(1.0), rev5(0.5), intraday_rev(1.0),")
    print("           mom252_skip21(1.0), mom120_skip20(0.5), vol_ratio(0.5)")
    print("=" * 60)

    periods = [
        ("Train (2020-2022)", "2020-06-01", "2022-12-31"),
        ("Val1  (2023 H1)", "2023-01-01", "2023-06-30"),
        ("Val2  (2023 H2)", "2023-07-01", "2023-12-31"),
        ("Test  (2024)", "2024-01-01", "2024-12-31"),
    ]

    for name, start, end in periods:
        result = rank_ic_series_fast(composite, fwd_ret, start, end)
        print(f"  {name}: {result}")

    # ---- Individual factor ICs for reference ----
    print("\nIndividual factor ICs (full sample 2020-2024):")
    for fname, sig in [
        ("Vol-norm 1d rev", rev_vnrev1),
        ("5d reversal", rev_rev5),
        ("Intraday rev", rev_intraday),
        ("252d mom skip21", mom_252),
        ("120d mom skip20", mom_120),
        ("Volume ratio", vol_r),
    ]:
        r = rank_ic_series_fast(sig, fwd_ret, "2020-06-01", "2024-12-31")
        print(f"  {fname}: {r}")

    # ---- Simulated PnL ----
    print("\nSimulated long-short PnL:")
    for pname, ps, pe in periods:
        mask = (composite.index >= ps) & (composite.index <= pe)
        sig_sub = composite[mask]
        fwd_sub = fwd_ret.reindex(sig_sub.index)
        daily_pnl = []
        for i in range(len(sig_sub)):
            s = sig_sub.iloc[i].dropna()
            r = fwd_sub.iloc[i].dropna()
            common = s.index.intersection(r.index)
            if len(common) < 100:
                continue
            n = len(common)
            ranks = s[common].rank()
            weights = (ranks - (n + 1) / 2) / n
            daily_pnl.append((weights * r[common]).sum())
        if daily_pnl:
            arr = np.array(daily_pnl)
            sharpe = arr.mean() / arr.std() * np.sqrt(252) if arr.std() > 0 else 0
            print(f"  {pname}: Sharpe={sharpe:.3f}, Mean={arr.mean()*10000:.1f}bps")


# =============================================================================
# PRODUCTION SIGNAL
# =============================================================================
def generate_signal(data):
    """
    Multi-factor stock ranking signal combining three uncorrelated alpha sources:
    1. Short-term reversal (vol-normalized 1d, raw 5d, intraday)
    2. Medium/long-term momentum (252d skip 21d, 120d skip 20d)
    3. Volume surprise (current vs 20d average)

    Parameters
    ----------
    data : dict[str, DataFrame]
        ticker -> DataFrame with columns [Date, open, high, low, close, volume]

    Returns
    -------
    dict[str, float]
        ticker -> signal score (higher = more long)
    """
    scores = {}
    raw_signals = {}

    for ticker, df in data.items():
        if len(df) < 252:
            continue

        c = df["close"].values
        o = df["open"].values
        v = df["volume"].values

        n = len(c)

        # --- Factor 1: Vol-normalized 1-day reversal ---
        if n >= 21:
            ret_1d = c[-1] / c[-2] - 1 if c[-2] != 0 else 0.0
            daily_rets = np.diff(c[-21:]) / c[-21:-1]
            daily_rets = daily_rets[np.isfinite(daily_rets)]
            vol_20d = np.std(daily_rets) if len(daily_rets) > 5 else 1e-6
            vol_20d = max(vol_20d, 1e-8)
            vnrev1 = -ret_1d / vol_20d
        else:
            vnrev1 = 0.0

        # --- Factor 2: 5-day reversal ---
        if n >= 6:
            ret_5d = c[-1] / c[-6] - 1 if c[-6] != 0 else 0.0
            rev5 = -ret_5d
        else:
            rev5 = 0.0

        # --- Factor 3: Intraday return reversal ---
        if o[-1] != 0:
            intraday_rev = -(c[-1] / o[-1] - 1)
        else:
            intraday_rev = 0.0

        # --- Factor 4: 252-day momentum, skip last 21 days ---
        if n >= 253:
            c_21 = c[-22]
            c_252 = c[-253]
            mom252 = (c_21 / c_252 - 1) if c_252 != 0 else 0.0
        else:
            mom252 = 0.0

        # --- Factor 5: 120-day momentum, skip last 20 days ---
        if n >= 141:
            c_20 = c[-21]
            c_120 = c[-141]
            mom120 = (c_20 / c_120 - 1) if c_120 != 0 else 0.0
        else:
            mom120 = 0.0

        # --- Factor 6: Volume ratio (current vs 20-day average) ---
        if n >= 20:
            vol_avg_20 = np.mean(v[-20:])
            vol_ratio = v[-1] / vol_avg_20 if vol_avg_20 > 0 else 1.0
        else:
            vol_ratio = 1.0

        raw_signals[ticker] = {
            "vnrev1": vnrev1,
            "rev5": rev5,
            "intraday_rev": intraday_rev,
            "mom252": mom252,
            "mom120": mom120,
            "vol_ratio": vol_ratio,
        }

    if not raw_signals:
        return {}

    tickers = list(raw_signals.keys())
    factor_names = ["vnrev1", "rev5", "intraday_rev", "mom252", "mom120", "vol_ratio"]
    weights = [1.0, 0.5, 1.0, 1.0, 0.5, 0.5]
    total_weight = sum(weights)

    factor_arrays = {}
    for fname in factor_names:
        vals = np.array([raw_signals[t][fname] for t in tickers], dtype=np.float64)
        factor_arrays[fname] = vals

    ranked_factors = {}
    for fname in factor_names:
        vals = factor_arrays[fname]
        finite_mask = np.isfinite(vals)
        ranks = np.full_like(vals, np.nan)
        if finite_mask.sum() > 0:
            temp = stats.rankdata(vals[finite_mask])
            temp = (
                (temp - 1) / (finite_mask.sum() - 1) * 2 - 1
                if finite_mask.sum() > 1
                else np.zeros(finite_mask.sum())
            )
            ranks[finite_mask] = temp
        ranked_factors[fname] = ranks

    composite = np.zeros(len(tickers))
    for fname, w in zip(factor_names, weights):
        r = ranked_factors[fname]
        valid = np.isfinite(r)
        composite[valid] += w * r[valid]
    composite /= total_weight

    for i, ticker in enumerate(tickers):
        scores[ticker] = float(composite[i])

    return scores


if __name__ == "__main__":
    run_research()

    print("\n" + "=" * 60)
    print("Testing generate_signal interface...")
    print("=" * 60)

    close = pd.read_parquet("close.parquet")
    high = pd.read_parquet("high.parquet")
    low = pd.read_parquet("low.parquet")
    opn = pd.read_parquet("open.parquet")
    volume = pd.read_parquet("volume.parquet")

    data = {}
    for ticker in close.columns:
        df = pd.DataFrame(
            {
                "Date": close.index,
                "open": opn[ticker].values,
                "high": high[ticker].values,
                "low": low[ticker].values,
                "close": close[ticker].values,
                "volume": volume[ticker].values,
            }
        )
        df = df.dropna(subset=["close"])
        data[ticker] = df

    signals = generate_signal(data)
    print(f"  Stocks scored: {len(signals)}")
    sorted_sigs = sorted(signals.items(), key=lambda x: x[1], reverse=True)
    print(f"  Top 5: {sorted_sigs[:5]}")
    print(f"  Bottom 5: {sorted_sigs[-5:]}")
    print(f"  Score range: [{sorted_sigs[-1][1]:.4f}, {sorted_sigs[0][1]:.4f}]")