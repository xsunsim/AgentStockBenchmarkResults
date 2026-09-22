"""
Daily S&P 500 stock ranking signal.

Predicts the signal-day-t return  close[t+2]/close[t+1] - 1
(enter next day's close, exit the following day's close).

The signal is an equal-weight blend of three cross-sectionally z-scored,
economically distinct and mutually uncorrelated features:

  1. mom   : intermediate-term momentum with a 1-month skip
             close[t-21] / close[t-126] - 1
             (classic 12-1 style momentum; winners keep winning while
              avoiding the well-known 1-month reversal contamination)

  2. overnight : overnight-gap reversal
             -(open[t] / close[t-1] - 1)
             (overnight jumps driven by sentiment/liquidity tend to
              partially reverse over the next day or two)

  3. rev5  : short-term reversal
             -(close[t] / close[t-5] - 1)
             (one-week price moves mean-revert at the 1-2 day horizon)

Rank-IC (cross-sectional Spearman vs the entry-to-exit forward return),
equal-weight blend:
    train (2020-2022) IC 0.0110
    val   (2023)      IC 0.0196
    test  (2024)      IC 0.0144   (single, post-selection evaluation)

Only the relative ordering of the returned scores matters.
"""

import numpy as np
import pandas as pd

# --- feature horizons (trading days) ---
MOM_NEAR = 21  # skip the most recent month
MOM_FAR = 126  # ~6 months
REV_LB = 5  # short-term reversal look-back
MIN_HISTORY = MOM_FAR + 2  # rows needed to compute every feature


def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score; NaNs -> 0 (neutral) so the name still ranks."""
    std = s.std()
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    z = (s - s.mean()) / std
    return z.fillna(0.0)


def _ticker_features(df: pd.DataFrame):
    """Return (mom, overnight, rev5) scalars for one ticker, or NaNs."""
    # Ensure chronological order and use positional access on the tail.
    if "Date" in df.columns:
        df = df.sort_values("Date")
    close = df["close"].to_numpy(dtype=float)
    openp = df["open"].to_numpy(dtype=float)
    n = close.shape[0]
    if n < MIN_HISTORY:
        return np.nan, np.nan, np.nan

    c_last = close[-1]
    c_prev = close[-2]
    c_near = close[-1 - MOM_NEAR]
    c_far = close[-1 - MOM_FAR]
    c_rev = close[-1 - REV_LB]
    o_last = openp[-1]

    mom = c_near / c_far - 1.0 if c_far > 0 else np.nan
    overnight = -(o_last / c_prev - 1.0) if c_prev > 0 else np.nan
    rev5 = -(c_last / c_rev - 1.0) if c_rev > 0 else np.nan
    return mom, overnight, rev5


def generate_signal(data: dict) -> dict:
    """
    Parameters
    ----------
    data : dict[str, pandas.DataFrame]
        ticker -> DataFrame with columns [Date, open, high, low, close, volume],
        containing all history up to (and including) the signal day.

    Returns
    -------
    dict[str, float]
        ticker -> signal score. Higher = more long. Only ordering matters.
    """
    tickers = list(data.keys())
    mom = {}
    overnight = {}
    rev5 = {}
    for t in tickers:
        df = data[t]
        if df is None or len(df) == 0:
            m = o = r = np.nan
        else:
            m, o, r = _ticker_features(df)
        mom[t] = m
        overnight[t] = o
        rev5[t] = r

    mom = pd.Series(mom)
    overnight = pd.Series(overnight)
    rev5 = pd.Series(rev5)

    # Equal-weight blend of cross-sectionally normalized features.
    signal = _zscore(mom) + _zscore(overnight) + _zscore(rev5)

    return {t: float(signal.get(t, 0.0)) for t in tickers}


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Research / validation. Runs only as a script, never on import.
    # ------------------------------------------------------------------
    close = pd.read_parquet("close.parquet")
    openp = pd.read_parquet("open.parquet")
    high = pd.read_parquet("high.parquet")
    low = pd.read_parquet("low.parquet")
    vol = pd.read_parquet("volume.parquet")

    fwd = close.shift(-2) / close.shift(-1) - 1.0  # entry->exit forward return
    ret1 = close.pct_change()
    fwd_rank = fwd.rank(axis=1)

    idx = close.index
    train = idx <= "2022-12-31"
    val = (idx >= "2023-01-01") & (idx <= "2023-12-31")
    test = idx >= "2024-01-01"

    def rank_ic(feat, mask):
        d = feat.rank(axis=1)[mask].corrwith(fwd_rank[mask], axis=1)  # spearman
        return d.mean(), d.mean() / d.std()

    def zc(df):
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

    # Panel versions of the three production features.
    F = {
        "mom": close.shift(MOM_NEAR) / close.shift(MOM_FAR) - 1.0,
        "overnight": -(openp / close.shift(1) - 1.0),
        "rev5": -(close / close.shift(REV_LB) - 1.0),
    }

    print(f"{'feature':<12}{'trainIC':>9}{'valIC':>9}{'testIC':>9}")
    for k, f in F.items():
        ti, _ = rank_ic(f, train)
        vi, _ = rank_ic(f, val)
        te, _ = rank_ic(f, test)
        print(f"{k:<12}{ti:>9.4f}{vi:>9.4f}{te:>9.4f}")

    combo = zc(F["mom"]) + zc(F["overnight"]) + zc(F["rev5"])
    print("\nequal-weight blend (mom + overnight + rev5):")
    for nm, m in [("train", train), ("val", val), ("test", test)]:
        i, ir = rank_ic(combo, m)
        print(f"  {nm:<6} IC {i:.4f}  IR {ir:.3f}")

    # Sanity check the production interface against the panel result.
    print("\nverifying generate_signal() on the most recent day ...")
    data = {}
    for tk in close.columns:
        d = pd.DataFrame(
            {
                "Date": close.index,
                "open": openp[tk].values,
                "high": high[tk].values,
                "low": low[tk].values,
                "close": close[tk].values,
                "volume": vol[tk].values,
            }
        ).dropna(subset=["close"])
        data[tk] = d
    scores = generate_signal(data)
    panel_last = combo.iloc[-1].dropna()
    s = pd.Series(scores).reindex(panel_last.index)
    corr = s.rank().corr(panel_last.rank())
    print(f"  scored {len(scores)} tickers; rank corr vs panel = {corr:.3f}")