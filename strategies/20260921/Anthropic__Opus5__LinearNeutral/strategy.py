"""
Daily cross-sectional stock ranking signal for ~500 S&P 500 names.

Prediction target for signal day t:  close[t+2] / close[t+1] - 1
(enter at next day's close, exit at the following close).

Signal = equal-weighted blend of four economically distinct, cross-sectionally
normalized blocks, then risk-scaled so that the rank-linear dollar weights are
not concentrated in the most volatile names:

  1. REV_SHORT  - 2-day idiosyncratic (market-demeaned) reversal, volatility
                  normalized, damped when the move came on heavy volume.
                  Uninformed price pressure reverses; news/earnings driven
                  moves (heavy volume) do not, so they are down-weighted.
  2. REV_MID    - deviation of the close from the 5-day average typical price
                  ((H+L+C)/3), i.e. a VWAP-style mean-reversion / overextension
                  measure over the trading week.
  3. ON_REV     - reversal of the last 3 days of overnight (close-to-open)
                  returns. Overnight moves are liquidity/sentiment driven and
                  systematically give back part of the move intraday.
  4. MOM        - vol-scaled intermediate momentum block: average of 12-1 and
                  6-1 month momentum (both skip the most recent month, which is
                  the reversal region). Captures slow information diffusion and
                  is essentially uncorrelated with the reversal blocks.

Blocks are equal weighted (no fitted coefficients). The composite is divided by
(0.25 + cross-sectional percentile rank of volatility) so that low-volatility
names receive the extreme ranks: with fixed +/-$250 rank-linear weights this is
the only way to express risk scaling, and it raised backtest Sharpe materially
without changing rank IC.

Backtest (rank-linear dollar-neutral portfolio, panel data 2020-2024):
  TRAIN 2021-2022  IC=+0.0050  ICIR=+0.38  PnL -0.32 bps/day  SR=-0.06
  VAL   2023       IC=+0.0143  ICIR=+1.24  PnL +3.54 bps/day  SR=+0.91
  TEST  2024       IC=+0.0171  ICIR=+1.85  PnL +4.60 bps/day  SR=+1.45
(2021 -- the meme-stock year -- is the single loss year; all blocks are
positive IC over train, validation and test taken as a whole.)
"""

import numpy as np
import pandas as pd

# Trading days of history needed: 252 (12m momentum) + 21 (skip month) + slack.
LOOKBACK = 300

MIN_PRICE = 3.0
MIN_NAMES = 5
VOL_SCALE_FLOOR = 0.25


def _xs_rank_z(frame, valid):
    """Cross-sectional (per-date) rank z-score, restricted to the valid universe."""
    r = frame.where(valid).rank(axis=1, pct=True).sub(0.5)
    sd = r.std(axis=1)
    return r.div(sd.where(sd > 0), axis=0)


def compute_score_panel(close, openp, high, low, volume):
    """Full (dates x tickers) score panel. Shared by research and production."""
    close = close.astype(float)
    openp = openp.astype(float)
    high = high.astype(float)
    low = low.astype(float)
    volume = volume.astype(float)

    valid = close.notna() & (close > MIN_PRICE) & volume.notna() & (volume > 0)

    ret1 = close.pct_change(fill_method=None)
    mkt = ret1.where(valid).mean(axis=1)
    idio = ret1.sub(mkt, axis=0)

    v20 = ret1.rolling(20, min_periods=12).std()
    v60 = ret1.rolling(60, min_periods=25).std()
    sigma = 0.5 * v20 + 0.5 * v60.fillna(v20)

    # relative volume over the last 2 sessions (1.0 = normal), capped at 2
    rel_vol = (
        (volume / volume.rolling(60, min_periods=20).median())
        .rolling(2)
        .mean()
        .clip(0.0, 2.0)
    )

    typ = (high + low + close) / 3.0
    overnight = openp / close.shift(1) - 1.0

    # --- blocks -------------------------------------------------------------
    rev_short = -(idio.rolling(2).sum() / v20) * (2.0 - rel_vol.fillna(1.0))
    rev_mid = -(close / typ.rolling(5).mean() - 1.0)
    on_rev = -overnight.rolling(3).sum()
    mom_long = (close.shift(21) / close.shift(252) - 1.0) / v20
    mom_mid = (close.shift(21) / close.shift(126) - 1.0) / v20

    z_rev_short = _xs_rank_z(rev_short, valid)
    z_rev_mid = _xs_rank_z(rev_mid, valid)
    z_on_rev = _xs_rank_z(on_rev, valid)
    z_mom_long = _xs_rank_z(mom_long, valid)
    z_mom_mid = _xs_rank_z(mom_mid, valid)

    # momentum block: average of the two horizons, tolerant of a missing one
    z_mom = (z_mom_long.fillna(z_mom_mid) + z_mom_mid.fillna(z_mom_long)) / 2.0

    combo = (
        z_rev_short.fillna(0.0)
        + z_rev_mid.fillna(0.0)
        + z_on_rev.fillna(0.0)
        + z_mom.fillna(0.0)
    ) / 4.0

    # risk scaling: push low-vol names to the extreme (largest dollar) ranks
    vol_rank = sigma.where(valid).rank(axis=1, pct=True)
    score = combo / (VOL_SCALE_FLOOR + vol_rank.fillna(1.0))

    # names outside the tradable universe get a neutral (middle-rank) score
    enough = valid.sum(axis=1) >= MIN_NAMES
    score = score.where(valid, 0.0)
    score = score.where(enough, 0.0)
    return score.fillna(0.0)


def _build_panels(data):
    """dict[ticker] -> DataFrame  ==>  aligned (dates x tickers) OHLCV panels."""
    cols = {"open": {}, "high": {}, "low": {}, "close": {}, "volume": {}}
    for ticker, df in data.items():
        if df is None or len(df) == 0:
            continue
        d = df
        if "Date" in d.columns:
            idx = pd.to_datetime(d["Date"])
        else:
            idx = pd.to_datetime(d.index)
        d = d.set_index(pd.Index(idx))
        d = d[~d.index.duplicated(keep="last")].sort_index()
        if len(d) > LOOKBACK:
            d = d.iloc[-LOOKBACK:]
        for field in cols:
            if field in d.columns:
                cols[field][ticker] = pd.to_numeric(d[field], errors="coerce")

    panels = {}
    for field, series in cols.items():
        if series:
            panels[field] = pd.DataFrame(series).sort_index()
        else:
            panels[field] = pd.DataFrame()
    # align all panels on the union of dates / tickers
    ref = panels["close"]
    for field in panels:
        panels[field] = panels[field].reindex(index=ref.index, columns=ref.columns)
    return panels


def generate_signal(data):
    """
    data: dict[ticker] -> DataFrame with columns [Date, open, high, low, close, volume]
          (full history up to and including the signal day).
    returns: dict[ticker] -> float score. Higher = more long.
    """
    tickers = list(data.keys())
    scores = {t: 0.0 for t in tickers}
    if not tickers:
        return scores

    p = _build_panels(data)
    close = p["close"]
    if close.empty or len(close.index) < 25:
        return scores

    panel = compute_score_panel(close, p["open"], p["high"], p["low"], p["volume"])
    if panel.empty:
        return scores

    last = panel.iloc[-1]
    for t in tickers:
        v = last.get(t, 0.0)
        scores[t] = float(v) if np.isfinite(v) else 0.0
    return scores


if __name__ == "__main__":
    # ---------------- research / validation (only runs as a script) ----------
    close = pd.read_parquet("close.parquet")
    openp = pd.read_parquet("open.parquet")
    high = pd.read_parquet("high.parquet")
    low = pd.read_parquet("low.parquet")
    volume = pd.read_parquet("volume.parquet")

    fwd = close.shift(-2) / close.shift(-1) - 1.0  # entry close -> exit close
    score = compute_score_panel(close, openp, high, low, volume)
    tradable = close.notna() & (close > MIN_PRICE) & volume.notna()
    score_v = score.where(tradable)

    ic = score_v.corrwith(fwd, axis=1, method="spearman")

    # rank-linear dollar-neutral portfolio: rank 1 -> +250 ... rank N -> -250
    rank = score_v.rank(axis=1, ascending=False)
    n = score_v.notna().sum(axis=1)
    w = pd.DataFrame(
        (n.values[:, None] + 1) / 2.0 - rank.values,
        index=score_v.index,
        columns=score_v.columns,
    )
    w = w.div(w.abs().sum(axis=1), axis=0) * 2.0  # gross = 2 (1 long / 1 short)
    pnl = (w * fwd).sum(axis=1, min_count=1).dropna()

    def show(label, sl):
        x = ic.loc[sl].dropna()
        q = pnl.loc[sl]
        print(
            f"{label:18s} IC={x.mean():+.4f} ICIR={x.mean()/x.std()*np.sqrt(252):+5.2f} "
            f"| PnL {q.mean()*1e4:+5.2f} bps/day  SR={q.mean()/q.std()*np.sqrt(252):+5.2f} "
            f"hit={(q > 0).mean():.3f}"
        )

    print("=== composite signal performance (target: close[t+2]/close[t+1]-1) ===")
    show("TRAIN 2021-2022", slice("2021-01-04", "2022-12-31"))
    show("VAL   2023", slice("2023-01-01", "2023-12-31"))
    show("TEST  2024", slice("2024-01-01", "2024-12-31"))
    print()
    for y in sorted(set(ic.loc["2021":].index.year)):
        show(f"  year {y}", slice(f"{y}-01-01", f"{y}-12-31"))

    # ---- individual block diagnostics -------------------------------------
    print("\n=== individual blocks (rank IC) ===")
    ret1 = close.pct_change(fill_method=None)
    mkt = ret1.where(tradable).mean(axis=1)
    idio = ret1.sub(mkt, axis=0)
    v20 = ret1.rolling(20, min_periods=12).std()
    rel_vol = (
        (volume / volume.rolling(60, min_periods=20).median())
        .rolling(2)
        .mean()
        .clip(0, 2)
    )
    typ = (high + low + close) / 3.0
    on = openp / close.shift(1) - 1.0
    blocks = {
        "rev_short (2d idio rev)": -(idio.rolling(2).sum() / v20)
        * (2 - rel_vol.fillna(1)),
        "rev_mid (close vs tp5)": -(close / typ.rolling(5).mean() - 1),
        "on_rev (3d overnight)": -on.rolling(3).sum(),
        "mom_long (12-1 / vol)": (close.shift(21) / close.shift(252) - 1) / v20,
        "mom_mid (6-1 / vol)": (close.shift(21) / close.shift(126) - 1) / v20,
    }
    for name, f in blocks.items():
        bic = (
            f.where(tradable)
            .loc["2021-01-04":]
            .corrwith(fwd, axis=1, method="spearman")
        )
        yr = bic.groupby(bic.index.year).mean()
        print(
            f"{name:26s} IC={bic.mean():+.4f} ICIR={bic.mean()/bic.std()*np.sqrt(252):+5.2f} | "
            + "  ".join(f"{y}:{v*1e4:+5.0f}bp" for y, v in yr.items())
        )

    # ---- production-interface equivalence check ---------------------------
    print("\n=== generate_signal() sanity check vs panel computation ===")
    data = {}
    for t in close.columns:
        df = pd.DataFrame(
            {
                "Date": close.index,
                "open": openp[t].values,
                "high": high[t].values,
                "low": low[t].values,
                "close": close[t].values,
                "volume": volume[t].values,
            }
        ).dropna(subset=["close"])
        if len(df):
            data[t] = df.reset_index(drop=True)
    out = generate_signal(data)
    ref = score.iloc[-1]
    common = [t for t in out if t in ref.index]
    live = pd.Series({t: out[t] for t in common})
    corr = live.rank().corr(ref[common].fillna(0.0).rank())
    print(f"tickers scored: {len(out)}  rank corr with panel last row: {corr:.4f}")
    print("top 5 long:", list(live.sort_values(ascending=False).head(5).index))
    print("top 5 short:", list(live.sort_values().head(5).index))