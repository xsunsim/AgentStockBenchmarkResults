import numpy as np
import pandas as pd


def _build_panel(data, field):
    return pd.DataFrame({t: df.set_index("Date")[field] for t, df in data.items()})


def generate_signal(data):
    """
    Combine four cross-sectionally normalized features into one score per ticker:
      1. rev_1d   - short-term (1-day) reversal
      2. mom_12_1 - 12-month minus most-recent-month momentum (classic momentum anomaly)
      3. gap_fade - fade of the overnight (prior close -> today's open) gap
      4. illiq_20 - Amihud-style illiquidity (20-day avg |return| / dollar volume)

    All four showed positive rank IC (vs close[t+2]/close[t+1]-1) on both a
    2020-2022 train split and a 2023 validation split, with low pairwise
    correlation, and are combined with equal weight after cross-sectional
    percentile ranking.
    """
    close = _build_panel(data, "close")
    open_ = _build_panel(data, "open")
    volume = _build_panel(data, "volume")

    ret1 = close.pct_change()

    rev_1d = -ret1.iloc[-1]

    mom_12_1 = (close.shift(21) / close.shift(252) - 1).iloc[-1]

    gap_fade = -(open_.iloc[-1] / close.shift(1).iloc[-1] - 1)

    dollar_vol = close * volume
    illiq_20 = (ret1.abs() / dollar_vol).rolling(20).mean().iloc[-1]

    features = pd.DataFrame(
        {
            "rev_1d": rev_1d,
            "mom_12_1": mom_12_1,
            "gap_fade": gap_fade,
            "illiq_20": illiq_20,
        }
    )

    ranks = features.rank(pct=True)
    ranks = ranks.fillna(0.5)  # neutral score for insufficient history

    score = ranks.mean(axis=1)

    return score.to_dict()


if __name__ == "__main__":
    close = pd.read_parquet("close.parquet")
    open_ = pd.read_parquet("open.parquet")
    high = pd.read_parquet("high.parquet")
    low = pd.read_parquet("low.parquet")
    volume = pd.read_parquet("volume.parquet")

    fwd_ret = close.shift(-2) / close.shift(-1) - 1

    def ic_stats(feat, label, mask=None):
        f = feat
        r = fwd_ret
        if mask is not None:
            f = f.loc[mask]
            r = r.loc[mask]
        f = f.rank(axis=1, pct=True)
        r = r.rank(axis=1, pct=True)
        daily_ic = f.corrwith(r, axis=1, method="pearson")
        mean_ic = daily_ic.mean()
        ir = mean_ic / daily_ic.std()
        print(
            f"{label:30s} mean IC={mean_ic: .4f}  IC-IR={ir: .4f}  n_days={daily_ic.count()}"
        )
        return daily_ic

    train_mask = (close.index >= "2020-01-01") & (close.index <= "2022-12-31")
    val_mask = (close.index >= "2023-01-01") & (close.index <= "2023-12-31")
    test_mask = (close.index >= "2024-01-01") & (close.index <= "2024-12-31")

    ret1 = close.pct_change()

    rev_1d = -ret1
    mom_12_1 = close.shift(21) / close.shift(252) - 1
    gap_fade = -(open_ / close.shift(1) - 1)
    dollar_vol = close * volume
    illiq_20 = (ret1.abs() / dollar_vol).rolling(20).mean()

    print("=== Individual features ===")
    print("-- Train (2020-2022) --")
    ic_stats(rev_1d, "rev_1d", train_mask)
    ic_stats(mom_12_1, "mom_12_1", train_mask)
    ic_stats(gap_fade, "gap_fade", train_mask)
    ic_stats(illiq_20, "illiq_20", train_mask)

    print("-- Validation (2023) --")
    ic_stats(rev_1d, "rev_1d", val_mask)
    ic_stats(mom_12_1, "mom_12_1", val_mask)
    ic_stats(gap_fade, "gap_fade", val_mask)
    ic_stats(illiq_20, "illiq_20", val_mask)

    print("-- Test (2024), final check --")
    ic_stats(rev_1d, "rev_1d", test_mask)
    ic_stats(mom_12_1, "mom_12_1", test_mask)
    ic_stats(gap_fade, "gap_fade", test_mask)
    ic_stats(illiq_20, "illiq_20", test_mask)

    print("\n=== Combined signal (equal-weight rank average) ===")
    combo = pd.concat(
        {
            "rev_1d": rev_1d.rank(axis=1, pct=True),
            "mom_12_1": mom_12_1.rank(axis=1, pct=True),
            "gap_fade": gap_fade.rank(axis=1, pct=True),
            "illiq_20": illiq_20.rank(axis=1, pct=True),
        }
    )
    combo_score = combo.groupby(level=1).mean()
    ic_stats(combo_score, "combined (train)", train_mask)
    ic_stats(combo_score, "combined (val)", val_mask)
    ic_stats(combo_score, "combined (test)", test_mask)

    print("\n=== Sanity check: generate_signal() on real data dict ===")
    sample_tickers = close.columns[:20]
    data = {}
    for t in sample_tickers:
        data[t] = pd.DataFrame(
            {
                "Date": close.index,
                "open": open_[t].values,
                "high": high[t].values,
                "low": low[t].values,
                "close": close[t].values,
                "volume": volume[t].values,
            }
        )
    scores = generate_signal(data)
    print(f"Generated {len(scores)} scores. Example: {list(scores.items())[:5]}")