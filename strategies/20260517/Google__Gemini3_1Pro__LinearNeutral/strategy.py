# Strategy: Google__Gemini3_1Pro__LinearNeutral_202605
# Model: Google__Gemini3_1Pro (api: gemini-3.1-pro-preview)
# File: strategies/Google__Gemini3_1Pro__LinearNeutral_202605/strategy.py

import numpy as np
import pandas as pd

def generate_signal(data):
    """
    Generates trading signals for a universe of stocks.
    
    Args:
        data: dict mapping ticker (str) -> DataFrame with columns [Date, open, high, low, close, volume]
              Each DataFrame contains history up to the current day.
              
    Returns:
        dict[str, float] mapping ticker -> signal score.
    """
    tickers = []
    n = len(data)
    
    f_intraday = np.zeros(n)
    f_overnight = np.zeros(n)
    f_ret5 = np.zeros(n)
    f_ret20 = np.zeros(n)
    f_mom12m = np.zeros(n)
    f_vol20 = np.zeros(n)
    
    valid_mask = np.zeros(n, dtype=bool)
    
    for i, (ticker, df) in enumerate(data.items()):
        tickers.append(ticker)
        
        if len(df) < 25:
            continue
            
        c = df['close'].values
        v = df['volume'].values
        o = df['open'].values
        
        c_val = c[-1]
        v_val = v[-1]
        
        if np.isnan(c_val) or c_val <= 0 or v_val <= 0:
            continue
            
        valid_mask[i] = True
        
        o_val = o[-1]
        c_prev = c[-2]
        c_5 = c[-6] if len(c) >= 6 else np.nan
        c_20 = c[-21] if len(c) >= 21 else np.nan
        
        f_intraday[i] = (c_val / o_val) - 1.0 if (o_val and not np.isnan(o_val) and o_val > 0) else 0
        f_overnight[i] = (o_val / c_prev) - 1.0 if (c_prev and not np.isnan(c_prev) and c_prev > 0) else 0
        f_ret5[i] = (c_val / c_5) - 1.0 if (c_5 and not np.isnan(c_5) and c_5 > 0) else 0
        f_ret20[i] = (c_val / c_20) - 1.0 if (c_20 and not np.isnan(c_20) and c_20 > 0) else 0
        
        if len(c) >= 253:
            c_253 = c[-253]
            c_22 = c[-22]
            if c_253 and not np.isnan(c_253) and c_253 > 0 and not np.isnan(c_22):
                f_mom12m[i] = (c_22 / c_253) - 1.0
            else:
                f_mom12m[i] = 0
                
        # 20-day volatility
        rets = (c[-20:] / c[-21:-1]) - 1.0
        with np.errstate(invalid='ignore'):
            vol = np.nanstd(rets)
            
        f_vol20[i] = vol if (vol > 0 and not np.isnan(vol)) else 0.01

    def cs_zscore(x, mask):
        z = np.zeros_like(x)
        if np.sum(mask) > 1:
            masked_x = x[mask].copy()
            nan_mask = np.isnan(masked_x)
            if np.any(nan_mask):
                masked_x[nan_mask] = np.nanmedian(masked_x)
                
            p99 = np.percentile(masked_x, 99)
            p01 = np.percentile(masked_x, 1)
            masked_x = np.clip(masked_x, p01, p99)
            std = np.std(masked_x)
            if std > 0:
                z[mask] = (masked_x - np.mean(masked_x)) / std
        return z

    z_intraday = cs_zscore(f_intraday, valid_mask)
    z_overnight = cs_zscore(f_overnight, valid_mask)
    z_ret5 = cs_zscore(f_ret5, valid_mask)
    z_ret20 = cs_zscore(f_ret20, valid_mask)
    z_mom12m = cs_zscore(f_mom12m, valid_mask)
    z_vol20 = cs_zscore(f_vol20, valid_mask)
    
    scores = np.zeros(n)
    
    for i in range(n):
        if valid_mask[i]:
            # Weights for stat-arb alpha
            score = (-0.30 * z_intraday[i]
                     - 0.30 * z_overnight[i]
                     - 0.20 * z_ret5[i]
                     - 0.10 * z_ret20[i]
                     + 0.15 * z_mom12m[i]
                     - 0.15 * z_vol20[i])
            scores[i] = score
            
    return {tickers[i]: float(scores[i]) for i in range(n)}

if __name__ == "__main__":
    print("Starting research...")
    close_df = pd.read_parquet("/tmp/tmp_yu5gkum/close.parquet")
    open_df = pd.read_parquet("/tmp/tmp_yu5gkum/open.parquet")
    high_df = pd.read_parquet("/tmp/tmp_yu5gkum/high.parquet")
    low_df = pd.read_parquet("/tmp/tmp_yu5gkum/low.parquet")
    vol_df = pd.read_parquet("/tmp/tmp_yu5gkum/volume.parquet")
    
    dates = close_df.index
    test_dates = dates[(dates >= '2023-01-01') & (dates < '2024-01-01')]
    
    print("Evaluating signal...")
    ics = []
    
    from scipy.stats import spearmanr
    
    for d in test_dates[::20]:
        t_idx = np.where(dates == d)[0][0]
        if t_idx < 253: continue
        
        data_dict = {}
        past_slice = slice(0, t_idx + 1)
        past_dates = dates[past_slice]
        
        for ticker in close_df.columns:
            data_dict[ticker] = pd.DataFrame({
                'Date': past_dates,
                'open': open_df[ticker].values[past_slice],
                'high': high_df[ticker].values[past_slice],
                'low': low_df[ticker].values[past_slice],
                'close': close_df[ticker].values[past_slice],
                'volume': vol_df[ticker].values[past_slice],
            })
            
        scores = generate_signal(data_dict)
        
        if t_idx + 1 < len(dates):
            next_d = dates[t_idx + 1]
            fwd_ret = (close_df.loc[next_d] / close_df.loc[d] - 1.0).fillna(0)
            
            score_arr = []
            ret_arr = []
            for ticker, score in scores.items():
                if score != 0:
                    score_arr.append(score)
                    ret_arr.append(fwd_ret[ticker])
                    
            if len(score_arr) > 0:
                ic, _ = spearmanr(score_arr, ret_arr)
                ics.append(ic)
                print(f"Date: {d.date() if hasattr(d, 'date') else d}, Rank IC: {ic:.4f}")
                
    print(f"Mean Rank IC (2023 sample): {np.nanmean(ics):.4f}")