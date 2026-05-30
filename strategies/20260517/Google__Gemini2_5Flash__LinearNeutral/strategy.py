# Strategy: Google__Gemini2_5Flash__LinearNeutral_202605
# Model: Google__Gemini2_5Flash (api: gemini-2.5-flash)
# File: strategies/Google__Gemini2_5Flash__LinearNeutral_202605/strategy.py

import os
import pandas as pd
import numpy as np

def generate_signal(data):
    """
    Generate daily trading signals based on mean reversion, momentum, and low volatility.
    
    data: dict mapping ticker (str) -> pd.DataFrame with columns ['Date', 'open', 'high', 'low', 'close', 'volume']
    Returns: dict mapping ticker -> float signal score (higher = more long)
    """
    features = []
    tickers = []
    
    for ticker, df in data.items():
        if len(df) < 252:
            continue
            
        c = df['close'].values
        h = df['high'].values
        l = df['low'].values
        
        # 1-day return
        ret_1 = c[-1] / c[-2] - 1.0 if c[-2] > 0 else 0.0
        
        # 5-day return
        ret_5 = c[-1] / c[-6] - 1.0 if c[-6] > 0 else 0.0
        
        # 20-day Volatility
        c_21 = c[-21:]
        rets = c_21[1:] / c_21[:-1] - 1.0
        vol_20 = np.std(rets)
        if vol_20 == 0 or np.isnan(vol_20):
            vol_20 = 1e-6
            
        # Volatility-adjusted 20-day return
        ret_20 = c[-1] / c[-21] - 1.0 if c[-21] > 0 else 0.0
        ret_20_vol_adj = ret_20 / vol_20
        
        # 252-day Momentum excluding latest 21 days
        mom_252 = c[-21] / c[-252] - 1.0 if c[-252] > 0 else 0.0
        
        # Close Location Value (CLV)
        clv = ((c[-1] - l[-1]) - (h[-1] - c[-1])) / (h[-1] - l[-1] + 1e-6)
        
        features.append([ret_1, ret_5, ret_20_vol_adj, mom_252, vol_20, clv])
        tickers.append(ticker)
        
    if not features:
        return {t: 0.0 for t in data.keys()}
        
    features = np.array(features)
    
    def rank(arr):
        arr = np.nan_to_num(arr, nan=np.nanmedian(arr))
        s = arr.argsort()
        r = np.empty_like(s)
        r[s] = np.arange(len(arr))
        return r / (len(arr) - 1) if len(arr) > 1 else r
        
    r_ret_1 = rank(features[:, 0])
    r_ret_5 = rank(features[:, 1])
    r_ret_20_adj = rank(features[:, 2])
    r_mom = rank(features[:, 3])
    r_vol = rank(features[:, 4])
    r_clv = rank(features[:, 5])
    
    # Combined score
    score = (
        -0.35 * r_ret_1 
        -0.25 * r_ret_5 
        -0.10 * r_ret_20_adj 
        + 0.15 * r_mom 
        - 0.10 * r_vol
        - 0.05 * r_clv
    )
    
    final_ranks = rank(score)
    final_scores = final_ranks - 0.5
    
    result = {t: 0.0 for t in data.keys()}
    for t, s in zip(tickers, final_scores):
        result[t] = float(s)
        
    return result

if __name__ == "__main__":
    print("Starting research and validation...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    close_df = pd.read_parquet(os.path.join(base_dir, "close.parquet"))
    open_df = pd.read_parquet(os.path.join(base_dir, "open.parquet"))
    high_df = pd.read_parquet(os.path.join(base_dir, "high.parquet"))
    low_df = pd.read_parquet(os.path.join(base_dir, "low.parquet"))
    volume_df = pd.read_parquet(os.path.join(base_dir, "volume.parquet"))
    
    ret_1 = close_df / close_df.shift(1) - 1.0
    ret_5 = close_df / close_df.shift(5) - 1.0
    ret_20 = close_df / close_df.shift(20) - 1.0
    
    vol_20 = ret_1.rolling(20).std().replace(0, 1e-6)
    ret_20_adj = ret_20 / vol_20
    
    mom_252 = close_df.shift(20) / close_df.shift(252) - 1.0
    
    clv = ((close_df - low_df) - (high_df - close_df)) / (high_df - low_df).replace(0, 1e-6)
    
    def cs_rank(df):
        return df.rank(axis=1, pct=True)
        
    r_ret_1 = cs_rank(ret_1)
    r_ret_5 = cs_rank(ret_5)
    r_ret_20_adj = cs_rank(ret_20_adj)
    r_mom = cs_rank(mom_252)
    r_vol = cs_rank(vol_20)
    r_clv = cs_rank(clv)
    
    combined_score = (
        -0.35 * r_ret_1 
        -0.25 * r_ret_5 
        -0.10 * r_ret_20_adj 
        + 0.15 * r_mom 
        - 0.10 * r_vol
        - 0.05 * r_clv
    )
    
    forward_ret = close_df.shift(-1) / close_df - 1.0
    
    df_eval = pd.concat([combined_score.stack(), forward_ret.stack()], axis=1).dropna()
    df_eval.columns = ['score', 'fwd_ret']
    
    daily_ic = df_eval.groupby(level=0).apply(lambda x: x['score'].corr(x['fwd_ret'], method='spearman'))
    
    print(f"Overall Mean Rank IC: {daily_ic.mean():.4f}")
    print(f"Annualized IC IR: {daily_ic.mean() / daily_ic.std() * np.sqrt(252):.4f}")
    
    try:
        daily_ic.index = pd.to_datetime(daily_ic.index)
        yearly_ic = daily_ic.groupby(daily_ic.index.year).mean()
        print("\nYearly Mean Rank IC:")
        print(yearly_ic)
    except Exception as e:
        pass
        
    print("\nResearch complete. Strategy is ready for production.")