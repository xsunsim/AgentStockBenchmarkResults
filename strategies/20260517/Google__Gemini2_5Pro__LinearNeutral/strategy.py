# Strategy: Google__Gemini2_5Pro__LinearNeutral_202605
# Model: Google__Gemini2_5Pro (api: gemini-2.5-pro)
# File: strategies/Google__Gemini2_5Pro__LinearNeutral_202605/strategy.py

import os
import numpy as np
import pandas as pd

def research_pipeline(data_dir="/tmp/tmp9n6yn98w"):
    """
    Load data, engineer features, evaluate performance over time.
    """
    print("Loading data...")
    close_df = pd.read_parquet(os.path.join(data_dir, "close.parquet"))
    open_df = pd.read_parquet(os.path.join(data_dir, "open.parquet"))
    high_df = pd.read_parquet(os.path.join(data_dir, "high.parquet"))
    low_df = pd.read_parquet(os.path.join(data_dir, "low.parquet"))
    volume_df = pd.read_parquet(os.path.join(data_dir, "volume.parquet"))
    
    # Basic data alignment
    close_df = close_df.sort_index()
    open_df = open_df.reindex(index=close_df.index, columns=close_df.columns)
    volume_df = volume_df.reindex(index=close_df.index, columns=close_df.columns)
    
    # Forward fill missing prices, limit to 5 days
    close_df = close_df.ffill(limit=5)
    
    print(f"Data shape: {close_df.shape}")
    print(f"Date range: {close_df.index[0]} to {close_df.index[-1]}")
    
    # Compute forward returns for evaluation
    returns_1d = close_df.pct_change(1).shift(-1)
    
    print("Computing features...")
    # 1. 5-day Reversal
    ret_5d = close_df.pct_change(5)
    
    # 2. 20-day Momentum (skip recent 5 days to avoid short-term reversal overlap)
    price_5d_ago = close_df.shift(5)
    price_25d_ago = close_df.shift(25)
    mom_20d = (price_5d_ago - price_25d_ago) / price_25d_ago
    
    # 3. 60-day Momentum
    price_65d_ago = close_df.shift(65)
    mom_60d = (price_5d_ago - price_65d_ago) / price_65d_ago
    
    # 4. 20-day Volatility
    daily_ret = close_df.pct_change(1)
    vol_20d = daily_ret.rolling(20).std()
    
    train_end = int(len(close_df) * 0.6)
    
    def evaluate_factor(factor_df, is_reversal=False):
        # rank cross-sectionally
        ranks = factor_df.rank(axis=1, pct=True)
        if is_reversal:
            ranks = 1.0 - ranks
            
        # Demean to get portfolio weights (long/short dollar neutral)
        weights = ranks.subtract(ranks.mean(axis=1), axis=0)
        
        # Normalize weights so sum of absolute weights is 1 on each day
        abs_sum = weights.abs().sum(axis=1)
        # Avoid division by zero
        weights = weights.div(abs_sum.replace(0, np.nan), axis=0).fillna(0)
        
        # Daily portfolio return
        port_ret = (weights * returns_1d).sum(axis=1)
        
        port_ret_train = port_ret.iloc[:train_end]
        port_ret_val = port_ret.iloc[train_end:]
        
        sharpe_train = port_ret_train.mean() / port_ret_train.std() * np.sqrt(252) if port_ret_train.std() > 0 else 0
        sharpe_val = port_ret_val.mean() / port_ret_val.std() * np.sqrt(252) if port_ret_val.std() > 0 else 0
        
        return sharpe_train, sharpe_val

    sharpe_rev5_train, sharpe_rev5_val = evaluate_factor(ret_5d, is_reversal=True)
    print(f"5-day Reversal -> Train Sharpe: {sharpe_rev5_train:.2f}, Val Sharpe: {sharpe_rev5_val:.2f}")
    
    sharpe_mom20_train, sharpe_mom20_val = evaluate_factor(mom_20d, is_reversal=False)
    print(f"20-day Momentum -> Train Sharpe: {sharpe_mom20_train:.2f}, Val Sharpe: {sharpe_mom20_val:.2f}")
    
    sharpe_mom60_train, sharpe_mom60_val = evaluate_factor(mom_60d, is_reversal=False)
    print(f"60-day Momentum -> Train Sharpe: {sharpe_mom60_train:.2f}, Val Sharpe: {sharpe_mom60_val:.2f}")
    
    sharpe_vol_train, sharpe_vol_val = evaluate_factor(vol_20d, is_reversal=True)
    print(f"Low Volatility -> Train Sharpe: {sharpe_vol_train:.2f}, Val Sharpe: {sharpe_vol_val:.2f}")
    
    print("Combining features for final model...")
    rank_rev5 = (1.0 - ret_5d.rank(axis=1, pct=True)).fillna(0.5)
    rank_mom20 = mom_20d.rank(axis=1, pct=True).fillna(0.5)
    rank_mom60 = mom_60d.rank(axis=1, pct=True).fillna(0.5)
    rank_vol = (1.0 - vol_20d.rank(axis=1, pct=True)).fillna(0.5)
    
    # Final model weights based on robust classic factor performance
    composite = 0.4 * rank_rev5 + 0.2 * rank_mom20 + 0.2 * rank_mom60 + 0.2 * rank_vol
    
    comp_train, comp_val = evaluate_factor(composite, is_reversal=False)
    print(f"Composite Strategy -> Train Sharpe: {comp_train:.2f}, Val Sharpe: {comp_val:.2f}")

if __name__ == "__main__":
    try:
        research_pipeline()
    except Exception as e:
        print(f"Research pipeline encountered an error: {e}")


def generate_signal(data: dict) -> dict:
    """
    Generate daily stock ranking signals.
    
    Args:
        data: dict mapping ticker (str) -> DataFrame with columns [Date, open, high, low, close, volume]
              Each DataFrame contains history up to the current day.
              
    Returns:
        dict mapping ticker (str) -> signal score (float). Higher = more long.
    """
    scores = {}
    features = {}
    
    for ticker, df in data.items():
        if len(df) < 66:
            scores[ticker] = 0.0
            continue
            
        # Extract and forward-fill close prices to handle missing data gracefully
        close = pd.Series(df['close'].values).ffill().values
        
        # 5-day Return (Reversal factor)
        ret_5d = (close[-1] - close[-6]) / close[-6] if close[-6] != 0 else 0.0
        
        # 20-day and 60-day Momentum (skipping the last 5 days)
        ret_20d = (close[-6] - close[-26]) / close[-26] if len(close) >= 26 and close[-26] != 0 else 0.0
        ret_60d = (close[-6] - close[-66]) / close[-66] if len(close) >= 66 and close[-66] != 0 else 0.0
        
        # 20-day Volatility
        if len(close) >= 21:
            rets = (close[-20:] - close[-21:-1]) / close[-21:-1]
            vol_20d = np.std(rets)
            if np.isnan(vol_20d) or vol_20d == 0:
                vol_20d = 1e-6
        else:
            vol_20d = 1e-6
            
        features[ticker] = {
            'rev_5d': ret_5d,
            'mom_20d': ret_20d,
            'mom_60d': ret_60d,
            'vol': vol_20d
        }
        
    if not features:
        return {}
        
    df_feat = pd.DataFrame.from_dict(features, orient='index')
    
    # Compute cross-sectional percentiles (ranks between 0 and 1)
    df_rank = df_feat.rank(pct=True)
    
    # Reverse ranks where smaller values are better
    score_rev = 1.0 - df_rank['rev_5d']
    score_lowvol = 1.0 - df_rank['vol']
    
    # Final blended factor score
    final_score = 0.4 * score_rev + 0.2 * df_rank['mom_20d'] + 0.2 * df_rank['mom_60d'] + 0.2 * score_lowvol
    
    # Handle any potential NaNs generated during ranking
    final_score = final_score.fillna(0.5)
    
    for ticker in final_score.index:
        scores[ticker] = float(final_score.loc[ticker])
        
    return scores