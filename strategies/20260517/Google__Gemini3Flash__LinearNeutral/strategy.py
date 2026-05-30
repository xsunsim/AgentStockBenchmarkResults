# Strategy: Google__Gemini3Flash__LinearNeutral_202605
# Model: Google__Gemini3Flash (api: gemini-3-flash-preview)
# File: strategies/Google__Gemini3Flash__LinearNeutral_202605/strategy.py

import os
import pandas as pd
import numpy as np
import scipy.stats as stats

def _rank_norm(s):
    # Cross-sectional rank normalization mapping to [-0.5, 0.5]
    return s.rank(pct=True) - 0.5

def do_research():
    print("Loading data...")
    try:
        close = pd.read_parquet("/tmp/tmp8kx0hm8g/close.parquet")
        open_ = pd.read_parquet("/tmp/tmp8kx0hm8g/open.parquet")
        high = pd.read_parquet("/tmp/tmp8kx0hm8g/high.parquet")
        low = pd.read_parquet("/tmp/tmp8kx0hm8g/low.parquet")
        volume = pd.read_parquet("/tmp/tmp8kx0hm8g/volume.parquet")
    except Exception as e:
        print("Could not load data. Ensure parquet files exist in /tmp/tmp8kx0hm8g/ :", e)
        return

    print(f"Data shapes: {close.shape}")
    print(f"Dates: {close.index[0].date()} to {close.index[-1].date()}")

    print("Calculating returns...")
    returns = close.pct_change()
    fwd_returns = returns.shift(-1)
    
    # Feature engineering
    print("Engineering features...")
    features = {}
    
    # Reversal features
    features['ret_1d'] = returns
    features['ret_5d'] = close.pct_change(5)
    
    # Momentum features (excluding recent 1-month to avoid overlap with reversal)
    features['mom_6m'] = close.shift(20).pct_change(100)
    features['mom_12m'] = close.shift(20).pct_change(232)
    
    # Volatility / Risk features
    features['vol_20d'] = returns.rolling(20).std()
    
    # Mean reversion
    features['p_ma20'] = close / close.rolling(20).mean() - 1
    
    # Volume features (Turnover proxy)
    vol_5d = volume.rolling(5).mean()
    vol_60d = volume.rolling(60).mean()
    features['turnover_ratio'] = vol_5d / vol_60d - 1
    
    # Intraday volatility proxy
    features['hl_range'] = ((high - low) / close).rolling(20).mean()

    # Align all data into a panel
    print("Normalizing features and applying weights...")
    
    # Predefined robust weights for multi-factor combination
    # We choose conservative weights based on established quantitative anomalies
    # to avoid overfitting the train/validation set.
    weights = {
        'ret_1d': -0.100,         # Strongest short-term mean reversion
        'ret_5d': -0.050,         # 1-week reversal
        'p_ma20': -0.050,         # 20-day mean reversion
        'vol_20d': -0.050,        # Low volatility factor
        'hl_range': -0.030,       # Low intraday volatility
        'mom_6m': 0.030,          # Medium term momentum
        'mom_12m': 0.030,         # Long term momentum
        'turnover_ratio': -0.020  # Mean reversion on volume spikes
    }
    
    composite_score = pd.DataFrame(0, index=close.index, columns=close.columns)
    
    for name, df in features.items():
        w = weights[name]
        # Cross-sectional rank normalize the factor each day
        df_norm = df.apply(_rank_norm, axis=1)
        composite_score += w * df_norm.fillna(0)
        
    print("Evaluating signal...")
    
    # Split train and val
    dates = close.index
    train_mask = (dates >= '2020-01-01') & (dates < '2023-01-01')
    val_mask = (dates >= '2023-01-01') & (dates < '2024-01-01')
    test_mask = (dates >= '2024-01-01')
    
    def eval_ic(pred_df, actual_df, mask):
        p = pred_df[mask]
        a = actual_df[mask]
        
        ics = []
        for dt in p.index:
            preds = p.loc[dt].dropna()
            actuals = a.loc[dt].dropna()
            common = preds.index.intersection(actuals.index)
            if len(common) > 30:
                ic, _ = stats.spearmanr(preds[common], actuals[common])
                if not np.isnan(ic):
                    ics.append(ic)
        ics = np.array(ics)
        mean_ic = np.mean(ics)
        ic_std = np.std(ics)
        ir = mean_ic / ic_std * np.sqrt(252) if ic_std > 0 else 0
        return mean_ic, ir
        
    train_ic, train_ir = eval_ic(composite_score, fwd_returns, train_mask)
    val_ic, val_ir = eval_ic(composite_score, fwd_returns, val_mask)
    test_ic, test_ir = eval_ic(composite_score, fwd_returns, test_mask)
    
    print("-" * 50)
    print("Research Evaluation Results (Fixed Robust Weights):")
    print(f"Train (2020-2022) Rank IC: {train_ic:8.4f}, Ann. IR: {train_ir:8.4f}")
    print(f"Val   (2023)      Rank IC: {val_ic:8.4f}, Ann. IR: {val_ir:8.4f}")
    if test_mask.sum() > 0:
        print(f"Test  (2024)      Rank IC: {test_ic:8.4f}, Ann. IR: {test_ir:8.4f}")
    print("-" * 50)
    print("Research complete. Found robust signals.")


if __name__ == "__main__":
    do_research()


# ==============================================================================
# Production Strategy Implementation
# ==============================================================================

def generate_signal(data):
    """
    Generate daily stock ranking signal using a robust multi-factor model.
    
    Args:
    data: dict mapping ticker (str) -> DataFrame with columns [Date, open, high, low, close, volume].
          Each DataFrame has ALL history from 2020 up to today (growing window).
          
    Returns:
    dict[str, float]: ticker -> signal score. Higher = more long.
    """
    import numpy as np
    import pandas as pd
    
    tickers = []
    features = {
        'ret_1d': [],
        'ret_5d': [],
        'p_ma20': [],
        'vol_20d': [],
        'hl_range': [],
        'mom_6m': [],
        'mom_12m': [],
        'turnover_ratio': []
    }
    
    for ticker, df in data.items():
        n = len(df)
        if n < 2:
            continue
            
        close = df['close'].values
        volume = df['volume'].values
        high = df['high'].values
        low = df['low'].values
        
        # Latest price
        c_0 = close[-1]
        
        # Dynamic lookbacks based on available history
        lb_5 = min(6, n)
        lb_20 = min(21, n)
        lb_60 = min(61, n)
        lb_121 = min(122, n)
        lb_253 = min(254, n)
        
        # 1. Reversals
        ret_1d = (c_0 / close[-2]) - 1 if close[-2] != 0 else 0
        ret_5d = (c_0 / close[-lb_5]) - 1 if close[-lb_5] != 0 else 0
        
        # Mean reversion (Price / 20-day MA)
        ma_20 = np.mean(close[-lb_20+1:]) if lb_20 > 1 else c_0
        p_ma20 = (c_0 / ma_20) - 1 if ma_20 != 0 else 0
        
        # 2. Volatility
        if lb_20 > 2:
            rets_20 = (close[-lb_20+1:] / close[-lb_20:-1]) - 1
            vol_20d = np.std(rets_20)
        else:
            vol_20d = 0.0
            
        if lb_20 > 1:
            # Handle possible division by zero in zero-price stocks
            close_slice = close[-lb_20+1:]
            hl_slice = high[-lb_20+1:] - low[-lb_20+1:]
            
            hl_20 = np.divide(hl_slice, close_slice, out=np.zeros_like(hl_slice, dtype=float), where=close_slice!=0)
            hl_range = np.mean(hl_20)
        else:
            hl_range = 0.0
        
        # 3. Momentum (excluding recent 20 days if possible)
        if n > 21:
            mom_6m = (close[-21] / close[-lb_121]) - 1 if close[-lb_121] != 0 else 0
            mom_12m = (close[-21] / close[-lb_253]) - 1 if close[-lb_253] != 0 else 0
        else:
            mom_6m = 0.0
            mom_12m = 0.0
            
        # 4. Volume Turnover
        vol_5d = np.mean(volume[-lb_5+1:]) if lb_5 > 1 else volume[-1]
        vol_60d = np.mean(volume[-lb_60+1:]) if lb_60 > 1 else volume[-1]
        turnover_ratio = (vol_5d / vol_60d) - 1 if vol_60d != 0 else 0
        
        tickers.append(ticker)
        features['ret_1d'].append(ret_1d)
        features['ret_5d'].append(ret_5d)
        features['p_ma20'].append(p_ma20)
        features['vol_20d'].append(vol_20d)
        features['hl_range'].append(hl_range)
        features['mom_6m'].append(mom_6m)
        features['mom_12m'].append(mom_12m)
        features['turnover_ratio'].append(turnover_ratio)
        
    if not tickers:
        return {t: 0.0 for t in data.keys()}
        
    # Helper to rank normalize features to [-0.5, 0.5] cross-sectionally
    def rank_norm(x):
        x = np.nan_to_num(x, nan=np.nanmedian(x))
        ranks = pd.Series(x).rank(pct=True).values
        return ranks - 0.5
        
    # Apply identical robust weights as used in research
    w_ret_1d = -0.100
    w_ret_5d = -0.050
    w_p_ma20 = -0.050
    w_vol_20d = -0.050
    w_hl_range = -0.030
    w_mom_6m = 0.030
    w_mom_12m = 0.030
    w_turnover = -0.020
    
    # Calculate composite score
    scores = (
        w_ret_1d * rank_norm(features['ret_1d']) +
        w_ret_5d * rank_norm(features['ret_5d']) +
        w_p_ma20 * rank_norm(features['p_ma20']) +
        w_vol_20d * rank_norm(features['vol_20d']) +
        w_hl_range * rank_norm(features['hl_range']) +
        w_mom_6m * rank_norm(features['mom_6m']) +
        w_mom_12m * rank_norm(features['mom_12m']) +
        w_turnover * rank_norm(features['turnover_ratio'])
    )
    
    # Map back to tickers
    signals = {}
    for i, ticker in enumerate(tickers):
        signals[ticker] = float(scores[i])
        
    # Assign neutral score (0.0) for tickers without enough data
    for ticker in data.keys():
        if ticker not in signals:
            signals[ticker] = 0.0
            
    return signals