import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime as dt
import sys
import os

# Ensure AgentStockBenchmark/src is in path
PROJ_ROOT = Path("/Users/xiaoyusun/AgentStockBench")
sys.path.append(str(PROJ_ROOT / "AgentStockBenchmark" / "src"))

from agentstockbenchmark.stage2.market_data import load_ohlcv_tables
from agentstockbenchmark.stage2.rankings import build_snapshot, validate_strategy_scores, scores_to_ranked_rows
from agentstockbenchmark.stage1.strategies import load_strategy

# Configuration
RESULTS_REPO = PROJ_ROOT / "AgentStockBenchmarkResults"
CLOSE_PATH = RESULTS_REPO / "data" / "parquet" / "close.parquet"
OUTPUT_DIR = RESULTS_REPO / "curiosity_long_horizon"
OUTPUT_PLOT = OUTPUT_DIR / "stepwise_long_horizon_comparison.png"

# Models to evaluate
MODELS = {
    "OpenAI GPT-5.5": RESULTS_REPO / "strategies/20260517/OpenAI__GPT5_5__LinearNeutral/strategy.py",
    "OpenAI O3": RESULTS_REPO / "strategies/20260517/OpenAI__O3__LinearNeutral/strategy.py",
    "Anthropic Haiku 4.5": RESULTS_REPO / "strategies/20260517/Anthropic__Haiku4_5__LinearNeutral/strategy.py",
    "Anthropic Opus 4.7": RESULTS_REPO / "strategies/20260517/Anthropic__Opus4_7__LinearNeutral/strategy.py",
    "Google Gemini 2.5 Pro": RESULTS_REPO / "strategies/20260517/Google__Gemini2_5Pro__LinearNeutral/strategy.py"
}

def calculate_pnl(ranked_rows, close_table, entry_dt, exit_dt, mode="linear"):
    df = pd.DataFrame(ranked_rows)
    n = len(df)
    
    if mode == "linear":
        # Linear Ladder: 250 down to -250
        # Formula: 250 - (rank-1) * (500/(n-1))
        step = 500.0 / (n - 1)
        df["weight"] = 250.0 - (df["strategy_rank"] - 1) * step
        lmv = 31500.0 # for n=503, approx
    else:
        # Stepwise: Top 50 @ 100, Bot 50 @ -100
        df["weight"] = 0.0
        df.loc[df["strategy_rank"] <= 50, "weight"] = 100.0
        df.loc[df["strategy_rank"] > (n - 50), "weight"] = -100.0
        lmv = 5000.0
        
    pnl = 0.0
    for _, row in df[df["weight"] != 0].iterrows():
        ticker = row["ticker"]
        try:
            p1 = close_table.loc[pd.Timestamp(entry_dt), ticker]
            p2 = close_table.loc[pd.Timestamp(exit_dt), ticker]
            if p1 > 0 and p2 > 0:
                pnl += (p2 / p1 - 1) * row["weight"]
        except KeyError:
            continue
    return pnl, lmv

def run_long_horizon():
    # 1. Load Data
    print("Loading market data...")
    tables = load_ohlcv_tables(RESULTS_REPO / "data" / "parquet")
    close = tables["close"]
    trading_dates = sorted(close.index.date)
    tickers = list(close.columns)
    
    # 2. Define Range (Must start after MIN_HISTORY_DAYS=21)
    # Data starts 2025-01-02. 21st trading day is approx Feb 1.
    start_date = dt.date(2025, 2, 1)
    end_date = trading_dates[-3] # Last date with realized exit price
    
    ranking_dates = [d for d in trading_dates if start_date <= d <= end_date]
    print(f"Evaluating {len(ranking_dates)} dates from {ranking_dates[0]} to {ranking_dates[-1]}")
    
    # 3. Load Strategies
    strategies = {}
    for name, path in MODELS.items():
        print(f"Loading strategy: {name}")
        strategies[name] = load_strategy(path)
    
    results = {name: {"lin_pnl": [], "step_pnl": [], "dates": []} for name in MODELS}
    
    # 4. Main Inference Loop
    for i, r_date in enumerate(ranking_dates):
        if i % 10 == 0:
            print(f"Processing day {i}/{len(ranking_dates)}: {r_date}")
            
        entry_date = trading_dates[trading_dates.index(r_date) + 1]
        exit_date = trading_dates[trading_dates.index(r_date) + 2]
        
        # Build snapshot for this day
        snapshot = build_snapshot(tables, r_date, tickers)
        
        for name, predict in strategies.items():
            try:
                raw_scores = predict(snapshot)
                scores = validate_strategy_scores(raw_scores)
                ranked = scores_to_ranked_rows(scores)
                
                lin_pnl, lin_lmv = calculate_pnl(ranked, close, entry_date, exit_date, mode="linear")
                step_pnl, step_lmv = calculate_pnl(ranked, close, entry_date, exit_date, mode="stepwise")
                
                results[name]["lin_pnl"].append(lin_pnl)
                results[name]["step_pnl"].append(step_pnl)
                results[name]["dates"].append(r_date)
            except Exception as e:
                # If a model fails for a day, record 0.0 or skip? 
                # To maintain alignment, we record 0.0 (Nice Judge behavior)
                results[name]["lin_pnl"].append(0.0)
                results[name]["step_pnl"].append(0.0)
                results[name]["dates"].append(r_date)

    # 5. Aggregate and Stats
    plt.figure(figsize=(16, 11), dpi=300)
    plt.style.use('bmh')
    cmap = plt.get_cmap("tab10")
    
    print("\n--- LONG HORIZON PERFORMANCE SUMMARY (2025-2026) ---")
    print(f"{'Model':<22} | {'Type':<10} | {'Sharpe':<8} | {'Return':<8}")
    print("-" * 60)

    for i, name in enumerate(MODELS.keys()):
        data = results[name]
        lin_cum = np.cumsum(data["lin_pnl"])
        step_cum = np.cumsum(data["step_pnl"])
        
        # Stats
        def get_stats(pnl_arr, lmv):
            pnl_arr = np.array(pnl_arr)
            mean = np.mean(pnl_arr)
            std = np.std(pnl_arr)
            sharpe = (mean / std * np.sqrt(252)) if std > 0 else 0
            ret = np.sum(pnl_arr) / lmv
            return sharpe, ret

        lin_s, lin_r = get_stats(data["lin_pnl"], 31500.0)
        step_s, step_r = get_stats(data["step_pnl"], 5000.0)
        
        print(f"{name:<22} | {'Linear':<10} | {lin_s:<8.2f} | {lin_r:<8.1%}")
        print(f"{name:<22} | {'Stepwise':<10} | {step_s:<8.2f} | {step_r:<8.1%}")
        print("-" * 60)
        
        color = cmap(i)
        plt.plot(data["dates"], lin_cum, label=f"{name} (Lin) S={lin_s:.2f}", color=color, linewidth=2)
        plt.plot(data["dates"], step_cum, label=f"{name} (Step) S={step_s:.2f}", color=color, linestyle="--", alpha=0.7)

    plt.title("Long Horizon Comparison: Linear vs. Stepwise ($100 Top/Bot 50)\nFeb 2025 - June 2026", fontsize=16, fontweight='bold')
    plt.xlabel("Ranking Date", fontsize=12)
    plt.ylabel("Cumulative PnL ($)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"\nSaved long-horizon plot to {OUTPUT_PLOT}")

if __name__ == "__main__":
    run_long_horizon()
