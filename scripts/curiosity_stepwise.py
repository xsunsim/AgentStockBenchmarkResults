import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime as dt

# Configuration
RESULTS_REPO = Path("AgentStockBenchmarkResults")
RANKINGS_DIR = RESULTS_REPO / "rankings"
CLOSE_PATH = RESULTS_REPO / "data" / "parquet" / "close.parquet"
OUTPUT_PLOT = Path("stepwise_curiosity_comparison.png")

# Expanded Model Set
MODELS = [
    "20260517__OpenAI__GPT5_5__LinearNeutral",
    "20260517__OpenAI__O3__LinearNeutral",
    "20260517__Anthropic__Haiku4_5__LinearNeutral",
    "20260517__Anthropic__Opus4_7__LinearNeutral",
    "20260517__Google__Gemini2_5Flash__LinearNeutral"
]

def calculate_stepwise_pnl(ranking_df, close_table, trading_dates):
    ranking_date = ranking_df["ranking_date"].iloc[0]
    if isinstance(ranking_date, (int, np.int64)):
        ranking_date = str(ranking_date)
    
    ranking_dt = pd.to_datetime(ranking_date).date()
    
    try:
        idx = trading_dates.index(ranking_dt)
        if idx + 2 >= len(trading_dates):
            return None # Not realized yet
        entry_dt = trading_dates[idx + 1]
        exit_dt = trading_dates[idx + 2]
    except ValueError:
        return None

    # Filter ranking
    df = ranking_df.sort_values("strategy_rank")
    
    # Top 50 and Bottom 50
    top_50 = df.head(50).copy()
    bot_50 = df.tail(50).copy()
    
    def get_pnl(subset, weight):
        pnl = 0.0
        for _, row in subset.iterrows():
            ticker = row["ticker"]
            try:
                p1 = close_table.loc[pd.Timestamp(entry_dt), ticker]
                p2 = close_table.loc[pd.Timestamp(exit_dt), ticker]
                if p1 > 0 and p2 > 0:
                    pnl += (p2 / p1 - 1) * weight
            except KeyError:
                continue
        return pnl

    total_pnl = get_pnl(top_50, 100.0) + get_pnl(bot_50, -100.0)
    return {
        "ranking_date": ranking_dt,
        "exit_date": exit_dt,
        "total_pnl": total_pnl
    }

def run_curiosity():
    # 1. Load Data
    close = pd.read_parquet(CLOSE_PATH)
    close.index = pd.to_datetime(close.index)
    trading_dates = sorted(close.index.date)
    
    results = {}
    
    for sid in MODELS:
        print(f"Processing {sid}...")
        # A. Load Linear PnL
        linear_file = RESULTS_REPO / "accounting" / "daily_pnl" / f"{sid}.csv"
        if not linear_file.exists():
            print(f"Skipping {sid} - PnL file not found")
            continue
        df_linear = pd.read_csv(linear_file)
        df_linear["ranking_date"] = pd.to_datetime(df_linear["ranking_date"].astype(str)).dt.date
        df_linear = df_linear.sort_values("ranking_date")
        df_linear["cum_pnl"] = df_linear["total_pnl"].cumsum()
        
        # B. Calculate Stepwise PnL
        stepwise_rows = []
        for folder in sorted(RANKINGS_DIR.iterdir()):
            if not folder.is_dir(): continue
            ranking_file = folder / f"{sid}.csv"
            if ranking_file.exists():
                df_rank = pd.read_csv(ranking_file)
                row = calculate_stepwise_pnl(df_rank, close, trading_dates)
                if row:
                    stepwise_rows.append(row)
        
        if not stepwise_rows:
            continue
            
        df_step = pd.DataFrame(stepwise_rows).sort_values("ranking_date")
        df_step["cum_pnl"] = df_step["total_pnl"].cumsum()
        
        # Calculate Stats
        def get_stats(pnl_series, lmv):
            mean = pnl_series.mean()
            std = pnl_series.std()
            sharpe = (mean / std * np.sqrt(252)) if std > 0 else 0
            ret = pnl_series.sum() / lmv
            return sharpe, ret

        lin_sharpe, lin_ret = get_stats(df_linear["total_pnl"], 31500.0)
        step_sharpe, step_ret = get_stats(df_step["total_pnl"], 5000.0)
        
        results[sid] = {
            "linear": df_linear,
            "stepwise": df_step,
            "lin_sharpe": lin_sharpe,
            "lin_ret": lin_ret,
            "step_sharpe": step_sharpe,
            "step_ret": step_ret
        }

    # 3. Plotting
    plt.figure(figsize=(16, 11), dpi=300)
    plt.style.use('bmh')
    
    # Generate distinct colors
    cmap = plt.get_cmap("tab10")
    
    for i, sid in enumerate(results.keys()):
        res = results[sid]
        model_name = sid.split("__")[2]
        color = cmap(i)
        
        # Plot Linear
        plt.plot(res["linear"]["ranking_date"], res["linear"]["cum_pnl"], 
                 label=f"{model_name} (Linear) | S={res['lin_sharpe']:.2f}, R={res['lin_ret']:.1%}", 
                 color=color, linestyle="-", linewidth=2.5)
        
        # Plot Stepwise
        plt.plot(res["stepwise"]["ranking_date"], res["stepwise"]["cum_pnl"], 
                 label=f"{model_name} (Stepwise) | S={res['step_sharpe']:.2f}, R={res['step_ret']:.1%}", 
                 color=color, linestyle="--", linewidth=1.5, alpha=0.8)

    plt.title("Curiosity Check: Linear vs. Stepwise Portfolio (Top/Bot 50 @ $100)", fontsize=18, fontweight='bold')
    plt.xlabel("Prediction Date", fontsize=12)
    plt.ylabel("Cumulative PnL ($)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"\nSaved curiosity plot to {OUTPUT_PLOT}")
    
    # Print summary table
    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"{'Model':<15} | {'Type':<10} | {'Sharpe':<8} | {'Return':<8}")
    print("-" * 50)
    # Sort printed output by Stepwise Sharpe for clarity
    sorted_sids = sorted(results.keys(), key=lambda x: results[x]["step_sharpe"], reverse=True)
    for sid in sorted_sids:
        res = results[sid]
        m = sid.split("__")[2]
        print(f"{m:<15} | {'Linear':<10} | {res['lin_sharpe']:<8.2f} | {res['lin_ret']:<8.1%}")
        print(f"{m:<15} | {'Stepwise':<10} | {res['step_sharpe']:<8.2f} | {res['step_ret']:<8.1%}")
        print("-" * 50)

if __name__ == "__main__":
    run_curiosity()
