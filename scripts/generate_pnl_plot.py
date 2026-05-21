import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import datetime as dt
import matplotlib.dates as mdates

# Configuration
RESULTS_REPO = Path("AgentStockBenchmarkResults")
PNL_DIR = RESULTS_REPO / "accounting" / "daily_pnl"
OUTPUT_PATH = RESULTS_REPO / "leaderboard" / "cumulative_pnl.png"
SUBMISSION_DATE = dt.date(2026, 5, 20)  # The date strategies were truly live

# Color mapping
COMPANY_COLORS = {
    "OpenAI": "#1f77b4",     # Blue
    "Anthropic": "#ff7f0e",  # Orange
    "Google": "#2ca02c",     # Green
}

def generate_plot():
    # 1. Load Leaderboard
    leaderboard = pd.read_csv(RESULTS_REPO / "leaderboard" / "leaderboard.csv")
    
    def parse_id(sid):
        parts = sid.split("__")
        return {
            "strategy_id": sid,
            "company": parts[1],
            "model": parts[2]
        }
    
    models = []
    for _, row in leaderboard.iterrows():
        minfo = parse_id(row["strategy_id"])
        minfo["sharpe"] = row["sharpe"]
        minfo["cum_pnl"] = row["cumulative_pnl"]
        minfo["avg_daily"] = row["avg_daily_pnl"]
        models.append(minfo)
    
    df_models = pd.DataFrame(models)
    
    # 2. Setup Figure
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    plt.style.use('bmh')
    
    # 3. Identify best model per company and collect plotting data
    best_models_data = []
    for company, group in df_models.groupby("company"):
        color = COMPANY_COLORS.get(company, "#7f7f7f")
        best_m = group.sort_values("sharpe", ascending=False).iloc[0]
        
        pnl_file = PNL_DIR / f"{best_m['strategy_id']}.csv"
        if not pnl_file.exists():
            continue
            
        df_pnl = pd.read_csv(pnl_file)
        df_pnl["ranking_date"] = pd.to_datetime(df_pnl["ranking_date"].astype(str))
        df_pnl = df_pnl.sort_values("ranking_date")
        df_pnl["cum_pnl"] = df_pnl["total_pnl"].cumsum()
        
        best_models_data.append({
            "company": company,
            "model": best_m["model"],
            "sharpe": best_m["sharpe"],
            "cum_pnl": best_m["cum_pnl"],
            "avg_daily": best_m["avg_daily"],
            "color": color,
            "df_pnl": df_pnl
        })

    # Sort all best models by Sharpe descending
    best_models_data = sorted(best_models_data, key=lambda x: x["sharpe"], reverse=True)

    summary_stats = []
    row_colors = []
    all_dates_list = []

    # 4. Plot and build table stats
    for m in best_models_data:
        df_pnl = m["df_pnl"]
        total_return = m['cum_pnl'] / 500.0
        
        ax.plot(df_pnl["ranking_date"], df_pnl["cum_pnl"], 
                 color=m["color"], linewidth=3, alpha=0.9)
        
        summary_stats.append([
            "", # Placeholder for color
            m["company"], 
            m['model'], 
            f"{m['sharpe']:.2f}", 
            f"${m['cum_pnl']:,.0f}", 
            f"{total_return:+.1%}"
        ])
        row_colors.append(m["color"])
        all_dates_list.extend(df_pnl["ranking_date"])

    # 5. Vertical Split Line (True OOS)
    split_date = pd.Timestamp(SUBMISSION_DATE)
    ax.axvline(split_date, color='#d62728', linestyle='--', linewidth=2, alpha=0.7)
    
    # Extend x-axis
    ax.set_xlim(left=min(all_dates_list), right=split_date + pd.Timedelta(days=5))
    
    # Labels for OOS regions
    ylim = ax.get_ylim()
    y_text = ylim[0] + (ylim[1] - ylim[0]) * 0.1
    ax.text(split_date - pd.Timedelta(days=0.5), y_text, 
            "SEMI-OOS (BACKTEST)", rotation=90, verticalalignment='bottom', 
            horizontalalignment='right', fontsize=10, fontweight='bold', alpha=0.6)
    ax.text(split_date + pd.Timedelta(days=0.5), y_text, 
            "TRUE OOS (LIVE)", rotation=90, verticalalignment='bottom', 
            horizontalalignment='left', fontsize=10, fontweight='bold', color='#d62728')

    # 6. Summary Table
    col_labels = ["Color", "Company", "Model", "Sharpe", "Total PnL", "Return"]
    the_table = plt.table(cellText=summary_stats, colLabels=col_labels, 
                          loc='upper left', bbox=[0.02, 0.65, 0.45, 0.25])
    
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    
    for i, color in enumerate(row_colors):
        cell = the_table[i+1, 0]
        cell.set_facecolor(color)
        cell.get_text().set_text("")
        
    col_widths = [0.08, 0.15, 0.35, 0.12, 0.15, 0.15]
    for i, w in enumerate(col_widths):
        for j in range(len(summary_stats) + 1):
            the_table[j, i].set_width(w)

    # 7. Formatting
    ax.set_title("AI Agent Benchmark: Cumulative PnL Performance", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Prediction Date", fontsize=12)
    ax.set_ylabel("Cumulative PnL ($)", fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"Successfully generated beautiful plot at {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_plot()
