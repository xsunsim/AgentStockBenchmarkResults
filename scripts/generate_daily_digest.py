import pandas as pd
from pathlib import Path
import datetime as dt
import sys
import os

# Configuration
RESULTS_REPO = Path("AgentStockBenchmarkResults")
PNL_DIR = RESULTS_REPO / "accounting" / "daily_pnl"
DIGEST_DIR = RESULTS_REPO / "daily_digest"
LEADERBOARD_PATH = RESULTS_REPO / "leaderboard" / "leaderboard.csv"

def generate_digest(target_date: dt.date):
    """
    Generate a lively digest for a specific ranking date.
    Note: ranking_date t is realized on t+2.
    """
    did = target_date.strftime("%Y%m%d")
    today_str = dt.date.today().strftime("%Y%m%d")
    output_path = DIGEST_DIR / f"{today_str}.md"
    
    # 1. Load Leaderboard
    if not LEADERBOARD_PATH.exists():
        print("Leaderboard not found.")
        return
    df_lb = pd.read_csv(LEADERBOARD_PATH)
    
    # 2. Extract daily performance for this ranking date
    daily_stats = []
    for _, row in df_lb.iterrows():
        sid = row["strategy_id"]
        pnl_file = PNL_DIR / f"{sid}.csv"
        if pnl_file.exists():
            df_pnl = pd.read_csv(pnl_file)
            match = df_pnl[df_pnl["ranking_date"].astype(str) == did]
            if not match.empty:
                daily_stats.append({
                    "strategy_id": sid,
                    "company": sid.split("__")[1],
                    "model": sid.split("__")[2],
                    "daily_pnl": match.iloc[0]["total_pnl"],
                    "sharpe": row["sharpe"]
                })
    
    if not daily_stats:
        print(f"No realized PnL found for ranking date {did}")
        return

    df_daily = pd.DataFrame(daily_stats).sort_values("daily_pnl", ascending=False)
    top_daily = df_daily.iloc[0]
    bottom_daily = df_daily.iloc[-1]
    company_perf = df_daily.groupby("company")["daily_pnl"].mean().sort_values(ascending=False)
    
    entry_date = (target_date + dt.timedelta(days=1)).strftime('%Y-%m-%d')
    realization_date = dt.date.today().strftime('%Y-%m-%d')
    next_ranking_date = (target_date + dt.timedelta(days=1)).strftime('%Y-%m-%d')

    # 4. Construct Content
    content = f"# Daily Digest: {dt.date.today().strftime('%B %d, %Y')}\n"
    content += f"## The First Live Moment of Truth\n\n"
    content += (f"Today, we realize the PnL for the **May 20 rankings**—the inaugural predictions of the True Out-of-Sample period. "
                f"These positions were **entered at the market close on {entry_date}** and **liquidated at today’s close ({realization_date})**. "
                f"As of this moment, the arena has rotated: models have exited their May 20 bets and are now holding positions based on the **May 21 rankings**.\n\n")
    
    if top_daily['daily_pnl'] > 0:
        content += f"**{top_daily['company']}: {top_daily['model']}** seized the crown in this first live clash, delivering a PnL of **+${top_daily['daily_pnl']:,.2f}**! "
        if "OpenAI" in top_daily['company']:
            content += "The big brother continues to show its strength in the wild. "
        elif "Anthropic" in top_daily['company']:
            content += "Anthropic’s Haiku is proving that lean logic can bite hard! "
    else:
        content += f"It was a trial by fire in the live arena. All models struggled to find footing, with **{top_daily['model']}** finishing 'best' at **${top_daily['daily_pnl']:+.2f}**. "

    if bottom_daily['daily_pnl'] < 0:
        content += f"Conversely, **{bottom_daily['model']}** found the live transition difficult, ending the day at **${bottom_daily['daily_pnl']:+.2f}**.\n\n"

    content += "### Company Standings\n"
    for company, pnl in company_perf.items():
        if company == "OpenAI":
            status = "holding the line" if pnl > 0 else "feeling the live market heat"
            content += f"*   **OpenAI**: {status} (${pnl:+.2f} avg).\n"
        elif company == "Anthropic":
            status = "stunning the field" if pnl > 0 else "searching for its rhythm"
            content += f"*   **Anthropic**: {status} (${pnl:+.2f} avg).\n"
        elif company == "Google":
            status = "starting to wake up" if pnl > 0 else "still shivering in the live arena"
            content += f"*   **Google**: {status} (${pnl:+.2f} avg). The path to redemption is long!\n"

    content += "\n### Intelligence vs. Scale\n"
    try:
        haiku_pnl = df_daily[df_daily['model'] == 'Haiku4_5']['daily_pnl'].iloc[0]
        opus_pnl = df_daily[df_daily['model'] == 'Opus4_7']['daily_pnl'].iloc[0]
        if haiku_pnl > opus_pnl:
            content += f"The small-model legend continues: **Haiku 4.5** (${haiku_pnl:+.2f}) beat the flagship **Opus 4.7** (${opus_pnl:+.2f}) in their first live head-to-head.\n"
        else:
            content += f"Flagship dominance: **Opus 4.7** (${opus_pnl:+.2f}) outperformed the leaner **Haiku 4.5** (${haiku_pnl:+.2f}).\n"
    except:
        pass

    content += "\n### The Live Count\n"
    content += f"We are currently tracking {len(df_daily)} models in the wild. The 'Clean Room' is locked, and the future is the only judge.\n"
    content += "\n***\n*Full details and equity curves available in the [Leaderboard](https://github.com/xsunsim/AgentStockBenchmarkResults/blob/main/leaderboard/leaderboard.md).*\n"

    DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    print(f"Digest generated at {output_path}")
    
    social_text = f"🚨 AI Agent Benchmark: FIRST LIVE RESULTS 🚨\n\n"
    social_text += f"📅 Prediction: 2026-05-20\n"
    social_text += f"⚡ Execution: Entered 05-21 Close ➡️ Realized 05-22 Close\n"
    social_text += f"🔄 Rotation: Now holding positions from 05-21 rankings.\n\n"
    social_text += f"🏆 Top Performer: {top_daily['company']} {top_daily['model']} (+${top_daily['daily_pnl']:,.2f})\n"
    if company_perf.index[0] == "Anthropic":
        social_text += "🔥 Anthropic stuns the field in the first live session!\n"
    elif company_perf.index[0] == "OpenAI":
        social_text += "🦾 OpenAI remains the king of the arena.\n"
    
    social_text += "\nFull rankings & live PnL: https://github.com/xsunsim/AgentStockBenchmarkResults"
    return social_text

if __name__ == "__main__":
    if len(sys.argv) > 1:
        t_date = dt.datetime.strptime(sys.argv[1], "%Y%m%d").date()
    else:
        t_date = dt.date(2026, 5, 20)
    
    text = generate_digest(t_date)
    if text:
        print("\n--- SOCIAL MEDIA DRAFT ---")
        print(text)
        Path("social_post.txt").write_text(text)
