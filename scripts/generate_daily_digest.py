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
                    "long_pnl": match.iloc[0]["long_pnl"],
                    "short_pnl": match.iloc[0]["short_pnl"],
                    "entry_date": str(match.iloc[0]["entry_date"]),
                    "exit_date": str(match.iloc[0]["exit_date"]),
                    "sharpe": row["sharpe"]
                })
    
    if not daily_stats:
        print(f"No realized PnL found for ranking date {did}")
        return

    df_daily = pd.DataFrame(daily_stats).sort_values("daily_pnl", ascending=False)
    top_daily = df_daily.iloc[0]
    bottom_daily = df_daily.iloc[-1]
    company_perf = df_daily.groupby("company")["daily_pnl"].mean().sort_values(ascending=False)
    
    entry_date = df_daily.iloc[0]["entry_date"]
    exit_date = df_daily.iloc[0]["exit_date"]
    
    # Check market context
    all_long_pos = (df_daily["long_pnl"] > 0).all()
    all_short_neg = (df_daily["short_pnl"] < 0).all()
    all_long_neg = (df_daily["long_pnl"] < 0).all()
    all_short_pos = (df_daily["short_pnl"] > 0).all()
    
    market_context = ""
    if all_long_pos and all_short_neg:
        market_context = "Today was a clear bull market—everyone rode the 'market train' on the long side, but those with weak shorts felt the rising tide."
    elif all_long_neg and all_short_pos:
        market_context = "The bears were out in force today. Longs bled across the board, but the best models protected themselves with rock-solid short selections."
    else:
        market_context = "A mixed day in the arena. Factor selection was key as the market showed no clear directional bias."

    # 4. Construct Content (English)
    content = f"# Daily Digest: {dt.date.today().strftime('%B %d, %Y')}\n"
    content += f"## The Arena Update\n\n"
    content += (f"Today, we realize the PnL for the **{target_date.strftime('%B %d')} rankings**. "
                f"These positions were **entered at the market close on {entry_date}** and **liquidated at today’s close ({exit_date})**.\n\n")
    
    content += f"### Market Context\n{market_context}\n\n"

    # PnL Table
    df_table = df_daily[["company", "model", "daily_pnl", "long_pnl", "short_pnl"]].copy()
    df_table["daily_pnl"] = df_table["daily_pnl"].map("${:,.2f}".format)
    df_table["long_pnl"] = df_table["long_pnl"].map("${:,.2f}".format)
    df_table["short_pnl"] = df_table["short_pnl"].map("${:,.2f}".format)
    content += "### Performance Snapshot\n\n" + df_table.to_markdown(index=False) + "\n\n"

    if top_daily['daily_pnl'] > 0:
        content += f"**{top_daily['company']}: {top_daily['model']}** seized the crown today with a daily PnL of **+${top_daily['daily_pnl']:,.2f}**! "
        if "OpenAI" in top_daily['company']:
            content += "The big brother continues to flex its muscles. "
        elif "Anthropic" in top_daily['company']:
            content += "Anthropic is showing incredible momentum in the live arena. "
    
    content += "\n\n### Company Standings\n"
    for company, pnl in company_perf.items():
        if company == "OpenAI":
            status = "holding the line" if pnl > 0 else "feeling the live market pressure"
            content += f"*   **OpenAI**: {status} (${pnl:+.2f} avg).\n"
        elif company == "Anthropic":
            status = "dominating the field" if pnl > 0 else "searching for its rhythm"
            content += f"*   **Anthropic**: {status} (${pnl:+.2f} avg).\n"
        elif company == "Google":
            status = "showing signs of life" if pnl > 0 else "still shivering in the live arena"
            content += f"*   **Google**: {status} (${pnl:+.2f} avg).\n"

    content += "\n### Intelligence vs. Scale\n"
    try:
        haiku_pnl = df_daily[df_daily['model'] == 'Haiku4_5']['daily_pnl'].iloc[0]
        opus_pnl = df_daily[df_daily['model'] == 'Opus4_7']['daily_pnl'].iloc[0]
        if haiku_pnl > opus_pnl:
            content += f"The small-model legend grows: **Haiku 4.5** (${haiku_pnl:+.2f}) beat the flagship **Opus 4.7** (${opus_pnl:+.2f}) today.\n"
        else:
            content += f"Order is restored: **Opus 4.7** (${opus_pnl:+.2f}) outperformed **Haiku 4.5** (${haiku_pnl:+.2f}).\n"
    except:
        pass

    content += "\n***\n*Check the [Full Leaderboard](https://github.com/xsunsim/AgentStockBenchmarkResults/blob/main/leaderboard/leaderboard.md) for cumulative stats.*\n"

    (DIGEST_DIR / f"{today_str}.md").write_text(content)
    
    # 5. Construct Content (Chinese)
    content_cn = f"# 每日摘要：{dt.date.today().strftime('%Y年%m月%d日')}\n"
    content_cn += f"## 竞技场动态\n\n"
    content_cn += (f"今天，我们结算了 **{target_date.month}月{target_date.day}日排名** 的收益。 "
                   f"这些头寸在 **{entry_date} 收盘时建立**，并在 **今日收盘 ({exit_date}) 结算**。\n\n")
    
    market_context_cn = ""
    if all_long_pos and all_short_neg:
        market_context_cn = "今日市场呈现明显的单边上涨趋势——所有模型都在多头端获利，但由于空头端也随大盘上涨，未能实现有效的对冲保护。"
    elif all_long_neg and all_short_pos:
        market_context_cn = "今日空头大获全胜。虽然多头端普遍亏损，但最优秀的模型通过精准的空头选择成功抵御了市场下跌。"
    else:
        market_context_cn = "今日市场震荡，风格切换频繁。在没有明显趋势的情况下，因子的选择成为了胜负的关键。"
    
    content_cn += f"### 市场背景\n{market_context_cn}\n\n"
    
    df_table_cn = df_daily[["company", "model", "daily_pnl", "long_pnl", "short_pnl"]].copy()
    df_table_cn.columns = ["公司", "模型", "今日收益", "多头收益", "空头收益"]
    for col in ["今日收益", "多头收益", "空头收益"]:
        df_table_cn[col] = df_table_cn[col].map("${:,.2f}".format)
    
    content_cn += "### 今日表现快照\n\n" + df_table_cn.to_markdown(index=False) + "\n\n"
    
    if top_daily['daily_pnl'] > 0:
        content_cn += f"**{top_daily['company']}: {top_daily['model']}** 夺得今日桂冠，单日收益达 **+${top_daily['daily_pnl']:,.2f}**！ "
    
    content_cn += "\n\n### 公司表现\n"
    for company, pnl in company_perf.items():
        if company == "OpenAI":
            status = "稳扎稳打" if pnl > 0 else "正感受实盘市场的压力"
            content_cn += f"*   **OpenAI**: {status} (平均收益 ${pnl:+.2f})。\n"
        elif company == "Anthropic":
            status = "统治全场" if pnl > 0 else "正在寻找节奏"
            content_cn += f"*   **Anthropic**: {status} (平均收益 ${pnl:+.2f})。\n"
        elif company == "Google":
            status = "展现生机" if pnl > 0 else "在实盘竞技场中仍显战栗"
            content_cn += f"*   **Google**: {status} (平均收益 ${pnl:+.2f})。\n"

    content_cn += "\n### 智能 vs. 规模\n"
    try:
        if haiku_pnl > opus_pnl:
            content_cn += f"小模型传奇在继续：**Haiku 4.5** (${haiku_pnl:+.2f}) 今日击败了旗舰模型 **Opus 4.7** (${opus_pnl:+.2f})。\n"
        else:
            content_cn += f"旗舰回归：**Opus 4.7** (${opus_pnl:+.2f}) 今日表现优于 **Haiku 4.5** (${haiku_pnl:+.2f})。\n"
    except:
        pass
    
    content_cn += "\n***\n*详细信息请查看 [完整排行榜](https://github.com/xsunsim/AgentStockBenchmarkResults/blob/main/leaderboard/leaderboard.md)。*\n"
    
    (DIGEST_DIR / f"{today_str}_CN.md").write_text(content_cn)
    print(f"Digests generated for {today_str}")

    # Social Media Draft
    social_text = f"🚨 AI Agent Benchmark Update 🚨\n\n"
    social_text += f"📅 Prediction Date: {target_date.strftime('%Y-%m-%d')}\n"
    social_text += f"🏆 Top Model: {top_daily['company']} {top_daily['model']} (${top_daily['daily_pnl']:+.2f})\n"
    social_text += f"📊 Market: {'Bullish 🐂' if all_long_pos else 'Bearish 🐻' if all_long_neg else 'Mixed ⚖️'}\n\n"
    social_text += f"Full results: https://github.com/xsunsim/AgentStockBenchmarkResults"
    
    return social_text

if __name__ == "__main__":
    if len(sys.argv) > 1:
        t_date = dt.datetime.strptime(sys.argv[1], "%Y%m%d").date()
    else:
        # Default to finding the latest realized ranking date
        # This is a bit complex to find purely from files, so we expect the caller to provide it
        # or we just guess T-2 for now if no arg provided.
        t_date = dt.date.today() - dt.timedelta(days=2)
    
    text = generate_digest(t_date)
    print("\n--- SOCIAL MEDIA DRAFT ---")
    print(text)
