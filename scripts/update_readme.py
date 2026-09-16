#!/usr/bin/env python3
"""Update README.md / README_CN.md LATEST sections after the daily digest.

Usage: python scripts/update_readme.py [YYYYMMDD]
Defaults to today. Must run after scripts/generate_daily_digest.py.
"""
import datetime as dt
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DIGEST_DIR = REPO / "daily_digest"
PNL_DIR = REPO / "accounting" / "daily_pnl"
RANKINGS_DIR = REPO / "rankings"
LEADERBOARD = REPO / "leaderboard" / "leaderboard.csv"

MODEL_DISPLAY = {
    "GPT5_5": "GPT-5.5", "GPT5_4": "GPT-5.4", "GPT5_4_mini": "GPT-5.4 mini",
    "GPT5_3_Codex": "GPT-5.3 Codex", "GPT4o": "GPT-4o", "O3": "O3",
    "O4_mini": "O4 mini", "Sonnet4_6": "Sonnet 4.6", "Opus4_6": "Opus 4.6",
    "Opus4_7": "Opus 4.7", "Opus4_8": "Opus 4.8", "Haiku4_5": "Haiku 4.5",
    "Gemini2_5Pro": "Gemini 2.5 Pro", "Gemini2_5Flash": "Gemini 2.5 Flash",
    "Gemini3_1Pro": "Gemini 3.1 Pro", "Gemini3Flash": "Gemini 3 Flash",
}

MONTHS_CN = ["1月", "2月", "3月", "4月", "5月", "6月", "7月",
             "8月", "9月", "10月", "11月", "12月"]


def parse_sid(sid):
    parts = sid.split("__")
    return parts[1], MODEL_DISPLAY.get(parts[2], parts[2])


def next_trading_day(d):
    d += dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def ticker_name(ticker, cache):
    if ticker in cache:
        return cache[ticker]
    name = ticker
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).get_info()
        name = info.get("longName") or info.get("shortName") or ticker
    except Exception:
        pass
    cache[ticker] = name
    return name


def realized_ranking_date(did):
    """Latest ranking_date whose positions were liquidated on `did`."""
    best = None
    for f in PNL_DIR.glob("*.csv"):
        df = pd.read_csv(f, usecols=["ranking_date", "exit_date"])
        m = df[df["exit_date"].astype(str) == did]
        if not m.empty:
            r = m["ranking_date"].astype(str).max()
            if best is None or r > best:
                best = r
    return best


def top_daily_from_digest(did):
    """Parse (company, model, pnl) of the crown winner from the digest table."""
    text = (DIGEST_DIR / f"{did}.md").read_text()
    m = re.search(r"^\| ([^|]+) \| ([^|]+) \| \$([0-9,.\-]+)", text, re.M)
    if not m:
        raise RuntimeError("could not parse digest table")
    return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()


def main():
    today = dt.datetime.strptime(sys.argv[1], "%Y%m%d").date() if len(sys.argv) > 1 else dt.date.today()
    did = today.strftime("%Y%m%d")
    rdate = realized_ranking_date(did)
    if not rdate:
        print(f"No realized PnL for exit date {did}; README not updated.")
        return
    rdate_d = dt.datetime.strptime(rdate, "%Y%m%d").date()

    # Top strategy per company by cumulative PnL
    lb = pd.read_csv(LEADERBOARD)
    lb[["company", "model"]] = lb["strategy_id"].str.split("__", n=2, expand=True)[[1, 2]]
    top = lb.sort_values("cumulative_pnl", ascending=False).drop_duplicates("company")
    top = top.set_index("company")["strategy_id"].to_dict()

    cache = {}
    picks = {}
    for company in ["OpenAI", "Anthropic", "Google"]:
        sid = top[company]
        rdf = pd.read_csv(RANKINGS_DIR / did / f"{sid}.csv")
        rdf = rdf.sort_values("strategy_rank")
        long_pick = rdf.iloc[0]["ticker"]
        short_pick = rdf.iloc[-1]["ticker"]
        _, model_disp = parse_sid(sid)
        picks[company] = (model_disp,
                          f"**{long_pick}** ({ticker_name(long_pick, cache)})",
                          f"**{short_pick}** ({ticker_name(short_pick, cache)})")

    win_company, win_model, win_pnl = top_daily_from_digest(did)
    nxt = next_trading_day(today)

    # ---------- English README ----------
    en = (REPO / "README.md").read_text()
    pred_en = (
        f"### LATEST AI PREDICTIONS (Generated {today.strftime('%B %d')}, for {nxt.strftime('%B %d')} cycle)\n"
        "Here is what the **Top Model from each Company** is betting on for tomorrow's market:\n\n"
        "| Company | Model | 📈 Top Pick (Rank 1) | 📉 Top Short (Bottom 1) |\n"
        "|:---|:---|:---|:---|\n"
    )
    for company in ["OpenAI", "Anthropic", "Google"]:
        model_disp, long_s, short_s = picks[company]
        pred_en += f"| **{company}** | {model_disp} | {long_s} | {short_s} |\n"
    en = re.sub(r"### LATEST AI PREDICTIONS.*?\n---\n",
                pred_en + "\n---\n", en, count=1, flags=re.S)

    title_en = f"{win_company} {win_model} Takes the Crown"
    digest_en = (
        f"### LATEST DAILY DIGEST: {today.strftime('%B %d, %Y')}\n"
        f"**{title_en}:** Today we realized the PnL for the {rdate_d.strftime('%B %d')} rankings. "
        f"**{win_company}'s {win_model}** led the arena with a **${win_pnl}** day. "
        f"[Read the full digest here.](daily_digest/{did}.md) ([中文版](daily_digest/{did}_CN.md))\n"
    )
    en = re.sub(r"### LATEST DAILY DIGEST:[^\n]*\n[^\n]*\n",
                digest_en, en, count=1)
    archive_en = (f"*   [{today.strftime('%B %d, %Y')}: {title_en}](daily_digest/{did}.md) "
                  f"([中文版](daily_digest/{did}_CN.md))\n")
    en = re.sub(r"(### ARCHIVE: DAILY DIGESTS\n)",
                r"\1" + archive_en, en, count=1)
    (REPO / "README.md").write_text(en)

    # ---------- Chinese README ----------
    cn = (REPO / "README_CN.md").read_text()
    pred_cn = (
        f"### AI 智能体最新多空头筹 ({today.month}月{today.day}日生成，针对 {nxt.month}月{nxt.day}日 周期)\n"
        "以下是各大厂当前表现最佳模型对明日市场的核心选择：\n\n"
        "| 公司 | 顶尖模型 | 📈 最看多 (Rank 1) | 📉 最看空 (Bottom 1) |\n"
        "|:---|:---|:---|:---|\n"
    )
    for company in ["OpenAI", "Anthropic", "Google"]:
        model_disp, long_s, short_s = picks[company]
        pred_cn += f"| **{company}** | {model_disp} | {long_s} | {short_s} |\n"
    cn = re.sub(r"### AI 智能体最新多空头筹.*?\n---\n",
                pred_cn + "\n---\n", cn, count=1, flags=re.S)

    title_cn = f"{win_company} {win_model} 夺得桂冠"
    digest_cn = (
        f"### 最新每日摘要：{today.year}年{today.month}月{today.day}日\n"
        f"**{title_cn}：** 今天我们结算了 {rdate_d.month}月{rdate_d.day}日排名 的收益。"
        f"**{win_company} 的 {win_model}** 以 **${win_pnl}** 的单日表现领跑全场。"
        f"[点击阅读完整摘要。](daily_digest/{did}.md) ([中文版](daily_digest/{did}_CN.md))\n"
    )
    cn = re.sub(r"### 最新每日摘要：[^\n]*\n[^\n]*\n",
                digest_cn, cn, count=1)
    archive_cn = (f"*   [{today.year}年{today.month}月{today.day}日：{title_cn}](daily_digest/{did}.md) "
                  f"([中文版](daily_digest/{did}_CN.md))\n")
    cn = re.sub(r"(### 存档：每日摘要\n)",
                r"\1" + archive_cn, cn, count=1)
    (REPO / "README_CN.md").write_text(cn)
    print(f"READMEs updated for {did} (realized ranking {rdate})")


if __name__ == "__main__":
    main()
