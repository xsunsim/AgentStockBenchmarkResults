from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from agentstockbenchmark_results.io import atomic_write_csv, atomic_write_text


def render_leaderboard(results_repo: Path) -> dict[str, Path]:
    metrics_path = results_repo / "accounting" / "latest_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics not found: {metrics_path}")

    df = pd.read_csv(metrics_path)
    out_dir = results_repo / "leaderboard"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "leaderboard.csv"
    md_path = out_dir / "leaderboard.md"
    html_path = out_dir / "leaderboard.html"

    atomic_write_csv(csv_path, df)
    atomic_write_text(
        md_path,
        "# Leaderboard\n\n```\n" + df.to_string(index=False) + "\n```\n",
    )
    atomic_write_text(html_path, render_html(df))
    return {"csv": csv_path, "markdown": md_path, "html": html_path}


def render_html(df: pd.DataFrame) -> str:
    rows = []
    for idx, row in df.iterrows():
        strategy_id = str(row["strategy_id"])
        rows.append(
            "<tr>"
            f"<td>{idx + 1}</td>"
            f"<td>{escape(strategy_id)}</td>"
            f"<td>{float(row['sharpe']):.3f}</td>"
            f"<td>{float(row['cumulative_pnl']):,.2f}</td>"
            f"<td>{float(row['max_drawdown']):,.2f}</td>"
            f"<td>{float(row['win_rate']):.1%}</td>"
            f"<td>{int(row['n_days'])}</td>"
            "</tr>"
        )

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AgentStockBenchmark Results</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px auto; max-width: 1100px; color: #172026; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border-bottom: 1px solid #d8dee4; padding: 9px 11px; text-align: right; }
    th { background: #f5f7f9; font-weight: 650; }
    th:nth-child(2), td:nth-child(2) { text-align: left; }
  </style>
</head>
<body>
  <h1>AgentStockBenchmark Results</h1>
  <table>
    <thead>
      <tr><th>Rank</th><th>Strategy</th><th>Sharpe</th><th>Cumulative PnL</th><th>Max Drawdown</th><th>Win Rate</th><th>Days</th></tr>
    </thead>
    <tbody>
""" + "\n".join(rows) + """
    </tbody>
  </table>
</body>
</html>
"""
