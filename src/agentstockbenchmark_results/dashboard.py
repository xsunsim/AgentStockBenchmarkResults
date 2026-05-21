from __future__ import annotations

import datetime as dt
import json
from html import escape
from pathlib import Path
from typing import Optional

from agentstockbenchmark_results.analysis import analyze_results
from agentstockbenchmark_results.io import atomic_write_text


def render_dashboard(
    results_repo: Path,
    as_of: Optional[dt.date] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Path]:
    if output_dir is None:
        output_dir = results_repo / "dashboard"
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = analyze_results(results_repo, as_of=as_of)
    json_path = output_dir / "dashboard.json"
    html_path = output_dir / "index.html"

    atomic_write_text(json_path, json.dumps(analysis, indent=2) + "\n")
    atomic_write_text(html_path, render_dashboard_html(analysis))
    return {"html": html_path, "json": json_path}


def render_dashboard_html(analysis: dict) -> str:
    summary = analysis["summary"]
    top_rows = table_rows(
        analysis["top_strategies"],
        ["strategy_id", "sharpe", "cumulative_pnl", "max_drawdown", "win_rate", "n_days"],
    )
    pnl_rows = table_rows(
        analysis["daily_pnl"],
        ["ranking_date", "total_pnl", "cumulative_pnl", "strategies"],
    )
    coverage_rows = table_rows(
        analysis["coverage"]["by_date"],
        ["date", "rankings", "portfolios", "metrics_snapshot", "audit"],
    )
    audit_rows = table_rows(
        analysis["audits"],
        ["audit_date", "status", "failures", "warnings"],
    )
    warnings = "\n".join(
        f"<li>{escape(str(item))}</li>" for item in analysis.get("warnings", [])
    )
    if not warnings:
        warnings = "<li>No warnings.</li>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AgentStockBenchmark Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #172026;
      --muted: #637083;
      --line: #d9dee7;
      --accent: #0f766e;
      --bad: #b42318;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.4;
    }}
    header {{
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 20px 28px;
    }}
    h1 {{ margin: 0; font-size: 24px; font-weight: 700; }}
    h2 {{ margin: 0 0 10px; font-size: 16px; font-weight: 700; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 22px 28px 40px; }}
    .meta {{ color: var(--muted); margin-top: 4px; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
      margin: 0 0 18px;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px 14px;
    }}
    .stat span {{ display: block; color: var(--muted); font-size: 12px; }}
    .stat strong {{ display: block; margin-top: 4px; font-size: 20px; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px;
      margin-top: 14px;
      overflow: auto;
    }}
    table {{ border-collapse: collapse; width: 100%; min-width: 720px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 9px; text-align: right; }}
    th {{ background: #f2f4f7; color: #344054; font-size: 12px; }}
    th:first-child, td:first-child {{ text-align: left; }}
    td:first-child {{ font-weight: 600; }}
    .warnings li {{ margin: 3px 0; }}
    .status-PASS {{ color: var(--accent); font-weight: 700; }}
    .status-FAIL, .status-INVALID_JSON {{ color: var(--bad); font-weight: 700; }}
    .chart-wrap {{ min-width: 720px; }}
    .chart {{
      display: block;
      width: 100%;
      height: auto;
      background: #fbfcfd;
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    .chart-label {{ fill: var(--muted); font-size: 12px; }}
    .chart-grid {{ stroke: #e7ebf0; stroke-width: 1; }}
    .chart-axis {{ stroke: #98a2b3; stroke-width: 1; }}
    .chart-line {{ fill: none; stroke: var(--accent); stroke-width: 3; }}
  </style>
</head>
<body>
  <header>
    <h1>AgentStockBenchmark Dashboard</h1>
    <div class="meta">Generated {escape(str(analysis["generated_at_utc"]))} - as of {escape(str(analysis.get("as_of") or "latest"))}</div>
  </header>
  <main>
    <div class="stats">
      {stat("Strategies", summary["strategy_count"])}
      {stat("Top Sharpe", f"{float(summary['top_sharpe']):.3f}")}
      {stat("Top Cumulative PnL", f"{float(summary['top_cumulative_pnl']):,.2f}")}
      {stat("Daily PnL Rows", summary["daily_pnl_rows"])}
      {stat("Ranking Dates", summary["ranking_dates"])}
      {stat("Portfolio Dates", summary["portfolio_dates"])}
      {stat("Audit Pass", summary["audit_pass"])}
      {stat("Audit Fail", summary["audit_fail"])}
    </div>
    <section>
      <h2>Warnings</h2>
      <ul class="warnings">{warnings}</ul>
    </section>
    <section>
      <h2>Cumulative PnL</h2>
      <div class="chart-wrap">{pnl_chart(analysis["daily_pnl"])}</div>
    </section>
    <section>
      <h2>Top Strategies</h2>
      {table(["strategy_id", "sharpe", "cumulative_pnl", "max_drawdown", "win_rate", "n_days"], top_rows)}
    </section>
    <section>
      <h2>Daily PnL</h2>
      {table(["ranking_date", "total_pnl", "cumulative_pnl", "strategies"], pnl_rows)}
    </section>
    <section>
      <h2>Coverage</h2>
      {table(["date", "rankings", "portfolios", "metrics_snapshot", "audit"], coverage_rows)}
    </section>
    <section>
      <h2>Audits</h2>
      {table(["audit_date", "status", "failures", "warnings"], audit_rows)}
    </section>
  </main>
</body>
</html>
"""


def stat(label: str, value) -> str:
    return f'<div class="stat"><span>{escape(label)}</span><strong>{escape(str(value))}</strong></div>'


def table(headers: list[str], rows: str) -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    if not rows:
        rows = f"<tr><td colspan=\"{len(headers)}\">No rows.</td></tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def table_rows(rows: list[dict], headers: list[str]) -> str:
    out = []
    for row in rows:
        cells = []
        for header in headers:
            value = row.get(header, "")
            text = format_value(value)
            class_name = ""
            if header == "status" or header == "audit":
                class_name = f' class="status-{escape(str(value))}"'
            cells.append(f"<td{class_name}>{escape(text)}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(out)


def format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:,.3f}"
    return str(value)


def pnl_chart(rows: list[dict]) -> str:
    if not rows:
        return "<p>No daily PnL rows.</p>"

    width = 980
    height = 280
    left = 56
    right = 18
    top = 18
    bottom = 38
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [float(row.get("cumulative_pnl") or 0.0) for row in rows]
    dates = [str(row.get("ranking_date") or "") for row in rows]
    lo = min(values)
    hi = max(values)
    if lo == hi:
        lo -= 1.0
        hi += 1.0

    points = []
    for idx, value in enumerate(values):
        x = left + (plot_width * idx / max(1, len(values) - 1))
        y = top + (hi - value) * plot_height / (hi - lo)
        points.append(f"{x:.1f},{y:.1f}")

    zero_y = None
    if lo <= 0.0 <= hi:
        zero_y = top + (hi - 0.0) * plot_height / (hi - lo)

    grid = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_height * frac
        value = hi - (hi - lo) * frac
        grid.append(
            f'<line class="chart-grid" x1="{left}" y1="{y:.1f}" '
            f'x2="{width - right}" y2="{y:.1f}"></line>'
        )
        grid.append(
            f'<text class="chart-label" x="8" y="{y + 4:.1f}">'
            f'{escape(f"{value:.2f}")}</text>'
        )

    if zero_y is not None:
        grid.append(
            f'<line class="chart-axis" x1="{left}" y1="{zero_y:.1f}" '
            f'x2="{width - right}" y2="{zero_y:.1f}"></line>'
        )

    first_date = escape(dates[0])
    last_date = escape(dates[-1])
    return f"""<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Cumulative PnL chart">
  {"".join(grid)}
  <polyline class="chart-line" points="{" ".join(points)}"></polyline>
  <line class="chart-axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"></line>
  <text class="chart-label" x="{left}" y="{height - 12}">{first_date}</text>
  <text class="chart-label" x="{width - right}" y="{height - 12}" text-anchor="end">{last_date}</text>
</svg>"""
