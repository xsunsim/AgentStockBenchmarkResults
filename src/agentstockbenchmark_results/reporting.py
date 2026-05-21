from __future__ import annotations

import datetime as dt
from html import escape
from pathlib import Path
from typing import Optional

from agentstockbenchmark_results.analysis import analyze_results
from agentstockbenchmark_results.io import atomic_write_text


def build_report(
    results_repo: Path,
    as_of: Optional[dt.date] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Path]:
    if output_dir is None:
        output_dir = results_repo / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = analyze_results(results_repo, as_of=as_of)
    suffix = analysis.get("as_of") or "latest"
    md_path = output_dir / f"report_{suffix}.md"
    html_path = output_dir / f"report_{suffix}.html"

    markdown = render_markdown(analysis)
    atomic_write_text(md_path, markdown)
    atomic_write_text(html_path, render_html_report(analysis))
    return {"markdown": md_path, "html": html_path}


def render_markdown(analysis: dict) -> str:
    summary = analysis["summary"]
    warnings = analysis.get("warnings", [])
    warning_text = "\n".join(f"- {item}" for item in warnings) or "- No warnings."
    top_rows = markdown_table(
        ["Strategy", "Sharpe", "Cum PnL", "Drawdown", "Win Rate", "Days"],
        [
            [
                row.get("strategy_id", ""),
                f"{float(row.get('sharpe') or 0):.3f}",
                f"{float(row.get('cumulative_pnl') or 0):.2f}",
                f"{float(row.get('max_drawdown') or 0):.2f}",
                f"{float(row.get('win_rate') or 0):.3f}",
                row.get("n_days", ""),
            ]
            for row in analysis.get("top_strategies", [])[:10]
        ],
    )
    coverage_rows = markdown_table(
        ["Date", "Rankings", "Portfolios", "Metric", "Audit"],
        [
            [
                row.get("date", ""),
                row.get("rankings", ""),
                row.get("portfolios", ""),
                row.get("metrics_snapshot", ""),
                row.get("audit", ""),
            ]
            for row in analysis["coverage"].get("by_date", [])[-20:]
        ],
    )
    return f"""# AgentStockBenchmark Report

Generated: `{analysis["generated_at_utc"]}`

As of: `{analysis.get("as_of") or "latest"}`

## Summary

- Strategies: `{summary["strategy_count"]}`
- Top strategy: `{summary["top_strategy_id"]}`
- Top Sharpe: `{float(summary["top_sharpe"]):.3f}`
- Top cumulative PnL: `{float(summary["top_cumulative_pnl"]):.2f}`
- Daily PnL rows: `{summary["daily_pnl_rows"]}`
- Latest PnL date: `{summary["latest_pnl_date"] or ""}`
- Audit pass/fail: `{summary["audit_pass"]}/{summary["audit_fail"]}`

## Warnings

{warning_text}

## Top Strategies

{top_rows}

## Recent Coverage

{coverage_rows}
"""


def render_html_report(analysis: dict) -> str:
    summary = analysis["summary"]
    warning_items = "\n".join(
        f"<li>{escape(str(item))}</li>" for item in analysis.get("warnings", [])
    )
    if not warning_items:
        warning_items = "<li>No warnings.</li>"
    top_rows = html_rows(
        [
            [
                row.get("strategy_id", ""),
                f"{float(row.get('sharpe') or 0):.3f}",
                f"{float(row.get('cumulative_pnl') or 0):.2f}",
                f"{float(row.get('max_drawdown') or 0):.2f}",
                f"{float(row.get('win_rate') or 0):.3f}",
                row.get("n_days", ""),
            ]
            for row in analysis.get("top_strategies", [])[:10]
        ]
    )
    coverage_rows = html_rows(
        [
            [
                row.get("date", ""),
                row.get("rankings", ""),
                row.get("portfolios", ""),
                row.get("metrics_snapshot", ""),
                row.get("audit", ""),
            ]
            for row in analysis["coverage"].get("by_date", [])[-20:]
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AgentStockBenchmark Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px auto; max-width: 1040px; color: #172026; line-height: 1.45; }}
    h1 {{ margin-bottom: 4px; }}
    h2 {{ margin-top: 28px; font-size: 18px; }}
    .meta {{ color: #667085; margin-bottom: 20px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; }}
    .summary div {{ border: 1px solid #d9dee7; border-radius: 6px; padding: 10px; }}
    .summary span {{ display: block; color: #667085; font-size: 12px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    th, td {{ border-bottom: 1px solid #d9dee7; padding: 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f2f4f7; font-size: 12px; color: #344054; }}
  </style>
</head>
<body>
  <h1>AgentStockBenchmark Report</h1>
  <div class="meta">Generated {escape(str(analysis["generated_at_utc"]))} - as of {escape(str(analysis.get("as_of") or "latest"))}</div>
  <section class="summary">
    {summary_item("Strategies", summary["strategy_count"])}
    {summary_item("Top strategy", summary["top_strategy_id"])}
    {summary_item("Top Sharpe", f"{float(summary['top_sharpe']):.3f}")}
    {summary_item("Top cumulative PnL", f"{float(summary['top_cumulative_pnl']):.2f}")}
    {summary_item("Daily PnL rows", summary["daily_pnl_rows"])}
    {summary_item("Latest PnL date", summary["latest_pnl_date"] or "")}
    {summary_item("Audit pass/fail", f"{summary['audit_pass']}/{summary['audit_fail']}")}
  </section>
  <h2>Warnings</h2>
  <ul>{warning_items}</ul>
  <h2>Top Strategies</h2>
  {html_table(["Strategy", "Sharpe", "Cum PnL", "Drawdown", "Win Rate", "Days"], top_rows)}
  <h2>Recent Coverage</h2>
  {html_table(["Date", "Rankings", "Portfolios", "Metric", "Audit"], coverage_rows)}
</body>
</html>
"""


def markdown_table(headers: list[str], rows: list[list]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def summary_item(label: str, value) -> str:
    return f"<div><span>{escape(label)}</span><strong>{escape(str(value))}</strong></div>"


def html_table(headers: list[str], rows: str) -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    if not rows:
        rows = f"<tr><td colspan=\"{len(headers)}\">No rows.</td></tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def html_rows(rows: list[list]) -> str:
    rendered = []
    for row in rows:
        rendered.append(
            "<tr>"
            + "".join(f"<td>{escape(str(value))}</td>" for value in row)
            + "</tr>"
        )
    return "\n".join(rendered)
