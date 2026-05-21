from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from agentstockbenchmark_results.dates import date_id
from agentstockbenchmark_results.io import atomic_write_text


def analyze_results(
    results_repo: Path,
    as_of: Optional[dt.date] = None,
    output: Optional[Path] = None,
) -> dict[str, Any]:
    repo = results_repo.resolve()
    as_of_id = date_id(as_of)
    metrics_path = metrics_source(repo, as_of_id)
    metrics = load_metrics(metrics_path)
    pnl = load_daily_pnl(repo, as_of_id)
    coverage = collect_coverage(repo)
    audits = collect_audits(repo)
    sources = {
        "metrics": str(metrics_path) if metrics_path is not None else None,
        "daily_pnl_dir": str(repo / "accounting" / "daily_pnl"),
        "audit_dir": str(repo / "manifests" / "audits"),
    }

    payload = {
        "schema_version": 1,
        "results_repo": str(repo),
        "as_of": as_of_id,
        "generated_at_utc": utc_now(),
        "sources": sources,
        "summary": build_summary(metrics, pnl, coverage, audits),
        "top_strategies": top_strategy_rows(metrics),
        "daily_pnl": daily_pnl_rows(pnl),
        "coverage": coverage,
        "audits": audits,
        "warnings": build_warnings(metrics, pnl, coverage, audits, as_of_id, sources),
    }
    safe_payload = json_safe(payload)
    if output is not None:
        atomic_write_text(output, json.dumps(safe_payload, indent=2) + "\n")
    return safe_payload


def metrics_source(results_repo: Path, as_of_id: Optional[str] = None) -> Optional[Path]:
    candidates = []
    if as_of_id:
        candidates.append(results_repo / "accounting" / "metrics" / f"{as_of_id}.csv")
    candidates.append(results_repo / "accounting" / "latest_metrics.csv")

    for path in candidates:
        if path.exists():
            return path
    return None


def load_metrics(source: Optional[Path] = None) -> pd.DataFrame:
    if source is not None and source.exists():
        return pd.read_csv(source)
    return pd.DataFrame(
        columns=[
            "strategy_id",
            "sharpe",
            "cumulative_pnl",
            "max_drawdown",
            "win_rate",
            "avg_daily_pnl",
            "n_days",
        ]
    )


def load_daily_pnl(results_repo: Path, as_of_id: Optional[str] = None) -> pd.DataFrame:
    pnl_dir = results_repo / "accounting" / "daily_pnl"
    frames = []
    if not pnl_dir.exists():
        return pd.DataFrame()

    for path in sorted(pnl_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        if "strategy_id" not in frame.columns:
            frame["strategy_id"] = path.stem
        frame["source_file"] = path.name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "ranking_date" in combined.columns:
        combined["ranking_date"] = combined["ranking_date"].astype(str)
        if as_of_id:
            combined = combined[combined["ranking_date"] <= as_of_id]
    return combined


def collect_coverage(results_repo: Path) -> dict[str, Any]:
    ranking_dates = date_dirs(results_repo / "rankings")
    portfolio_dates = date_dirs(results_repo / "portfolios")
    metric_dates = sorted(
        path.stem
        for path in (results_repo / "accounting" / "metrics").glob("*.csv")
        if path.stem.isdigit()
    ) if (results_repo / "accounting" / "metrics").exists() else []

    all_dates = sorted(set(ranking_dates) | set(portfolio_dates) | set(metric_dates))
    by_date = []
    for did in all_dates:
        by_date.append(
            {
                "date": did,
                "rankings": count_csv(results_repo / "rankings" / did),
                "portfolios": count_csv(results_repo / "portfolios" / did),
                "metrics_snapshot": (
                    results_repo / "accounting" / "metrics" / f"{did}.csv"
                ).exists(),
                "audit": audit_status(results_repo, did),
            }
        )

    return {
        "ranking_dates": ranking_dates,
        "portfolio_dates": portfolio_dates,
        "metric_dates": metric_dates,
        "by_date": by_date,
    }


def collect_audits(results_repo: Path) -> list[dict[str, Any]]:
    audit_dir = results_repo / "manifests" / "audits"
    if not audit_dir.exists():
        return []
    rows = []
    for path in sorted(audit_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            rows.append({"audit_date": path.stem, "status": "INVALID_JSON"})
            continue
        rows.append(
            {
                "audit_date": str(payload.get("audit_date", path.stem)),
                "status": str(payload.get("status", "UNKNOWN")),
                "failures": len(payload.get("failures", [])),
                "warnings": len(payload.get("warnings", [])),
            }
        )
    return rows


def build_summary(
    metrics: pd.DataFrame,
    pnl: pd.DataFrame,
    coverage: dict[str, Any],
    audits: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered = ordered_metrics(metrics)
    top = ordered.iloc[0].to_dict() if not ordered.empty else {}
    latest_pnl_date = None
    if not pnl.empty and "ranking_date" in pnl.columns:
        latest_pnl_date = str(pnl["ranking_date"].max())

    return {
        "strategy_count": int(len(metrics)),
        "top_strategy_id": str(top.get("strategy_id", "")) if top else "",
        "top_sharpe": float(top.get("sharpe", 0.0)) if top else 0.0,
        "top_cumulative_pnl": float(top.get("cumulative_pnl", 0.0)) if top else 0.0,
        "daily_pnl_rows": int(len(pnl)),
        "latest_pnl_date": latest_pnl_date,
        "ranking_dates": len(coverage.get("ranking_dates", [])),
        "portfolio_dates": len(coverage.get("portfolio_dates", [])),
        "audit_pass": sum(1 for row in audits if row.get("status") == "PASS"),
        "audit_fail": sum(1 for row in audits if row.get("status") == "FAIL"),
    }


def top_strategy_rows(metrics: pd.DataFrame, limit: int = 25) -> list[dict[str, Any]]:
    if metrics.empty:
        return []
    return json_safe(ordered_metrics(metrics).head(limit).to_dict(orient="records"))


def ordered_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    frame = metrics.copy()
    sort_cols = [col for col in ["sharpe", "cumulative_pnl"] if col in frame.columns]
    for col in sort_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.sort_values(sort_cols, ascending=False) if sort_cols else frame


def daily_pnl_rows(pnl: pd.DataFrame) -> list[dict[str, Any]]:
    if pnl.empty or "ranking_date" not in pnl.columns:
        return []
    numeric_pnl = pd.to_numeric(pnl.get("total_pnl"), errors="coerce")
    frame = pnl.assign(total_pnl=numeric_pnl)
    grouped = (
        frame.groupby("ranking_date", as_index=False)
        .agg(total_pnl=("total_pnl", "sum"), strategies=("strategy_id", "nunique"))
        .sort_values("ranking_date")
    )
    grouped["cumulative_pnl"] = grouped["total_pnl"].cumsum()
    return json_safe(grouped.to_dict(orient="records"))


def build_warnings(
    metrics: pd.DataFrame,
    pnl: pd.DataFrame,
    coverage: dict[str, Any],
    audits: list[dict[str, Any]],
    as_of_id: Optional[str],
    sources: dict[str, Any],
) -> list[str]:
    warnings = []
    if metrics.empty:
        warnings.append("No metrics file found.")
    elif as_of_id and not str(sources.get("metrics") or "").endswith(f"{as_of_id}.csv"):
        warnings.append(
            f"No dated metrics snapshot found for {as_of_id}; using latest metrics."
        )
    if pnl.empty:
        warnings.append("No daily PnL files found.")
    failed = [row for row in audits if row.get("status") == "FAIL"]
    if failed:
        warnings.append(f"{len(failed)} audit manifest(s) are failing.")
    for row in coverage.get("by_date", []):
        if row["rankings"] and not row["portfolios"]:
            warnings.append(f"{row['date']} has rankings but no portfolios.")
    return warnings


def date_dirs(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(
        child.name
        for child in path.iterdir()
        if child.is_dir() and child.name.isdigit()
    )


def count_csv(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.glob("*.csv") if item.is_file())


def audit_status(results_repo: Path, did: str) -> str:
    path = results_repo / "manifests" / "audits" / f"{did}.json"
    if not path.exists():
        return ""
    try:
        return str(json.loads(path.read_text()).get("status", "UNKNOWN"))
    except json.JSONDecodeError:
        return "INVALID_JSON"


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    try:
        is_missing = pd.isna(value)
    except (TypeError, ValueError):
        is_missing = False
    if isinstance(is_missing, bool) and is_missing:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
