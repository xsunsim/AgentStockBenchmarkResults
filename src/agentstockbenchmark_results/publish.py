from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

from agentstockbenchmark_results.dashboard import render_dashboard
from agentstockbenchmark_results.dates import date_id
from agentstockbenchmark_results.io import atomic_write_text
from agentstockbenchmark_results.reporting import build_report
from agentstockbenchmark_results.visualization import render_leaderboard


def publish_results(
    results_repo: Path,
    publish_date: dt.date,
    push: bool = False,
) -> dict[str, object]:
    did = date_id(publish_date)
    if did is None:
        raise ValueError("publish date is required")
    audit_path = results_repo / "manifests" / "audits" / f"{did}.json"
    if not audit_path.exists():
        raise FileNotFoundError(f"passed audit manifest required: {audit_path}")

    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "PASS":
        raise ValueError(f"audit is not PASS for {did}: {audit.get('status')}")
    audit_date = audit.get("audit_date")
    if audit_date is not None and str(audit_date) != did:
        raise ValueError(f"audit manifest date {audit_date} does not match {did}")

    rendered = {}
    rendered.update(prefixed_paths("leaderboard", render_leaderboard(results_repo)))
    rendered.update(
        prefixed_paths("dashboard", render_dashboard(results_repo, as_of=publish_date))
    )
    rendered.update(
        prefixed_paths("report", build_report(results_repo, as_of=publish_date))
    )
    manifest = {
        "schema_version": 1,
        "publish_date": did,
        "audit_manifest": str(audit_path),
        "rendered": {key: str(path) for key, path in rendered.items()},
        "push": None,
    }

    if push:
        manifest["push"] = best_effort_git_push(results_repo, did)

    out_path = results_repo / "manifests" / "published" / f"{did}.json"
    atomic_write_text(out_path, json.dumps(manifest, indent=2) + "\n")
    return manifest


def prefixed_paths(prefix: str, paths: dict[str, Path]) -> dict[str, Path]:
    return {f"{prefix}_{key}": path for key, path in paths.items()}


def best_effort_git_push(results_repo: Path, did: str) -> dict[str, object]:
    try:
        subprocess.run(["git", "add", "."], cwd=results_repo, check=True)
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=results_repo,
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout.strip():
            subprocess.run(
                ["git", "commit", "-m", f"publish benchmark results {did}"],
                cwd=results_repo,
                check=True,
            )
        subprocess.run(["git", "push"], cwd=results_repo, check=True)
        return {"status": "PASS"}
    except Exception as exc:
        return {"status": "MANUAL_REQUIRED", "error": str(exc)}
