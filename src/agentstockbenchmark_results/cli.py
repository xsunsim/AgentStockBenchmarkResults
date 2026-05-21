from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Optional


def parse_date(value: str) -> dt.date:
    from agentstockbenchmark_results.dates import require_date

    try:
        return require_date(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentstockbenchmark-results",
        description="Fetch and visualize AgentStockBenchmark result artifacts",
    )
    sub = parser.add_subparsers(dest="command")

    copy_p = sub.add_parser("copy-artifacts", help="Copy artifacts from a local source")
    copy_p.add_argument("--source", type=Path, required=True)
    copy_p.add_argument("--dest", type=Path, required=True)
    copy_p.add_argument("--overwrite", action="store_true")

    download_p = sub.add_parser("download-artifact", help="Download one artifact by URL")
    download_p.add_argument("--url", required=True)
    download_p.add_argument("--dest", type=Path, required=True)
    download_p.add_argument("--overwrite", action="store_true")

    render_p = sub.add_parser("render-leaderboard", help="Render leaderboard files")
    render_p.add_argument("--results-repo", type=Path, default=Path.cwd())

    analyze_p = sub.add_parser("analyze", help="Summarize result artifacts as JSON")
    analyze_p.add_argument("--results-repo", type=Path, default=Path.cwd())
    analyze_p.add_argument("--as-of", type=parse_date, default=None)
    analyze_p.add_argument("--output", type=Path, default=None)

    dashboard_p = sub.add_parser(
        "render-dashboard",
        help="Render dashboard/dashboard.json and dashboard/index.html",
    )
    dashboard_p.add_argument("--results-repo", type=Path, default=Path.cwd())
    dashboard_p.add_argument("--as-of", type=parse_date, default=None)
    dashboard_p.add_argument("--output-dir", type=Path, default=None)

    report_p = sub.add_parser("build-report", help="Render Markdown and HTML reports")
    report_p.add_argument("--results-repo", type=Path, default=Path.cwd())
    report_p.add_argument("--as-of", type=parse_date, default=None)
    report_p.add_argument("--output-dir", type=Path, default=None)

    pull_p = sub.add_parser(
        "pull-results",
        help="Pull selected public/date-scoped result artifacts from local or HTTP source",
    )
    pull_p.add_argument("--source", required=True)
    pull_p.add_argument("--dest", type=Path, default=Path.cwd())
    pull_p.add_argument("--date", type=parse_date, default=None)
    pull_p.add_argument("--overwrite", action="store_true")
    pull_p.add_argument("--no-verify-checksums", action="store_true")

    publish_p = sub.add_parser(
        "publish",
        help="Require a passed audit, render public artifacts, and optionally push",
    )
    publish_p.add_argument("--date", type=parse_date, required=True)
    publish_p.add_argument("--results-repo", type=Path, default=Path.cwd())
    publish_p.add_argument("--push", action="store_true")

    args = parser.parse_args(argv)
    try:
        return run_command(args, parser)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def run_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.command == "copy-artifacts":
        from agentstockbenchmark_results.results_fetching import copy_artifacts

        copied = copy_artifacts(args.source, args.dest, args.overwrite)
        for path in copied:
            print(path)
        print(f"{len(copied)} files")
        return 0

    if args.command == "download-artifact":
        from agentstockbenchmark_results.results_fetching import download_artifact

        print(download_artifact(args.url, args.dest, args.overwrite))
        return 0

    if args.command == "render-leaderboard":
        from agentstockbenchmark_results.visualization import render_leaderboard

        paths = render_leaderboard(args.results_repo)
        for kind, path in paths.items():
            print(f"{kind}\t{path}")
        return 0

    if args.command == "analyze":
        from agentstockbenchmark_results.analysis import analyze_results

        payload = analyze_results(args.results_repo, as_of=args.as_of, output=args.output)
        if args.output is not None:
            print(f"analysis\t{args.output}")
        else:
            print(json.dumps(payload, indent=2))
        return 0

    if args.command == "render-dashboard":
        from agentstockbenchmark_results.dashboard import render_dashboard

        paths = render_dashboard(
            args.results_repo,
            as_of=args.as_of,
            output_dir=args.output_dir,
        )
        for kind, path in paths.items():
            print(f"{kind}\t{path}")
        return 0

    if args.command == "build-report":
        from agentstockbenchmark_results.reporting import build_report

        paths = build_report(
            args.results_repo,
            as_of=args.as_of,
            output_dir=args.output_dir,
        )
        for kind, path in paths.items():
            print(f"{kind}\t{path}")
        return 0

    if args.command == "pull-results":
        from agentstockbenchmark_results.results_fetching import pull_results

        records = pull_results(
            args.source,
            args.dest,
            date=args.date,
            overwrite=args.overwrite,
            verify_checksums=not args.no_verify_checksums,
        )
        for record in records:
            dest = record.dest if str(record.dest) != "." else record.source
            detail = f"\t{record.message}" if record.message else ""
            print(f"{record.status}\t{dest}\t{record.bytes}{detail}")
        if any(record.status == "error" for record in records):
            return 1
        if not any(
            record.status in {"copied", "downloaded", "skipped"} for record in records
        ):
            print("error: no result files were pulled or already present", file=sys.stderr)
            return 1
        return 0

    if args.command == "publish":
        from agentstockbenchmark_results.dates import date_id
        from agentstockbenchmark_results.publish import publish_results

        manifest = publish_results(args.results_repo, args.date, push=args.push)
        print(f"{date_id(args.date)}\tPUBLISHED")
        for kind, path in manifest["rendered"].items():
            print(f"{kind}\t{path}")
        if manifest["push"] is not None:
            print(f"push\t{manifest['push']['status']}")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
