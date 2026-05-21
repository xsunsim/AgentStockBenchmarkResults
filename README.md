# AgentStockBenchmarkResults

This repo stores benchmark artifacts and lightweight result tooling for
AgentStockBenchmark.

It is intentionally separate from `AgentStockBenchmark`:

- `data/raw/` contains source market-data CSV snapshots.
- `data/parquet/` contains derived field-level OHLCV parquets.
- `rankings/` contains frozen model ranking artifacts written before scoring.
- `portfolios/` contains frozen portfolio artifacts built from rankings and the
  applicable S&P 500 universe.
- `accounting/` contains realized PnL and metrics after returns exist.
- `leaderboard/` contains generated CSV, Markdown, and HTML views.
- `manifests/` contains run, audit, artifact, and publish manifests keyed by
  compact `YYYYMMDD` dates.

Only result-facing code lives here: selected artifact fetching, analysis,
dashboard rendering, reporting, and audit-gated publication. Benchmark
execution, strategy loading, and accounting logic live in `AgentStockBenchmark`.

All dated commands use compact `YYYYMMDD` dates. `YYYY-MM-DD` is accepted on the
CLI for convenience, but generated file names use `YYYYMMDD`.

## Analyze Results

Print a JSON summary of metrics, daily PnL, coverage, audits, and warnings:

```bash
cd AgentStockBenchmarkResults
PYTHONPATH=src python -m agentstockbenchmark_results analyze \
  --results-repo . \
  --as-of 20260519
```

Write the same payload to a file:

```bash
PYTHONPATH=src python -m agentstockbenchmark_results analyze \
  --results-repo . \
  --as-of 20260519 \
  --output analysis_20260519.json
```

## Render Leaderboard

```bash
cd AgentStockBenchmarkResults
PYTHONPATH=src python -m agentstockbenchmark_results render-leaderboard \
  --results-repo .
```

This writes:

```text
leaderboard/leaderboard.csv
leaderboard/leaderboard.md
leaderboard/leaderboard.html
```

## Render Dashboard

Generate a static HTML dashboard and the JSON backing payload:

```bash
PYTHONPATH=src python -m agentstockbenchmark_results render-dashboard \
  --results-repo . \
  --as-of 20260519
```

This writes:

```text
dashboard/index.html
dashboard/dashboard.json
```

Open `dashboard/index.html` in a browser.

## Build Report

Generate Markdown and HTML reports:

```bash
PYTHONPATH=src python -m agentstockbenchmark_results build-report \
  --results-repo . \
  --as-of 20260519
```

This writes:

```text
reports/report_20260519.md
reports/report_20260519.html
```

Omit `--as-of` to write `reports/report_latest.md` and
`reports/report_latest.html`.

## Pull Results

Pull public and date-scoped result artifacts from another local result checkout:

```bash
PYTHONPATH=src python -m agentstockbenchmark_results pull-results \
  --source /path/to/another_AgentStockBenchmarkResults \
  --dest . \
  --date 20260519
```

Pull from an HTTP root that exposes the result repo layout:

```bash
PYTHONPATH=src python -m agentstockbenchmark_results pull-results \
  --source https://example.com/AgentStockBenchmarkResults/ \
  --dest . \
  --date 20260519
```

`pull-results` copies public outputs, date manifests, dated raw/universe files,
dated rankings/portfolios, dated metrics, and daily PnL files. It verifies
artifact checksums when `manifests/artifacts/<YYYYMMDD>.json` is available.
Use `--overwrite` to replace existing files. Use `--no-verify-checksums` only
when you intentionally want a partial mirror.

## Publish

Publishing requires a passed audit manifest from `AgentStockBenchmark` and uses compact
dates:

```bash
PYTHONPATH=src python -m agentstockbenchmark_results publish \
  --date 20260519 \
  --results-repo .
```

Publish renders leaderboard, dashboard, and report artifacts, then writes:

```text
manifests/published/20260519.json
```

Add `--push` only when you want the best-effort local Git add/commit/push path.
