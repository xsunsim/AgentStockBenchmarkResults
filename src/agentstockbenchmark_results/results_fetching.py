from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

from agentstockbenchmark_results.dates import date_id
from agentstockbenchmark_results.io import atomic_write_bytes


@dataclass(frozen=True)
class PullRecord:
    source: str
    dest: Path
    status: str
    bytes: int = 0
    message: str = ""


def copy_artifacts(source: Path, dest: Path, overwrite: bool = False) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"source not found: {source}")

    copied: list[Path] = []
    if source.is_file():
        dest_path = dest if dest.suffix else dest / source.name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not dest_path.exists():
            shutil.copy2(source, dest_path)
        copied.append(dest_path)
        return copied

    for source_path in sorted(p for p in source.rglob("*") if p.is_file()):
        rel = source_path.relative_to(source)
        dest_path = dest / rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not dest_path.exists():
            shutil.copy2(source_path, dest_path)
        copied.append(dest_path)
    return copied


def download_artifact(url: str, dest: Path, overwrite: bool = False) -> Path:
    if dest.exists() and not overwrite:
        return dest
    with urllib.request.urlopen(url) as response:
        atomic_write_bytes(dest, response.read())
    return dest


def pull_results(
    source: Union[str, Path],
    dest: Path,
    date: Optional[dt.date] = None,
    overwrite: bool = False,
    verify_checksums: bool = True,
) -> list[PullRecord]:
    did = date_id(date)
    dest = dest.expanduser()
    source_text = str(source)

    if is_url(source_text):
        records = pull_results_from_url(source_text, dest, did, overwrite)
    else:
        records = pull_results_from_local(
            Path(source_text).expanduser(),
            dest,
            did,
            overwrite,
        )

    if verify_checksums and did:
        failures = verify_artifact_checksums(dest, did)
        if failures:
            raise ValueError(
                "checksum verification failed:\n"
                + "\n".join(f"- {item}" for item in failures)
            )
    return records


def pull_results_from_local(
    source_root: Path,
    dest: Path,
    did: Optional[str],
    overwrite: bool,
) -> list[PullRecord]:
    if not source_root.exists():
        raise FileNotFoundError(f"source not found: {source_root}")
    if not source_root.is_dir():
        raise ValueError(f"source must be a result directory: {source_root}")
    if source_root.resolve() == dest.resolve():
        raise ValueError("source and destination must be different directories")

    records: list[PullRecord] = []
    for rel in selected_local_files(source_root, did, records):
        source_path = source_root / rel
        if source_path.is_file():
            records.append(copy_one_file(source_path, dest / rel, overwrite))
    return records


def pull_results_from_url(
    source_root: str,
    dest: Path,
    did: Optional[str],
    overwrite: bool,
) -> list[PullRecord]:
    base = source_root.rstrip("/") + "/"
    records: list[PullRecord] = []
    rels = list(selected_public_files(did))

    seen = {rel.as_posix() for rel in rels}
    if did:
        manifest_rel = Path("manifests") / "artifacts" / f"{did}.json"
        if manifest_rel.as_posix() not in seen:
            rels.append(manifest_rel)
            seen.add(manifest_rel.as_posix())

    for rel in rels:
        records.append(download_one_file(base, rel, dest / rel, overwrite))

    if did:
        manifest_path = dest / "manifests" / "artifacts" / f"{did}.json"
        for rel in artifact_manifest_paths(manifest_path):
            key = rel.as_posix()
            if key in seen:
                continue
            seen.add(key)
            records.append(download_one_file(base, rel, dest / rel, overwrite))
    return records


def selected_local_files(
    source_root: Path,
    did: Optional[str],
    records: list[PullRecord],
) -> list[Path]:
    seen: set[str] = set()
    rels: list[Path] = []

    for rel in selected_public_files(did):
        add_selected_file(source_root, rel, rels, seen, records)

    if did:
        for rel in selected_date_files(did):
            add_selected_file(source_root, rel, rels, seen, records)
        manifest_path = source_root / "manifests" / "artifacts" / f"{did}.json"
        for rel in artifact_manifest_paths(manifest_path):
            add_selected_file(source_root, rel, rels, seen, records)
        roots = [
            Path("rankings") / did,
            Path("portfolios") / did,
            Path("accounting") / "daily_pnl",
            Path("leaderboard"),
            Path("dashboard"),
            Path("reports"),
        ]
    else:
        roots = [
            Path("accounting"),
            Path("leaderboard"),
            Path("dashboard"),
            Path("reports"),
            Path("manifests"),
            Path("rankings"),
            Path("portfolios"),
            Path("data") / "universe",
            Path("data") / "raw" / "daily",
        ]

    for root_rel in roots:
        root = source_root / root_rel
        if not root.exists():
            records.append(
                PullRecord(
                    source=str(root),
                    dest=Path(),
                    status="missing",
                    message="optional result path does not exist",
                )
            )
            continue
        if root.is_file():
            add_selected_file(source_root, root_rel, rels, seen, records)
            continue
        for source_path in sorted(path for path in root.rglob("*") if path.is_file()):
            rel = source_path.relative_to(source_root)
            key = rel.as_posix()
            if key not in seen:
                seen.add(key)
                rels.append(rel)

    return rels


def add_selected_file(
    source_root: Path,
    rel: Path,
    rels: list[Path],
    seen: set[str],
    records: list[PullRecord],
) -> None:
    key = rel.as_posix()
    if key in seen:
        return
    seen.add(key)
    source_path = source_root / rel
    if source_path.exists():
        if source_path.is_file():
            rels.append(rel)
        return
    records.append(
        PullRecord(
            source=str(source_path),
            dest=Path(),
            status="missing",
            message="optional result file does not exist",
        )
    )


def selected_public_files(did: Optional[str]) -> Iterable[Path]:
    report_suffix = did or "latest"
    rels = [
        Path("accounting") / "latest_metrics.csv",
        Path("leaderboard") / "leaderboard.csv",
        Path("leaderboard") / "leaderboard.md",
        Path("leaderboard") / "leaderboard.html",
        Path("dashboard") / "dashboard.json",
        Path("dashboard") / "index.html",
        Path("reports") / f"report_{report_suffix}.md",
        Path("reports") / f"report_{report_suffix}.html",
        Path("manifests") / "strategies.json",
    ]
    if did:
        rels.extend(
            [
                Path("manifests") / "audits" / f"{did}.json",
                Path("manifests") / "artifacts" / f"{did}.json",
                Path("manifests") / "runs" / f"{did}.json",
                Path("manifests") / "published" / f"{did}.json",
            ]
        )
    return rels


def selected_date_files(did: str) -> Iterable[Path]:
    return [
        Path("data") / "universe" / f"{did}.txt",
        Path("data") / "raw" / "daily" / f"{did}.csv",
        Path("accounting") / "metrics" / f"{did}.csv",
    ]


def copy_one_file(source: Path, dest: Path, overwrite: bool) -> PullRecord:
    if dest.exists() and not overwrite:
        return PullRecord(
            source=str(source),
            dest=dest,
            status="skipped",
            bytes=dest.stat().st_size,
            message="destination exists",
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return PullRecord(
        source=str(source),
        dest=dest,
        status="copied",
        bytes=dest.stat().st_size,
    )


def download_one_file(
    source_root: str,
    rel: Path,
    dest: Path,
    overwrite: bool,
) -> PullRecord:
    url = urllib.parse.urljoin(source_root, rel.as_posix())
    if dest.exists() and not overwrite:
        return PullRecord(
            source=url,
            dest=dest,
            status="skipped",
            bytes=dest.stat().st_size,
            message="destination exists",
        )
    try:
        with urllib.request.urlopen(url) as response:
            content = response.read()
    except urllib.error.HTTPError as exc:
        status = "missing" if exc.code == 404 else "error"
        return PullRecord(source=url, dest=dest, status=status, message=str(exc))
    except urllib.error.URLError as exc:
        return PullRecord(source=url, dest=dest, status="error", message=str(exc))

    atomic_write_bytes(dest, content)
    return PullRecord(source=url, dest=dest, status="downloaded", bytes=len(content))


def artifact_manifest_paths(manifest_path: Path) -> list[Path]:
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return []
    paths = []
    for entry in manifest.get("artifacts", []):
        rel = entry.get("path")
        if isinstance(rel, str) and rel:
            paths.append(Path(rel))
    return paths


def verify_artifact_checksums(results_repo: Path, did: str) -> list[str]:
    manifest_path = results_repo / "manifests" / "artifacts" / f"{did}.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return [f"invalid artifact manifest JSON: {manifest_path}: {exc}"]

    failures = []
    for entry in manifest.get("artifacts", []):
        rel = entry.get("path")
        expected = entry.get("sha256")
        if not rel or not expected:
            continue
        path = results_repo / str(rel)
        if not path.exists():
            failures.append(f"missing artifact {rel}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(f"checksum mismatch for {rel}")
    return failures


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"}
