"""Promote a verified mapping into the file-browser runtime table.

Everything upstream is deliberately staging-only: the mapping service refuses to
write anywhere under /data/file-browser-data, so a run can be repeated, audited
and thrown away without touching what the runtime serves. Publication is the one
audited door through that wall, and it opens on evidence rather than on a flag:
the run must carry a quality report whose holdout acceptance passed.

Two properties matter more than convenience here.

A batch is a slice, not the whole table. Exeter's runtime CSV holds 24,908 rows
across four work packages; a WP3 run accounts for 759 of them. Publication
therefore merges by charge identifier and leaves every row it did not map alone,
because writing the file wholesale would silently delete the other three work
packages.

Some columns are the runtime's, not the mapping's. `file_count` and the
`local_*` paths record what the downloader has fetched, so they are carried
across from the existing row rather than overwritten with blanks.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class PublicationError(RuntimeError):
    pass


# Written by the mapping run; everything else in the runtime row is the
# runtime's own state and is preserved.
MAPPING_COLUMNS = (
    "amazons3_path",
    "amazons3_confidence",
    "portal_path",
    "file_found",
    "further-information-reference",
)

SOURCE_ID = "originating-authority-charge-identifier"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def runtime_row(mapped: dict[str, str]) -> dict[str, str]:
    """Translate one exported mapping row into the runtime table's columns."""
    status = (mapped.get("path_found") or "").strip()
    return {
        "amazons3_path": (mapped.get("amazons3_path") or "").strip(),
        "amazons3_confidence": (mapped.get("amazons3_path_cfd") or "0.00").strip(),
        "portal_path": (mapped.get("portal_path") or "").strip(),
        "file_found": "yes" if status == "found" else "no scan",
        "further-information-reference": (mapped.get("further-information-reference") or "").strip(),
    }


@dataclass(frozen=True)
class PublicationResult:
    council: str
    batch: str
    runtime_path: Path
    backup_path: Path
    rows_total: int
    rows_updated: int
    rows_unchanged: int
    rows_added: int
    verified_rate: float

    def describe(self) -> dict[str, Any]:
        return {
            "production_published": True,
            "council": self.council,
            "batch": self.batch,
            "runtime_path": str(self.runtime_path),
            "backup_path": str(self.backup_path),
            "rows_total": self.rows_total,
            "rows_updated": self.rows_updated,
            "rows_unchanged": self.rows_unchanged,
            "rows_added": self.rows_added,
            "verified_rate": self.verified_rate,
        }


def clearance(run_directory: Path) -> dict[str, Any]:
    """Read the run's quality report and refuse anything it did not clear.

    Publication rests on the holdout acceptance, which is the only sample the
    mapping spec was never adjusted against. A run that never reached acceptance
    has no evidence to publish on, whatever its working rounds said.
    """
    report_path = run_directory / "quality-report" / "quality-report.json"
    if not report_path.is_file():
        raise PublicationError(
            f"{run_directory} carries no quality report; a mapping is published on its "
            "acceptance evidence, not on request"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("acceptance_reviewed"):
        raise PublicationError(
            "The run has no holdout acceptance round. Working rounds test cases the spec was "
            "adjusted against and cannot clear a mapping for publication."
        )
    if not report.get("passed"):
        raise PublicationError("The holdout acceptance did not pass; the mapping is not cleared")
    return report


def publish_mapping(
    *,
    council: str,
    batch: str,
    mapping_csv: Path,
    runtime_csv: Path,
    run_directory: Path,
    dry_run: bool = False,
) -> PublicationResult:
    report = clearance(run_directory)
    rounds = report.get("rounds") or []
    accepted = [r for r in rounds if r.get("label", "").startswith("acceptance")]
    verified_rate = float(
        (accepted[-1].get("outcome", {}).get("metrics", {}) or {}).get("verified_rate", 0.0)
        if accepted
        else 0.0
    )

    if not runtime_csv.is_file():
        raise PublicationError(f"Runtime table does not exist: {runtime_csv}")
    fieldnames, existing = read_csv_rows(runtime_csv)
    if "oachargeid" not in fieldnames:
        raise PublicationError(f"Runtime table has no oachargeid column: {runtime_csv}")
    by_id = {row["oachargeid"]: row for row in existing}
    if len(by_id) != len(existing):
        raise PublicationError("Runtime table contains duplicate charge identifiers")

    _, mapped_rows = read_csv_rows(mapping_csv)
    if not mapped_rows:
        raise PublicationError(f"Mapping export is empty: {mapping_csv}")

    updated = unchanged = added = 0
    for mapped in mapped_rows:
        oachargeid = (mapped.get(SOURCE_ID) or "").strip()
        if not oachargeid:
            raise PublicationError("Mapping export contains a blank charge identifier")
        replacement = runtime_row(mapped)
        current = by_id.get(oachargeid)
        if current is None:
            # A charge the runtime has never seen. Runtime-owned columns start
            # empty because nothing has been downloaded for it yet.
            row = {name: "" for name in fieldnames}
            row["oachargeid"] = oachargeid
            row.update(replacement)
            by_id[oachargeid] = row
            existing.append(row)
            added += 1
            continue
        if all(current.get(key, "") == value for key, value in replacement.items()):
            unchanged += 1
            continue
        current.update(replacement)
        updated += 1

    backup = runtime_csv.with_name(f"{runtime_csv.name}.bak-{council}-{batch}-{utc_stamp()}")
    if dry_run:
        return PublicationResult(
            council=council,
            batch=batch,
            runtime_path=runtime_csv,
            backup_path=backup,
            rows_total=len(existing),
            rows_updated=updated,
            rows_unchanged=unchanged,
            rows_added=added,
            verified_rate=verified_rate,
        )

    shutil.copy2(runtime_csv, backup)
    temporary = runtime_csv.with_name(f".{runtime_csv.name}.publishing")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing)
    # Replace in one step so a reader never sees a partial table.
    os.replace(temporary, runtime_csv)

    return PublicationResult(
        council=council,
        batch=batch,
        runtime_path=runtime_csv,
        backup_path=backup,
        rows_total=len(existing),
        rows_updated=updated,
        rows_unchanged=unchanged,
        rows_added=added,
        verified_rate=verified_rate,
    )
