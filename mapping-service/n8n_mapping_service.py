#!/usr/bin/env python3
"""Small localhost bridge between n8n and the isolated mapping engine.

The service accepts only evidence paths below /data (excluding the protected
file-browser runtime tree), freezes them into an explicit mapping manifest,
runs the existing autonomous capture-rule compiler, and exports two
case-complete mapping tables plus the full audit trail.
"""

from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import re
import shutil
import subprocess
import threading
import traceback
from collections import Counter
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable

import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent
AUTONOMOUS_ENTRYPOINT = REPOSITORY_ROOT / "amazons3-mapping" / "autonomous_mapping.py"
sys.path.insert(0, str(REPOSITORY_ROOT / "amazons3-mapping"))
from autonomous.path_policy import read_only_job_isolated_command  # noqa: E402
DATA_ROOT = Path("/data")
PROTECTED_ROOT = DATA_ROOT / "file-browser-data"
JOBS_ROOT = DATA_ROOT / "mapping-jobs"
CODEX_RUNTIME_HOME = JOBS_ROOT / "codex-runtime"
CODEX_AUTH = Path("/home/rmsi/.codex/auth.json")

MAPPING_TOTAL_FIELDS = (
    "batch",
    "originating-authority-charge-identifier",
    "further-information-reference",
    "supplementary-information",
    "charge-address",
    "charge-geographic-description",
    "amazons3_path",
    "amazons3_path_cfd",
    "amazons3_path_mappingrule",
    "amazons3_path_note",
    "portal_path",
    "portal_path_cfd",
    "portal_path_mappingrule",
    "portal_path_note",
    "path_source",
    "path_found",
)

MAPPINGJ_FIELDS = (
    "batch",
    "originating-authority-charge-identifier",
    "further-information-reference",
    "supplementary-information",
    "charge-address",
    "charge-geographic-description",
    "amazons3_path",
    "portal_path",
    "path_source",
    "path_found",
)

SUPPORTED_TABLES = {".csv", ".json", ".jsonl", ".xlsx"}
SUPPORTED_RULES = {".docx", ".pdf", ".txt"}
REQUEST_LOCK = threading.Lock()


class MappingServiceError(RuntimeError):
    pass


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slug(value: str, *, label: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    if not cleaned:
        raise MappingServiceError(f"{label} must contain letters or digits")
    return cleaned


def safe_data_path(value: str, *, kind: str = "file") -> Path:
    if not value or "\x00" in value:
        raise MappingServiceError(f"A nonblank {kind} path is required")
    path = Path(value).expanduser().resolve(strict=True)
    if path != DATA_ROOT and DATA_ROOT not in path.parents:
        raise MappingServiceError(f"{kind} path must be below /data: {path}")
    if path == PROTECTED_ROOT or PROTECTED_ROOT in path.parents:
        raise MappingServiceError("Access to /data/file-browser-data is forbidden")
    if kind == "directory" and not path.is_dir():
        raise MappingServiceError(f"Input directory does not exist: {path}")
    if kind == "file" and not path.is_file():
        raise MappingServiceError(f"Evidence file does not exist: {path}")
    return path


def list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in re.split(r"[\r\n,]+", str(value)) if part.strip()]


def classify_input_files(directory: Path) -> tuple[list[Path], list[Path], list[Path]]:
    files = sorted((path for path in directory.iterdir() if path.is_file()), key=lambda path: path.name.casefold())
    if len(files) > 40:
        raise MappingServiceError("Input directory contains more than 40 files")
    source: list[Path] = []
    rules: list[Path] = []
    breakdown: list[Path] = []
    for path in files:
        words = " ".join(re.findall(r"[a-z0-9]+", path.name.casefold()))
        suffix = path.suffix.casefold()
        if suffix in SUPPORTED_RULES and "capture" in words and "rule" in words:
            rules.append(path)
        elif suffix in SUPPORTED_TABLES and ("source breakdown" in words or "data source breakdown" in words):
            breakdown.append(path)
        elif suffix in SUPPORTED_TABLES and (
            "source information" in words
            or "source record" in words
            or "landmark" in words
            or re.search(r"\b[a-z]{2,5}\s+prod\d+\b", words)
        ):
            source.append(path)
    return source, rules, breakdown


def select_latest_rule_versions(paths: Iterable[Path]) -> list[Path]:
    """Keep the highest vN within an otherwise identical filename family."""
    selected: dict[str, tuple[int, Path]] = {}
    for path in paths:
        stem = re.sub(r"^\d{8}T\d{6}Z_", "", path.stem, flags=re.IGNORECASE)
        match = re.search(r"(?:^|[-_\s])v(\d+)(?:$|[-_\s])", stem, flags=re.IGNORECASE)
        version = int(match.group(1)) if match else 0
        family = re.sub(r"(?:^|[-_\s])v\d+(?:$|[-_\s])", " ", stem, flags=re.IGNORECASE)
        family = " ".join(re.findall(r"[a-z0-9]+", family.casefold()))
        current = selected.get(family)
        if current is None or version > current[0]:
            selected[family] = (version, path)
    return [item[1] for item in sorted(selected.values(), key=lambda item: item[1].name.casefold())]


def write_csv(path: Path, rows: Iterable[dict[str, str]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="", errors="replace") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def source_value(row: dict[str, str], field: str) -> str:
    """Read a requested Landmark field without rewriting its source value."""
    if field in row:
        return (row.get(field) or "").strip()
    normalized = re.sub(r"[^a-z0-9]+", "", field.casefold())
    for key, value in row.items():
        if re.sub(r"[^a-z0-9]+", "", key.casefold()) == normalized:
            return (value or "").strip()
    return ""


def export_outputs(*, workspace: Path, council: str, batch: str, output_directory: Path) -> dict[str, Any]:
    mapping_path = workspace / "mapping" / "proposed-mapping.csv"
    audit_path = workspace / "mapping" / "mapping-audit.csv"
    spec_path = workspace / "spec" / "mapping-spec.json"
    validation_path = workspace / "validation" / "validation.json"
    source_path = workspace / "prepared" / "source-records.csv"
    for path in (mapping_path, audit_path, spec_path, validation_path, source_path):
        if not path.is_file():
            raise MappingServiceError(f"Mapping job did not produce required artifact: {path}")

    mappings = read_csv(mapping_path)
    audits = read_csv(audit_path)
    source_rows = read_csv(source_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    source_id_field = str(spec.get("source_id_field") or "oachargeid")
    if len(mappings) != len(audits):
        raise MappingServiceError("Mapping and audit row counts differ")
    audit_by_id = {row.get("oachargeid", ""): row for row in audits}
    if len(audit_by_id) != len(audits):
        raise MappingServiceError("Mapping audit contains blank or duplicate case identifiers")
    source_by_id = {source_value(row, source_id_field): row for row in source_rows}
    if len(source_by_id) != len(source_rows) or any(not oid for oid in source_by_id):
        raise MappingServiceError("Source evidence contains blank or duplicate case identifiers")

    mapping_total_rows: list[dict[str, str]] = []
    mappingj_rows: list[dict[str, str]] = []
    for mapping in mappings:
        oid = (mapping.get("oachargeid") or "").strip()
        audit = audit_by_id.get(oid)
        source = source_by_id.get(oid)
        if audit is None:
            raise MappingServiceError(f"Missing audit row for case {oid!r}")
        if source is None:
            raise MappingServiceError(f"Missing source row for case {oid!r}")
        raw_status = (audit.get("match_status") or "").strip()
        if raw_status.startswith("accepted_"):
            status = "found"
        elif raw_status == "not_found":
            status = "no_found"
        else:
            status = "no_match"
        route = (audit.get("route") or "").strip()
        source_type = "amazons3" if route == "s3" else ("portal" if route == "portal" else "")
        reason = (audit.get("rejection_reason") or "").strip() or raw_status
        amazon_path = (mapping.get("amazons3_path") or "").strip()
        portal_path = (mapping.get("portal_path") or "").strip()
        row = {
                "batch": batch,
                "originating-authority-charge-identifier": oid,
                "further-information-reference": source_value(source, "further-information-reference"),
                "supplementary-information": source_value(source, "supplementary-information"),
                "charge-address": source_value(source, "charge-address"),
                "charge-geographic-description": source_value(source, "charge-geographic-description"),
                "amazons3_path": amazon_path,
                "amazons3_path_cfd": (
                    (mapping.get("amazons3_confidence") or "0.00").strip() if route == "s3" else "0.00"
                ),
                "amazons3_path_mappingrule": (audit.get("rule_id") or "").strip() if route == "s3" else "",
                "amazons3_path_note": reason if route == "s3" else "",
                "portal_path": portal_path,
                "portal_path_cfd": (
                    (audit.get("decision_confidence") or "0.00").strip() if route == "portal" else "0.00"
                ),
                "portal_path_mappingrule": (audit.get("rule_id") or "").strip() if route == "portal" else "",
                "portal_path_note": reason if route == "portal" else "",
                "path_source": source_type,
                "path_found": status,
            }
        mapping_total_rows.append(row)
        mappingj_rows.append({field: row[field] for field in MAPPINGJ_FIELDS})

    ids = [row["originating-authority-charge-identifier"] for row in mapping_total_rows]
    if any(not oid for oid in ids) or len(ids) != len(set(ids)):
        raise MappingServiceError("Output contains blank or duplicate case identifiers")

    output_directory.mkdir(parents=True, exist_ok=False)
    prefix = f"{council}-{slug(batch, label='batch')}"
    mapping_total_path = output_directory / f"{prefix}-mapping.csv"
    mappingj_path = output_directory / f"{prefix}-mappingj.csv"
    exported_audit_path = output_directory / f"{prefix}-mapping-audit.csv"
    exported_spec_path = output_directory / "routing-spec.json"
    report_path = output_directory / "mapping-run-report.json"
    write_csv(mapping_total_path, mapping_total_rows, MAPPING_TOTAL_FIELDS)
    write_csv(mappingj_path, mappingj_rows, MAPPINGJ_FIELDS)
    shutil.copy2(audit_path, exported_audit_path)
    shutil.copy2(spec_path, exported_spec_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    counts = Counter(row["path_found"] for row in mapping_total_rows)
    source_counts = Counter(row["path_source"] or "none" for row in mapping_total_rows)
    report = {
        "status": "completed_staged",
        "production_published": False,
        "council": council,
        "batch": batch,
        "case_count": len(mapping_total_rows),
        "match_status_counts": {key: counts.get(key, 0) for key in ("found", "no_found", "no_match")},
        "source_type_counts": dict(sorted(source_counts.items())),
        "coverage": {
            "mapping_rows": len(mappings),
            "audit_rows": len(audits),
            "unique_case_ids": len(set(ids)),
            "exact": len(mappings) == len(audits) == len(set(ids)),
        },
        "validation": validation,
        "outputs": {
            "mapping_total": str(mapping_total_path),
            "mappingj": str(mappingj_path),
            "audit": str(exported_audit_path),
            "routing_spec": str(exported_spec_path),
            "run_report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def copy_artifact(path: Path, staging: Path, index: int) -> str:
    destination = staging / f"{index:02d}_{path.name}"
    shutil.copy2(path, destination)
    return destination.name


def run_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    council = slug(str(payload.get("council") or ""), label="council")
    batch = str(payload.get("batch") or "").strip()
    if not batch:
        raise MappingServiceError("batch is required and must be a registered file-matching batch")
    input_directory = safe_data_path(str(payload.get("input_directory") or ""), kind="directory")

    discovered_source, discovered_rules, discovered_breakdown = classify_input_files(input_directory)
    explicit_source = list_value(payload.get("source_path"))
    source_paths = [safe_data_path(value) for value in explicit_source] if explicit_source else discovered_source
    if len(source_paths) != 1:
        raise MappingServiceError(
            f"Expected exactly one source table; found {len(source_paths)}. Supply source_path explicitly."
        )
    explicit_rules = list_value(payload.get("capture_rules_paths"))
    rule_paths = [safe_data_path(value) for value in explicit_rules] if explicit_rules else select_latest_rule_versions(discovered_rules)
    if not rule_paths:
        raise MappingServiceError("No capture-rules DOCX, PDF, or TXT file was found")

    s3_paths = [safe_data_path(value) for value in list_value(payload.get("s3_inventory_paths"))]
    portal_paths = [safe_data_path(value) for value in list_value(payload.get("portal_evidence_paths"))]
    if not s3_paths and not portal_paths:
        conventional = DATA_ROOT / council / "file-matching" / f"{council}-s3-folder-index.csv"
        if conventional.is_file():
            s3_paths = [safe_data_path(str(conventional))]
    if not s3_paths and not portal_paths:
        raise MappingServiceError("At least one S3 inventory or Portal evidence path is required")

    stamp = utc_stamp()
    submission = JOBS_ROOT / "submissions" / f"{council}-{slug(batch, label='batch')}-{stamp}"
    submission.mkdir(parents=True, exist_ok=False)
    artifacts: list[dict[str, str]] = []
    index = 1
    for role, paths in (
        ("source_records", source_paths),
        ("capture_rules", rule_paths),
        ("source_breakdown", discovered_breakdown),
        ("s3_inventory", s3_paths),
        ("portal_evidence", portal_paths),
    ):
        for path in paths:
            artifacts.append({"url": copy_artifact(path, submission, index), "role": role})
            index += 1
    manifest_path = submission / "mapping-job.json"
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "council-mapping-job",
                "schema_version": 1,
                "council": council,
                "batch": batch,
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    command = [
        str(AUTONOMOUS_ENTRYPOINT),
        "--jobs-root",
        str(JOBS_ROOT),
        "start",
        "--url",
        manifest_path.as_uri(),
        "--council",
        council,
        "--batch",
        batch,
        "--requested-by",
        "n8n-file-path-mapping",
    ]
    result = subprocess.run(
        read_only_job_isolated_command(
            command,
            writable_root=JOBS_ROOT,
            codex_home=CODEX_RUNTIME_HOME,
            codex_auth=CODEX_AUTH,
        ),
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=1800,
        check=False,
    )
    try:
        snapshot = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MappingServiceError(
            f"Mapping engine returned unreadable output (exit {result.returncode}): {(result.stderr or result.stdout)[-2000:]}"
        ) from exc
    job_record = snapshot.get("job") if isinstance(snapshot.get("job"), dict) else snapshot
    if result.returncode != 0 or job_record.get("status") not in {
        "completed_staged",
        "completed_safe_with_exceptions",
    }:
        detail = (
            job_record.get("error")
            or snapshot.get("error")
            or snapshot.get("detail")
            or job_record.get("status")
            or "mapping failed"
        )
        raise MappingServiceError(f"Mapping engine stopped: {detail}")
    workspace = safe_data_path(str(job_record.get("workspace") or ""), kind="directory")
    output_directory = DATA_ROOT / council / "file-matching" / f"{slug(batch, label='batch')}_{stamp}"
    report = export_outputs(
        workspace=workspace,
        council=council,
        batch=batch,
        output_directory=output_directory,
    )
    report["job_id"] = job_record.get("job_id")
    report["input_directory"] = str(input_directory)
    report["selected_source"] = [str(path) for path in source_paths]
    report["selected_capture_rules"] = [str(path) for path in rule_paths]
    report["selected_s3_inventory"] = [str(path) for path in s3_paths]
    report["selected_portal_evidence"] = [str(path) for path in portal_paths]
    return report


class Handler(BaseHTTPRequestHandler):
    server_version = "n8n-file-path-mapping/1.0"

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)

    def respond(self, status: int, payload: dict[str, Any]) -> None:
        body = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def client_allowed(self) -> bool:
        address = ipaddress.ip_address(self.client_address[0])
        return address.is_loopback or address in ipaddress.ip_network("172.16.0.0/12")

    def do_GET(self) -> None:  # noqa: N802
        if not self.client_allowed():
            self.respond(HTTPStatus.FORBIDDEN, {"error": "forbidden_client"})
            return
        if self.path == "/healthz":
            self.respond(HTTPStatus.OK, {"status": "ok", "service": "file-path-mapping"})
            return
        self.respond(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self.client_allowed():
            self.respond(HTTPStatus.FORBIDDEN, {"error": "forbidden_client"})
            return
        if self.path != "/run":
            self.respond(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1024 * 1024:
                raise MappingServiceError("Request body must be JSON and no larger than 1 MiB")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise MappingServiceError("Request JSON must be an object")
            if not REQUEST_LOCK.acquire(blocking=False):
                self.respond(HTTPStatus.CONFLICT, {"error": "mapping_job_already_running"})
                return
            try:
                result = run_mapping(payload)
            finally:
                REQUEST_LOCK.release()
            self.respond(HTTPStatus.OK, result)
        except (MappingServiceError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
            self.respond(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": type(exc).__name__, "detail": str(exc), "production_published": False},
            )
        except Exception as exc:  # pragma: no cover - last-resort service boundary
            traceback.print_exc()
            self.respond(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": type(exc).__name__, "detail": str(exc), "production_published": False},
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5680)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
