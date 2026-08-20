from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

import openpyxl

from .engine import clean, write_csv_bytes
from .path_policy import require_unprotected_path
from .schemas import (
    ArtifactRole,
    DiscoveredArtifact,
    DiscoveryManifest,
    PreparationReport,
    RuleChunk,
)
from .storage import ArtifactStore


class PreparationError(RuntimeError):
    pass


def scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value).strip()


def _json_rows(payload: Any, *, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
        return payload
    if isinstance(payload, dict):
        candidates = [value for value in payload.values() if isinstance(value, list) and all(isinstance(row, dict) for row in value)]
        if len(candidates) == 1:
            return candidates[0]
    raise PreparationError(f"JSON evidence must be an array of objects or contain exactly one such array: {path}")


def load_tabular(path: Path, *, original_name: str) -> list[dict[str, str]]:
    path = require_unprotected_path(path, operation="read tabular evidence")
    suffix = Path(original_name).suffix.casefold()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            return [
                {str(key): scalar_text(value) for key, value in row.items() if key is not None}
                for row in csv.DictReader(handle)
            ]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return [
            {str(key): scalar_text(value) for key, value in row.items()}
            for row in _json_rows(payload, path=path)
        ]
    if suffix == ".jsonl":
        rows: list[dict[str, str]] = []
        with path.open(encoding="utf-8-sig") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise PreparationError(f"JSONL row {line_number} is not an object: {path}")
                rows.append({str(key): scalar_text(item) for key, item in value.items()})
        return rows
    if suffix == ".xlsx":
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        populated: list[tuple[str, list[dict[str, str]]]] = []
        try:
            for sheet in workbook.worksheets:
                values = sheet.iter_rows(values_only=True)
                header_values = next(values, None)
                if header_values is None:
                    continue
                headers = [scalar_text(value) for value in header_values]
                if not any(headers):
                    continue
                nonblank_headers = [header for header in headers if header]
                if len(nonblank_headers) != len(set(nonblank_headers)):
                    raise PreparationError(f"Duplicate headers in sheet {sheet.title!r}: {path}")
                rows = [
                    {
                        header: scalar_text(value)
                        for header, value in zip(headers, row, strict=False)
                        if header
                    }
                    for row in values
                    if any(value is not None and scalar_text(value) for value in row)
                ]
                if rows:
                    populated.append((sheet.title, rows))
        finally:
            workbook.close()
        if len(populated) != 1:
            names = [name for name, _ in populated]
            raise PreparationError(
                f"XLSX evidence requires exactly one populated sheet until an explicit selector exists; "
                f"found {names}: {path}"
            )
        return populated[0][1]
    if suffix == ".xls":
        raise PreparationError(f"Legacy .xls evidence is discovered but requires conversion to .xlsx or CSV: {path}")
    raise PreparationError(f"Unsupported tabular evidence type {suffix!r}: {path}")


def ordered_fields(rows: Iterable[dict[str, str]]) -> tuple[str, ...]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    return tuple(fields)


def combine_tabular(artifacts: Iterable[DiscoveredArtifact], *, add_role: bool) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    combined: list[dict[str, str]] = []
    for artifact in artifacts:
        rows = load_tabular(artifact.local_path, original_name=artifact.original_name)
        for row in rows:
            enriched = dict(row)
            enriched["_artifact_id"] = artifact.artifact_id
            if add_role:
                enriched["_evidence_role"] = artifact.role.value
            combined.append(enriched)
    fields = ordered_fields(combined)
    return combined, fields


WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    paragraphs: list[str] = []
    for paragraph in root.iter(WORD_NS + "p"):
        text = "".join(node.text or "" for node in paragraph.iter(WORD_NS + "t")).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _pdf_text(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def extract_capture_text(artifact: DiscoveredArtifact) -> str:
    suffix = Path(artifact.original_name).suffix.casefold()
    if suffix == ".txt":
        return artifact.local_path.read_text(encoding="utf-8-sig", errors="replace")
    if suffix == ".docx":
        return _docx_text(artifact.local_path)
    if suffix == ".pdf":
        return _pdf_text(artifact.local_path)
    raise PreparationError(f"Unsupported capture-rules type {suffix!r}: {artifact.original_name}")


def _clean_rule_text(value: str) -> str:
    value = value.replace("\u00a0", " ")
    value = re.sub(r"[ \t\r]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def chunk_rules(text: str, *, max_chars: int = 3500) -> tuple[RuleChunk, ...]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for paragraph in paragraphs:
        if current and current_size + len(paragraph) + 2 > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_size = 0
        if len(paragraph) > max_chars:
            for offset in range(0, len(paragraph), max_chars):
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_size = 0
                chunks.append(paragraph[offset : offset + max_chars])
            continue
        current.append(paragraph)
        current_size += len(paragraph) + 2
    if current:
        chunks.append("\n\n".join(current))
    return tuple(
        RuleChunk(
            location=f"chunk:{index}",
            text=chunk,
            excerpt_sha256=hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
        )
        for index, chunk in enumerate(chunks, start=1)
    )


def _registry_state(council: str, batch: str) -> tuple[bool, bool]:
    try:
        from registry import PIPELINES
    except ImportError as exc:  # pragma: no cover - wrapper normally places amazons3-mapping on sys.path
        raise PreparationError("Could not import amazons3-mapping registry") from exc
    pipeline = PIPELINES.get(council)
    if pipeline is None:
        return False, False
    return True, batch in {builder.name for builder in pipeline.builders}


def _ensure_nonempty_rows(rows: list[dict[str, str]], *, label: str) -> None:
    if not rows:
        raise PreparationError(f"Prepared {label} contains no rows")


def prepare_evidence(
    *,
    job_id: str,
    council: str,
    batch: str,
    discovery: DiscoveryManifest,
    artifacts: ArtifactStore,
) -> PreparationReport:
    if not discovery.evidence_complete:
        missing = ", ".join(role.value for role in discovery.missing_roles)
        raise PreparationError(f"Evidence discovery is incomplete; missing roles: {missing}")
    source_artifacts = [item for item in discovery.artifacts if item.role == ArtifactRole.SOURCE_RECORDS]
    inventory_artifacts = [
        item
        for item in discovery.artifacts
        if item.role in {ArtifactRole.S3_INVENTORY, ArtifactRole.PORTAL_EVIDENCE}
    ]
    capture_artifacts = [item for item in discovery.artifacts if item.role == ArtifactRole.CAPTURE_RULES]

    source_rows, source_fields = combine_tabular(source_artifacts, add_role=False)
    inventory_rows, inventory_fields = combine_tabular(inventory_artifacts, add_role=True)
    _ensure_nonempty_rows(source_rows, label="source records")
    _ensure_nonempty_rows(inventory_rows, label="candidate inventory")

    capture_sections: list[str] = []
    for artifact in capture_artifacts:
        extracted = _clean_rule_text(extract_capture_text(artifact))
        if not extracted:
            raise PreparationError(f"Capture-rules extraction produced no text: {artifact.original_name}")
        capture_sections.append(f"SOURCE: {artifact.original_name}\nARTIFACT: {artifact.artifact_id}\n\n{extracted}")
    capture_text = "\n\n===== NEXT CAPTURE-RULE DOCUMENT =====\n\n".join(capture_sections)
    chunks = chunk_rules(capture_text)
    if not chunks:
        raise PreparationError("Capture-rules extraction produced no citable chunks")

    source_path = artifacts.write_immutable(
        "prepared/source-records.csv",
        write_csv_bytes(source_rows, source_fields),
    )
    inventory_path = artifacts.write_immutable(
        "prepared/candidate-inventory.csv",
        write_csv_bytes(inventory_rows, inventory_fields),
    )
    capture_path = artifacts.write_immutable(
        "prepared/capture-rules.txt",
        (capture_text + "\n").encode("utf-8"),
    )
    council_known, batch_known = _registry_state(council, batch)
    warnings: list[str] = []
    if not council_known:
        warnings.append(f"Council {council!r} is not registered; compilation may proceed but mapping must not run")
    elif not batch_known:
        warnings.append(
            f"Batch {batch!r} is not registered for council {council!r}; compilation may proceed but mapping must not run"
        )
    return PreparationReport(
        job_id=job_id,
        council=council,
        batch=batch,
        registry_council_known=council_known,
        registry_batch_known=batch_known,
        source_path=source_path,
        inventory_path=inventory_path,
        capture_rules_path=capture_path,
        source_rows=len(source_rows),
        inventory_rows=len(inventory_rows),
        source_fields=source_fields,
        inventory_fields=inventory_fields,
        inventory_roles=tuple(sorted({item.role for item in inventory_artifacts}, key=lambda role: role.value)),
        capture_rule_chunks=chunks,
        warnings=tuple(warnings),
    )


# Bounds the scan used to find distinct shapes; the sample itself stays small.
SAMPLE_SCAN_LIMIT = 20000


def value_shape(value: str) -> str:
    """Collapse a value to its shape, so 92_0007P and 94_1325P look alike."""
    shape = re.sub(r"\d+", "9", value.strip())
    shape = re.sub(r"[A-Za-z]+", "A", shape)
    return shape[:40]


# A column with more distinct values than this identifies records rather than
# grouping them, and listing its values would be a data dump, not a summary.
MAX_DISTINCT_FOR_DISTRIBUTION = 25


def value_distributions(
    path: Path, *, max_distinct: int = MAX_DISTINCT_FOR_DISTRIBUTION
) -> dict[str, dict[str, int]]:
    """Every value of each low-cardinality column, with its row count.

    Sample rows show what records look like; they do not show what values a
    column can take. A Sheffield spec written from eight sampled rows routed
    "Microfiche" and "Aperture", and so dropped the 14,340 U-Drive rows it never
    saw and missed that the value is spelled "Aperture cards". Both are visible
    the moment the column's values are counted, and neither is inferable from a
    sample however it is drawn.
    """
    path = require_unprotected_path(path, operation="read compiler distribution")
    counts: dict[str, dict[str, int]] = {}
    dropped: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index >= SAMPLE_SCAN_LIMIT:
                break
            for column, value in row.items():
                if column is None or column in dropped:
                    continue
                bucket = counts.setdefault(str(column), {})
                cleaned = clean(value)[:120]
                bucket[cleaned] = bucket.get(cleaned, 0) + 1
                if len(bucket) > max_distinct:
                    dropped.add(str(column))
                    counts.pop(str(column), None)
    return {
        column: dict(sorted(values.items(), key=lambda item: -item[1]))
        for column, values in counts.items()
    }


def sample_rows(path: Path, *, limit: int = 8) -> list[dict[str, str]]:
    """Show the compiler every naming convention present, not the first few rows.

    Evidence files are usually ordered, so the head of one is not a sample of it.
    Sheffield's scan index holds two populations -- vendor files named
    SHE_1988_88-392-P and legacy U-Drive files named 92_0007P -- and the first
    rows show only one. A spec written from that head templates one convention
    and silently fails to join the other, which measured as 45.7% reachable
    where the delivered mapping reaches 93.4%.

    So rows are grouped by the shape of their most various column, and the
    sample is drawn across those groups largest-first.
    """
    path = require_unprotected_path(path, operation="read compiler sample")
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        scanned: list[dict[str, str]] = []
        for row in csv.DictReader(handle):
            scanned.append(
                {str(key): clean(value)[:500] for key, value in row.items() if key is not None}
            )
            if len(scanned) >= SAMPLE_SCAN_LIMIT:
                break
    if len(scanned) <= limit:
        return scanned

    # The column whose values take the most shapes is the one that identifies a
    # record; grouping on anything else would not separate the conventions.
    columns = list(scanned[0])
    key_column = max(
        columns,
        key=lambda column: len({value_shape(row.get(column, "")) for row in scanned}),
        default=columns[0] if columns else "",
    )
    groups: dict[str, list[dict[str, str]]] = {}
    for row in scanned:
        groups.setdefault(value_shape(row.get(key_column, "")), []).append(row)

    ordered = sorted(groups.values(), key=len, reverse=True)
    sample: list[dict[str, str]] = []
    index = 0
    while len(sample) < limit and any(len(group) > index for group in ordered):
        for group in ordered:
            if len(sample) >= limit:
                break
            if len(group) > index:
                sample.append(group[index])
        index += 1
    return sample
