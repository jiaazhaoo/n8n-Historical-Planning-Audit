from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence
from urllib.parse import unquote, urlparse

from PIL import Image, ImageOps

from .compiler import CodexOAuthCompiler, strict_output_schema
from .engine import read_csv, write_csv_bytes
from .path_policy import file_browser_isolated_command, require_unprotected_path
from .preparation import load_tabular
from .qa import QA_FIELDS, select_stratified_rows
from .schemas import (
    AcquisitionBatchReport,
    AcquisitionStatus,
    ContentCaseResult,
    ContentExpectation,
    ContentObservation,
    ContentVerificationReport,
    QaVerdict,
    utc_now,
)
from .storage import ArtifactStore, sha256_file


class ContentQaError(RuntimeError):
    pass


class ObservationExtractor(Protocol):
    def extract(
        self,
        *,
        case_token: str,
        images: tuple[Path, ...],
        artifacts: ArtifactStore,
    ) -> tuple[ContentObservation, Path]: ...


class DocumentAcquirer(Protocol):
    documents_root: Path | None

    def acquire(
        self,
        expectations: tuple[ContentExpectation, ...],
        artifacts: ArtifactStore,
    ) -> AcquisitionBatchReport: ...


@dataclass(frozen=True)
class IdentityFieldProfile:
    reference_fields: tuple[str, ...] = ()
    address_fields: tuple[str, ...] = ()
    description_fields: tuple[str, ...] = ()
    date_fields: tuple[str, ...] = ()
    document_type_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContentQaConfig:
    council: str
    batch: str
    source_path: Path
    source_original_name: str
    source_id_field: str
    audit_path: Path
    documents_root: Path | None = None
    scope: str = "stratified_sample"
    sample_size: int = 5
    include_ids: tuple[str, ...] = ()
    max_images_per_case: int = 12
    max_image_dimension: int = 2400
    max_image_pixels: int = 40_000_000
    field_profile: IdentityFieldProfile = IdentityFieldProfile()

    def validate(self) -> None:
        if not self.council.strip() or not self.batch.strip():
            raise ContentQaError("council and batch are required")
        if self.scope not in {"targeted", "stratified_sample", "full_population"}:
            raise ContentQaError(f"Unsupported content QA scope: {self.scope}")
        if not 1 <= self.sample_size <= 40:
            raise ContentQaError("sample_size must be between 1 and 40")
        if not 1 <= self.max_images_per_case <= 24:
            raise ContentQaError("max_images_per_case must be between 1 and 24")
        if self.scope == "targeted" and not self.include_ids:
            raise ContentQaError("targeted content QA requires at least one include_id")
        for label, path in (("source", self.source_path), ("audit", self.audit_path)):
            path = require_unprotected_path(path, operation=f"read content QA {label}")
            if not path.is_file():
                raise ContentQaError(f"{label} is not a file: {path}")
        if self.documents_root is not None:
            documents_root = require_unprotected_path(
                self.documents_root,
                operation="read content QA documents",
            )
            if not documents_root.is_dir():
                raise ContentQaError(f"documents_root is not a directory: {documents_root}")


def safe_token(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._")
    return token or "blank"


def dedupe(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return tuple(result)


def natural_key(path: Path) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name))


def evenly_spaced(items: list[Path], limit: int) -> list[Path]:
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]
    indexes = []
    for position in range(limit):
        index = round(position * (len(items) - 1) / (limit - 1))
        if index not in indexes:
            indexes.append(index)
    return [items[index] for index in indexes]


def canonical_audit_row(row: dict[str, str]) -> dict[str, str]:
    result = dict(row)
    s3_path = str(row.get("amazons3_path") or "").strip()
    portal_path = str(row.get("portal_path") or "").strip()
    result["route"] = str(row.get("route") or ("s3" if s3_path else "portal" if portal_path else "none"))
    explicit_confidence = str(row.get("decision_confidence") or "").strip()
    if explicit_confidence:
        decision_confidence = explicit_confidence
    elif portal_path:
        portal_confidence = str(row.get("portal_confidence") or "").strip()
        status = str(row.get("match_status") or row.get("status") or "").casefold()
        if portal_confidence:
            decision_confidence = portal_confidence
        elif status.startswith("accepted_") and (
            "content_verified" in status or "content_match" in status
        ):
            decision_confidence = "0.74"
        elif status.startswith("accepted_"):
            decision_confidence = "0.66"
        else:
            decision_confidence = "0.00"
    else:
        decision_confidence = str(row.get("amazons3_confidence") or "0.00")
    result["decision_confidence"] = decision_confidence
    result["candidate_count"] = str(
        row.get("candidate_count") or row.get("candidate_file_count") or ("1" if s3_path or portal_path else "0")
    )
    result["match_basis"] = str(row.get("match_basis") or row.get("match_method") or "")
    return result


def _field_names(
    source_fields: Iterable[str],
    *,
    explicit: tuple[str, ...],
    exact: tuple[str, ...],
    contains: tuple[str, ...],
    excluded_contains: tuple[str, ...] = (),
) -> tuple[str, ...]:
    available = tuple(source_fields)
    if explicit:
        missing = [field for field in explicit if field not in available]
        if missing:
            raise ContentQaError(f"Configured source fields are absent: {missing}")
        return explicit
    selected: list[str] = []
    exact_set = {value.casefold() for value in exact}
    for field in available:
        lowered = field.casefold().replace("_", "-")
        if lowered in exact_set or (
            any(fragment in lowered for fragment in contains)
            and not any(fragment in lowered for fragment in excluded_contains)
        ):
            selected.append(field)
    return tuple(selected)


def identity_fields(
    source_fields: Iterable[str], profile: IdentityFieldProfile
) -> dict[str, tuple[str, ...]]:
    return {
        "address": _field_names(
            source_fields,
            explicit=profile.address_fields,
            exact=("charge-geographic-description", "charge-address", "site-address", "property-address"),
            contains=("site-address", "property-address", "geographic-description"),
            excluded_contains=("applicant", "agent", "council", "office", "correspondence"),
        ),
        "description": _field_names(
            source_fields,
            explicit=profile.description_fields,
            exact=("supplementary-information", "land-works-particulars", "proposal", "description"),
            contains=("proposal", "development-description", "works-particulars"),
        ),
        "date": _field_names(
            source_fields,
            explicit=profile.date_fields,
            exact=("registration-date", "decision-date", "application-date", "charge-creation-date"),
            contains=("registration-date", "decision-date", "application-date"),
        ),
        "document_type": _field_names(
            source_fields,
            explicit=profile.document_type_fields,
            exact=("instrument", "charge-sub-category", "charge-type", "document-type"),
            contains=("document-type",),
        ),
    }


def _row_values(row: dict[str, str], fields: Iterable[str]) -> tuple[str, ...]:
    return dedupe(str(row.get(field) or "") for field in fields)


def build_expectation(
    *,
    council: str,
    batch: str,
    audit: dict[str, str],
    source: dict[str, str],
    fields: dict[str, tuple[str, ...]],
    configured_reference_fields: tuple[str, ...],
) -> ContentExpectation:
    route = str(audit.get("route") or "none")
    if route not in {"s3", "portal"}:
        route = "none"
    mapping_path = str(
        audit.get("amazons3_path") if route == "s3" else audit.get("portal_path") if route == "portal" else ""
    ).strip()
    match_basis = str(audit.get("match_basis") or audit.get("authoritative_key") or "").strip()
    reference_fields: list[str] = list(configured_reference_fields)
    if not reference_fields and match_basis in source:
        reference_fields.append(match_basis)
    authoritative_key = str(audit.get("authoritative_key") or "").strip()
    if not reference_fields and authoritative_key in source:
        reference_fields.append(authoritative_key)
    reference_values = list(_row_values(source, reference_fields))
    authoritative_value = str(audit.get("authoritative_value") or "").strip()
    if not reference_values and authoritative_value:
        reference_values.append(authoritative_value)
    try:
        mapping_confidence = float(str(audit.get("decision_confidence") or "0.00").strip())
    except ValueError as exc:
        raise ContentQaError(
            f"Audit row {audit.get('oachargeid')!r} has invalid decision confidence"
        ) from exc
    return ContentExpectation(
        council=council,
        batch=batch,
        oachargeid=str(audit.get("oachargeid") or "").strip(),
        route=route,
        mapping_path=mapping_path,
        mapping_confidence=mapping_confidence,
        mapping_status=str(audit.get("match_status") or audit.get("status") or "").strip(),
        match_basis=match_basis,
        reference_fields=tuple(reference_fields),
        reference_values=dedupe(reference_values),
        address_fields=fields["address"],
        address_values=_row_values(source, fields["address"]),
        description_fields=fields["description"],
        description_values=_row_values(source, fields["description"]),
        date_fields=fields["date"],
        date_values=_row_values(source, fields["date"]),
        document_type_fields=fields["document_type"],
        document_type_values=_row_values(source, fields["document_type"]),
    )


def select_audit_rows(
    rows: list[dict[str, str]], config: ContentQaConfig
) -> list[dict[str, str]]:
    canonical = [canonical_audit_row(row) for row in rows]
    by_id = {row.get("oachargeid", ""): row for row in canonical}
    missing = [value for value in config.include_ids if value not in by_id]
    if missing:
        raise ContentQaError(f"Included oachargeid values are absent from audit: {missing}")
    if config.scope == "targeted":
        return [by_id[value] for value in config.include_ids]
    if config.scope == "full_population":
        if len(canonical) > 40:
            raise ContentQaError(
                "Full-population OAuth content QA is capped at 40 cases per run; select a bounded population"
            )
        return canonical
    return select_stratified_rows(
        canonical,
        seed=f"{config.council}:{config.batch}:{sha256_file(config.audit_path)}",
        sample_size=config.sample_size,
        include_ids=config.include_ids,
    )


def _safe_inside(root: Path, candidate: Path) -> Path | None:
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if resolved == resolved_root or resolved_root in resolved.parents:
        return resolved
    return None


def find_document_directory(expectation: ContentExpectation, documents_root: Path) -> Path | None:
    root = documents_root.resolve()
    if not expectation.accepted:
        return None
    parsed = urlparse(expectation.mapping_path)
    if expectation.route == "s3":
        leaf = Path(unquote(parsed.path).rstrip("/")).name
        if not leaf:
            return None
        names = [Path(leaf).stem, leaf] if Path(leaf).suffix.casefold() == ".pdf" else [leaf]
    else:
        names = [safe_token(expectation.oachargeid)]
    for name in names:
        candidate = _safe_inside(root, root / name)
        if candidate and candidate.is_dir():
            return candidate
        if candidate and candidate.is_file():
            return candidate.parent
    return None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def discover_images(directory: Path, *, max_images: int, max_files_scanned: int = 500) -> list[Path]:
    files: list[Path] = []
    for path in directory.rglob("*"):
        if path.is_file():
            files.append(path)
            if len(files) > max_files_scanned:
                raise ContentQaError(
                    f"Case directory exceeds the {max_files_scanned}-file inspection limit: {directory}"
                )
    images = sorted(
        (path for path in files if path.suffix.casefold() in IMAGE_SUFFIXES),
        key=natural_key,
    )
    return evenly_spaced(images, max_images)


def pdf_page_count(path: Path) -> int:
    if shutil.which("pdfinfo") is None:
        raise ContentQaError("pdfinfo is required to inspect PDF-only cases")
    result = subprocess.run(
        ["pdfinfo", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ContentQaError(f"pdfinfo failed for {path}: {(result.stderr or '').strip()}")
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout or "", flags=re.MULTILINE)
    if not match or int(match.group(1)) < 1:
        raise ContentQaError(f"Could not determine PDF page count: {path}")
    return int(match.group(1))


def selected_pdf_pages(directory: Path, limit: int) -> list[tuple[Path, int]]:
    pdfs = sorted(directory.rglob("*.pdf"), key=natural_key)
    pdfs.extend(sorted(directory.rglob("*.PDF"), key=natural_key))
    unique_pdfs = list(dict.fromkeys(path.resolve() for path in pdfs if path.is_file()))
    pages = [(pdf, page) for pdf in unique_pdfs for page in range(1, pdf_page_count(pdf) + 1)]
    if len(pages) <= limit:
        return pages
    if limit == 1:
        return [pages[0]]
    indexes = []
    for position in range(limit):
        index = round(position * (len(pages) - 1) / (limit - 1))
        if index not in indexes:
            indexes.append(index)
    return [pages[index] for index in indexes]


def render_pdf_page(pdf: Path, page: int, temporary_root: Path, index: int, max_dimension: int) -> Path:
    if shutil.which("pdftoppm") is None:
        raise ContentQaError("pdftoppm is required to render PDF-only cases")
    prefix = temporary_root / f"render-{index:04d}"
    result = subprocess.run(
        [
            "pdftoppm",
            "-f",
            str(page),
            "-l",
            str(page),
            "-singlefile",
            "-jpeg",
            "-scale-to",
            str(max_dimension),
            str(pdf),
            str(prefix),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = prefix.with_suffix(".jpg")
    if result.returncode != 0 or not output.is_file():
        raise ContentQaError(
            f"pdftoppm failed for {pdf} page {page}: {(result.stderr or '').strip()}"
        )
    return output


def _neutral_image(
    source: Path,
    *,
    relative: str,
    artifacts: ArtifactStore,
    max_dimension: int,
    max_pixels: int,
) -> Path:
    # A neutral image is a deterministic function of its source and the size
    # limits, and re-deriving one costs a 50 MP decode plus a LANCZOS resize.
    # Within a round's artifact store those inputs do not change, so an existing
    # rendering is the same rendering.
    existing = artifacts.resolve(relative)
    if existing.is_file() and existing.stat().st_size > 0:
        return existing
    with Image.open(source) as opened:
        if opened.width * opened.height > max_pixels:
            raise ContentQaError(f"Image exceeds the {max_pixels}-pixel safety limit: {source}")
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        from io import BytesIO

        output = BytesIO()
        image.save(output, format="JPEG", quality=88, optimize=True)
    return artifacts.write_immutable(relative, output.getvalue())


def prepare_neutral_images(
    *,
    case_token: str,
    directory: Path,
    artifacts: ArtifactStore,
    config: ContentQaConfig,
) -> tuple[Path, ...]:
    sources = discover_images(directory, max_images=config.max_images_per_case)
    rendered_temporary: tempfile.TemporaryDirectory[str] | None = None
    if not sources:
        pages = selected_pdf_pages(directory, config.max_images_per_case)
        if pages:
            rendered_temporary = tempfile.TemporaryDirectory(prefix="mapping-content-pages-")
            temporary_root = Path(rendered_temporary.name)
            sources = [
                render_pdf_page(pdf, page, temporary_root, index, config.max_image_dimension)
                for index, (pdf, page) in enumerate(pages, start=1)
            ]
    prepared: list[Path] = []
    try:
        for index, source in enumerate(sources, start=1):
            prepared.append(
                _neutral_image(
                    source,
                    relative=f"qa/cases/{case_token}/images/page-{index:04d}.jpg",
                    artifacts=artifacts,
                    max_dimension=config.max_image_dimension,
                    max_pixels=config.max_image_pixels,
                )
            )
    finally:
        if rendered_temporary is not None:
            rendered_temporary.cleanup()
    return tuple(prepared)


def observation_prompt(image_names: Iterable[str]) -> str:
    names = ", ".join(image_names)
    return f"""Perform blind factual extraction from the attached council planning-record images.

The attachments have neutral names ({names}). Do not infer any value from filenames, directory names,
the repository, or prior knowledge. Do not decide whether the record matches another record. Extract only
what is visibly supported by the pages.

For each distinct planning identity that is actually visible, return its application/reference number,
application-site/property address, proposal/description, relevant application/decision/registration date,
and document type. Use null for a field that is not visible. Property address means the application site;
exclude applicant, agent, council-office, correspondence, and neighbouring-consultee addresses. Preserve
spelling and punctuation as seen. Cite the neutral image name and a short visible phrase in evidence.

Set readable=false only when none of the attachments contains enough legible record content to extract an
identity. List every reviewed attachment, do not invent missing fields, and return only the schema object.
"""


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class CodexOAuthVisionExtractor:
    def __init__(
        self,
        *,
        repository_root: Path,
        command_runner: CommandRunner = subprocess.run,
        command_isolator: Callable[[list[str]], list[str]] = file_browser_isolated_command,
        timeout_seconds: int = 900,
    ):
        self.repository_root = require_unprotected_path(
            repository_root,
            operation="use content QA repository root",
        )
        self.command_runner = command_runner
        self.command_isolator = command_isolator
        self.timeout_seconds = timeout_seconds

    def _require_chatgpt_login(self) -> None:
        result = self.command_runner(
            ["codex", "login", "status"],
            env=CodexOAuthCompiler._oauth_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        if result.returncode != 0 or "Logged in using ChatGPT" not in (result.stdout or ""):
            raise ContentQaError(
                "Content extraction requires `codex login` with ChatGPT; API-key fallback is disabled"
            )

    def extract(
        self,
        *,
        case_token: str,
        images: tuple[Path, ...],
        artifacts: ArtifactStore,
    ) -> tuple[ContentObservation, Path]:
        if not images:
            raise ContentQaError("Vision extraction requires at least one image")
        images = tuple(
            require_unprotected_path(image, operation="read content QA image")
            for image in images
        )
        self._require_chatgpt_login()
        case_root = f"qa/cases/{case_token}"
        raw_output = artifacts.resolve(f"{case_root}/observation.raw.json")
        if raw_output.exists():
            return ContentObservation.model_validate_json(raw_output.read_text(encoding="utf-8")), raw_output
        schema_path = artifacts.write_immutable_json(
            "qa/content-observation.schema.json",
            strict_output_schema(ContentObservation.model_json_schema(mode="validation")),
        )
        attempt = artifacts.resolve(f"{case_root}/observation.attempt.json")
        attempt.parent.mkdir(parents=True, exist_ok=True)
        if attempt.exists():
            attempt.unlink()
        command = [
            "codex",
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--color",
            "never",
        ]
        for image in images:
            command.extend(("--image", str(image)))
        command.extend(
            (
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(attempt),
                "--cd",
                str(self.repository_root),
                "-",
            )
        )
        result = self.command_runner(
            self.command_isolator(command),
            input=observation_prompt(path.name for path in images),
            env=CodexOAuthCompiler._oauth_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_seconds,
            check=False,
        )
        log = (
            f"returncode={result.returncode}\n\nSTDOUT\n{(result.stdout or '')[-100000:]}"
            f"\n\nSTDERR\n{(result.stderr or '')[-100000:]}\n"
        )
        log_path = artifacts.write_mutable(f"{case_root}/codex-vision.log", log.encode("utf-8"))
        if result.returncode != 0 or not attempt.is_file():
            if attempt.exists():
                artifacts.write_mutable(f"{case_root}/observation.failed.json", attempt.read_bytes())
                attempt.unlink()
            raise ContentQaError(f"Codex vision extraction failed; inspect {log_path}")
        try:
            observation = ContentObservation.model_validate_json(attempt.read_text(encoding="utf-8"))
        except Exception as exc:
            failed = artifacts.write_mutable(f"{case_root}/observation.failed.json", attempt.read_bytes())
            attempt.unlink()
            raise ContentQaError(f"Invalid structured content observation; inspect {failed}: {exc}") from exc
        allowed_names = {path.name for path in images}
        cited_names = {name for identity in observation.identities for name in identity.image_names}
        unknown_names = sorted(cited_names - allowed_names)
        if unknown_names:
            raise ContentQaError(f"Observation cites unattached image names: {unknown_names}")
        raw_output = artifacts.write_immutable(f"{case_root}/observation.raw.json", attempt.read_bytes())
        attempt.unlink()
        return observation, raw_output


def reference_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def canonical_reference_key(value: str, council: str) -> str:
    if council.casefold() == "testvalley":
        match = re.match(r"^\s*(TVS|TVN)[.\s_-]*0*(\d+)(.*)$", value, flags=re.I)
        if match:
            main_number = str(int(match.group(2)))
            suffix = reference_key(match.group(3))
            return f"{match.group(1).casefold()}{main_number}{suffix}"
    return reference_key(value)


def reference_search_patterns(value: str) -> tuple[str, ...]:
    """How a reference is written on the document it belongs to.

    A source row joins the parts one way and the document another -- Exeter
    records EXE_1979_79-854-03 while its scan says 79/854 -- so what has to be
    reconstructed is the two-digit year and the number that follows it, with any
    of the usual separators and any leading zeros between the two spellings.

    One pattern, not every adjacent pair of number groups. Both rules scored
    12/12 on the sample's own references and 0/132 against the other cases', so
    the measurement could not separate them; the narrower rule wins on the
    asymmetry instead. A case that cannot be verified only fails to count as
    verified, while a case verified against a coincidental number match is a
    silent error, and each extra pattern is only more surface for one.
    """
    groups = re.findall(r"\d+", value or "")
    if not groups:
        return ()
    index: int | None = None
    year = next((group for group in groups if len(group) == 4), None)
    if year is not None:
        short = year[2:]
        after = groups.index(year)
        index = next(
            (position for position, group in enumerate(groups) if group == short and position > after),
            None,
        )
    if index is None:
        index = next((position for position, group in enumerate(groups) if len(group) == 2), None)
    if index is None or index + 1 >= len(groups):
        return ()
    return (rf"\b{re.escape(groups[index])}\s*[/\-.]\s*0*{int(groups[index + 1])}\b",)


def document_carries_reference(text: str, expectation: "ContentExpectation") -> bool:
    for value in expectation.reference_values:
        for pattern in reference_search_patterns(value):
            if re.search(pattern, text):
                return True
    return False


ADDRESS_STOPWORDS = {
    "the", "and", "of", "at", "land", "site", "property", "sheffield", "road", "street",
    "avenue", "lane", "drive", "close", "way", "uk",
}
DESCRIPTION_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "onto", "of", "to", "at", "a", "an",
    "proposed", "proposal", "application", "planning",
}


def word_tokens(value: str, *, stopwords: set[str]) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 1 and token not in stopwords
    }


def house_numbers(value: str) -> set[str]:
    return set(re.findall(r"\b\d+[a-z]?\b", value.casefold()))


def address_matches(expected: str, observed: str) -> bool:
    expected_numbers = house_numbers(expected)
    observed_numbers = house_numbers(observed)
    if expected_numbers and observed_numbers and expected_numbers.isdisjoint(observed_numbers):
        return False
    expected_tokens = word_tokens(expected, stopwords=ADDRESS_STOPWORDS)
    observed_tokens = word_tokens(observed, stopwords=ADDRESS_STOPWORDS)
    if not expected_tokens or not observed_tokens:
        return False
    overlap = expected_tokens & observed_tokens
    return len(overlap) >= 2 and len(overlap) / min(len(expected_tokens), len(observed_tokens)) >= 0.6


def description_matches(expected: str, observed: str) -> bool:
    expected_tokens = word_tokens(expected, stopwords=DESCRIPTION_STOPWORDS)
    observed_tokens = word_tokens(observed, stopwords=DESCRIPTION_STOPWORDS)
    if not expected_tokens or not observed_tokens:
        return False
    overlap = expected_tokens & observed_tokens
    return len(overlap) >= 2 and len(overlap) / min(len(expected_tokens), len(observed_tokens)) >= 0.5


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def date_keys(value: str) -> set[str]:
    keys: set[str] = set()
    digits = "".join(character for character in value if character.isdigit())
    if len(digits) == 8:
        if digits[:4].startswith(("19", "20")):
            keys.add(digits)
        elif digits[-4:].startswith(("19", "20")):
            keys.add(f"{digits[-4:]}{digits[2:4]}{digits[:2]}")
    for match in re.finditer(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b", value):
        day, month, year = (int(item) for item in match.groups())
        if year < 100:
            year += 1900 if year >= 50 else 2000
        if 1 <= day <= 31 and 1 <= month <= 12:
            keys.add(f"{year:04d}{month:02d}{day:02d}")
    month_pattern = "|".join(sorted(MONTHS, key=len, reverse=True))
    for match in re.finditer(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_pattern})\s+((?:19|20)\d{{2}})\b",
        value.casefold(),
    ):
        day = int(match.group(1))
        month = MONTHS[match.group(2)]
        year = int(match.group(3))
        if 1 <= day <= 31:
            keys.add(f"{year:04d}{month:02d}{day:02d}")
    keys.update(re.findall(r"(?:19|20)\d{2}", value))
    return keys


def _any_match(expected: Iterable[str], observed: str | None, comparator: Callable[[str, str], bool]) -> bool:
    return bool(observed) and any(comparator(value, observed or "") for value in expected)


def judge_observation(
    expectation: ContentExpectation,
    observation: ContentObservation,
) -> tuple[QaVerdict, float, bool, dict[str, bool], int | None, str]:
    if not expectation.accepted:
        return (
            QaVerdict.NOT_APPLICABLE,
            0.0,
            False,
            {"reference": False, "address": False, "description": False, "date": False},
            None,
            "No accepted path; content identity review is not applicable",
        )
    if not observation.readable or not observation.identities:
        return (
            QaVerdict.UNREADABLE,
            0.0,
            False,
            {"reference": False, "address": False, "description": False, "date": False},
            None,
            "No legible planning identity was extracted from the selected pages",
        )
    expected_refs = {
        canonical_reference_key(value, expectation.council)
        for value in expectation.reference_values
        if canonical_reference_key(value, expectation.council)
    }
    identity_signals: list[dict[str, bool]] = []
    for identity in observation.identities:
        signals = {
            "reference": bool(
                identity.reference
                and canonical_reference_key(identity.reference, expectation.council) in expected_refs
            ),
            "address": _any_match(expectation.address_values, identity.property_address, address_matches),
            "description": _any_match(
                expectation.description_values, identity.description, description_matches
            ),
            "date": _any_match(
                expectation.date_values,
                identity.relevant_date,
                lambda left, right: bool(date_keys(left) & date_keys(right)),
            ),
        }
        identity_signals.append(signals)
    for index, signals in enumerate(identity_signals):
        if signals["reference"] and any(signals[key] for key in ("address", "description", "date")):
            return (
                QaVerdict.VERIFIED_SAME,
                0.90,
                True,
                signals,
                index,
                "Exact observed reference plus at least one independent source fact identifies the same case",
            )
    observed_refs = [
        canonical_reference_key(identity.reference or "", expectation.council)
        for identity in observation.identities
        if canonical_reference_key(identity.reference or "", expectation.council)
    ]
    observed_addresses = [identity.property_address for identity in observation.identities if identity.property_address]
    explicit_reference_conflict = bool(expected_refs and observed_refs and expected_refs.isdisjoint(observed_refs))
    explicit_address_conflict = bool(
        expectation.address_values
        and observed_addresses
        and not any(
            address_matches(expected, observed)
            for expected in expectation.address_values
            for observed in observed_addresses
        )
    )
    combined = {
        key: any(signals[key] for signals in identity_signals)
        for key in ("reference", "address", "description", "date")
    }
    if explicit_reference_conflict and explicit_address_conflict:
        return (
            QaVerdict.VERIFIED_WRONG,
            0.0,
            False,
            combined,
            None,
            "Observed reference and application-site address both conflict with the source record",
        )
    # Where the source records nothing but a reference, there is no second fact
    # to corroborate with, and demanding one makes the batch permanently
    # unverifiable. On Exeter's Microfiche 1977-1985 sample the reference alone
    # separates the cases cleanly: every one of the twelve scans carried its own
    # reference, and none carried any of the other eleven (0/132).
    if combined["reference"] and not (
        expectation.address_values or expectation.description_values or expectation.date_values
    ):
        return (
            QaVerdict.VERIFIED_REFERENCE_ONLY,
            0.80,
            True,
            combined,
            next(
                (index for index, signals in enumerate(identity_signals) if signals["reference"]),
                None,
            ),
            "The document carries the expected reference; the source records no address, "
            "description or date to corroborate it against",
        )
    if len(set(observed_refs)) > 1 and not combined["reference"]:
        return (
            QaVerdict.AMBIGUOUS,
            0.0,
            False,
            combined,
            None,
            "Selected pages contain multiple different references and none matches the expected mapping key",
        )
    return (
        QaVerdict.RULE_SUPPORTED_UNVERIFIED,
        0.0,
        False,
        combined,
        None,
        "Readable content did not provide the required reference-plus-secondary-evidence combination",
    )


def _queue_bytes(rows: list[dict[str, str]]) -> bytes:
    return write_csv_bytes(rows, QA_FIELDS)


def _case_result_row(result: ContentCaseResult, observation: ContentObservation | None) -> dict[str, str]:
    identity = None
    if observation and result.matched_identity_index is not None:
        identity = observation.identities[result.matched_identity_index]
    return {
        "oachargeid": result.oachargeid,
        "route": result.route,
        "match_basis": result.match_basis,
        "amazons3_path": result.mapping_path if result.route == "s3" else "",
        "portal_path": result.mapping_path if result.route == "portal" else "",
        "decision_confidence": f"{result.recommended_confidence:.2f}",
        "qa_reference": identity.reference if identity and identity.reference else "",
        "qa_property_address": identity.property_address if identity and identity.property_address else "",
        "qa_description": identity.description if identity and identity.description else "",
        "qa_date": identity.relevant_date if identity and identity.relevant_date else "",
        "qa_document_type": identity.document_type if identity and identity.document_type else "",
        "qa_verdict": result.verdict.value,
        "qa_notes": result.reason,
    }


class CaseVerifier(Protocol):
    """Decides a case directly, instead of extracting then judging.

    A deterministic verifier already knows what the source says, so it answers
    "is this record present in this document" without an extraction step. It
    therefore replaces both the extractor and `judge_observation`.
    """

    def prepare(self, expectations: Sequence[ContentExpectation]) -> None: ...

    def verify(
        self,
        *,
        expectation: ContentExpectation,
        images: tuple[Path, ...],
        artifacts: ArtifactStore,
        case_token: str,
    ) -> tuple[QaVerdict, float, bool, dict[str, bool], int | None, str, Path]: ...


def run_content_qa(
    *,
    run_id: str,
    config: ContentQaConfig,
    artifacts: ArtifactStore,
    extractor: ObservationExtractor | None = None,
    acquirer: DocumentAcquirer | None = None,
    verifier: "CaseVerifier | None" = None,
) -> ContentVerificationReport:
    config.validate()
    if extractor is None and verifier is None:
        raise ContentQaError("Content QA needs either an extractor or a verifier")
    try:
        from registry import PIPELINES
    except ImportError as exc:  # pragma: no cover - entrypoint places amazons3-mapping on sys.path
        raise ContentQaError("Could not import the mapping registry") from exc
    pipeline = PIPELINES.get(config.council)
    if pipeline is None:
        raise ContentQaError(f"Council {config.council!r} is not registered")
    # The same admission rule as the mapping path: a batch mapped by a compiled
    # spec has no builder and can only be known from its declaration.
    from .preparation import autonomous_batches

    known = {builder.name for builder in pipeline.builders} | autonomous_batches(config.council)
    if config.batch not in known:
        raise ContentQaError(
            f"Batch {config.batch!r} is not registered for council {config.council!r}. Add it to "
            f"/data/{config.council}/file-matching/autonomous-batches.json to allow the autonomous "
            "path to check it."
        )
    source_rows = load_tabular(config.source_path, original_name=config.source_original_name)
    if not source_rows:
        raise ContentQaError("Source evidence contains no rows")
    if config.source_id_field not in source_rows[0]:
        raise ContentQaError(f"source_id_field {config.source_id_field!r} is absent from source evidence")
    source_by_id: dict[str, dict[str, str]] = {}
    for row in source_rows:
        oid = str(row.get(config.source_id_field) or "").strip()
        if not oid:
            raise ContentQaError("Source evidence contains a blank source ID")
        if oid in source_by_id:
            raise ContentQaError(f"Source evidence contains duplicate ID {oid!r}")
        source_by_id[oid] = row
    source_fields = tuple(source_rows[0])
    fields = identity_fields(source_fields, config.field_profile)
    _, audit_rows = read_csv(config.audit_path)
    if not audit_rows:
        raise ContentQaError("Audit evidence contains no rows")
    selected = select_audit_rows(audit_rows, config)
    expectations: list[ContentExpectation] = []
    for audit in selected:
        oid = str(audit.get("oachargeid") or "").strip()
        audit_batch = str(audit.get("batch") or "").strip()
        if audit_batch and audit_batch != config.batch:
            raise ContentQaError(
                f"Audit oachargeid {oid!r} belongs to batch {audit_batch!r}, not {config.batch!r}"
            )
        source = source_by_id.get(oid)
        if source is None:
            raise ContentQaError(f"Audit oachargeid {oid!r} is absent from source evidence")
        expectations.append(
            build_expectation(
                council=config.council,
                batch=config.batch,
                audit=audit,
                source=source,
                fields=fields,
                configured_reference_fields=config.field_profile.reference_fields,
            )
        )
    if verifier is not None:
        # Swap detection compares a case against the rest of the sample, so the
        # verifier needs the whole set before any case is judged.
        verifier.prepare(expectations)
    expectations_path = artifacts.write_immutable_json(
        "qa/source-expectations.json",
        [expectation.model_dump(mode="json") for expectation in expectations],
    )
    # Nothing in the source to compare a scan against means the verdict is
    # settled before the scan is fetched. Exeter's Microfiche 1977-1985 table
    # fills only the charge identifier, and 388 scans were downloaded to reach
    # rule_supported_unverified on every one of them. Acquisition is skipped in
    # that case; the verdict below is the same one the judge would return, so
    # the gate reads "the source states no address or description" rather than
    # the misleading "acquisition produced no document set".
    nothing_to_verify = not any(
        expectation.address_values
        or expectation.description_values
        or expectation.date_values
        or expectation.reference_values
        for expectation in expectations
    )
    if nothing_to_verify:
        acquirer = None

    acquisition_report: AcquisitionBatchReport | None = None
    acquisition_by_id = {}
    documents_root = config.documents_root
    if acquirer is not None:
        acquisition_report = acquirer.acquire(tuple(expectations), artifacts)
        documents_root = acquisition_report.documents_root
        acquisition_by_id = {
            report.oachargeid: report for report in acquisition_report.case_reports
        }
    if documents_root is None and not nothing_to_verify:
        raise ContentQaError(
            "documents_root is required unless an automatic document acquirer is configured"
        )
    results: list[ContentCaseResult] = []
    observations: dict[str, ContentObservation] = {}
    queue_rows: list[dict[str, str]] = []
    for expectation in expectations:
        case_token = safe_token(expectation.oachargeid)
        expectation_path = artifacts.write_immutable_json(
            f"qa/cases/{case_token}/expectation.json",
            expectation.model_dump(mode="json"),
        )
        acquisition = acquisition_by_id.get(expectation.oachargeid)
        acquisition_status = acquisition.status if acquisition is not None else None
        if acquisition is not None:
            directory = (
                acquisition.destination
                if acquisition.status
                in {AcquisitionStatus.COMPLETED, AcquisitionStatus.SKIPPED_EXISTING}
                and acquisition.destination is not None
                and acquisition.destination.is_dir()
                else None
            )
        elif documents_root is None:
            directory = None
        else:
            directory = find_document_directory(expectation, documents_root)
        selected_images: tuple[Path, ...] = ()
        observation: ContentObservation | None = None
        observation_path: Path | None = None
        warnings: list[str] = []
        if not expectation.accepted:
            verdict, confidence, eligible, signals, matched_index, reason = judge_observation(
                expectation,
                ContentObservation(
                    readable=False,
                    images_reviewed=0,
                    identities=(),
                    unreadable_image_names=(),
                    warnings=(),
                ),
            )
        elif nothing_to_verify:
            # Judged on the source alone: with no address, description or date
            # recorded, no document could confirm or contradict the match.
            verdict = QaVerdict.RULE_SUPPORTED_UNVERIFIED
            confidence = 0.0
            eligible = False
            signals = {"reference": False, "address": False, "description": False, "date": False}
            matched_index = None
            reason = (
                "The source record states no address or description, so its document cannot be "
                "checked against it"
            )
        elif directory is None:
            verdict = QaVerdict.MISSING_DOCUMENT
            confidence = 0.0
            eligible = False
            signals = {"reference": False, "address": False, "description": False, "date": False}
            matched_index = None
            reason = (
                f"Automatic acquisition did not produce a complete document set ({acquisition_status.value})"
                if acquisition_status is not None
                else "Accepted mapping has no resolvable local document directory"
            )
        else:
            selected_images = prepare_neutral_images(
                case_token=case_token,
                directory=directory,
                artifacts=artifacts,
                config=config,
            )
            if not selected_images:
                verdict = QaVerdict.MISSING_DOCUMENT
                confidence = 0.0
                eligible = False
                signals = {"reference": False, "address": False, "description": False, "date": False}
                matched_index = None
                reason = "Document directory contains no supported page images"
            elif verifier is not None:
                (
                    verdict,
                    confidence,
                    eligible,
                    signals,
                    matched_index,
                    reason,
                    observation_path,
                ) = verifier.verify(
                    expectation=expectation,
                    images=selected_images,
                    artifacts=artifacts,
                    case_token=case_token,
                )
            else:
                observation, observation_path = extractor.extract(
                    case_token=case_token,
                    images=selected_images,
                    artifacts=artifacts,
                )
                observations[expectation.oachargeid] = observation
                verdict, confidence, eligible, signals, matched_index, reason = judge_observation(
                    expectation, observation
                )
                warnings.extend(observation.warnings)
        result = ContentCaseResult(
            council=config.council,
            batch=config.batch,
            oachargeid=expectation.oachargeid,
            route=expectation.route,
            mapping_path=expectation.mapping_path,
            match_basis=expectation.match_basis,
            verdict=verdict,
            recommended_confidence=confidence,
            eligible_for_confidence_upgrade=eligible,
            signals=signals,
            matched_identity_index=matched_index,
            document_directory=directory,
            selected_images=selected_images,
            expectation_path=expectation_path,
            observation_path=observation_path,
            acquisition_status=acquisition_status,
            reason=reason,
            warnings=tuple(warnings),
        )
        artifacts.write_immutable_json(
            f"qa/cases/{case_token}/result.json", result.model_dump(mode="json")
        )
        results.append(result)
        queue_rows.append(_case_result_row(result, observation))
    queue_path = artifacts.write_immutable("qa/content-review-results.csv", _queue_bytes(queue_rows))
    case_results_path = artifacts.write_immutable_json(
        "qa/content-case-results.json", [result.model_dump(mode="json") for result in results]
    )
    verdict_counts = Counter(result.verdict.value for result in results)
    accepted_ids = {expectation.oachargeid for expectation in expectations if expectation.accepted}
    accepted_results = [result for result in results if result.oachargeid in accepted_ids]
    sample_passed = bool(accepted_results) and all(
        result.verdict in {QaVerdict.VERIFIED_SAME, QaVerdict.VERIFIED_REFERENCE_ONLY}
        for result in accepted_results
    )
    failure_signatures: dict[tuple[str, str], int] = defaultdict(int)
    for result in results:
        if result.verdict == QaVerdict.VERIFIED_WRONG:
            failure_signatures[(result.route, result.match_basis)] += 1
    systematic_failures = sum(count for count in failure_signatures.values() if count >= 2)
    full_scope = config.scope == "full_population" and len(selected) == len(audit_rows)
    passed = full_scope and sample_passed and systematic_failures == 0
    warnings: list[str] = []
    if config.scope != "full_population":
        warnings.append(
            "A targeted or stratified sample can demonstrate defects or support calibration but cannot verify the full mapping population"
        )
    if sample_passed and not passed:
        warnings.append("The reviewed accepted sample passed; production promotion remains blocked because scope is not full population")
    report = ContentVerificationReport(
        run_id=run_id,
        council=config.council,
        batch=config.batch,
        scope=config.scope,
        population=len(audit_rows),
        selected=len(results),
        accepted_selected=len(accepted_results),
        reviewed=sum(result.observation_path is not None for result in results),
        verdict_counts=dict(sorted(verdict_counts.items())),
        systematic_content_failures=systematic_failures,
        sample_passed=sample_passed,
        passed=passed,
        queue_path=queue_path,
        expectations_path=expectations_path,
        case_results_path=case_results_path,
        acquisition_report_path=(
            artifacts.resolve("qa/acquisition/report.json") if acquisition_report is not None else None
        ),
        generated_at=utc_now(),
        warnings=tuple(warnings),
    )
    artifacts.write_immutable_json("qa/content-verification-report.json", report.model_dump(mode="json"))
    return report
