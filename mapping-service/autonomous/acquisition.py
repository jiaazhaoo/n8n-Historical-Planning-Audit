from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Callable, Protocol
from urllib.parse import parse_qs, unquote, urlparse

from .ingestion import validate_remote_url
from .schemas import (
    AcquiredFile,
    AcquisitionBatchReport,
    AcquisitionStatus,
    CaseAcquisitionReport,
    ContentExpectation,
    utc_now,
)
from .storage import ArtifactStore, sha256_bytes, sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


REGISTERED_IDOX_PORTAL_ORIGINS = {
    "braintree": "https://publicaccess.braintree.gov.uk",
    "charnwood": "https://planningexplorer.charnwood.gov.uk",
    "exeter": "https://publicaccess.exeter.gov.uk",
    "mansfield": "https://planning.mansfield.gov.uk",
    "monmouthshire": "https://planningonline.monmouthshire.gov.uk",
    "sheffield": "https://planningapps.sheffield.gov.uk",
    "testvalley": "https://view-applications.testvalley.gov.uk",
}


class AcquisitionError(RuntimeError):
    pass


class PartialAcquisitionError(AcquisitionError):
    def __init__(self, message: str, *, files: tuple[AcquiredFile, ...], retryable: bool):
        super().__init__(message)
        self.files = files
        self.retryable = retryable


class PortalAdapter(Protocol):
    def acquire(
        self,
        *,
        expectation: ContentExpectation,
        destination: Path,
        limits: "AcquisitionLimits",
        remaining_total_bytes: int,
    ) -> tuple[AcquiredFile, ...]: ...


# One bounded acquisition call fetches at most this many accepted cases. A
# larger review acquires in successive batches rather than raising the ceiling.
MAX_ACCEPTED_CASES_PER_BATCH = 5


@dataclass(frozen=True)
class AcquisitionLimits:
    max_accepted_cases: int = MAX_ACCEPTED_CASES_PER_BATCH
    max_objects_per_case: int = 100
    max_file_bytes: int = 128 * 1024 * 1024
    max_case_bytes: int = 512 * 1024 * 1024
    max_total_bytes: int = 512 * 1024 * 1024

    def validate(self) -> None:
        if not 1 <= self.max_accepted_cases <= MAX_ACCEPTED_CASES_PER_BATCH:
            raise AcquisitionError(
                f"max_accepted_cases must be between 1 and {MAX_ACCEPTED_CASES_PER_BATCH}"
            )
        if not 1 <= self.max_objects_per_case <= 500:
            raise AcquisitionError("max_objects_per_case must be between 1 and 500")
        for name in ("max_file_bytes", "max_case_bytes", "max_total_bytes"):
            if getattr(self, name) < 1:
                raise AcquisitionError(f"{name} must be positive")
        if self.max_file_bytes > self.max_case_bytes:
            raise AcquisitionError("max_file_bytes must not exceed max_case_bytes")
        if self.max_case_bytes > self.max_total_bytes:
            raise AcquisitionError("max_case_bytes must not exceed max_total_bytes")


def _safe_component(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value.strip())
    value = re.sub(r"\s+", " ", value).strip(" ._")
    if not value or value in {".", ".."}:
        raise AcquisitionError("Mapping path has no safe final path component")
    return value


def _safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise AcquisitionError(f"Unsafe remote relative path: {value!r}")
    return Path(*(_safe_component(part) for part in pure.parts))


def _safe_inside(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    absolute_root = root.absolute()
    absolute_candidate = candidate.absolute()
    try:
        relative = absolute_candidate.relative_to(absolute_root)
    except ValueError as exc:
        raise AcquisitionError(f"Acquisition target escapes documents root: {candidate}") from exc
    cursor = absolute_root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise AcquisitionError(f"Acquisition target traverses a symbolic link: {cursor}")
    resolved = candidate.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise AcquisitionError(f"Acquisition target escapes documents root: {candidate}")
    return resolved


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    bucket = parsed.netloc.strip()
    key = unquote(parsed.path).lstrip("/")
    if parsed.scheme != "s3" or not bucket or not key:
        raise AcquisitionError(f"Invalid accepted S3 mapping path: {uri}")
    if any(part in {".", ".."} for part in PurePosixPath(key).parts):
        raise AcquisitionError("Accepted S3 mapping contains an unsafe path component")
    return bucket, key.rstrip("/")


def _case_directory_name(expectation: ContentExpectation) -> str:
    if expectation.route == "portal":
        return _safe_component(expectation.oachargeid)
    _, key = _parse_s3_uri(expectation.mapping_path)
    leaf = _safe_component(PurePosixPath(key).name)
    if Path(leaf).suffix.casefold() == ".pdf":
        return _safe_component(Path(leaf).stem)
    return leaf


def _redacted_error(exc: BaseException) -> str:
    text = str(exc).replace("\n", " ").strip()
    text = re.sub(
        r"(?i)(aws_access_key_id|aws_secret_access_key|aws_session_token|authorization)\s*[=:]\s*\S+",
        r"\1=<redacted>",
        text,
    )
    return f"{type(exc).__name__}: {text[:2000]}"


def _error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        if isinstance(error, dict):
            return str(error.get("Code") or "")
    return ""


def _not_found(exc: BaseException) -> bool:
    return _error_code(exc) in {"404", "NoSuchKey", "NotFound"}


def _retryable(exc: BaseException) -> bool:
    code = _error_code(exc)
    if code in {
        "403",
        "408",
        "429",
        "500",
        "502",
        "503",
        "504",
        "AccessDenied",
        "ExpiredToken",
        "InternalError",
        "RequestTimeout",
        "SlowDown",
        "Throttling",
    }:
        return True
    text = str(exc).casefold()
    return any(
        marker in text
        for marker in (
            "timed out",
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "rate limit",
            "too many requests",
            "http 403",
            "http 429",
            "slowdown",
        )
    )


def _default_s3_client(region: str) -> Any:
    try:
        import boto3
        from key_manager import get_aws_client_kwargs
    except ImportError as exc:  # pragma: no cover - depends on runtime installation
        raise AcquisitionError("boto3 and key_manager are required for S3 acquisition") from exc
    return boto3.client("s3", **get_aws_client_kwargs(region=region))


def _read_marker(
    *,
    marker: Path,
    expectation: ContentExpectation,
    documents_root: Path,
) -> CaseAcquisitionReport | None:
    if not marker.is_file():
        return None
    try:
        report = CaseAcquisitionReport.model_validate_json(marker.read_text(encoding="utf-8"))
    except Exception:
        return None
    if (
        report.status != AcquisitionStatus.COMPLETED
        or report.council != expectation.council
        or report.batch != expectation.batch
        or report.oachargeid != expectation.oachargeid
        or report.route != expectation.route
        or report.mapping_path != expectation.mapping_path
        or report.mapping_confidence != expectation.mapping_confidence
        or report.mapping_status != expectation.mapping_status
        or not report.files
        or report.destination is None
    ):
        return None
    try:
        destination = _safe_inside(documents_root, report.destination)
    except AcquisitionError:
        return None
    if not destination.is_dir():
        return None
    for item in report.files:
        try:
            relative = _safe_relative(item.relative_path)
            expected_path = _safe_inside(destination, destination / relative)
            path = _safe_inside(destination, item.local_path)
        except AcquisitionError:
            return None
        if (
            path != expected_path
            or not path.is_file()
            or path.stat().st_size != item.actual_size
            or item.actual_size <= 0
            or sha256_file(path) != item.sha256
        ):
            return None
        if expectation.route == "s3":
            accepted = expectation.mapping_path.rstrip("/")
            if item.source_uri != accepted and not item.source_uri.startswith(accepted + "/"):
                return None
        elif expectation.route == "portal":
            accepted_url = urlparse(expectation.mapping_path)
            source_url = urlparse(item.source_uri)
            if source_url.scheme != "https" or source_url.hostname != accepted_url.hostname:
                return None
    return report


def _atomic_s3_download(client: Any, *, bucket: str, key: str, target: Path, expected_size: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    try:
        client.download_file(bucket, key, str(temporary))
        actual_size = temporary.stat().st_size if temporary.is_file() else -1
        if actual_size != expected_size or actual_size <= 0:
            raise AcquisitionError(
                f"S3 size verification failed for s3://{bucket}/{key}: "
                f"expected {expected_size}, got {actual_size}"
            )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _list_s3_objects(client: Any, *, bucket: str, prefix: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        objects.extend(item for item in page.get("Contents", []) if str(item.get("Key") or ""))
        if not page.get("IsTruncated"):
            break
        token = str(page.get("NextContinuationToken") or "")
        if not token:
            raise AcquisitionError("S3 listing was truncated without a continuation token")
    return objects


class RegisteredPortalAdapter:
    """Bounded exact-URL adapter for councils supported by the existing Idox downloader."""

    def __init__(self, *, request_interval_seconds: float = 5.0):
        self.request_interval_seconds = max(request_interval_seconds, 0.0)

    @staticmethod
    def _declared_size(value: object) -> int | None:
        text = str(value or "").strip().replace(",", "")
        match = re.search(r"(\d+(?:\.\d+)?)\s*(bytes?|kb|kib|mb|mib|gb|gib)\b", text, flags=re.I)
        if not match:
            return None
        multipliers = {
            "byte": 1,
            "bytes": 1,
            "kb": 1000,
            "kib": 1024,
            "mb": 1000**2,
            "mib": 1024**2,
            "gb": 1000**3,
            "gib": 1024**3,
        }
        return int(float(match.group(1)) * multipliers[match.group(2).casefold()])

    def acquire(
        self,
        *,
        expectation: ContentExpectation,
        destination: Path,
        limits: AcquisitionLimits,
        remaining_total_bytes: int,
    ) -> tuple[AcquiredFile, ...]:
        from council_config import load_council_config
        from download_case import load_portal_module

        registered_origin = REGISTERED_IDOX_PORTAL_ORIGINS.get(expectation.council)
        if registered_origin is None:
            raise AcquisitionError(
                f"No registered Idox Portal origin for council {expectation.council!r}"
            )
        try:
            cfg = load_council_config(expectation.council)
        except ValueError:
            cfg = SimpleNamespace(portal_base_url=registered_origin)
        if cfg.portal_base_url.rstrip("/") != registered_origin:
            raise AcquisitionError("Council Portal configuration conflicts with the registered origin")
        accepted = urlparse(expectation.mapping_path)
        official = urlparse(registered_origin)
        query = parse_qs(accepted.query)
        if (
            accepted.scheme != "https"
            or accepted.hostname != official.hostname
            or accepted.port != official.port
            or accepted.path != "/online-applications/applicationDetails.do"
            or len(query.get("keyVal", [])) != 1
            or not query["keyVal"][0].strip()
        ):
            raise AcquisitionError(
                "Accepted Portal URL must be an exact Idox applicationDetails URL on the registered HTTPS host"
            )
        validate_remote_url(expectation.mapping_path)
        module = load_portal_module(cfg, self.request_interval_seconds)
        client = module.CurlSession()
        try:
            return self._acquire_documents(
                expectation=expectation,
                destination=destination,
                limits=limits,
                remaining_total_bytes=remaining_total_bytes,
                official=official,
                module=module,
                client=client,
            )
        finally:
            shutil.rmtree(client.temp_dir, ignore_errors=True)

    def _acquire_documents(
        self,
        *,
        expectation: ContentExpectation,
        destination: Path,
        limits: AcquisitionLimits,
        remaining_total_bytes: int,
        official: Any,
        module: Any,
        client: Any,
    ) -> tuple[AcquiredFile, ...]:
        summary_url = module.to_summary_url(expectation.mapping_path)
        documents_url = module.to_documents_url(expectation.mapping_path)
        summary_html = client.request_text(summary_url, referer=module.SEARCH_PAGE)
        expected_count = module.extract_documents_count(summary_html)
        documents_html = client.request_text(documents_url, referer=summary_url)
        if module.is_rate_limited_html(documents_html):
            raise AcquisitionError("Portal returned an HTTP 429/rate-limit document page")
        documents = module.parse_documents(documents_html)
        if expected_count not in (None, len(documents)):
            raise AcquisitionError(
                f"Portal summary declared {expected_count} documents but {len(documents)} were listed"
            )
        if not documents:
            raise AcquisitionError("Accepted Portal page contains no downloadable documents")
        if len(documents) > limits.max_objects_per_case:
            raise AcquisitionError(
                f"Portal case has {len(documents)} documents; limit is {limits.max_objects_per_case}"
            )
        names = [str(document.get("filename") or "") for document in documents]
        if any(not name or Path(name).name != name for name in names):
            raise AcquisitionError("Portal returned an unsafe or blank document filename")
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        if duplicates:
            raise AcquisitionError(f"Portal returned duplicate document filenames: {duplicates[:5]}")
        declared_sizes = [self._declared_size(document.get("measure")) for document in documents]
        known_total = sum(size or 0 for size in declared_sizes)
        if any(size is not None and size > limits.max_file_bytes for size in declared_sizes):
            raise AcquisitionError("Portal metadata declares a document above the per-file byte limit")
        allowed_case_bytes = min(limits.max_case_bytes, remaining_total_bytes)
        if known_total > allowed_case_bytes:
            raise AcquisitionError("Portal metadata declares a case above the remaining byte limit")

        destination.mkdir(parents=True, exist_ok=True)
        completed: list[AcquiredFile] = []
        bytes_completed = 0
        try:
            for document, declared_size in zip(documents, declared_sizes):
                url = str(document.get("url") or "")
                parsed = urlparse(url)
                if parsed.scheme != "https" or parsed.hostname != official.hostname:
                    raise AcquisitionError("Portal document URL escaped the registered council host")
                validate_remote_url(url)
                name = str(document["filename"])
                target = _safe_inside(destination, destination / name)
                remaining = allowed_case_bytes - bytes_completed
                if remaining <= 0:
                    raise AcquisitionError("Portal case exhausted the remaining byte limit")
                client.max_file_bytes = min(limits.max_file_bytes, remaining)
                temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
                try:
                    module.download_document_with_retries(
                        client=client,
                        doc=document,
                        target=temporary,
                        referer=documents_url,
                        document_retries=3,
                        document_retry_sleep_sec=max(self.request_interval_seconds, 1.0),
                        rate_limit_base_sleep_sec=max(self.request_interval_seconds, 5.0),
                    )
                    actual_size = temporary.stat().st_size if temporary.is_file() else -1
                    if actual_size <= 0 or actual_size > client.max_file_bytes:
                        raise AcquisitionError(
                            f"Portal document size is invalid or above the byte limit: {name}"
                        )
                    if declared_size is not None and abs(actual_size - declared_size) > max(
                        4096, declared_size // 20
                    ):
                        raise AcquisitionError(
                            f"Portal document size differs materially from declared metadata: {name}"
                        )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temporary, target)
                finally:
                    if temporary.exists():
                        temporary.unlink()
                bytes_completed += actual_size
                completed.append(
                    AcquiredFile(
                        source_uri=url,
                        relative_path=name,
                        local_path=target,
                        expected_size=declared_size,
                        actual_size=actual_size,
                        sha256=sha256_file(target),
                    )
                )
        except Exception as exc:
            raise PartialAcquisitionError(
                _redacted_error(exc), files=tuple(completed), retryable=_retryable(exc)
            ) from exc
        return tuple(completed)


class BoundedAcquirer:
    def __init__(
        self,
        *,
        run_id: str,
        council: str,
        batch: str,
        limits: AcquisitionLimits = AcquisitionLimits(),
        aws_region: str = "eu-west-2",
        s3_client_factory: Callable[[str], Any] = _default_s3_client,
        portal_adapter: PortalAdapter | None = None,
    ):
        self.run_id = run_id
        self.council = council
        self.batch = batch
        self.limits = limits
        self.aws_region = aws_region
        self.s3_client_factory = s3_client_factory
        self.portal_adapter = portal_adapter or RegisteredPortalAdapter()
        self.documents_root: Path | None = None

    def _report(
        self,
        *,
        expectation: ContentExpectation,
        status: AcquisitionStatus,
        destination: Path | None,
        files: tuple[AcquiredFile, ...] = (),
        error: str | None = None,
        started_at: str,
        completion_marker: Path | None = None,
    ) -> CaseAcquisitionReport:
        return CaseAcquisitionReport(
            council=expectation.council,
            batch=expectation.batch,
            oachargeid=expectation.oachargeid,
            route=expectation.route,
            mapping_path=expectation.mapping_path,
            mapping_confidence=expectation.mapping_confidence,
            mapping_status=expectation.mapping_status,
            status=status,
            destination=destination,
            files=files,
            files_completed=len(files),
            bytes_completed=sum(item.actual_size for item in files),
            error=error,
            started_at=started_at,
            ended_at=utc_now(),
            completion_marker=completion_marker,
        )

    def _s3_files(
        self,
        *,
        expectation: ContentExpectation,
        destination: Path,
        remaining_total_bytes: int,
    ) -> tuple[AcquiredFile, ...]:
        bucket, key = _parse_s3_uri(expectation.mapping_path)
        client = self.s3_client_factory(self.aws_region)
        exact = False
        try:
            metadata = client.head_object(Bucket=bucket, Key=key)
            objects = [{"Key": key, "Size": int(metadata.get("ContentLength") or 0)}]
            exact = True
        except Exception as exc:
            if not _not_found(exc):
                raise
            prefix = key.rstrip("/") + "/"
            objects = _list_s3_objects(client, bucket=bucket, prefix=prefix)
        objects = [item for item in objects if not str(item.get("Key") or "").endswith("/")]
        if not objects:
            raise AcquisitionError("Accepted S3 mapping resolves to no objects")
        if len(objects) > self.limits.max_objects_per_case:
            raise AcquisitionError(
                f"S3 case has {len(objects)} objects; limit is {self.limits.max_objects_per_case}"
            )
        prefix = key.rstrip("/") + "/"
        planned: list[tuple[str, Path, int]] = []
        for item in objects:
            object_key = str(item.get("Key") or "")
            expected_size = int(item.get("Size") or 0)
            if expected_size <= 0:
                raise AcquisitionError(f"S3 object is empty: s3://{bucket}/{object_key}")
            if expected_size > self.limits.max_file_bytes:
                raise AcquisitionError(f"S3 object exceeds the per-file byte limit: {object_key}")
            relative = PurePosixPath(object_key).name if exact else object_key[len(prefix) :] if object_key.startswith(prefix) else ""
            if not relative:
                raise AcquisitionError(f"S3 listing returned an object outside the accepted prefix: {object_key}")
            relative_path = _safe_relative(relative)
            target = _safe_inside(destination, destination / relative_path)
            planned.append((object_key, target, expected_size))
        case_bytes = sum(item[2] for item in planned)
        if case_bytes > self.limits.max_case_bytes:
            raise AcquisitionError("S3 case exceeds the per-case byte limit")
        if case_bytes > remaining_total_bytes:
            raise AcquisitionError("S3 case exceeds the remaining batch byte limit")

        destination.mkdir(parents=True, exist_ok=True)
        completed: list[AcquiredFile] = []
        try:
            for object_key, target, expected_size in planned:
                if not target.is_file() or target.stat().st_size != expected_size:
                    _atomic_s3_download(
                        client,
                        bucket=bucket,
                        key=object_key,
                        target=target,
                        expected_size=expected_size,
                    )
                completed.append(
                    AcquiredFile(
                        source_uri=f"s3://{bucket}/{object_key}",
                        relative_path=str(target.relative_to(destination)),
                        local_path=target,
                        expected_size=expected_size,
                        actual_size=target.stat().st_size,
                        sha256=sha256_file(target),
                    )
                )
        except Exception as exc:
            raise PartialAcquisitionError(
                _redacted_error(exc), files=tuple(completed), retryable=_retryable(exc)
            ) from exc
        return tuple(completed)

    def acquire(
        self,
        expectations: tuple[ContentExpectation, ...],
        artifacts: ArtifactStore,
    ) -> AcquisitionBatchReport:
        self.limits.validate()
        documents_root = artifacts.resolve("qa/acquisition/documents")
        documents_root.mkdir(parents=True, exist_ok=True)
        self.documents_root = documents_root
        for expectation in expectations:
            if expectation.council != self.council or expectation.batch != self.batch:
                raise AcquisitionError("Acquisition expectation council/batch does not match the run")
        accepted = [expectation for expectation in expectations if expectation.accepted]
        if len(accepted) > self.limits.max_accepted_cases:
            raise AcquisitionError(
                f"Selected QA queue has {len(accepted)} accepted mappings; bounded acquisition limit is "
                f"{self.limits.max_accepted_cases}"
            )
        destinations: dict[Path, str] = {}
        for expectation in accepted:
            destination = _safe_inside(documents_root, documents_root / _case_directory_name(expectation))
            previous = destinations.get(destination)
            if previous and previous != expectation.mapping_path:
                raise AcquisitionError(
                    f"Two accepted mappings collide on acquisition directory {destination.name!r}"
                )
            destinations[destination] = expectation.mapping_path

        reports: list[CaseAcquisitionReport] = []
        total_bytes = 0
        for expectation in expectations:
            started_at = utc_now()
            token = re.sub(r"[^A-Za-z0-9._-]+", "_", expectation.oachargeid).strip("._") or "blank"
            report_relative = f"qa/acquisition/cases/{token}/report.json"
            marker = artifacts.resolve(f"qa/acquisition/cases/{token}/completion.json")
            if not expectation.accepted:
                report = self._report(
                    expectation=expectation,
                    status=AcquisitionStatus.MAPPING_REJECTED,
                    destination=None,
                    error="Mapping is rejected, has no accepted path, or has zero confidence; no network request was made",
                    started_at=started_at,
                )
            else:
                destination = _safe_inside(
                    documents_root, documents_root / _case_directory_name(expectation)
                )
                prior = _read_marker(
                    marker=marker,
                    expectation=expectation,
                    documents_root=documents_root,
                )
                if prior is not None:
                    report = self._report(
                        expectation=expectation,
                        status=AcquisitionStatus.SKIPPED_EXISTING,
                        destination=prior.destination,
                        files=prior.files,
                        started_at=started_at,
                        completion_marker=marker,
                    )
                else:
                    try:
                        remaining = self.limits.max_total_bytes - total_bytes
                        if remaining <= 0:
                            raise AcquisitionError("Acquisition batch exhausted the total byte limit")
                        if expectation.route == "s3":
                            files = self._s3_files(
                                expectation=expectation,
                                destination=destination,
                                remaining_total_bytes=remaining,
                            )
                        else:
                            files = self.portal_adapter.acquire(
                                expectation=expectation,
                                destination=destination,
                                limits=self.limits,
                                remaining_total_bytes=remaining,
                            )
                        if not files:
                            raise AcquisitionError("Accepted mapping produced no verified local files")
                        report = self._report(
                            expectation=expectation,
                            status=AcquisitionStatus.COMPLETED,
                            destination=destination,
                            files=files,
                            started_at=started_at,
                            completion_marker=marker,
                        )
                        restored_marker = _read_marker(
                            marker=marker,
                            expectation=expectation,
                            documents_root=documents_root,
                        )
                        if restored_marker is None:
                            if marker.is_file():
                                previous = marker.read_bytes()
                                artifacts.write_immutable(
                                    f"qa/acquisition/cases/{token}/completion-history/"
                                    f"{sha256_bytes(previous)}.json",
                                    previous,
                                )
                            artifacts.write_mutable_json(
                                marker.relative_to(artifacts.workspace),
                                report.model_dump(mode="json"),
                            )
                    except PartialAcquisitionError as exc:
                        report = self._report(
                            expectation=expectation,
                            status=(
                                AcquisitionStatus.PARTIAL
                                if exc.files
                                else AcquisitionStatus.RETRYABLE
                                if exc.retryable
                                else AcquisitionStatus.FAILED
                            ),
                            destination=destination,
                            files=exc.files,
                            error=str(exc),
                            started_at=started_at,
                        )
                    except Exception as exc:
                        report = self._report(
                            expectation=expectation,
                            status=(AcquisitionStatus.RETRYABLE if _retryable(exc) else AcquisitionStatus.FAILED),
                            destination=destination,
                            error=_redacted_error(exc),
                            started_at=started_at,
                        )
            reports.append(report)
            total_bytes += report.bytes_completed
            artifacts.write_mutable_json(report_relative, report.model_dump(mode="json"))

        counts = Counter(report.status.value for report in reports)
        batch_report = AcquisitionBatchReport(
            run_id=self.run_id,
            council=self.council,
            batch=self.batch,
            requested_cases=len(expectations),
            accepted_cases=len(accepted),
            files_completed=sum(report.files_completed for report in reports),
            bytes_completed=sum(report.bytes_completed for report in reports),
            status_counts=dict(sorted(counts.items())),
            case_reports=tuple(reports),
            documents_root=documents_root,
            generated_at=utc_now(),
        )
        artifacts.write_mutable_json("qa/acquisition/report.json", batch_report.model_dump(mode="json"))
        return batch_report
