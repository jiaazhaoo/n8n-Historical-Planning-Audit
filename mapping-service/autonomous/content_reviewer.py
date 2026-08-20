"""Adapt the OAuth vision content-QA run to the quality loop's reviewer contract.

The loop decides which cases to review; this decides nothing. It reviews exactly
the identifiers it is handed by running content QA in `targeted` scope, so the
holdout and round-without-replacement guarantees stay in the loop rather than
being re-derived by a component that cannot see the loop's state.

This module is kept apart from `quality_loop` so the loop stays importable, and
testable, without the vision stack, the registry, or network credentials.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from .acquisition import (
    AcquisitionLimits,
    BoundedAcquirer,
    MAX_ACCEPTED_CASES_PER_BATCH,
    RegisteredPortalAdapter,
)
from .schemas import AcquisitionBatchReport
from .content_qa import ContentQaConfig, IdentityFieldProfile, run_content_qa
from .openrouter_vision import (
    DEFAULT_MODEL,
    OpenRouterSettings,
    OpenRouterVisionExtractor,
    read_api_key,
)
from .ocr_extractor import PaddleOcrObservationExtractor
from .paddle_ocr import PaddleOcrRunner
from .review_budget import DEFAULT_ESTIMATE_USD_PER_CASE, DEFAULT_LIMIT_USD, ReviewBudget
from .storage import ArtifactStore


class ContentReviewError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReviewerSettings:
    """Everything the review needs that the loop has no opinion about."""

    council: str
    batch: str
    audit_path: Path
    source_path: Path
    source_id_field: str
    source_original_name: str = ""
    documents_root: Path | None = None
    acquire: bool = True
    max_images_per_case: int = 12
    # Archival microfiche frames are far larger than a page scan. Measured over
    # 219 real Exeter WP1 frames on 2026-08-20: median 6.9 MP, but 5.9% of them
    # are 52-56 MP large-format frames that a 40 MP guard rejects outright. The
    # guard exists to stop decompression bombs, which are orders of magnitude
    # larger again, so it is raised to clear genuine scans with headroom.
    max_image_pixels: int = 80_000_000
    field_profile: IdentityFieldProfile = IdentityFieldProfile()
    aws_region: str = "eu-west-2"
    portal_request_interval: float = 5.0
    max_objects_per_case: int = 100
    max_file_bytes: int = 128 * 1024 * 1024
    max_case_bytes: int = 512 * 1024 * 1024
    max_total_bytes: int = 512 * 1024 * 1024
    openrouter_key_path: Path = Path("/env/key/spatial_capture.keys.md")
    model: str = DEFAULT_MODEL
    budget_usd: float = DEFAULT_LIMIT_USD
    estimate_usd_per_case: float = DEFAULT_ESTIMATE_USD_PER_CASE
    # "ocr" reads pages locally with PaddleOCR and sends only their text;
    # "vision" sends the page images themselves.
    extractor_mode: str = "ocr"

    def validate(self) -> None:
        if not self.acquire and self.documents_root is None:
            raise ContentReviewError(
                "documents_root is required when automatic acquisition is disabled"
            )
        if self.budget_usd <= 0:
            raise ContentReviewError("budget_usd must be positive")
        if self.extractor_mode not in {"ocr", "vision"}:
            raise ContentReviewError(
                f"extractor_mode must be 'ocr' or 'vision', got {self.extractor_mode!r}"
            )


class ChunkedAcquirer:
    """Acquire a sample larger than one bounded batch, without unbounding it.

    `BoundedAcquirer` accepts at most five accepted cases per call. That ceiling
    made a review as small as the acquisition batch, which is too thin for a
    stratified sample to say anything. Here the sample is acquired in successive
    batches instead, and the byte ceiling -- the limit that actually bounds how
    much data leaves the bucket -- is carried across them and decremented, so
    the total egress stays within one `max_total_bytes` no matter how many
    batches it takes.
    """

    def __init__(
        self,
        *,
        run_id: str,
        council: str,
        batch: str,
        limits: AcquisitionLimits,
        aws_region: str,
        portal_adapter: RegisteredPortalAdapter,
    ) -> None:
        self.run_id = run_id
        self.council = council
        self.batch = batch
        self.limits = limits
        self.aws_region = aws_region
        self.portal_adapter = portal_adapter
        self.documents_root: Path | None = None

    def _batches(self, expectations: Sequence[Any]) -> list[tuple[Any, ...]]:
        """Group so no batch holds more accepted cases than one call allows."""
        batches: list[list[Any]] = []
        current: list[Any] = []
        accepted_in_current = 0
        for expectation in expectations:
            if expectation.accepted and accepted_in_current >= MAX_ACCEPTED_CASES_PER_BATCH:
                batches.append(current)
                current, accepted_in_current = [], 0
            current.append(expectation)
            accepted_in_current += int(bool(expectation.accepted))
        if current:
            batches.append(current)
        return [tuple(item) for item in batches]

    def acquire(self, expectations: tuple[Any, ...], artifacts: ArtifactStore):
        batches = self._batches(expectations)
        remaining_bytes = self.limits.max_total_bytes
        reports: list[AcquisitionBatchReport] = []
        for index, group in enumerate(batches, start=1):
            if remaining_bytes <= 0:
                break
            limits = replace(
                self.limits,
                max_accepted_cases=min(
                    MAX_ACCEPTED_CASES_PER_BATCH,
                    max(1, sum(1 for item in group if item.accepted)),
                ),
                max_total_bytes=remaining_bytes,
                max_case_bytes=min(self.limits.max_case_bytes, remaining_bytes),
                max_file_bytes=min(self.limits.max_file_bytes, remaining_bytes),
            )
            acquirer = BoundedAcquirer(
                run_id=f"{self.run_id}-acq{index:02d}",
                council=self.council,
                batch=self.batch,
                limits=limits,
                aws_region=self.aws_region,
                portal_adapter=self.portal_adapter,
            )
            report = acquirer.acquire(group, artifacts)
            self.documents_root = acquirer.documents_root
            remaining_bytes -= report.bytes_completed
            reports.append(report)

        if not reports:
            raise ContentReviewError("Acquisition produced no batches")

        status_counts: dict[str, int] = {}
        for report in reports:
            for status, count in report.status_counts.items():
                status_counts[status] = status_counts.get(status, 0) + count
        merged = AcquisitionBatchReport(
            run_id=self.run_id,
            council=self.council,
            batch=self.batch,
            requested_cases=sum(item.requested_cases for item in reports),
            accepted_cases=sum(item.accepted_cases for item in reports),
            files_completed=sum(item.files_completed for item in reports),
            bytes_completed=sum(item.bytes_completed for item in reports),
            status_counts=status_counts,
            case_reports=tuple(
                case for report in reports for case in report.case_reports
            ),
            documents_root=reports[-1].documents_root,
            generated_at=reports[-1].generated_at,
        )
        # The per-batch file is overwritten by each batch; keep the merged view
        # as the one the verification report cites.
        artifacts.write_mutable_json(
            "qa/acquisition/report.json", merged.model_dump(mode="json")
        )
        return merged


class ContentQaReviewer:
    """Reviews a named case set with the blind vision extractor."""

    def __init__(self, settings: ReviewerSettings, *, repository_root: Path) -> None:
        settings.validate()
        self.settings = settings
        self.repository_root = repository_root

    def review(
        self,
        *,
        run_id: str,
        include_ids: Sequence[str],
        output_dir: Path,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not include_ids:
            raise ContentReviewError("A review needs at least one case identifier")
        settings = self.settings
        budget = ReviewBudget(
            limit_usd=settings.budget_usd,
            estimate_usd_per_case=settings.estimate_usd_per_case,
        )

        # Trim the sample before any document is fetched. Discovering the
        # ceiling mid-round would leave a partly reviewed sample whose pass rate
        # describes whichever cases happened to be cheap.
        affordable = budget.affordable_cases()
        if affordable < 1:
            raise ContentReviewError(
                f"A budget of ${settings.budget_usd:.2f} does not cover one case at an "
                f"estimated ${settings.estimate_usd_per_case:.4f} per case"
            )
        if len(include_ids) > affordable:
            budget.note_truncation(len(include_ids) - affordable)
            include_ids = tuple(include_ids)[:affordable]
        config = ContentQaConfig(
            council=settings.council,
            batch=settings.batch,
            audit_path=settings.audit_path.resolve(),
            source_path=settings.source_path.resolve(),
            source_original_name=settings.source_original_name or settings.source_path.name,
            source_id_field=settings.source_id_field,
            documents_root=(
                settings.documents_root.resolve() if settings.documents_root else None
            ),
            # The loop already chose the sample; targeted scope reviews that set
            # verbatim instead of drawing a second, unrelated one.
            scope="targeted",
            sample_size=min(max(len(include_ids), 1), 40),
            include_ids=tuple(include_ids),
            max_images_per_case=settings.max_images_per_case,
            max_image_pixels=settings.max_image_pixels,
            field_profile=settings.field_profile,
        )
        config.validate()

        artifacts = ArtifactStore(output_dir)
        acquirer = (
            ChunkedAcquirer(
                run_id=run_id,
                council=settings.council,
                batch=settings.batch,
                limits=AcquisitionLimits(
                    max_accepted_cases=MAX_ACCEPTED_CASES_PER_BATCH,
                    max_objects_per_case=settings.max_objects_per_case,
                    max_file_bytes=settings.max_file_bytes,
                    max_case_bytes=settings.max_case_bytes,
                    max_total_bytes=settings.max_total_bytes,
                ),
                aws_region=settings.aws_region,
                portal_adapter=RegisteredPortalAdapter(
                    request_interval_seconds=settings.portal_request_interval
                ),
            )
            if settings.acquire
            else None
        )

        openrouter = OpenRouterSettings(
            api_key=read_api_key(settings.openrouter_key_path),
            model=settings.model,
        )
        extractor = (
            PaddleOcrObservationExtractor(
                settings=openrouter, budget=budget, runner=PaddleOcrRunner()
            )
            if settings.extractor_mode == "ocr"
            else OpenRouterVisionExtractor(openrouter, budget=budget)
        )
        report = run_content_qa(
            run_id=run_id,
            config=config,
            artifacts=artifacts,
            extractor=extractor,
            acquirer=acquirer,
        )
        case_results = json.loads(
            Path(report.case_results_path).read_text(encoding="utf-8")
        )
        reviewed = {str(result.get("oachargeid") or "") for result in case_results}
        missing = [value for value in include_ids if value not in reviewed]
        if missing:
            raise ContentReviewError(
                f"Content QA did not return results for requested cases: {missing}"
            )
        summary = report.model_dump(mode="json")
        summary["budget"] = budget.describe()
        summary["model"] = settings.model
        summary["extractor_mode"] = settings.extractor_mode
        artifacts.write_mutable(
            "qa/review-budget.json",
            json.dumps(summary["budget"], ensure_ascii=False, indent=2).encode("utf-8"),
        )
        return summary, case_results
