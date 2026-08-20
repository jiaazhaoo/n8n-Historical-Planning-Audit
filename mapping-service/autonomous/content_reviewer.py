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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .acquisition import AcquisitionLimits, BoundedAcquirer, RegisteredPortalAdapter
from .content_qa import (
    CodexOAuthVisionExtractor,
    ContentQaConfig,
    IdentityFieldProfile,
    run_content_qa,
)
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
    field_profile: IdentityFieldProfile = IdentityFieldProfile()
    aws_region: str = "eu-west-2"
    portal_request_interval: float = 5.0
    max_objects_per_case: int = 100
    max_file_bytes: int = 128 * 1024 * 1024
    max_case_bytes: int = 512 * 1024 * 1024
    max_total_bytes: int = 512 * 1024 * 1024

    def validate(self) -> None:
        if not self.acquire and self.documents_root is None:
            raise ContentReviewError(
                "documents_root is required when automatic acquisition is disabled"
            )


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
            field_profile=settings.field_profile,
        )
        config.validate()

        artifacts = ArtifactStore(output_dir)
        acquirer = (
            BoundedAcquirer(
                run_id=run_id,
                council=settings.council,
                batch=settings.batch,
                limits=AcquisitionLimits(
                    max_accepted_cases=len(include_ids),
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

        report = run_content_qa(
            run_id=run_id,
            config=config,
            artifacts=artifacts,
            extractor=CodexOAuthVisionExtractor(repository_root=self.repository_root),
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
        return report.model_dump(mode="json"), case_results
