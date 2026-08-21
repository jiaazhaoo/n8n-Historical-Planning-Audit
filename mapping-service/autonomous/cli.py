from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .acquisition import AcquisitionLimits, BoundedAcquirer, RegisteredPortalAdapter
from .content_qa import (
    CodexOAuthVisionExtractor,
    ContentQaConfig,
    IdentityFieldProfile,
    run_content_qa,
)
from .replay import load_replay_suite, run_replay
from .runner import STAGES, AutonomousRunner, JobPaused
from .schemas import ContentVerificationReport, JobOperation, JobRequest, JobStatus, MappingSpec
from .ingestion import IngestionLimits
from .path_policy import policy_record, require_unprotected_path, require_unprotected_url
from .single_link import SingleLinkRunner
from .storage import ArtifactStore, JobStore, fingerprint, sha256_file


DEFAULT_JOBS_ROOT = Path("/data/mapping-jobs")


def render(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Staging-only autonomous council mapping job foundation."
    )
    parser.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create a durable mapping job without running it.")
    create.add_argument("--url", required=True)
    create.add_argument("--council")
    create.add_argument("--batch")
    create.add_argument("--operation", choices=[item.value for item in JobOperation], default="build")
    create.add_argument("--requested-by", default="")

    start = subparsers.add_parser(
        "start",
        help="Create and run a staging-only job from one evidence link using Codex ChatGPT OAuth.",
    )
    start.add_argument("--url", required=True)
    start.add_argument("--council")
    start.add_argument("--batch")
    start.add_argument("--operation", choices=[item.value for item in JobOperation], default="build")
    start.add_argument("--requested-by", default="")
    start.add_argument("--allow-private-hosts", action="store_true")
    start.add_argument(
        "--approved-spec",
        type=Path,
        help="Use an already-verified MappingSpec instead of compiling one. It is still verified.",
    )
    start.add_argument(
        "--prior-findings",
        type=Path,
        help=(
            "JSON array of arrays: what a quality round found wrong with an earlier spec for this "
            "batch. Carried into the compiler prompt from the first attempt, so a rework starts "
            "from what the scans showed rather than rediscovering it."
        ),
    )

    continue_job = subparsers.add_parser("continue", help="Resume a single-link job from durable state.")
    continue_job.add_argument("--job-id", required=True)
    continue_job.add_argument("--council")
    continue_job.add_argument("--batch")
    continue_job.add_argument("--allow-private-hosts", action="store_true")

    run = subparsers.add_parser("run", help="Run or resume a job against frozen local evidence.")
    run.add_argument("--job-id", required=True)
    run.add_argument("--source-csv", type=Path, required=True)
    run.add_argument("--inventory-csv", type=Path, required=True)
    run.add_argument("--capture-rules", type=Path, required=True)
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--stop-after", choices=STAGES)

    status = subparsers.add_parser("status", help="Show durable job, stage, and artifact state.")
    status.add_argument("--job-id", required=True)

    validate_spec = subparsers.add_parser("validate-spec", help="Validate a MappingSpec JSON file.")
    validate_spec.add_argument("--spec", type=Path, required=True)

    replay = subparsers.add_parser("replay", help="Run golden expectations against mapping artifacts.")
    replay.add_argument("--mapping", type=Path, required=True)
    replay.add_argument("--audit", type=Path)
    replay.add_argument("--suite", type=Path, required=True)

    content_qa = subparsers.add_parser(
        "content-qa",
        help="Run bounded blind image extraction and deterministic identity QA using ChatGPT OAuth.",
    )
    content_qa.add_argument("--council", required=True)
    content_qa.add_argument("--batch", required=True)
    content_qa.add_argument("--audit", type=Path, required=True)
    content_qa.add_argument("--source", type=Path, required=True)
    content_qa.add_argument("--source-original-name")
    content_qa.add_argument("--source-id-field", required=True)
    content_qa.add_argument("--documents-root", type=Path)
    content_qa.add_argument(
        "--acquire",
        action="store_true",
        help="Acquire accepted QA documents into the run workspace before blind content review.",
    )
    content_qa.add_argument("--output-dir", type=Path, required=True)
    content_qa.add_argument(
        "--scope",
        choices=("targeted", "stratified_sample", "full_population"),
        default="stratified_sample",
    )
    content_qa.add_argument("--sample-size", type=int, default=5)
    content_qa.add_argument("--max-images", type=int, default=12)
    content_qa.add_argument("--max-acquired-cases", type=int, default=5)
    content_qa.add_argument("--max-objects-per-case", type=int, default=100)
    content_qa.add_argument("--max-file-bytes", type=int, default=128 * 1024 * 1024)
    content_qa.add_argument("--max-case-bytes", type=int, default=512 * 1024 * 1024)
    content_qa.add_argument("--max-total-bytes", type=int, default=512 * 1024 * 1024)
    content_qa.add_argument("--aws-region", default=os.getenv("AWS_REGION", "eu-west-2"))
    content_qa.add_argument("--portal-request-interval", type=float, default=5.0)
    content_qa.add_argument("--include-id", action="append", default=[])
    content_qa.add_argument("--reference-field", action="append", default=[])
    content_qa.add_argument("--address-field", action="append", default=[])
    content_qa.add_argument("--description-field", action="append", default=[])
    content_qa.add_argument("--date-field", action="append", default=[])
    content_qa.add_argument("--document-type-field", action="append", default=[])
    return parser.parse_args(argv)


def load_prior_findings(path: Path | None) -> tuple[tuple[str, ...], ...]:
    """Read what a quality round found wrong with an earlier spec.

    A rework that starts from a blank prompt asks the compiler to rediscover
    what the scans have already shown, and nothing stops it from proposing
    again the spec that was just rejected.
    """
    if path is None:
        return ()
    resolved = require_unprotected_path(path, operation="read prior quality findings")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"--prior-findings must hold a JSON array: {resolved}")
    findings: list[tuple[str, ...]] = []
    for entry in payload:
        items = entry if isinstance(entry, list) else [entry]
        cleaned = tuple(str(item).strip() for item in items if str(item).strip())
        if cleaned:
            findings.append(cleaned)
    return tuple(findings)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "validate-spec":
            spec_path = require_unprotected_path(args.spec, operation="read MappingSpec")
            spec = MappingSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
            print(render({"valid": True, "spec": spec.model_dump(mode="json")}))
            return 0

        if args.command == "replay":
            mapping_path = require_unprotected_path(args.mapping, operation="read replay mapping")
            audit_path = (
                require_unprotected_path(args.audit, operation="read replay audit")
                if args.audit
                else None
            )
            suite_path = require_unprotected_path(args.suite, operation="read replay suite")
            result = run_replay(
                mapping_path=mapping_path,
                audit_path=audit_path,
                suite=load_replay_suite(suite_path),
            )
            print(render(result))
            return 1 if result.failed else 0

        if args.command == "content-qa":
            if not args.acquire and args.documents_root is None:
                raise ValueError("--documents-root is required unless --acquire is enabled")
            config = ContentQaConfig(
                council=args.council,
                batch=args.batch,
                audit_path=args.audit.resolve(),
                source_path=args.source.resolve(),
                source_original_name=args.source_original_name or args.source.name,
                source_id_field=args.source_id_field,
                documents_root=(args.documents_root.resolve() if args.documents_root else None),
                scope=args.scope,
                sample_size=args.sample_size,
                include_ids=tuple(args.include_id),
                max_images_per_case=args.max_images,
                field_profile=IdentityFieldProfile(
                    reference_fields=tuple(args.reference_field),
                    address_fields=tuple(args.address_field),
                    description_fields=tuple(args.description_field),
                    date_fields=tuple(args.date_field),
                    document_type_fields=tuple(args.document_type_field),
                ),
            )
            config.validate()
            artifacts = ArtifactStore(args.output_dir)
            run_config = {
                "council": config.council,
                "batch": config.batch,
                "audit_path": str(config.audit_path),
                "audit_sha256": sha256_file(config.audit_path),
                "source_path": str(config.source_path),
                "source_sha256": sha256_file(config.source_path),
                "source_original_name": config.source_original_name,
                "source_id_field": config.source_id_field,
                "documents_root": str(config.documents_root) if config.documents_root else None,
                "acquire": args.acquire,
                "acquisition_limits": {
                    "max_accepted_cases": args.max_acquired_cases,
                    "max_objects_per_case": args.max_objects_per_case,
                    "max_file_bytes": args.max_file_bytes,
                    "max_case_bytes": args.max_case_bytes,
                    "max_total_bytes": args.max_total_bytes,
                },
                "aws_region": args.aws_region,
                "portal_request_interval": args.portal_request_interval,
                "scope": config.scope,
                "sample_size": config.sample_size,
                "include_ids": config.include_ids,
                "max_images_per_case": config.max_images_per_case,
                "field_profile": config.field_profile.__dict__,
                "oauth_only": True,
                "path_access_policy": policy_record(),
            }
            artifacts.write_immutable_json("qa/run-config.json", run_config)
            run_id = f"content_{fingerprint(run_config)[:16]}"
            report_path = artifacts.resolve("qa/content-verification-report.json")
            if report_path.exists():
                report = ContentVerificationReport.model_validate_json(
                    report_path.read_text(encoding="utf-8")
                )
            else:
                report = run_content_qa(
                    run_id=run_id,
                    config=config,
                    artifacts=artifacts,
                    extractor=CodexOAuthVisionExtractor(
                        repository_root=Path(__file__).resolve().parents[2]
                    ),
                    acquirer=(
                        BoundedAcquirer(
                            run_id=run_id,
                            council=config.council,
                            batch=config.batch,
                            limits=AcquisitionLimits(
                                max_accepted_cases=args.max_acquired_cases,
                                max_objects_per_case=args.max_objects_per_case,
                                max_file_bytes=args.max_file_bytes,
                                max_case_bytes=args.max_case_bytes,
                                max_total_bytes=args.max_total_bytes,
                            ),
                            aws_region=args.aws_region,
                            portal_adapter=RegisteredPortalAdapter(
                                request_interval_seconds=args.portal_request_interval
                            ),
                        )
                        if args.acquire
                        else None
                    ),
                )
            print(render(report))
            return 0 if report.sample_passed else 4

        store = JobStore(args.jobs_root)
        if args.command in {"create", "start"}:
            require_unprotected_url(args.url, operation="read job source")
            request = JobRequest(
                source_url=args.url,
                council_hint=args.council,
                batch_hint=args.batch,
                operation=JobOperation(args.operation),
                requested_by=args.requested_by,
            )
            job = store.create_job(request)
            ArtifactStore(job.workspace).write_mutable_json("job.json", store.snapshot(job.job_id))
            if args.command == "create":
                print(render(job))
                return 0
            approved = getattr(args, "approved_spec", None)
            runner = SingleLinkRunner(
                store,
                repository_root=Path(__file__).resolve().parents[2],
                ingestion_limits=IngestionLimits(allow_private_hosts=args.allow_private_hosts),
                approved_spec=(
                    require_unprotected_path(approved, operation="read approved MappingSpec")
                    if approved
                    else None
                ),
                prior_findings=load_prior_findings(getattr(args, "prior_findings", None)),
            )
            result = runner.run(job_id=job.job_id)
            print(render(store.snapshot(result.job_id)))
            return 0 if result.status != JobStatus.AWAITING_INPUT else 3

        if args.command == "continue":
            if bool(args.council) != bool(args.batch):
                raise ValueError("--council and --batch must be supplied together")
            job = store.get_job(args.job_id)
            if args.council and args.batch:
                job = store.update_job(
                    args.job_id,
                    status=JobStatus.CREATED,
                    council=args.council,
                    batch=args.batch,
                )
            approved = getattr(args, "approved_spec", None)
            runner = SingleLinkRunner(
                store,
                repository_root=Path(__file__).resolve().parents[2],
                ingestion_limits=IngestionLimits(allow_private_hosts=args.allow_private_hosts),
                approved_spec=(
                    require_unprotected_path(approved, operation="read approved MappingSpec")
                    if approved
                    else None
                ),
                prior_findings=load_prior_findings(getattr(args, "prior_findings", None)),
            )
            result = runner.run(job_id=job.job_id)
            print(render(store.snapshot(result.job_id)))
            return 0 if result.status != JobStatus.AWAITING_INPUT else 3

        if args.command == "status":
            print(render(store.snapshot(args.job_id)))
            return 0

        if args.command == "run":
            runner = AutonomousRunner(store)
            try:
                job = runner.run(
                    job_id=args.job_id,
                    source_csv=args.source_csv,
                    inventory_csv=args.inventory_csv,
                    capture_rules=args.capture_rules,
                    spec_path=args.spec,
                    stop_after=args.stop_after,
                )
            except JobPaused:
                print(render(store.snapshot(args.job_id)))
                return 0
            print(render(job))
            return 0
    except (KeyError, OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(render({"error": type(exc).__name__, "detail": str(exc)}))
        return 2
    return 1
