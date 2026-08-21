from __future__ import annotations

from pathlib import Path

from .compiler import CodexOAuthCompiler
from .ingestion import IngestionLimits, discover_link
from .path_policy import require_unprotected_path
from .preparation import prepare_evidence
from .qa import build_qa_queue
from .replay import load_replay_suite, run_replay
from .runner import AutonomousRunner, JobPaused
from .schemas import (
    DiscoveryManifest,
    ContentQaReport,
    HistoricalReplayReport,
    JobRecord,
    JobStatus,
    MappingSpec,
    PreparationReport,
    PublishPolicyInputs,
    SpecVerificationReport,
)
from .spec_verifier import verify_mapping_spec
from .storage import ArtifactStore, JobStore, sha256_file


LINK_STAGES = ("discover", "prepare", "compile_spec", "verify_spec", "historical_replay", "qa_sample")


class SingleLinkRunner:
    def __init__(
        self,
        store: JobStore,
        *,
        repository_root: Path,
        compiler: CodexOAuthCompiler | None = None,
        approved_spec: Path | None = None,
        compile_attempts: int = 3,
        ingestion_limits: IngestionLimits | None = None,
    ):
        self.store = store
        self.repository_root = require_unprotected_path(
            repository_root,
            operation="use mapping repository root",
        )
        self.compiler = compiler or CodexOAuthCompiler(repository_root=self.repository_root)
        # A spec that has already been verified against this council and batch.
        # Supplying one replaces compilation, not verification: the spec still
        # faces the same checks a compiled one does.
        self.approved_spec = approved_spec
        # Bounded: an unbounded retry would let the compiler search until
        # something passes, which is not the same as getting it right.
        self.compile_attempts = max(1, compile_attempts)
        self.ingestion_limits = ingestion_limits or IngestionLimits()

    def _await_input(
        self,
        *,
        job_id: str,
        artifacts: ArtifactStore,
        reason: str,
        required: list[str],
    ) -> JobRecord:
        path = artifacts.write_mutable_json(
            "needs-input.json",
            {"job_id": job_id, "reason": reason, "required": required, "production_mutated": False},
        )
        self.store.register_artifact(job_id, "needs_input", "needs_input", path)
        job = self.store.update_job(
            job_id,
            status=JobStatus.AWAITING_INPUT,
            current_stage="needs_input",
            error=reason,
        )
        self.store.event(job_id, "job_awaiting_input", stage="needs_input", payload={"reason": reason, "required": required})
        artifacts.write_mutable_json("job.json", self.store.snapshot(job_id))
        return job

    def _context(self, job: JobRecord, discovery: DiscoveryManifest) -> tuple[str | None, str | None, str | None]:
        if bool(discovery.declared_council) != bool(discovery.declared_batch):
            return None, None, "Link manifest must declare council and batch together"
        if job.council and discovery.declared_council and job.council != discovery.declared_council:
            return None, None, "Job council conflicts with the link manifest council"
        if job.batch and discovery.declared_batch and job.batch != discovery.declared_batch:
            return None, None, "Job batch conflicts with the link manifest batch"
        return (
            job.council or discovery.declared_council,
            job.batch or discovery.declared_batch,
            None,
        )

    def run(self, *, job_id: str) -> JobRecord:
        job = self.store.get_job(job_id)
        artifacts = ArtifactStore(job.workspace)
        stage_runner = AutonomousRunner(self.store)

        discovery_path = artifacts.resolve("ingestion/discovery-manifest.json")

        def discover() -> list[tuple[str, Path]]:
            manifest = discover_link(
                job_id=job_id,
                root_url=job.request.source_url,
                artifacts=artifacts,
                limits=self.ingestion_limits,
                created_at=job.created_at,
            )
            path = artifacts.write_immutable_json(
                "ingestion/discovery-manifest.json",
                manifest.model_dump(mode="json"),
            )
            outputs: list[tuple[str, Path]] = [("discovery_manifest", path)]
            outputs.extend(("discovered_evidence", item.local_path) for item in manifest.artifacts)
            return outputs

        stage_runner._stage(
            job_id=job_id,
            artifacts=artifacts,
            stage="discover",
            input_value={
                "root_url": job.request.source_url,
                "limits": self.ingestion_limits.__dict__,
            },
            execute=discover,
        )
        discovery = DiscoveryManifest.model_validate_json(discovery_path.read_text(encoding="utf-8"))
        if not discovery.evidence_complete:
            missing = [role.value for role in discovery.missing_roles]
            return self._await_input(
                job_id=job_id,
                artifacts=artifacts,
                reason="The link did not resolve every required evidence role",
                required=missing,
            )
        council, batch, conflict = self._context(job, discovery)
        if conflict:
            return self._await_input(
                job_id=job_id,
                artifacts=artifacts,
                reason=conflict,
                required=["consistent council", "consistent batch"],
            )
        if not council or not batch:
            return self._await_input(
                job_id=job_id,
                artifacts=artifacts,
                reason="Council and batch are not declared by the job or link manifest",
                required=["council", "batch"],
            )
        if (job.council, job.batch) != (council, batch):
            job = self.store.update_job(job_id, status=JobStatus.RUNNING, council=council, batch=batch)

        preparation_path = artifacts.resolve("prepared/preparation-report.json")

        def prepare() -> list[tuple[str, Path]]:
            report = prepare_evidence(
                job_id=job_id,
                council=council,
                batch=batch,
                discovery=discovery,
                artifacts=artifacts,
            )
            path = artifacts.write_immutable_json(
                "prepared/preparation-report.json",
                report.model_dump(mode="json"),
            )
            return [
                ("preparation_report", path),
                ("prepared_source", report.source_path),
                ("prepared_inventory", report.inventory_path),
                ("prepared_capture_rules", report.capture_rules_path),
            ]

        stage_runner._stage(
            job_id=job_id,
            artifacts=artifacts,
            stage="prepare",
            input_value={
                "discovery_sha256": sha256_file(discovery_path),
                "council": council,
                "batch": batch,
            },
            execute=prepare,
        )
        preparation = PreparationReport.model_validate_json(preparation_path.read_text(encoding="utf-8"))
        if not preparation.registry_ready:
            # Checked before compiling: no proposal can satisfy it, so retrying
            # against it only spends attempts on a precondition.
            raise ValueError(
                f"Batch {preparation.batch!r} is not registered for council "
                f"{preparation.council!r}. Add it to /data/{preparation.council}"
                "/file-matching/autonomous-batches.json to allow the autonomous path to map it."
            )

        spec_path = artifacts.resolve("spec/mapping-spec.json")

        def compile_spec() -> list[tuple[str, Path]]:
            if self.approved_spec is not None:
                approved = MappingSpec.model_validate_json(
                    self.approved_spec.read_text(encoding="utf-8")
                )
                if (approved.council, approved.batch) != (council, batch):
                    raise ValueError(
                        f"Approved spec is for {approved.council}/{approved.batch}, "
                        f"not {council}/{batch}"
                    )
                path = artifacts.write_immutable_json(
                    "spec/mapping-spec.json", approved.model_dump(mode="json")
                )
                source = artifacts.write_immutable_json(
                    "spec/approved-spec-source.json",
                    {
                        "approved_spec_path": str(self.approved_spec),
                        "approved_spec_sha256": sha256_file(self.approved_spec),
                        "compiled": False,
                    },
                )
                return [("approved_spec", path), ("approved_spec_source", source)]
            # The verifier measures a proposal against this job's own evidence,
            # so its findings are the most useful thing to hand back on a retry.
            def check(candidate: MappingSpec) -> list[str]:
                return list(
                    verify_mapping_spec(
                        job_id=job_id, spec=candidate, preparation=preparation
                    ).errors
                )

            _, paths = self.compiler.compile(
                report=preparation,
                artifacts=artifacts,
                verifier=check,
                max_attempts=self.compile_attempts,
            )
            return [("compiler_artifact", path) for path in paths]

        stage_runner._stage(
            job_id=job_id,
            artifacts=artifacts,
            stage="compile_spec",
            input_value={
                "preparation_sha256": sha256_file(preparation_path),
                "auth": "none" if self.approved_spec else "codex_chatgpt_oauth",
                "approved_spec_sha256": (
                    sha256_file(self.approved_spec) if self.approved_spec else ""
                ),
            },
            execute=compile_spec,
        )
        spec = MappingSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))

        verification_path = artifacts.resolve("verification/spec-verification.json")

        def verify_spec() -> list[tuple[str, Path]]:
            report = verify_mapping_spec(job_id=job_id, spec=spec, preparation=preparation)
            path = artifacts.write_immutable_json(
                "verification/spec-verification.json",
                report.model_dump(mode="json"),
            )
            return [("spec_verification", path)]

        stage_runner._stage(
            job_id=job_id,
            artifacts=artifacts,
            stage="verify_spec",
            input_value={
                "spec_sha256": sha256_file(spec_path),
                "preparation_sha256": sha256_file(preparation_path),
            },
            execute=verify_spec,
        )
        verification = SpecVerificationReport.model_validate_json(
            verification_path.read_text(encoding="utf-8")
        )
        if not verification.passed:
            return self._await_input(
                job_id=job_id,
                artifacts=artifacts,
                reason="The proposed MappingSpec failed independent verification",
                required=list(verification.errors),
            )

        try:
            stage_runner.run(
                job_id=job_id,
                source_csv=preparation.source_path,
                inventory_csv=preparation.inventory_path,
                capture_rules=preparation.capture_rules_path,
                spec_path=spec_path,
                stop_after="validation",
            )
        except JobPaused:
            pass

        mapping_path = artifacts.resolve("mapping/proposed-mapping.csv")
        audit_path = artifacts.resolve("mapping/mapping-audit.csv")
        replay_path = artifacts.resolve("verification/historical-replay.json")

        golden_directory = Path(__file__).resolve().parent / "golden"
        suite_paths = sorted(golden_directory.glob(f"{council}*.json"))

        def historical_replay() -> list[tuple[str, Path]]:
            results = tuple(
                run_replay(
                    mapping_path=mapping_path,
                    audit_path=audit_path,
                    suite=load_replay_suite(suite_path),
                )
                for suite_path in suite_paths
            )
            warnings = () if results else (f"No historical replay suite is registered for {council}",)
            report = HistoricalReplayReport(
                job_id=job_id,
                applicable_suites=len(results),
                passed=bool(results) and all(result.failed == 0 for result in results),
                results=results,
                warnings=warnings,
            )
            path = artifacts.write_immutable_json(
                "verification/historical-replay.json",
                report.model_dump(mode="json"),
            )
            return [("historical_replay", path)]

        stage_runner._stage(
            job_id=job_id,
            artifacts=artifacts,
            stage="historical_replay",
            input_value={
                "mapping_sha256": sha256_file(mapping_path),
                "audit_sha256": sha256_file(audit_path),
                "suites": {str(path): sha256_file(path) for path in suite_paths},
            },
            execute=historical_replay,
        )
        replay = HistoricalReplayReport.model_validate_json(replay_path.read_text(encoding="utf-8"))
        qa_report_path = artifacts.resolve("qa/content-qa-report.json")
        replay_ids = [
            case.oachargeid
            for suite_path in suite_paths
            for case in load_replay_suite(suite_path).cases
        ]

        def qa_sample() -> list[tuple[str, Path]]:
            report, queue_path = build_qa_queue(
                job_id=job_id,
                audit_path=audit_path,
                artifacts=artifacts,
                sample_size=40,
                include_ids=replay_ids,
            )
            path = artifacts.write_immutable_json(
                "qa/content-qa-report.json",
                report.model_dump(mode="json"),
            )
            return [("content_qa_queue", queue_path), ("content_qa_report", path)]

        stage_runner._stage(
            job_id=job_id,
            artifacts=artifacts,
            stage="qa_sample",
            input_value={
                "audit_sha256": sha256_file(audit_path),
                "sample_size": 40,
                "include_ids": replay_ids,
            },
            execute=qa_sample,
        )
        qa_report = ContentQaReport.model_validate_json(qa_report_path.read_text(encoding="utf-8"))
        policy_inputs = PublishPolicyInputs(
            staging_only=True,
            spec_verification_passed=verification.passed,
            negative_tests_passed=verification.gates.get("ambiguity_negative_tests", False),
            historical_regressions_passed=replay.passed,
            content_qa_passed=qa_report.passed,
            systematic_content_failures=qa_report.systematic_content_failures,
            target_unchanged=False,
        )
        return stage_runner.run(
            job_id=job_id,
            source_csv=preparation.source_path,
            inventory_csv=preparation.inventory_path,
            capture_rules=preparation.capture_rules_path,
            spec_path=spec_path,
            policy_inputs=policy_inputs,
        )
