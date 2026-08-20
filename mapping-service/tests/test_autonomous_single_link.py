from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


AUTONOMOUS_ROOT = Path(__file__).resolve().parents[1] / "amazons3-mapping"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(AUTONOMOUS_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTONOMOUS_ROOT))

from autonomous.compiler import (  # noqa: E402
    CodexOAuthCompiler,
    CompilerError,
    normalize_schema_boilerplate,
    strict_output_schema,
)
from autonomous.ingestion import (  # noqa: E402
    IngestionError,
    classify_artifact,
    discover_link,
    resolve_reference,
    validate_remote_url,
)
from autonomous.qa import select_stratified_rows, stratum  # noqa: E402
from autonomous.schemas import (  # noqa: E402
    ArtifactRole,
    EvidenceCitation,
    JobRequest,
    JobStatus,
    MappingSpec,
    PreparationReport,
)
from autonomous.single_link import SingleLinkRunner  # noqa: E402
from autonomous.spec_verifier import verify_mapping_spec  # noqa: E402
from autonomous.storage import ArtifactStore, JobStore  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    path: Path,
    *,
    council: str = "sheffield",
    batch: str = "wp3",
    artifacts: list[dict[str, str]] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "kind": "council-mapping-job",
                "schema_version": 1,
                "council": council,
                "batch": batch,
                "artifacts": artifacts
                or [
                    {"url": "source.csv", "role": "source_records"},
                    {"url": "Capture Rules.txt", "role": "capture_rules"},
                    {"url": "inventory.csv", "role": "s3_inventory"},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def spec_for(report: PreparationReport) -> MappingSpec:
    chunk = report.capture_rule_chunks[0]
    return MappingSpec.model_validate(
        {
            "schema_version": 1,
            "spec_id": "oauth-proposed-sheffield-wp3-v1",
            "council": report.council,
            "batch": report.batch,
            "source_id_field": "oachargeid",
            "inventory_key_field": "candidate_key",
            "inventory_path_field": "candidate_path",
            "routes": [
                {
                    "rule_id": "source-reference-exact",
                    "priority": 10,
                    "conditions": [{"operator": "always"}],
                    "target": "s3",
                    "authoritative_key": "source_reference",
                    "fallback_key": "planning_reference",
                    "fallback_only_when_authoritative_blank": True,
                    "normalizers": ["trim", "casefold", "slash_to_hyphen"],
                    "automatic_confidence": 0.74,
                    "content_verified": False,
                    "ambiguity_action": "reject",
                    "citations": [
                        EvidenceCitation(
                            artifact_id=chunk.artifact_id,
                            location=chunk.location,
                            statement="Use the source reference as the archive filename key.",
                            excerpt_sha256=chunk.excerpt_sha256,
                        ).model_dump(mode="json")
                    ],
                }
            ],
        }
    )


class FakeCompiler:
    """Mirrors the real compiler's signature, including verifier-driven retry."""

    def __init__(self, reject_first: int = 0) -> None:
        self.calls = 0
        self.reject_first = reject_first
        self.feedback: list[list[str]] = []

    def compile(self, *, report: PreparationReport, artifacts: ArtifactStore,
                verifier=None, max_attempts: int = 1):
        rejected: list[list[str]] = []
        for attempt in range(1, max(1, max_attempts) + 1):
            self.calls += 1
            spec = spec_for(report)
            errors = (
                ["synthetic rejection"] if attempt <= self.reject_first else
                (list(verifier(spec)) if verifier else [])
            )
            if not errors:
                path = artifacts.write_immutable_json(
                    "spec/mapping-spec.json", spec.model_dump(mode="json")
                )
                self.feedback = rejected
                return spec, (path,)
            rejected.append(errors)
        self.feedback = rejected
        raise CompilerError(f"No proposal passed verification in {len(rejected)} attempt(s)")


class IngestionTest(unittest.TestCase):
    def test_blocks_loopback_remote_target(self) -> None:
        with self.assertRaises(IngestionError):
            validate_remote_url("http://127.0.0.1/private")

    def test_classifier_distinguishes_s3_and_portal_evidence(self) -> None:
        self.assertEqual(classify_artifact("Council S3 Folder Index.csv")[0], ArtifactRole.S3_INVENTORY)
        self.assertEqual(classify_artifact("Public Access Portal Evidence.jsonl")[0], ArtifactRole.PORTAL_EVIDENCE)

    def test_manifest_cannot_escape_local_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence"
            evidence.mkdir()
            (root / "outside.csv").write_text("x\n1\n", encoding="utf-8")
            manifest = evidence / "mapping-job.json"
            write_manifest(
                manifest,
                artifacts=[
                    {"url": "../outside.csv", "role": "source_records"},
                    {"url": "rules.txt", "role": "capture_rules"},
                    {"url": "inventory.csv", "role": "s3_inventory"},
                ],
            )
            with self.assertRaises(IngestionError):
                discover_link(
                    job_id="traversal",
                    root_url=manifest.as_uri(),
                    artifacts=ArtifactStore(root / "workspace"),
                )

    def test_s3_manifest_relative_reference_is_resolved_without_guessing(self) -> None:
        self.assertEqual(
            resolve_reference("s3://evidence/jobs/mapping-job.json", "source.csv"),
            "s3://evidence/jobs/source.csv",
        )

    def test_directory_heuristics_build_complete_evidence_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Council Source Information.csv").write_text("oachargeid\n1\n", encoding="utf-8")
            (root / "Capture Rules.txt").write_text("Use source reference.\n", encoding="utf-8")
            (root / "Council S3 Folder Index.csv").write_text(
                "candidate_key,candidate_path\nA,s3://evidence/a\n", encoding="utf-8"
            )
            result = discover_link(
                job_id="directory",
                root_url=root.as_uri(),
                artifacts=ArtifactStore(root / "workspace"),
            )
            self.assertTrue(result.evidence_complete)
            self.assertEqual(
                {artifact.role for artifact in result.artifacts},
                {ArtifactRole.SOURCE_RECORDS, ArtifactRole.CAPTURE_RULES, ArtifactRole.S3_INVENTORY},
            )


class CompilerTest(unittest.TestCase):
    def test_schema_boilerplate_normalization_does_not_change_mapping_fields(self) -> None:
        payload = {
            "routes": [
                {
                    "authoritative_key": "source_reference",
                    "conditions": [{"field": "oachargeid", "operator": "always", "value": None}],
                    "inventory_conditions": [],
                    "fallback_key": "",
                    "inventory_key_field": "",
                    "inventory_path_field": "",
                }
            ]
        }
        normalized = normalize_schema_boilerplate(payload)
        self.assertEqual(normalized["routes"][0]["conditions"][0]["field"], "")
        self.assertEqual(normalized["routes"][0]["authoritative_key"], "source_reference")
        self.assertIsNone(normalized["routes"][0]["fallback_key"])

    def test_strict_schema_requires_every_object_property(self) -> None:
        schema = strict_output_schema(MappingSpec.model_json_schema(mode="validation"))

        def inspect(value):
            if isinstance(value, dict):
                if isinstance(value.get("properties"), dict):
                    self.assertEqual(set(value["required"]), set(value["properties"]))
                    self.assertFalse(value["additionalProperties"])
                for item in value.values():
                    inspect(item)
            elif isinstance(value, list):
                for item in value:
                    inspect(item)

        inspect(schema)

    def test_codex_compiler_requires_oauth_and_read_only_ephemeral_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            inventory = root / "inventory.csv"
            rules = root / "rules.txt"
            write_csv(source, [{"oachargeid": "1", "source_reference": "A", "planning_reference": ""}])
            write_csv(inventory, [{"candidate_key": "A", "candidate_path": "s3://evidence/a"}])
            rules.write_text("Use source reference.\n", encoding="utf-8")
            from autonomous.schemas import RuleChunk

            chunk = RuleChunk(
                location="chunk:1",
                text="Use source reference.",
                excerpt_sha256="11661e8dbe5e9e517925547501d622478d86b3c8c888e6a453f7a32f2dbb2b03",
            )
            report = PreparationReport(
                job_id="compiler",
                council="sheffield",
                batch="wp3",
                registry_council_known=True,
                registry_batch_known=True,
                source_path=source,
                inventory_path=inventory,
                capture_rules_path=rules,
                source_rows=1,
                inventory_rows=1,
                source_fields=("oachargeid", "source_reference", "planning_reference"),
                inventory_fields=("candidate_key", "candidate_path", "_evidence_role"),
                inventory_roles=(ArtifactRole.S3_INVENTORY,),
                capture_rule_chunks=(chunk,),
            )
            commands: list[list[str]] = []
            environments: list[dict[str, str]] = []

            def fake_run(command: list[str], **kwargs):
                commands.append(command)
                if command[:3] == ["codex", "login", "status"]:
                    return subprocess.CompletedProcess(command, 0, stdout="Logged in using ChatGPT\n")
                environments.append(kwargs["env"])
                output = Path(command[command.index("--output-last-message") + 1])
                output.write_text(spec_for(report).model_dump_json(indent=2), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

            compiler = CodexOAuthCompiler(
                repository_root=REPOSITORY_ROOT,
                command_runner=fake_run,
                command_isolator=lambda command: command,
            )
            with patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "must-not-be-forwarded", "CODEX_API_KEY": "must-not-be-forwarded"},
            ):
                spec, _ = compiler.compile(report=report, artifacts=ArtifactStore(root / "job"))
            self.assertEqual(spec.council, "sheffield")
            execution = commands[1]
            self.assertIn("--ephemeral", execution)
            self.assertEqual(execution[execution.index("--sandbox") + 1], "read-only")
            self.assertNotIn("OPENAI_API_KEY", " ".join(execution))
            self.assertNotIn("OPENAI_API_KEY", environments[0])
            self.assertNotIn("CODEX_API_KEY", environments[0])


class QaSamplingTest(unittest.TestCase):
    def test_sampling_is_deterministic_and_covers_every_stratum(self) -> None:
        rows = [
            {
                "oachargeid": "accepted-a",
                "route": "s3",
                "match_status": "accepted",
                "match_basis": "authoritative",
                "decision_confidence": "0.74",
                "candidate_count": "1",
            },
            {
                "oachargeid": "accepted-b",
                "route": "s3",
                "match_status": "accepted",
                "match_basis": "authoritative",
                "decision_confidence": "0.74",
                "candidate_count": "1",
            },
            {
                "oachargeid": "rejected",
                "route": "portal",
                "match_status": "rejected_ambiguous_multiple_candidates",
                "match_basis": "authoritative",
                "decision_confidence": "0.00",
                "candidate_count": "2",
            },
            {
                "oachargeid": "missing",
                "route": "none",
                "match_status": "unmatched_no_candidate",
                "match_basis": "none",
                "decision_confidence": "0.00",
                "candidate_count": "0",
            },
        ]
        first = select_stratified_rows(rows, seed="job", sample_size=1, include_ids=("accepted-b",))
        second = select_stratified_rows(rows, seed="job", sample_size=1, include_ids=("accepted-b",))
        self.assertEqual(first, second)
        self.assertEqual({stratum(row) for row in first}, {stratum(row) for row in rows})
        self.assertIn("accepted-b", {row["oachargeid"] for row in first})


class SingleLinkEndToEndTest(unittest.TestCase):
    def test_one_link_reaches_staging_without_mutating_production(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence"
            evidence.mkdir()
            write_csv(
                evidence / "source.csv",
                [
                    {
                        "oachargeid": "103512",
                        "source_reference": "93/0063T",
                        "planning_reference": "93/0063P",
                    }
                ],
            )
            write_csv(
                evidence / "inventory.csv",
                [{"candidate_key": "93/0063P", "candidate_path": "s3://evidence/93-0063P"}],
            )
            (evidence / "Capture Rules.txt").write_text(
                "Use source_reference as the archive filename key.\n",
                encoding="utf-8",
            )
            manifest = evidence / "mapping-job.json"
            write_manifest(manifest)

            store = JobStore(root / "jobs")
            job = store.create_job(JobRequest(source_url=manifest.as_uri()), job_id="single_link")
            fake_compiler = FakeCompiler()
            runner = SingleLinkRunner(
                store,
                repository_root=REPOSITORY_ROOT,
                compiler=fake_compiler,
            )
            result = runner.run(job_id=job.job_id)
            self.assertEqual(result.status, JobStatus.COMPLETED_STAGED)
            self.assertEqual(fake_compiler.calls, 1)
            mapping = (result.workspace / "mapping/proposed-mapping.csv").read_text(encoding="utf-8")
            self.assertIn("103512,,0.00,", mapping)
            replay = json.loads(
                (result.workspace / "verification/historical-replay.json").read_text(encoding="utf-8")
            )
            decision = json.loads(
                (result.workspace / "publish/publish-decision.json").read_text(encoding="utf-8")
            )
            qa_report = json.loads(
                (result.workspace / "qa/content-qa-report.json").read_text(encoding="utf-8")
            )
            self.assertTrue(replay["passed"])
            self.assertFalse(qa_report["passed"])
            self.assertEqual(qa_report["selected_sample_size"], 1)
            self.assertTrue((result.workspace / "qa/content-review-queue.csv").is_file())
            self.assertFalse(decision["allowed"])
            self.assertIn("not_staging_only", decision["failed_gates"])
            self.assertIn("content_qa_passed", decision["failed_gates"])

            resumed = runner.run(job_id=job.job_id)
            self.assertEqual(resumed.status, JobStatus.COMPLETED_STAGED)
            self.assertEqual(fake_compiler.calls, 1)

    def test_verifier_rejects_invented_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            inventory = root / "inventory.csv"
            rules = root / "rules.txt"
            write_csv(source, [{"oachargeid": "1", "source_reference": "A", "planning_reference": ""}])
            write_csv(inventory, [{"candidate_key": "A", "candidate_path": "s3://evidence/a"}])
            rules.write_text("Use source reference.\n", encoding="utf-8")
            from autonomous.schemas import RuleChunk

            chunk = RuleChunk(
                location="chunk:1",
                text="Use source reference.",
                excerpt_sha256="11661e8dbe5e9e517925547501d622478d86b3c8c888e6a453f7a32f2dbb2b03",
            )
            report = PreparationReport(
                job_id="verify",
                council="sheffield",
                batch="wp3",
                registry_council_known=True,
                registry_batch_known=True,
                source_path=source,
                inventory_path=inventory,
                capture_rules_path=rules,
                source_rows=1,
                inventory_rows=1,
                source_fields=("oachargeid", "source_reference", "planning_reference"),
                inventory_fields=("candidate_key", "candidate_path"),
                inventory_roles=(ArtifactRole.S3_INVENTORY,),
                capture_rule_chunks=(chunk,),
            )
            payload = spec_for(report).model_dump(mode="json")
            payload["routes"][0]["authoritative_key"] = "invented_reference"
            result = verify_mapping_spec(
                job_id="verify",
                spec=MappingSpec.model_validate(payload),
                preparation=report,
            )
            self.assertFalse(result.passed)
            self.assertTrue(any("invented_reference" in error for error in result.errors))


if __name__ == "__main__":
    unittest.main()
