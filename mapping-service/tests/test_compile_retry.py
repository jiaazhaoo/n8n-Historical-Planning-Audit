from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

AUTONOMOUS_ROOT = Path(__file__).resolve().parents[1] / "amazons3-mapping"
if str(AUTONOMOUS_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTONOMOUS_ROOT))

from autonomous.compiler import (  # noqa: E402
    CodexOAuthCompiler,
    CompilerError,
    compiler_prompt,
    rejection_notes,
)
from autonomous.schemas import MappingSpec, PreparationReport, RuleChunk  # noqa: E402
from autonomous.storage import ArtifactStore  # noqa: E402


CHUNK = RuleChunk(
    artifact_id="capture_rules",
    location="chunk:1",
    text="Folders are named EXE_<year>_<yy>-<number>-<code>.",
    excerpt_sha256="a" * 64,
)


def preparation(tmp: Path) -> PreparationReport:
    for name in ("source.csv", "inventory.csv", "rules.txt"):
        (tmp / name).write_text("oachargeid\n1\n", encoding="utf-8")
    return PreparationReport(
        job_id="job-1",
        council="exeter",
        batch="wp3",
        registry_council_known=True,
        registry_batch_known=True,
        source_path=tmp / "source.csv",
        inventory_path=tmp / "inventory.csv",
        capture_rules_path=tmp / "rules.txt",
        source_rows=1,
        inventory_rows=1,
        source_fields=("oachargeid", "reference"),
        inventory_fields=("folder", "amazons3_path"),
        inventory_roles=(),
        capture_rule_chunks=(CHUNK,),
    )


def spec_json(spec_id: str) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "spec_id": spec_id,
            "council": "exeter",
            "batch": "wp3",
            "source_id_field": "oachargeid",
            "inventory_key_field": "folder",
            "inventory_path_field": "amazons3_path",
            "routes": [
                {
                    "rule_id": "reject_all",
                    "priority": 100,
                    "conditions": [{"operator": "always"}],
                    "target": "reject",
                }
            ],
        }
    )


class ScriptedCodex:
    """Stands in for the codex CLI, writing one structured output per call."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, command, **kwargs):
        self.prompts.append(kwargs.get("input", ""))
        output = Path(command[command.index("--output-last-message") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(spec_json(f"attempt-{len(self.prompts)}"), encoding="utf-8")
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")


def compiler(runner) -> CodexOAuthCompiler:
    instance = CodexOAuthCompiler(repository_root=Path("/env/code/file-matching"))
    instance.command_runner = runner
    instance.command_isolator = lambda command: command
    instance._require_chatgpt_login = lambda: None
    return instance


class RejectionNotesTests(unittest.TestCase):
    def test_no_attempts_adds_nothing(self) -> None:
        self.assertEqual(rejection_notes([]), "")

    def test_each_attempt_is_shown_with_its_findings(self) -> None:
        notes = rejection_notes([["joins nothing"], ["citation missing"]])
        self.assertIn("Attempt 1", notes)
        self.assertIn("joins nothing", notes)
        self.assertIn("Attempt 2", notes)
        self.assertIn("citation missing", notes)

    def test_the_prompt_carries_the_findings(self) -> None:
        prompt = compiler_prompt(Path("/tmp/packet.json"), {"a": 1}, [["derived key joins none"]])
        self.assertIn("PREVIOUS ATTEMPTS", prompt)
        self.assertIn("derived key joins none", prompt)

    def test_a_first_attempt_prompt_has_no_feedback_section(self) -> None:
        self.assertNotIn("PREVIOUS ATTEMPTS", compiler_prompt(Path("/tmp/packet.json"), {"a": 1}))


class CompileRetryTests(unittest.TestCase):
    def test_a_passing_first_attempt_does_not_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            runner = ScriptedCodex()
            spec, _ = compiler(runner).compile(
                report=preparation(tmp),
                artifacts=ArtifactStore(tmp / "job"),
                verifier=lambda candidate: [],
                max_attempts=3,
            )
            self.assertEqual(len(runner.prompts), 1)
            self.assertEqual(spec.spec_id, "attempt-1")

    def test_a_rejected_attempt_is_retried_with_the_findings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            runner = ScriptedCodex()
            calls = {"n": 0}

            def verifier(candidate: MappingSpec) -> list[str]:
                calls["n"] += 1
                return ["the derived key joins none of 759 routed cases"] if calls["n"] == 1 else []

            spec, _ = compiler(runner).compile(
                report=preparation(tmp),
                artifacts=ArtifactStore(tmp / "job"),
                verifier=verifier,
                max_attempts=3,
            )
            self.assertEqual(len(runner.prompts), 2)
            self.assertEqual(spec.spec_id, "attempt-2")
            # The second prompt must contain what the verifier said about the first.
            self.assertIn("joins none of 759", runner.prompts[1])
            self.assertNotIn("PREVIOUS ATTEMPTS", runner.prompts[0])

    def test_retries_are_bounded_and_report_the_last_findings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            runner = ScriptedCodex()
            with self.assertRaises(CompilerError) as caught:
                compiler(runner).compile(
                    report=preparation(tmp),
                    artifacts=ArtifactStore(tmp / "job"),
                    verifier=lambda candidate: ["still wrong"],
                    max_attempts=3,
                )
            self.assertEqual(len(runner.prompts), 3)
            self.assertIn("still wrong", str(caught.exception))
            self.assertIn("3 attempt", str(caught.exception))

    def test_every_attempt_is_kept_for_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            store = ArtifactStore(tmp / "job")
            with self.assertRaises(CompilerError):
                compiler(ScriptedCodex()).compile(
                    report=preparation(tmp),
                    artifacts=store,
                    verifier=lambda candidate: ["no"],
                    max_attempts=2,
                )
            self.assertTrue(store.resolve("compiler/attempt-01/mapping-spec.raw.json").is_file())
            self.assertTrue(store.resolve("compiler/attempt-02/mapping-spec.raw.json").is_file())
            recorded = json.loads(store.resolve("compiler/attempts.json").read_text())
            self.assertEqual(recorded["attempts"], 2)

    def test_a_settled_spec_is_reused_instead_of_recompiled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            store = ArtifactStore(tmp / "job")
            store.write_immutable_json("spec/mapping-spec.json", json.loads(spec_json("settled")))
            runner = ScriptedCodex()
            spec, _ = compiler(runner).compile(
                report=preparation(tmp), artifacts=store, max_attempts=3
            )
            self.assertEqual(spec.spec_id, "settled")
            self.assertEqual(runner.prompts, [])

    def test_without_a_verifier_the_first_attempt_stands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            runner = ScriptedCodex()
            spec, _ = compiler(runner).compile(
                report=preparation(tmp), artifacts=ArtifactStore(tmp / "job"), max_attempts=3
            )
            self.assertEqual(len(runner.prompts), 1)
            self.assertEqual(spec.spec_id, "attempt-1")


if __name__ == "__main__":
    unittest.main()
