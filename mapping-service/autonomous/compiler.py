from __future__ import annotations

import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Sequence

from .path_policy import (
    file_browser_isolated_command,
    read_only_job_isolation_active,
    require_unprotected_path,
)
from .preparation import sample_rows
from .schemas import MappingSpec, PreparationReport
from .storage import ArtifactStore


class CompilerError(RuntimeError):
    pass


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def strict_output_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [strict_output_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {
        key: strict_output_schema(item)
        for key, item in value.items()
        if key not in {"default", "title"}
    }
    properties = result.get("properties")
    if isinstance(properties, dict):
        result["additionalProperties"] = False
        result["required"] = list(properties)
    return result


def normalize_schema_boilerplate(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize only representation details that carry no mapping semantics."""
    result = deepcopy(value)
    for route in result.get("routes", []):
        for collection in ("conditions", "inventory_conditions"):
            for predicate in route.get(collection, []):
                operator = predicate.get("operator")
                if operator == "always":
                    predicate["field"] = ""
                    predicate["value"] = None
                elif operator in {"is_blank", "not_blank"}:
                    predicate["value"] = None
        for optional_field in ("fallback_key", "inventory_key_field", "inventory_path_field"):
            if route.get(optional_field) == "":
                route[optional_field] = None
    return result


def load_compiled_spec(path: Path) -> MappingSpec:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("root output is not an object")
        return MappingSpec.model_validate(normalize_schema_boilerplate(payload))
    except Exception as exc:
        raise CompilerError(f"Codex output is not a valid MappingSpec: {exc}") from exc


def compiler_packet(report: PreparationReport) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "job_id": report.job_id,
        "council": report.council,
        "batch": report.batch,
        "source": {
            "fields": list(report.source_fields),
            "row_count": report.source_rows,
            "sample_rows": sample_rows(report.source_path),
        },
        "inventory": {
            "fields": list(report.inventory_fields),
            "row_count": report.inventory_rows,
            "sample_rows": sample_rows(report.inventory_path),
        },
        "capture_rule_chunks": [chunk.model_dump(mode="json") for chunk in report.capture_rule_chunks],
        "hard_constraints": {
            "no_path_construction": True,
            "accept_only_unique_inventory_candidate": True,
            "fallback_only_when_authoritative_blank": True,
            "automatic_confidence_cap": 0.74,
            "content_verified": False,
            "ambiguity_action": "reject",
        },
    }


def rejection_notes(attempts: Sequence[Sequence[str]]) -> str:
    """Show the verifier's own words back to the compiler, newest last."""
    if not attempts:
        return ""
    blocks = []
    for index, errors in enumerate(attempts, start=1):
        listed = "\n".join(f"  - {error}" for error in errors)
        blocks.append(f"Attempt {index} was rejected by the independent verifier:\n{listed}")
    joined = "\n\n".join(blocks)
    return f"""
PREVIOUS ATTEMPTS
{joined}

Those errors are measured against the real evidence in the packet, not style opinions. Fix the named
cause rather than rewriting the whole spec, and do not repeat a spelling the verifier has already
rejected.
END PREVIOUS ATTEMPTS
"""


def compiler_prompt(
    packet_path: Path,
    packet: dict[str, Any] | None = None,
    rejected: Sequence[Sequence[str]] = (),
) -> str:
    embedded_packet = json.dumps(packet, ensure_ascii=False, indent=2) if packet is not None else ""
    previous = rejection_notes(rejected)
    return f"""You are a capture-rules compiler for council planning-record mapping.

The compiler packet is embedded below and is also frozen at {packet_path}. Do not call tools or
execute commands; use the embedded packet directly. Return only one JSON object conforming to the supplied
MappingSpec schema. Your output is a proposal and will be rejected by an independent verifier.

Non-negotiable rules:
- Use exactly the packet council and batch. Never infer another batch.
- Routes are evaluated from the lowest priority number upwards, and the first match wins. Give accepting
  routes low numbers and put any catch-all reject route at the highest number, or the reject shadows
  everything and the mapping accepts nothing.
- Use only source and inventory field names listed in the packet.
- Translate capture-rule prose into ordered route predicates. Do not invent S3 paths or Portal URLs.
- S3 and Portal candidates must come from exact inventory rows through inventory key/path fields.
- In a mixed inventory, filter each accepting route with _evidence_role.
- An authoritative field may use a fallback only when the authoritative value is blank.
- Retain ambiguity_action=reject, content_verified=false, and automatic_confidence <= 0.74.
- Every accepting route must cite one or more capture_rule_chunks. Copy artifact_id, location, and
  excerpt_sha256 exactly from a cited chunk and write a concise statement of what that chunk supports.
- If the evidence cannot justify an accepting route, emit an explicit reject route instead of guessing.
- When the source reference and the inventory key are written differently -- different separators,
  leading zeros, a prefix or year the reference does not carry -- set derived_key instead of relying on
  normalizers. Whole-field normalizers cannot rebuild a key from parts. Declare source_templates and
  inventory_templates over the same named parts, list in key_parts only the parts that genuinely identify
  the record, and give per-part normalizers such as strip_zeros or pad:5 as part_normalizers entries,
  each naming one part and the normalizers to apply to it. Per-part normalizer names are the same
  vocabulary as the route normalizers, plus strip_zeros, year2to4 and pad:N. Give each side
  more than one template when references arrive in shape variants. Use inventory_match_mode "prefix" when
  inventory entries append free text after the key. A part written {{name:d}} matches digits, {{name:a}}
  matches letters and digits, and a bare {{name}} runs up to the next literal character.
- Do not put a part in key_parts unless both sides mean the same thing by it. For example a reference
  88/1061/FUL and its folder EXE_1988_88-1061-02 share a year and a number, but FUL is an application
  type while 02 is a document type: key_parts must be the year and number only. A part that means two
  different things joins nothing, and the spec will be rejected for joining nothing.
- A part name carries one set of normalizers on both sides. year2to4 accepts a two- or four-digit year
  and returns four, so naming both sides "year" and normalizing it with year2to4 is correct when the
  reference writes two digits and the folder writes four.
- Conditions are ANDed. Use an always predicate only for a genuine catch-all route.
- For an always predicate set field to the empty string and value to null. For is_blank and not_blank,
  set value to null.

BEGIN COMPILER PACKET
{embedded_packet}
END COMPILER PACKET
{previous}"""


class CodexOAuthCompiler:
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
            operation="use compiler repository root",
        )
        self.command_runner = command_runner
        self.command_isolator = command_isolator
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _oauth_environment() -> dict[str, str]:
        environment = os.environ.copy()
        for variable in ("OPENAI_API_KEY", "CODEX_API_KEY"):
            environment.pop(variable, None)
        return environment

    def _require_chatgpt_login(self) -> None:
        result = self.command_runner(
            ["codex", "login", "status"],
            env=self._oauth_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        if result.returncode != 0 or "Logged in using ChatGPT" not in (result.stdout or ""):
            raise CompilerError(
                "Codex OAuth compiler requires `codex login` with ChatGPT; API-key fallback is disabled"
            )

    def _attempt(
        self,
        *,
        attempt: int,
        packet_path: Path,
        packet: dict[str, Any],
        schema_path: Path,
        artifacts: ArtifactStore,
        rejected: list[list[str]],
    ) -> tuple[MappingSpec, Path, Path]:
        """Run one compilation and return its spec, raw output and log."""
        root = f"compiler/attempt-{attempt:02d}"
        attempt_output = artifacts.resolve(f"{root}/mapping-spec.attempt.json")
        attempt_output.parent.mkdir(parents=True, exist_ok=True)
        if attempt_output.exists():
            attempt_output.unlink()
        # n8n runs the entire job in one stricter outer sandbox whose root is
        # read-only and whose only writable mount is /data/mapping-jobs. Do not
        # try to nest bubblewrap there; Codex's "danger-full-access" is still
        # bounded by that outer read-only mount namespace.
        sandbox_mode = "danger-full-access" if read_only_job_isolation_active() else "read-only"
        command = [
            "codex",
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            sandbox_mode,
            "--color",
            "never",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(attempt_output),
            "--cd",
            str(self.repository_root),
            "-",
        ]
        result = self.command_runner(
            self.command_isolator(command),
            env=self._oauth_environment(),
            input=compiler_prompt(packet_path, packet, rejected),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_seconds,
            check=False,
        )
        log_text = (
            f"returncode={result.returncode}\n\nSTDOUT\n{(result.stdout or '')[-100000:]}"
            f"\n\nSTDERR\n{(result.stderr or '')[-100000:]}\n"
        )
        log_path = artifacts.write_mutable(f"{root}/codex-compiler.log", log_text.encode("utf-8"))
        if result.returncode != 0:
            if attempt_output.exists():
                artifacts.write_mutable(f"{root}/mapping-spec.failed.json", attempt_output.read_bytes())
                attempt_output.unlink()
            raise CompilerError(
                f"Codex compiler failed with exit code {result.returncode}; inspect {log_path}"
            )
        if not attempt_output.is_file():
            raise CompilerError("Codex compiler completed without writing its structured output")
        try:
            spec = load_compiled_spec(attempt_output)
        except CompilerError:
            failed_path = artifacts.write_mutable(
                f"{root}/mapping-spec.failed.json", attempt_output.read_bytes()
            )
            attempt_output.unlink()
            raise CompilerError(f"Codex output failed MappingSpec validation; inspect {failed_path}")
        raw_output = artifacts.write_immutable(
            f"{root}/mapping-spec.raw.json", attempt_output.read_bytes()
        )
        attempt_output.unlink()
        return spec, raw_output, log_path

    def compile(
        self,
        *,
        report: PreparationReport,
        artifacts: ArtifactStore,
        verifier: Callable[[MappingSpec], list[str]] | None = None,
        max_attempts: int = 1,
    ) -> tuple[MappingSpec, tuple[Path, ...]]:
        """Compile a spec, optionally retrying against the verifier's findings.

        The verifier measures a proposal against the real evidence -- whether its
        fields exist, whether its citations match frozen chunks, whether its
        derived key joins anything at all. Those findings are precise enough to
        act on, so a rejected attempt is fed back rather than thrown away.

        Retries are bounded. An unbounded loop would let the compiler search
        until something passes, which is a different thing from getting it right.
        """
        self._require_chatgpt_login()
        packet = compiler_packet(report)
        packet_path = artifacts.write_immutable_json("compiler/compiler-packet.json", packet)
        schema_path = artifacts.write_immutable_json(
            "compiler/mapping-spec.schema.json",
            strict_output_schema(MappingSpec.model_json_schema(mode="validation")),
        )

        canonical_path = artifacts.resolve("spec/mapping-spec.json")
        if canonical_path.exists():
            # Resuming a job that already settled on a spec.
            return load_compiled_spec(canonical_path), (packet_path, schema_path, canonical_path)

        rejected: list[list[str]] = []
        produced: list[Path] = [packet_path, schema_path]
        last_errors: list[str] = []
        for attempt in range(1, max(1, max_attempts) + 1):
            spec, raw_output, log_path = self._attempt(
                attempt=attempt,
                packet_path=packet_path,
                packet=packet,
                schema_path=schema_path,
                artifacts=artifacts,
                rejected=rejected,
            )
            produced.extend((raw_output, log_path))
            errors = verifier(spec) if verifier else []
            if not errors:
                canonical = artifacts.write_immutable_json(
                    "spec/mapping-spec.json", spec.model_dump(mode="json")
                )
                if rejected:
                    produced.append(
                        artifacts.write_mutable_json(
                            "compiler/attempts.json",
                            {"attempts": attempt, "rejected": rejected},
                        )
                    )
                produced.append(canonical)
                return spec, tuple(produced)
            rejected.append(errors)
            last_errors = errors

        produced.append(
            artifacts.write_mutable_json(
                "compiler/attempts.json", {"attempts": len(rejected), "rejected": rejected}
            )
        )
        raise CompilerError(
            f"No proposal passed verification in {len(rejected)} attempt(s). "
            f"Last findings: {'; '.join(last_errors)[:600]}"
        )
