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
from .preparation import sample_rows, value_distributions
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


APPROVED_SPECS_DIRNAME = "approved-specs"


def accepted_precedents(council: str, *, exclude_batch: str = "") -> list[dict[str, Any]]:
    """How this council's already-accepted specs spelled their joins.

    Every compile starts from nothing today, so each new work package
    rediscovers the council's own conventions. Exeter's Microfiche batch spent
    three attempts arriving at a folder shape that the accepted WP3 spec
    already recorded.

    Read from approved-specs rather than from any run that happened to pass
    verification. That directory holds only specs someone accepted after a
    holdout round, so a precedent is something that was measured to be right,
    not merely something that parsed.
    """
    directory = Path("/data") / council / "file-matching" / APPROVED_SPECS_DIRNAME
    if not directory.is_dir():
        return []
    precedents: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        if exclude_batch and path.stem == exclude_batch:
            continue
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        joins: list[dict[str, Any]] = []
        for route in spec.get("routes", []):
            derived = route.get("derived_key")
            if not derived:
                continue
            joins.append(
                {
                    "authoritative_key": route.get("authoritative_key"),
                    "inventory_key_field": route.get("inventory_key_field")
                    or spec.get("inventory_key_field"),
                    "source_templates": derived.get("source_templates"),
                    "inventory_templates": derived.get("inventory_templates"),
                    "inventory_match_mode": derived.get("inventory_match_mode"),
                    "key_parts": derived.get("key_parts"),
                    "part_normalizers": derived.get("part_normalizers"),
                }
            )
        if joins:
            precedents.append({"batch": spec.get("batch") or path.stem, "joins": joins})
    return precedents


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
            # Sample rows show the shape of a record; these show what values a
            # column can hold, which is what a route condition is written against.
            "value_counts": value_distributions(report.source_path),
        },
        "inventory": {
            "fields": list(report.inventory_fields),
            "row_count": report.inventory_rows,
            "sample_rows": sample_rows(report.inventory_path),
            "value_counts": value_distributions(report.inventory_path),
        },
        "capture_rule_chunks": [chunk.model_dump(mode="json") for chunk in report.capture_rule_chunks],
        # Specs already accepted for this council, so a new work package starts
        # from its conventions instead of rediscovering them.
        "accepted_precedents": accepted_precedents(report.council, exclude_batch=report.batch),
        "hard_constraints": {
            "no_path_construction": True,
            "accept_only_unique_inventory_candidate": True,
            "fallback_only_when_authoritative_blank": True,
            "automatic_confidence_cap": 0.74,
            "content_verified": False,
            "ambiguity_action": "reject",
        },
    }


def rejection_notes(attempts: Sequence[Sequence[str]], *, from_quality: int = 0) -> str:
    """Show the findings back to the compiler in their own words, newest last.

    Quality-round findings are labelled apart from verifier findings because
    they answer different questions. The verifier asks whether a proposal is
    coherent against the evidence; a quality round compared accepted mappings
    against the scans and asks whether it was right. A spec can satisfy the
    first completely and still match the wrong document.
    """
    if not attempts:
        return ""
    blocks = []
    for index, errors in enumerate(attempts, start=1):
        listed = "\n".join(f"  - {error}" for error in errors)
        if index <= from_quality:
            blocks.append(
                f"An earlier spec passed verification and was then found wrong by a quality round "
                f"that compared its accepted mappings against the scans themselves ({index}):\n{listed}"
            )
        else:
            blocks.append(
                f"Attempt {index - from_quality} was rejected by the independent verifier:\n{listed}"
            )
    joined = "\n\n".join(blocks)
    return f"""
PREVIOUS ATTEMPTS
{joined}

Those findings are measured against the real evidence, not style opinions. Fix the named cause rather
than rewriting the whole spec, and do not repeat a spelling that has already been rejected.
END PREVIOUS ATTEMPTS
"""


def precedent_notes(precedents: Sequence[dict[str, Any]]) -> str:
    """Render this council's accepted joins into the prompt itself.

    Carried in the packet first, where it changed nothing: two compiles read a
    packet containing Exeter WP3's folder templates and opened with the same
    naive whole-value key as the run that had no precedent at all. What does
    move the compiler is text in the prompt body -- the rejection notes are
    read and acted on -- so the precedent goes there too.
    """
    if not precedents:
        return ""
    blocks = []
    for precedent in precedents:
        for join in precedent["joins"]:
            blocks.append(
                f"  batch {precedent['batch']}, key {join.get('authoritative_key')!r} against "
                f"{join.get('inventory_key_field')!r}:\n"
                f"    source_templates:    {join.get('source_templates')}\n"
                f"    inventory_templates: {join.get('inventory_templates')}\n"
                f"    inventory_match_mode: {join.get('inventory_match_mode')!r}\n"
                f"    key_parts:           {join.get('key_parts')}\n"
                f"    part_normalizers:    {join.get('part_normalizers')}"
            )
    joined = "\n\n".join(blocks)
    return f"""
ACCEPTED FOR THIS COUNCIL ALREADY
{joined}

Each of those passed a holdout round, so it is measured evidence about how this council writes its
references and names its folders -- not a guess. Start from those shapes and adapt them to the values
you can see in this packet, rather than inventing a key from scratch. They say nothing about this
batch, so never cite one in place of a capture rule, and drop any that cannot parse the values in
front of you.
END ACCEPTED FOR THIS COUNCIL
"""


def compiler_prompt(
    packet_path: Path,
    packet: dict[str, Any] | None = None,
    rejected: Sequence[Sequence[str]] = (),
    from_quality: int = 0,
) -> str:
    embedded_packet = json.dumps(packet, ensure_ascii=False, indent=2) if packet is not None else ""
    previous = rejection_notes(rejected, from_quality=from_quality)
    precedents = precedent_notes((packet or {}).get("accepted_precedents", []))
    return f"""You are a capture-rules compiler for council planning-record mapping.

The compiler packet is embedded below and is also frozen at {packet_path}. Do not call tools or
execute commands; use the embedded packet directly. Return only one JSON object conforming to the supplied
MappingSpec schema. Your output is a proposal and will be rejected by an independent verifier.

Non-negotiable rules:
- Use exactly the packet council and batch. Never infer another batch.
- value_counts lists every value a low-cardinality column takes, with its row count. Account for all of
  them: give each value that has its own evidence source a route, and let the catch-all reject the rest
  deliberately. Copy values from value_counts exactly rather than from a sample row, because a condition
  written as "Aperture" matches nothing when the column says "Aperture cards".
- A route owns every row its conditions select. A key that fails to join does NOT fall through to a
  later route, so a second accepting route for the same population is unreachable dead weight. When
  one population writes its reference in more than one shape, put every shape in a single
  derived_key: list them as source_templates alternatives, list the folder shapes as
  inventory_templates alternatives, and give part_defaults for the parts a shorter shape omits.
- Routes are evaluated from the lowest priority number upwards, and the first match wins. Give accepting
  routes low numbers and put any catch-all reject route at the highest number, or the reject shadows
  everything and the mapping accepts nothing.
- accepted_precedents holds joins from specs already accepted for this council after a holdout
  round. A council names its folders one way across work packages, so start from a precedent's
  template shapes, normalizers and key_parts and adapt them to this packet's real values. They are
  evidence about the council, not about this batch: never cite one in place of a capture rule, and
  drop a precedent that does not parse the values you can see in the packet.
- Use only source and inventory field names listed in the packet.
- Translate capture-rule prose into ordered route predicates. Do not invent S3 paths or Portal URLs.
- S3 and Portal candidates must come from exact inventory rows through inventory key/path fields.
- In a mixed inventory, filter each accepting route with _evidence_role.
- Never condition on _artifact_id: preparation assigns it per download, so a
  spec carrying it rejects the next delivery of the same work package.
- Do not condition on a field whose value_counts show one value across every
  row. It selects everything it is shown, and silently rejects any future row
  that differs. Condition only on what actually separates the routes.
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
{precedents}{previous}"""


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
        from_quality: int = 0,
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
            input=compiler_prompt(packet_path, packet, rejected, from_quality=from_quality),
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
        prior_findings: Sequence[Sequence[str]] = (),
    ) -> tuple[MappingSpec, tuple[Path, ...]]:
        """Compile a spec, optionally retrying against the verifier's findings.

        The verifier measures a proposal against the real evidence -- whether its
        fields exist, whether its citations match frozen chunks, whether its
        derived key joins anything at all. Those findings are precise enough to
        act on, so a rejected attempt is fed back rather than thrown away.

        Retries are bounded. An unbounded loop would let the compiler search
        until something passes, which is a different thing from getting it right.

        prior_findings carries what an earlier spec was found wrong about after
        it had already passed verification -- a quality round comparing accepted
        mappings against the scans themselves. Verification asks whether a
        proposal is coherent against the evidence; a quality round asks whether
        it was right. Only the second can see a spec that joins cleanly and
        still matches the wrong document, so its findings belong in the prompt
        from the first attempt rather than being rediscovered.
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

        seeded = [list(finding) for finding in prior_findings if finding]
        rejected: list[list[str]] = list(seeded)
        produced: list[Path] = [packet_path, schema_path]
        last_errors: list[str] = []
        for attempt in range(1, max(1, max_attempts) + 1):
            try:
                spec, raw_output, log_path = self._attempt(
                    attempt=attempt,
                    packet_path=packet_path,
                    packet=packet,
                    schema_path=schema_path,
                    artifacts=artifacts,
                    rejected=rejected,
                    from_quality=len(seeded),
                )
            except CompilerError as exc:
                # A proposal that will not even parse is the kind of mistake
                # feedback fixes best -- a malformed template, a part named
                # twice. Retrying it is the same loop as a verification failure.
                if attempt >= max(1, max_attempts):
                    raise
                last_errors = [str(exc)]
                rejected.append(last_errors)
                continue
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
