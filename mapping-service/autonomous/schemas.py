from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .key_derivation import DerivationError, KeyDerivation, Template


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


NonEmpty = Annotated[str, Field(min_length=1)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactRole(str, Enum):
    SOURCE_RECORDS = "source_records"
    CAPTURE_RULES = "capture_rules"
    SOURCE_BREAKDOWN = "source_breakdown"
    DELIVERY_REPORT = "delivery_report"
    S3_INVENTORY = "s3_inventory"
    PORTAL_EVIDENCE = "portal_evidence"
    UNKNOWN = "unknown"


class EvidenceArtifact(StrictModel):
    artifact_id: NonEmpty
    uri: NonEmpty
    local_path: Path
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    size_bytes: Annotated[int, Field(ge=0)]
    media_type: NonEmpty
    role: ArtifactRole
    discovered_from: NonEmpty


class EvidenceBundle(StrictModel):
    schema_version: Literal[1] = 1
    job_id: NonEmpty
    root_url: NonEmpty
    council: NonEmpty
    batch: NonEmpty
    created_at: NonEmpty
    artifacts: tuple[EvidenceArtifact, ...]

    @model_validator(mode="after")
    def unique_artifact_ids(self) -> "EvidenceBundle":
        ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(ids) != len(set(ids)):
            raise ValueError("Evidence artifact_id values must be unique")
        return self

    def roles(self) -> set[ArtifactRole]:
        return {artifact.role for artifact in self.artifacts}


class EvidenceCitation(StrictModel):
    artifact_id: NonEmpty
    location: NonEmpty
    statement: NonEmpty
    excerpt_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class PredicateOperator(str, Enum):
    ALWAYS = "always"
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IN = "in"
    NOT_IN = "not_in"
    IS_BLANK = "is_blank"
    NOT_BLANK = "not_blank"
    YEAR_BETWEEN = "year_between"


class Predicate(StrictModel):
    field: str = ""
    operator: PredicateOperator
    value: str | tuple[str, ...] | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "Predicate":
        if self.operator == PredicateOperator.ALWAYS:
            if self.field:
                raise ValueError("always predicates must not name a field")
            return self
        if not self.field:
            raise ValueError(f"{self.operator.value} predicates require a field")
        if self.operator in {
            PredicateOperator.EQUALS,
            PredicateOperator.NOT_EQUALS,
        } and not isinstance(self.value, str):
            raise ValueError(f"{self.operator.value} predicates require a string value")
        if self.operator in {
            PredicateOperator.IN,
            PredicateOperator.NOT_IN,
            PredicateOperator.YEAR_BETWEEN,
        } and not isinstance(self.value, tuple):
            raise ValueError(f"{self.operator.value} predicates require a list value")
        if self.operator == PredicateOperator.YEAR_BETWEEN and len(self.value or ()) != 2:
            raise ValueError("year_between requires exactly two boundary years")
        if self.operator == PredicateOperator.YEAR_BETWEEN:
            try:
                low, high = (int(item) for item in (self.value or ()))
            except ValueError as exc:
                raise ValueError("year_between boundaries must be integer years") from exc
            if low > high:
                raise ValueError("year_between lower boundary must not exceed upper boundary")
        return self


class NormalizerName(str, Enum):
    TRIM = "trim"
    CASEFOLD = "casefold"
    COLLAPSE_SPACE = "collapse_space"
    SLASH_TO_HYPHEN = "slash_to_hyphen"
    ALNUM = "alnum"


class SecondaryOperator(str, Enum):
    EXACT_NORMALIZED = "exact_normalized"
    DATE_EQUAL = "date_equal"
    TOKEN_OVERLAP = "token_overlap"


class SecondaryCheck(StrictModel):
    source_field: NonEmpty
    candidate_field: NonEmpty
    operator: SecondaryOperator
    normalizers: tuple[NormalizerName, ...] = (
        NormalizerName.TRIM,
        NormalizerName.CASEFOLD,
    )
    threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    required: bool = True


class RouteTarget(str, Enum):
    S3 = "s3"
    PORTAL = "portal"
    REJECT = "reject"


class PartNormalizers(StrictModel):
    """Normalisers applied to one named part before it enters the key."""

    part: NonEmpty
    normalizers: tuple[NonEmpty, ...]


class PartDefault(StrictModel):
    """Value used for a part an alternative template does not capture."""

    part: NonEmpty
    value: str


class DerivedKey(StrictModel):
    """Build the join key from parts of a reference, on both sides at once.

    A whole-field normaliser cannot turn `98/0538/CAC` into the folder token
    `EXE_1988_88-1061`: that needs the reference decomposed, its number stripped
    of leading zeros, and the result rebuilt. Declaring both sides as templates
    over the same named parts keeps the two in step, and lets a spec be dry-run
    against real evidence before it is executed.

    Alternatives are tried in order, because real references arrive in shape
    variants -- an optional classification segment, a trailing code that is
    sometimes absent.
    """

    schema_version: Literal[1] = 1
    source_templates: tuple[NonEmpty, ...]
    inventory_templates: tuple[NonEmpty, ...]
    key_parts: tuple[NonEmpty, ...]
    source_match_mode: Literal["exact", "prefix"] = "exact"
    inventory_match_mode: Literal["exact", "prefix"] = "exact"
    # Declared as lists rather than maps: a structured-output schema has to name
    # every property it allows, so an open-keyed object cannot be requested from
    # the compiler at all.
    part_normalizers: tuple[PartNormalizers, ...] = ()
    part_defaults: tuple[PartDefault, ...] = ()

    def build(self) -> KeyDerivation:
        return KeyDerivation(
            source_templates=tuple(
                Template(pattern, self.source_match_mode) for pattern in self.source_templates
            ),
            inventory_templates=tuple(
                Template(pattern, self.inventory_match_mode)
                for pattern in self.inventory_templates
            ),
            key_parts=tuple(self.key_parts),
            normalizers={
                entry.part: tuple(entry.normalizers) for entry in self.part_normalizers
            },
            defaults={entry.part: entry.value for entry in self.part_defaults},
        )

    @model_validator(mode="after")
    def validate_derivation(self) -> "DerivedKey":
        if not self.source_templates or not self.inventory_templates:
            raise ValueError("A derived key needs a template on both sides")
        try:
            self.build()
        except DerivationError as exc:
            raise ValueError(str(exc)) from exc
        return self


class RouteRule(StrictModel):
    rule_id: NonEmpty
    # Evaluated lowest first; the first matching route wins. A catch-all reject
    # therefore belongs at the highest number, not the lowest.
    priority: int
    conditions: tuple[Predicate, ...]
    target: RouteTarget
    authoritative_key: str = ""
    fallback_key: str | None = None
    fallback_only_when_authoritative_blank: bool = True
    inventory_key_field: str | None = None
    inventory_path_field: str | None = None
    normalizers: tuple[NormalizerName, ...] = (
        NormalizerName.TRIM,
        NormalizerName.CASEFOLD,
    )
    inventory_conditions: tuple[Predicate, ...] = ()
    # When set, replaces whole-field normalisation on both sides of the join.
    derived_key: DerivedKey | None = None
    secondary_checks: tuple[SecondaryCheck, ...] = ()
    automatic_confidence: Confidence = 0.0
    content_verified: bool = False
    ambiguity_action: Literal["reject"] = "reject"
    citations: tuple[EvidenceCitation, ...] = ()

    @model_validator(mode="after")
    def validate_acceptance_contract(self) -> "RouteRule":
        if self.target != RouteTarget.REJECT and not self.authoritative_key:
            raise ValueError("S3 and Portal rules require authoritative_key")
        if self.target == RouteTarget.REJECT and self.automatic_confidence != 0:
            raise ValueError("Reject rules must use confidence 0.00")
        if not self.fallback_only_when_authoritative_blank:
            raise ValueError("MappingSpec v1 permits fallback only when the authoritative field is blank")
        if self.content_verified:
            raise ValueError("MappingSpec v1 has no content-verification stage; content_verified must be false")
        if self.automatic_confidence > 0.74:
            raise ValueError("MappingSpec v1 automatic matches cannot exceed confidence 0.74")
        return self


class MappingSpec(StrictModel):
    schema_version: Literal[1] = 1
    spec_id: NonEmpty
    council: NonEmpty
    batch: NonEmpty
    source_id_field: NonEmpty = "oachargeid"
    inventory_key_field: NonEmpty = "candidate_key"
    inventory_path_field: NonEmpty = "candidate_path"
    routes: tuple[RouteRule, ...]

    @model_validator(mode="after")
    def validate_rules(self) -> "MappingSpec":
        if not self.routes:
            raise ValueError("MappingSpec requires at least one route")
        ids = [route.rule_id for route in self.routes]
        if len(ids) != len(set(ids)):
            raise ValueError("Route rule_id values must be unique")
        return self

    def ordered_routes(self) -> tuple[RouteRule, ...]:
        return tuple(sorted(self.routes, key=lambda route: (route.priority, route.rule_id)))


class JobOperation(str, Enum):
    BUILD = "build"
    REPAIR = "repair"
    APPEND = "append"


class JobStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    AWAITING_INPUT = "awaiting_input"
    COMPLETED_STAGED = "completed_staged"
    COMPLETED_SAFE_WITH_EXCEPTIONS = "completed_safe_with_exceptions"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_PERMANENT = "failed_permanent"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobRequest(StrictModel):
    source_url: NonEmpty
    council_hint: str | None = None
    batch_hint: str | None = None
    operation: JobOperation = JobOperation.BUILD
    requested_by: str = ""

    @field_validator("source_url")
    @classmethod
    def supported_url_scheme(cls, value: str) -> str:
        value = value.strip()
        if not value.startswith(("https://", "http://", "s3://", "file://")):
            raise ValueError("source_url must use https, http, s3, or file")
        return value

    @model_validator(mode="after")
    def council_and_batch_hints_travel_together(self) -> "JobRequest":
        if bool(self.council_hint) != bool(self.batch_hint):
            raise ValueError("council_hint and batch_hint must be supplied together")
        return self


class JobRecord(StrictModel):
    schema_version: Literal[1] = 1
    job_id: NonEmpty
    request: JobRequest
    council: str | None = None
    batch: str | None = None
    status: JobStatus
    current_stage: str | None = None
    workspace: Path
    created_at: NonEmpty
    updated_at: NonEmpty
    error: str | None = None


class ValidationIssue(StrictModel):
    level: Literal["error", "warning"]
    code: NonEmpty
    oachargeid: str = ""
    detail: str = ""


class ValidationReport(StrictModel):
    schema_version: Literal[1] = 1
    source_rows: Annotated[int, Field(ge=0)]
    mapping_rows: Annotated[int, Field(ge=0)]
    audit_rows: Annotated[int, Field(ge=0)]
    accepted_s3: Annotated[int, Field(ge=0)]
    accepted_portal: Annotated[int, Field(ge=0)]
    rejected_or_unmatched: Annotated[int, Field(ge=0)]
    gates: dict[str, bool]
    issues: tuple[ValidationIssue, ...]

    @property
    def passed(self) -> bool:
        return all(self.gates.values()) and not any(issue.level == "error" for issue in self.issues)


class PublishPolicyInputs(StrictModel):
    staging_only: bool = True
    spec_verification_passed: bool = False
    negative_tests_passed: bool = False
    historical_regressions_passed: bool = False
    content_qa_passed: bool = False
    systematic_content_failures: Annotated[int, Field(ge=0)] = 0
    target_unchanged: bool = False


class PublishDecision(StrictModel):
    schema_version: Literal[1] = 1
    job_id: NonEmpty
    allowed: bool
    mode: Literal["staging_only", "production"]
    gates: dict[str, bool]
    failed_gates: tuple[str, ...]
    proposed_mapping: Path
    audit_path: Path
    generated_at: NonEmpty


class ReplayExpectation(StrictModel):
    oachargeid: NonEmpty
    expected_amazons3_path: str | None = None
    expected_confidence: Confidence | None = None
    expected_status: str | None = None
    expected_status_prefix: str | None = None
    forbidden_paths: tuple[str, ...] = ()


class ReplaySuite(StrictModel):
    schema_version: Literal[1] = 1
    name: NonEmpty
    cases: tuple[ReplayExpectation, ...]


class ReplayResult(StrictModel):
    schema_version: Literal[1] = 1
    suite: NonEmpty
    cases: Annotated[int, Field(ge=0)]
    passed: Annotated[int, Field(ge=0)]
    failed: Annotated[int, Field(ge=0)]
    failures: tuple[str, ...]


class DiscoveryMethod(str, Enum):
    EXPLICIT_MANIFEST = "explicit_manifest"
    HTML_LINK = "html_link"
    DIRECTORY_ENTRY = "directory_entry"
    DIRECT_RESOURCE = "direct_resource"


class DiscoveredArtifact(StrictModel):
    artifact_id: NonEmpty
    role: ArtifactRole
    source_url: NonEmpty
    local_path: Path
    original_name: NonEmpty
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    size_bytes: Annotated[int, Field(ge=0)]
    media_type: NonEmpty
    method: DiscoveryMethod
    classification_reason: NonEmpty


class DiscoveryManifest(StrictModel):
    schema_version: Literal[1] = 1
    job_id: NonEmpty
    root_url: NonEmpty
    declared_council: str | None = None
    declared_batch: str | None = None
    created_at: NonEmpty
    artifacts: tuple[DiscoveredArtifact, ...]
    missing_roles: tuple[ArtifactRole, ...]
    warnings: tuple[str, ...] = ()
    evidence_complete: bool


class RuleChunk(StrictModel):
    artifact_id: Literal["capture_rules"] = "capture_rules"
    location: NonEmpty
    text: NonEmpty
    excerpt_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class PreparationReport(StrictModel):
    schema_version: Literal[1] = 1
    job_id: NonEmpty
    council: NonEmpty
    batch: NonEmpty
    registry_council_known: bool
    registry_batch_known: bool
    source_path: Path
    inventory_path: Path
    capture_rules_path: Path
    source_rows: Annotated[int, Field(ge=0)]
    inventory_rows: Annotated[int, Field(ge=0)]
    source_fields: tuple[str, ...]
    inventory_fields: tuple[str, ...]
    inventory_roles: tuple[ArtifactRole, ...]
    capture_rule_chunks: tuple[RuleChunk, ...]
    warnings: tuple[str, ...] = ()

    @property
    def registry_ready(self) -> bool:
        return self.registry_council_known and self.registry_batch_known


class SpecVerificationReport(StrictModel):
    schema_version: Literal[1] = 1
    job_id: NonEmpty
    spec_id: NonEmpty
    passed: bool
    gates: dict[str, bool]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    accepting_routes: Annotated[int, Field(ge=0)]
    cited_chunks: Annotated[int, Field(ge=0)]


class HistoricalReplayReport(StrictModel):
    schema_version: Literal[1] = 1
    job_id: NonEmpty
    applicable_suites: Annotated[int, Field(ge=0)]
    passed: bool
    results: tuple[ReplayResult, ...]
    warnings: tuple[str, ...] = ()


class QaVerdict(str, Enum):
    VERIFIED_SAME = "verified_same"
    # Kept distinct from VERIFIED_SAME so a report never implies an address was
    # compared when the source held none. Only reachable where the source row
    # carries nothing but its reference.
    VERIFIED_REFERENCE_ONLY = "verified_reference_only"
    VERIFIED_WRONG = "verified_wrong"
    RULE_SUPPORTED_UNVERIFIED = "rule_supported_unverified"
    AMBIGUOUS = "ambiguous"
    UNREADABLE = "unreadable"
    MISSING_DOCUMENT = "missing_document"
    NOT_APPLICABLE = "not_applicable"


class ContentQaReport(StrictModel):
    schema_version: Literal[1] = 1
    job_id: NonEmpty
    population: Annotated[int, Field(ge=0)]
    requested_sample_size: Annotated[int, Field(ge=0)]
    selected_sample_size: Annotated[int, Field(ge=0)]
    strata: Annotated[int, Field(ge=0)]
    reviewed: Annotated[int, Field(ge=0)] = 0
    verdict_counts: dict[str, int] = Field(default_factory=dict)
    systematic_content_failures: Annotated[int, Field(ge=0)] = 0
    passed: bool = False
    queue_path: Path
    warnings: tuple[str, ...] = ()


class ObservedIdentity(StrictModel):
    reference: str | None
    property_address: str | None
    description: str | None
    relevant_date: str | None
    document_type: str | None
    image_names: tuple[str, ...]
    evidence: tuple[str, ...]


class ContentObservation(StrictModel):
    schema_version: Literal[1] = 1
    readable: bool
    images_reviewed: Annotated[int, Field(ge=0)]
    identities: tuple[ObservedIdentity, ...]
    unreadable_image_names: tuple[str, ...]
    warnings: tuple[str, ...]


class ContentExpectation(StrictModel):
    schema_version: Literal[1] = 1
    council: NonEmpty
    batch: NonEmpty
    oachargeid: NonEmpty
    route: Literal["s3", "portal", "none"]
    mapping_path: str
    mapping_confidence: Confidence = 0.0
    mapping_status: str = ""
    match_basis: str
    reference_fields: tuple[str, ...]
    reference_values: tuple[str, ...]
    address_fields: tuple[str, ...]
    address_values: tuple[str, ...]
    description_fields: tuple[str, ...]
    description_values: tuple[str, ...]
    date_fields: tuple[str, ...]
    date_values: tuple[str, ...]
    document_type_fields: tuple[str, ...]
    document_type_values: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        status = self.mapping_status.casefold().replace("-", "_")
        rejected_status = any(
            marker in status
            for marker in (
                "ambiguous",
                "missing",
                "no_scan",
                "no scan",
                "not_found",
                "not found",
                "reject",
                "unmatched",
                "unreadable",
                "unsupported",
                "wrong_scan",
                "wrong scan",
            )
        )
        return (
            self.route in {"s3", "portal"}
            and bool(self.mapping_path)
            and self.mapping_confidence > 0
            and not rejected_status
        )


class AcquisitionStatus(str, Enum):
    COMPLETED = "completed"
    SKIPPED_EXISTING = "skipped_existing"
    PARTIAL = "partial"
    RETRYABLE = "retryable"
    FAILED = "failed"
    MAPPING_REJECTED = "mapping_rejected"


class AcquiredFile(StrictModel):
    schema_version: Literal[1] = 1
    source_uri: NonEmpty
    relative_path: NonEmpty
    local_path: Path
    expected_size: Annotated[int | None, Field(ge=0)] = None
    actual_size: Annotated[int, Field(ge=0)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class CaseAcquisitionReport(StrictModel):
    schema_version: Literal[1] = 1
    council: NonEmpty
    batch: NonEmpty
    oachargeid: NonEmpty
    route: Literal["s3", "portal", "none"]
    mapping_path: str
    mapping_confidence: Confidence
    mapping_status: str
    status: AcquisitionStatus
    destination: Path | None
    files: tuple[AcquiredFile, ...]
    files_completed: Annotated[int, Field(ge=0)]
    bytes_completed: Annotated[int, Field(ge=0)]
    error: str | None
    started_at: NonEmpty
    ended_at: NonEmpty
    completion_marker: Path | None


class AcquisitionBatchReport(StrictModel):
    schema_version: Literal[1] = 1
    run_id: NonEmpty
    council: NonEmpty
    batch: NonEmpty
    requested_cases: Annotated[int, Field(ge=0)]
    accepted_cases: Annotated[int, Field(ge=0)]
    files_completed: Annotated[int, Field(ge=0)]
    bytes_completed: Annotated[int, Field(ge=0)]
    status_counts: dict[str, int]
    case_reports: tuple[CaseAcquisitionReport, ...]
    documents_root: Path
    generated_at: NonEmpty


class ContentCaseResult(StrictModel):
    schema_version: Literal[1] = 1
    council: NonEmpty
    batch: NonEmpty
    oachargeid: NonEmpty
    route: Literal["s3", "portal", "none"]
    mapping_path: str
    match_basis: str
    verdict: QaVerdict
    recommended_confidence: Confidence
    eligible_for_confidence_upgrade: bool
    signals: dict[str, bool]
    matched_identity_index: int | None
    document_directory: Path | None
    selected_images: tuple[Path, ...]
    expectation_path: Path
    observation_path: Path | None
    acquisition_status: AcquisitionStatus | None = None
    reason: NonEmpty
    warnings: tuple[str, ...] = ()


class ContentVerificationReport(StrictModel):
    schema_version: Literal[1] = 1
    run_id: NonEmpty
    council: NonEmpty
    batch: NonEmpty
    scope: Literal["targeted", "stratified_sample", "full_population"]
    population: Annotated[int, Field(ge=0)]
    selected: Annotated[int, Field(ge=0)]
    accepted_selected: Annotated[int, Field(ge=0)]
    reviewed: Annotated[int, Field(ge=0)]
    verdict_counts: dict[str, int]
    systematic_content_failures: Annotated[int, Field(ge=0)]
    sample_passed: bool
    passed: bool
    queue_path: Path
    expectations_path: Path
    case_results_path: Path
    acquisition_report_path: Path | None = None
    generated_at: NonEmpty
    warnings: tuple[str, ...] = ()
