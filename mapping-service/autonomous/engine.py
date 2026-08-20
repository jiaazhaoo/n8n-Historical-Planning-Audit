from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .path_policy import require_unprotected_path
from .schemas import (
    MappingSpec,
    NormalizerName,
    Predicate,
    PredicateOperator,
    RouteRule,
    RouteTarget,
    SecondaryCheck,
    SecondaryOperator,
)


MAPPING_FIELDS = (
    "oachargeid",
    "amazons3_path",
    "amazons3_confidence",
    "portal_path",
)

AUDIT_FIELDS = (
    "oachargeid",
    "rule_id",
    "route",
    "authoritative_key",
    "authoritative_value",
    "match_basis",
    "candidate_count",
    "eligible_candidate_count",
    "candidate_paths",
    "amazons3_path",
    "portal_path",
    "decision_confidence",
    "match_status",
    "rejection_reason",
)


class MappingEngineError(RuntimeError):
    pass


@dataclass(frozen=True)
class MappingResult:
    source_rows: tuple[dict[str, str], ...]
    mapping_rows: tuple[dict[str, str], ...]
    audit_rows: tuple[dict[str, str], ...]


def clean(value: object) -> str:
    return str(value or "").strip()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    path = require_unprotected_path(path, operation="read mapping/audit CSV")
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        rows = [{str(key): clean(value) for key, value in row.items() if key is not None} for row in reader]
    return headers, rows


def write_csv_bytes(rows: Iterable[dict[str, str]], fields: tuple[str, ...]) -> bytes:
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def normalize(value: object, normalizers: Iterable[NormalizerName]) -> str:
    result = str(value or "")
    for normalizer in normalizers:
        if normalizer == NormalizerName.TRIM:
            result = result.strip()
        elif normalizer == NormalizerName.CASEFOLD:
            result = result.casefold()
        elif normalizer == NormalizerName.COLLAPSE_SPACE:
            result = " ".join(result.split())
        elif normalizer == NormalizerName.SLASH_TO_HYPHEN:
            result = re.sub(r"[\\/／]+", "-", result)
        elif normalizer == NormalizerName.ALNUM:
            result = "".join(character for character in result if character.isalnum())
        else:  # pragma: no cover - enum validation makes this defensive only
            raise MappingEngineError(f"Unsupported normalizer: {normalizer}")
    return result


def extract_year(value: object) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", clean(value))
    return int(match.group(0)) if match else None


def predicate_matches(row: dict[str, str], predicate: Predicate) -> bool:
    if predicate.operator == PredicateOperator.ALWAYS:
        return True
    actual = clean(row.get(predicate.field))
    if predicate.operator == PredicateOperator.IS_BLANK:
        return not actual
    if predicate.operator == PredicateOperator.NOT_BLANK:
        return bool(actual)
    if predicate.operator == PredicateOperator.EQUALS:
        return actual.casefold() == clean(predicate.value).casefold()
    if predicate.operator == PredicateOperator.NOT_EQUALS:
        return actual.casefold() != clean(predicate.value).casefold()
    if predicate.operator in {PredicateOperator.IN, PredicateOperator.NOT_IN}:
        expected = {clean(value).casefold() for value in (predicate.value or ())}
        found = actual.casefold() in expected
        return found if predicate.operator == PredicateOperator.IN else not found
    if predicate.operator == PredicateOperator.YEAR_BETWEEN:
        year = extract_year(actual)
        if year is None:
            return False
        low, high = (int(value) for value in (predicate.value or ()))
        return low <= year <= high
    raise MappingEngineError(f"Unsupported predicate operator: {predicate.operator}")


def route_for(row: dict[str, str], spec: MappingSpec) -> RouteRule | None:
    for route in spec.ordered_routes():
        if all(predicate_matches(row, condition) for condition in route.conditions):
            return route
    return None


def date_token(value: object) -> str:
    digits = "".join(character for character in clean(value) if character.isdigit())
    if len(digits) == 8:
        return digits
    year = extract_year(value)
    return str(year or "")


def word_tokens(value: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", clean(value).casefold())
        if len(token) > 1
    }


def secondary_check_passes(
    source: dict[str, str],
    candidate: dict[str, str],
    check: SecondaryCheck,
) -> bool:
    source_value = clean(source.get(check.source_field))
    candidate_value = clean(candidate.get(check.candidate_field))
    if not source_value or not candidate_value:
        return not check.required
    if check.operator == SecondaryOperator.EXACT_NORMALIZED:
        return normalize(source_value, check.normalizers) == normalize(candidate_value, check.normalizers)
    if check.operator == SecondaryOperator.DATE_EQUAL:
        return bool(date_token(source_value)) and date_token(source_value) == date_token(candidate_value)
    if check.operator == SecondaryOperator.TOKEN_OVERLAP:
        source_tokens = word_tokens(source_value)
        candidate_tokens = word_tokens(candidate_value)
        if not source_tokens or not candidate_tokens:
            return not check.required
        overlap = len(source_tokens & candidate_tokens) / len(source_tokens | candidate_tokens)
        return overlap >= check.threshold
    raise MappingEngineError(f"Unsupported secondary operator: {check.operator}")


def candidate_uri_valid(target: RouteTarget, path: str) -> bool:
    if target == RouteTarget.S3:
        return path.startswith("s3://")
    if target == RouteTarget.PORTAL:
        return path.startswith(("https://", "http://"))
    return False


def _blank_decision(oid: str, *, status: str, reason: str, rule: RouteRule | None = None) -> tuple[dict[str, str], dict[str, str]]:
    mapping = {
        "oachargeid": oid,
        "amazons3_path": "",
        "amazons3_confidence": "0.00",
        "portal_path": "",
    }
    audit = {
        "oachargeid": oid,
        "rule_id": rule.rule_id if rule else "",
        "route": rule.target.value if rule else "none",
        "authoritative_key": "",
        "authoritative_value": "",
        "match_basis": "",
        "candidate_count": "0",
        "eligible_candidate_count": "0",
        "candidate_paths": "[]",
        "amazons3_path": "",
        "portal_path": "",
        "decision_confidence": "0.00",
        "match_status": status,
        "rejection_reason": reason,
    }
    return mapping, audit


def execute_mapping(
    source_rows: Iterable[dict[str, str]],
    inventory_rows: Iterable[dict[str, str]],
    spec: MappingSpec,
) -> MappingResult:
    source = tuple({key: clean(value) for key, value in row.items()} for row in source_rows)
    inventory = tuple({key: clean(value) for key, value in row.items()} for row in inventory_rows)

    ids = [clean(row.get(spec.source_id_field)) for row in source]
    blanks = [index + 2 for index, oid in enumerate(ids) if not oid]
    if blanks:
        raise MappingEngineError(f"Blank source IDs at CSV rows: {blanks[:20]}")
    duplicates = sorted(oid for oid, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise MappingEngineError(f"Duplicate source IDs require an explicit precedence rule: {duplicates[:20]}")

    mapping_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []
    for row, oid in zip(source, ids, strict=True):
        rule = route_for(row, spec)
        if rule is None:
            mapping, audit = _blank_decision(
                oid,
                status="rejected_no_route_rule",
                reason="No MappingSpec route matched the source row",
            )
            mapping_rows.append(mapping)
            audit_rows.append(audit)
            continue
        if rule.target == RouteTarget.REJECT:
            mapping, audit = _blank_decision(
                oid,
                status="rejected_by_capture_rule",
                reason=f"Route {rule.rule_id} explicitly rejects this row",
                rule=rule,
            )
            mapping_rows.append(mapping)
            audit_rows.append(audit)
            continue

        authoritative_value = clean(row.get(rule.authoritative_key))
        match_basis = rule.authoritative_key
        if not authoritative_value and rule.fallback_key:
            authoritative_value = clean(row.get(rule.fallback_key))
            match_basis = rule.fallback_key
        if not authoritative_value:
            mapping, audit = _blank_decision(
                oid,
                status="rejected_missing_authoritative_value",
                reason=f"Neither {rule.authoritative_key!r} nor the permitted fallback contains a value",
                rule=rule,
            )
            audit["authoritative_key"] = match_basis
            mapping_rows.append(mapping)
            audit_rows.append(audit)
            continue

        # A derived key decomposes both sides into the same named parts; without
        # one, the whole field is normalised and compared as it stands.
        derivation = rule.derived_key.build() if rule.derived_key else None
        if derivation is not None:
            derived = derivation.source_key(authoritative_value)
            if derived is None:
                mapping, audit = _blank_decision(
                    oid,
                    status="rejected_unparsable_authoritative_value",
                    reason=(
                        f"{match_basis!r} value {authoritative_value!r} does not fit any declared "
                        "source template, so no key could be derived"
                    ),
                    rule=rule,
                )
                audit["authoritative_key"] = match_basis
                audit["authoritative_value"] = authoritative_value
                mapping_rows.append(mapping)
                audit_rows.append(audit)
                continue
            key: object = derived
        else:
            key = normalize(authoritative_value, rule.normalizers)
        inventory_key_field = rule.inventory_key_field or spec.inventory_key_field
        inventory_path_field = rule.inventory_path_field or spec.inventory_path_field
        discovered: list[dict[str, str]] = []
        for candidate in inventory:
            if not all(predicate_matches(candidate, condition) for condition in rule.inventory_conditions):
                continue
            if derivation is not None:
                candidate_key: object = derivation.inventory_key(candidate.get(inventory_key_field) or "")
                if candidate_key is None:
                    continue
            else:
                candidate_key = normalize(candidate.get(inventory_key_field), rule.normalizers)
            if candidate_key == key:
                discovered.append(candidate)

        paths: list[str] = []
        candidates_by_path: dict[str, dict[str, str]] = {}
        for candidate in discovered:
            path = clean(candidate.get(inventory_path_field))
            if path and path not in candidates_by_path:
                paths.append(path)
                candidates_by_path[path] = candidate

        eligible: list[tuple[str, dict[str, str]]] = []
        invalid_paths = 0
        for path in paths:
            candidate = candidates_by_path[path]
            if not candidate_uri_valid(rule.target, path):
                invalid_paths += 1
                continue
            if all(secondary_check_passes(row, candidate, check) for check in rule.secondary_checks):
                eligible.append((path, candidate))

        mapping = {
            "oachargeid": oid,
            "amazons3_path": "",
            "amazons3_confidence": "0.00",
            "portal_path": "",
        }
        audit = {
            "oachargeid": oid,
            "rule_id": rule.rule_id,
            "route": rule.target.value,
            "authoritative_key": match_basis,
            "authoritative_value": authoritative_value,
            "match_basis": match_basis,
            "candidate_count": str(len(paths)),
            "eligible_candidate_count": str(len(eligible)),
            "candidate_paths": json.dumps(paths, ensure_ascii=False, separators=(",", ":")),
            "amazons3_path": "",
            "portal_path": "",
            "decision_confidence": "0.00",
            "match_status": "",
            "rejection_reason": "",
        }

        if len(eligible) == 1:
            accepted_path = eligible[0][0]
            if rule.target == RouteTarget.S3:
                mapping["amazons3_path"] = accepted_path
                mapping["amazons3_confidence"] = f"{rule.automatic_confidence:.2f}"
                audit["amazons3_path"] = accepted_path
            else:
                mapping["portal_path"] = accepted_path
                audit["portal_path"] = accepted_path
            audit["decision_confidence"] = f"{rule.automatic_confidence:.2f}"
            audit["match_status"] = "accepted_unique_rule_supported"
        elif len(eligible) > 1:
            audit["match_status"] = "rejected_ambiguous_multiple_candidates"
            audit["rejection_reason"] = "More than one eligible candidate remains after rule checks"
        elif invalid_paths and invalid_paths == len(paths):
            audit["match_status"] = "rejected_invalid_candidate_uri"
            audit["rejection_reason"] = "Candidate URI scheme does not match the selected route"
        elif paths and rule.secondary_checks:
            audit["match_status"] = "rejected_secondary_evidence"
            audit["rejection_reason"] = "Candidates exist but required secondary evidence did not agree"
        else:
            audit["match_status"] = "not_found"
            audit["rejection_reason"] = "No inventory candidate matched the authoritative value"

        mapping_rows.append(mapping)
        audit_rows.append(audit)

    return MappingResult(source, tuple(mapping_rows), tuple(audit_rows))
