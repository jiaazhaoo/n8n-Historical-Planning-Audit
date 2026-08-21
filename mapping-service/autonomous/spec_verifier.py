from __future__ import annotations

from .engine import MappingEngineError, execute_mapping, read_csv, route_for
from .schemas import (
    MappingSpec,
    PredicateOperator,
    PreparationReport,
    RouteTarget,
    SpecVerificationReport,
)


def _condition_value(condition: object) -> str:
    operator = getattr(condition, "operator")
    value = getattr(condition, "value")
    if operator == PredicateOperator.EQUALS:
        return str(value)
    if operator == PredicateOperator.IN:
        return str((value or ("value",))[0])
    if operator == PredicateOperator.YEAR_BETWEEN:
        return str((value or ("2000",))[0])
    if operator == PredicateOperator.NOT_BLANK:
        return "value"
    if operator == PredicateOperator.IS_BLANK:
        return ""
    if operator == PredicateOperator.NOT_EQUALS:
        return "__different__"
    if operator == PredicateOperator.NOT_IN:
        return "__outside__"
    return ""


def ambiguity_negative_test_errors(spec: MappingSpec, preparation: PreparationReport) -> list[str]:
    _, source_rows = read_csv(preparation.source_path)
    errors: list[str] = []
    if not source_rows:
        return ["ambiguity negative tests require at least one prepared source row"]
    for route in spec.routes:
        if route.target == RouteTarget.REJECT:
            continue
        existing = next((row for row in source_rows if route_for(row, spec) == route), None)
        source = dict(existing or source_rows[0])
        for condition in route.conditions:
            if condition.operator == PredicateOperator.ALWAYS:
                continue
            # A not_blank condition is already satisfied by whatever the row
            # holds. Overwriting a real reference with a placeholder makes a
            # derived key unparsable and turns this test into a false alarm.
            if condition.operator == PredicateOperator.NOT_BLANK and str(
                source.get(condition.field) or ""
            ).strip():
                continue
            source[condition.field] = _condition_value(condition)
        source.setdefault(spec.source_id_field, f"negative-{route.rule_id}")
        if not str(source.get(spec.source_id_field) or "").strip():
            source[spec.source_id_field] = f"negative-{route.rule_id}"
        selected = route_for(source, spec)
        if selected != route:
            selected_name = selected.rule_id if selected else "none"
            errors.append(
                f"route {route.rule_id}: accepting route is unreachable; synthetic row selects {selected_name}"
            )
            continue
        authoritative_value = str(source.get(route.authoritative_key) or "").strip()
        if not authoritative_value and route.fallback_key:
            authoritative_value = str(source.get(route.fallback_key) or "").strip()
        if not authoritative_value:
            source[route.authoritative_key] = f"NEGATIVE-{route.rule_id}"
            authoritative_value = source[route.authoritative_key]
        inventory_key = route.inventory_key_field or spec.inventory_key_field
        inventory_path = route.inventory_path_field or spec.inventory_path_field
        # With a derived key the candidate must look like an inventory entry,
        # not like the raw reference, or the synthetic pair never collides and
        # the test would silently prove nothing.
        candidate_key_value = authoritative_value
        if route.derived_key is not None:
            rendered = route.derived_key.build().inventory_value_for(authoritative_value)
            if rendered is None:
                errors.append(
                    f"route {route.rule_id}: derived key cannot render an inventory form for "
                    f"{authoritative_value!r}, so ambiguity cannot be demonstrated"
                )
                continue
            candidate_key_value = rendered
        base: dict[str, str] = {inventory_key: candidate_key_value}
        for condition in route.inventory_conditions:
            if condition.operator != PredicateOperator.ALWAYS:
                base[condition.field] = _condition_value(condition)
        for check in route.secondary_checks:
            source_value = str(source.get(check.source_field) or "").strip()
            if not source_value:
                source_value = "2001-01-01" if check.operator.value == "date_equal" else "negative evidence"
                source[check.source_field] = source_value
            base[check.candidate_field] = source_value
        paths = (
            ("s3://negative-test/a", "s3://negative-test/b")
            if route.target == RouteTarget.S3
            else ("https://negative.invalid/a", "https://negative.invalid/b")
        )
        candidates = [{**base, inventory_path: path} for path in paths]
        try:
            result = execute_mapping([source], candidates, spec)
        except (KeyError, MappingEngineError, ValueError) as exc:
            errors.append(
                f"route {route.rule_id}: ambiguity negative test could not execute: {type(exc).__name__}: {exc}"
            )
            continue
        mapping = result.mapping_rows[0]
        audit = result.audit_rows[0]
        if mapping["amazons3_path"] or mapping["portal_path"]:
            errors.append(f"route {route.rule_id}: ambiguity negative test accepted a path")
        if audit["match_status"] != "rejected_ambiguous_multiple_candidates":
            errors.append(
                f"route {route.rule_id}: ambiguity negative test returned {audit['match_status']!r}"
            )
    return errors


MIN_STRATUM_CASES = 30
MAX_STRATUM_VALUES = 60
STRATUM_SHORTFALL = 0.25
STRATUM_EVIDENCE_RATIO = 0.5


def _stratum_findings(
    rule_id: str,
    source_keys: list[tuple[str, ...]],
    index: set[tuple[str, ...]],
    key_parts: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    """Report a subpopulation that joins far below the rest of its route.

    An overall rate hides a local failure. Exeter's Microfiche spec joined 90.1%
    and passed verification while all 859 records of 1977 -- whose folders append
    the site address -- matched nothing; the other eight years carried the rate.

    Requiring a *complete* miss is not enough: the next spec reached 2.1% of 1977
    against 100% everywhere else, and eighteen accidental hits would have hidden
    it again. What marks a defect is the shortfall against the route's own rate,
    not a zero.

    A shortfall is only a defect when the scans are actually there. Where the
    inventory holds few keys in the stratum, they were never delivered, no spec
    can join them, and reporting it would cost the compiler an attempt it has no
    way to use -- so that case is a warning.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if len(source_keys) < MIN_STRATUM_CASES or not index:
        return errors, warnings
    overall = sum(1 for key in source_keys if key in index) / len(source_keys)
    if overall <= 0:
        return errors, warnings

    for position, part in enumerate(key_parts):
        if any(len(key) <= position for key in source_keys):
            continue
        totals: dict[str, int] = {}
        joins: dict[str, int] = {}
        for key in source_keys:
            value = key[position]
            totals[value] = totals.get(value, 0) + 1
            if key in index:
                joins[value] = joins.get(value, 0) + 1
        if not 2 <= len(totals) <= MAX_STRATUM_VALUES:
            continue

        stocked: dict[str, int] = {}
        for key in index:
            if len(key) > position:
                stocked[key[position]] = stocked.get(key[position], 0) + 1

        for value, count in sorted(totals.items(), key=lambda item: -item[1]):
            joined = joins.get(value, 0)
            if count < MIN_STRATUM_CASES:
                continue
            rate = joined / count
            if rate >= overall * STRATUM_SHORTFALL:
                continue
            available = stocked.get(value, 0)
            if available < count * STRATUM_EVIDENCE_RATIO:
                warnings.append(
                    f"route {rule_id}: {count} cases have {part}={value!r} and join {rate:.1%}, but "
                    f"the inventory holds only {available} keys there, so those scans are absent "
                    "rather than unreachable"
                )
                continue
            errors.append(
                f"route {rule_id}: cases with {part}={value!r} join {joined}/{count} ({rate:.1%}) "
                f"while the route joins {overall:.1%} overall, and the inventory holds {available} "
                f"keys for {part}={value!r}. Those scans exist and the derivation is not reaching "
                f"them, so {part}={value!r} is written differently on the two sides. Read real key "
                "values from both sides for that stratum and add the template alternative it needs. "
                "Do not widen key_parts to absorb the difference: a part the source leaves at its "
                "default can never equal one the inventory fills in."
            )
    return errors, warnings


def derived_key_join_errors(
    spec: MappingSpec, preparation: PreparationReport
) -> tuple[list[str], list[str]]:
    """Dry-run every derived key against the real evidence before executing it.

    A derivation can be structurally valid and still join nothing, because a
    part means different things on the two sides -- a folder's trailing code is
    a document type, while a reference's trailing code is an application type.
    Nothing in the schema can catch that, but running the derivation over the
    prepared source and inventory answers it in seconds, before a mapping run
    reports every case as not_found.
    """
    errors: list[str] = []
    warnings: list[str] = []
    routes = [
        route
        for route in spec.routes
        if route.target != RouteTarget.REJECT and route.derived_key is not None
    ]
    if not routes:
        return errors, warnings

    _, source_rows = read_csv(preparation.source_path)
    _, inventory_rows = read_csv(preparation.inventory_path)
    if not source_rows or not inventory_rows:
        return errors, warnings

    # Route selection is a property of the row, not of the route being examined.
    # Asking it once per row per route made this quadratic in the number of
    # routes, which a four-route Sheffield spec over 26,384 rows felt as minutes.
    rows_by_rule: dict[str, list[dict[str, str]]] = {}
    for row in source_rows:
        chosen = route_for(row, spec)
        if chosen is not None:
            rows_by_rule.setdefault(chosen.rule_id, []).append(row)

    # Routes that declare the same derivation over the same column produce the
    # same index; building it once per distinct derivation keeps a large
    # inventory from being walked once per route.
    indexes: dict[str, set[tuple[str, ...]]] = {}

    for route in routes:
        derivation = route.derived_key.build()
        key_field = route.inventory_key_field or spec.inventory_key_field
        signature = f"{key_field}\0{route.derived_key.model_dump_json()}"
        index = indexes.get(signature)
        if index is None:
            index = set()
            for candidate in inventory_rows:
                key = derivation.inventory_key(str(candidate.get(key_field) or ""))
                if key is not None:
                    index.add(key)
            indexes[signature] = index

        selected = rows_by_rule.get(route.rule_id, [])
        joined = 0
        keyed = 0
        source_keys: list[tuple[str, ...]] = []
        for row in selected:
            value = str(row.get(route.authoritative_key) or "").strip()
            if not value and route.fallback_key:
                value = str(row.get(route.fallback_key) or "").strip()
            key = derivation.source_key(value) if value else None
            if key is None:
                continue
            keyed += 1
            source_keys.append(key)
            if key in index:
                joined += 1

        stratum_errors, stratum_warnings = _stratum_findings(
            route.rule_id, source_keys, index, derivation.key_parts
        )
        errors.extend(stratum_errors)
        warnings.extend(stratum_warnings)

        rate = joined / len(selected) if selected else 0.0
        warnings.append(
            f"route {route.rule_id}: derived key joins {joined}/{len(selected)} routed cases "
            f"({rate:.1%}); {keyed} produced a key and the inventory yielded {len(index)} distinct keys"
        )
        if not index and len(inventory_rows) >= 20:
            # No inventory row keyed at all. That is never a property of the
            # data; the inventory side of the derivation is simply wrong.
            sample = str(inventory_rows[0].get(key_field) or "")
            errors.append(
                f"route {route.rule_id}: the derived key produces no key for any of "
                f"{len(inventory_rows)} inventory rows. {derivation.explain(sample, side='inventory')}"
            )
        elif joined == 0 and len(selected) >= 20 and len(index) >= 20:
            errors.append(
                f"route {route.rule_id}: the derived key joins none of {len(selected)} routed cases "
                f"against {len(index)} inventory keys. Check that every part in key_parts means the "
                "same thing on both sides; a folder's trailing code is usually a document type, not "
                "the reference's application-type suffix."
            )
    return errors, warnings


def inert_prefix_errors(spec: MappingSpec) -> list[str]:
    """Refuse a prefix template that can never match a tail.

    Declaring prefix mode and then ending the template in an untyped part makes
    the part greedy, so the tail always matches empty and the join is exact
    after all. Exeter's Microfiche 1977-1985 spec did this: 1978-1985 folders
    have no trailing text so it looked correct, while all 859 records of 1977 --
    whose folders append the site address -- reported zero candidates and the
    run still passed at 90.1%. A join dry-run cannot catch it either, because
    the other eight years carry the rate.
    """
    errors: list[str] = []
    for route in spec.routes:
        if route.derived_key is None:
            continue
        try:
            derivation = route.derived_key.build()
        except Exception:  # reported by the schema, not here
            continue
        for template in derivation.inventory_templates:
            if not template.prefix_is_inert:
                continue
            errors.append(
                f"route {route.rule_id!r} declares inventory_match_mode 'prefix' but template "
                f"{template.pattern!r} ends in an untyped part, which matches greedily and leaves "
                "nothing for the trailing text, so the join is exact after all. Either give the "
                "final part a kind (:d digits, :a letters and digits) so the trailing text is "
                "excluded from the key, or set inventory_match_mode to 'exact' if the folder name "
                "really is the key. Where references have different shapes, list several templates "
                "as alternatives -- 'EXE_{year:d}_{yy:d}-{number:d}-{code:a}' alongside "
                "'EXE_{year:d}_{yy:d}-{number:d}' -- with part_defaults for the parts a shorter "
                "alternative does not supply."
            )
    return errors


BOOKKEEPING_FIELDS = {"_artifact_id"}


def undiscriminating_condition_findings(
    spec: MappingSpec, preparation: PreparationReport
) -> tuple[list[str], list[str]]:
    """Report route conditions that select every row they are shown.

    Exeter's Microfiche spec conditioned one route on six fields, each holding a
    single value across all 8,898 rows -- `always` written six times. Every such
    condition is inert today and a silent reject tomorrow: the next delivery
    whose failed-checks reads anything but TA-29 routes nowhere, and the run
    reports no_match rather than an error.

    `_artifact_id` is worse than redundant, so it is an error rather than a
    warning: preparation assigns it per download, so a spec carrying it can
    never be reused on the next delivery of the same work package.
    """
    errors: list[str] = []
    warnings: list[str] = []
    accepting = [route for route in spec.routes if route.target != RouteTarget.REJECT]
    if not accepting:
        return errors, warnings

    counts: dict[str, set[str]] = {}
    try:
        _, source_rows = read_csv(preparation.source_path)
    except OSError:
        return errors, warnings
    for route in accepting:
        for condition in route.conditions:
            field = getattr(condition, "field", None)
            if field and field not in counts:
                counts[field] = {row.get(field, "") for row in source_rows}

    for route in accepting:
        for condition in route.conditions:
            field = getattr(condition, "field", None)
            if not field:
                continue
            if field in BOOKKEEPING_FIELDS:
                errors.append(
                    f"route {route.rule_id!r} conditions on {field!r}, which preparation assigns "
                    "per download rather than reading from the source. The spec would reject "
                    "every row of the next delivery of this work package. Route on a field the "
                    "source itself carries, or on _evidence_role when the inventory is mixed."
                )
            elif len(counts.get(field, set())) == 1:
                only = next(iter(counts[field]))
                warnings.append(
                    f"route {route.rule_id!r} conditions on {field}={only!r}, the only value in "
                    f"all {len(source_rows)} source rows, so the condition selects everything it "
                    "is shown. It discriminates nothing now and silently rejects any future row "
                    "that differs."
                )
    return errors, warnings


def verify_mapping_spec(
    *,
    job_id: str,
    spec: MappingSpec,
    preparation: PreparationReport,
) -> SpecVerificationReport:
    errors: list[str] = []
    warnings: list[str] = []
    source_fields = set(preparation.source_fields)
    inventory_fields = set(preparation.inventory_fields)
    chunks = {
        (chunk.artifact_id, chunk.location, chunk.excerpt_sha256): chunk
        for chunk in preparation.capture_rule_chunks
    }
    cited_chunks: set[tuple[str, str, str]] = set()
    accepting_routes = 0

    errors.extend(inert_prefix_errors(spec))
    condition_errors, condition_warnings = undiscriminating_condition_findings(spec, preparation)
    errors.extend(condition_errors)
    warnings.extend(condition_warnings)

    if spec.council != preparation.council:
        errors.append(f"spec council {spec.council!r} does not equal prepared council {preparation.council!r}")
    if spec.batch != preparation.batch:
        errors.append(f"spec batch {spec.batch!r} does not equal prepared batch {preparation.batch!r}")
    if spec.source_id_field not in source_fields:
        errors.append(f"source_id_field {spec.source_id_field!r} is absent from source evidence")
    default_key_used = any(
        route.target != RouteTarget.REJECT and not route.inventory_key_field for route in spec.routes
    )
    default_path_used = any(
        route.target != RouteTarget.REJECT and not route.inventory_path_field for route in spec.routes
    )
    if default_key_used and spec.inventory_key_field not in inventory_fields:
        errors.append(f"default inventory_key_field {spec.inventory_key_field!r} is absent from inventory evidence")
    if default_path_used and spec.inventory_path_field not in inventory_fields:
        errors.append(f"default inventory_path_field {spec.inventory_path_field!r} is absent from inventory evidence")

    _, all_source_rows = read_csv(preparation.source_path)
    source_ids = [str(row.get(spec.source_id_field) or "").strip() for row in all_source_rows]
    if any(not source_id for source_id in source_ids):
        errors.append(f"source_id_field {spec.source_id_field!r} contains blank values")
    if len(source_ids) != len(set(source_ids)):
        errors.append(f"source_id_field {spec.source_id_field!r} contains duplicate values")

    evidence_roles = {role.value for role in preparation.inventory_roles}

    for route in spec.routes:
        for condition in route.conditions:
            if condition.operator != PredicateOperator.ALWAYS and condition.field not in source_fields:
                errors.append(f"route {route.rule_id}: source condition field {condition.field!r} is absent")
        if route.target == RouteTarget.REJECT:
            continue
        accepting_routes += 1
        for field, label in (
            (route.authoritative_key, "authoritative_key"),
            (route.fallback_key, "fallback_key"),
        ):
            if field and field not in source_fields:
                errors.append(f"route {route.rule_id}: {label} {field!r} is absent from source evidence")
        effective_key = route.inventory_key_field or spec.inventory_key_field
        effective_path = route.inventory_path_field or spec.inventory_path_field
        if effective_key not in inventory_fields:
            errors.append(f"route {route.rule_id}: inventory key field {effective_key!r} is absent")
        if effective_path not in inventory_fields:
            errors.append(f"route {route.rule_id}: inventory path field {effective_path!r} is absent")
        for condition in route.inventory_conditions:
            if condition.operator != PredicateOperator.ALWAYS and condition.field not in inventory_fields:
                errors.append(f"route {route.rule_id}: inventory condition field {condition.field!r} is absent")
        for check in route.secondary_checks:
            if check.source_field not in source_fields:
                errors.append(f"route {route.rule_id}: secondary source field {check.source_field!r} is absent")
            if check.candidate_field not in inventory_fields:
                errors.append(f"route {route.rule_id}: secondary inventory field {check.candidate_field!r} is absent")
        if not route.citations:
            errors.append(f"route {route.rule_id}: accepting route has no capture-rule citation")
        for citation in route.citations:
            key = (citation.artifact_id, citation.location, citation.excerpt_sha256)
            if key not in chunks:
                errors.append(
                    f"route {route.rule_id}: citation {citation.artifact_id}/{citation.location} does not match a frozen chunk"
                )
            else:
                cited_chunks.add(key)
        if len(evidence_roles) > 1:
            expected_role = "s3_inventory" if route.target == RouteTarget.S3 else "portal_evidence"
            role_conditions = [
                condition
                for condition in route.inventory_conditions
                if condition.field == "_evidence_role" and condition.operator == PredicateOperator.EQUALS
            ]
            if not any(str(condition.value) == expected_role for condition in role_conditions):
                errors.append(
                    f"route {route.rule_id}: mixed inventory requires _evidence_role={expected_role!r}"
                )

    if accepting_routes == 0:
        warnings.append("MappingSpec contains only reject routes; this is safe but produces no accepted mappings")
    negative_errors = ambiguity_negative_test_errors(spec, preparation)
    errors.extend(negative_errors)
    join_errors, join_warnings = derived_key_join_errors(spec, preparation)
    errors.extend(join_errors)
    warnings.extend(join_warnings)
    gates = {
        "council_batch_exact": spec.council == preparation.council and spec.batch == preparation.batch,
        "source_fields_exist": not any("source" in error and "absent" in error for error in errors),
        "source_ids_nonblank_unique": not any("source_id_field" in error and "values" in error for error in errors),
        "inventory_fields_exist": not any("inventory" in error and "absent" in error for error in errors),
        "citations_frozen": not any("citation" in error for error in errors),
        "accepting_routes_cited": not any("has no capture-rule citation" in error for error in errors),
        "mixed_routes_separated": not any("mixed inventory requires" in error for error in errors),
        "ambiguity_negative_tests": not negative_errors,
        "registry_ready": preparation.registry_ready,
    }
    if not preparation.registry_ready:
        errors.append("Council and batch must be registered before executing the proposed mapping")
    passed = all(gates.values()) and not errors
    return SpecVerificationReport(
        job_id=job_id,
        spec_id=spec.spec_id,
        passed=passed,
        gates=gates,
        errors=tuple(errors),
        warnings=tuple(warnings),
        accepting_routes=accepting_routes,
        cited_chunks=len(cited_chunks),
    )
