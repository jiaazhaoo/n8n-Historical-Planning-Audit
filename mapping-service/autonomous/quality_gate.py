"""Turn content-QA case results into a pass/fail decision with attribution.

`ContentVerificationReport.passed` is only ever true for a full-population
review, because a sample cannot verify a population. The quality loop still
needs a decision from each sampled round, so this module states the sampling
gate explicitly: what counts as good enough to stop, and when a failure is
systematic enough to point at a rule rather than at one bad case.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


VERIFIED_SAME = "verified_same"
# Where the source records only a reference, a document carrying that reference
# is the strongest evidence available, so it counts as verified -- reported
# under its own name so nobody reads it as an address having been compared.
VERIFIED_REFERENCE_ONLY = "verified_reference_only"
VERIFIED_WRONG = "verified_wrong"
MISSING_DOCUMENT = "missing_document"
UNREADABLE = "unreadable"
NOT_APPLICABLE = "not_applicable"

SIGNAL_ORDER = ("reference", "address", "description", "date")


@dataclass(frozen=True)
class GateThresholds:
    """What the reviewed sample has to show before the loop may stop.

    A confirmed-wrong case fails the round outright: unlike an unreadable scan
    or a missing document, it is direct evidence that the mapping sent a case to
    the wrong document, and no pass rate excuses it.
    """

    min_verified_rate: float = 0.95
    max_verified_wrong: int = 0
    max_systematic_failures: int = 0
    max_missing_document_rate: float = 0.10

    def validate(self) -> None:
        if not 0.0 < self.min_verified_rate <= 1.0:
            raise ValueError("min_verified_rate must be in (0, 1]")
        if not 0.0 <= self.max_missing_document_rate <= 1.0:
            raise ValueError("max_missing_document_rate must be in [0, 1]")
        if self.max_verified_wrong < 0 or self.max_systematic_failures < 0:
            raise ValueError("gate counts cannot be negative")


@dataclass(frozen=True)
class FailureSignature:
    """A group of confirmed-wrong cases that share a route, basis, and symptom."""

    route: str
    match_basis: str
    failing_signals: tuple[str, ...]
    count: int
    example_ids: tuple[str, ...]

    @property
    def systematic(self) -> bool:
        return self.count >= 2

    def describe(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "match_basis": self.match_basis,
            "failing_signals": list(self.failing_signals),
            "count": self.count,
            "systematic": self.systematic,
            "example_ids": list(self.example_ids),
        }


@dataclass(frozen=True)
class GateOutcome:
    passed: bool
    metrics: dict[str, Any]
    reasons: tuple[str, ...] = ()
    signatures: tuple[FailureSignature, ...] = field(default=())

    @property
    def systematic_signatures(self) -> tuple[FailureSignature, ...]:
        return tuple(signature for signature in self.signatures if signature.systematic)

    def describe(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "metrics": self.metrics,
            "reasons": list(self.reasons),
            "failure_signatures": [signature.describe() for signature in self.signatures],
        }


@dataclass(frozen=True)
class CoverageOutcome:
    """How much of the population the mapping actually resolved.

    The sample gate measures whether accepted mappings are right. It cannot
    measure how many cases were accepted at all, because cases the mapping
    rejected are `not_applicable` to a content review. A spec that accepts a
    small fraction of the population and gets those right therefore passes the
    sample gate while delivering almost nothing, so coverage is judged here,
    against the whole population, and reported separately.
    """

    passed: bool
    metrics: dict[str, Any]
    reasons: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        return {"passed": self.passed, "metrics": self.metrics, "reasons": list(self.reasons)}


def evaluate_coverage(
    match_status_counts: dict[str, Any] | None,
    *,
    min_accepted_rate: float = 0.0,
    max_unmatched_rate: float = 1.0,
) -> CoverageOutcome:
    """Judge population coverage from the mapping run's own status counts.

    `no_found` means the source itself reports no scan, which no spec can fix,
    so it is reported but never counted against the mapping. `no_match` is the
    spec failing to resolve a case that should have resolved.
    """
    counts = {key: int(value or 0) for key, value in (match_status_counts or {}).items()}
    found = counts.get("found", 0)
    no_found = counts.get("no_found", 0)
    no_match = counts.get("no_match", 0)
    total = found + no_found + no_match

    accepted_rate = found / total if total else 0.0
    unmatched_rate = no_match / total if total else 0.0
    resolvable = found + no_match
    accepted_of_resolvable = found / resolvable if resolvable else 0.0

    reasons: list[str] = []
    if not total:
        reasons.append("The mapping run reported no cases; coverage cannot be judged")
    if total and accepted_rate < min_accepted_rate:
        reasons.append(
            f"Accepted rate {accepted_rate:.2%} is below the required {min_accepted_rate:.2%}; "
            "the mapping is accurate on what it accepted but resolved too little to deliver"
        )
    if total and unmatched_rate > max_unmatched_rate:
        reasons.append(
            f"Unmatched rate {unmatched_rate:.2%} exceeds {max_unmatched_rate:.2%}"
        )

    return CoverageOutcome(
        passed=not reasons,
        metrics={
            "population": total,
            "accepted": found,
            "source_reports_no_scan": no_found,
            "unmatched": no_match,
            "accepted_rate": round(accepted_rate, 4),
            "unmatched_rate": round(unmatched_rate, 4),
            # Excludes cases the source itself says have no scan, so this is the
            # share of genuinely resolvable cases the spec resolved.
            "accepted_rate_of_resolvable": round(accepted_of_resolvable, 4),
            "thresholds": {
                "min_accepted_rate": min_accepted_rate,
                "max_unmatched_rate": max_unmatched_rate,
            },
        },
        reasons=tuple(reasons),
    )


def _verdict(result: dict[str, Any]) -> str:
    value = result.get("verdict")
    return str(getattr(value, "value", value) or "").strip()


def _failing_signals(result: dict[str, Any]) -> tuple[str, ...]:
    signals = result.get("signals") or {}
    if not isinstance(signals, dict):
        return ()
    return tuple(name for name in SIGNAL_ORDER if name in signals and not signals[name])


def _case_id(result: dict[str, Any]) -> str:
    return str(result.get("oachargeid") or "").strip()


def build_signatures(results: Iterable[dict[str, Any]]) -> tuple[FailureSignature, ...]:
    """Group confirmed-wrong cases by the shape of their failure.

    Two cases that fail the same way on the same route and basis point at a
    rule; two cases that fail differently point at two cases.
    """
    grouped: dict[tuple[str, str, tuple[str, ...]], list[str]] = defaultdict(list)
    for result in results:
        if _verdict(result) != VERIFIED_WRONG:
            continue
        key = (
            str(result.get("route") or "none"),
            str(result.get("match_basis") or "none"),
            _failing_signals(result),
        )
        grouped[key].append(_case_id(result))
    signatures = [
        FailureSignature(
            route=route,
            match_basis=match_basis,
            failing_signals=signals,
            count=len(ids),
            example_ids=tuple(sorted(ids)[:5]),
        )
        for (route, match_basis, signals), ids in grouped.items()
    ]
    signatures.sort(key=lambda item: (-item.count, item.route, item.match_basis))
    return tuple(signatures)


def evaluate_gate(
    results: Sequence[dict[str, Any]],
    *,
    thresholds: GateThresholds | None = None,
) -> GateOutcome:
    """Decide whether a reviewed sample clears the bar, and say why if not."""
    thresholds = thresholds or GateThresholds()
    thresholds.validate()

    counts: dict[str, int] = defaultdict(int)
    for result in results:
        counts[_verdict(result) or "unknown"] += 1

    # not_applicable covers cases the mapping deliberately did not accept, so
    # they are outside the population this gate is measuring.
    judged = [result for result in results if _verdict(result) != NOT_APPLICABLE]
    judged_total = len(judged)
    verified_same = counts[VERIFIED_SAME] + counts[VERIFIED_REFERENCE_ONLY]
    verified_wrong = counts[VERIFIED_WRONG]
    missing = counts[MISSING_DOCUMENT]

    verified_rate = verified_same / judged_total if judged_total else 0.0
    missing_rate = missing / judged_total if judged_total else 0.0
    signatures = build_signatures(results)
    systematic = sum(signature.count for signature in signatures if signature.systematic)

    reasons: list[str] = []
    if not judged_total:
        reasons.append("No case in the sample was judged; the round proves nothing")
    if verified_wrong > thresholds.max_verified_wrong:
        reasons.append(
            f"{verified_wrong} case(s) were confirmed wrong; the limit is {thresholds.max_verified_wrong}"
        )
    if systematic > thresholds.max_systematic_failures:
        reasons.append(
            f"{systematic} confirmed-wrong case(s) share a route/basis signature, which points at a rule rather than a case"
        )
    if judged_total and verified_rate < thresholds.min_verified_rate:
        reasons.append(
            f"Verified rate {verified_rate:.2%} is below the required {thresholds.min_verified_rate:.2%}"
        )
    if judged_total and missing_rate > thresholds.max_missing_document_rate:
        reasons.append(
            f"Missing-document rate {missing_rate:.2%} exceeds {thresholds.max_missing_document_rate:.2%}; "
            "the sample could not be reviewed rather than the mapping being wrong"
        )

    metrics = {
        "sample_size": len(results),
        "judged": judged_total,
        "verdict_counts": dict(sorted(counts.items())),
        "verified_rate": round(verified_rate, 4),
        "missing_document_rate": round(missing_rate, 4),
        "verified_wrong": verified_wrong,
        "systematic_failures": systematic,
        "thresholds": {
            "min_verified_rate": thresholds.min_verified_rate,
            "max_verified_wrong": thresholds.max_verified_wrong,
            "max_systematic_failures": thresholds.max_systematic_failures,
            "max_missing_document_rate": thresholds.max_missing_document_rate,
        },
    }
    return GateOutcome(
        passed=not reasons,
        metrics=metrics,
        reasons=tuple(reasons),
        signatures=signatures,
    )
