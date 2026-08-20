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
    verified_same = counts[VERIFIED_SAME]
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
