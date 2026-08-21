"""Orchestrate the sample -> review -> gate -> adjust loop over a mapping run.

The loop exists because a mapping spec is rarely right first time. Each round
reviews unseen cases from a working pool, and a failing round reports which rule
family the failures share so the spec can be corrected. Acceptance is measured
once, on a holdout that no adjustment ever saw.

The holdout is frozen as a set of case identifiers on the first round. Strata are
derived from audit columns such as route and confidence, which change whenever
the mapping changes, so recomputing the split every round would let a case drift
between the holdout and the working pool and quietly destroy the separation the
holdout exists to provide.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

from .engine import read_csv
from .quality_gate import GateOutcome, GateThresholds, evaluate_gate
from .quality_report import RoundRecord, render_quality_report
from .sampling_plan import (
    DEFAULT_HOLDOUT_FRACTION,
    SamplingPlan,
    case_id,
    partition_population,
    select_acceptance,
    select_round,
)


STATE_SCHEMA_VERSION = 1
WORKING = "working"
ACCEPTANCE = "acceptance"


class QualityLoopError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SampleReviewer(Protocol):
    """Reviews a named set of cases and returns their content-QA results."""

    def review(
        self,
        *,
        run_id: str,
        include_ids: Sequence[str],
        output_dir: Path,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        ...


@dataclass
class LoopState:
    council: str
    batch: str
    seed: str
    holdout_fraction: float
    working_ids: tuple[str, ...]
    holdout_ids: tuple[str, ...]
    strata_without_holdout: tuple[str, ...]
    rounds: list[dict[str, Any]] = field(default_factory=list)

    @property
    def sampled_ids(self) -> tuple[str, ...]:
        seen: list[str] = []
        for record in self.rounds:
            if record.get("stage") == WORKING:
                seen.extend(str(value) for value in record.get("sampled_ids", ()))
        return tuple(seen)

    @property
    def next_round_index(self) -> int:
        return sum(1 for record in self.rounds if record.get("stage") == WORKING) + 1

    def plan(self) -> SamplingPlan:
        return SamplingPlan(
            seed=self.seed,
            holdout_fraction=self.holdout_fraction,
            working_ids=self.working_ids,
            holdout_ids=self.holdout_ids,
            strata_without_holdout=self.strata_without_holdout,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "council": self.council,
            "batch": self.batch,
            "seed": self.seed,
            "holdout_fraction": self.holdout_fraction,
            "working_ids": list(self.working_ids),
            "holdout_ids": list(self.holdout_ids),
            "strata_without_holdout": list(self.strata_without_holdout),
            "rounds": self.rounds,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "LoopState":
        if payload.get("schema_version") != STATE_SCHEMA_VERSION:
            raise QualityLoopError(
                f"Unsupported quality loop state version {payload.get('schema_version')!r}"
            )
        return cls(
            council=str(payload["council"]),
            batch=str(payload["batch"]),
            seed=str(payload["seed"]),
            holdout_fraction=float(payload["holdout_fraction"]),
            working_ids=tuple(str(value) for value in payload.get("working_ids", ())),
            holdout_ids=tuple(str(value) for value in payload.get("holdout_ids", ())),
            strata_without_holdout=tuple(
                str(value) for value in payload.get("strata_without_holdout", ())
            ),
            rounds=list(payload.get("rounds", [])),
        )


def load_state(path: Path) -> LoopState | None:
    if not path.is_file():
        return None
    return LoopState.from_json(json.loads(path.read_text(encoding="utf-8")))


def save_state(path: Path, state: LoopState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def open_state(
    path: Path,
    *,
    council: str,
    batch: str,
    audit_rows: Sequence[dict[str, str]],
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> LoopState:
    """Load the frozen split, or create it from the first mapping's strata.

    A reopened loop keeps its original split even though the audit it was derived
    from has since changed. Cases that appear only in a later mapping are added to
    the working pool: they were never reserved, so they cannot serve as holdout.
    """
    population_ids = [case_id(row) for row in audit_rows]
    if len(set(population_ids)) != len(population_ids):
        raise QualityLoopError("Audit contains duplicate case identifiers")

    state = load_state(path)
    if state is None:
        plan = partition_population(
            audit_rows, seed=f"{council}:{batch}", holdout_fraction=holdout_fraction
        )
        state = LoopState(
            council=council,
            batch=batch,
            seed=plan.seed,
            holdout_fraction=plan.holdout_fraction,
            working_ids=plan.working_ids,
            holdout_ids=plan.holdout_ids,
            strata_without_holdout=plan.strata_without_holdout,
        )
        save_state(path, state)
        return state

    if state.council != council or state.batch != batch:
        raise QualityLoopError(
            f"State at {path} belongs to {state.council}/{state.batch}, not {council}/{batch}"
        )
    known = set(state.working_ids) | set(state.holdout_ids)
    added = tuple(value for value in population_ids if value not in known)
    if added:
        state.working_ids = state.working_ids + added
        save_state(path, state)
    return state


@dataclass(frozen=True)
class RoundResult:
    stage: str
    index: int
    record: RoundRecord
    outcome: GateOutcome
    sampled_ids: tuple[str, ...]
    exhausted: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "round_index": self.index,
            "sampled_ids": list(self.sampled_ids),
            "exhausted": self.exhausted,
            **self.outcome.describe(),
        }


def run_quality_round(
    *,
    state: LoopState,
    state_path: Path,
    audit_rows: Sequence[dict[str, str]],
    reviewer: SampleReviewer,
    artifacts_root: Path,
    stage: str = WORKING,
    sample_size: int = 12,
    thresholds: GateThresholds | None = None,
) -> RoundResult:
    """Sample, review, and judge one round; persist what was sampled."""
    if stage not in {WORKING, ACCEPTANCE}:
        raise QualityLoopError(f"Unknown stage {stage!r}")
    plan = state.plan()
    index = state.next_round_index if stage == WORKING else 0

    if stage == WORKING:
        selected = select_round(
            audit_rows,
            plan,
            round_index=index,
            sample_size=sample_size,
            already_sampled=state.sampled_ids,
        )
        if not selected:
            # Every working case has been reviewed. Continuing would resample
            # cases the spec was already adjusted against.
            return RoundResult(
                stage=stage,
                index=index,
                record=RoundRecord(
                    index=index,
                    label="working pool exhausted",
                    case_results=(),
                    outcome=evaluate_gate((), thresholds=thresholds),
                ),
                outcome=evaluate_gate((), thresholds=thresholds),
                sampled_ids=(),
                exhausted=True,
            )
        label = f"round {index} · working pool"
    else:
        selected = select_acceptance(audit_rows, plan, sample_size=sample_size)
        label = "acceptance · holdout"

    include_ids = tuple(case_id(row) for row in selected)
    run_id = f"{state.council}-{state.batch}-{stage}-{index:02d}"
    output_dir = artifacts_root / run_id
    verification, case_results = reviewer.review(
        run_id=run_id, include_ids=include_ids, output_dir=output_dir
    )
    outcome = evaluate_gate(case_results, thresholds=thresholds)
    record = RoundRecord(
        index=index,
        label=label,
        case_results=tuple(case_results),
        outcome=outcome,
        verification_report=verification,
    )

    state.rounds.append(
        {
            "stage": stage,
            "index": index,
            "label": label,
            "run_id": run_id,
            "sampled_ids": list(include_ids),
            "passed": outcome.passed,
            "metrics": outcome.metrics,
            "reasons": list(outcome.reasons),
            "failure_signatures": [item.describe() for item in outcome.signatures],
            "artifacts": str(output_dir),
            # Kept per round so the whole mapping can be held to one ceiling.
            # The budget object is rebuilt for every /quality call, so without
            # this the limit is per round and a loop multiplies it.
            "spent_usd": float((verification.get("budget") or {}).get("spent_usd") or 0.0),
            "reviewed_at": utc_now(),
        }
    )
    save_state(state_path, state)
    return RoundResult(
        stage=stage,
        index=index,
        record=record,
        outcome=outcome,
        sampled_ids=include_ids,
    )


def spent_so_far(state: LoopState) -> float:
    """What every round of this mapping has already cost."""
    return sum(float(record.get("spent_usd") or 0.0) for record in state.rounds)


def shared_reason(
    case_results: Sequence[dict[str, Any]], *, threshold: float = 0.5
) -> tuple[str, int, int] | None:
    """Return the reason most unverified cases give, when one dominates."""
    unverified = [
        result
        for result in case_results
        if str(getattr(result.get("verdict"), "value", result.get("verdict")) or "")
        not in {"verified_same", "not_applicable"}
    ]
    if not unverified:
        return None
    counts: dict[str, int] = {}
    for result in unverified:
        reason = str(result.get("reason") or "").strip()
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return None
    reason, count = max(counts.items(), key=lambda item: item[1])
    if count < max(2, threshold * len(unverified)):
        return None
    return reason, count, len(unverified)


def next_action(result: RoundResult) -> dict[str, Any]:
    """Say what the loop should do next, and what to look at if it must adjust."""
    if result.exhausted:
        return {
            "action": "stop_exhausted",
            "detail": (
                "Every working-pool case has been reviewed. Adjusting further would fit the sample; "
                "run the acceptance stage or widen the population."
            ),
        }
    if result.outcome.passed:
        return {
            "action": "accept" if result.stage == WORKING else "publish",
            "detail": (
                "The round cleared the gate. Run the acceptance stage on the holdout before publishing."
                if result.stage == WORKING
                else "The holdout cleared the gate; the mapping may be published."
            ),
        }
    systematic = result.outcome.systematic_signatures
    if systematic:
        focus = [
            f"{item.route}/{item.match_basis} failing on {', '.join(item.failing_signals) or 'all signals'}"
            f" ({item.count} cases, e.g. {', '.join(item.example_ids[:3])})"
            for item in systematic
        ]
        return {
            "action": "adjust_spec",
            "detail": "Repeated failures share a route and matching basis, which points at a rule.",
            "focus": focus,
        }
    # Signatures only group confirmed-wrong cases. A round can also fail because
    # every case failed the same way without any being wrong -- a batch whose
    # source carries nothing to check against, for instance -- and reporting
    # that as "no shared signature" sends the reader case-hunting for a cause
    # that is the same in all of them.
    shared = shared_reason(result.record.case_results)
    if shared:
        reason, count, total = shared
        return {
            "action": "review_inputs",
            "detail": (
                f"{count} of {total} judged cases failed for the same reason, which points at the "
                "batch rather than at individual mappings."
            ),
            "focus": [reason],
        }
    return {
        "action": "investigate_cases",
        "detail": (
            "Failures do not share a signature, so they read as individual bad cases rather than a "
            "broken rule. Inspect them before changing the spec."
        ),
        "focus": [item.describe() for item in result.outcome.signatures],
    }


def write_report(
    *,
    destination: Path,
    state: LoopState,
    rounds: Sequence[RoundRecord],
    acceptance: RoundRecord | None,
    mapping_summary: dict[str, Any],
) -> Path:
    return render_quality_report(
        destination=destination,
        council=state.council,
        batch=state.batch,
        run_id=f"{state.council}-{state.batch}",
        plan=state.plan(),
        rounds=rounds,
        acceptance=acceptance,
        mapping_summary=mapping_summary,
        generated_at=utc_now(),
    )


def load_round_records(state: LoopState) -> tuple[list[RoundRecord], RoundRecord | None]:
    """Rebuild every round from persisted state so a report can span the loop.

    Case results live in each round's own artifact directory rather than in the
    state file, which stays small and readable. A round whose artifacts have
    been pruned is kept with its recorded metrics but no case detail, so the
    report shows that the round happened instead of silently dropping it.
    """
    working: list[RoundRecord] = []
    acceptance: RoundRecord | None = None
    for record in state.rounds:
        results: list[dict[str, Any]] = []
        artifacts = record.get("artifacts")
        if artifacts:
            case_results_path = Path(str(artifacts)) / "qa" / "content-case-results.json"
            if case_results_path.is_file():
                results = json.loads(case_results_path.read_text(encoding="utf-8"))
        outcome = evaluate_gate(results) if results else GateOutcome(
            passed=bool(record.get("passed")),
            metrics=dict(record.get("metrics") or {}),
            reasons=tuple(str(value) for value in record.get("reasons") or ()),
        )
        rebuilt = RoundRecord(
            index=int(record.get("index") or 0),
            label=str(record.get("label") or ""),
            case_results=tuple(results),
            outcome=outcome,
            adjustment=str(record.get("adjustment") or ""),
        )
        if record.get("stage") == ACCEPTANCE:
            acceptance = rebuilt
        else:
            working.append(rebuilt)
    return working, acceptance


def load_audit_rows(audit_path: Path) -> list[dict[str, str]]:
    _, rows = read_csv(audit_path)
    if not rows:
        raise QualityLoopError(f"Audit evidence contains no rows: {audit_path}")
    return rows
