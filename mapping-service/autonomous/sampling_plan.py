"""Round-based sampling with a reserved holdout for the mapping quality loop.

The quality loop repeatedly samples cases, reviews them, and adjusts the mapping
spec when the review fails. Sampling the same cases every round would make the
loop fit the sample instead of improving the mapping, so this module splits the
population once into a holdout that never informs an adjustment and a working
pool that the optimisation rounds draw from, without replacement.

The split and every round are derived from a seed, so a rerun of the same job
reviews the same cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .qa import stable_rank, stratum


DEFAULT_HOLDOUT_FRACTION = 0.2

# Every reviewed case costs a document acquisition and a vision call, so a
# sample must be bounded no matter how the population is shaped.
MAX_REVIEW_SAMPLE = 40

# Strata are meant to group cases. When nearly every case lands in its own
# stratum the audit is carrying per-case detail in a column this module treats
# as categorical, and any "stratified" sample drawn from it is meaningless.
MAX_STRATA_RATIO = 0.5
MIN_ROWS_FOR_STRATA_CHECK = 20


class SamplingPlanError(RuntimeError):
    pass


def case_id(row: dict[str, str]) -> str:
    return str(row.get("oachargeid") or "").strip()


@dataclass(frozen=True)
class SamplingPlan:
    """A population split into an optimisation pool and an untouched holdout."""

    seed: str
    holdout_fraction: float
    working_ids: tuple[str, ...]
    holdout_ids: tuple[str, ...]
    strata_without_holdout: tuple[str, ...]

    @property
    def population(self) -> int:
        return len(self.working_ids) + len(self.holdout_ids)

    def describe(self) -> dict[str, object]:
        return {
            "population": self.population,
            "working_cases": len(self.working_ids),
            "holdout_cases": len(self.holdout_ids),
            "holdout_fraction_requested": self.holdout_fraction,
            "holdout_fraction_actual": (
                round(len(self.holdout_ids) / self.population, 4) if self.population else 0.0
            ),
            "strata_without_holdout": list(self.strata_without_holdout),
        }


def validate_stratification(rows: Sequence[dict[str, str]]) -> None:
    """Refuse a population whose strata do not group anything.

    A legacy council-builder audit stores the matched reference inside
    `match_status`, which makes the stratum key unique per case. Sampling that
    would review a large share of the population while reporting itself as a
    small stratified sample.
    """
    if len(rows) < MIN_ROWS_FOR_STRATA_CHECK:
        return
    distinct = len({stratum(row) for row in rows})
    if distinct > len(rows) * MAX_STRATA_RATIO:
        raise SamplingPlanError(
            f"Stratification is not informative: {distinct} strata across {len(rows)} cases. "
            "This audit is probably not in the autonomous engine's schema; check that "
            "match_status holds a category rather than a per-case value."
        )


def _allocate(groups: dict[tuple[str, ...], list[dict[str, str]]], budget: int) -> dict[tuple[str, ...], int]:
    """Split a fixed budget across strata in proportion to their size.

    Largest-remainder allocation, so the sample mirrors the population instead
    of over-representing rare strata. When the budget is smaller than the number
    of strata some strata necessarily get nothing; that shortfall is reported by
    the caller rather than papered over by raising the budget.
    """
    total = sum(len(group) for group in groups.values())
    if not total:
        return {}
    exact = {key: len(group) * budget / total for key, group in groups.items()}
    allocation = {key: min(int(value), len(groups[key])) for key, value in exact.items()}

    order = sorted(groups, key=lambda key: (-(exact[key] - int(exact[key])), key))
    while sum(allocation.values()) < budget:
        progressed = False
        for key in order:
            if sum(allocation.values()) >= budget:
                break
            if allocation[key] < len(groups[key]):
                allocation[key] += 1
                progressed = True
        if not progressed:
            break
    return allocation


def bounded_stratified_sample(
    rows: Sequence[dict[str, str]],
    *,
    seed: str,
    sample_size: int,
) -> list[dict[str, str]]:
    """Draw at most `sample_size` cases, proportionally across strata."""
    budget = min(max(sample_size, 1), MAX_REVIEW_SAMPLE, len(rows))
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(stratum(row), []).append(row)

    selected: list[dict[str, str]] = []
    for key, count in sorted(_allocate(groups, budget).items()):
        if count <= 0:
            continue
        group = sorted(groups[key], key=lambda row: stable_rank(seed, row))
        selected.extend(group[:count])
    return selected[:budget]


def partition_population(
    rows: Sequence[dict[str, str]],
    *,
    seed: str,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> SamplingPlan:
    """Split rows into a working pool and a holdout, stratum by stratum.

    Each stratum contributes to the holdout so acceptance mirrors the shape of
    the population, but always keeps at least one case in the working pool: a
    stratum the optimisation rounds can never see is a stratum the loop cannot
    fix. Strata of one therefore stay entirely in the working pool, and are
    reported so the shortfall is visible rather than silent.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise SamplingPlanError("holdout_fraction must be between 0 and 1 exclusive")
    population = list(rows)
    if not population:
        raise SamplingPlanError("Cannot build a sampling plan for an empty population")
    validate_stratification(population)

    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in population:
        if not case_id(row):
            raise SamplingPlanError("Population contains a row without an oachargeid")
        groups.setdefault(stratum(row), []).append(row)

    holdout_seed = f"{seed}\0holdout"
    working: list[str] = []
    holdout: list[str] = []
    unrepresented: list[str] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: stable_rank(holdout_seed, row))
        # ceil keeps small strata represented; the cap keeps one case workable.
        reserved = min(math.ceil(len(group) * holdout_fraction), len(group) - 1)
        if reserved <= 0:
            unrepresented.append("|".join(key))
        holdout.extend(case_id(row) for row in group[:reserved])
        working.extend(case_id(row) for row in group[reserved:])

    return SamplingPlan(
        seed=seed,
        holdout_fraction=holdout_fraction,
        working_ids=tuple(working),
        holdout_ids=tuple(holdout),
        strata_without_holdout=tuple(unrepresented),
    )


def _subset(rows: Sequence[dict[str, str]], keep: Iterable[str]) -> list[dict[str, str]]:
    wanted = set(keep)
    return [row for row in rows if case_id(row) in wanted]


def select_round(
    rows: Sequence[dict[str, str]],
    plan: SamplingPlan,
    *,
    round_index: int,
    sample_size: int,
    already_sampled: Iterable[str] = (),
) -> list[dict[str, str]]:
    """Draw one optimisation round from the working pool, without replacement.

    Cases reviewed in earlier rounds are excluded, so a later round cannot
    re-confirm an earlier fix instead of testing new ground.
    """
    if round_index < 1:
        raise SamplingPlanError("round_index starts at 1")
    if sample_size < 1:
        raise SamplingPlanError("sample_size must be at least 1")
    seen = set(already_sampled)
    available = [row for row in _subset(rows, plan.working_ids) if case_id(row) not in seen]
    if not available:
        return []
    return bounded_stratified_sample(
        available,
        seed=f"{plan.seed}\0round\0{round_index}",
        sample_size=sample_size,
    )


def select_acceptance(
    rows: Sequence[dict[str, str]],
    plan: SamplingPlan,
    *,
    sample_size: int,
) -> list[dict[str, str]]:
    """Draw the final acceptance sample from the holdout.

    This is the only sample whose result describes the mapping rather than the
    loop's ability to satisfy its own reviews, because no adjustment was ever
    made in response to these cases.
    """
    if sample_size < 1:
        raise SamplingPlanError("sample_size must be at least 1")
    available = _subset(rows, plan.holdout_ids)
    if not available:
        raise SamplingPlanError(
            "The holdout is empty; every stratum was too small to reserve a case. "
            "Acceptance cannot be measured independently for this population."
        )
    return bounded_stratified_sample(
        available,
        seed=f"{plan.seed}\0acceptance",
        sample_size=sample_size,
    )
