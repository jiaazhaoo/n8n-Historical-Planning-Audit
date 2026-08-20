"""A spend ceiling for one content-QA round.

Every reviewed case costs a vision call whose price depends on how many pages
the scan has, so a fixed case count is not a fixed cost. This tracks the actual
spend reported by the provider and stops the round before it goes over, rather
than discovering the overrun on a bill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_LIMIT_USD = 3.0

# Used only until the round has priced a real case; observed cost takes over
# from the second case onward. Measured 2026-08-20 on google/gemini-3.7-flash:
# $0.0021 for a one-page case, $0.0070 for a twelve-page case. Held above the
# measured worst case because under-estimating spends money while
# over-estimating only reviews fewer cases.
DEFAULT_ESTIMATE_USD_PER_CASE = 0.01


class BudgetExhausted(RuntimeError):
    """Raised instead of making a call that would exceed the ceiling."""


@dataclass
class ReviewBudget:
    limit_usd: float = DEFAULT_LIMIT_USD
    estimate_usd_per_case: float = DEFAULT_ESTIMATE_USD_PER_CASE
    spent_usd: float = 0.0
    calls: int = 0
    unpriced_calls: int = 0
    truncated_cases: int = 0
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.limit_usd <= 0:
            raise ValueError("limit_usd must be positive")
        if self.estimate_usd_per_case <= 0:
            raise ValueError("estimate_usd_per_case must be positive")

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)

    @property
    def observed_usd_per_case(self) -> float | None:
        priced = self.calls - self.unpriced_calls
        if priced <= 0:
            return None
        return self.spent_usd / priced

    @property
    def projected_usd_per_case(self) -> float:
        """What the next case is expected to cost.

        Once real prices are in, they beat any prior estimate; the estimate only
        covers the first case of a round.
        """
        observed = self.observed_usd_per_case
        return observed if observed is not None else self.estimate_usd_per_case

    def affordable_cases(self) -> int:
        return int(self.remaining_usd // self.projected_usd_per_case)

    def ensure_affordable(self) -> None:
        if self.remaining_usd < self.projected_usd_per_case:
            raise BudgetExhausted(
                f"Review budget of ${self.limit_usd:.2f} is spent "
                f"(${self.spent_usd:.4f} over {self.calls} calls); "
                f"the next case is projected at ${self.projected_usd_per_case:.4f}"
            )

    def record(self, cost_usd: float | None) -> None:
        """Record one call. A provider that reports no cost is counted, not guessed at."""
        self.calls += 1
        if cost_usd is None:
            self.unpriced_calls += 1
            # Charge the running estimate so an unpriced provider cannot make the
            # round look free and run past the ceiling.
            self.spent_usd += self.projected_usd_per_case
            self.notes.append("A call returned no cost; the running estimate was charged instead")
            return
        if cost_usd < 0:
            raise ValueError("cost_usd cannot be negative")
        self.spent_usd += cost_usd

    def note_truncation(self, dropped: int) -> None:
        if dropped <= 0:
            return
        self.truncated_cases += dropped
        self.notes.append(
            f"{dropped} case(s) were dropped from the sample to stay within "
            f"${self.limit_usd:.2f}; the review is smaller than requested"
        )

    def describe(self) -> dict[str, Any]:
        return {
            "limit_usd": round(self.limit_usd, 4),
            "spent_usd": round(self.spent_usd, 4),
            "remaining_usd": round(self.remaining_usd, 4),
            "calls": self.calls,
            "unpriced_calls": self.unpriced_calls,
            "observed_usd_per_case": (
                round(self.observed_usd_per_case, 5)
                if self.observed_usd_per_case is not None
                else None
            ),
            "truncated_cases": self.truncated_cases,
            "notes": list(dict.fromkeys(self.notes)),
        }
