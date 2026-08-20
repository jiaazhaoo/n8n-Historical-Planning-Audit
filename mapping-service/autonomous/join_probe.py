"""Dry-run a key derivation over real evidence and predict what it can deliver.

Running a mapping to find out how much of it matched is expensive: it compiles a
spec, executes the engine, and only then does a sampled content review report
that coverage was thin. A derivation that parses and formats can instead be run
over the source table and the folder inventory directly, with no model call and
no document fetch, and answer the question that actually decides feasibility:
how many cases would resolve, and where do the rest fall over.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .key_derivation import KeyDerivation, index_inventory


@dataclass(frozen=True)
class Precedence:
    """Order competing candidates by a field that ranks them.

    Competing candidates are usually not ambiguous, they are graded. Test Valley
    holds the same reference as both microfiche and paper and the capture rules
    make fiche primary; Braintree's portal evidence holds both verified official
    documents and unverified search hits for one reference. Same primitive, a
    different field each time. Whatever the order cannot separate stays
    genuinely ambiguous and is still rejected.
    """

    field_name: str = "parent_prefix"
    order: tuple[str, ...] = ()

    def rank(self, row: dict[str, str]) -> int:
        value = str(row.get(self.field_name) or "")
        for index, fragment in enumerate(self.order):
            if fragment.casefold() in value.casefold():
                return index
        return len(self.order)

    def resolve(self, rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
        if not self.order:
            return list(rows)
        best = min(self.rank(row) for row in rows)
        return [row for row in rows if self.rank(row) == best]


@dataclass
class JoinReport:
    council: str
    batch: str
    derivation: dict[str, Any]
    source_rows: int = 0
    source_keyed: int = 0
    inventory_rows: int = 0
    inventory_keyed: int = 0
    zero_candidates: int = 0
    unique_candidates: int = 0
    multi_candidates: int = 0
    resolved_by_precedence: int = 0
    still_ambiguous: int = 0
    source_unparsed_examples: list[str] = field(default_factory=list)
    inventory_unparsed_examples: list[str] = field(default_factory=list)
    ambiguous_examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def resolvable(self) -> int:
        """Cases a mapping built on this derivation could actually accept."""
        return self.unique_candidates + self.resolved_by_precedence

    def _rate(self, value: int) -> float:
        return round(value / self.source_rows, 4) if self.source_rows else 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "council": self.council,
            "batch": self.batch,
            "derivation": self.derivation,
            "source": {
                "rows": self.source_rows,
                "keyed": self.source_keyed,
                "keyed_rate": self._rate(self.source_keyed),
                "unparsed_examples": self.source_unparsed_examples[:5],
            },
            "inventory": {
                "rows": self.inventory_rows,
                "keyed": self.inventory_keyed,
                "keyed_rate": (
                    round(self.inventory_keyed / self.inventory_rows, 4)
                    if self.inventory_rows
                    else 0.0
                ),
                "unparsed_examples": self.inventory_unparsed_examples[:5],
            },
            "join": {
                "zero_candidates": self.zero_candidates,
                "unique_candidates": self.unique_candidates,
                "multi_candidates": self.multi_candidates,
                "resolved_by_precedence": self.resolved_by_precedence,
                "still_ambiguous": self.still_ambiguous,
                "ambiguous_examples": self.ambiguous_examples[:5],
            },
            "predicted": {
                # What a unique-candidate-only engine would accept today.
                "resolvable_without_precedence": self.unique_candidates,
                "resolvable_without_precedence_rate": self._rate(self.unique_candidates),
                # What the same derivation would accept once precedence exists.
                "resolvable": self.resolvable,
                "resolvable_rate": self._rate(self.resolvable),
            },
        }


def probe_join(
    source_rows: Iterable[dict[str, str]],
    inventory_rows: Iterable[dict[str, str]],
    derivation: KeyDerivation,
    *,
    reference_field: str,
    folder_field: str = "folder",
    council: str = "",
    batch: str = "",
    precedence: Precedence | None = None,
) -> JoinReport:
    inventory = list(inventory_rows)
    index, unparsed_inventory = index_inventory(
        inventory, derivation, folder_field=folder_field
    )
    report = JoinReport(
        council=council,
        batch=batch,
        derivation=derivation.describe(),
        inventory_rows=len(inventory),
        inventory_keyed=len(inventory) - len(unparsed_inventory),
        inventory_unparsed_examples=unparsed_inventory[:5],
    )

    precedence = precedence or Precedence()
    for row in source_rows:
        report.source_rows += 1
        reference = str(row.get(reference_field) or "")
        key = derivation.source_key(reference)
        if key is None:
            if len(report.source_unparsed_examples) < 5:
                report.source_unparsed_examples.append(reference)
            report.zero_candidates += 1
            continue
        report.source_keyed += 1
        candidates = index.get(key, [])
        if not candidates:
            report.zero_candidates += 1
        elif len(candidates) == 1:
            report.unique_candidates += 1
        else:
            report.multi_candidates += 1
            remaining = precedence.resolve(candidates)
            if len(remaining) == 1:
                report.resolved_by_precedence += 1
            else:
                report.still_ambiguous += 1
                if len(report.ambiguous_examples) < 5:
                    report.ambiguous_examples.append(
                        {
                            "reference": reference,
                            "key": list(key),
                            "candidates": [
                                str(item.get(folder_field) or "") for item in remaining[:4]
                            ],
                            "locations": sorted(
                                {str(item.get(precedence.field_name) or "") for item in remaining}
                            )[:4],
                        }
                    )
    return report


def summarise(reports: Sequence[JoinReport]) -> str:
    """A one-line-per-probe table, for deciding which councils the spec can serve."""
    header = (
        f"{'council/batch':<26}{'源可解析':>10}{'唯一命中':>10}"
        f"{'优先级消解':>12}{'仍歧义':>9}{'预测覆盖':>10}"
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        data = report.describe()
        lines.append(
            f"{(report.council + '/' + report.batch)[:25]:<26}"
            f"{data['source']['keyed_rate']:>9.1%}"
            f"{report._rate(report.unique_candidates):>10.1%}"
            f"{report._rate(report.resolved_by_precedence):>12.1%}"
            f"{report._rate(report.still_ambiguous):>9.1%}"
            f"{data['predicted']['resolvable_rate']:>10.1%}"
        )
    return "\n".join(lines)
