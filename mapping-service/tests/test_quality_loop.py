from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

AUTONOMOUS_ROOT = Path(__file__).resolve().parents[1] / "amazons3-mapping"
if str(AUTONOMOUS_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTONOMOUS_ROOT))

from autonomous.quality_gate import (  # noqa: E402
    GateThresholds,
    build_signatures,
    evaluate_coverage,
    evaluate_gate,
)
from autonomous.quality_loop import (  # noqa: E402
    ACCEPTANCE,
    WORKING,
    QualityLoopError,
    next_action,
    open_state,
    run_quality_round,
    write_report,
)
from autonomous.quality_report import RoundRecord  # noqa: E402
from autonomous.sampling_plan import (  # noqa: E402
    MAX_REVIEW_SAMPLE,
    SamplingPlanError,
    bounded_stratified_sample,
    validate_stratification,
    case_id,
    partition_population,
    select_acceptance,
    select_round,
)
from autonomous.qa import stratum  # noqa: E402


def audit_row(
    oid: str,
    *,
    route: str = "s3",
    match_status: str = "accepted_rule",
    match_basis: str = "reference",
    confidence: str = "0.80",
    candidates: str = "1",
) -> dict[str, str]:
    return {
        "oachargeid": oid,
        "route": route,
        "match_status": match_status,
        "match_basis": match_basis,
        "decision_confidence": confidence,
        "candidate_count": candidates,
    }


def population(count: int = 60) -> list[dict[str, str]]:
    rows = []
    for index in range(count):
        rows.append(
            audit_row(
                f"CASE-{index:03d}",
                route="s3" if index % 3 else "portal",
                match_basis="reference" if index % 2 else "address",
                confidence="0.80" if index % 4 else "0.40",
            )
        )
    return rows


def case_result(
    oid: str,
    verdict: str,
    *,
    route: str = "s3",
    match_basis: str = "reference",
    signals: dict[str, bool] | None = None,
) -> dict[str, object]:
    return {
        "oachargeid": oid,
        "verdict": verdict,
        "route": route,
        "match_basis": match_basis,
        "mapping_path": f"s3://bucket/{oid}.pdf",
        "reason": "",
        "signals": signals or {"reference": True, "address": True, "description": True, "date": True},
        "selected_images": [],
    }


class StubReviewer:
    """Returns a scripted verdict per case without calling a vision model."""

    def __init__(self, verdicts: dict[str, str], default: str = "verified_same") -> None:
        self.verdicts = verdicts
        self.default = default
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def review(self, *, run_id, include_ids, output_dir):
        self.calls.append((run_id, tuple(include_ids)))
        results = [
            case_result(oid, self.verdicts.get(oid, self.default)) for oid in include_ids
        ]
        return {"run_id": run_id, "selected": len(results)}, results


class SamplingPlanTests(unittest.TestCase):
    def test_holdout_and_working_pools_are_disjoint_and_cover_population(self) -> None:
        rows = population()
        plan = partition_population(rows, seed="exeter:wp1")
        self.assertEqual(
            set(plan.working_ids) | set(plan.holdout_ids),
            {case_id(row) for row in rows},
        )
        self.assertFalse(set(plan.working_ids) & set(plan.holdout_ids))

    def test_every_stratum_keeps_at_least_one_working_case(self) -> None:
        rows = population()
        plan = partition_population(rows, seed="exeter:wp1")
        working = {case_id(row) for row in rows} & set(plan.working_ids)
        strata_with_working = {
            stratum(row) for row in rows if case_id(row) in working
        }
        self.assertEqual(strata_with_working, {stratum(row) for row in rows})

    def test_singleton_stratum_is_reported_as_unrepresented_in_holdout(self) -> None:
        rows = population(4) + [audit_row("LONELY", route="portal", match_basis="date", confidence="0.10")]
        plan = partition_population(rows, seed="exeter:wp1")
        self.assertIn("LONELY", plan.working_ids)
        self.assertTrue(plan.strata_without_holdout)

    def test_partition_is_deterministic_for_the_same_seed(self) -> None:
        rows = population()
        first = partition_population(rows, seed="exeter:wp1")
        second = partition_population(list(reversed(rows)), seed="exeter:wp1")
        self.assertEqual(set(first.holdout_ids), set(second.holdout_ids))

    def test_rounds_never_resample_a_reviewed_case(self) -> None:
        rows = population()
        plan = partition_population(rows, seed="exeter:wp1")
        first = select_round(rows, plan, round_index=1, sample_size=8)
        first_ids = [case_id(row) for row in first]
        second = select_round(
            rows, plan, round_index=2, sample_size=8, already_sampled=first_ids
        )
        second_ids = [case_id(row) for row in second]
        self.assertFalse(set(first_ids) & set(second_ids))

    def test_rounds_only_draw_from_the_working_pool(self) -> None:
        rows = population()
        plan = partition_population(rows, seed="exeter:wp1")
        selected = select_round(rows, plan, round_index=1, sample_size=10)
        self.assertTrue({case_id(row) for row in selected} <= set(plan.working_ids))

    def test_acceptance_only_draws_from_the_holdout(self) -> None:
        rows = population()
        plan = partition_population(rows, seed="exeter:wp1")
        selected = select_acceptance(rows, plan, sample_size=6)
        self.assertTrue({case_id(row) for row in selected} <= set(plan.holdout_ids))

    def test_empty_population_is_rejected(self) -> None:
        with self.assertRaises(SamplingPlanError):
            partition_population([], seed="exeter:wp1")


class BoundedSamplingTests(unittest.TestCase):
    def test_sample_never_exceeds_the_requested_size(self) -> None:
        rows = population(400)
        selected = bounded_stratified_sample(rows, seed="s", sample_size=15)
        self.assertEqual(len(selected), 15)

    def test_many_singleton_strata_do_not_inflate_the_sample(self) -> None:
        # One stratum per case is the shape that previously forced the sample to
        # cover the whole population, one case per stratum.
        rows = [
            audit_row(f"CASE-{index:04d}", confidence=f"0.{index:02d}", candidates=str(index))
            for index in range(300)
        ]
        selected = bounded_stratified_sample(rows, seed="s", sample_size=12)
        self.assertEqual(len(selected), 12)

    def test_sample_is_capped_at_the_review_ceiling(self) -> None:
        rows = population(500)
        selected = bounded_stratified_sample(rows, seed="s", sample_size=10_000)
        self.assertEqual(len(selected), MAX_REVIEW_SAMPLE)

    def test_allocation_follows_stratum_size(self) -> None:
        rows = [audit_row(f"BIG-{index}", route="s3") for index in range(90)]
        rows += [audit_row(f"SMALL-{index}", route="portal") for index in range(10)]
        selected = bounded_stratified_sample(rows, seed="s", sample_size=20)
        big = sum(1 for row in selected if row["oachargeid"].startswith("BIG"))
        self.assertEqual(len(selected), 20)
        self.assertGreater(big, 12)

    def test_sample_is_deterministic_for_a_seed(self) -> None:
        rows = population(200)
        first = bounded_stratified_sample(rows, seed="s", sample_size=15)
        second = bounded_stratified_sample(rows, seed="s", sample_size=15)
        self.assertEqual([case_id(row) for row in first], [case_id(row) for row in second])


class StratificationValidationTests(unittest.TestCase):
    def test_per_case_stratum_values_are_refused(self) -> None:
        # A legacy council audit stores the matched reference in match_status.
        rows = [
            audit_row(f"CASE-{index:04d}", match_status=f"matched:92/{index:04d}P")
            for index in range(100)
        ]
        with self.assertRaises(SamplingPlanError) as caught:
            validate_stratification(rows)
        self.assertIn("not informative", str(caught.exception))

    def test_a_normal_audit_passes_validation(self) -> None:
        validate_stratification(population(200))

    def test_small_populations_skip_the_check(self) -> None:
        rows = [audit_row(f"CASE-{index}", match_status=f"matched:{index}") for index in range(5)]
        validate_stratification(rows)

    def test_partition_refuses_an_uninformative_population(self) -> None:
        rows = [
            audit_row(f"CASE-{index:04d}", match_status=f"matched:92/{index:04d}P")
            for index in range(100)
        ]
        with self.assertRaises(SamplingPlanError):
            partition_population(rows, seed="sheffield:wp3")


class QualityGateTests(unittest.TestCase):
    def test_a_single_confirmed_wrong_case_fails_the_gate(self) -> None:
        results = [case_result(f"CASE-{index}", "verified_same") for index in range(19)]
        results.append(case_result("CASE-19", "verified_wrong"))
        outcome = evaluate_gate(results)
        self.assertFalse(outcome.passed)
        self.assertTrue(any("confirmed wrong" in reason for reason in outcome.reasons))

    def test_all_verified_passes(self) -> None:
        results = [case_result(f"CASE-{index}", "verified_same") for index in range(10)]
        self.assertTrue(evaluate_gate(results).passed)

    def test_repeated_failures_on_one_route_and_basis_are_systematic(self) -> None:
        failing = {"reference": False, "address": True, "description": True, "date": True}
        results = [
            case_result("A", "verified_wrong", route="s3", match_basis="reference", signals=failing),
            case_result("B", "verified_wrong", route="s3", match_basis="reference", signals=failing),
        ]
        signatures = build_signatures(results)
        self.assertEqual(len(signatures), 1)
        self.assertTrue(signatures[0].systematic)
        self.assertEqual(signatures[0].failing_signals, ("reference",))

    def test_differing_failures_are_not_treated_as_systematic(self) -> None:
        results = [
            case_result("A", "verified_wrong", route="s3", match_basis="reference"),
            case_result("B", "verified_wrong", route="portal", match_basis="address"),
        ]
        self.assertFalse(any(item.systematic for item in build_signatures(results)))

    def test_not_applicable_cases_are_excluded_from_the_rate(self) -> None:
        results = [case_result("A", "verified_same"), case_result("B", "not_applicable")]
        outcome = evaluate_gate(results)
        self.assertEqual(outcome.metrics["judged"], 1)
        self.assertEqual(outcome.metrics["verified_rate"], 1.0)

    def test_missing_documents_fail_the_gate_separately_from_wrong_mappings(self) -> None:
        results = [case_result(f"CASE-{index}", "verified_same") for index in range(8)]
        results += [case_result("X", "missing_document"), case_result("Y", "missing_document")]
        outcome = evaluate_gate(results)
        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.metrics["verified_wrong"], 0)
        self.assertTrue(any("Missing-document" in reason for reason in outcome.reasons))

    def test_an_empty_sample_never_passes(self) -> None:
        outcome = evaluate_gate([])
        self.assertFalse(outcome.passed)

    def test_thresholds_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_gate([], thresholds=GateThresholds(min_verified_rate=0.0))


class CoverageGateTests(unittest.TestCase):
    def test_an_accurate_but_thin_mapping_fails_coverage(self) -> None:
        # The precision gate passes this: everything it accepted was correct.
        results = [case_result(f"C{index}", "verified_same") for index in range(3)]
        results += [case_result(f"R{index}", "not_applicable") for index in range(17)]
        self.assertTrue(evaluate_gate(results).passed)

        coverage = evaluate_coverage(
            {"found": 3, "no_found": 2, "no_match": 15}, min_accepted_rate=0.5
        )
        self.assertFalse(coverage.passed)
        self.assertTrue(any("resolved too little" in reason for reason in coverage.reasons))

    def test_cases_the_source_says_have_no_scan_are_not_held_against_the_spec(self) -> None:
        coverage = evaluate_coverage({"found": 40, "no_found": 55, "no_match": 5})
        self.assertAlmostEqual(coverage.metrics["accepted_rate"], 0.40)
        self.assertAlmostEqual(coverage.metrics["accepted_rate_of_resolvable"], 40 / 45, places=4)

    def test_coverage_is_reported_even_when_no_floor_is_set(self) -> None:
        coverage = evaluate_coverage({"found": 1, "no_found": 0, "no_match": 99})
        self.assertTrue(coverage.passed)
        self.assertAlmostEqual(coverage.metrics["unmatched_rate"], 0.99)

    def test_an_unmatched_ceiling_can_fail_the_run(self) -> None:
        coverage = evaluate_coverage(
            {"found": 1, "no_found": 0, "no_match": 99}, max_unmatched_rate=0.2
        )
        self.assertFalse(coverage.passed)

    def test_an_empty_population_cannot_be_judged(self) -> None:
        self.assertFalse(evaluate_coverage({}).passed)


class LoopStateTests(unittest.TestCase):
    def test_holdout_stays_frozen_when_the_mapping_changes_strata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "state.json"
            rows = population()
            state = open_state(
                state_path, council="exeter", batch="wp1", audit_rows=rows
            )
            original_holdout = set(state.holdout_ids)

            # A spec change reroutes and rescores every case, so strata shift.
            changed = [
                dict(row, route="portal", match_basis="address", decision_confidence="0.95")
                for row in rows
            ]
            reopened = open_state(
                state_path, council="exeter", batch="wp1", audit_rows=changed
            )
            self.assertEqual(set(reopened.holdout_ids), original_holdout)

    def test_cases_new_to_a_later_mapping_join_the_working_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "state.json"
            rows = population(20)
            open_state(state_path, council="exeter", batch="wp1", audit_rows=rows)
            extended = rows + [audit_row("BRAND-NEW")]
            reopened = open_state(
                state_path, council="exeter", batch="wp1", audit_rows=extended
            )
            self.assertIn("BRAND-NEW", reopened.working_ids)
            self.assertNotIn("BRAND-NEW", reopened.holdout_ids)

    def test_state_from_another_batch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "state.json"
            rows = population(20)
            open_state(state_path, council="exeter", batch="wp1", audit_rows=rows)
            with self.assertRaises(QualityLoopError):
                open_state(state_path, council="exeter", batch="wp4", audit_rows=rows)

    def test_duplicate_case_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "state.json"
            rows = population(10) + [audit_row("CASE-000")]
            with self.assertRaises(QualityLoopError):
                open_state(state_path, council="exeter", batch="wp1", audit_rows=rows)


class QualityRoundTests(unittest.TestCase):
    def _setup(self, temporary: str, rows):
        state_path = Path(temporary) / "state.json"
        state = open_state(state_path, council="exeter", batch="wp1", audit_rows=rows)
        return state_path, state, Path(temporary) / "artifacts"

    def test_a_clean_round_passes_and_asks_for_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = population()
            state_path, state, artifacts = self._setup(temporary, rows)
            result = run_quality_round(
                state=state,
                state_path=state_path,
                audit_rows=rows,
                reviewer=StubReviewer({}),
                artifacts_root=artifacts,
                sample_size=8,
            )
            self.assertTrue(result.outcome.passed)
            self.assertEqual(next_action(result)["action"], "accept")

    def test_a_systematic_failure_asks_for_a_spec_adjustment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = population()
            state_path, state, artifacts = self._setup(temporary, rows)
            plan = state.plan()
            sampled = [case_id(row) for row in select_round(rows, plan, round_index=1, sample_size=8)]
            reviewer = StubReviewer({sampled[0]: "verified_wrong", sampled[1]: "verified_wrong"})
            result = run_quality_round(
                state=state,
                state_path=state_path,
                audit_rows=rows,
                reviewer=reviewer,
                artifacts_root=artifacts,
                sample_size=8,
            )
            action = next_action(result)
            self.assertFalse(result.outcome.passed)
            self.assertEqual(action["action"], "adjust_spec")
            self.assertTrue(action["focus"])

    def test_consecutive_rounds_review_different_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = population()
            state_path, state, artifacts = self._setup(temporary, rows)
            reviewer = StubReviewer({})
            first = run_quality_round(
                state=state, state_path=state_path, audit_rows=rows,
                reviewer=reviewer, artifacts_root=artifacts, sample_size=8,
            )
            second = run_quality_round(
                state=state, state_path=state_path, audit_rows=rows,
                reviewer=reviewer, artifacts_root=artifacts, sample_size=8,
            )
            self.assertEqual(second.index, 2)
            self.assertFalse(set(first.sampled_ids) & set(second.sampled_ids))

    def test_acceptance_reviews_only_holdout_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = population()
            state_path, state, artifacts = self._setup(temporary, rows)
            reviewer = StubReviewer({})
            run_quality_round(
                state=state, state_path=state_path, audit_rows=rows,
                reviewer=reviewer, artifacts_root=artifacts, sample_size=8,
            )
            acceptance = run_quality_round(
                state=state, state_path=state_path, audit_rows=rows,
                reviewer=reviewer, artifacts_root=artifacts,
                stage=ACCEPTANCE, sample_size=6,
            )
            self.assertTrue(set(acceptance.sampled_ids) <= set(state.holdout_ids))
            self.assertEqual(next_action(acceptance)["action"], "publish")

    def test_exhausted_working_pool_stops_instead_of_resampling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = population(12)
            state_path, state, artifacts = self._setup(temporary, rows)
            reviewer = StubReviewer({})
            for _ in range(6):
                result = run_quality_round(
                    state=state, state_path=state_path, audit_rows=rows,
                    reviewer=reviewer, artifacts_root=artifacts, sample_size=10,
                )
                if result.exhausted:
                    break
            self.assertTrue(result.exhausted)
            self.assertEqual(next_action(result)["action"], "stop_exhausted")

    def test_state_records_every_round_for_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = population()
            state_path, state, artifacts = self._setup(temporary, rows)
            run_quality_round(
                state=state, state_path=state_path, audit_rows=rows,
                reviewer=StubReviewer({}), artifacts_root=artifacts, sample_size=8,
            )
            saved = json.loads(state_path.read_text())
            self.assertEqual(len(saved["rounds"]), 1)
            self.assertIn("metrics", saved["rounds"][0])


class QualityReportTests(unittest.TestCase):
    def test_report_folder_contains_html_and_machine_readable_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = population()
            state_path = Path(temporary) / "state.json"
            state = open_state(state_path, council="exeter", batch="wp1", audit_rows=rows)
            results = tuple(case_result(f"CASE-{index:03d}", "verified_same") for index in range(4))
            record = RoundRecord(
                index=1, label="round 1", case_results=results, outcome=evaluate_gate(results)
            )
            destination = Path(temporary) / "quality-report"
            write_report(
                destination=destination,
                state=state,
                rounds=[record],
                acceptance=record,
                mapping_summary={"case_count": len(rows)},
            )
            html = (destination / "index.html").read_text(encoding="utf-8")
            summary = json.loads((destination / "quality-report.json").read_text())
            self.assertIn("Passed on the holdout sample", html)
            self.assertIn("CASE-000", html)
            self.assertTrue(summary["passed"])
            self.assertTrue(summary["acceptance_reviewed"])

    def test_report_says_so_when_no_holdout_was_reviewed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = population()
            state_path = Path(temporary) / "state.json"
            state = open_state(state_path, council="exeter", batch="wp1", audit_rows=rows)
            results = (case_result("CASE-000", "verified_same"),)
            record = RoundRecord(
                index=1, label="round 1", case_results=results, outcome=evaluate_gate(results)
            )
            destination = Path(temporary) / "quality-report"
            write_report(
                destination=destination,
                state=state,
                rounds=[record],
                acceptance=None,
                mapping_summary={"case_count": len(rows)},
            )
            html = (destination / "index.html").read_text(encoding="utf-8")
            self.assertIn("No independent acceptance sample", html)
            self.assertFalse(json.loads((destination / "quality-report.json").read_text())["passed"])

    def test_a_short_coverage_downgrades_a_passing_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = population()
            state_path = Path(temporary) / "state.json"
            state = open_state(state_path, council="exeter", batch="wp1", audit_rows=rows)
            results = tuple(case_result(f"CASE-{i:03d}", "verified_same") for i in range(4))
            record = RoundRecord(
                index=1, label="round 1", case_results=results, outcome=evaluate_gate(results)
            )
            destination = Path(temporary) / "quality-report"
            write_report(
                destination=destination,
                state=state,
                rounds=[record],
                acceptance=record,
                mapping_summary={
                    "case_count": 100,
                    "coverage": evaluate_coverage(
                        {"found": 15, "no_found": 5, "no_match": 80}, min_accepted_rate=0.5
                    ).describe(),
                },
            )
            html = (destination / "index.html").read_text(encoding="utf-8")
            summary = json.loads((destination / "quality-report.json").read_text())
            self.assertIn("coverage is short", html)
            self.assertFalse(summary["passed"])

    def test_report_escapes_case_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = population()
            state_path = Path(temporary) / "state.json"
            state = open_state(state_path, council="exeter", batch="wp1", audit_rows=rows)
            hostile = case_result("CASE-000", "verified_same")
            hostile["mapping_path"] = "<script>alert(1)</script>"
            record = RoundRecord(
                index=1, label="round 1", case_results=(hostile,),
                outcome=evaluate_gate([hostile]),
            )
            destination = Path(temporary) / "quality-report"
            write_report(
                destination=destination, state=state, rounds=[record],
                acceptance=None, mapping_summary={},
            )
            html = (destination / "index.html").read_text(encoding="utf-8")
            self.assertNotIn("<script>alert(1)</script>", html)
            self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()
