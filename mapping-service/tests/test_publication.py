from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

AUTONOMOUS_ROOT = Path(__file__).resolve().parents[1] / "amazons3-mapping"
if str(AUTONOMOUS_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTONOMOUS_ROOT))

from autonomous.publication import (  # noqa: E402
    PublicationError,
    clearance,
    publish_mapping,
    runtime_row,
)


RUNTIME_FIELDS = [
    "oachargeid",
    "amazons3_path",
    "amazons3_confidence",
    "portal_path",
    "file_count",
    "file_found",
    "further-information-reference",
    "local_portal_path",
    "local_amazon_path",
]


def write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def runtime_table(path: Path, rows) -> None:
    write_csv(path, RUNTIME_FIELDS, rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def existing(oachargeid, **overrides):
    row = {name: "" for name in RUNTIME_FIELDS}
    row["oachargeid"] = oachargeid
    row.update(overrides)
    return row


def mapped(oachargeid, *, path="", confidence="0.74", found="found", reference=""):
    return {
        "originating-authority-charge-identifier": oachargeid,
        "amazons3_path": path,
        "amazons3_path_cfd": confidence,
        "portal_path": "",
        "path_found": found,
        "further-information-reference": reference,
    }


def quality_report(path: Path, *, passed=True, reviewed=True, rate=0.964) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "quality-report.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "acceptance_reviewed": reviewed,
                "rounds": [
                    {"label": "acceptance · holdout", "outcome": {"metrics": {"verified_rate": rate}}}
                ],
            }
        ),
        encoding="utf-8",
    )


class RuntimeRowTests(unittest.TestCase):
    def test_a_found_mapping_becomes_a_runtime_row(self) -> None:
        row = runtime_row(mapped("A", path="s3://b/a", reference="88/1061/FUL"))
        self.assertEqual(row["amazons3_path"], "s3://b/a")
        self.assertEqual(row["file_found"], "yes")
        self.assertEqual(row["further-information-reference"], "88/1061/FUL")

    def test_an_unmapped_case_is_marked_as_having_no_scan(self) -> None:
        self.assertEqual(runtime_row(mapped("A", found="no_found"))["file_found"], "no scan")


class ClearanceTests(unittest.TestCase):
    def test_a_run_without_a_quality_report_cannot_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(PublicationError):
                clearance(Path(temporary))

    def test_working_rounds_alone_do_not_clear_a_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            quality_report(run / "quality-report", passed=False, reviewed=False)
            with self.assertRaises(PublicationError) as caught:
                clearance(run)
            self.assertIn("holdout acceptance", str(caught.exception))

    def test_a_failed_acceptance_does_not_clear_a_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            quality_report(run / "quality-report", passed=False, reviewed=True)
            with self.assertRaises(PublicationError):
                clearance(run)


class PublishMappingTests(unittest.TestCase):
    def _setup(self, temporary, runtime_rows, mapping_rows, **report):
        root = Path(temporary)
        runtime = root / "file-browser-data" / "exeter" / "exeter-matching.csv"
        runtime_table(runtime, runtime_rows)
        mapping = root / "run" / "exeter-wp3-mapping.csv"
        write_csv(mapping, list(mapping_rows[0]), mapping_rows)
        run = root / "run"
        quality_report(run / "quality-report", **report)
        return runtime, mapping, run

    def test_only_the_batch_rows_change(self) -> None:
        # Exeter's runtime table holds four work packages; a WP3 run must not
        # remove the other three.
        with tempfile.TemporaryDirectory() as temporary:
            runtime, mapping, run = self._setup(
                temporary,
                [
                    existing("WP1-A", amazons3_path="s3://old/wp1"),
                    existing("WP3-A", amazons3_path="s3://old/wp3"),
                    existing("WP4-A", amazons3_path="s3://old/wp4"),
                ],
                [mapped("WP3-A", path="s3://new/wp3")],
            )
            result = publish_mapping(
                council="exeter", batch="wp3", mapping_csv=mapping,
                runtime_csv=runtime, run_directory=run,
            )
            rows = {r["oachargeid"]: r for r in read_rows(runtime)}
            self.assertEqual(rows["WP3-A"]["amazons3_path"], "s3://new/wp3")
            self.assertEqual(rows["WP1-A"]["amazons3_path"], "s3://old/wp1")
            self.assertEqual(rows["WP4-A"]["amazons3_path"], "s3://old/wp4")
            self.assertEqual((result.rows_total, result.rows_updated), (3, 1))

    def test_runtime_owned_columns_survive_publication(self) -> None:
        # file_count and the local_* paths record what the downloader fetched.
        with tempfile.TemporaryDirectory() as temporary:
            runtime, mapping, run = self._setup(
                temporary,
                [existing("A", file_count="12", local_amazon_path="/data/x", amazons3_path="s3://old")],
                [mapped("A", path="s3://new")],
            )
            publish_mapping(
                council="exeter", batch="wp3", mapping_csv=mapping,
                runtime_csv=runtime, run_directory=run,
            )
            row = read_rows(runtime)[0]
            self.assertEqual(row["amazons3_path"], "s3://new")
            self.assertEqual(row["file_count"], "12")
            self.assertEqual(row["local_amazon_path"], "/data/x")

    def test_the_previous_table_is_kept_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime, mapping, run = self._setup(
                temporary,
                [existing("A", amazons3_path="s3://old")],
                [mapped("A", path="s3://new")],
            )
            result = publish_mapping(
                council="exeter", batch="wp3", mapping_csv=mapping,
                runtime_csv=runtime, run_directory=run,
            )
            self.assertTrue(result.backup_path.is_file())
            backed = read_rows(result.backup_path)[0]
            self.assertEqual(backed["amazons3_path"], "s3://old")

    def test_a_charge_the_runtime_has_not_seen_is_added(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime, mapping, run = self._setup(
                temporary, [existing("A")], [mapped("NEW", path="s3://new")]
            )
            result = publish_mapping(
                council="exeter", batch="wp3", mapping_csv=mapping,
                runtime_csv=runtime, run_directory=run,
            )
            self.assertEqual(result.rows_added, 1)
            rows = {r["oachargeid"] for r in read_rows(runtime)}
            self.assertEqual(rows, {"A", "NEW"})

    def test_an_unchanged_row_is_counted_not_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime, mapping, run = self._setup(
                temporary,
                [existing("A", amazons3_path="s3://same", amazons3_confidence="0.74", file_found="yes")],
                [mapped("A", path="s3://same")],
            )
            result = publish_mapping(
                council="exeter", batch="wp3", mapping_csv=mapping,
                runtime_csv=runtime, run_directory=run,
            )
            self.assertEqual((result.rows_updated, result.rows_unchanged), (0, 1))

    def test_a_dry_run_reports_without_touching_the_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime, mapping, run = self._setup(
                temporary, [existing("A", amazons3_path="s3://old")], [mapped("A", path="s3://new")]
            )
            before = runtime.read_bytes()
            result = publish_mapping(
                council="exeter", batch="wp3", mapping_csv=mapping,
                runtime_csv=runtime, run_directory=run, dry_run=True,
            )
            self.assertEqual(result.rows_updated, 1)
            self.assertEqual(runtime.read_bytes(), before)
            self.assertFalse(result.backup_path.exists())

    def test_an_uncleared_run_cannot_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime, mapping, run = self._setup(
                temporary, [existing("A")], [mapped("A", path="s3://new")], passed=False
            )
            with self.assertRaises(PublicationError):
                publish_mapping(
                    council="exeter", batch="wp3", mapping_csv=mapping,
                    runtime_csv=runtime, run_directory=run,
                )

    def test_a_duplicated_runtime_identifier_stops_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime, mapping, run = self._setup(
                temporary, [existing("A"), existing("A")], [mapped("A", path="s3://new")]
            )
            with self.assertRaises(PublicationError):
                publish_mapping(
                    council="exeter", batch="wp3", mapping_csv=mapping,
                    runtime_csv=runtime, run_directory=run,
                )


if __name__ == "__main__":
    unittest.main()
