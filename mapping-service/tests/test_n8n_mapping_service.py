from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "n8n_mapping_service.py"
SPEC = importlib.util.spec_from_file_location("n8n_mapping_service", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_rows(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class N8nMappingServiceTests(unittest.TestCase):
    def test_select_latest_rule_versions_keeps_distinct_rule_families(self) -> None:
        paths = [
            Path("20260820T110605Z_Exeter-Microfiche-CaptureRules-v1.docx"),
            Path("20260820T110605Z_Exeter-Microfiche-CaptureRules-v2.docx"),
            Path("Exeter-Public Access-CaptureRules.docx"),
        ]
        selected = MODULE.select_latest_rule_versions(paths)
        self.assertEqual(
            [path.name for path in selected],
            [
                "20260820T110605Z_Exeter-Microfiche-CaptureRules-v2.docx",
                "Exeter-Public Access-CaptureRules.docx",
            ],
        )

    def test_export_outputs_emits_primary_full_and_audit_with_complete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            output = root / "output"
            write_rows(
                workspace / "mapping" / "proposed-mapping.csv",
                ("oachargeid", "amazons3_path", "amazons3_confidence", "portal_path"),
                [
                    {
                        "oachargeid": "case-1",
                        "amazons3_path": "s3://bucket/case-1",
                        "amazons3_confidence": "0.74",
                        "portal_path": "",
                    },
                    {
                        "oachargeid": "case-2",
                        "amazons3_path": "",
                        "amazons3_confidence": "0.00",
                        "portal_path": "",
                    },
                ],
            )
            audit_fields = (
                "oachargeid",
                "route",
                "authoritative_value",
                "candidate_count",
                "match_basis",
                "match_status",
                "rejection_reason",
            )
            write_rows(
                workspace / "mapping" / "mapping-audit.csv",
                audit_fields,
                [
                    {
                        "oachargeid": "case-1",
                        "route": "s3",
                        "authoritative_value": "REF-1",
                        "candidate_count": "1",
                        "match_basis": "reference",
                        "match_status": "accepted_unique_rule_supported",
                        "rejection_reason": "",
                    },
                    {
                        "oachargeid": "case-2",
                        "route": "portal",
                        "authoritative_value": "REF-2",
                        "candidate_count": "0",
                        "match_basis": "reference",
                        "match_status": "not_found",
                        "rejection_reason": "No candidate",
                    },
                ],
            )
            (workspace / "spec").mkdir(parents=True)
            (workspace / "spec" / "mapping-spec.json").write_text("{}\n", encoding="utf-8")
            (workspace / "validation").mkdir(parents=True)
            (workspace / "validation" / "validation.json").write_text(
                json.dumps({"gates": {"row_counts_equal": True}}), encoding="utf-8"
            )

            report = MODULE.export_outputs(
                workspace=workspace,
                council="exeter",
                batch="wp1",
                output_directory=output,
            )

            full_path = Path(report["outputs"]["full_mapping"])
            with full_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(tuple(reader.fieldnames or ()), MODULE.WORKFLOW_FIELDS)
            self.assertEqual([row["match_status"] for row in rows], ["found", "no_found"])
            self.assertEqual(report["coverage"]["exact"], True)
            self.assertEqual(report["match_status_counts"], {"found": 1, "no_found": 1, "no_match": 0})


if __name__ == "__main__":
    unittest.main()
