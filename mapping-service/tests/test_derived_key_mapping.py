from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

AUTONOMOUS_ROOT = Path(__file__).resolve().parents[1] / "amazons3-mapping"
if str(AUTONOMOUS_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTONOMOUS_ROOT))

from autonomous.engine import execute_mapping  # noqa: E402
from autonomous.schemas import (  # noqa: E402
    DerivedKey,
    MappingSpec,
    Predicate,
    PredicateOperator,
    RouteRule,
    RouteTarget,
)
from autonomous.spec_verifier import ambiguity_negative_test_errors  # noqa: E402


# Exeter WP3 references read 88/1061/FUL while the folder that holds the scans
# reads EXE_1988_88-1061-02. The trailing FUL and the trailing 02 are unrelated,
# so only the year and number identify the record.
EXETER_KEY = DerivedKey(
    source_templates=("{yy:d}/{number:d}/{kind:a}", "{yy:d}/{number:d}"),
    inventory_templates=("EXE_{year4:d}_{yy:d}-{number:d}-{code:a}", "EXE_{year4:d}_{yy:d}-{number:d}"),
    inventory_match_mode="prefix",
    key_parts=("yy", "number"),
    normalizers={"number": ("strip_zeros",)},
)


def spec(derived: DerivedKey | None = EXETER_KEY) -> MappingSpec:
    return MappingSpec(
        spec_id="exeter-wp3",
        council="exeter",
        batch="wp3",
        source_id_field="oachargeid",
        inventory_key_field="folder",
        inventory_path_field="amazons3_path",
        routes=(
            RouteRule(
                rule_id="derived_reference_s3",
                priority=10,
                conditions=(Predicate(operator=PredicateOperator.NOT_BLANK, field="reference"),),
                target=RouteTarget.S3,
                authoritative_key="reference",
                inventory_key_field="folder",
                inventory_path_field="amazons3_path",
                derived_key=derived,
                automatic_confidence=0.74,
            ),
            RouteRule(
                rule_id="reject_rest",
                priority=100,
                conditions=(Predicate(operator=PredicateOperator.ALWAYS),),
                target=RouteTarget.REJECT,
            ),
        ),
    )


def inventory(*folders: str) -> list[dict[str, str]]:
    return [
        {"folder": folder, "amazons3_path": f"s3://localauthorityscans/Exeter/{folder}"}
        for folder in folders
    ]


class DerivedKeyMappingTests(unittest.TestCase):
    def test_a_reference_matches_a_differently_written_folder(self) -> None:
        source = [{"oachargeid": "A", "reference": "88/1061/FUL"}]
        result = execute_mapping(source, inventory("EXE_1988_88-1061-02"), spec())
        self.assertEqual(
            result.mapping_rows[0]["amazons3_path"],
            "s3://localauthorityscans/Exeter/EXE_1988_88-1061-02",
        )
        self.assertEqual(result.audit_rows[0]["match_status"], "accepted_unique_rule_supported")

    def test_whole_field_normalisation_cannot_do_the_same_join(self) -> None:
        # The reason the derived key exists: without one, nothing matches.
        source = [{"oachargeid": "A", "reference": "88/1061/FUL"}]
        result = execute_mapping(source, inventory("EXE_1988_88-1061-02"), spec(derived=None))
        self.assertEqual(result.mapping_rows[0]["amazons3_path"], "")
        self.assertEqual(result.audit_rows[0]["match_status"], "not_found")

    def test_leading_zeros_are_reconciled_between_the_two_sides(self) -> None:
        source = [{"oachargeid": "A", "reference": "98/0538/CAC"}]
        result = execute_mapping(source, inventory("EXE_1998_98-538-CAC"), spec())
        self.assertTrue(result.mapping_rows[0]["amazons3_path"])

    def test_a_folder_with_a_trailing_address_still_matches(self) -> None:
        # 642 Exeter folders append the site address after the code.
        source = [{"oachargeid": "A", "reference": "77/1/ADV"}]
        result = execute_mapping(
            source, inventory("EXE_1977_77-1-ADV-UNIT 2 GUILDHALL DEVELOPMENT"), spec()
        )
        self.assertTrue(result.mapping_rows[0]["amazons3_path"])

    def test_an_alternative_template_covers_a_reference_without_its_suffix(self) -> None:
        source = [{"oachargeid": "A", "reference": "88/1061"}]
        result = execute_mapping(source, inventory("EXE_1988_88-1061-02"), spec())
        self.assertTrue(result.mapping_rows[0]["amazons3_path"])

    def test_a_reference_no_template_fits_is_rejected_and_named(self) -> None:
        source = [{"oachargeid": "A", "reference": "TPO 5/2011"}]
        result = execute_mapping(source, inventory("EXE_1988_88-1061-02"), spec())
        audit = result.audit_rows[0]
        self.assertEqual(audit["match_status"], "rejected_unparsable_authoritative_value")
        self.assertIn("TPO 5/2011", audit["rejection_reason"])
        self.assertEqual(result.mapping_rows[0]["amazons3_path"], "")

    def test_two_folders_sharing_a_derived_key_are_rejected_as_ambiguous(self) -> None:
        source = [{"oachargeid": "A", "reference": "85/0752/FUL"}]
        result = execute_mapping(
            source, inventory("EXE_1985_85-0752-02", "EXE_1985_85-0752-03"), spec()
        )
        self.assertEqual(
            result.audit_rows[0]["match_status"], "rejected_ambiguous_multiple_candidates"
        )
        self.assertEqual(result.mapping_rows[0]["amazons3_path"], "")

    def test_an_unrelated_folder_does_not_match(self) -> None:
        source = [{"oachargeid": "A", "reference": "88/1061/FUL"}]
        result = execute_mapping(source, inventory("EXE_1988_88-9999-02"), spec())
        self.assertEqual(result.audit_rows[0]["match_status"], "not_found")

    def test_every_case_survives_regardless_of_outcome(self) -> None:
        source = [
            {"oachargeid": "A", "reference": "88/1061/FUL"},
            {"oachargeid": "B", "reference": "TPO 5/2011"},
            {"oachargeid": "C", "reference": ""},
        ]
        result = execute_mapping(source, inventory("EXE_1988_88-1061-02"), spec())
        self.assertEqual(len(result.mapping_rows), 3)
        self.assertEqual([row["oachargeid"] for row in result.mapping_rows], ["A", "B", "C"])


class DerivedKeyVerifierTests(unittest.TestCase):
    def _prepared(self, rows):
        temporary = tempfile.mkdtemp()
        path = Path(temporary) / "source.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        class Prepared:
            source_path = path

        return Prepared()

    def test_ambiguity_is_still_demonstrated_under_a_derived_key(self) -> None:
        prepared = self._prepared([{"oachargeid": "A", "reference": "88/1061/FUL"}])
        self.assertEqual(ambiguity_negative_test_errors(spec(), prepared), [])

    def test_a_real_reference_is_not_replaced_by_a_placeholder(self) -> None:
        # A not_blank condition on the key field used to overwrite the reference
        # with "value", which no template can parse.
        prepared = self._prepared([{"oachargeid": "A", "reference": "88/1061/FUL"}])
        errors = ambiguity_negative_test_errors(spec(), prepared)
        self.assertFalse([e for e in errors if "cannot render" in e])


class DerivedKeyValidationTests(unittest.TestCase):
    def test_a_key_part_absent_from_one_side_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            DerivedKey(
                source_templates=("{yy:d}/{number:d}",),
                inventory_templates=("EXE_{year4:d}_{yy:d}",),
                key_parts=("yy", "number"),
            )

    def test_a_template_without_parts_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            DerivedKey(
                source_templates=("literal",),
                inventory_templates=("EXE_{yy:d}",),
                key_parts=("yy",),
            )

    def test_the_declaration_round_trips_through_json(self) -> None:
        restored = DerivedKey.model_validate_json(EXETER_KEY.model_dump_json())
        self.assertEqual(restored.build().source_key("88/1061/FUL"), ("88", "1061"))


if __name__ == "__main__":
    unittest.main()
