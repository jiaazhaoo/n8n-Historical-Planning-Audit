from __future__ import annotations

import sys
import unittest
from pathlib import Path

AUTONOMOUS_ROOT = Path(__file__).resolve().parents[1] / "amazons3-mapping"
if str(AUTONOMOUS_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTONOMOUS_ROOT))

from autonomous.join_probe import Precedence, probe_join  # noqa: E402
from autonomous.key_derivation import (  # noqa: E402
    DerivationError,
    KeyDerivation,
    Template,
    from_declaration,
    index_inventory,
    normalise,
)


EXETER_SOURCE = Template("EXE_{year4:d}_{yy:d}-{number:d}-{code:a}", "exact")
EXETER_FOLDER = Template("EXE_{year4:d}_{yy:d}-{number:d}-{code:a}", "prefix")


def exeter_derivation(**overrides) -> KeyDerivation:
    settings = dict(
        source_templates=(EXETER_SOURCE,),
        inventory_templates=(EXETER_FOLDER,),
        key_parts=("yy", "number"),
        normalizers={"number": ("strip_zeros",)},
    )
    settings.update(overrides)
    return KeyDerivation(**settings)


class TemplateTests(unittest.TestCase):
    def test_parts_split_on_the_following_literal(self) -> None:
        parsed = EXETER_SOURCE.parse("EXE_1978_78-660-03")
        self.assertEqual(parsed, {"year4": "1978", "yy": "78", "number": "660", "code": "03"})

    def test_prefix_mode_tolerates_a_trailing_free_text_address(self) -> None:
        # 642 Exeter folders append the site address after the code.
        parsed = EXETER_FOLDER.parse("EXE_1977_77-1-ADV-UNIT 2 GUILDHALL DEVELOPMENT")
        self.assertEqual(parsed["number"], "1")
        self.assertEqual(parsed["code"], "ADV")

    def test_exact_mode_rejects_the_same_trailing_text(self) -> None:
        self.assertIsNone(EXETER_SOURCE.parse("EXE_1977_77-1-ADV-UNIT 2 GUILDHALL"))

    def test_a_template_formats_what_it_parses(self) -> None:
        parsed = EXETER_SOURCE.parse("EXE_1978_78-660-03")
        self.assertEqual(EXETER_SOURCE.format(parsed), "EXE_1978_78-660-03")

    def test_a_template_without_parts_is_refused(self) -> None:
        with self.assertRaises(DerivationError):
            Template("EXE_literal_only")

    def test_an_unknown_match_mode_is_refused(self) -> None:
        with self.assertRaises(DerivationError):
            Template("{a}-{b}", "fuzzy")

    def test_formatting_without_a_part_is_refused(self) -> None:
        with self.assertRaises(DerivationError):
            EXETER_SOURCE.format({"year4": "1978"})


class NormaliserTests(unittest.TestCase):
    def test_strip_zeros_and_pad_are_inverses(self) -> None:
        self.assertEqual(normalise("00010", ("strip_zeros",)), "10")
        self.assertEqual(normalise("10", ("pad:5",)), "00010")

    def test_stripping_zeros_keeps_a_zero_value(self) -> None:
        self.assertEqual(normalise("000", ("strip_zeros",)), "0")

    def test_two_digit_years_expand_around_the_pivot(self) -> None:
        self.assertEqual(normalise("78", ("year2to4",)), "1978")
        self.assertEqual(normalise("06", ("year2to4",)), "2006")

    def test_a_four_digit_year_passes_through_unchanged(self) -> None:
        # One part name carries the same normalisers on both sides of a join, and
        # a reference writes 78 where its folder writes 1978.
        self.assertEqual(normalise("1978", ("year2to4",)), "1978")

    def test_year_expansion_refuses_a_length_that_is_neither(self) -> None:
        with self.assertRaises(DerivationError):
            normalise("198", ("year2to4",))

    def test_an_unknown_normaliser_is_refused(self) -> None:
        with self.assertRaises(DerivationError):
            normalise("x", ("titlecase",))


class KeyDerivationTests(unittest.TestCase):
    def test_both_sides_agree_despite_leading_zeros(self) -> None:
        derivation = exeter_derivation()
        self.assertEqual(
            derivation.source_key("EXE_1978_78-01-03"),
            derivation.inventory_key("EXE_1978_78-1-03-SOME ADDRESS"),
        )

    def test_a_key_part_missing_from_a_template_is_refused(self) -> None:
        with self.assertRaises(DerivationError) as caught:
            exeter_derivation(key_parts=("yy", "number", "district"))
        self.assertIn("district", str(caught.exception))

    def test_an_alternative_may_supply_a_missing_part_from_defaults(self) -> None:
        derivation = exeter_derivation(
            source_templates=(
                EXETER_SOURCE,
                Template("EXE_{year4:d}_{yy:d}-{number:d}", "exact"),
            ),
            key_parts=("yy", "number", "code"),
            defaults={"code": ""},
        )
        # 270 of Exeter's 289 unparsed identifiers are this shorter shape.
        self.assertEqual(derivation.source_key("EXE_1977_77-187"), ("77", "187", ""))

    def test_alternatives_are_tried_in_order(self) -> None:
        derivation = KeyDerivation(
            source_templates=(
                Template("{prefix:a}.{number:d}/{part:d}", "exact"),
                Template("{prefix:a}.{cls:a}.{number:d}/{part:d}", "exact"),
            ),
            inventory_templates=(Template("{prefix:a}.{number:d}_{part:d}", "exact"),),
            key_parts=("prefix", "number", "part"),
            normalizers={"number": ("strip_zeros",), "prefix": ("upper",)},
        )
        self.assertEqual(derivation.source_key("TVN.00234/1"), ("TVN", "234", "1"))
        self.assertEqual(derivation.source_key("TVN.LB.00220/2"), ("TVN", "220", "2"))

    def test_an_unparseable_value_yields_no_key(self) -> None:
        self.assertIsNone(exeter_derivation().source_key("16/00249/LBWN"))

    def test_a_declaration_round_trips(self) -> None:
        derivation = from_declaration(
            {
                "source_templates": ["EXE_{year4:d}_{yy:d}-{number:d}-{code:a}"],
                "inventory_templates": ["EXE_{year4:d}_{yy:d}-{number:d}-{code:a}"],
                "inventory_match_mode": "prefix",
                "key_parts": ["yy", "number"],
                "normalizers": {"number": ["strip_zeros"]},
            }
        )
        self.assertEqual(derivation.source_key("EXE_1978_78-01-03"), ("78", "1"))

    def test_a_declaration_missing_a_field_is_refused(self) -> None:
        with self.assertRaises(DerivationError):
            from_declaration({"source_templates": ["{a}-{b}"]})

    def test_inventory_indexing_reports_rows_it_could_not_key(self) -> None:
        rows = [{"folder": "EXE_1978_78-1-03"}, {"folder": "not-a-folder"}]
        index, unparsed = index_inventory(rows, exeter_derivation())
        self.assertEqual(list(index), [("78", "1")])
        self.assertEqual(unparsed, ["not-a-folder"])


class PrecedenceTests(unittest.TestCase):
    def test_the_preferred_location_wins(self) -> None:
        # Test Valley holds the same reference as fiche and as paper; the
        # capture rules make fiche primary, so this is not ambiguity.
        precedence = Precedence("parent_prefix", ("/Fiche/", "/Paper Files"))
        rows = [
            {"parent_prefix": "Test Valley/Paper Files (South)/Batch 3/"},
            {"parent_prefix": "Test Valley/Fiche/South_Part 3/"},
        ]
        resolved = precedence.resolve(rows)
        self.assertEqual(len(resolved), 1)
        self.assertIn("Fiche", resolved[0]["parent_prefix"])

    def test_two_candidates_in_the_same_tier_stay_ambiguous(self) -> None:
        precedence = Precedence("parent_prefix", ("/Fiche/",))
        rows = [
            {"parent_prefix": "Test Valley/Fiche/North _Part 1/"},
            {"parent_prefix": "Test Valley/Fiche/North _Part 3/"},
        ]
        self.assertEqual(len(precedence.resolve(rows)), 2)

    def test_without_an_order_nothing_is_resolved(self) -> None:
        rows = [{"parent_prefix": "a"}, {"parent_prefix": "b"}]
        self.assertEqual(len(Precedence().resolve(rows)), 2)


class JoinProbeTests(unittest.TestCase):
    def _probe(self, source, inventory, **kwargs):
        return probe_join(
            source,
            inventory,
            exeter_derivation(),
            reference_field="oachargeid",
            council="exeter",
            batch="wp1",
            **kwargs,
        )

    def test_a_clean_join_predicts_full_coverage(self) -> None:
        source = [{"oachargeid": "EXE_1978_78-1-03"}, {"oachargeid": "EXE_1978_78-2-03"}]
        inventory = [{"folder": "EXE_1978_78-1-03"}, {"folder": "EXE_1978_78-02-03"}]
        report = self._probe(source, inventory)
        self.assertEqual(report.unique_candidates, 2)
        self.assertEqual(report.describe()["predicted"]["resolvable_rate"], 1.0)

    def test_an_unkeyable_reference_counts_as_zero_candidates(self) -> None:
        report = self._probe([{"oachargeid": "16/00249/LBWN"}], [{"folder": "EXE_1978_78-1-03"}])
        self.assertEqual(report.zero_candidates, 1)
        self.assertEqual(report.source_keyed, 0)

    def test_precedence_converts_ambiguity_into_coverage(self) -> None:
        source = [{"oachargeid": "EXE_1978_78-1-03"}]
        inventory = [
            {"folder": "EXE_1978_78-1-03", "parent_prefix": "Exeter/Pilot/"},
            {"folder": "EXE_1978_78-1-03", "parent_prefix": "Exeter/1978/"},
        ]
        without = self._probe(source, inventory)
        self.assertEqual(without.still_ambiguous, 1)
        self.assertEqual(without.resolvable, 0)

        with_order = self._probe(
            source, inventory, precedence=Precedence("parent_prefix", ("Exeter/19",))
        )
        self.assertEqual(with_order.resolved_by_precedence, 1)
        self.assertEqual(with_order.resolvable, 1)

    def test_the_report_separates_what_today_and_precedence_would_deliver(self) -> None:
        source = [{"oachargeid": "EXE_1978_78-1-03"}]
        inventory = [
            {"folder": "EXE_1978_78-1-03", "parent_prefix": "Exeter/Pilot/"},
            {"folder": "EXE_1978_78-1-03", "parent_prefix": "Exeter/1978/"},
        ]
        data = self._probe(
            source, inventory, precedence=Precedence("parent_prefix", ("Exeter/19",))
        ).describe()
        self.assertEqual(data["predicted"]["resolvable_without_precedence"], 0)
        self.assertEqual(data["predicted"]["resolvable"], 1)

    def test_an_empty_source_reports_zero_rather_than_dividing_by_zero(self) -> None:
        report = self._probe([], [{"folder": "EXE_1978_78-1-03"}])
        self.assertEqual(report.describe()["predicted"]["resolvable_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
