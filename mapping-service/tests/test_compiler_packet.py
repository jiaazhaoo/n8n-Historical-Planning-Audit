from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

AUTONOMOUS_ROOT = Path(__file__).resolve().parents[1] / "amazons3-mapping"
if str(AUTONOMOUS_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTONOMOUS_ROOT))

from autonomous.compiler import compiler_prompt  # noqa: E402
from autonomous.preparation import sample_rows, value_distributions  # noqa: E402


def write(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


# Sheffield's real proportions: a spec written from eight sampled rows routed
# Microfiche and Aperture and dropped the majority U-Drive population.
def sheffield_rows():
    rows = []
    for source, count in (
        ("U-Drive", 60),
        ("Microfiche", 30),
        ("Aperture cards", 8),
        ("Public Access", 2),
    ):
        for index in range(count):
            rows.append(
                {
                    "oachargeid": f"{source[:2]}-{index:04d}",
                    "source": source,
                    "xml_altref": f"9{index % 10}/{index:04d}P",
                }
            )
    return rows


class ValueDistributionTests(unittest.TestCase):
    def test_every_value_of_a_low_cardinality_column_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary) / "source.csv", sheffield_rows())
            counts = value_distributions(path)
            self.assertEqual(
                counts["source"],
                {"U-Drive": 60, "Microfiche": 30, "Aperture cards": 8, "Public Access": 2},
            )

    def test_the_exact_spelling_is_preserved(self) -> None:
        # A condition written as "Aperture" matches nothing when the column
        # says "Aperture cards".
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary) / "source.csv", sheffield_rows())
            self.assertIn("Aperture cards", value_distributions(path)["source"])
            self.assertNotIn("Aperture", value_distributions(path)["source"])

    def test_an_identifying_column_is_left_out(self) -> None:
        # Listing every charge identifier would be a data dump, not a summary.
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary) / "source.csv", sheffield_rows())
            counts = value_distributions(path)
            self.assertNotIn("oachargeid", counts)
            self.assertIn("source", counts)

    def test_the_cardinality_ceiling_is_honoured(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = [{"kind": f"value-{index}"} for index in range(40)]
            path = write(Path(temporary) / "source.csv", rows)
            self.assertNotIn("kind", value_distributions(path, max_distinct=25))
            self.assertIn("kind", value_distributions(path, max_distinct=50))

    def test_a_minority_population_survives_where_sampling_loses_it(self) -> None:
        # The failure this exists for: sampling eight rows can miss a value that
        # accounts for thousands of records.
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary) / "source.csv", sheffield_rows())
            sampled = {row["source"] for row in sample_rows(path)}
            counted = set(value_distributions(path)["source"])
            self.assertEqual(counted, {"U-Drive", "Microfiche", "Aperture cards", "Public Access"})
            self.assertTrue(counted >= sampled)


class PromptTests(unittest.TestCase):
    def test_the_prompt_tells_the_compiler_to_account_for_every_value(self) -> None:
        prompt = compiler_prompt(Path("/tmp/packet.json"), {"a": 1})
        self.assertIn("value_counts", prompt)
        self.assertIn("Aperture cards", prompt)


if __name__ == "__main__":
    unittest.main()
