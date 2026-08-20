from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

AUTONOMOUS_ROOT = Path(__file__).resolve().parents[1] / "amazons3-mapping"
if str(AUTONOMOUS_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTONOMOUS_ROOT))

from autonomous.local_verifier import MIN_TEXT_CHARACTERS, OcrTextVerifier  # noqa: E402
from autonomous.paddle_ocr import PageText  # noqa: E402
from autonomous.schemas import ContentExpectation, QaVerdict  # noqa: E402
from autonomous.storage import ArtifactStore  # noqa: E402


# Real Exeter WP3 source values and the shape of their OCR'd documents.
HERON = "Unit 13 Heron Road Sowton Industrial Estate Exeter EX2"
QUEENS = "11 Queens Road Exeter EX2 9ER"
FILLER = "planning application form part 1 to be completed by the applicant " * 6


def expectation(
    oachargeid: str,
    *,
    address: str = "",
    description: str = "",
    reference: str = "",
    route: str = "s3",
    path: str = "s3://localauthorityscans/Exeter/1988/EXE_1988_88-1061-02",
) -> ContentExpectation:
    return ContentExpectation(
        council="exeter",
        batch="wp3",
        oachargeid=oachargeid,
        route=route,
        mapping_path=path,
        mapping_confidence=0.74,
        mapping_status="accepted_rule",
        match_basis="further-information-reference",
        reference_fields=("further-information-reference",),
        reference_values=(reference,) if reference else (),
        address_fields=("charge-geographic-description",),
        address_values=(address,) if address else (),
        description_fields=("supplementary-information",),
        description_values=(description,) if description else (),
        date_fields=(),
        date_values=(),
        document_type_fields=(),
        document_type_values=(),
    )


class StoredRunner:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.calls = 0

    def read(self, images):
        self.calls += 1
        return (PageText(name="page-0001.jpg", blocks=tuple(self.text.split("\n"))),)


class OcrTextVerifierTests(unittest.TestCase):
    def _verify(self, verifier, exp, token="CASE-1"):
        with tempfile.TemporaryDirectory() as temporary:
            return verifier.verify(
                expectation=exp,
                images=(Path("page-0001.jpg"),),
                artifacts=ArtifactStore(Path(temporary)),
                case_token=token,
            )

    def test_the_source_address_found_in_the_document_verifies_the_mapping(self) -> None:
        runner = StoredRunner(FILLER + "\nUNIT 13 HERON ROAD SOWTON INDUSTRIAL ESTATE EXETER")
        verifier = OcrTextVerifier(runner=runner)
        exp = expectation("88/1061/FUL", address=HERON)
        verifier.prepare([exp])
        with tempfile.TemporaryDirectory() as temporary:
            verdict, confidence, eligible, signals, _, _, evidence = verifier.verify(
                expectation=exp,
                images=(Path("page-0001.jpg"),),
                artifacts=ArtifactStore(Path(temporary)),
                case_token="CASE-1",
            )
            self.assertTrue(evidence.is_file())
        self.assertEqual(verdict, QaVerdict.VERIFIED_SAME)
        self.assertEqual(confidence, 0.90)
        self.assertTrue(eligible)
        self.assertTrue(signals["address"])

    def test_a_document_carrying_another_sampled_case_is_reported_as_a_swap(self) -> None:
        runner = StoredRunner(FILLER + "\n11 QUEENS ROAD EXETER EX2 9ER")
        verifier = OcrTextVerifier(runner=runner)
        mine = expectation("88/1061/FUL", address=HERON)
        theirs = expectation("88/1002/FUL", address=QUEENS)
        verifier.prepare([mine, theirs])
        verdict, _, _, _, _, reason, _ = self._verify(verifier, mine)
        self.assertEqual(verdict, QaVerdict.VERIFIED_WRONG)
        self.assertIn("88/1002/FUL", reason)

    def test_an_unrelated_document_is_unconfirmed_rather_than_wrong(self) -> None:
        # Absence of the address is not evidence of a different record.
        runner = StoredRunner(FILLER + "\nSOME OTHER STREET ALTOGETHER")
        verifier = OcrTextVerifier(runner=runner)
        exp = expectation("88/1061/FUL", address=HERON)
        verifier.prepare([exp])
        verdict, _, _, _, _, _, _ = self._verify(verifier, exp)
        self.assertEqual(verdict, QaVerdict.RULE_SUPPORTED_UNVERIFIED)

    def test_a_document_with_almost_no_recognised_text_is_unreadable(self) -> None:
        runner = StoredRunner("blur")
        verifier = OcrTextVerifier(runner=runner)
        exp = expectation("88/1061/FUL", address=HERON)
        verifier.prepare([exp])
        verdict, _, _, _, _, reason, _ = self._verify(verifier, exp)
        self.assertEqual(verdict, QaVerdict.UNREADABLE)
        self.assertIn(str(len("blur")), reason)
        self.assertLess(len("blur"), MIN_TEXT_CHARACTERS)

    def test_a_source_record_with_nothing_to_match_says_so(self) -> None:
        # Exeter WP1 carries no address or description at all, so its mapping
        # cannot be checked against document content by any method.
        runner = StoredRunner(FILLER + "\nUNIT 13 HERON ROAD")
        verifier = OcrTextVerifier(runner=runner)
        exp = expectation("EXE_1978_78-660-03")
        verifier.prepare([exp])
        verdict, _, _, _, _, reason, _ = self._verify(verifier, exp)
        self.assertEqual(verdict, QaVerdict.RULE_SUPPORTED_UNVERIFIED)
        self.assertIn("states no address or description", reason)

    def test_reference_alone_does_not_verify(self) -> None:
        # The mapping was made on the reference, so finding it again proves
        # nothing the mapping did not already assume.
        runner = StoredRunner(FILLER + "\nApplication No: 88/1061/FUL")
        verifier = OcrTextVerifier(runner=runner)
        exp = expectation("88/1061/FUL", address=HERON, reference="88/1061/FUL")
        verifier.prepare([exp])
        verdict, _, _, signals, _, _, _ = self._verify(verifier, exp)
        self.assertTrue(signals["reference"])
        self.assertFalse(signals["address"])
        self.assertEqual(verdict, QaVerdict.RULE_SUPPORTED_UNVERIFIED)

    def test_reference_plus_a_description_verifies(self) -> None:
        runner = StoredRunner(
            FILLER + "\nApplication No: 88/1061/FUL\nErection of industrial unit and ancillary works"
        )
        verifier = OcrTextVerifier(runner=runner)
        exp = expectation(
            "88/1061/FUL",
            address=HERON,
            description="Erection of industrial unit and ancillary external works",
            reference="88/1061/FUL",
        )
        verifier.prepare([exp])
        verdict, _, _, signals, _, _, _ = self._verify(verifier, exp)
        self.assertTrue(signals["description"])
        self.assertEqual(verdict, QaVerdict.VERIFIED_SAME)

    def test_a_none_placeholder_is_not_treated_as_a_source_fact(self) -> None:
        runner = StoredRunner(FILLER + "\nnone of this matters")
        verifier = OcrTextVerifier(runner=runner)
        exp = expectation("88/1002/FUL", description="None")
        verifier.prepare([exp])
        verdict, _, _, _, _, reason, _ = self._verify(verifier, exp)
        self.assertIn("states no address or description", reason)
        self.assertEqual(verdict, QaVerdict.RULE_SUPPORTED_UNVERIFIED)

    def test_the_ocr_text_and_the_decision_are_both_kept(self) -> None:
        runner = StoredRunner(FILLER + "\nUNIT 13 HERON ROAD SOWTON")
        verifier = OcrTextVerifier(runner=runner)
        exp = expectation("88/1061/FUL", address=HERON)
        verifier.prepare([exp])
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = ArtifactStore(Path(temporary))
            verifier.verify(
                expectation=exp,
                images=(Path("page-0001.jpg"),),
                artifacts=artifacts,
                case_token="CASE-1",
            )
            self.assertTrue(artifacts.resolve("qa/cases/CASE-1/local-match.json").is_file())
            self.assertTrue(artifacts.resolve("qa/cases/CASE-1/ocr-text.json").is_file())

    def test_the_verifier_reads_each_case_once(self) -> None:
        runner = StoredRunner(FILLER + "\nUNIT 13 HERON ROAD SOWTON")
        verifier = OcrTextVerifier(runner=runner)
        exp = expectation("88/1061/FUL", address=HERON)
        verifier.prepare([exp])
        self._verify(verifier, exp)
        self.assertEqual(runner.calls, 1)


if __name__ == "__main__":
    unittest.main()
