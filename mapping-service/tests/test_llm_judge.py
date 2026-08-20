from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

AUTONOMOUS_ROOT = Path(__file__).resolve().parents[1] / "amazons3-mapping"
if str(AUTONOMOUS_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTONOMOUS_ROOT))

from autonomous.llm_judge import LlmJudgeVerifier, judge_prompt  # noqa: E402
from autonomous.openrouter_vision import OpenRouterSettings  # noqa: E402
from autonomous.paddle_ocr import PageText  # noqa: E402
from autonomous.review_budget import ReviewBudget  # noqa: E402
from autonomous.schemas import ContentExpectation, QaVerdict  # noqa: E402
from autonomous.storage import ArtifactStore  # noqa: E402


HERON = "Unit 13 Heron Road Sowton Industrial Estate Exeter EX2"
QUEENS = "11 Queens Road Exeter EX2 9ER"
FILLER = "planning application form part 1 to be completed by the applicant " * 8


def expectation(oachargeid: str, *, address: str = "", description: str = "") -> ContentExpectation:
    return ContentExpectation(
        council="exeter",
        batch="wp3",
        oachargeid=oachargeid,
        route="s3",
        mapping_path="s3://localauthorityscans/Exeter/1988/EXE_1988_88-1061-02",
        mapping_confidence=0.74,
        mapping_status="accepted_rule",
        match_basis="further-information-reference",
        reference_fields=(),
        reference_values=(),
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
    def __init__(self, text: str) -> None:
        self.text = text

    def read(self, images):
        return (PageText(name="page-0001.jpg", blocks=tuple(self.text.split("\n"))),)


class ScriptedTransport:
    def __init__(self, *judgements, cost: float = 0.00035) -> None:
        self.judgements = list(judgements)
        self.cost = cost
        self.payloads: list[dict] = []

    def __call__(self, payload):
        self.payloads.append(payload)
        return {
            "model": "deepseek/deepseek-v4-flash",
            "usage": {"cost": self.cost},
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(self.judgements.pop(0))}}
            ],
        }


def judgement(**overrides):
    base = {
        "readable": True,
        "belongs": True,
        "matched_address": True,
        "matched_description": True,
        "document_site": HERON,
        "evidence": "UNIT 13 HERON ROAD SOWTON",
    }
    base.update(overrides)
    return base


class JudgePromptTests(unittest.TestCase):
    def test_the_prompt_states_the_record_and_the_ocr_caveat(self) -> None:
        prompt = judge_prompt(expectation("A", address=HERON, description="Erection of unit"), "TEXT")
        self.assertIn(HERON, prompt)
        self.assertIn("Erection of unit", prompt)
        self.assertIn("recognition errors", prompt)
        # The distinction the string matcher could not make.
        self.assertIn("applicant, agent and council addresses", prompt)

    def test_a_record_without_facts_is_still_described_honestly(self) -> None:
        self.assertIn("(not stated)", judge_prompt(expectation("A"), "TEXT"))


class LlmJudgeVerifierTests(unittest.TestCase):
    def _verify(self, verifier, exp, token="CASE-1"):
        with tempfile.TemporaryDirectory() as temporary:
            return verifier.verify(
                expectation=exp,
                images=(Path("page-0001.jpg"),),
                artifacts=ArtifactStore(Path(temporary)),
                case_token=token,
            )

    def _verifier(self, text, transport, budget=None):
        return LlmJudgeVerifier(
            settings=OpenRouterSettings(api_key="sk-or-v1-test", retry_backoff_seconds=0),
            budget=budget or ReviewBudget(limit_usd=3.0),
            runner=StoredRunner(text),
            transport=transport,
        )

    def test_a_matching_site_verifies_and_charges_the_call(self) -> None:
        budget = ReviewBudget(limit_usd=3.0)
        transport = ScriptedTransport(judgement())
        verifier = self._verifier(FILLER + "\nUNIT 13 HERON ROAD SOWTON", transport, budget)
        exp = expectation("88/1061/FUL", address=HERON)
        verifier.prepare([exp])
        verdict, confidence, eligible, signals, _, reason, _ = self._verify(verifier, exp)
        self.assertEqual(verdict, QaVerdict.VERIFIED_SAME)
        self.assertEqual(confidence, 0.90)
        self.assertTrue(eligible)
        self.assertTrue(signals["address"])
        self.assertIn("UNIT 13 HERON ROAD", reason)
        self.assertAlmostEqual(budget.spent_usd, 0.00035)

    def test_only_ocr_text_is_sent_never_an_image(self) -> None:
        transport = ScriptedTransport(judgement())
        verifier = self._verifier(FILLER + "\nUNIT 13", transport)
        exp = expectation("A", address=HERON)
        verifier.prepare([exp])
        self._verify(verifier, exp)
        body = json.dumps(transport.payloads[0])
        self.assertNotIn("image_url", body)
        self.assertNotIn("base64", body)

    def test_a_file_about_another_sampled_case_is_named_as_a_swap(self) -> None:
        transport = ScriptedTransport(
            judgement(belongs=False, matched_address=False, matched_description=False,
                      document_site=QUEENS)
        )
        verifier = self._verifier(FILLER + "\n11 QUEENS ROAD EXETER", transport)
        mine = expectation("88/1061/FUL", address=HERON)
        theirs = expectation("88/1002/FUL", address=QUEENS)
        verifier.prepare([mine, theirs])
        verdict, _, _, _, _, reason, _ = self._verify(verifier, mine)
        self.assertEqual(verdict, QaVerdict.VERIFIED_WRONG)
        self.assertIn("88/1002/FUL", reason)

    def test_a_file_about_an_unsampled_site_is_unconfirmed_not_wrong(self) -> None:
        # Not matching is not evidence of a swap; only another sampled record is.
        transport = ScriptedTransport(
            judgement(belongs=False, matched_address=False, matched_description=False,
                      document_site="99 Somewhere Else Lane")
        )
        verifier = self._verifier(FILLER + "\nSOMEWHERE ELSE", transport)
        exp = expectation("88/1061/FUL", address=HERON)
        verifier.prepare([exp])
        verdict, _, _, _, _, _, _ = self._verify(verifier, exp)
        self.assertEqual(verdict, QaVerdict.RULE_SUPPORTED_UNVERIFIED)

    def test_a_description_match_alone_can_verify(self) -> None:
        transport = ScriptedTransport(
            judgement(matched_address=False, belongs=True, matched_description=True,
                      document_site="", evidence="Erection of industrial unit")
        )
        verifier = self._verifier(FILLER + "\nERECTION OF INDUSTRIAL UNIT", transport)
        exp = expectation("A", address=HERON, description="Erection of industrial unit")
        verifier.prepare([exp])
        verdict, _, _, _, _, _, _ = self._verify(verifier, exp)
        self.assertEqual(verdict, QaVerdict.VERIFIED_SAME)

    def test_thin_text_is_unreadable_without_calling_the_model(self) -> None:
        budget = ReviewBudget(limit_usd=3.0)
        transport = ScriptedTransport()
        verifier = self._verifier("blur", transport, budget)
        exp = expectation("A", address=HERON)
        verifier.prepare([exp])
        verdict, _, _, _, _, _, _ = self._verify(verifier, exp)
        self.assertEqual(verdict, QaVerdict.UNREADABLE)
        self.assertEqual(transport.payloads, [])
        self.assertEqual(budget.calls, 0)

    def test_a_record_with_nothing_to_check_does_not_call_the_model(self) -> None:
        budget = ReviewBudget(limit_usd=3.0)
        transport = ScriptedTransport()
        verifier = self._verifier(FILLER + "\nANYTHING", transport, budget)
        exp = expectation("A")
        verifier.prepare([exp])
        verdict, _, _, _, _, reason, _ = self._verify(verifier, exp)
        self.assertEqual(verdict, QaVerdict.RULE_SUPPORTED_UNVERIFIED)
        self.assertIn("states no address or description", reason)
        self.assertEqual(budget.calls, 0)

    def test_the_model_reporting_illegible_text_is_unreadable(self) -> None:
        transport = ScriptedTransport(judgement(readable=False, matched_address=False))
        verifier = self._verifier(FILLER + "\nSMUDGE", transport)
        exp = expectation("A", address=HERON)
        verifier.prepare([exp])
        verdict, _, _, _, _, _, _ = self._verify(verifier, exp)
        self.assertEqual(verdict, QaVerdict.UNREADABLE)

    def test_the_judgement_is_kept_with_its_evidence(self) -> None:
        transport = ScriptedTransport(judgement())
        verifier = self._verifier(FILLER + "\nUNIT 13", transport)
        exp = expectation("A", address=HERON)
        verifier.prepare([exp])
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = ArtifactStore(Path(temporary))
            verifier.verify(
                expectation=exp,
                images=(Path("page-0001.jpg"),),
                artifacts=artifacts,
                case_token="CASE-1",
            )
            saved = json.loads(artifacts.resolve("qa/cases/CASE-1/judge.json").read_text())
            self.assertEqual(saved["judgement"]["evidence"], "UNIT 13 HERON ROAD SOWTON")
            self.assertTrue(artifacts.resolve("qa/cases/CASE-1/ocr-text.json").is_file())


if __name__ == "__main__":
    unittest.main()
