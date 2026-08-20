from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

AUTONOMOUS_ROOT = Path(__file__).resolve().parents[1] / "amazons3-mapping"
if str(AUTONOMOUS_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTONOMOUS_ROOT))

from autonomous.openrouter_vision import (  # noqa: E402
    DEFAULT_MODEL,
    OpenRouterError,
    OpenRouterSettings,
    OpenRouterVisionExtractor,
    read_api_key,
)
from autonomous.review_budget import BudgetExhausted, ReviewBudget  # noqa: E402
from autonomous.storage import ArtifactStore  # noqa: E402


OBSERVATION = {
    "schema_version": 1,
    "readable": True,
    "images_reviewed": 1,
    "identities": [
        {
            "reference": "92/0301P",
            "property_address": "12 Fore Street, Exeter",
            "description": "Tree preservation order",
            "relevant_date": "1992-04-01",
            "document_type": "decision notice",
            "image_names": ["page-01.jpg"],
            "evidence": ["page-01.jpg: 92/0301P"],
        }
    ],
    "unreadable_image_names": [],
    "warnings": [],
}


def response(cost: float | None = 0.004, content: dict | None = None) -> dict:
    usage = {"prompt_tokens": 12000, "completion_tokens": 400}
    if cost is not None:
        usage["cost"] = cost
    return {
        "model": DEFAULT_MODEL,
        "usage": usage,
        "choices": [{"message": {"content": json.dumps(content or OBSERVATION)}}],
    }


class RecordingTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads: list[dict] = []

    def __call__(self, payload):
        self.payloads.append(payload)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def make_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 40), "white").save(path)
    return path


class ReviewBudgetTests(unittest.TestCase):
    def test_projection_uses_observed_cost_once_a_case_is_priced(self) -> None:
        budget = ReviewBudget(limit_usd=3.0, estimate_usd_per_case=0.03)
        self.assertEqual(budget.projected_usd_per_case, 0.03)
        budget.record(0.005)
        self.assertAlmostEqual(budget.projected_usd_per_case, 0.005)

    def test_affordable_case_count_grows_once_real_prices_are_known(self) -> None:
        budget = ReviewBudget(limit_usd=3.0, estimate_usd_per_case=0.03)
        self.assertEqual(budget.affordable_cases(), 100)
        budget.record(0.005)
        self.assertGreater(budget.affordable_cases(), 100)

    def test_spent_budget_refuses_another_call(self) -> None:
        budget = ReviewBudget(limit_usd=0.01, estimate_usd_per_case=0.005)
        budget.record(0.009)
        with self.assertRaises(BudgetExhausted):
            budget.ensure_affordable()

    def test_an_unpriced_call_is_charged_the_estimate(self) -> None:
        # A provider that reports no cost must not make the round look free.
        budget = ReviewBudget(limit_usd=1.0, estimate_usd_per_case=0.02)
        budget.record(None)
        self.assertAlmostEqual(budget.spent_usd, 0.02)
        self.assertEqual(budget.unpriced_calls, 1)
        self.assertTrue(budget.notes)

    def test_negative_cost_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ReviewBudget().record(-1.0)

    def test_truncation_is_recorded_so_a_smaller_sample_is_visible(self) -> None:
        budget = ReviewBudget()
        budget.note_truncation(4)
        self.assertEqual(budget.describe()["truncated_cases"], 4)
        self.assertTrue(budget.describe()["notes"])


class ApiKeyTests(unittest.TestCase):
    def test_key_is_found_by_prefix_not_by_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "keys.md"
            path.write_text(
                "deepseek = sk-abc123\nonerouter = sk-or-v1-abcdef0123456789abcdef\n",
                encoding="utf-8",
            )
            self.assertEqual(read_api_key(path), "sk-or-v1-abcdef0123456789abcdef")

    def test_a_file_without_an_openrouter_key_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "keys.md"
            path.write_text("openai = sk-proj-abc\n", encoding="utf-8")
            with self.assertRaises(OpenRouterError):
                read_api_key(path)


class OpenRouterVisionExtractorTests(unittest.TestCase):
    def _extract(self, transport, budget=None, temporary=None):
        root = Path(temporary)
        image = make_image(root / "images" / "page-01.jpg")
        extractor = OpenRouterVisionExtractor(
            OpenRouterSettings(api_key="sk-or-v1-test", retry_backoff_seconds=0),
            budget=budget or ReviewBudget(),
            transport=transport,
        )
        return extractor, extractor.extract(
            case_token="CASE-1", images=(image,), artifacts=ArtifactStore(root / "artifacts")
        )

    def test_observation_is_parsed_and_the_call_is_charged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            budget = ReviewBudget(limit_usd=3.0)
            transport = RecordingTransport([response(cost=0.004)])
            extractor, (observation, path) = self._extract(transport, budget, temporary)
            self.assertTrue(observation.readable)
            self.assertEqual(observation.identities[0].reference, "92/0301P")
            self.assertTrue(path.is_file())
            self.assertAlmostEqual(budget.spent_usd, 0.004)

    def test_request_asks_for_the_configured_model_and_a_strict_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            transport = RecordingTransport([response()])
            self._extract(transport, None, temporary)
            payload = transport.payloads[0]
            self.assertEqual(payload["model"], DEFAULT_MODEL)
            self.assertEqual(payload["temperature"], 0)
            self.assertTrue(payload["usage"]["include"])
            self.assertEqual(payload["response_format"]["json_schema"]["strict"], True)

    def test_each_image_is_labelled_with_its_neutral_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            transport = RecordingTransport([response()])
            self._extract(transport, None, temporary)
            parts = transport.payloads[0]["messages"][0]["content"]
            texts = [part["text"] for part in parts if part["type"] == "text"]
            images = [part for part in parts if part["type"] == "image_url"]
            self.assertIn("Attachment: page-01.jpg", texts)
            self.assertEqual(len(images), 1)
            self.assertTrue(images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_a_spent_budget_stops_the_call_before_it_is_made(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            budget = ReviewBudget(limit_usd=0.001, estimate_usd_per_case=0.03)
            transport = RecordingTransport([response()])
            with self.assertRaises(BudgetExhausted):
                self._extract(transport, budget, temporary)
            self.assertEqual(transport.payloads, [])

    def test_a_cached_observation_is_not_charged_again(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            budget = ReviewBudget(limit_usd=3.0)
            root = Path(temporary)
            image = make_image(root / "images" / "page-01.jpg")
            artifacts = ArtifactStore(root / "artifacts")
            transport = RecordingTransport([response(cost=0.004)])
            extractor = OpenRouterVisionExtractor(
                OpenRouterSettings(api_key="sk-or-v1-test"), budget=budget, transport=transport
            )
            extractor.extract(case_token="CASE-1", images=(image,), artifacts=artifacts)
            extractor.extract(case_token="CASE-1", images=(image,), artifacts=artifacts)
            self.assertEqual(budget.calls, 1)
            self.assertEqual(len(transport.payloads), 1)

    def test_an_observation_citing_an_unattached_image_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hostile = json.loads(json.dumps(OBSERVATION))
            hostile["identities"][0]["image_names"] = ["some-real-filename.pdf"]
            transport = RecordingTransport([response(content=hostile)])
            with self.assertRaises(OpenRouterError):
                self._extract(transport, None, temporary)

    def test_unparseable_content_is_saved_for_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            broken = dict(response())
            broken["choices"] = [{"message": {"content": "not json"}}]
            transport = RecordingTransport([broken])
            with self.assertRaises(OpenRouterError):
                self._extract(transport, None, temporary)
            failed = list(Path(temporary).rglob("observation.failed.json"))
            self.assertTrue(failed)

    def test_a_provider_error_returned_with_http_200_is_retried(self) -> None:
        # OpenRouter delivers upstream failures inside a normal response, with
        # finish_reason "error" and zero usage. Observed live on 2026-08-20.
        soft_error = {
            "model": DEFAULT_MODEL,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0},
            "choices": [{"finish_reason": "error", "message": {"content": None}}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            budget = ReviewBudget(limit_usd=3.0)
            transport = RecordingTransport([soft_error, response(cost=0.004)])
            extractor, (observation, _) = self._extract(transport, budget, temporary)
            self.assertTrue(observation.readable)
            self.assertEqual(len(transport.payloads), 2)
            # Only the call that produced content is charged.
            self.assertAlmostEqual(budget.spent_usd, 0.004)

    def test_a_persistent_provider_error_still_fails(self) -> None:
        soft_error = {
            "model": DEFAULT_MODEL,
            "usage": {},
            "choices": [{"finish_reason": "error", "message": {"content": None}}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(OpenRouterError):
                self._extract(RecordingTransport([soft_error] * 3), None, temporary)

    def test_an_error_field_in_the_body_is_retried(self) -> None:
        body_error = {"error": {"code": 502, "message": "upstream unavailable"}}
        with tempfile.TemporaryDirectory() as temporary:
            transport = RecordingTransport([body_error, response()])
            self._extract(transport, None, temporary)
            self.assertEqual(len(transport.payloads), 2)

    def test_an_empty_choice_list_is_retried_then_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            empty = dict(response())
            empty["choices"] = []
            transport = RecordingTransport([empty] * 3)
            with self.assertRaises(OpenRouterError):
                self._extract(transport, None, temporary)
            self.assertEqual(len(transport.payloads), 3)

    def test_review_without_images_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            extractor = OpenRouterVisionExtractor(
                OpenRouterSettings(api_key="sk-or-v1-test"), transport=RecordingTransport([])
            )
            with self.assertRaises(OpenRouterError):
                extractor.extract(
                    case_token="CASE-1",
                    images=(),
                    artifacts=ArtifactStore(Path(temporary) / "artifacts"),
                )


if __name__ == "__main__":
    unittest.main()
