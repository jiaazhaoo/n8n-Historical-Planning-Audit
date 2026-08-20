from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

AUTONOMOUS_ROOT = Path(__file__).resolve().parents[1] / "amazons3-mapping"
if str(AUTONOMOUS_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTONOMOUS_ROOT))

from autonomous.ocr_extractor import (  # noqa: E402
    PaddleOcrObservationExtractor,
    ocr_observation_prompt,
)
from autonomous.openrouter_vision import OpenRouterError, OpenRouterSettings  # noqa: E402
from autonomous.paddle_ocr import PaddleOcrError, PaddleOcrRunner, PageText  # noqa: E402
from autonomous.review_budget import ReviewBudget  # noqa: E402
from autonomous.storage import ArtifactStore  # noqa: E402


OBSERVATION = {
    "schema_version": 1,
    "readable": True,
    "images_reviewed": 1,
    "identities": [
        {
            "reference": "03/81/1514",
            "property_address": "23 Wendover Way, Countess Wear, Exeter",
            "description": "Extension",
            "relevant_date": "1982-01-18",
            "document_type": "Grant of Conditional Planning Permission",
            "image_names": ["page-0001.jpg"],
            "evidence": ["page-0001.jpg: 23,WENDOVER WAY EXETER"],
        }
    ],
    "unreadable_image_names": [],
    "warnings": [],
}


def completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def ocr_payload(path: Path, blocks) -> str:
    # Paddle prints load warnings before the payload, which the marker skips.
    warning = "UserWarning: No ccache found\n"
    return warning + "@@OCR_JSON@@" + json.dumps({str(path): {"blocks": blocks}})


class RecordingTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads: list[dict] = []

    def __call__(self, payload):
        self.payloads.append(payload)
        return self.responses.pop(0)


def response(content=None, cost: float = 0.005) -> dict:
    return {
        "model": "google/gemini-3.7-flash",
        "usage": {"cost": cost},
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content or OBSERVATION)}}],
    }


class PaddleOcrRunnerTests(unittest.TestCase):
    def test_text_below_the_confidence_floor_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "page-0001.jpg"
            image.write_bytes(b"x")
            blocks = [
                {"text": "23,WENDOVER WAY EXETER", "score": 0.94},
                {"text": "gibberish", "score": 0.21},
            ]
            runner = PaddleOcrRunner(python_executable=Path(sys.executable), min_score=0.5)
            with patch("subprocess.run", return_value=completed(ocr_payload(image, blocks))):
                pages = runner.read([image])
            self.assertEqual(pages[0].blocks, ("23,WENDOVER WAY EXETER",))

    def test_load_warnings_before_the_payload_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "page-0001.jpg"
            image.write_bytes(b"x")
            runner = PaddleOcrRunner(python_executable=Path(sys.executable))
            with patch(
                "subprocess.run",
                return_value=completed(ocr_payload(image, [{"text": "1514", "score": 0.99}])),
            ):
                pages = runner.read([image])
            self.assertEqual(pages[0].blocks, ("1514",))

    def test_a_per_page_error_is_reported_without_failing_the_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "page-0001.jpg"
            image.write_bytes(b"x")
            payload = "@@OCR_JSON@@" + json.dumps({str(image): {"error": "ValueError: bad image"}})
            runner = PaddleOcrRunner(python_executable=Path(sys.executable))
            with patch("subprocess.run", return_value=completed(payload)):
                pages = runner.read([image])
            self.assertFalse(pages[0].readable)
            self.assertIn("bad image", pages[0].error)

    def test_a_nonzero_exit_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "page-0001.jpg"
            image.write_bytes(b"x")
            runner = PaddleOcrRunner(python_executable=Path(sys.executable))
            with patch("subprocess.run", return_value=completed("", returncode=1)):
                with self.assertRaises(PaddleOcrError):
                    runner.read([image])

    def test_output_without_the_marker_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "page-0001.jpg"
            image.write_bytes(b"x")
            runner = PaddleOcrRunner(python_executable=Path(sys.executable))
            with patch("subprocess.run", return_value=completed("only warnings, no payload")):
                with self.assertRaises(PaddleOcrError):
                    runner.read([image])

    def test_a_missing_interpreter_is_reported_before_running(self) -> None:
        runner = PaddleOcrRunner(python_executable=Path("/nonexistent/python"))
        self.assertFalse(runner.available())
        with self.assertRaises(PaddleOcrError):
            runner.read([Path("/tmp/page-0001.jpg")])

    def test_no_images_needs_no_subprocess(self) -> None:
        runner = PaddleOcrRunner(python_executable=Path("/nonexistent/python"))
        self.assertEqual(runner.read([]), ())


class PromptTests(unittest.TestCase):
    def test_the_prompt_carries_page_text_under_neutral_names(self) -> None:
        pages = (PageText(name="page-0001.jpg", blocks=("23,WENDOVER WAY EXETER", "1514")),)
        prompt = ocr_observation_prompt(pages)
        self.assertIn("page-0001.jpg", prompt)
        self.assertIn("23,WENDOVER WAY EXETER", prompt)
        # The model must be told the text is noisy, or it treats OCR slips as fact.
        self.assertIn("recognition errors", prompt)

    def test_an_unreadable_page_is_stated_rather_than_omitted(self) -> None:
        pages = (PageText(name="page-0002.jpg", blocks=(), error="ValueError: bad image"),)
        self.assertIn("OCR failed", ocr_observation_prompt(pages))


class PaddleOcrObservationExtractorTests(unittest.TestCase):
    def _extractor(self, transport, pages, budget=None):
        class FakeRunner:
            def read(self, images):
                return pages

        return PaddleOcrObservationExtractor(
            settings=OpenRouterSettings(api_key="sk-or-v1-test", retry_backoff_seconds=0),
            budget=budget or ReviewBudget(limit_usd=3.0),
            runner=FakeRunner(),
            transport=transport,
        )

    def test_page_text_is_structured_and_the_call_is_charged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "page-0001.jpg"
            image.write_bytes(b"x")
            budget = ReviewBudget(limit_usd=3.0)
            transport = RecordingTransport([response(cost=0.005)])
            extractor = self._extractor(
                transport, (PageText(name="page-0001.jpg", blocks=("1514",)),), budget
            )
            observation, path = extractor.extract(
                case_token="CASE-1",
                images=(image,),
                artifacts=ArtifactStore(Path(temporary) / "artifacts"),
            )
            self.assertEqual(observation.identities[0].reference, "03/81/1514")
            self.assertTrue(path.is_file())
            self.assertAlmostEqual(budget.spent_usd, 0.005)

    def test_no_image_is_sent_only_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "page-0001.jpg"
            image.write_bytes(b"x")
            transport = RecordingTransport([response()])
            extractor = self._extractor(
                transport, (PageText(name="page-0001.jpg", blocks=("1514",)),)
            )
            extractor.extract(
                case_token="CASE-1",
                images=(image,),
                artifacts=ArtifactStore(Path(temporary) / "artifacts"),
            )
            body = json.dumps(transport.payloads[0])
            self.assertNotIn("image_url", body)
            self.assertNotIn("base64", body)

    def test_pages_without_text_are_unreadable_without_calling_the_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "page-0001.jpg"
            image.write_bytes(b"x")
            budget = ReviewBudget(limit_usd=3.0)
            transport = RecordingTransport([])
            extractor = self._extractor(
                transport, (PageText(name="page-0001.jpg", blocks=()),), budget
            )
            observation, _ = extractor.extract(
                case_token="CASE-1",
                images=(image,),
                artifacts=ArtifactStore(Path(temporary) / "artifacts"),
            )
            self.assertFalse(observation.readable)
            self.assertEqual(transport.payloads, [])
            self.assertEqual(budget.calls, 0)

    def test_the_ocr_text_is_kept_next_to_the_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "page-0001.jpg"
            image.write_bytes(b"x")
            artifacts = ArtifactStore(Path(temporary) / "artifacts")
            extractor = self._extractor(
                RecordingTransport([response()]),
                (PageText(name="page-0001.jpg", blocks=("1514",)),),
            )
            extractor.extract(case_token="CASE-1", images=(image,), artifacts=artifacts)
            saved = json.loads(
                artifacts.resolve("qa/cases/CASE-1/ocr-text.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved[0]["blocks"], ["1514"])

    def test_an_observation_citing_an_unattached_page_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "page-0001.jpg"
            image.write_bytes(b"x")
            hostile = json.loads(json.dumps(OBSERVATION))
            hostile["identities"][0]["image_names"] = ["EXE_1981_81-1514-03_0001.jpg"]
            extractor = self._extractor(
                RecordingTransport([response(hostile)]),
                (PageText(name="page-0001.jpg", blocks=("1514",)),),
            )
            with self.assertRaises(OpenRouterError):
                extractor.extract(
                    case_token="CASE-1",
                    images=(image,),
                    artifacts=ArtifactStore(Path(temporary) / "artifacts"),
                )


if __name__ == "__main__":
    unittest.main()
