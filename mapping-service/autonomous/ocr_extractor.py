"""Identify a record from locally-OCR'd page text rather than from page images.

Same `ObservationExtractor` contract as the vision extractor, and the same blind
discipline: pages are still referred to by their neutral names, and the model is
still told nothing about the mapping it is checking.

What changes is where the reading happens. PaddleOCR reads the pages on the
local GPU and only its text is sent out, which takes a case request from roughly
a megabyte of base64 images to a few kilobytes. Measured on real Exeter
microfiche, OCR recovers the printed title block cleanly -- the address and the
reference, which are the strongest identity signals -- while degrading
hand-lettered text, so the model is asked to read through that noise rather than
to trust it verbatim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .compiler import strict_output_schema
from .content_qa import ContentQaError
from .openrouter_vision import OpenRouterError, OpenRouterSettings, OpenRouterVisionExtractor
from .paddle_ocr import PaddleOcrError, PaddleOcrRunner, PageText
from .path_policy import require_unprotected_path
from .review_budget import ReviewBudget
from .schemas import ContentObservation
from .storage import ArtifactStore


def ocr_observation_prompt(pages: tuple[PageText, ...]) -> str:
    names = ", ".join(page.name for page in pages)
    body = "\n\n".join(page.as_prompt_block() for page in pages)
    return f"""Perform blind factual extraction from OCR text taken from council planning-record pages.

The pages have neutral names ({names}). Do not infer any value from the names, and do not decide whether
the record matches another record. Extract only what the text supports.

The text comes from optical character recognition of archival microfiche and contains recognition errors:
run-together words, transposed letters, and digits lost from dates. Correct an obvious recognition error
when the intended value is unambiguous from context, and leave a field null when it is not. Do not invent
a value that the text does not support.

For each distinct planning identity present, return its application/reference number, application-site or
property address, proposal/description, relevant application/decision/registration date, and document
type. Property address means the application site; exclude applicant, agent, council-office,
correspondence, and neighbouring-consultee addresses. Cite the neutral page name and a short quoted phrase
in evidence, quoting the OCR text as it appears.

Set readable=false only when no page carries enough legible record content to extract an identity. List
every page reviewed, and return only the schema object.

BEGIN PAGE TEXT
{body}
END PAGE TEXT
"""


@dataclass
class PaddleOcrObservationExtractor:
    """OCR locally, then structure the text with a text-only model call."""

    settings: OpenRouterSettings
    budget: ReviewBudget
    runner: PaddleOcrRunner = PaddleOcrRunner()
    transport: object | None = None

    def __post_init__(self) -> None:
        # Reuse the vision extractor's transport, retry, and soft-failure
        # handling rather than maintaining a second HTTP client.
        self._client = OpenRouterVisionExtractor(
            self.settings, budget=self.budget, transport=self.transport
        )

    def extract(
        self,
        *,
        case_token: str,
        images: tuple[Path, ...],
        artifacts: ArtifactStore,
    ) -> tuple[ContentObservation, Path]:
        if not images:
            raise ContentQaError("Content extraction requires at least one page image")
        images = tuple(
            require_unprotected_path(image, operation="read content QA image")
            for image in images
        )
        case_root = f"qa/cases/{case_token}"
        raw_output = artifacts.resolve(f"{case_root}/observation.raw.json")
        if raw_output.exists():
            return (
                ContentObservation.model_validate_json(raw_output.read_text(encoding="utf-8")),
                raw_output,
            )

        self.budget.ensure_affordable()
        try:
            pages = self.runner.read(images)
        except PaddleOcrError as exc:
            raise ContentQaError(f"Local OCR failed for {case_token}: {exc}") from exc

        artifacts.write_immutable_json(
            f"{case_root}/ocr-text.json",
            [
                {"name": page.name, "error": page.error, "blocks": list(page.blocks)}
                for page in pages
            ],
        )
        if not any(page.readable for page in pages):
            # No page yielded text. This is a readability fact about the scans,
            # not a mapping defect, and the judge treats it as such.
            observation = ContentObservation(
                readable=False,
                images_reviewed=len(pages),
                identities=(),
                unreadable_image_names=tuple(page.name for page in pages),
                warnings=("PaddleOCR recognised no text on any page",),
            )
            stored = artifacts.write_immutable(
                f"{case_root}/observation.raw.json",
                observation.model_dump_json(indent=2).encode("utf-8"),
            )
            return observation, stored

        schema = strict_output_schema(ContentObservation.model_json_schema(mode="validation"))
        artifacts.write_immutable_json("qa/content-observation.schema.json", schema)
        payload = {
            "model": self.settings.model,
            "messages": [{"role": "user", "content": ocr_observation_prompt(pages)}],
            "temperature": 0,
            "usage": {"include": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "content_observation",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        response = self._client._call(payload)

        usage = response.get("usage") or {}
        cost = usage.get("cost")
        self.budget.record(float(cost) if isinstance(cost, (int, float)) else None)
        artifacts.write_mutable(
            f"{case_root}/openrouter-usage.json",
            json.dumps(
                {
                    "model": response.get("model"),
                    "mode": "paddleocr+text",
                    "usage": usage,
                    "budget": self.budget.describe(),
                },
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
        )

        text = ((response.get("choices") or [{}])[0].get("message") or {}).get("content")
        try:
            observation = ContentObservation.model_validate_json(text)
        except Exception as exc:
            failed = artifacts.write_mutable(
                f"{case_root}/observation.failed.json", (text or "").encode("utf-8")
            )
            raise OpenRouterError(
                f"Invalid structured content observation; inspect {failed}: {exc}"
            ) from exc

        allowed = {page.name for page in pages}
        cited = {name for identity in observation.identities for name in identity.image_names}
        unknown = sorted(cited - allowed)
        if unknown:
            raise OpenRouterError(f"Observation cites unattached page names: {unknown}")

        stored = artifacts.write_immutable(
            f"{case_root}/observation.raw.json",
            observation.model_dump_json(indent=2).encode("utf-8"),
        )
        return observation, stored
