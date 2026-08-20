"""Judge document/record correspondence with a cheap model over local OCR text.

The pages are still read on this host by PaddleOCR, so no image leaves it; what
goes out is a few kilobytes of text and what comes back is a judgement with the
phrase it rested on.

Why a model rather than the string matcher: measured head to head on the same
ten real Exeter WP3 records cross-matched against the same ten documents, the
matcher scored 8/10 with a 4% false-positive rate while the model scored 9/10 at
1%. Eighteen tunings of the matcher never beat both at once, because token
overlap cannot tell an application-site address from the agent's address printed
on the same page. At $0.00035 a case that distinction is worth buying.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .content_qa import ContentQaError, address_matches
from .local_verifier import MIN_TEXT_CHARACTERS, NO_SIGNALS, _usable
from .openrouter_vision import OpenRouterError, OpenRouterSettings, OpenRouterVisionExtractor
from .paddle_ocr import PaddleOcrError, PaddleOcrRunner
from .review_budget import ReviewBudget
from .schemas import ContentExpectation, QaVerdict
from .storage import ArtifactStore


DEFAULT_JUDGE_MODEL = "deepseek/deepseek-v4-flash"

# Long enough to carry a planning file's identifying pages, short enough to keep
# a case at a third of a cent.
MAX_TEXT_CHARACTERS = 12000

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "readable",
        "belongs",
        "matched_address",
        "matched_description",
        "document_site",
        "evidence",
    ],
    "properties": {
        "readable": {"type": "boolean"},
        "belongs": {"type": "boolean"},
        "matched_address": {"type": "boolean"},
        "matched_description": {"type": "boolean"},
        "document_site": {"type": "string"},
        "evidence": {"type": "string"},
    },
}


def judge_prompt(expectation: ContentExpectation, text: str) -> str:
    address = "; ".join(_usable(expectation.address_values)) or "(not stated)"
    description = "; ".join(_usable(expectation.description_values)) or "(not stated)"
    return f"""Decide whether this scanned council planning file is the file for one specific land-charge record.

The record states:
  application site address: {address}
  proposal/description: {description}

Below is OCR text from the scanned file. It is archival microfiche, so it contains recognition errors,
run-together words and missing characters. Judge through that noise rather than treating it as exact.

A file belongs to the record when the site it concerns is the same place the record names. Planning files
also carry applicant, agent and council addresses; those do not make a file belong. A shared town or
industrial estate is not enough on its own -- the street or building name has to agree.

Set matched_address true only when the application site in the file agrees with the record's, and quote
the phrase you relied on in evidence. In document_site, write the application site the file is actually
about, as the file states it, whether or not it matches; leave it empty only if no site is legible. Set
readable false when the text carries too little record content to judge at all.

BEGIN OCR TEXT
{text[:MAX_TEXT_CHARACTERS]}
END OCR TEXT"""


@dataclass
class LlmJudgeVerifier:
    """Reads pages locally, then asks a cheap model whether they fit the record."""

    settings: OpenRouterSettings
    budget: ReviewBudget
    runner: PaddleOcrRunner = field(default_factory=PaddleOcrRunner)
    transport: object | None = None
    _addresses_by_case: dict[str, tuple[str, ...]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        # Reuse the existing client's retry and soft-failure handling rather
        # than maintaining a second HTTP path.
        self._client = OpenRouterVisionExtractor(
            self.settings, budget=self.budget, transport=self.transport
        )

    def prepare(self, expectations: Sequence[ContentExpectation]) -> None:
        self._addresses_by_case = {
            expectation.oachargeid: tuple(_usable(expectation.address_values))
            for expectation in expectations
        }

    def _swapped_with(self, expectation: ContentExpectation, document_site: str) -> str | None:
        """Name the case a document actually belongs to, if it is in the sample.

        The model reports the site the file is about; comparing that one short
        address against the other sampled records is an address-to-address
        question, which is what `address_matches` is for.
        """
        if not document_site.strip():
            return None
        for oachargeid, addresses in self._addresses_by_case.items():
            if oachargeid == expectation.oachargeid:
                continue
            if any(address_matches(value, document_site) for value in addresses):
                return oachargeid
        return None

    def _ask(self, expectation: ContentExpectation, text: str) -> dict:
        payload = {
            "model": self.settings.model,
            "messages": [{"role": "user", "content": judge_prompt(expectation, text)}],
            "temperature": 0,
            "usage": {"include": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "belongs", "strict": True, "schema": JUDGE_SCHEMA},
            },
        }
        response = self._client._call(payload)
        usage = response.get("usage") or {}
        cost = usage.get("cost")
        self.budget.record(float(cost) if isinstance(cost, (int, float)) else None)
        content = ((response.get("choices") or [{}])[0].get("message") or {}).get("content")
        try:
            return json.loads(content)
        except Exception as exc:
            raise OpenRouterError(f"Judge returned unreadable JSON: {str(content)[:200]}") from exc

    def verify(
        self,
        *,
        expectation: ContentExpectation,
        images: tuple[Path, ...],
        artifacts: ArtifactStore,
        case_token: str,
    ) -> tuple[QaVerdict, float, bool, dict[str, bool], int | None, str, Path]:
        try:
            pages = self.runner.read(images)
        except PaddleOcrError as exc:
            raise ContentQaError(f"Local OCR failed for {case_token}: {exc}") from exc
        text = " \n".join(block for page in pages for block in page.blocks)
        artifacts.write_immutable_json(
            f"qa/cases/{case_token}/ocr-text.json",
            [{"name": page.name, "error": page.error, "blocks": list(page.blocks)} for page in pages],
        )

        signals = dict(NO_SIGNALS)
        judgement: dict = {}
        swapped: str | None = None

        if not _usable(expectation.address_values) and not _usable(expectation.description_values):
            verdict, confidence, reason = (
                QaVerdict.RULE_SUPPORTED_UNVERIFIED,
                0.0,
                "The source record states no address or description, so its document cannot be "
                "checked against it",
            )
        elif len(text) < MIN_TEXT_CHARACTERS:
            verdict, confidence, reason = (
                QaVerdict.UNREADABLE,
                0.0,
                f"Only {len(text)} characters were recognised across {len(pages)} page(s)",
            )
        else:
            self.budget.ensure_affordable()
            judgement = self._ask(expectation, text)
            signals = {
                "reference": False,
                "address": bool(judgement.get("matched_address")),
                "description": bool(judgement.get("matched_description")),
                "date": False,
            }
            swapped = self._swapped_with(expectation, str(judgement.get("document_site") or ""))
            evidence = str(judgement.get("evidence") or "").strip()[:200]

            if not judgement.get("readable", True):
                verdict, confidence, reason = (
                    QaVerdict.UNREADABLE,
                    0.0,
                    "The reviewer found too little legible record content to judge",
                )
            elif judgement.get("matched_address"):
                verdict, confidence, reason = (
                    QaVerdict.VERIFIED_SAME,
                    0.90,
                    f"The file's application site matches the source record: {evidence}",
                )
            elif swapped:
                verdict, confidence, reason = (
                    QaVerdict.VERIFIED_WRONG,
                    0.0,
                    f"The file concerns {str(judgement.get('document_site'))[:80]!r}, which is "
                    f"{swapped}'s site, not this record's",
                )
            elif judgement.get("belongs") and judgement.get("matched_description"):
                verdict, confidence, reason = (
                    QaVerdict.VERIFIED_SAME,
                    0.90,
                    f"The file's proposal matches the source record: {evidence}",
                )
            else:
                verdict, confidence, reason = (
                    QaVerdict.RULE_SUPPORTED_UNVERIFIED,
                    0.0,
                    "The file's application site could not be reconciled with the source record; "
                    f"the file appears to concern {str(judgement.get('document_site'))[:80]!r}",
                )

        evidence_path = artifacts.write_immutable_json(
            f"qa/cases/{case_token}/judge.json",
            {
                "oachargeid": expectation.oachargeid,
                "model": self.settings.model,
                "verdict": verdict.value,
                "reason": reason,
                "signals": signals,
                "swapped_with": swapped,
                "judgement": judgement,
                "expected": {
                    "address": list(expectation.address_values),
                    "description": list(expectation.description_values),
                },
                "text_characters": len(text),
                "pages": len(pages),
            },
        )
        return verdict, confidence, verdict == QaVerdict.VERIFIED_SAME, signals, None, reason, evidence_path
