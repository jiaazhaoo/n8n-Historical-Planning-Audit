"""Verify a mapping by looking for the source record inside the document text.

Extraction and verification are different problems. Extracting the record's
identity from 480 OCR fragments needs to know which fragment is the site address
and which is the agent's; verifying only needs to know whether the address the
source already states appears in the document at all. The second question is
answerable with a deterministic matcher, so no model is involved and no page
text leaves the host.

Measured on 10 real Exeter WP3 cases whose source addresses and delivered S3
paths were both known, cross-matching every address against every document:
address alone identified the right document in 8 of 10 cases with a 3% false
positive rate. Of the two misses, one is a typo in the source record itself
("National Wesminster Bank"), which a matcher is right to flag and a generative
model would likely have smoothed over.

Blindness is not needed here for the reason it is needed with a model. A model
told the expected answer can report finding it; a string matcher either finds
the text or does not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .content_qa import (
    ContentQaError,
    address_matches,
    canonical_reference_key,
    date_keys,
    description_matches,
)
from .paddle_ocr import PaddleOcrError, PaddleOcrRunner
from .schemas import ContentExpectation, QaVerdict
from .storage import ArtifactStore


# Below this, a document has too little recognised text for a non-match to mean
# anything about the mapping.
MIN_TEXT_CHARACTERS = 200

NO_SIGNALS = {"reference": False, "address": False, "description": False, "date": False}


def _usable(values: Sequence[str]) -> list[str]:
    return [value for value in values if value and value.strip().lower() not in {"", "none"}]


@dataclass
class OcrTextVerifier:
    """Decide a case by matching its source facts against the document's text."""

    runner: PaddleOcrRunner = field(default_factory=PaddleOcrRunner)
    council: str = ""
    _addresses_by_case: dict[str, tuple[str, ...]] = field(default_factory=dict, init=False)

    def prepare(self, expectations: Sequence[ContentExpectation]) -> None:
        """Remember every case's expected address, so a swap can be named.

        A document that does not match its own case but does match another
        sampled case is direct evidence the two were exchanged, which is a
        stronger finding than "could not verify".
        """
        self._addresses_by_case = {
            expectation.oachargeid: tuple(_usable(expectation.address_values))
            for expectation in expectations
        }

    def _signals(self, expectation: ContentExpectation, text: str) -> dict[str, bool]:
        lowered = text.lower()
        expected_refs = {
            canonical_reference_key(value, expectation.council)
            for value in expectation.reference_values
            if canonical_reference_key(value, expectation.council)
        }
        reference_hit = any(
            key and key in canonical_reference_key(lowered, expectation.council)
            for key in expected_refs
        ) or any(value.strip().lower() in lowered for value in _usable(expectation.reference_values))
        return {
            "reference": bool(reference_hit),
            "address": any(
                address_matches(value, text) for value in _usable(expectation.address_values)
            ),
            "description": any(
                description_matches(value, text)
                for value in _usable(expectation.description_values)
            ),
            "date": any(
                date_keys(value) & date_keys(text) for value in _usable(expectation.date_values)
            ),
        }

    def _swapped_with(self, expectation: ContentExpectation, text: str) -> str | None:
        for oachargeid, addresses in self._addresses_by_case.items():
            if oachargeid == expectation.oachargeid:
                continue
            if any(address_matches(value, text) for value in addresses):
                return oachargeid
        return None

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
        signals = self._signals(expectation, text) if text else dict(NO_SIGNALS)
        swapped = self._swapped_with(expectation, text) if text else None

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
        elif signals["address"]:
            verdict, confidence, reason = (
                QaVerdict.VERIFIED_SAME,
                0.90,
                "The source application-site address appears in the document text",
            )
        elif signals["reference"] and (signals["description"] or signals["date"]):
            verdict, confidence, reason = (
                QaVerdict.VERIFIED_SAME,
                0.90,
                "The source reference plus an independent source fact appear in the document text",
            )
        elif swapped:
            verdict, confidence, reason = (
                QaVerdict.VERIFIED_WRONG,
                0.0,
                f"The document does not carry this record's address but carries {swapped}'s, "
                "which is evidence the two mappings were exchanged",
            )
        else:
            verdict, confidence, reason = (
                QaVerdict.RULE_SUPPORTED_UNVERIFIED,
                0.0,
                "No source fact was found in the document text; the mapping is unconfirmed rather "
                "than shown to be wrong",
            )

        evidence = artifacts.write_immutable_json(
            f"qa/cases/{case_token}/local-match.json",
            {
                "oachargeid": expectation.oachargeid,
                "verdict": verdict.value,
                "reason": reason,
                "signals": signals,
                "swapped_with": swapped,
                "expected": {
                    "reference": list(expectation.reference_values),
                    "address": list(expectation.address_values),
                    "description": list(expectation.description_values),
                },
                "pages": [
                    {"name": page.name, "blocks": len(page.blocks), "error": page.error}
                    for page in pages
                ],
                "text_characters": len(text),
            },
        )
        artifacts.write_immutable_json(
            f"qa/cases/{case_token}/ocr-text.json",
            [{"name": page.name, "error": page.error, "blocks": list(page.blocks)} for page in pages],
        )
        return verdict, confidence, verdict == QaVerdict.VERIFIED_SAME, signals, None, reason, evidence
