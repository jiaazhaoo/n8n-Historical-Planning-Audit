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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .content_qa import (
    ADDRESS_STOPWORDS,
    ContentQaError,
    canonical_reference_key,
    date_keys,
    description_matches,
    document_carries_reference,
    word_tokens,
)
from .paddle_ocr import PaddleOcrError, PaddleOcrRunner
from .schemas import ContentExpectation, QaVerdict
from .storage import ArtifactStore


# Below this, a document has too little recognised text for a non-match to mean
# anything about the mapping.
MIN_TEXT_CHARACTERS = 200

NO_SIGNALS = {"reference": False, "address": False, "description": False, "date": False}


# A UK postcode, outward or inward half. Source records carry one; a 1980s
# planning drawing almost never prints it.
POSTCODE = re.compile(r"^(?:[a-z]{1,2}\d{1,2}[a-z]?|\d[a-z]{2})$")

# Share of the address's own words that must appear in the document.
ADDRESS_OVERLAP = 0.6


def address_in_document(expected: str, text: str, *, council: str = "") -> bool:
    """Match a source address against a whole document.

    `address_matches` compares one address with another, where a differing house
    number means a different property and both sides carry a postcode. A
    document is not an address: it prints the street and the building name but
    often neither the number nor the postcode, and it holds hundreds of
    unrelated numbers. Counting those against the address rejected documents
    that plainly are the right ones -- "8 Oxford Road Exeter EX4 6QU" scored 2
    of 4 because ex4 and 6qu appear nowhere on a 1989 drawing, and a Topsham
    Road site was rejected outright because the plan does not print "398".

    So the ratio is taken over the words that name the place, and the house
    number corroborates a match rather than being required for one.
    """
    # The council's own name is on every document it issued, so it can raise a
    # score without telling one site from another.
    generic = {word for word in re.findall(r"[a-z']{3,}", council.lower())}
    expected_tokens = {
        token
        for token in word_tokens(expected, stopwords=ADDRESS_STOPWORDS)
        if not POSTCODE.match(token) and not token.isdigit() and token not in generic
    }
    if not expected_tokens:
        return False
    observed = word_tokens(text, stopwords=ADDRESS_STOPWORDS)
    if len(expected_tokens) == 1:
        # A one-word address such as "8 Oxford Road Exeter" has a single name to
        # go on; a ratio would be meaningless, so require that one word.
        return expected_tokens <= observed
    overlap = expected_tokens & observed
    return len(overlap) >= 2 and len(overlap) / len(expected_tokens) >= ADDRESS_OVERLAP


def _address_words(value: str) -> set[str]:
    """Words long enough to identify a place, lowercased."""
    return {word for word in re.findall(r"[a-z']{4,}", value.lower())}


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
                address_in_document(value, text, council=expectation.council)
                for value in _usable(expectation.address_values)
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
        """Name another sampled case only when the document says so distinctly.

        Neighbouring records share a locality: "Marsh Barton, Exeter" appears in
        both a sorting office on Alphinbrook Road and land off Manaton Close.
        Matching on shared words alone accused a correct mapping of being a
        swap, which is the most damaging verdict this gate can return, so the
        evidence has to be a word that tells the two cases apart.
        """
        lowered = text.lower()
        own = {
            word
            for value in _usable(expectation.address_values)
            for word in _address_words(value)
        }
        for oachargeid, addresses in self._addresses_by_case.items():
            if oachargeid == expectation.oachargeid:
                continue
            for value in addresses:
                if not address_in_document(value, text, council=expectation.council):
                    continue
                distinguishing = _address_words(value) - own
                if any(word in lowered for word in distinguishing):
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
            # Same reasoning as the judge: with no address recorded, whether the
            # scan filed under this folder carries the case reference is the
            # only question left, and it is a real one.
            if document_carries_reference(text, expectation):
                signals["reference"] = True
                verdict, confidence, reason = (
                    QaVerdict.VERIFIED_REFERENCE_ONLY,
                    0.80,
                    "The document carries the expected reference; the source records no address "
                    "or description to corroborate it against",
                )
            else:
                verdict, confidence, reason = (
                    QaVerdict.RULE_SUPPORTED_UNVERIFIED,
                    0.0,
                    "The source records no address or description, and the document does not "
                    f"carry the expected reference {list(expectation.reference_values)[:2]}",
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
        eligible = verdict in {QaVerdict.VERIFIED_SAME, QaVerdict.VERIFIED_REFERENCE_ONLY}
        return verdict, confidence, eligible, signals, None, reason, evidence
