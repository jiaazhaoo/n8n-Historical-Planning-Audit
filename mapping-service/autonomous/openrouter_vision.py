"""Blind page extraction through OpenRouter, priced per call against a budget.

Interchangeable with `CodexOAuthVisionExtractor`: same `ObservationExtractor`
contract, same neutral-name discipline, same structured `ContentObservation`.
What differs is that every call reports its own cost, so a round can be held to
a spend ceiling instead of a guessed case count.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .compiler import strict_output_schema
from .content_qa import ContentQaError, observation_prompt
from .path_policy import require_unprotected_path
from .review_budget import BudgetExhausted, ReviewBudget
from .schemas import ContentObservation
from .storage import ArtifactStore


DEFAULT_MODEL = "google/gemini-3.7-flash"
DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT_SECONDS = 300
MAX_ATTEMPTS = 3
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# Only the OpenRouter key shape is accepted, so a differently-scoped key cannot
# be pasted in and silently billed to another provider's account.
KEY_PATTERN = re.compile(r"sk-or-v\d+-[A-Za-z0-9_-]{16,}")


class OpenRouterError(ContentQaError):
    pass


def read_api_key(source: Path) -> str:
    """Pull the OpenRouter key out of a local key file.

    The keys file holds several providers under free-form labels, so the key is
    located by its own prefix rather than by the label next to it.
    """
    source = require_unprotected_path(source, operation="read OpenRouter API key")
    match = KEY_PATTERN.search(source.read_text(encoding="utf-8", errors="replace"))
    if not match:
        raise OpenRouterError(f"No OpenRouter key (sk-or-v...) found in {source}")
    return match.group(0)


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise OpenRouterError(f"Unsupported image type for vision review: {path.name}")
    mime = mimetypes.types_map.get(suffix) or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@dataclass(frozen=True)
class OpenRouterSettings:
    api_key: str
    model: str = DEFAULT_MODEL
    endpoint: str = DEFAULT_ENDPOINT
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    referer: str = "https://localhost/n8n-file-path-mapping"
    title: str = "council mapping content QA"


class OpenRouterVisionExtractor:
    """Extracts one case's visible identities, charging the round's budget."""

    def __init__(
        self,
        settings: OpenRouterSettings,
        *,
        budget: ReviewBudget | None = None,
        transport=None,
    ) -> None:
        if not settings.api_key:
            raise OpenRouterError("An OpenRouter API key is required")
        self.settings = settings
        self.budget = budget or ReviewBudget()
        # Injectable so tests exercise the extractor without a network call.
        self._transport = transport or self._post

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            self.settings.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.settings.referer,
                "X-Title": self.settings.title,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _call(self, payload: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return self._transport(payload)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:2000]
                # 4xx other than rate limiting will not improve on retry.
                if exc.code not in {408, 429} and exc.code < 500:
                    raise OpenRouterError(f"OpenRouter rejected the request ({exc.code}): {body}") from exc
                last_error = OpenRouterError(f"OpenRouter error {exc.code}: {body}")
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = OpenRouterError(f"OpenRouter request failed: {exc}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(2 ** attempt)
        raise last_error or OpenRouterError("OpenRouter request failed")

    def _build_payload(self, images: tuple[Path, ...], schema: dict) -> dict:
        content: list[dict] = [
            {"type": "text", "text": observation_prompt(path.name for path in images)}
        ]
        for path in images:
            # The name is stated next to its image so the model can cite the
            # neutral name the schema requires, without seeing a real filename.
            content.append({"type": "text", "text": f"Attachment: {path.name}"})
            content.append({"type": "image_url", "image_url": {"url": image_data_url(path)}})
        return {
            "model": self.settings.model,
            "messages": [{"role": "user", "content": content}],
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

    def extract(
        self,
        *,
        case_token: str,
        images: tuple[Path, ...],
        artifacts: ArtifactStore,
    ) -> tuple[ContentObservation, Path]:
        if not images:
            raise OpenRouterError("Vision extraction requires at least one image")
        images = tuple(
            require_unprotected_path(image, operation="read content QA image")
            for image in images
        )
        case_root = f"qa/cases/{case_token}"
        raw_output = artifacts.resolve(f"{case_root}/observation.raw.json")
        if raw_output.exists():
            # Already reviewed in an earlier attempt at this round; re-reading it
            # costs nothing and must not be charged to the budget again.
            return (
                ContentObservation.model_validate_json(raw_output.read_text(encoding="utf-8")),
                raw_output,
            )

        self.budget.ensure_affordable()
        schema = strict_output_schema(ContentObservation.model_json_schema(mode="validation"))
        artifacts.write_immutable_json("qa/content-observation.schema.json", schema)
        response = self._call(self._build_payload(images, schema))

        usage = response.get("usage") or {}
        cost = usage.get("cost")
        self.budget.record(float(cost) if isinstance(cost, (int, float)) else None)
        artifacts.write_mutable(
            f"{case_root}/openrouter-usage.json",
            json.dumps(
                {
                    "model": response.get("model"),
                    "usage": usage,
                    "budget": self.budget.describe(),
                },
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
        )

        choices = response.get("choices") or []
        if not choices:
            raise OpenRouterError(f"OpenRouter returned no choices: {json.dumps(response)[:1000]}")
        message = choices[0].get("message") or {}
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            raise OpenRouterError("OpenRouter returned an empty content observation")

        try:
            observation = ContentObservation.model_validate_json(text)
        except Exception as exc:
            failed = artifacts.write_mutable(
                f"{case_root}/observation.failed.json", text.encode("utf-8")
            )
            raise OpenRouterError(
                f"Invalid structured content observation; inspect {failed}: {exc}"
            ) from exc

        allowed_names = {path.name for path in images}
        cited_names = {name for identity in observation.identities for name in identity.image_names}
        unknown_names = sorted(cited_names - allowed_names)
        if unknown_names:
            raise OpenRouterError(f"Observation cites unattached image names: {unknown_names}")

        stored = artifacts.write_immutable(
            f"{case_root}/observation.raw.json",
            observation.model_dump_json(indent=2).encode("utf-8"),
        )
        return observation, stored


__all__ = [
    "BudgetExhausted",
    "DEFAULT_MODEL",
    "OpenRouterError",
    "OpenRouterSettings",
    "OpenRouterVisionExtractor",
    "read_api_key",
]
