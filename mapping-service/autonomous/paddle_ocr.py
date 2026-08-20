"""Read page text locally with PaddleOCR instead of shipping images to a model.

Sending page scans to a vision model costs a megabyte-scale request per case and
fails whenever the provider hiccups. PaddleOCR runs on the local GPU, so the
pages never leave the host and the only thing that goes out is a few kilobytes
of text.

PaddleOCR lives in its own virtualenv with its own pinned paddle build, so it is
driven as a subprocess rather than imported. One call handles every page of a
case: model load costs about 1.4s and dominates a 2.7s inference, so per-image
processes would spend most of their time loading.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_PADDLE_PYTHON = Path("/env/venv/paddleocr_v6/bin/python")
DEFAULT_TIMEOUT_SECONDS = 900

# Recognition confidence below this is dropped. Measured on real Exeter
# microfiche: hand-lettered construction notes score 0.7-0.9 while the printed
# title block scores 0.94+, and the title block is what identifies a record.
DEFAULT_MIN_SCORE = 0.5

# Executed by the PaddleOCR virtualenv, which cannot import this package.
RUNNER = r'''
import json, sys
from paddleocr import PaddleOCR

paths = sys.argv[1:]
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
)
out = {}
for path in paths:
    blocks = []
    try:
        for result in ocr.predict(path):
            data = result.json["res"] if hasattr(result, "json") else result
            for text, score in zip(data.get("rec_texts", []), data.get("rec_scores", [])):
                blocks.append({"text": text, "score": float(score)})
    except Exception as exc:
        out[path] = {"error": f"{type(exc).__name__}: {exc}"}
        continue
    out[path] = {"blocks": blocks}
print("@@OCR_JSON@@" + json.dumps(out, ensure_ascii=False))
'''


class PaddleOcrError(RuntimeError):
    pass


@dataclass(frozen=True)
class PageText:
    name: str
    blocks: tuple[str, ...]
    error: str = ""

    @property
    def readable(self) -> bool:
        return bool(self.blocks)

    def as_prompt_block(self) -> str:
        if self.error:
            return f"[{self.name}] OCR failed: {self.error}"
        if not self.blocks:
            return f"[{self.name}] no text recognised"
        return f"[{self.name}]\n" + "\n".join(self.blocks)


@dataclass(frozen=True)
class PaddleOcrRunner:
    python_executable: Path = DEFAULT_PADDLE_PYTHON
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    min_score: float = DEFAULT_MIN_SCORE

    def available(self) -> bool:
        return self.python_executable.is_file() and os.access(self.python_executable, os.X_OK)

    def read(self, images: Sequence[Path]) -> tuple[PageText, ...]:
        if not images:
            return ()
        if not self.available():
            raise PaddleOcrError(
                f"PaddleOCR interpreter is not executable: {self.python_executable}"
            )
        command = [str(self.python_executable), "-c", RUNNER, *(str(item) for item in images)]
        try:
            result = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PaddleOcrError(f"PaddleOCR timed out after {self.timeout_seconds}s") from exc
        if result.returncode != 0:
            raise PaddleOcrError(
                f"PaddleOCR exited {result.returncode}: {(result.stderr or '')[-1500:]}"
            )

        # Paddle writes load warnings to stdout, so the payload is delimited
        # rather than assumed to be the whole stream.
        marker = "@@OCR_JSON@@"
        index = (result.stdout or "").rfind(marker)
        if index < 0:
            raise PaddleOcrError(
                f"PaddleOCR produced no result payload: {(result.stdout or '')[-1500:]}"
            )
        try:
            payload = json.loads(result.stdout[index + len(marker) :])
        except json.JSONDecodeError as exc:
            raise PaddleOcrError("PaddleOCR result payload is not JSON") from exc

        pages: list[PageText] = []
        for image in images:
            entry = payload.get(str(image)) or {}
            if entry.get("error"):
                pages.append(PageText(name=image.name, blocks=(), error=str(entry["error"])))
                continue
            blocks = tuple(
                str(block["text"]).strip()
                for block in entry.get("blocks", [])
                if float(block.get("score", 0)) >= self.min_score and str(block.get("text", "")).strip()
            )
            pages.append(PageText(name=image.name, blocks=blocks))
        return tuple(pages)
