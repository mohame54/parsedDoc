"""Thin wrapper over the google-genai Batch API. The only module that talks to Gemini.

Wire-format details (JSONL keys, result line shape, file_data parts) follow the Batch API docs as
read on 2026-11-12 but are not yet confirmed against a live job; the smoke test must confirm them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SUCCEEDED = "JOB_STATE_SUCCEEDED"
FAILED_STATES = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
TERMINAL_STATES = FAILED_STATES | {SUCCEEDED}

MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}


def prepare_image(path: Path, scratch: Path, max_side: int = 0) -> tuple[Path, str]:
    """Returns a path Gemini accepts plus its MIME type. Other formats (tif, bmp) are converted to PNG copies, and
    with max_side an image whose longest side is larger is shrunk to a copy (smaller upload, same readable text)."""
    from PIL import Image

    suffix = path.suffix.lower()
    mime = MIME_BY_SUFFIX.get(suffix)
    if mime == "application/pdf":
        return path, mime
    if mime and not max_side:
        return path, mime
    scratch.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as image:
        longest = max(image.size)
        if mime and longest <= max_side:
            return path, mime
        jpeg = suffix in (".jpg", ".jpeg")
        out = scratch / (f"{path.stem}.up.jpg" if jpeg else f"{path.stem}.png")
        if not out.exists():
            picture = image.convert("RGB")
            if max_side and longest > max_side:
                scale = max_side / longest
                picture = picture.resize((round(image.width * scale), round(image.height * scale)))
            picture.save(out, quality=90) if jpeg else picture.save(out)
    return out, "image/jpeg" if out.suffix == ".jpg" else "image/png"


def text_part(text: str) -> dict[str, Any]:
    return {"text": text}


def file_part(uri: str, mime_type: str) -> dict[str, Any]:
    return {"file_data": {"file_uri": uri, "mime_type": mime_type}}


def build_request_line(key: str, parts: list[dict], schema: dict, max_output_tokens: int) -> dict[str, Any]:
    return {
        "key": key,
        "request": {
            "contents": [{"role": "user", "parts": parts}],
            "generation_config": {
                "response_mime_type": "application/json",
                "response_schema": schema,
                "max_output_tokens": max_output_tokens,
                "temperature": 0.0,
            },
        },
    }


def write_jsonl(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines), encoding="utf-8")


def extract_text(response: dict) -> str:
    candidates = response.get("candidates") or []
    if not candidates:
        raise ValueError("response has no candidates")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts)


def parse_output_text(jsonl_text: str) -> dict[str, tuple[str | None, str | None]]:
    """Maps request key -> (text, error). Exactly one of the two is set."""
    results: dict[str, tuple[str | None, str | None]] = {}
    for line in jsonl_text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        key = obj.get("key")
        if key is None:
            continue
        if obj.get("error"):
            results[key] = (None, json.dumps(obj["error"], ensure_ascii=False)[:300])
            continue
        try:
            results[key] = (extract_text(obj.get("response") or {}), None)
        except ValueError as exc:
            results[key] = (None, str(exc))
    return results


def _client(api_key: str | None):
    from google import genai

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY before submitting or collecting batches.")
    return genai.Client(api_key=key)


class BatchApi:
    def __init__(self, client: Any = None, api_key: str | None = None) -> None:
        self.client = client if client is not None else _client(api_key)

    def upload_file(self, path: Path, mime_type: str) -> str:
        """Returns the file URI to reference from request parts. Files expire after 48 hours."""
        uploaded = self.client.files.upload(file=str(path), config={"mime_type": mime_type})
        return uploaded.uri

    def upload_jsonl(self, path: Path) -> str:
        uploaded = self.client.files.upload(
            file=str(path), config={"mime_type": "jsonl", "display_name": path.name[:200]}
        )
        return uploaded.name

    def find_job(self, display_name: str) -> str | None:
        """Finds a non-failed job with this display name. Makes re-submission after a crash idempotent."""
        for job in self.client.batches.list(config={"page_size": 100}):
            if getattr(job, "display_name", None) != display_name:
                continue
            if _state_name(job.state) not in FAILED_STATES:
                return job.name
        return None

    def create_job(self, model: str, src_name: str, display_name: str) -> str:
        job = self.client.batches.create(model=model, src=src_name, config={"display_name": display_name})
        return job.name

    def find_or_create(self, display_name: str, model: str, src_name: str) -> str:
        existing = self.find_job(display_name)
        if existing:
            return existing
        return self.create_job(model, src_name, display_name)

    def state(self, job_name: str) -> str:
        return _state_name(self.client.batches.get(name=job_name).state)

    def progress(self, job_name: str) -> dict[str, Any]:
        """State plus request counters. Counters are None when the API does not report them."""
        job = self.client.batches.get(name=job_name)
        stats = getattr(job, "completion_stats", None)

        def count(attr: str) -> int | None:
            value = getattr(stats, attr, None) if stats is not None else None
            return int(value) if value is not None else None

        return {
            "state": _state_name(job.state),
            "succeeded": count("successful_count"),
            "failed": count("failed_count"),
            "incomplete": count("incomplete_count"),
        }

    def download_output(self, job_name: str) -> str:
        job = self.client.batches.get(name=job_name)
        data = self.client.files.download(file=job.dest.file_name)
        return data.decode("utf-8") if isinstance(data, bytes) else str(data)


def _state_name(state: Any) -> str:
    return str(getattr(state, "name", state))
