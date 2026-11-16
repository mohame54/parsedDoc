"""Optional Gemini pass that writes one-line descriptions for the attributes in the exported records.

Opt-in and separate from the labelling pipeline: it costs API calls, and the training data works without
it (train/describe.py falls back to templates). One request per document type lists that type's distinct
(section, attribute, data type) triples, so the cost is a handful of small text requests, not one per document.

    labeller describe-submit --records data/export/train.jsonl     # uploads and submits one batch job
    labeller describe-collect                                      # writes data/descriptions/descriptions.json
    DatasetProcessor(descriptions="data/descriptions/descriptions.json")

Like the other Gemini code, the wire format follows the Batch API docs and has not been run against a live job.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from train.describe import key

from . import gemini

PROMPT = """You are writing documentation for a data schema used to extract information from "{document_type}" documents.
Below are the attributes, each as section.name (data type).

For every attribute write ONE short sentence (at most 20 words) saying what it holds on this kind of document.
- Describe the attribute only. Never include an example value, a number or a date.
- Do not repeat the attribute name as the whole answer, and do not mention data types or formats.
- Return every attribute exactly once, with the section and name spelled exactly as given.

Attributes:
{attributes}

Return JSON only, matching the response schema."""

SCHEMA = {
    "type": "object",
    "properties": {"descriptions": {"type": "array", "items": {
        "type": "object",
        "properties": {"section": {"type": "string"}, "name": {"type": "string"}, "description": {"type": "string"}},
        "required": ["section", "name", "description"]}}},
    "required": ["descriptions"],
}

MAX_DESCRIPTION_CHARS = 200
LINE_PREFIX = "describe:"


def read_records(paths: Iterable[Path]) -> Iterable[dict]:
    """Records from export JSONL files (the canonical record, or a line that wraps it)."""
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                yield obj.get("canonical", obj)


def collect_attributes(records: Iterable[dict]) -> dict[str, dict[tuple[str, str], str]]:
    """document type -> {(section, name): data_type}, first occurrence wins."""
    out: dict[str, dict[tuple[str, str], str]] = {}
    for record in records:
        doc_type = record.get("document_type")
        if not doc_type:
            continue
        attributes = out.setdefault(doc_type, {})
        for field in record.get("fields") or []:
            attributes.setdefault((field["section"], field["name"]), field.get("data_type", "string"))
    return out


def build_lines(attributes: dict[str, dict[tuple[str, str], str]], max_output_tokens: int = 4096) -> list[dict]:
    lines = []
    for doc_type, attrs in sorted(attributes.items()):
        listing = "\n".join(f"- {section}.{name} ({data_type})" for (section, name), data_type in sorted(attrs.items()))
        prompt = PROMPT.format(document_type=doc_type.replace("_", " "), attributes=listing)
        lines.append(gemini.build_request_line(LINE_PREFIX + doc_type, [gemini.text_part(prompt)], SCHEMA,
                                               max_output_tokens))
    return lines


def parse_results(output_text: str, attributes: dict[str, dict[tuple[str, str], str]]) -> dict[str, str]:
    """Table keyed by train.describe.key. Anything not asked for, empty, or too long is dropped, and the
    templates cover the gaps."""
    table: dict[str, str] = {}
    for line_key, (text, error) in gemini.parse_output_text(output_text).items():
        doc_type = line_key.removeprefix(LINE_PREFIX)
        if error or text is None or doc_type not in attributes:
            continue
        try:
            items = json.loads(text).get("descriptions", [])
        except (ValueError, AttributeError):
            continue
        for item in items:
            pair = (item.get("section"), item.get("name"))
            description = " ".join(str(item.get("description", "")).split())
            if pair in attributes[doc_type] and description and len(description) <= MAX_DESCRIPTION_CHARS:
                table[key(doc_type, *pair)] = description
    return table


def submit(model: str, api: Any, record_paths: list[Path], out_dir: Path) -> int:
    """Uploads the requests and creates (or reuses) the batch job. Returns the number of document types."""
    attributes = collect_attributes(read_records(record_paths))
    if not attributes:
        raise SystemExit("no document types with fields found in the given records")
    out_dir.mkdir(parents=True, exist_ok=True)
    requests = out_dir / "requests.jsonl"
    gemini.write_jsonl(requests, build_lines(attributes))
    job = api.find_or_create("describe-attributes", model, api.upload_jsonl(requests))
    (out_dir / "job.json").write_text(json.dumps({
        "job": job, "attributes": {t: [[s, n, d] for (s, n), d in a.items()] for t, a in attributes.items()},
    }), encoding="utf-8")
    return len(attributes)


def collect(api: Any, out_dir: Path) -> str:
    """Returns the job state; once it has succeeded, writes descriptions.json."""
    state_file = out_dir / "job.json"
    if not state_file.exists():
        raise SystemExit(f"{state_file} not found; run describe-submit first")
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    state = api.state(saved["job"])
    if state != gemini.SUCCEEDED:
        return state
    attributes = {t: {(s, n): d for s, n, d in rows} for t, rows in saved["attributes"].items()}
    table = parse_results(api.download_output(saved["job"]), attributes)
    (out_dir / "descriptions.json").write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"{state}: {len(table)} descriptions written"
