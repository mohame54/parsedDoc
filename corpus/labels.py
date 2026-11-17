"""Annotation text that ships with a dataset, saved next to each image as `<stem>.labels.json`.

    {"origin": "<source id>", "entities": [{"box": [x0, y0, x1, y1], "text": "..."}, ...]}

Every dataset stores its text differently (word lists with boxes, FUNSD-style entities, one split-wide JSON).
The functions here turn each into the same list of boxed text, so the labeller needs a single reader
(labeller/processors/sidecar.py) and documents that have such a file skip OCR and Stage 1.
Documents without one are untouched and take the normal vision route.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

SUFFIX = ".labels.json"
QA_SUFFIX = ".qa.json"


def _entity(box: Any, text: Any) -> dict | None:
    """One boxed text, or None when the box is not four numbers or the text is empty."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        values = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if len(values) != 4:
        return None
    return {"box": [int(v) if v == int(v) else v for v in values], "text": text.strip()}


def entities_from_words(words: list[str], boxes: list[list[float]]) -> list[dict]:
    """Word list plus one box per word (WildReceipt, token-level datasets)."""
    return [e for e in (_entity(b, w) for w, b in zip(words or [], boxes or [])) if e]


def entities_from_funsd(entities: list[dict] | str) -> list[dict]:
    """FUNSD-style entities, as a list or a JSON string (the CMS-1500 `funsd_json` column)."""
    if isinstance(entities, str):
        try:
            entities = json.loads(entities)
        except ValueError:
            return []
    if isinstance(entities, dict):                 # the original FUNSD files nest them under "form"
        entities = entities.get("form", [])
    return [e for e in (_entity(item.get("box"), item.get("text")) for item in entities or []
                        if isinstance(item, dict)) if e]


def entities_from_row(spec: dict[str, Any], row: dict[str, Any]) -> list[dict]:
    """Entities for one Hugging Face row, as described by the source's `labels:` block.

        labels: {format: words_boxes, words_column: words, boxes_column: bboxes}
        labels: {format: funsd_json, column: funsd_json}
    """
    kind = spec.get("format")
    if kind == "words_boxes":
        return entities_from_words(row.get(spec.get("words_column", "words")),
                                   row.get(spec.get("boxes_column", "bboxes")))
    if kind == "funsd_json":
        return entities_from_funsd(row.get(spec.get("column", "funsd_json")))
    raise ValueError(f"unknown labels format {kind!r}; known: words_boxes, funsd_json")


def write_sidecar(image: Path, entities: list[dict], origin: str) -> Path | None:
    """Writes `<image stem>.labels.json`; nothing is written for an empty list."""
    if not entities:
        return None
    target = image.with_name(image.stem + SUFFIX)
    target.write_text(json.dumps({"origin": origin, "entities": entities}, ensure_ascii=False), encoding="utf-8")
    return target


def attach_xfund(annotations: Path, images: list[Path], origin: str) -> int:
    """Splits an XFUND split file (`{"documents": [{"img": {"fname"}, "document": [...]}]}`) into sidecars for
    the images that were downloaded. Returns how many images received one."""
    documents = json.loads(annotations.read_text(encoding="utf-8")).get("documents", [])
    by_name = {Path(d["img"]["fname"]).name: d.get("document") or [] for d in documents if d.get("img")}
    attached = 0
    for image in images:
        entities = entities_from_funsd(by_name.get(image.name, []))
        if write_sidecar(image, entities, origin):
            attached += 1
    return attached


def parse_answers(value: Any) -> list[str]:
    """Answers as a list of strings. ChartQA stores them as the text of a Python list, e.g. "['2014']"."""
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value.strip()]
        if isinstance(parsed, (list, tuple)):
            return [str(v).strip() for v in parsed if str(v).strip()]
        return [str(parsed).strip()]
    return []


def write_qa_sidecar(image: Path, qa: list[dict], origin: str) -> Path | None:
    """Writes `<image stem>.qa.json` with every question asked about the image; nothing for an empty list."""
    if not qa:
        return None
    target = image.with_name(image.stem + QA_SUFFIX)
    target.write_text(json.dumps({"origin": origin, "qa": qa}, ensure_ascii=False), encoding="utf-8")
    return target
