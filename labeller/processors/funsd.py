"""FUNSD: per-form JSON with entities (box, text, label, words, linking).

Native text is rebuilt in reading order. FUNSD has no per-document schema, so the route is
native_text: Stage 2 builds the schema from the text, with no image read.
Format assumed from the public FUNSD release (dataset/training_data/annotations/<id>.json,
key "form"). Verify on the real archive before the pilot.
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import DatasetProcessor, FileIndex, NativeLabels


def layout_text(items: list[tuple[list[float], str]]) -> str:
    """Groups boxes into lines by vertical overlap, then orders each line left to right."""
    cleaned = [(box, text.strip()) for box, text in items if text and text.strip() and len(box) == 4]
    cleaned.sort(key=lambda item: (item[0][1], item[0][0]))
    lines: list[dict] = []
    for box, text in cleaned:
        height = max(box[3] - box[1], 1.0)
        if lines and abs(box[1] - lines[-1]["y"]) < 0.5 * lines[-1]["h"]:
            lines[-1]["parts"].append((box[0], text))
        else:
            lines.append({"y": box[1], "h": height, "parts": [(box[0], text)]})
    return "\n".join(" ".join(text for _, text in sorted(line["parts"])) for line in lines)


class FunsdProcessor(DatasetProcessor):
    name = "funsd"

    def load(self, doc: dict, root: Path) -> NativeLabels:
        index = FileIndex(self.source_dir(doc, root), ".json", parent_name="annotations")
        sidecar = index.get(self.stem(doc))
        if sidecar is None:
            return NativeLabels(route="vision", reason="no_annotation_sidecar")
        entities = json.loads(sidecar.read_text(encoding="utf-8")).get("form", [])
        text = layout_text([(entity.get("box") or [], entity.get("text") or "") for entity in entities])
        if not text.strip():
            return NativeLabels(route="vision", reason="empty_annotation")
        return NativeLabels(route="native_text", text_pages=[text], document_type="form")
