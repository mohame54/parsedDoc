"""SROIE receipts: OCR lines plus key-information JSON, both shipped with the dataset.

Assumed layout (ICDAR 2019 release): box/<id>.txt has lines "x1,y1,...,x4,y4,text" and
entities/<id>.json has company, date, address and total. Route native_full: no Gemini call.
The Hugging Face mirror in config.yaml currently saves images only; sidecars must be exported
before this processor can fire (until then the route falls back to vision).
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import DatasetProcessor, FileIndex, NativeLabels

ENTITY_TYPES = {"company": "string", "date": "date", "address": "string", "total": "number"}


class SroieProcessor(DatasetProcessor):
    name = "sroie"

    def load(self, doc: dict, root: Path) -> NativeLabels:
        base = self.source_dir(doc, root)
        stem = self.stem(doc)
        ocr = FileIndex(base, ".txt", parent_name="box").get(stem)
        entities_path = FileIndex(base, ".json", parent_name="entities").get(stem)
        if ocr is None or entities_path is None:
            return NativeLabels(route="vision", reason="no_sroie_sidecars")

        lines = []
        for raw in ocr.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = raw.split(",", 8)
            if len(parts) == 9:
                lines.append(parts[8])
        entities = json.loads(entities_path.read_text(encoding="utf-8-sig"))

        fields = []
        for name, data_type in ENTITY_TYPES.items():
            value = str(entities.get(name, "") or "").strip()
            fields.append({
                "name": name,
                "section": "receipt",
                "data_type": data_type,
                "raw_value": value,
                "normalized_value": value,
                "status": "extracted" if value else "not_present",
                "confidence": "high",
            })
        return NativeLabels(route="native_full", text_pages=["\n".join(lines)], fields=fields, document_type="receipt")
