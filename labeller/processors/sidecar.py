"""Datasets that ship their own text, saved by the downloader as `<stem>.labels.json` (see corpus/labels.py).

A document with such a file skips OCR and Stage 1: its text is rebuilt in reading order and goes straight to
Stage 2 (route native_text). A document without one, or with an empty one, takes the normal vision route
(OCR, Stage 1 correction, Stage 2), so a dataset can mix labelled and unlabelled documents.
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import DatasetProcessor, FileIndex, NativeLabels
from .funsd import layout_text

SUFFIX = ".labels.json"


class SidecarTextProcessor(DatasetProcessor):
    name = "sidecar_text"

    def load(self, doc: dict, root: Path) -> NativeLabels:
        sidecar = FileIndex(self.source_dir(doc, root), SUFFIX).get(self.stem(doc) + ".labels")
        if sidecar is None:
            return NativeLabels(route="vision", reason="no_labels_sidecar")
        try:
            entities = json.loads(sidecar.read_text(encoding="utf-8")).get("entities", [])
        except ValueError:
            return NativeLabels(route="vision", reason="unreadable_labels_sidecar")
        text = layout_text([(entity.get("box") or [], entity.get("text") or "") for entity in entities])
        if not text.strip():
            return NativeLabels(route="vision", reason="empty_annotation")
        return NativeLabels(route="native_text", text_pages=[text])
