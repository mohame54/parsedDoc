"""Bilingual handwritten prescriptions with a transcription per image.

ASSUMPTION, NOT VERIFIED: the transcription is a <image-stem>.txt next to the image. If the
downloaded release stores transcriptions differently, change the lookup here.
With a transcription present the route is native_text (Stage 2 only); otherwise vision.
"""
from __future__ import annotations

from pathlib import Path

from .base import DatasetProcessor, FileIndex, NativeLabels


class BilingualRxProcessor(DatasetProcessor):
    name = "bilingual_rx"

    def load(self, doc: dict, root: Path) -> NativeLabels:
        transcript = FileIndex(self.source_dir(doc, root), ".txt").get(self.stem(doc))
        if transcript is None:
            return NativeLabels(route="vision", reason="no_transcription_sidecar")
        text = transcript.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return NativeLabels(route="vision", reason="empty_transcription")
        return NativeLabels(route="native_text", text_pages=[text], document_type="prescription")
