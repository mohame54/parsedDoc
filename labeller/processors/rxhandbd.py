"""RxHandBD: word-level crops from handwritten prescriptions, not whole documents.

Each sample is one word, so it cannot train whole-document extraction. It is skipped and
recorded as excluded, which keeps it out of the export without deleting it from the corpus.
"""
from __future__ import annotations

from pathlib import Path

from .base import DatasetProcessor, NativeLabels


class RxHandBDProcessor(DatasetProcessor):
    name = "rxhandbd"

    def load(self, doc: dict, root: Path) -> NativeLabels:
        return NativeLabels(route="skip", reason="word_level_crop_not_whole_document")
