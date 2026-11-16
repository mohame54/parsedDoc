"""Fallback processor: vision route for any dataset without a dedicated class.

Also provides the PDF text-layer check: a born-digital PDF is labelled from its text instead of
paying for a vision read. The pipeline calls this for every vision-route PDF.
"""
from __future__ import annotations

from pathlib import Path

from .base import DatasetProcessor, NativeLabels


def pdf_text_pages(pdf: Path, min_chars_per_page: int, max_pages: int) -> list[str] | None:
    """Returns per-page text if every page has a usable text layer, else None."""
    try:
        import fitz
    except ImportError:
        return None
    try:
        with fitz.open(pdf) as doc:
            if len(doc) == 0 or len(doc) > max_pages:
                return None
            texts = [doc[i].get_text("text") for i in range(len(doc))]
    except Exception:
        return None
    if all(len(text.strip()) >= min_chars_per_page for text in texts):
        return texts
    return None


class GenericProcessor(DatasetProcessor):
    name = "generic"

    def load(self, doc: dict, root: Path) -> NativeLabels:
        return NativeLabels(route="vision")
