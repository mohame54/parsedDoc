"""Dataset processors. Each dataset gets one class that decides the cheapest correct route for its documents.

Routes:
  native_full  labels and text already exist and cover the whole document. No Gemini call.
  native_text  text exists (from the dataset or a text layer). Gemini Stage 2 only, text in, no image.
  vision       nothing usable. Gemini Stage 1 reads the images, then Stage 2 runs on the result.
  skip         not a whole-document sample; excluded with the given reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class NativeLabels:
    route: str
    text_pages: list[str] | None = None
    fields: list[dict] | None = None
    document_type: str | None = None
    reason: str | None = None


_INDEX_CACHE: dict[tuple[str, str, str | None], dict[str, Path]] = {}


class FileIndex:
    """Stem -> path lookup over a directory tree, built once per (dir, suffix, parent name)."""

    def __init__(self, base: Path, suffix: str, parent_name: str | None = None) -> None:
        key = (str(base), suffix, parent_name)
        if key not in _INDEX_CACHE:
            index: dict[str, Path] = {}
            if base.exists():
                for path in sorted(base.rglob(f"*{suffix}")):
                    if parent_name is not None and path.parent.name != parent_name:
                        continue
                    index.setdefault(path.stem, path)
            _INDEX_CACHE[key] = index
        self._index = _INDEX_CACHE[key]

    def get(self, stem: str) -> Path | None:
        return self._index.get(stem)


class DatasetProcessor:
    name = "generic"

    def load(self, doc: dict, root: Path) -> NativeLabels:
        return NativeLabels(route="vision")

    @staticmethod
    def source_dir(doc: dict, root: Path) -> Path:
        return root / "extracted" / doc["source_id"]

    @staticmethod
    def stem(doc: dict) -> str:
        return Path(doc["original_path"]).stem
