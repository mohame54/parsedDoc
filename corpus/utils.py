from __future__ import annotations


import hashlib
import zipfile
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DOC_EXTS = IMAGE_EXTS | {".pdf"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def bytes_to_gb(n: int) -> float:
    return n / (1024 ** 3)


def safe_extract_zip(archive: Path, destination: Path, progress: bool = True) -> None:
    """Extract without allowing absolute paths or ../ traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        for member in members:      # validate every name before writing anything
            target = (destination / member.filename).resolve()
            if target != base and base not in target.parents:
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")
        total = sum(m.file_size for m in members)
        with tqdm(total=total, desc=f"extract {archive.name}", unit="B", unit_scale=True, leave=False,
                  disable=not progress) as bar:
            for member in members:
                zf.extract(member, destination)
                bar.update(member.file_size)


class Budget:
    """Disk budget. The tree is scanned once; later checks are O(1) over the
    cached total plus bytes reported through ``check`` / ``rescan``."""

    def __init__(self, root: Path, max_total_gb: float):
        self.root = root
        self.max_bytes = int(max_total_gb * 1024**3)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cached: int | None = None

    def _scan(self) -> int:
        total = 0
        for p in self.root.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

    def rescan(self) -> int:
        """Force a full scan (call after bulk extraction)."""
        self._cached = self._scan()
        return self._cached

    def used(self) -> int:
        if self._cached is None:
            self._cached = self._scan()
        return self._cached

    def check(self, incoming: int = 0) -> None:
        if self.used() + incoming > self.max_bytes:
            raise RuntimeError(
                f"Storage budget exceeded: {bytes_to_gb(self.used() + incoming):.2f} GB "
                f"> configured {bytes_to_gb(self.max_bytes):.2f} GB"
            )
        # Bytes announced here are about to be written; keep the running total current.
        self._cached = self.used() + incoming


def download_stream(url: str, destination: Path, budget: Budget, max_bytes: int) -> None:
    if not url:
        raise ValueError("Missing download URL")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": "real-document-corpus-builder/0.1"}
    with requests.get(url, stream=True, timeout=(20, 120), headers=headers) as response:
        response.raise_for_status()
        declared = int(response.headers.get("content-length", 0))
        if declared and declared > max_bytes:
            raise RuntimeError(f"Download is {bytes_to_gb(declared):.2f} GB, above source limit")
        budget.check(declared)
        written = 0
        with tmp.open("wb") as f, tqdm(
            total=declared or None, unit="B", unit_scale=True, desc=destination.name
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                written += len(chunk)
                if written > max_bytes:
                    raise RuntimeError("Source download-size limit exceeded")
                budget.check(len(chunk))
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(destination)



def classify(path: Path, source_category: str, rules: dict[str, list[str]],
             allowed: set[str]) -> tuple[str, str]:
    text = str(path).lower().replace("-", "_")
    for category, keywords in rules.items():
        if any(k.lower() in text for k in keywords):
            return (category if category in allowed else "other", "filename_rule")
    return (source_category if source_category in allowed else "other", "source_default")


UNFILLED_VALUES = {"", "off", "0", "false", "no"}


def pdf_form_fill(pdf_path: Path) -> dict | None:
    """Counts fillable widgets and how many carry a value. None if the PDF has no widgets or cannot be read.

    A checkbox/text value of "", None, "Off", "0" or "false" counts as unfilled.
    """
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required: pip install pymupdf") from exc
    try:
        with fitz.open(pdf_path) as doc:
            widgets = [w for page in doc for w in (page.widgets() or [])]
    except Exception:
        return None
    if not widgets:
        return {"widgets": 0, "filled": 0}
    filled = sum(1 for w in widgets if str(w.field_value or "").strip().lower() not in UNFILLED_VALUES)
    return {"widgets": len(widgets), "filled": filled}


def is_blank_form(form_fill: dict, project: dict) -> bool:
    """True when the project rejects blank forms and this PDF's widgets are mostly unfilled."""
    if not project.get("reject_blank_forms", False):
        return False
    widgets = int(form_fill.get("widgets", 0))
    if widgets < int(project.get("min_widgets", 5)):
        return False
    return form_fill.get("filled", 0) / widgets < float(project.get("max_fill_ratio", 0.1))


def pdf_page_count(pdf_path: Path) -> int | None:
    """Real page count, independent of how many pages are rendered. None if unreadable."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required: pip install pymupdf") from exc
    try:
        with fitz.open(pdf_path) as doc:
            return len(doc)
    except Exception:
        return None


def render_pdf(pdf_path: Path, page_dir: Path, dpi: int, max_pages: int) -> list[Path]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF rendering: pip install pymupdf") from exc
    page_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    with fitz.open(pdf_path) as doc:
        scale = dpi / 72
        matrix = fitz.Matrix(scale, scale)
        for page_index in range(min(len(doc), max_pages)):
            pix = doc[page_index].get_pixmap(matrix=matrix, alpha=False)
            target = page_dir / f"page_{page_index + 1:04d}.png"
            pix.save(target)
            rendered.append(target)
    return rendered


def is_archive_junk(path: Path) -> bool:
    """macOS zip metadata: the __MACOSX folder and AppleDouble '._' files are not documents."""
    return "__MACOSX" in path.parts or path.name.startswith("._")


def discover_files(folder: Path, max_files: int) -> list[Path]:
    found = [p for p in folder.rglob("*")
             if p.is_file() and p.suffix.lower() in DOC_EXTS and not is_archive_junk(p)]
    found.sort()
    return found[:max_files]
