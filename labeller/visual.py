"""Local, free visual analysis: does a document need the page images at Stage 2, or is its text enough?

Signals (a document is "multimodal" when any one fires):
  PDF structure   embedded pictures that are not full-page scans, highlight/stamp/ink annotations,
                  checkbox and radio form fields.
  Page pixels     coloured areas (stamps, highlights, coloured rows or logos), light shaded blocks (table
                  row and header backgrounds), and large solid dark regions (logos, photos, halftone
                  headers). Blobs that touch the page edge are ignored: they are scanner or microfilm
                  borders, not content. Thin strokes (text, rules, box outlines) never count.
  Stage 1 flags   has_table / has_checkboxes / has_signature (configurable), added after Stage 1 for scans.

Pixel work is done on a 600-pixel-wide copy with Pillow only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .config import VisualConfig

WORK_WIDTH = 600
BORDER = 0.03                  # share of each side cropped off before analysis (scanner borders)
SCAN_PAGE_COVER = 0.85         # an embedded picture covering this much of a page is a scanned page
ANNOTATION_KINDS = {"highlight", "stamp", "ink", "square", "circle", "underline", "strikeout", "freetext"}


def _share(mask: Image.Image) -> float:
    return mask.histogram()[255] / max(mask.width * mask.height, 1)


def _odd(n: int) -> int:
    return n if n % 2 else n + 1


def _clear_edge_blobs(mask: Image.Image) -> Image.Image:
    """Removes blobs touching the image edge. Seeds are only the white pixels along the border."""
    mask = mask.copy()
    width, height = mask.size
    pixels = mask.load()
    edge = ([(x, 0) for x in range(width)] + [(x, height - 1) for x in range(width)]
            + [(0, y) for y in range(height)] + [(width - 1, y) for y in range(height)])
    for point in edge:
        if pixels[point] == 255:
            ImageDraw.floodfill(mask, point, 0)
    return mask


def page_signals(image: Image.Image, cfg: VisualConfig) -> dict[str, float]:
    """Shares of the page that are coloured, lightly shaded, or covered by large dark blobs."""
    image = image.convert("RGB")
    height = max(round(image.height * WORK_WIDTH / max(image.width, 1)), 1)
    image = image.resize((WORK_WIDTH, height), Image.BILINEAR)
    margin = round(BORDER * WORK_WIDTH)
    if height > 2 * margin + 10:
        image = image.crop((margin, margin, WORK_WIDTH - margin, height - margin))

    _, saturation, value = image.convert("HSV").split()
    coloured = ImageChops.multiply(
        saturation.point(lambda x: 255 if x > cfg.color_saturation else 0),
        value.point(lambda x: 255 if x > cfg.color_value else 0),
    )

    gray = image.convert("L")
    histogram, total, seen, background = gray.histogram(), gray.width * gray.height, 0, 255
    for level, count in enumerate(histogram):
        seen += count
        if seen >= total / 2:
            background = level     # the paper tone is the median grey level
            break

    thin = _odd(max(5, round(0.012 * WORK_WIDTH)))     # removes text strokes and rules
    dark = gray.point(lambda x: 255 if x < background - 60 else 0)
    blobs = _clear_edge_blobs(dark.filter(ImageFilter.MinFilter(thin)).filter(ImageFilter.MaxFilter(thin)))

    wide = _odd(round(0.04 * WORK_WIDTH))               # light shading must form wide solid blocks
    light = gray.point(lambda x: 255 if background - 55 <= x <= background - 10 else 0)
    shaded = _clear_edge_blobs(light.filter(ImageFilter.MinFilter(wide)).filter(ImageFilter.MaxFilter(wide)))

    return {"color": _share(coloured), "shaded": _share(shaded), "blob": _share(blobs)}


def pdf_structure(path: Path, cfg: VisualConfig, max_pages: int) -> dict[str, Any]:
    """Embedded pictures, annotations and checkbox fields in a PDF. Never raises: unreadable means no signal."""
    result: dict[str, Any] = {"image_area": 0.0, "annotations": 0, "checkboxes": 0, "scan_pages": 0}
    try:
        import fitz

        fitz.TOOLS.mupdf_display_errors(False)   # MuPDF prints a warning for every odd form widget
        with fitz.open(path) as pdf:
            for number, page in enumerate(pdf):
                if number >= max_pages:
                    break
                area = max(page.rect.width * page.rect.height, 1.0)
                page_images = 0.0
                for info in page.get_image_info():
                    covered = (fitz.Rect(info["bbox"]) & page.rect).get_area() / area
                    if covered >= SCAN_PAGE_COVER:
                        result["scan_pages"] += 1          # a scanned page: judged by its pixels instead
                    else:
                        page_images += covered
                result["image_area"] = max(result["image_area"], page_images)
                if cfg.pdf_annotations:
                    kinds = {(annot.type[1] or "").lower() for annot in (page.annots() or [])}
                    result["annotations"] += len(kinds & ANNOTATION_KINDS)
                if cfg.pdf_checkboxes:
                    boxes = (fitz.PDF_WIDGET_TYPE_CHECKBOX, fitz.PDF_WIDGET_TYPE_RADIOBUTTON)
                    result["checkboxes"] += sum(1 for widget in (page.widgets() or []) if widget.field_type in boxes)
    except Exception:
        pass
    return result


def analyze(doc: dict, root: Path, cfg: VisualConfig, max_pages: int) -> dict[str, Any]:
    """Visual routing for one document: {"multimodal", "reasons", "signals", "source"}. Writes nothing."""
    signals: dict[str, Any] = {"color": 0.0, "shaded": 0.0, "blob": 0.0}
    reasons: list[str] = []

    pages = doc.get("page_paths") or [doc["original_path"]]
    for relative in pages[:max_pages]:
        try:
            with Image.open(root / relative) as image:
                found = page_signals(image, cfg)
        except Exception:
            continue
        for name, share in found.items():
            signals[name] = max(signals[name], share)

    if doc.get("file_type") == "pdf":
        structure = pdf_structure(root / doc["original_path"], cfg, max_pages)
        signals.update({"image_area": structure["image_area"], "annotations": structure["annotations"],
                        "checkboxes": structure["checkboxes"]})
        if structure["image_area"] >= cfg.min_image_area:
            reasons.append(f"pdf_images:{structure['image_area']:.3f}")
        if structure["annotations"]:
            reasons.append(f"pdf_annotations:{structure['annotations']}")
        if structure["checkboxes"]:
            reasons.append(f"pdf_checkboxes:{structure['checkboxes']}")

    if signals["color"] >= cfg.min_color_area:
        reasons.append(f"color:{signals['color']:.3f}")
    if signals["shaded"] >= cfg.min_shaded_area:
        reasons.append(f"shading:{signals['shaded']:.3f}")
    if signals["blob"] >= cfg.min_blob_area:
        reasons.append(f"dark_regions:{signals['blob']:.3f}")

    return {"multimodal": bool(reasons), "reasons": reasons,
            "signals": {name: round(value, 4) if isinstance(value, float) else value
                        for name, value in signals.items()},
            "source": "local"}


def add_stage1_flags(visual: dict | None, stage1: dict, cfg: VisualConfig) -> dict:
    """Adds Stage 1's layout flags (for scans) to the local result. Flags only ever add reasons."""
    merged = dict(visual or {"multimodal": False, "reasons": [], "signals": {}, "source": "local"})
    reasons = list(merged.get("reasons") or [])
    for flag in cfg.stage1_flags:
        if any(page.get(flag) for page in stage1.get("pages", [])):
            reasons.append(f"stage1:{flag}")
    merged["reasons"] = reasons
    merged["multimodal"] = bool(reasons)
    merged["source"] = "local+stage1" if any(r.startswith("stage1:") for r in reasons) else merged.get("source", "local")
    return merged
