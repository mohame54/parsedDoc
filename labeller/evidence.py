"""Spatial grounding from the OCR's own coordinates.

Stage 2 v3 components cite OCR line ids as evidence (for example "P1-L3"). This module turns those ids
into boxes using the boxes the OCR engine measured. Gemini's own coordinates are never used: a language
model's boxes are not pixel-accurate, and the OCR's are.
"""
from __future__ import annotations

COMPONENTS_WITH_EVIDENCE = ("fields", "tables", "visual_elements", "layout")


def build_line_index(ocr_pages: list[dict]) -> dict[str, dict]:
    """OCR line id -> {"page", "box_px", "box"}. box is normalised to 0-1 when the page size is known."""
    index: dict[str, dict] = {}
    for page in ocr_pages:
        size = page.get("size")
        for line in page["lines"]:
            box = line.get("box")
            if not box:
                continue
            entry = {"page": page["page"], "box_px": box}
            if size and size[0] and size[1]:
                entry["box"] = [round(box[0] / size[0], 4), round(box[1] / size[1], 4),
                                round(box[2] / size[0], 4), round(box[3] / size[1], 4)]
            index[line["id"]] = entry
    return index


def attach_boxes(annotation: dict, index: dict[str, dict]) -> None:
    """Adds "boxes" to every component that has evidence. Ids with no measured box (for example lines Gemini
    added that OCR missed) are skipped, so a component may have fewer boxes than evidence ids."""
    for key in COMPONENTS_WITH_EVIDENCE:
        for item in annotation.get(key, []):
            item["boxes"] = [{"line": line_id, **index[line_id]}
                             for line_id in item.get("evidence", []) if line_id in index]
