"""Local OCR (RapidOCR with PP-OCRv5 server models) producing the draft that Stage 1 corrects.

Each page becomes ordered text lines with stable ids (P<page>-L<n>). The result is cached in
annotations/<doc>/ocr.json, so a re-run never repeats OCR. The local gates run before any Gemini call.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

from PIL import Image

from .config import OcrConfig
from .gemini import prepare_image
from .store import Store, read_json, utc_now, write_json


class OcrEngine(Protocol):
    def recognise(self, image: Path) -> list[dict[str, Any]]:
        """Returns detections: [{"text": str, "score": float, "box": [[x, y], x4]}]."""


class RapidOcrEngine:
    def __init__(self, model_type: str = "server", ocr_version: str = "PP-OCRv5") -> None:
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import ModelType, OCRVersion

        version = OCRVersion(ocr_version)
        model = ModelType(model_type)
        self._engine = RapidOCR(params={
            "Det.ocr_version": version, "Det.model_type": model,
            "Rec.ocr_version": version, "Rec.model_type": model,
        })

    def recognise(self, image: Path) -> list[dict[str, Any]]:
        result = self._engine(str(image))
        texts = list(result.txts or [])
        scores = [float(s) for s in (result.scores or [])]
        boxes = result.boxes.tolist() if result.boxes is not None else []
        return [{"text": t, "score": s, "box": b} for t, s, b in zip(texts, scores, boxes)]


def page_paths(doc: dict) -> list[str]:
    return doc.get("page_paths") or [doc["original_path"]]


def group_lines(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Joins detections that share a row into one reading-order line."""
    items = []
    for det in detections:
        text = str(det["text"]).strip()
        if not text:
            continue
        xs = [point[0] for point in det["box"]]
        ys = [point[1] for point in det["box"]]
        items.append({"text": text, "score": float(det["score"]),
                      "x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)})
    items.sort(key=lambda item: (item["y0"], item["x0"]))

    rows: list[dict[str, Any]] = []
    for item in items:
        height = max(item["y1"] - item["y0"], 1.0)
        if rows and abs(item["y0"] - rows[-1]["y0"]) < 0.5 * rows[-1]["h"]:
            rows[-1]["items"].append(item)
        else:
            rows.append({"y0": item["y0"], "h": height, "items": [item]})

    lines = []
    for row in rows:
        ordered = sorted(row["items"], key=lambda item: item["x0"])
        lines.append({
            "text": " ".join(item["text"] for item in ordered),
            "score": round(sum(item["score"] for item in ordered) / len(ordered), 4),
            # Pixel box of the whole line, from the OCR's own detections (not from Gemini).
            "box": [round(min(i["x0"] for i in ordered), 1), round(min(i["y0"] for i in ordered), 1),
                    round(max(i["x1"] for i in ordered), 1), round(max(i["y1"] for i in ordered), 1)],
        })
    return lines


def ocr_gates(result: dict[str, Any], ocr_cfg: OcrConfig) -> str | None:
    """Local gates on the draft. Returns an exclusion reason, or None when the draft passes."""
    all_lines = [line for page in result["pages"] for line in page["lines"]]
    if not all_lines:
        return "ocr_no_text"
    for page in result["pages"]:
        chars = sum(len(line["text"]) for line in page["lines"])
        if chars < ocr_cfg.min_chars_per_page:
            return f"ocr_blank_page:{page['page']}"
    mean = sum(line["score"] for line in all_lines) / len(all_lines)
    if mean < ocr_cfg.min_mean_confidence:
        return f"ocr_low_confidence:{mean:.2f}"
    return None


def run_ocr(store: Store, engine: OcrEngine, doc: dict, ocr_cfg: OcrConfig,
            on_page: Callable[[], None] | None = None) -> str | None:
    """OCR once per document (cached in ocr.json), then apply the local gates. Returns an exclusion reason or None.

    on_page is called after each page is recognised, so the caller can show a page-level bar.
    """
    doc_id = doc["document_id"]
    path = store.ann_dir(doc_id) / "ocr.json"
    cached = read_json(path)
    if cached is None:
        pages = []
        scratch = store.ann_dir(doc_id) / "input"
        for index, rel in enumerate(page_paths(doc), start=1):
            image, _ = prepare_image(store.resolve(rel), scratch)
            lines = group_lines(engine.recognise(image))
            with Image.open(image) as picture:
                size = [picture.width, picture.height]
            pages.append({
                "page": index,
                "size": size,
                "lines": [{"id": f"P{index}-L{n}", "text": line["text"], "score": line["score"], "box": line["box"]}
                          for n, line in enumerate(lines, start=1)],
            })
            if on_page is not None:
                on_page()
        cached = {"engine": "rapidocr", "model_type": ocr_cfg.model_type, "ocr_version": ocr_cfg.ocr_version,
                  "pages": pages, "created_at": utc_now()}
        write_json(path, cached)
    return ocr_gates(cached, ocr_cfg)
