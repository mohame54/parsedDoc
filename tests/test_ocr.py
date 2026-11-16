from pathlib import Path

from PIL import Image

from labeller.config import OcrConfig, load_config
from labeller.ocr import group_lines, ocr_gates, run_ocr
from labeller.store import Store

OCR_CFG = OcrConfig(enabled=True, min_mean_confidence=0.6, min_chars_per_page=5)


def det(text, score, x0, y0, x1, y1):
    return {"text": text, "score": score, "box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


class FakeEngine:
    def __init__(self, detections):
        self.detections = detections
        self.calls = 0

    def recognise(self, image):
        self.calls += 1
        return list(self.detections)


def test_group_lines_orders_reading_order_and_joins_rows():
    lines = group_lines([
        det("2020-01-02", 0.9, 60, 40, 120, 50),
        det("Date:", 0.8, 10, 40, 50, 50),
        det("John Smith", 0.95, 60, 10, 120, 20),
        det("Name:", 0.9, 10, 10, 50, 20),
    ])
    assert [line["text"] for line in lines] == ["Name: John Smith", "Date: 2020-01-02"]
    assert lines[0]["score"] == 0.925


def test_gates_report_no_text_blank_page_and_low_confidence():
    empty = {"pages": [{"page": 1, "lines": []}]}
    assert ocr_gates(empty, OCR_CFG) == "ocr_no_text"

    blank_second = {"pages": [
        {"page": 1, "lines": [{"id": "P1-L1", "text": "Name: John", "score": 0.9}]},
        {"page": 2, "lines": [{"id": "P2-L1", "text": "x", "score": 0.9}]},
    ]}
    assert ocr_gates(blank_second, OCR_CFG) == "ocr_blank_page:2"

    low = {"pages": [{"page": 1, "lines": [{"id": "P1-L1", "text": "Name: John", "score": 0.3}]}]}
    assert ocr_gates(low, OCR_CFG).startswith("ocr_low_confidence")

    good = {"pages": [{"page": 1, "lines": [{"id": "P1-L1", "text": "Name: John", "score": 0.9}]}]}
    assert ocr_gates(good, OCR_CFG) is None


def test_run_ocr_caches_result_so_engine_runs_once(tmp_path: Path):
    config_path = tmp_path / "config" / "annotation.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        "data_root: ./data\nmax_pages: 4\nmin_text_chars_per_page: 80\nbatch_size: 5\nmax_attempts: 2\n"
        "stage1: {model: m, prompt_version: stage1_v2}\nstage2: {model: m, prompt_version: stage2_v1}\n"
        "ocr: {enabled: true}\nexport: {train: 0.8, val: 0.1, seed: 1}\n",
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    store = Store(cfg)
    image = tmp_path / "data" / "extracted" / "src" / "scan.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (40, 40), "white").save(image)
    doc = {"document_id": "src:1", "source_id": "src", "original_path": "extracted/src/scan.png",
           "page_paths": [], "file_type": "png"}

    engine = FakeEngine([det("Name: John Smith", 0.9, 10, 10, 120, 20)])
    assert run_ocr(store, engine, doc, OCR_CFG) is None
    assert run_ocr(store, engine, doc, OCR_CFG) is None
    assert engine.calls == 1

    saved = (store.ann_dir("src:1") / "ocr.json").read_text(encoding="utf-8")
    assert '"P1-L1"' in saved and "John Smith" in saved
