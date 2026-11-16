import json
from pathlib import Path

from labeller.processors import get_processor
from labeller.processors.funsd import layout_text


def test_layout_text_orders_by_line_then_column():
    items = [
        ([60, 40, 120, 50], "2020-01-02"),
        ([10, 40, 50, 50], "Date:"),
        ([60, 10, 120, 20], "John Smith"),
        ([10, 10, 50, 20], "Name:"),
    ]
    assert layout_text(items) == "Name: John Smith\nDate: 2020-01-02"


def test_funsd_builds_native_text_from_sidecar(tmp_path: Path):
    ann_dir = tmp_path / "extracted" / "funsd" / "dataset" / "training_data" / "annotations"
    ann_dir.mkdir(parents=True)
    (ann_dir / "00001.json").write_text(json.dumps({"form": [
        {"id": 0, "text": "Name:", "box": [10, 10, 50, 20]},
        {"id": 1, "text": "John Smith", "box": [60, 10, 120, 20]},
    ]}))
    doc = {"source_id": "funsd", "original_path": "extracted/funsd/dataset/training_data/images/00001.png"}
    labels = get_processor("funsd").load(doc, tmp_path)
    assert labels.route == "native_text"
    assert labels.text_pages == ["Name: John Smith"]


def test_funsd_without_sidecar_falls_back_to_vision(tmp_path: Path):
    doc = {"source_id": "funsd", "original_path": "extracted/funsd/x/00002.png"}
    assert get_processor("funsd").load(doc, tmp_path).route == "vision"


def test_sroie_builds_native_full_labels(tmp_path: Path):
    base = tmp_path / "extracted" / "sroie"
    (base / "box").mkdir(parents=True)
    (base / "entities").mkdir(parents=True)
    (base / "box" / "r1.txt").write_text(
        "10,10,90,10,90,20,10,20,ACME SDN BHD\n10,30,90,30,90,40,10,40,TOTAL 12.50\n"
    )
    (base / "entities" / "r1.json").write_text(json.dumps(
        {"company": "ACME SDN BHD", "date": "01/02/2019", "address": "", "total": "12.50"}
    ))
    doc = {"source_id": "sroie", "original_path": "extracted/sroie/images/r1.jpg"}
    labels = get_processor("sroie").load(doc, tmp_path)
    assert labels.route == "native_full"
    assert labels.text_pages == ["ACME SDN BHD\nTOTAL 12.50"]
    by_name = {f["name"]: f for f in labels.fields}
    assert by_name["address"]["status"] == "not_present"
    assert by_name["total"]["raw_value"] == "12.50"


def test_rxhandbd_word_crops_are_skipped(tmp_path: Path):
    labels = get_processor("rxhandbd").load({"source_id": "rx", "original_path": "a.png"}, tmp_path)
    assert labels.route == "skip"
