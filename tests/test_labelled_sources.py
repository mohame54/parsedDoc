"""Datasets that ship annotation text skip OCR and Stage 1; everything else keeps the normal vision route."""
import json
from pathlib import Path

import pytest
from PIL import Image

from corpus import datasets as corpus_datasets
from corpus.labels import attach_xfund, entities_from_funsd, entities_from_words, write_sidecar
from labeller.config import load_config
from labeller.pipeline import decide
from labeller.processors import PROCESSORS
from tests.test_download_progress import FakeStream, _jpeg
from tests.test_pipeline import CONFIG


def test_each_dataset_format_becomes_boxed_text():
    words = entities_from_words(["Total", "", "12.50"], [[10, 10, 50, 20], [0, 0, 1, 1], [60, 10, 100, 20]])
    assert [e["text"] for e in words] == ["Total", "12.50"]                    # empty text dropped
    assert entities_from_words(["x"], [[1, 2, 3]]) == []                        # a box needs four numbers
    funsd = entities_from_funsd(json.dumps([{"text": "Name:", "box": [1, 2, 3, 4], "label": "question"}]))
    assert funsd == [{"box": [1, 2, 3, 4], "text": "Name:"}]
    assert entities_from_funsd({"form": [{"text": "A", "box": [1, 2, 3, 4]}]})[0]["text"] == "A"   # original FUNSD
    assert entities_from_funsd("not json") == []


def test_xfund_split_file_is_split_into_one_sidecar_per_downloaded_image(tmp_path):
    split = {"documents": [
        {"img": {"fname": "fr_train_0.jpg"}, "document": [{"text": "Formulaire", "box": [1, 2, 30, 9]}]},
        {"img": {"fname": "fr_train_1.jpg"}, "document": [{"text": "Nom", "box": [1, 2, 30, 9]}]}]}
    ann = tmp_path / "fr.train.json"
    ann.write_text(json.dumps(split))
    images = [tmp_path / "fr_train_0.jpg", tmp_path / "fr_train_7.jpg"]       # _7 is not in the file
    assert attach_xfund(ann, images, "xfund_fr") == 1
    sidecar = json.loads((tmp_path / "fr_train_0.labels.json").read_text())
    assert sidecar["entities"][0]["text"] == "Formulaire" and sidecar["origin"] == "xfund_fr"
    assert not (tmp_path / "fr_train_7.labels.json").exists()


def test_huggingface_download_saves_the_rows_text_next_to_the_image(tmp_path, monkeypatch):
    import sys
    import types

    rows = [{"image": {"bytes": _jpeg(), "path": None}, "words": ["Hello", "World"],
             "bboxes": [[0, 0, 10, 5], [12, 0, 30, 5]]}]
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(
        load_dataset=lambda *a, **k: FakeStream(rows), Image=lambda decode: None))
    source = {"id": "wild", "dataset_id": "o/d",
              "labels": {"format": "words_boxes", "words_column": "words", "boxes_column": "bboxes"}}
    (image,) = corpus_datasets.download_huggingface(source, tmp_path, max_documents=5)
    sidecar = json.loads(image.with_name(image.stem + ".labels.json").read_text())
    assert [e["text"] for e in sidecar["entities"]] == ["Hello", "World"]


@pytest.fixture
def labelled_project(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "annotation.yaml").write_text(CONFIG, encoding="utf-8")
    source = tmp_path / "data" / "extracted" / "wild" / "images"
    source.mkdir(parents=True)
    for stem in ("a", "b", "c"):
        Image.new("RGB", (40, 40), "white").save(source / f"{stem}.png")
    write_sidecar(source / "a.png", [{"box": [60, 10, 120, 20], "text": "John Smith"},
                                     {"box": [10, 10, 50, 20], "text": "Name:"}], "wild")
    (source / "c.labels.json").write_text(json.dumps({"origin": "wild", "entities": []}))   # empty annotation
    cfg = load_config(config_dir / "annotation.yaml")
    docs = {f"wild:{s}": {"document_id": f"wild:{s}", "source_id": "wild", "processor": "sidecar_text",
                          "phi_risk": False, "original_path": f"extracted/wild/images/{s}.png", "page_paths": [],
                          "file_type": "png", "page_count": 1} for s in ("a", "b", "c")}
    return cfg, docs


def test_labelled_documents_skip_stage_1_and_the_rest_fall_back_to_the_vision_route(labelled_project):
    cfg, docs = labelled_project
    labelled = decide(docs["wild:a"], cfg)
    assert labelled["state"] == "needs_stage2" and labelled["origin"] == "native_text"
    assert labelled["source_text"] == ["Name: John Smith"]                  # reading order, one line
    for doc_id in ("wild:b", "wild:c"):                                      # no sidecar / empty sidecar
        fallback = decide(docs[doc_id], cfg)
        assert fallback["state"] == "needs_stage1" and fallback["origin"] == "gemini_stage1"


def test_sidecar_processor_is_registered_and_missing_labels_say_why(labelled_project):
    cfg, docs = labelled_project
    assert "sidecar_text" in PROCESSORS
    labels = PROCESSORS["sidecar_text"]().load(docs["wild:b"], cfg.data_root)
    assert labels.route == "vision" and labels.reason == "no_labels_sidecar"
    assert PROCESSORS["sidecar_text"]().load(docs["wild:c"], cfg.data_root).reason == "empty_annotation"
