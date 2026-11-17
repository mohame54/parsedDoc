"""Stage 2 v3: multimodal annotation (fields, tables, visual elements, layout, relationships)."""
import dataclasses
import json

from labeller.config import StageConfig, load_config
from labeller.evidence import attach_boxes, build_line_index
from labeller.pipeline import collect, ocr_step, prepare, prompt_version_for, submit
from labeller.validate import validate_stage2_v3
from tests.test_pipeline import FakeApi, gemini_text
from tests.test_pipeline_ocr import CONFIG_OCR, FakeOcr, project, stage1_v2_response  # noqa: F401  (fixture)

SOURCE = "P1-L1: Name: John Smith\nP1-L2: Date: 2020-01-02"
SOURCE_TEXT = "Name: John Smith\nDate: 2020-01-02"
LINE_IDS = {"P1-L1", "P1-L2"}


def v3(**overrides):
    obj = {
        "document_type": "form",
        "fields": [
            {"name": "patient_name", "section": "patient", "data_type": "string", "raw_value": "John Smith",
             "normalized_value": "John Smith", "status": "extracted", "confidence": "high", "description": "What this field holds.", "page": 1,
             "evidence": ["P1-L1"]},
            {"name": "visit_date", "section": "visit", "data_type": "date", "raw_value": "2020-01-02",
             "normalized_value": "2020-01-02", "status": "extracted", "confidence": "high", "description": "What this field holds.", "page": 1,
             "evidence": ["P1-L2"]},
        ],
        "tables": [{
            "table_id": "table_1", "page": 1, "headers": ["Name", "Date"], "evidence": ["P1-L1", "P1-L2"],
            "rows": [{
                "row_id": "t1_r1", "row_type": "data",
                "cells": [{"column": 1, "text": "John Smith", "row_span": 1, "col_span": 1},
                          {"column": 2, "text": "2020-01-02", "row_span": 1, "col_span": 1}],
                "visual": {"background_color": "light_yellow", "text_color": "black", "highlighted": True,
                           "bold": False},
            }],
        }],
        "visual_elements": [{"element_id": "visual_1", "type": "highlight", "page": 1,
                             "description": "yellow row", "color": "light_yellow", "state": "present",
                             "evidence": ["P1-L1"]}],
        "layout": [{"layout_id": "layout_1", "type": "title", "name": "Patient form", "page": 1,
                    "evidence": ["P1-L1"]}],
        "relationships": [{"from_id": "visual_1", "to_id": "t1_r1", "type": "highlights"}],
        "figures": [],
    }
    obj.update(overrides)
    return obj


def check(obj, line_ids=LINE_IDS, page_count=1, gap=0):
    return validate_stage2_v3(obj, SOURCE_TEXT, line_ids, page_count, max_value_gap=gap)


def test_a_consistent_answer_passes():
    assert check(v3()) == []


def test_missing_components_are_a_schema_error():
    obj = v3()
    del obj["tables"]
    assert check(obj)[0].startswith("schema:")


def test_table_cell_text_must_come_from_the_source():
    obj = v3()
    obj["tables"][0]["rows"][0]["cells"][0]["text"] = "Jane Doe"
    assert check(obj) == ["table_not_in_source:table_1.t1_r1"]


def test_ids_must_be_unique_across_components():
    obj = v3()
    obj["layout"][0]["layout_id"] = "visual_1"
    assert "duplicate_id:visual_1" in check(obj)


def test_relationships_must_point_at_existing_ids():
    obj = v3(relationships=[{"from_id": "visual_1", "to_id": "nope", "type": "highlights"}])
    assert check(obj) == ["unknown_reference:nope"]


def test_evidence_must_cite_real_line_ids_when_ids_were_given():
    obj = v3()
    obj["fields"][0]["evidence"] = ["P9-L9"]
    assert check(obj) == ["unknown_evidence:P9-L9"]
    assert check(obj, line_ids=None) == []   # no ids were shown to Gemini, so evidence is not judged


def test_page_numbers_and_spans_are_checked():
    obj = v3()
    obj["layout"][0]["page"] = 3
    obj["tables"][0]["rows"][0]["cells"][0]["col_span"] = 0
    reasons = check(obj)
    assert "bad_page:3" in reasons and "bad_cell:t1_r1" in reasons


def test_boxes_come_from_the_ocr_and_are_normalised():
    ocr_pages = [{"page": 1, "size": [200, 100], "lines": [
        {"id": "P1-L1", "text": "a", "score": 0.9, "box": [20, 10, 100, 30]},
        {"id": "P1-L2", "text": "b", "score": 0.9},                      # no box measured
    ]}]
    obj = v3()
    attach_boxes(obj, build_line_index(ocr_pages))
    name_box = obj["fields"][0]["boxes"][0]
    assert name_box == {"line": "P1-L1", "page": 1, "box_px": [20, 10, 100, 30], "box": [0.1, 0.1, 0.5, 0.3]}
    assert obj["fields"][1]["boxes"] == []                              # cited a line with no box
    assert [b["line"] for b in obj["tables"][0]["boxes"]] == ["P1-L1"]


def test_stage2_is_text_only_unless_images_are_switched_on(tmp_path):
    config = tmp_path / "config" / "annotation.yaml"
    config.parent.mkdir()
    config.write_text(CONFIG_OCR, encoding="utf-8")
    cfg = load_config(config)
    assert cfg.stage2.send_images is False
    assert cfg.stage1.send_images is True
    assert prompt_version_for(2, cfg, {"origin": "gemini_stage1"}) == "stage2_v5"
    with_images = dataclasses.replace(cfg, stage2=dataclasses.replace(cfg.stage2, send_images=True))
    assert prompt_version_for(2, with_images, {"origin": "gemini_stage1"}) == "stage2_v7"
    assert prompt_version_for(2, with_images, {"origin": "native_text"}) == "stage2_v7"


def responder_v3(request):
    props = request["generation_config"]["response_schema"]["properties"]
    if "corrections" in props:
        return stage1_v2_response(request)
    assert "tables" in props, "Stage 2 should have been asked for the v3 schema"
    return gemini_text(v3())


def test_pipeline_sends_images_and_stores_components_with_ocr_boxes(project):  # noqa: F811
    cfg, store, docs = project
    cfg = dataclasses.replace(cfg, stage2=StageConfig(model="test-flash", prompt_version="stage2_v1",
                                                     max_output_tokens=4096, send_images=True))
    prepare(cfg, store, docs)
    ocr_step(cfg, store, FakeOcr, docs, out=lambda _: None)
    api = FakeApi(responder_v3)
    submit(cfg, store, api, 1, docs, out=lambda _: None)
    collect(cfg, store, api, 1, docs, out=lambda _: None)
    submit(cfg, store, api, 2, docs, out=lambda _: None)

    stage2_request = api.inputs[sorted(api.inputs)[-1]]
    assert ":stage2:stage2_v7" in stage2_request
    assert '"file_data"' in stage2_request                      # the page image goes to Stage 2 as well
    assert "P1-L2: Date: 2020-01-02" in stage2_request          # corrected lines carry their ids

    assert collect(cfg, store, api, 2, docs, out=lambda _: None)["accepted"] == 1
    annotation = json.loads((store.ann_dir("imgsrc:aaa") / "annotation.json").read_text())
    assert annotation["provenance"]["stage2"]["prompt_version"] == "stage2_v7"
    assert [key for key in ("tables", "visual_elements", "layout", "relationships") if annotation[key]] == [
        "tables", "visual_elements", "layout", "relationships"]
    name_box = annotation["fields"][0]["boxes"][0]
    assert name_box["line"] == "P1-L1" and name_box["box_px"] == [10.0, 10.0, 120.0, 20.0]
    assert annotation["tables"][0]["rows"][0]["visual"]["highlighted"] is True


def test_default_stage2_request_has_no_images(project):  # noqa: F811
    cfg, store, docs = project
    prepare(cfg, store, docs)
    ocr_step(cfg, store, FakeOcr, docs, out=lambda _: None)

    def responder(request):
        props = request["generation_config"]["response_schema"]["properties"]
        if "corrections" in props:
            return stage1_v2_response(request)
        return gemini_text({"document_type": "form", "fields": [
            {"name": "n", "section": "s", "data_type": "string", "raw_value": "John Smith",
             "normalized_value": "", "status": "extracted", "confidence": "high", "description": "What this field holds."}]})

    api = FakeApi(responder)
    submit(cfg, store, api, 1, docs, out=lambda _: None)
    collect(cfg, store, api, 1, docs, out=lambda _: None)
    submit(cfg, store, api, 2, docs, out=lambda _: None)
    request = api.inputs[sorted(api.inputs)[-1]]
    assert '"file_data"' not in request
    assert ":stage2:stage2_v5" in request
    assert "tables" not in json.loads(request.splitlines()[0])["request"]["generation_config"]["response_schema"]["properties"]
