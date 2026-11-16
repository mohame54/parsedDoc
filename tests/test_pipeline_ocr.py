"""OCR -> Stage 1 (v2, corrections only) -> Stage 2, against the fake Batch API and fake OCR engine."""
import json
from pathlib import Path

import pytest
from PIL import Image

from labeller.config import load_config
from labeller.pipeline import collect, load_manifest, ocr_step, prepare, submit
from labeller.store import Store
from tests.test_pipeline import FakeApi, gemini_text, stage2_response

CONFIG_OCR = """
data_root: ./data
max_pages: 4
min_text_chars_per_page: 80
batch_size: 500
max_attempts: 2
stage1: {model: test-flash-lite, prompt_version: stage1_v2, max_output_tokens: 4096}
stage2: {model: test-flash, prompt_version: stage2_v1, max_output_tokens: 4096}
ocr: {enabled: true, min_mean_confidence: 0.6, min_chars_per_page: 5}
gates: {max_illegible_regions: 0, max_change_ratio: 0.3, max_value_gap: 0, exclude_blank_forms: true}
phi_allowed_sources: []
export: {train: 0.8, val: 0.1, seed: 42}
"""


class FakeOcr:
    """Returns the same two draft lines for every page. Line 2 has a common OCR error (O for 0)."""

    def __init__(self):
        self.calls = 0

    def recognise(self, image):
        self.calls += 1
        return [
            {"text": "Name: John Smith", "score": 0.95, "box": [[10, 10], [120, 10], [120, 20], [10, 20]]},
            {"text": "Date: 2O2O-01-02", "score": 0.85, "box": [[10, 40], [120, 40], [120, 50], [10, 50]]},
        ]


def stage1_v2_response(_request):
    return gemini_text({
        "usable": True, "document_type": "form", "language": "en", "fill_level": "filled",
        "illegible_regions": 0,
        "pages": [{"page": 1, "legibility": "good", "has_handwriting": False, "has_checkboxes": False,
                   "has_signature": False, "has_table": False}],
        "corrections": [{"line": "P1-L2", "text": "Date: 2020-01-02"}],
        "missed": [],
    })


def responder(request):
    props = request["generation_config"]["response_schema"]["properties"]
    return stage1_v2_response(request) if "corrections" in props else stage2_response(request)


@pytest.fixture
def project(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "annotation.yaml"
    config_path.write_text(CONFIG_OCR, encoding="utf-8")
    img_dir = tmp_path / "data" / "extracted" / "imgsrc"
    img_dir.mkdir(parents=True)
    Image.new("RGB", (40, 40), "white").save(img_dir / "scan.png")
    row = {"document_id": "imgsrc:aaa", "source_id": "imgsrc", "processor": "generic", "phi_risk": False,
           "original_path": "extracted/imgsrc/scan.png", "page_paths": [], "file_type": "png",
           "page_count": 1, "license_review_status": "UNVERIFIED", "license_declared": "test",
           "category": "form", "source_url": None}
    (tmp_path / "data" / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    cfg = load_config(config_path)
    return cfg, Store(cfg), load_manifest(cfg)


def test_scan_goes_ocr_then_stage1_corrections_then_stage2(project):
    cfg, store, docs = project
    report = prepare(cfg, store, docs)
    assert report["counts"] == {"needs_ocr": 1}

    engine = FakeOcr()
    result = ocr_step(cfg, store, lambda: engine, docs, out=lambda _: None)
    assert result["ocr_ok"] == 1 and engine.calls == 1
    assert store.status("imgsrc:aaa")["state"] == "needs_stage1"

    api = FakeApi(responder)
    submit(cfg, store, api, 1, docs, out=lambda _: None)
    request_text = api.inputs[next(iter(api.inputs))]
    assert "OCR DRAFT" in request_text and "P1-L2: Date: 2O2O-01-02" in request_text
    assert collect(cfg, store, api, 1, docs, out=lambda _: None)["stage1_ok"] == 1

    source = json.loads((store.ann_dir("imgsrc:aaa") / "source_text.json").read_text())
    assert "Date: 2020-01-02" in source["pages"][0]
    assert source["correction_ratio"] == 0.5          # one of two draft lines was touched
    assert 0 < source["change_ratio"] < 0.3           # but only a couple of characters really changed

    submit(cfg, store, api, 2, docs, out=lambda _: None)
    stage2_text = api.inputs[sorted(api.inputs)[-1]]
    assert "came from OCR" in stage2_text
    assert ":stage2:stage2_v5" in stage2_text  # the request key records the prompt version used
    assert collect(cfg, store, api, 2, docs, out=lambda _: None)["accepted"] == 1
    assert store.status("imgsrc:aaa")["state"] == "accepted"


def test_prompt_versions_are_selected_per_document():
    from labeller import prompts
    from labeller.pipeline import prompt_version_for

    assert "OCR" in prompts.get_prompt("stage1_v3")
    assert "stage1_v3" in prompts.OCR_STAGE1_PROMPTS
    assert "came from OCR" in prompts.get_prompt("stage2_v2")
    assert "came from OCR" not in prompts.get_prompt("stage2_v1")
    assert "came from OCR" in prompts.get_prompt("stage2_v5")   # v5 = v2 + field descriptions

    class Cfg:
        stage1 = type("S", (), {"prompt_version": "stage1_v3"})
        stage2 = type("S", (), {"prompt_version": "stage2_v1", "send_images": False})
        visual = type("V", (), {"mode": "never"})

    assert prompt_version_for(2, Cfg, {"origin": "gemini_stage1"}) == "stage2_v5"
    assert prompt_version_for(2, Cfg, {"origin": "native_text"}) == "stage2_v1"
    assert prompt_version_for(1, Cfg, {}) == "stage1_v3"


def test_run_all_drives_every_step_with_progress_on(project):
    from labeller.pipeline import run_all

    cfg, store, docs = project
    api = FakeApi(responder)
    lines: list[str] = []
    summary = run_all(cfg, store, docs, FakeOcr, lambda: api, interval=0, out=lines.append, progress=True)
    assert list(summary) == ["prepare", "OCR", "Stage 1 submit", "Stage 1 wait", "Stage 2 submit", "Stage 2 wait"]
    assert store.status("imgsrc:aaa")["state"] == "accepted"  # with progress on, lines go to tqdm.write, not `out`


def test_reevaluate_rescues_a_document_after_a_gate_is_loosened(project):
    import dataclasses

    from labeller.pipeline import reevaluate

    cfg, store, docs = project
    strict = dataclasses.replace(cfg, max_change_ratio=0.01)
    prepare(strict, store, docs)
    ocr_step(strict, store, FakeOcr, docs, out=lambda _: None)
    api = FakeApi(responder)
    submit(strict, store, api, 1, docs, out=lambda _: None)
    collect(strict, store, api, 1, docs, out=lambda _: None)
    assert store.status("imgsrc:aaa")["reason"].startswith("too_many_changes")
    created = api.created

    result = reevaluate(cfg, store, docs, out=lambda _: None)  # cfg keeps the normal 0.30 limit
    assert result["stage1_stage1_ok"] == 1
    assert store.status("imgsrc:aaa")["state"] == "needs_stage2"
    assert store.status("imgsrc:aaa")["reason"] is None
    assert api.created == created  # re-checking made no API calls

    submit(cfg, store, api, 2, docs, out=lambda _: None)
    assert collect(cfg, store, api, 2, docs, out=lambda _: None)["accepted"] == 1


def test_reevaluate_accepts_a_value_split_across_columns_after_widening_the_gap(project):
    import dataclasses

    from labeller.pipeline import reevaluate

    class Columns(FakeOcr):
        def recognise(self, image):
            self.calls += 1
            return [
                {"text": "Sender 450 Main Street", "score": 0.95, "box": [[10, 10], [120, 10], [120, 20], [10, 20]]},
                {"text": "Charles Duggan New York", "score": 0.95, "box": [[10, 40], [120, 40], [120, 50], [10, 50]]},
            ]

    def columns_responder(request):
        props = request["generation_config"]["response_schema"]["properties"]
        if "corrections" in props:
            return gemini_text({
                "usable": True, "document_type": "fax", "language": "en", "fill_level": "filled",
                "illegible_regions": 0,
                "pages": [{"page": 1, "legibility": "good", "has_handwriting": False, "has_checkboxes": False,
                           "has_signature": False, "has_table": False}],
                "corrections": [], "missed": [],
            })
        return gemini_text({"document_type": "fax", "fields": [
            {"name": "sender_address", "section": "sender", "data_type": "string", "raw_value": "450 Main New York",
             "normalized_value": "", "status": "extracted", "confidence": "high", "description": "What this field holds."}]})

    cfg, store, docs = project  # max_value_gap is 0 here
    prepare(cfg, store, docs)
    ocr_step(cfg, store, Columns, docs, out=lambda _: None)
    api = FakeApi(columns_responder)
    submit(cfg, store, api, 1, docs, out=lambda _: None)
    collect(cfg, store, api, 1, docs, out=lambda _: None)
    submit(cfg, store, api, 2, docs, out=lambda _: None)
    collect(cfg, store, api, 2, docs, out=lambda _: None)
    assert store.status("imgsrc:aaa")["reason"] == "not_in_source:sender_address"
    created = api.created

    wide = dataclasses.replace(cfg, max_value_gap=6)
    result = reevaluate(wide, store, docs, out=lambda _: None)
    assert result["stage2_accepted"] == 1
    assert store.status("imgsrc:aaa")["state"] == "accepted"
    assert (store.ann_dir("imgsrc:aaa") / "annotation.json").exists()
    assert api.created == created


def test_reevaluate_also_tightens_documents_that_were_already_accepted(project):
    import dataclasses

    from labeller.pipeline import reevaluate, run_all

    cfg, store, docs = project
    api = FakeApi(responder)
    run_all(cfg, store, docs, FakeOcr, lambda: api, interval=0, out=lambda _: None)
    assert store.status("imgsrc:aaa")["state"] == "accepted"

    # Unchanged settings: re-checking an accepted document changes nothing.
    assert dict(reevaluate(cfg, store, docs, out=lambda _: None)) == {}
    assert store.status("imgsrc:aaa")["state"] == "accepted"

    # A stricter Stage 1 gate must not leave the document accepted.
    strict = dataclasses.replace(cfg, max_change_ratio=0.01)
    result = reevaluate(strict, store, docs, out=lambda _: None)
    assert result["stage1_now_excluded"] == 1
    assert store.status("imgsrc:aaa")["state"] == "excluded"
    assert store.status("imgsrc:aaa")["reason"].startswith("too_many_changes")


def test_provenance_records_the_model_that_ran_not_the_current_config(project):
    import dataclasses

    from labeller.config import StageConfig

    cfg, store, docs = project
    prepare(cfg, store, docs)
    ocr_step(cfg, store, FakeOcr, docs, out=lambda _: None)
    api = FakeApi(responder)
    submit(cfg, store, api, 1, docs, out=lambda _: None)
    collect(cfg, store, api, 1, docs, out=lambda _: None)
    submit(cfg, store, api, 2, docs, out=lambda _: None)

    # The config changes after submitting but before the results are collected.
    changed = dataclasses.replace(cfg, stage2=StageConfig(model="someone-else", prompt_version="stage2_v1",
                                                          max_output_tokens=10))
    collect(changed, store, api, 2, docs, out=lambda _: None)
    provenance = json.loads((store.ann_dir("imgsrc:aaa") / "annotation.json").read_text())["provenance"]
    assert provenance["stage2"]["model"] == "test-flash"
    assert provenance["stage1"]["model"] == "test-flash-lite"
    assert provenance["stage2"]["job"].startswith("batches/")


def test_repair_provenance_corrects_annotations_written_with_the_wrong_model(project):
    from labeller.pipeline import repair_provenance, run_all

    cfg, store, docs = project
    api = FakeApi(responder)
    run_all(cfg, store, docs, FakeOcr, lambda: api, interval=0, out=lambda _: None)
    path = store.ann_dir("imgsrc:aaa") / "annotation.json"
    annotation = json.loads(path.read_text())
    annotation["provenance"]["stage2"]["model"] = "wrong-model"      # what the old code could write
    path.write_text(json.dumps(annotation))
    status = store.status("imgsrc:aaa")
    store.set_status("imgsrc:aaa", status["state"], models=None)      # and no recorded models

    assert repair_provenance(store, out=lambda _: None) == 1
    assert json.loads(path.read_text())["provenance"]["stage2"]["model"] == "test-flash"
    assert store.status("imgsrc:aaa")["models"]["stage2"]["model"] == "test-flash"
    assert repair_provenance(store, out=lambda _: None) == 0          # nothing left to fix


def test_reevaluate_leaves_exclusions_that_gates_do_not_control(project):
    from labeller.pipeline import reevaluate

    cfg, store, docs = project
    prepare(cfg, store, docs)
    store.set_status("imgsrc:aaa", "excluded", reason="unreadable_page:1")
    assert dict(reevaluate(cfg, store, docs, out=lambda _: None)) == {}
    assert store.status("imgsrc:aaa")["state"] == "excluded"


def test_ocr_is_not_repeated_on_resume(project):
    cfg, store, docs = project
    prepare(cfg, store, docs)
    engine = FakeOcr()
    ocr_step(cfg, store, lambda: engine, docs, out=lambda _: None)
    store.set_status("imgsrc:aaa", "needs_ocr")
    ocr_step(cfg, store, lambda: engine, docs, out=lambda _: None)
    assert engine.calls == 1


def test_low_confidence_ocr_excludes_before_any_gemini_call(project):
    cfg, store, docs = project
    prepare(cfg, store, docs)

    class Weak(FakeOcr):
        def recognise(self, image):
            return [{"text": "Name: ?", "score": 0.2, "box": [[10, 10], [120, 10], [120, 20], [10, 20]]}]

    ocr_step(cfg, store, Weak, docs, out=lambda _: None)
    assert store.status("imgsrc:aaa")["state"] == "excluded"
    assert store.status("imgsrc:aaa")["reason"].startswith("ocr_low_confidence")


def test_stage1_v2_without_ocr_is_a_config_error(tmp_path):
    config_path = tmp_path / "config" / "annotation.yaml"
    config_path.parent.mkdir()
    config_path.write_text(CONFIG_OCR.replace("ocr: {enabled: true", "ocr: {enabled: false"), encoding="utf-8")
    with pytest.raises(ValueError, match="stage1_v2 needs the local OCR draft"):
        load_config(config_path)
