"""Stage 2 v7 figures, per-source visual routing, grouped question sidecars (ChartQA) and config sanity."""
import copy
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from corpus import datasets as corpus_datasets
from corpus.labels import parse_answers
from labeller import prompts, schemas
from labeller.pipeline import _config_visual, _document_visual, stage2_uses_images
from labeller.validate import check_figures, trend_of, validate_stage2_v3
from tests.test_download_progress import FakeStream, _jpeg
from tests.test_stage2_v3 import SOURCE_TEXT, v3


def figure(figure_type="line_chart", values=(10, 14.5, 20), estimated=False, trend="increasing", **over):
    obj = {"figure_id": "figure_1", "figure_type": figure_type, "page": 1, "title": "Sales", "caption": "",
           "summary": "Sales per month.", "x_axis": {"label": "month", "unit": ""},
           "y_axis": {"label": "sales", "unit": "units"}, "trend": trend, "evidence": [],
           "series": [{"name": "units", "points": [{"label": f"m{i}", "value": v, "estimated": estimated}
                                                  for i, v in enumerate(values)]}]}
    if figure_type == "photo":
        obj["series"] = []
    obj.update(over)
    return obj


def v7(figures, **over):
    return v3(figures=figures, **over)


def test_valid_chart_passes_and_figures_are_required_only_in_v7():
    assert validate_stage2_v3(v7([figure()]), SOURCE_TEXT, {"P1-L1", "P1-L2"}, 1, figures=True) == []
    missing = v3()
    del missing["figures"]
    assert validate_stage2_v3(missing, SOURCE_TEXT, {"P1-L1", "P1-L2"}, 1, figures=True)   # schema error
    assert validate_stage2_v3(missing, SOURCE_TEXT, {"P1-L1", "P1-L2"}, 1) == []           # v3 unaffected


def test_chart_without_data_and_non_finite_values_are_rejected():
    empty = figure(values=())
    assert "chart_without_data:figure_1" in check_figures([empty])
    nan = figure(values=(1.0, float("nan")))
    assert "bad_value:figure_1" in check_figures([nan])


def test_trend_is_recomputed_from_the_points_and_the_correction_is_marked():
    wrong = figure(values=(1, 2, 3), trend="decreasing")
    check_figures([wrong])
    assert wrong["trend"] == "increasing" and wrong["trend_corrected"] is True
    assert trend_of(figure(values=(3, 2, 1))) == "decreasing"
    assert trend_of(figure(values=(1, 3, 2))) == "mixed"
    assert trend_of(figure(values=(5, 5.1, 5))) == "stable"                  # within 5% of the largest value
    assert trend_of(figure("pie_chart", values=(1, 2))) == "not_applicable"  # pie values are shares
    assert trend_of(figure("photo")) == "not_applicable"


def test_figure_ids_are_references_and_evidence_must_be_real():
    rel = v7([figure()], relationships=[{"from_id": "figure_1", "to_id": "t1_r1", "type": "other"}])
    assert validate_stage2_v3(rel, SOURCE_TEXT, {"P1-L1", "P1-L2"}, 1, figures=True) == []
    bad_ev = v7([figure(evidence=["P9-L9"])])
    assert any(r.startswith("unknown_evidence") for r in
               validate_stage2_v3(bad_ev, SOURCE_TEXT, {"P1-L1", "P1-L2"}, 1, figures=True))


def test_v7_prompt_and_schema_are_registered_and_v6_stays_frozen():
    assert prompts.MULTIMODAL_STAGE2_PROMPT == "stage2_v7"
    assert "stage2_v7" in prompts.FIGURE_STAGE2_PROMPTS and "stage2_v6" not in prompts.FIGURE_STAGE2_PROMPTS
    assert "F. FIGURES" in prompts.get_prompt("stage2_v7") and "F. FIGURES" not in prompts.get_prompt("stage2_v6")
    v7_types = schemas.STAGE2_V7_RESPONSE_SCHEMA["properties"]["visual_elements"]["items"]["properties"]["type"]["enum"]
    v6_types = schemas.STAGE2_V3_DESCRIBED_RESPONSE_SCHEMA["properties"]["visual_elements"]["items"]["properties"]["type"]["enum"]
    assert "strikethrough" in v7_types and "strikethrough" not in v6_types
    assert schemas.stage2_schema(True, True, True) is schemas.STAGE2_V7_RESPONSE_SCHEMA
    Draft202012Validator.check_schema(schemas.STAGE2_V7_RESPONSE_SCHEMA)


def test_parse_answers_reads_chartqa_lists():
    assert parse_answers("['2014']") == ["2014"]
    assert parse_answers("['Yes', '6']") == ["Yes", "6"]
    assert parse_answers(["a", " "]) == ["a"]
    assert parse_answers("plain text") == ["plain text"]
    assert parse_answers(None) == []


def _jpg_row(data, question, label="['2014']"):
    return {"image": {"bytes": data, "path": None}, "query": question, "label": label, "human_or_machine": "0"}


@pytest.fixture
def chartqa_hub(monkeypatch):
    chart_a, chart_b = _jpeg((200, 10, 10)), _jpeg((10, 200, 10))
    rows = [_jpg_row(chart_a, "In which year is it highest?"),
            _jpg_row(chart_a, "What is the sum?", "['95']"),          # same image again: a second question
            _jpg_row(chart_b, "Which bar is tallest?", "['Mar']"),
            _jpg_row(_jpeg((10, 10, 200)), "Third chart?")]
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(
        load_dataset=lambda *a, **k: FakeStream(rows), Image=lambda decode: None))


def test_grouped_download_keeps_one_image_per_chart_with_all_its_questions(tmp_path, chartqa_hub):
    source = {"id": "chartqa", "dataset_id": "x/y", "group_by_image": True}
    files = corpus_datasets.download_grouped_qa(source, tmp_path, max_documents=10)
    assert len(files) == 3                                           # 4 rows, 3 distinct images
    sidecars = {f.stem: json.loads(f.with_name(f.stem + ".qa.json").read_text()) for f in files}
    first = next(v for v in sidecars.values() if len(v["qa"]) == 2)
    assert [q["question"] for q in first["qa"]] == ["In which year is it highest?", "What is the sum?"]
    assert first["qa"][1]["answers"] == ["95"] and first["origin"] == "chartqa"


def test_grouped_download_counts_images_not_rows_for_the_document_cap(tmp_path, chartqa_hub):
    source = {"id": "chartqa", "dataset_id": "x/y", "group_by_image": True}
    files = corpus_datasets.download_grouped_qa(source, tmp_path, max_documents=2)
    assert len(files) == 2
    total_questions = sum(len(json.loads(f.with_name(f.stem + ".qa.json").read_text())["qa"]) for f in files)
    assert total_questions == 3                                      # the repeat of chart A is still kept


@dataclass
class _Visual:
    mode: str = "auto"


@dataclass
class _Stage2:
    send_images: bool = False


@dataclass
class _Cfg:
    visual: _Visual = field(default_factory=_Visual)
    stage2: _Stage2 = field(default_factory=_Stage2)
    data_root: Path = Path("/nonexistent")
    max_pages: int = 4


def test_a_source_setting_overrides_the_local_analysis_per_document(tmp_path):
    always = {"original_path": "x.png", "visual_mode": "always"}
    never = {"original_path": "x.png", "visual_mode": "never"}
    auto = {"original_path": "x.png"}
    cfg = _Cfg()                                                     # visual.mode auto: no file exists, so
    assert _config_visual(auto) is None                              # any analysis would crash: none must run
    assert _document_visual(always, cfg)["multimodal"] is True
    assert _document_visual(never, cfg)["multimodal"] is False
    assert _document_visual(always, cfg)["source"] == "config"
    assert stage2_uses_images(cfg, {"visual": _document_visual(always, cfg)}) is True
    assert stage2_uses_images(cfg, {"visual": _document_visual(never, cfg)}) is False


def test_global_modes_still_apply_to_documents_without_a_source_setting():
    assert stage2_uses_images(_Cfg(visual=_Visual("always")), {}) is True
    assert stage2_uses_images(_Cfg(visual=_Visual("never")), {"visual": {"multimodal": True}}) is False
    assert stage2_uses_images(_Cfg(visual=_Visual("auto")), {"visual": {"multimodal": True}}) is True
    assert stage2_uses_images(_Cfg(stage2=_Stage2(True), visual=_Visual("never")), {}) is True


def test_config_sources_fit_the_budget_and_carry_valid_routing():
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    enabled = [s for s in config["sources"] if s.get("enabled")]
    assert sum(float(s["max_download_gb"]) for s in enabled) <= float(config["project"]["max_total_gb"])
    for s in enabled:
        assert s.get("visual_mode") in (None, "auto", "always", "never"), s["id"]
    chartqa = next(s for s in enabled if s["id"] == "chartqa")
    assert chartqa["group_by_image"] is True and chartqa["visual_mode"] == "always"
    assert {s["id"] for s in enabled if s.get("visual_mode") == "never"} == {"nova_invoices_receipts", "wildreceipt"}
