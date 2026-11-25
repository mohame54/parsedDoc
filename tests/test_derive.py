import ast
import json
import random

import pytest
import yaml
from PIL import Image

from train.dataset import DocumentDataset
from train.derive import TASKS, derive
from train.processors import DatasetProcessor
from train.render import render_schema, serialise, values


def _record():
    field = lambda n, v, st="extracted", s="s", t="string": {  # noqa: E731
        "name": n, "section": s, "data_type": t, "raw_value": v, "normalized_value": v, "status": st,
        "confidence": "high"}
    return {
        "id": "d:1", "images": ["p.png"], "text": ["Hello"], "document_type": "invoice_form", "language": "en",
        "fields": [field("total", "10", t="number"), field("date", "", "not_present", t="date"),
                   field("buyer", "ACME & Co", s="parties")],
        "tables": [{"table_id": "t", "page": 1, "headers": ["a", "b"], "rows": [
            {"row_id": "r", "row_type": "data", "cells": [{"column": 1, "text": "y"}, {"column": 0, "text": "x"}],
             "visual": {}}]}],
        "visual_elements": [{"element_id": "e", "type": "checkbox", "state": "checked", "description": "d", "page": 1}],
        "relationships": [],
    }


def test_every_task_is_derived_and_deterministic():
    a = derive(_record(), random.Random(1))
    assert a == derive(_record(), random.Random(1))
    figure_tasks = {"figure", "chart_qa", "chart_trend", "table_qa"}      # need figures or chart questions in the record
    assert {e["task"] for e in a} == set(TASKS) - figure_tasks
    table = next(e for e in a if e["task"] == "table")
    assert table["messages"][-1]["content"].splitlines()[2] == "| x | y |"
    schema = next(e for e in a if e["task"] == "schema")
    assert None in json.loads(schema["messages"][-1]["content"]).values()   # absent fields -> null


def test_schema_renderers_produce_valid_code():
    record = _record()
    json.loads(render_schema(record, "json_schema"))
    for fmt in ("pydantic", "dataclass"):
        ast.parse(render_schema(record, fmt))               # syntactically valid Python
    assert "class InvoiceForm" in render_schema(record, "pydantic")
    assert "interface Parties" in render_schema(record, "typescript")
    assert render_schema(record, "sql").startswith("CREATE TABLE invoice_form")


def test_output_formats_roundtrip():
    data = values(_record()["fields"])
    assert data["s"]["date"] is None and data["parties"]["buyer"] == "ACME & Co"
    assert json.loads(serialise(data, "json")) == data
    assert yaml.safe_load(serialise(data, "yaml")) == data
    assert "ACME &amp; Co" in serialise(data, "xml")
    assert serialise(data, "csv").splitlines()[0] == "s.total,s.date,parties.buyer"


def test_extract_to_schema_embeds_the_schema_in_the_prompt():
    for seed in range(20):
        example = derive(_record(), random.Random(seed), ["extract_to_schema"])[0]
        prompt = example["messages"][0]["content"][-1]["text"]
        answer = json.loads(example["messages"][1]["content"])
        for section, fields in answer.items():          # whatever the answer holds, the prompt names
            assert f'"{section}"' in prompt
            assert all(f'"{name}"' in prompt for name in fields)
    full = [json.loads(derive(_record(), random.Random(s), ["extract_to_schema"])[0]["messages"][1]["content"])
            for s in range(20)]
    assert any(a.get("parties", {}).get("buyer") == "ACME & Co" for a in full)


def test_processor_mixes_tasks_and_varies_by_epoch():
    processor = DatasetProcessor(preset="structured")
    seen_tasks = {processor.sample(_record(), epoch=e, index=0)["task"] for e in range(40)}
    assert {"extract_to_schema", "propose_schema", "extract"} <= seen_tasks
    assert "extract_format" not in seen_tasks   # JSON-only serving: not in the structured preset
    assert processor.sample(_record(), epoch=2, index=0) == processor.sample(_record(), epoch=2, index=0)
    only = DatasetProcessor(tasks=["ocr"])
    assert only.sample(_record())["task"] == "ocr"
    with pytest.raises(ValueError):
        DatasetProcessor(tasks=["nope"])


def test_dataset_items_are_prompt_images_label_and_redraw_in_train(tmp_path):
    Image.new("RGB", (4, 4)).save(tmp_path / "p.png")
    for split in ("train", "val"):
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(_record()))
    processor = DatasetProcessor(preset="structured")
    train = DocumentDataset(tmp_path, "train", processor)
    item = train[0]
    assert set(item) == {"id", "task", "prompt", "images", "label", "image_paths"}
    assert isinstance(item["prompt"], str) and isinstance(item["label"], str)
    assert isinstance(item["images"], list) and item["images"][0].size == (4, 4)

    prompts = set()
    for epoch in range(15):
        train.set_epoch(epoch)
        prompts.add(train[0]["prompt"])
    assert len(prompts) > 4

    val = DocumentDataset(tmp_path, "val", processor)
    first = (val[0]["prompt"], val[0]["label"])
    val.set_epoch(5)
    assert (val[0]["prompt"], val[0]["label"]) == first


def test_extract_and_visual_targets_validate_against_the_schema_shown_in_the_prompt():
    from jsonschema import Draft202012Validator

    from train.render import EXTRACT_SCHEMA, VISUAL_SCHEMA

    record = _record()
    record["tables"][0]["evidence"] = ["P1-L3"]
    record["tables"][0]["boxes"] = [{"line": "P1-L3", "box_px": [1, 2, 3, 4]}]
    record["visual_elements"][0]["evidence"] = ["P1-L9"]
    examples = {e["task"]: e for e in derive(record, random.Random(5), ["extract", "visual"])}
    for task, schema in (("extract", EXTRACT_SCHEMA), ("visual", VISUAL_SCHEMA)):
        prompt = examples[task]["messages"][0]["content"][-1]["text"]
        answer = examples[task]["messages"][1]["content"]
        assert "JSON Schema:" in prompt and '"additionalProperties"' in prompt
        assert list(Draft202012Validator(schema).iter_errors(json.loads(answer))) == []
        assert "evidence" not in answer and "boxes" not in answer and "P1-L" not in answer
    assert json.loads(examples["extract"]["messages"][1]["content"])["relationships"] == []


def test_extract_to_schema_styles_vary_and_answers_always_validate_against_the_prompt_schema():
    from jsonschema import Draft202012Validator

    record = _record()
    seen = {"described": 0, "preamble": 0, "subset": 0, "bare": 0}
    for seed in range(120):
        example = derive(record, random.Random(seed), ["extract_to_schema"])[0]
        prompt = example["messages"][0]["content"][-1]["text"]
        schema = json.loads(prompt.split("JSON Schema:\n", 1)[1])
        answer = json.loads(example["messages"][1]["content"])
        assert list(Draft202012Validator(schema).iter_errors(answer)) == [], (seed, prompt)
        described = '"description"' in prompt.split("JSON Schema:\n", 1)[1]
        preamble = prompt.startswith(("You are extracting", "The document is", "Task: structured"))
        subset = sum(len(v) for v in answer.values()) < 3
        seen["described"] += described
        seen["preamble"] += preamble
        seen["subset"] += subset
        seen["bare"] += not (described or preamble or subset)
        if preamble:
            assert "Use null for anything that is missing or unreadable" in prompt
    assert all(count > 0 for count in seen.values()), seen


def test_number_and_checkbox_values_are_typed_to_match_the_schema():
    from train.render import typed_value

    f = lambda t, v, st="extracted": {"data_type": t, "normalized_value": v, "raw_value": v, "status": st}  # noqa: E731
    assert typed_value(f("number", "1,200.50")) == ("number", 1200.5)
    assert typed_value(f("number", "007")) == ("string", "007")          # leading zeros are identifiers, not numbers
    assert typed_value(f("number", "n/a")) == ("string", "n/a")
    assert typed_value(f("boolean", "[x]")) == ("boolean", True)
    assert typed_value(f("boolean", "[ ]")) == ("boolean", False)
    assert typed_value(f("date", "2024-01-02")) == ("string", "2024-01-02")
    assert typed_value(f("number", "5", st="not_present")) == ("number", None)


def test_descriptions_come_from_the_table_with_a_template_fallback():
    from train.describe import Describer, key

    field = {"name": "invoice_date", "section": "header", "data_type": "date"}
    assert "YYYY-MM-DD" in Describer().field("invoice", field)
    table = Describer({key("invoice", "header", "invoice_date"): "Date the invoice was issued."})
    assert table.field("invoice", field) == "Date the invoice was issued."
    assert "Invoice date" in table.field("receipt", field)   # other document type: template


def test_describe_pass_builds_one_request_per_document_type_and_parses_the_answer():
    import json as _json

    from labeller import describe as d
    from train.describe import key

    attributes = d.collect_attributes([_record(), _record()])
    assert set(attributes) == {"invoice_form"}
    (line,) = d.build_lines(attributes)
    assert line["key"] == "describe:invoice_form" and "parties.buyer (string)" in line["request"]["contents"][0]["parts"][0]["text"]
    reply = {"descriptions": [
        {"section": "parties", "name": "buyer", "description": "Name of the buying party."},
        {"section": "parties", "name": "invented", "description": "Not requested."},
        {"section": "s", "name": "total", "description": "x" * 500}]}
    output = _json.dumps({"key": line["key"], "response": {"candidates": [{"content": {"parts": [
        {"text": _json.dumps(reply)}]}}]}})
    table = d.parse_results(output, attributes)
    assert table == {key("invoice_form", "parties", "buyer"): "Name of the buying party."}


def test_a_fields_own_description_wins_over_the_table_and_the_template():
    from train.describe import Describer, key

    field = {"name": "invoice_date", "section": "header", "data_type": "date",
             "description": "Date the invoice was issued."}
    table = Describer({key("invoice", "header", "invoice_date"): "From the table."})
    assert table.field("invoice", field) == "Date the invoice was issued."
    assert table.field("invoice", dict(field, description="")) == "From the table."
    assert "YYYY-MM-DD" in Describer().field("invoice", dict(field, description=""))


def _chart_record():
    rec = _record()
    rec["figures"] = [{"figure_id": "figure_1", "figure_type": "line_chart", "page": 1, "title": "Sales",
                       "caption": "", "summary": "Sales per month.", "x_axis": {"label": "month", "unit": ""},
                       "y_axis": {"label": "sales", "unit": "units"}, "trend": "increasing", "evidence": ["P1-L1"],
                       "series": [{"name": "units", "points": [
                           {"label": "Jan", "value": 10, "estimated": False},
                           {"label": "Feb", "value": 14.5, "estimated": False},
                           {"label": "Mar", "value": 20, "estimated": False}]}]},
                      {"figure_id": "figure_2", "figure_type": "bar_chart", "page": 1, "title": "Estimated",
                       "caption": "", "summary": "", "x_axis": {"label": "", "unit": ""},
                       "y_axis": {"label": "", "unit": ""}, "trend": "not_applicable", "evidence": [],
                       "series": [{"name": "s", "points": [{"label": "A", "value": 3, "estimated": True},
                                                          {"label": "B", "value": 5, "estimated": True}]}]}]
    rec["qa"] = [{"question": "What is the peak?", "answers": ["Mar"], "annotator": "0"},
                 {"question": "Empty?", "answers": [], "annotator": "0"}]
    return rec


def test_figure_and_chart_tasks_answers_come_from_the_data_and_validate_against_their_schema():
    from jsonschema import Draft202012Validator

    from train.render import EXTRACT_SCHEMA

    examples = derive(_chart_record(), random.Random(2), ["figure", "chart_qa", "chart_trend", "extract"])
    by_task = {}
    for e in examples:
        by_task.setdefault(e["task"], []).append(e)
    figure = json.loads(by_task["figure"][0]["messages"][1]["content"])
    assert [f["figure_id"] for f in figure] == ["figure_1", "figure_2"] and "evidence" not in figure[0]
    assert list(Draft202012Validator({"type": "array", "items": __import__("train.render", fromlist=["x"]).FIGURE_SCHEMA}).iter_errors(figure)) == []
    assert by_task["chart_qa"][0]["messages"][1]["content"] == "Mar"          # the annotated answer, not a guess
    assert "Empty?" not in json.dumps([e["messages"][0]["content"] for e in by_task["chart_qa"]])
    answers = {e["messages"][0]["content"][-1]["text"]: e["messages"][1]["content"] for e in by_task["chart_trend"]}
    assert any(a == "increasing" for a in answers.values()) and "Mar" in " ".join(answers.values())
    assert all("estimated" not in t for t in answers) and not any("Estimated" in q for q in answers)  # estimated chart skipped
    extract = json.loads(by_task["extract"][0]["messages"][1]["content"])
    assert Draft202012Validator(EXTRACT_SCHEMA).iter_errors(extract) is not None
    assert list(Draft202012Validator(EXTRACT_SCHEMA).iter_errors(extract)) == []
    assert len(extract["figures"]) == 2
