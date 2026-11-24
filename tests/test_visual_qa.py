import random

import pytest

from train.derive import TASKS, derive
from train.processors.mixture import PRESETS, DatasetProcessor


def _rec(**kw):
    base = {"id": "d:1", "images": ["p.png"], "text": ["x"], "document_type": "form", "fields": [],
            "tables": [], "visual_elements": [], "figures": []}
    return {**base, **kw}


def _box(desc, state, page=1):
    return {"element_id": desc, "type": "checkbox", "state": state, "description": desc, "page": page}


def _qa(record, task, seed=0):
    out = derive(record, random.Random(seed), [task])
    return {e["messages"][0]["content"][-1]["text"] if isinstance(e["messages"][0]["content"], list)
            else e["messages"][0]["content"]: e["messages"][1]["content"] for e in out}


def test_checkbox_state_and_counts():
    rec = _rec(visual_elements=[_box("next to 'Yes'", "checked"), _box("next to 'No'", "unchecked"),
                                _box("next to 'Maybe'", "unchecked")])
    answers = {}
    for seed in range(10):
        answers.update(_qa(rec, "checkbox_qa", seed))
    assert answers["Is the checkbox described as 'next to 'Yes'' checked? Answer yes or no."] == "yes"
    assert answers["Is the checkbox described as 'next to 'No'' checked? Answer yes or no."] == "no"
    assert answers["How many checkboxes in this document are checked?"] == "1"
    assert answers["How many checkboxes in this document are unchecked?"] == "2"


def test_checkbox_skips_leaky_duplicate_and_unknown():
    rec = _rec(visual_elements=[_box("checked box by Yes", "checked"), _box("dup", "checked"), _box("dup", "unchecked"),
                                _box("odd", "unknown")])
    answers = {}
    for seed in range(10):
        answers.update(_qa(rec, "checkbox_qa", seed))
    assert answers == {}          # leaky and duplicate skipped, an unknown state suppresses the counts


def _table(rows, headers=("name", "price"), **cell):
    return {"table_id": "t", "page": 1, "headers": list(headers), "rows": [
        {"row_id": str(i), "row_type": "data", "cells": [{"column": 1, "text": b, **cell}, {"column": 0, "text": a}]}
        for i, (a, b) in enumerate(rows)]}


def test_table_lookup_extreme_and_count():
    rec = _rec(tables=[_table([("pen", "$1,200"), ("cup", "$30"), ("ink", "$450")])])
    answers = {}
    for seed in range(20):
        answers.update(_qa(rec, "table_qa", seed))
    assert answers["How many data rows does the table on page 1 have?"] == "3"
    assert answers["In the table on page 1, which row has the highest 'price'? Give the 'name' value."] == "pen"
    lookup = {q: a for q, a in answers.items() if "what is the 'price' value" in q}
    assert lookup and all(a == {"pen": "$1,200", "cup": "$30", "ink": "$450"}[q.split("is '")[-1].rstrip("'?")]
                          for q, a in lookup.items())


def test_table_skips_text_column_spans_and_short_tables():
    text = _rec(tables=[_table([("a", "x"), ("b", "y"), ("c", "z")])])
    assert not any("highest" in q for q in _qa(text, "table_qa"))
    assert _qa(_rec(tables=[_table([("a", "1"), ("b", "2")], row_span=2)]), "table_qa") == {}
    assert _qa(_rec(tables=[_table([("a", "1")])]), "table_qa") == {}


def test_deterministic_and_presets_known():
    rec = _rec(tables=[_table([("a", "1"), ("b", "2"), ("c", "3")])], visual_elements=[_box("Yes", "checked")])
    assert derive(rec, random.Random(3), ["table_qa", "checkbox_qa"]) == derive(rec, random.Random(3), ["table_qa", "checkbox_qa"])
    assert set(PRESETS["visual"]) <= set(TASKS) and {"checkbox_qa", "table_qa"} <= set(PRESETS["visual"])
    DatasetProcessor(preset="visual")


def test_train_vision_picks_the_visual_preset(tmp_path):
    from train.train import parse_args
    base = ["--root", "r", "--model-id", "m"]
    assert parse_args(base).preset == "structured"
    assert parse_args(base + ["--train-vision"]).preset == "visual"
    assert parse_args(base + ["--train-vision", "--preset", "structured"]).preset == "structured"
