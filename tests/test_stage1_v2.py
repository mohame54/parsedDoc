from labeller.validate import validate_stage1_v2

DRAFT = [{"page": 1, "lines": [
    {"id": "P1-L1", "text": "Name: John Smith", "score": 0.9},
    {"id": "P1-L2", "text": "Date: 2O2O-01-02", "score": 0.7},
    {"id": "P1-L3", "text": "[ ] Check box", "score": 0.8},
]}]


def stage1(**overrides):
    obj = {
        "usable": True, "document_type": "form", "language": "en", "fill_level": "filled",
        "illegible_regions": 0,
        "pages": [{"page": 1, "legibility": "good", "has_handwriting": False, "has_checkboxes": True,
                   "has_signature": False, "has_table": False}],
        "corrections": [], "missed": [],
    }
    obj.update(overrides)
    return obj


def run(obj, **kw):
    args = dict(page_count=1, max_illegible_regions=0, exclude_blank_forms=True, max_change_ratio=0.3)
    args.update(kw)
    return validate_stage1_v2(obj, DRAFT, **args)[:3]   # (reasons, texts, metrics); lines are tested below


def test_corrected_lines_keep_their_ids_for_evidence():
    obj = stage1(corrections=[{"line": "P1-L2", "text": "Date: 2020-01-02"}],
                 missed=[{"after_line": "P1-L1", "text": "Phone: 555-0100"}])
    reasons, _, _, lines = validate_stage1_v2(obj, DRAFT, 1, 0, True, 1.0)
    assert reasons == []
    assert [line["id"] for line in lines[0]] == ["P1-L1", "added-0", "P1-L2", "P1-L3"]
    assert lines[0][2]["text"] == "Date: 2020-01-02"


def test_no_corrections_keeps_the_draft_text():
    reasons, texts, metrics = run(stage1())
    assert reasons == []
    assert texts == ["Name: John Smith\nDate: 2O2O-01-02\n[ ] Check box"]
    assert metrics == {"correction_ratio": 0.0, "change_ratio": 0.0}


def test_corrections_replace_only_the_named_lines():
    obj = stage1(corrections=[{"line": "P1-L2", "text": "Date: 2020-01-02"}])
    reasons, texts, metrics = run(obj)
    assert reasons == []
    assert "Date: 2020-01-02" in texts[0] and "John Smith" in texts[0]
    assert round(metrics["correction_ratio"], 3) == 0.333
    assert 0 < metrics["change_ratio"] < 0.1  # two characters (O -> 0) out of 33


def test_markup_and_punctuation_only_corrections_do_not_count_as_changes():
    obj = stage1(corrections=[
        {"line": "P1-L3", "text": "[x] Check box"},                       # checkbox state
        {"line": "P1-L1", "text": "Name: [handwritten: John Smith]"},     # handwriting tag
    ])
    reasons, _, metrics = run(obj)
    assert reasons == []
    assert round(metrics["correction_ratio"], 2) == 0.67   # two of three lines touched...
    assert metrics["change_ratio"] == 0.0                  # ...but no real character changed


def test_heavy_rewrites_exceed_the_change_limit():
    obj = stage1(corrections=[{"line": "P1-L1", "text": "b"}, {"line": "P1-L2", "text": "c"}])
    reasons, texts, metrics = run(obj)
    assert reasons[0].startswith("too_many_changes")
    assert texts == []
    assert metrics["change_ratio"] > 0.3


def test_missed_lines_are_inserted_after_their_anchor():
    obj = stage1(missed=[{"after_line": "P1-L1", "text": "Phone: 555-0100"}])
    # The draft is tiny, so one added line is a large share of it; relax the limit to test ordering only.
    _, texts, metrics = run(obj, max_change_ratio=1.0)
    assert metrics["change_ratio"] > 0.3
    assert texts[0].split("\n")[:2] == ["Name: John Smith", "Phone: 555-0100"]


def test_unknown_line_ids_are_rejected():
    reasons, texts, _ = run(stage1(corrections=[{"line": "P9-L9", "text": "x"}]))
    assert reasons == ["unknown_line:P9-L9"]
    assert texts == []


def test_blank_fill_level_is_excluded():
    reasons, _, _ = run(stage1(fill_level="blank"))
    assert reasons == ["blank_form:stage1"]


def test_unreadable_page_and_schema_errors():
    obj = stage1(pages=[{"page": 1, "legibility": "unreadable", "has_handwriting": False,
                         "has_checkboxes": False, "has_signature": False, "has_table": False}])
    reasons, _, _ = run(obj)
    assert reasons == ["unreadable_page:1"]

    reasons, _, _ = run({"usable": True})
    assert reasons and reasons[0].startswith("schema:")
