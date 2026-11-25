import json

from PIL import Image

from train.show import differences, load, render_html, render_text


def _row(**kw):
    return {"id": "d#extract_to_schema0", "task": "extract_to_schema", "score": 0.5, "images": ["p.png"],
            "prompt": "Extract... JSON Schema:\n{}", "reference": json.dumps({"a": "1", "b": "2"}),
            "prediction": json.dumps({"a": "1", "b": "9", "c": "x"}), **kw}


def test_differences_name_wrong_missing_and_extra():
    out = differences(_row()["prediction"], _row()["reference"])
    assert any("b: expected '2', got '9'" in d for d in out) and any(d.startswith("c: not in the reference") for d in out)
    assert differences("not json", _row()["reference"]) == ["the answer is not valid JSON"]
    assert differences("x", "plain text") == []


def test_renders_text_and_html_with_and_without_images(tmp_path):
    Image.new("RGB", (40, 30), "white").save(tmp_path / "p.png")
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps(_row()) + "\n" + json.dumps(_row(id="o", task="ocr", reference="hi", prediction="hi")) + "\n")
    rows = load(path)
    assert "--- request ---" in render_text(rows) and "expected '2'" in render_text(rows)
    page = render_html(rows, tmp_path)
    assert "data:image/jpeg;base64," in page and "Reference" in page
    assert "page image not found" in render_html(rows, None)
    assert load(path, task="ocr")[0]["id"] == "o" and load(path, worst_first=True, limit=1)
    old = {k: v for k, v in _row().items() if k not in ("prompt", "images")}
    assert "not stored" in render_text([old])
