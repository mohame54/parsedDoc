from pathlib import Path

import fitz

from corpus.utils import is_blank_form, pdf_form_fill
from labeller.validate import validate_fill

PROJECT = {"reject_blank_forms": True, "min_widgets": 5, "max_fill_ratio": 0.1}


def make_form(path: Path, values: list[str]) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    for index, value in enumerate(values):
        widget = fitz.Widget()
        widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
        widget.field_name = f"field_{index}"
        widget.rect = fitz.Rect(72, 80 + index * 30, 300, 100 + index * 30)
        widget.field_value = value
        page.add_widget(widget)
    doc.save(path)
    doc.close()
    return path


def test_pdf_form_fill_counts_widgets_and_filled_values(tmp_path):
    blank = make_form(tmp_path / "blank.pdf", ["", "", "", "", "", "", ""])
    partly = make_form(tmp_path / "partly.pdf", ["John", "", "", "", "", "", ""])
    assert pdf_form_fill(blank) == {"widgets": 7, "filled": 0}
    assert pdf_form_fill(partly) == {"widgets": 7, "filled": 1}


def test_pdf_without_widgets_reports_zero(tmp_path):
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "A plain text document with no fields.")
    path = tmp_path / "plain.pdf"
    doc.save(path)
    doc.close()
    assert pdf_form_fill(path) == {"widgets": 0, "filled": 0}


def test_blank_rule_respects_thresholds():
    assert is_blank_form({"widgets": 92, "filled": 0}, PROJECT)
    assert not is_blank_form({"widgets": 92, "filled": 60}, PROJECT)
    # Too few widgets to judge: never rejected on this rule alone.
    assert not is_blank_form({"widgets": 3, "filled": 0}, PROJECT)
    # Switched off by configuration.
    assert not is_blank_form({"widgets": 92, "filled": 0}, {**PROJECT, "reject_blank_forms": False})


def test_fill_backstop_flags_sparse_extraction_only_for_larger_documents():
    def fields(extracted, total):
        return ([{"name": f"f{i}", "status": "extracted"} for i in range(extracted)]
                + [{"name": f"g{i}", "status": "not_present"} for i in range(total - extracted)])

    assert validate_fill(fields(3, 25), 0.25, 8) == ["blank_form:extracted=3/25"]
    assert validate_fill(fields(10, 25), 0.25, 8) == []
    # A 5-field form is exempt even if sparse.
    assert validate_fill(fields(0, 5), 0.25, 8) == []
