import json

from labeller.audit import audit
from labeller.clean import clean_export, clean_field, clean_record, date_from_raw, number_from_raw


def _f(raw, norm="", kind="string", status="extracted", name="x"):
    return {"name": name, "section": "s", "data_type": kind, "raw_value": raw, "normalized_value": norm, "status": status}


def test_strings_use_the_raw_text():
    out, notes = clean_field(_f("Nov.", "November"))
    assert out["normalized_value"] == "" and notes[0]["action"] == "string_rewrite_dropped"
    out, notes = clean_field(_f("T.G.I.FRIDAY'S", "T.G.I. FRIDAY'S"))
    assert out["normalized_value"] == "" and notes[0]["action"] == "string_respaced_dropped"
    out, notes = clean_field(_f("ACME", "ACME"))
    assert notes == [] and out["normalized_value"] == "ACME"
    assert clean_field(_f("[x] .5 % METHYL", "0.5 % METHYL"))[0]["normalized_value"] == ""
    assert clean_field(_f("line one\nline  two", "line one line two"))[1] == []         # whitespace only: kept
    assert clean_field(_f("J. E. Mann\nD. E. Cawthon", "J. E. Mann / D. E. Cawthon"))[0]["normalized_value"] == ""


def test_numbers_are_kept_only_when_the_raw_text_says_so():
    assert number_from_raw("0.5", ".5") and number_from_raw("1200.5", "1,200.50") and number_from_raw("12", "$ 12")
    assert number_from_raw("1234.5", "1.234,50")
    assert not number_from_raw("514.35", "514.53")
    assert clean_field(_f("514.53", "514.35", "number"))[0]["normalized_value"] == ""
    assert clean_field(_f("1,200", "1200", "number"))[1] == []


def test_dates_must_be_real_and_written_in_the_raw_text():
    assert date_from_raw("1984-07-09", "7/9/84") is None
    assert date_from_raw("2019-11-05", "Nov. 5, 2019") is None
    assert date_from_raw("2019-11-05", "5. November 2019") is None
    assert date_from_raw("2002-02-29", "29.02.2002") == "invalid_date"
    assert date_from_raw("2019-12-05", "Nov. 5, 2019") == "date_not_derivable"
    assert date_from_raw("5 Nov 2019", "5 Nov 2019") == "date_not_derivable"
    assert date_from_raw("2017-03", "March 2017") is None and date_from_raw("2019-02", "2 2019") is None
    assert date_from_raw("2019-02", "Feb. 2019") is None and date_from_raw("2019", "FY 2019") is None
    assert date_from_raw("2019-03", "February 2019") == "date_not_derivable"      # the wrong month
    assert date_from_raw("2017-13", "13 2017") == "invalid_date"
    out, notes = clean_field(_f("29.02.2002", "2002-02-29", "date"))
    assert out["normalized_value"] == "" and notes[0]["action"] == "invalid_date"


def test_booleans_and_status():
    assert clean_field(_f("[x]", "true", "boolean"))[1] == []
    assert clean_field(_f("[x] AIR", "true", "boolean"))[1] == [] and clean_field(_f("[ ] HEAT", "false", "boolean"))[1] == []
    assert clean_field(_f("[ ] HEAT", "true", "boolean"))[0]["normalized_value"] == ""      # the mark and the value disagree
    assert clean_field(_f("maybe", "true", "boolean"))[0]["normalized_value"] == ""
    out, notes = clean_field(_f("Zzz Corp", "", "string"), source_text="something else entirely")
    assert out["status"] == "uncertain" and notes[0]["action"] == "not_in_source"
    absent = _f("", "", status="not_present")
    assert clean_field(absent)[1] == []


def test_record_cleaning_is_idempotent_and_does_not_mutate():
    record = {"id": "a:1", "text": ["Nov. 5 2019 ACME"],
              "fields": [_f("Nov.", "November", name="m"), _f("ACME", "", name="c")]}
    once = clean_record(record)
    assert record["fields"][0]["normalized_value"] == "November"          # the input is untouched
    assert once["fields"][0]["normalized_value"] == "" and len(once["label_notes"]) == 1
    assert clean_record(once) == once


def _write_export(root, rows):
    root.mkdir(parents=True)
    (root / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (root / "images").mkdir()


def _row(i):
    return {"id": f"funsd:{i}", "source": "funsd", "images": [f"images/{i}.png"], "page_count": 1, "text": ["Nov. 5 ACME"],
            "document_type": "form", "family_id": "f", "language": "en", "tables": [], "visual_elements": [],
            "figures": [], "relationships": [], "layout": [], "qa": [], "split": "train",
            "fields": [_f("Nov.", "November", name="month"), _f("ACME", "ACME", name="company")]}


def test_audit_counts_and_clean_export_round_trip(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_export(src, [_row(1), _row(2)])
    report = audit(src)
    assert report["records"] == 2 and report["by_source"]["funsd"]["string_rewrite_dropped"] == 2
    assert report["examples"][0]["raw"] == "Nov."
    counts = clean_export(src, dst)
    assert counts == {"records": 2, "fields_changed": 2}
    rows = [json.loads(line) for line in (dst / "train.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["funsd:1", "funsd:2"] and rows[0]["fields"][0]["normalized_value"] == ""
    assert audit(dst)["by_source"]["funsd"].get("string_rewrite_dropped", 0) == 0
    extract = [json.loads(line) for line in (dst / "sft" / "extract_train.jsonl").read_text().splitlines()]
    assert json.loads(extract[0]["messages"][-1]["content"])["fields"]["s"]["month"] == "Nov."
