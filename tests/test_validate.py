from labeller.validate import normalize_text, validate_fields, validate_stage1, validate_stage2


def field(name, status="extracted", raw="John Smith", confidence="high", section="s", data_type="string"):
    return {"name": name, "section": section, "data_type": data_type, "raw_value": raw,
            "normalized_value": raw, "status": status, "confidence": confidence}


def test_normalize_collapses_whitespace_and_case():
    assert normalize_text("  ACME   Ltd\n") == "acme ltd"


def test_extracted_value_must_appear_in_source():
    reasons = validate_fields([field("name", raw="Jane Doe")], "Name: John Smith")
    assert reasons == ["not_in_source:name"]


def test_extracted_value_matches_despite_layout_whitespace():
    assert validate_fields([field("name", raw="John  Smith")], "Name:\nJohn Smith") == []


def test_uncertain_and_illegible_block_acceptance():
    reasons = validate_fields(
        [field("a", status="uncertain", raw=""), field("b", status="illegible", raw="")], "text"
    )
    assert reasons == ["uncertain:a", "illegible:b"]


def test_absent_fields_must_not_carry_a_value():
    reasons = validate_fields([field("dob", status="not_present", raw="1951")], "text")
    assert reasons == ["value_for_absent_field:dob"]


def test_low_confidence_and_duplicates_are_reported():
    reasons = validate_fields([field("x", confidence="low"), field("x")], "John Smith")
    assert "low_confidence:x" in reasons
    assert "duplicate_field:x" in reasons


def test_value_split_across_columns_matches_with_a_gap_but_not_without():
    source = "Sender 450 Lexington Avenue Charles Duggan New York, NY 10017"
    raw = "450 Lexington Avenue New York, NY 10017"
    assert validate_fields([field("addr", raw=raw)], source, max_gap=0) == ["not_in_source:addr"]
    assert validate_fields([field("addr", raw=raw)], source, max_gap=3) == []


def test_markup_inside_the_source_does_not_break_a_match():
    source = "Test market geography: Divisions 621 and 627 [handwritten: (Wisconsin)] Price point"
    assert validate_fields([field("geo", raw="Divisions 621 and 627 (Wisconsin)")], source) == []


def test_a_word_that_is_not_in_the_source_is_never_accepted():
    source = "Sender 450 Lexington Avenue Charles Duggan New York, NY 10017"
    raw = "450 Lexington Avenue Boston, NY 10017"
    assert validate_fields([field("addr", raw=raw)], source, max_gap=10) == ["not_in_source:addr"]


def test_words_in_the_wrong_order_are_rejected():
    assert validate_fields([field("n", raw="Smith John")], "Name: John Smith", max_gap=6) == ["not_in_source:n"]


def test_gap_limit_is_respected():
    source = "alpha one two three four five six seven beta"
    assert validate_fields([field("v", raw="alpha beta")], source, max_gap=7) == []
    assert validate_fields([field("v", raw="alpha beta")], source, max_gap=6) == ["not_in_source:v"]


def test_stage2_schema_errors_short_circuit():
    assert validate_stage2({"document_type": "form"}, "text")[0].startswith("schema:")


def test_stage1_gates():
    ok = {"usable": True, "document_type": "form", "illegible_regions": 0,
          "pages": [{"page": 1, "legibility": "good", "text": "hello"}]}
    assert validate_stage1(ok, 1, 0) == []

    unreadable = {**ok, "pages": [{"page": 1, "legibility": "unreadable", "text": "hello"}]}
    assert validate_stage1(unreadable, 1, 0) == ["unreadable_page:1"]

    assert validate_stage1(ok, 2, 0) == ["page_count_mismatch:1!=2"]
    assert validate_stage1({**ok, "illegible_regions": 2}, 1, 0) == ["illegible_regions:2"]
    assert validate_stage1({**ok, "usable": False}, 1, 0) == ["unusable_document"]


def test_stage2_descriptions_are_requested_required_and_cleaned():
    import copy

    from labeller import prompts, schemas
    from labeller.validate import clean_descriptions, validate_stage2

    # v4-v6 ask for a description; v1-v3 are frozen and do not
    for version in ("stage2_v4", "stage2_v5", "stage2_v6"):
        assert "FIELD DESCRIPTIONS" in prompts.get_prompt(version)
        assert version in prompts.DESCRIBED_STAGE2_PROMPTS
    for version in ("stage2_v1", "stage2_v2", "stage2_v3"):
        assert "FIELD DESCRIPTIONS" not in prompts.get_prompt(version)
    assert prompts.MULTIMODAL_STAGE2_PROMPT in prompts.DESCRIBED_STAGE2_PROMPTS
    assert "description" in schemas.STAGE2_V3_DESCRIBED_RESPONSE_SCHEMA["properties"]["fields"]["items"]["required"]
    assert "description" not in schemas.STAGE2_RESPONSE_SCHEMA["properties"]["fields"]["items"]["required"]

    field = {"name": "patient_name", "section": "s", "data_type": "string", "raw_value": "John Smith",
             "normalized_value": "John Smith", "status": "extracted", "confidence": "high"}
    text = "Patient: John Smith"
    old = {"document_type": "form", "fields": [dict(field)]}
    new = {"document_type": "form", "fields": [dict(field, description="Full name of the patient.")]}
    assert validate_stage2(old, text, described=False) == []
    assert any("description" in r for r in validate_stage2(old, text, described=True))   # now required
    assert validate_stage2(new, text, described=True) == []

    fields = [dict(field, description="  Full name \n of the patient. "),
              dict(field, description="It is John Smith."),            # leaks the value -> blanked
              dict(field, description="x" * 500),                      # too long -> blanked
              dict(field, description="")]
    assert clean_descriptions(copy.deepcopy(fields)) == 2
    cleaned = copy.deepcopy(fields)
    clean_descriptions(cleaned)
    assert [f["description"] for f in cleaned] == ["Full name of the patient.", "", "", ""]
