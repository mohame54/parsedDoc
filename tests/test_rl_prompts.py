"""RL answer formats: the wrapped schema validates the reference answer, the prompt keeps its task wording, and the
model output unwraps back to plain values with confidences, reasons and rotation."""
import json
import random

import pytest

from train.gen_eval import schema_from_prompt, schema_score
from train.rl_prompts import (AnswerSpec, build_prompt, confidence_value, sample_spec, unwrap_answer, wrap_answer,
                              wrap_schema)

SCHEMA = {"type": "object", "properties": {
    "header": {"type": "object", "properties": {"total": {"type": ["number", "null"]},
                                                "name": {"type": ["string", "null"]}},
               "required": ["total", "name"], "additionalProperties": False},
    "items": {"type": "array", "items": {"type": "object", "properties": {"desc": {"type": ["string", "null"]}},
                                         "required": ["desc"], "additionalProperties": False}}},
    "required": ["header", "items"], "additionalProperties": False}
PLAIN = {"header": {"total": 12.5, "name": None}, "items": [{"desc": "tea"}]}
PROMPT = "Read the receipt. Use null when unreadable.\n\nJSON Schema:\n" + json.dumps(SCHEMA)
SPECS = [AnswerSpec("value", "confidence", "percent", "reason", "rotation"),
         AnswerSpec("val", "certainty", "unit", None, None),
         AnswerSpec("answer", "score", "level", "why", "orientation")]


@pytest.mark.parametrize("spec", SPECS)
def test_the_reference_answer_validates_against_the_wrapped_schema(spec):
    schema = wrap_schema(SCHEMA, spec)
    answer = json.loads(wrap_answer(PLAIN, SCHEMA, spec, upright=90))
    assert schema_score(answer, schema) == 1.0


def test_plain_format_is_untouched():
    assert wrap_schema(SCHEMA, None) is SCHEMA
    assert json.loads(wrap_answer(PLAIN, SCHEMA, None)) == PLAIN
    assert build_prompt(PROMPT, SCHEMA, None) == PROMPT


def test_the_prompt_keeps_its_wording_and_shows_the_wrapped_schema():
    spec = SPECS[0]
    prompt = build_prompt(PROMPT, wrap_schema(SCHEMA, spec), spec)
    assert prompt.startswith("Read the receipt. Use null when unreadable.")
    assert "guessing" in prompt and f"`{spec.rotation_key}`" in prompt
    assert schema_from_prompt(prompt) == wrap_schema(SCHEMA, spec)


def test_null_values_get_a_low_confidence_and_present_ones_a_sure_one():
    spec = SPECS[0]
    answer = json.loads(wrap_answer(PLAIN, SCHEMA, spec, upright=90, reasons={"header.name": "occluded"}))
    assert answer["header"]["name"] == {"value": None, "confidence": 0, "reason": "occluded"}
    assert answer["header"]["total"]["confidence"] == 100
    assert answer["rotation"] == 90


def test_unwrap_returns_plain_values_confidences_reasons_and_rotation():
    spec = SPECS[0]
    output = {"header": {"total": {"value": 12.5, "confidence": 80, "reason": "none"},
                         "name": {"value": "ana", "confidence": 30, "reason": "blurry"}},
              "items": [{"desc": {"value": "tea", "confidence": 95, "reason": "none"}}], "rotation": 270}
    plain, conf, reason, rotation = unwrap_answer(output, spec)
    assert plain == {"header": {"total": 12.5, "name": "ana"}, "items": [{"desc": "tea"}]}
    assert conf["header"]["name"] == 30 and conf["items"][0]["desc"] == 95
    assert reason["header"]["name"] == "blurry" and rotation == 270


def test_unwrap_of_a_plain_output_is_the_identity():
    assert unwrap_answer(PLAIN, None) == (PLAIN, None, None, None)


def test_confidence_scales_map_to_zero_one():
    assert confidence_value(80, SPECS[0]) == 0.8
    assert confidence_value(0.4, SPECS[1]) == 0.4
    assert confidence_value("high", SPECS[2]) == 0.9
    assert confidence_value("sure-ish", SPECS[2]) is None and confidence_value(None, SPECS[0]) is None


def test_sampling_covers_every_spelling_and_the_plain_format():
    rng = random.Random(0)
    specs = [sample_spec(rng, p_extended=0.8) for _ in range(300)]
    seen = [s for s in specs if s]
    assert {s.value_key for s in seen} == {"value", "val", "answer"}
    assert {s.confidence_key for s in seen} == {"confidence", "confi", "certainty", "score"}
    assert {s.scale for s in seen} == {"percent", "unit", "level"}
    assert sum(1 for s in specs if s is None) > 0
    assert AnswerSpec.from_dict(seen[0].to_dict()) == seen[0] and AnswerSpec.from_dict(None) is None


def _extended_dataset(tmp_path, **kwargs):
    from PIL import Image

    from tests.test_derive import _record
    from train.dataset import DocumentDataset
    from train.processors import DatasetProcessor
    record = _record()
    rel = record["images"][0]
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (90, 60), (255, 255, 255)).save(tmp_path / rel)
    (tmp_path / "train.jsonl").write_text("\n".join(json.dumps(dict(record, id=f"doc:{i}")) for i in range(4)))
    return DocumentDataset(tmp_path, "train", DatasetProcessor(preset="structured"), samples_per_document=6, **kwargs)


def test_sft_extended_examples_validate_against_their_own_prompt_schema(tmp_path):
    ds = _extended_dataset(tmp_path, extended_p=1.0)
    checked = 0
    for index in range(len(ds)):
        item = ds[(0, index)]
        if item["task"] != "extract_to_schema":
            continue
        schema = schema_from_prompt(item["prompt"])
        assert schema_score(json.loads(item["label"]), schema) == 1.0
        checked += 1
    assert checked > 0


def test_extended_p_zero_changes_nothing_and_other_tasks_are_never_touched(tmp_path):
    plain = _extended_dataset(tmp_path)
    extended = _extended_dataset(tmp_path, extended_p=1.0)
    for index in range(len(plain)):
        a, b = plain[(0, index)], extended[(0, index)]
        assert a["task"] == b["task"]
        if a["task"] != "extract_to_schema":
            assert a["prompt"] == b["prompt"] and a["label"] == b["label"]


def test_planning_sees_the_extended_label_and_the_rotation_matches_the_applied_turn(tmp_path):
    from train.degrade import upright_turn
    ds = _extended_dataset(tmp_path, extended_p=1.0, augment="strong", augment_p=1.0)

    def length(sizes, prompt, label):
        return len(prompt) + len(label)
    rotated = 0
    for index in range(len(ds)):
        item = ds[(0, index)]
        assert ds.length_of(0, index, length) == len(item["prompt"]) + len(item["label"])
        if item["task"] == "extract_to_schema":
            answer = json.loads(item["label"])
            keys = [k for k in ("rotation", "page_rotation", "orientation") if k in answer]
            if keys:
                assert answer[keys[0]] == pytest.approx(upright_turn(item["augment"][0]))
                rotated += 1
    assert rotated > 0
