"""Generation-based validation: the scorers, the prompt/left-padding builder and the hook in the loop."""
import json

import pytest
import torch
from PIL import Image

from train.gen_eval import extract_json, flatten, normalise, score_example, summarise, value_credit, reward, suspicious_reference, schema_from_prompt, schema_score


def test_json_is_found_inside_fences_and_surrounding_text():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": [1, 2]} done') == {"a": [1, 2]}
    assert extract_json("not json at all") is None
    assert extract_json('{"a": ') is None


def test_flatten_matches_records_by_name_not_by_position():
    one = {"fields": [{"name": "total", "value": "10"}, {"name": "date", "value": "May"}]}
    other = {"fields": [{"name": "date", "value": "May"}, {"name": "total", "value": "10"}]}
    assert flatten(one) == flatten(other)
    assert flatten({"a": None, "b": " X  y "}) == {"a": "", "b": "x y"}


def test_json_answers_score_by_value_and_by_structure():
    ref = json.dumps({"vendor": "ACME", "total": "10.00", "date": "2020-01-02"})
    perfect = score_example(ref, ref)
    assert perfect["kind"] == "json" and perfect["json_valid"] and perfect["score"] == 1.0
    wrong_value = score_example(json.dumps({"vendor": "ACME", "total": "99", "date": "2020-01-02"}), ref)
    assert wrong_value["path_f1"] == 1.0 and 0 < wrong_value["value_f1"] < 1          # right structure, one wrong value
    broken = score_example('{"vendor": "ACME", "total"', ref)
    assert broken["json_valid"] is False and broken["score"] == 0.0
    assert score_example("```json\n" + ref + "\n```", ref)["score"] == 1.0           # a code fence is not an error


def test_short_answers_are_exact_matches_and_long_text_gets_a_similarity():
    assert score_example("  Invoice ", "invoice")["score"] == 1.0
    assert score_example("receipt", "invoice")["score"] == 0.0
    text = "the quick brown fox jumps over the lazy dog and keeps running far away"
    close = score_example(text.replace("fox", "fax"), text)
    assert close["kind"] == "text" and 0.9 < close["score"] < 1.0
    assert score_example("", text)["score"] == 0.0
    assert normalise("A \n b") == "a b"


@pytest.mark.parametrize("pred, ref", [
    ("30/06/2017", "2017-06-30"), ("June 30, 2017", "2017-06-30"), ("30. Juni 2017", "2017-06-30"),
    ("7:30pm", "19:30"), ("1:45 P. M.", "1: 45 p. m."), ("$7,569,000.00", "7569000.0"), ("1.234,56", "1234.56"),
    ("(12)", "-12"), ("N/A", ""), ("Yes", "true"), ("Straße", "STRASSE"), ("Carlton, Private Stock", "Private Stock, Carlton")])
def test_values_that_mean_the_same_get_full_credit(pred, ref):
    assert value_credit(pred, ref) == 1.0


@pytest.mark.parametrize("pred, ref", [("2017-06-29", "2017-06-30"), ("7:30am", "19:30"), ("514.35", "514.36"),
                                       ("2010-09-01", "05-05"), ("\u2612", "\u2610"), ("\u2612", "")])
def test_values_that_differ_still_get_no_credit(pred, ref):
    assert value_credit(pred, ref) == 0.0


def test_a_key_left_out_where_the_reference_is_null_is_correct():
    ref = json.dumps({"a": "x", "b": None})
    assert score_example('{"a": "x"}', ref)["score"] == 1.0
    assert score_example('{"a": "x", "b": "invented"}', ref)["score"] < 1.0


def test_reward_floors_valid_json_penalises_invention_and_flags_bad_references():
    ref = json.dumps({"a": "x", "b": None})
    assert reward(ref, ref) == 1.0 and reward("not json", ref) == 0.0
    assert reward('{"a": "x", "b": "guess"}', ref) < reward('{"a": "x", "b": null}', ref)
    assert reward('{"a": "wrong", "b": null}', ref) >= 0.05                      # valid JSON keeps a little
    assert suspicious_reference('{"d": "2094-04-28"}') and not suspicious_reference('{"d": "2017-06-30"}')


PROMPT = 'Extract. JSON Schema: {"type":"object","properties":{"n":{"type":["number","null"]},"s":{"type":["string","null"]}},"required":["n","s"],"additionalProperties":false} Reference'


def test_schema_compliance_is_part_of_the_reward():
    schema = schema_from_prompt(PROMPT)
    assert schema["required"] == ["n", "s"] and schema_from_prompt("no schema here") is None
    assert schema_score({"n": 1, "s": "a"}, schema) == 1.0
    assert schema_score({"n": "1", "s": "a", "extra": 1}, schema) < 1.0
    ref = '{"n": 5, "s": "a"}'
    assert reward(ref, ref, PROMPT) == 1.0
    assert reward('{"n": "5", "s": "a"}', ref, PROMPT) < 1.0 and reward('{"n": "5", "s": "a"}', ref) == 1.0
    assert reward('{"n": 5, "s": "a"', ref, PROMPT) == 0.0                       # cut off: the gate is closed


def test_line_items_are_matched_by_content_not_position():
    ref = json.dumps({"items": [{"d": "tea", "p": 2.5}, {"d": "cake", "p": 4}, {"d": "milk", "p": 1}]})
    shuffled = json.dumps({"items": [{"d": "milk", "p": 1.0}, {"d": "tea", "p": "2.50"}, {"d": "cake", "p": 4}]})
    assert score_example(shuffled, ref)["value_f1"] == 1.0
    missing = json.dumps({"items": [{"d": "cake", "p": 4}, {"d": "tea", "p": 2.5}]})
    assert 0.5 < score_example(missing, ref)["value_f1"] < 1.0


def test_the_summary_averages_per_task_and_counts_valid_json():
    rows = [{"task": "extract", "kind": "json", "json_valid": True, "path_f1": 1.0, "score": 0.5},
            {"task": "extract", "kind": "json", "json_valid": False, "path_f1": 0.0, "score": 0.0},
            {"task": "classify", "kind": "short", "exact_match": True, "score": 1.0}]
    out = summarise(rows)
    assert out["examples"] == 3 and out["per_task"]["extract"] == {"n": 2, "score": 0.25, "json_valid": 0.5,
                                                                    "path_f1": 0.5, "strict_f1": None, "hallucination": None, "kind": "json"}
    assert out["per_task"]["classify"]["score"] == 1.0 and out["json_valid"] == 0.5


# ---- the generator, with a stand-in model --------------------------------------------------------------

class Tok:
    pad_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [10 + ord(c) % 50 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + (i % 26)) for i in ids)


class FakeBase:
    def __init__(self):
        self.calls = []

    def generate(self, input_ids, attention_mask, **kw):
        self.calls.append({"input_ids": input_ids.clone(), "attention_mask": attention_mask.clone(), **kw})
        tail = torch.tensor([[5, 6, 7, 1, 9]] * input_ids.shape[0])            # three tokens, end-of-text, then junk
        return torch.cat([input_ids, tail], dim=1)


class FakeWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.base = FakeBase()


def _evaluator(tmp_path):
    from train.gen_eval import GenerationEvaluator
    from train.processors import DeepSeekOCRV2Transform, batch_adapter_for

    Image.new("RGB", (60, 80), "white").save(tmp_path / "p.png")
    from .test_derive import _record
    rows = [dict(_record(), id=f"d:{i}") for i in range(3)]
    (tmp_path / "val.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    transform = DeepSeekOCRV2Transform(tokenizer=Tok(), mode="base", fixed_length=False)
    return GenerationEvaluator(tmp_path, transform, batch_adapter_for("deepseek-community/DeepSeek-OCR-2"),
                               samples_per_task=1, max_new_tokens=4096, batch_size=2, precision="fp32",
                               out_dir=tmp_path / "gen")


def test_the_prompt_is_cut_where_the_answer_starts_and_left_padded(tmp_path):
    evaluator = _evaluator(tmp_path)
    assert evaluator.items, "the validation file produced no examples"
    chunk = evaluator.items[:2]
    kwargs, width, budget = evaluator._inputs(chunk)
    assert kwargs["input_ids"].shape == (len(chunk), width) and "labels" not in kwargs
    assert (kwargs["attention_mask"][:, -1] == 1).all()                      # generation continues from the last column
    for row in range(len(chunk)):                                             # no answer tokens leaked into the prompt
        features = evaluator.transform(chunk[row]["images"], chunk[row]["prompt"], chunk[row]["label"])
        prompt = int((features["labels"] != -100).nonzero()[0])
        padding = width - prompt
        assert torch.equal(kwargs["input_ids"][row, padding:], features["input_ids"][:prompt])
        assert (kwargs["attention_mask"][row, :padding] == 0).all()
    assert "pixel_values" in kwargs and kwargs["pixel_values"].shape[0] == len(chunk)
    assert 16 < budget <= 4096


def test_generate_decodes_up_to_the_end_of_text_token(tmp_path):
    evaluator = _evaluator(tmp_path)
    wrapper = FakeWrapper()
    answers = evaluator.generate(wrapper, evaluator.items)
    assert len(answers) == len(evaluator.items) and set(answers) == {"fgh"}       # tokens 5, 6, 7 and nothing after id 1
    assert wrapper.base.calls[0]["do_sample"] is False and wrapper.base.calls[0]["eos_token_id"] == 1


def test_run_scores_writes_the_answers_and_logs_scalars(tmp_path):
    evaluator = _evaluator(tmp_path)
    scalars = {}

    class Logger:
        def scalars(self, step, values):
            scalars.update(values)

    summary = evaluator.run(FakeWrapper(), step=7, logger=Logger())
    written = (tmp_path / "gen" / "generations_step_0000007.jsonl").read_text().splitlines()
    assert len(written) == summary["examples"] == len(evaluator.items)
    row = json.loads(written[0])
    assert {"id", "task", "reference", "prediction", "score"} <= set(row) and row["prediction"] == "fgh"
    assert "gen/score" in scalars and any(k.startswith("gen/") and k != "gen/score" for k in scalars)


def test_the_loop_runs_the_generation_check_at_the_start_every_n_steps_and_at_the_end(tmp_path):
    from train.dataset import make_loader
    from train.processors import DatasetProcessor
    from train.sft import SFTConfig, train_sft
    from .test_bucketing import StubModel, _docs, _loader

    calls = []

    class Spy:
        def run(self, model, step, logger=None):
            calls.append(step)

    loader = _loader(_docs(tmp_path), batch_tokens=400, max_batch_size=4)
    model = StubModel()
    cfg = SFTConfig(epochs=2, grad_accum=2, precision="fp32", log_every=100, save_every_steps=0, tensorboard=False,
                    out_dir=str(tmp_path / "o"), gen_eval=Spy(), gen_eval_every=3,
                    batch_adapter=lambda b: {k: b[k] for k in ("input_ids", "attention_mask", "labels")})
    result = train_sft(2, model, torch.optim.SGD(model.parameters(), lr=0.01), loader, None, 0, cfg=cfg)
    assert calls[0] == 0 and calls[-1] == result["step"]
    assert set(range(3, result["step"] + 1, 3)) <= set(calls) and len(calls) == len(set(calls))


def test_scoring_forgives_spacing_punctuation_and_slips_but_not_wrong_values():
    from train.gen_eval import value_credit
    ref = json.dumps({"name": "Bistecca RestaurantPtaLtd", "addr": "25 Mohamed Sultan Rd S(238969)", "total": "514.35"})
    close = json.dumps({"name": "Bistecca RestaurantPteLtd", "addr": "25 Mohamed Sultan Rd S (238969)", "total": "514.350"})
    out = score_example(close, ref)
    assert out["strict_f1"] < 0.5 and out["score"] > 0.95                  # spacing, a one-letter slip, 514.350
    assert value_credit("T.G.I.FRIDAY'S", "T.G.I. FRIDAY'S") == 1.0
    assert value_credit("0359 Table 997 #Party 1", "211 W 34th St, New York, NY 10001") == 0.0
    assert value_credit("12", "21") == 0.0 and value_credit("1,200", "1200") == 1.0
    assert score_example("ACME-Inc", "acme inc")["score"] == 1.0           # short answers: punctuation ignored
