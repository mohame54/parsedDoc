"""RL reward: honest answers beat guessing and blanket abstention, calibration and rotation count, bad JSON scores 0."""
import json

import pytest

from train.rl_prompts import AnswerSpec, wrap_answer, wrap_schema
from train.rl_reward import aggregate, reward_rl, score_rl

SCHEMA = {"type": "object", "properties": {"total": {"type": ["number", "null"]},
                                           "name": {"type": ["string", "null"]},
                                           "date": {"type": ["string", "null"]}},
          "required": ["total", "name", "date"], "additionalProperties": False}
REF = {"total": 12.5, "name": "ana", "date": "2017-06-30"}
SPEC = AnswerSpec("value", "confidence", "percent", "reason", "rotation")
STATES = {"total": "answerable", "name": "must_abstain", "date": "partial"}
TAGS = {"name": "occluded", "date": "blurry"}


def out(total=(12.5, 95), name=(None, 5), date=("2017-06-30", 70), reason=("none", "none", "blurry"), rot=0):
    return json.dumps({
        "total": {"value": total[0], "confidence": total[1], "reason": reason[0]},
        "name": {"value": name[0], "confidence": name[1], "reason": reason[1]},
        "date": {"value": date[0], "confidence": date[1], "reason": reason[2]},
        "rotation": rot})


def r(prediction, upright=0.0, states=STATES, tags=TAGS, spec=SPEC):
    return score_rl(prediction, json.dumps(REF), spec, states, upright, tags, wrap_schema(SCHEMA, spec))


def test_the_honest_answer_scores_best():
    honest = r(out())["total"]
    assert honest > 0.9
    assert honest > r(out(name=("ana", 90), reason=("none", "none", "blurry")))["total"]   # guessed an unreadable name
    assert honest > r(out(total=(None, 5), reason=("none", "none", "blurry")))["total"]   # abstained on a readable total


def test_abstaining_on_a_readable_field_is_worse_than_answering_it_right():
    assert r(out())["total"] > r(out(total=(None, 10)))["total"]


def test_guessing_an_unreadable_field_is_worse_than_abstaining_on_it():
    guessed = r(out(name=("ana", 60)))
    abstained = r(out(name=(None, 5)))
    assert abstained["total"] > guessed["total"]
    assert guessed["stats"]["must_abstain"]["wrong"] == 1 and abstained["stats"]["must_abstain"]["abstained"] == 1


def test_blanket_abstention_and_blanket_guessing_both_lose_to_the_honest_answer():
    honest = r(out())["total"]
    always_abstain = r(out(total=(None, 0), name=(None, 0), date=(None, 0)))["total"]
    always_guess = r(out(total=(99.0, 99), name=("ana", 99), date=("2017-06-29", 99)))["total"]
    assert honest > always_abstain and honest > always_guess


def test_confident_and_wrong_costs_more_than_unsure_and_wrong():
    confident_wrong = r(out(total=(99.0, 99)))
    unsure_wrong = r(out(total=(99.0, 10)))
    assert confident_wrong["parts"]["calibration"] < unsure_wrong["parts"]["calibration"]
    assert unsure_wrong["total"] > confident_wrong["total"]


def test_rotation_is_circular_and_exact_at_the_right_turn():
    assert r(out(rot=90), upright=90)["parts"]["rotation"] == 1.0
    assert r(out(rot=359), upright=1)["parts"]["rotation"] == pytest.approx(0.8)   # 2 degrees apart across zero
    assert r(out(rot=180), upright=90)["parts"]["rotation"] == 0.0
    assert r(out(rot=95), upright=90)["parts"]["rotation"] == pytest.approx(0.5)


def test_invalid_or_cut_off_json_scores_zero():
    assert reward_rl('{"total": {"value": 12.5', json.dumps(REF), SPEC, STATES, 0.0, TAGS) == 0.0
    assert reward_rl("not json", json.dumps(REF), SPEC) == 0.0


def test_reason_tags_are_rewarded_only_when_they_match_the_degradation():
    assert r(out(reason=("none", "occluded", "blurry")))["parts"]["reason"] == 1.0
    assert r(out(reason=("none", "none", "none")))["parts"]["reason"] < 1.0      # did not flag the blurred date
    assert r(out(reason=("blurry", "none", "blurry")))["parts"]["reason"] < 1.0  # flagged a readable field


def test_a_list_is_matched_by_content_not_position():
    ref = {"items": [{"desc": "tea", "price": 2}, {"desc": "cake", "price": 4}]}
    spec = AnswerSpec("value", "confidence", "unit", None, None)
    schema = {"type": "object", "properties": {"items": {"type": "array", "items": {
        "type": "object", "properties": {"desc": {"type": "string"}, "price": {"type": "number"}},
        "required": ["desc", "price"], "additionalProperties": False}}}, "required": ["items"],
        "additionalProperties": False}
    swapped = {"items": [{"desc": {"value": "cake", "confidence": 1.0}, "price": {"value": 4, "confidence": 1.0}},
                         {"desc": {"value": "tea", "confidence": 1.0}, "price": {"value": 2, "confidence": 1.0}}]}
    total = score_rl(json.dumps(swapped), json.dumps(ref), spec, {}, None, {}, wrap_schema(schema, spec))
    assert total["stats"]["answerable"]["wrong"] == 0 and total["parts"]["content"] == 1.0


def test_the_plain_format_scores_without_confidence_or_rotation_terms():
    plain = json.dumps(REF)
    result = score_rl(plain, plain, None, {}, None, {}, None)
    assert result["total"] == 1.0 and set(result["parts"]) == {"content"}


def test_aggregate_reports_rates_and_auroc():
    good = [r(out(total=(12.5, 95)))]
    bad = [r(out(total=(99.0, 99), name=("ana", 90)))]
    summary = aggregate(good + bad)
    assert summary["examples"] == 2
    assert summary["per_state"]["must_abstain"]["wrong_rate"] == 0.5
    assert summary["high_conf_wrong"] > 0 and summary["auroc"] is not None
    perfect = aggregate([r(out(total=(99.0, 99))), r(out(total=(0.0, 1)))])
    assert perfect["auroc"] is not None
