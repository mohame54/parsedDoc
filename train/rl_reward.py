"""Reward for the RL answers (extended format, see train/rl_prompts.py). Plain SFT scoring is in train/gen_eval.py and
is not changed by this module.

Every reference field has a state, set when the example was built (train/rl_data.py):
  answerable   the field is readable; a correct value earns +1, an abstention 0, a wrong value -LAMBDA.
  partial      the field is hard to read; a value is scored on its merits (no penalty for a wrong guess, no
               abstention bonus), so a correct read of a blurred field still counts.
  must_abstain the field is unreadable on this page; abstaining (null) earns +BETA, inventing a value -LAMBDA.

Components, each in [0, 1], averaged with weights (a component that does not apply is left out and the rest are
renormalised):
  content    the field rewards above, mapped from [-1, 1] to [0, 1]
  calibration  1 - mean Brier score of the stated confidences against being right (proper scoring rule)
  schema     the answer follows the prompt's JSON Schema (the wrapped one)
  rotation   the reported page rotation against the true upright turn, circular distance, linear to ROT_TOL
  reason     the reason tag matches the degradation (flagged fields) or is `none` (readable fields)
An unparseable answer, or one cut off, scores 0 whatever the components say.
"""
from __future__ import annotations

from typing import Any

from .gen_eval import align, extract_json, flatten, is_null, record_label, schema_score, value_credit
from .rl_prompts import AnswerSpec, confidence_value, unwrap_answer

LAMBDA = 1.0          # cost of a wrong value on a readable field (and of an invented value on an unreadable one)
BETA = 0.5            # reward for correctly abstaining on an unreadable field; below a correct answer, so guessing is
                      # never better than abstaining on what is really unreadable, and abstaining is never free either
ROT_TOL = 10.0        # degrees of rotation error that still earn partial credit
WEIGHTS = {"content": 0.55, "calibration": 0.2, "schema": 0.1, "rotation": 0.1, "reason": 0.05}
STATES = ("answerable", "partial", "must_abstain")


def _circular(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def score_rl(prediction: str, reference: str, spec: AnswerSpec | dict | None, states: dict[str, str] | None = None,
             upright: float | None = None, tags: dict[str, str] | None = None, prompt_schema: dict | None = None) -> dict[str, Any]:
    """Breakdown of the reward for one answer.

    reference: the PLAIN reference answer (JSON, no wrapping); states: field path -> state (default answerable);
    upright: the true clockwise turn that makes the page upright; tags: field path -> the degradation tag applied
    to it (blurry, occluded, ...); prompt_schema: the wrapped schema the prompt showed (for the schema term)."""
    spec = spec if isinstance(spec, AnswerSpec) or spec is None else AnswerSpec.from_dict(spec)
    states, tags = states or {}, tags or {}
    model = extract_json(prediction)
    if model is None or not isinstance(model, (dict, list)):
        return {"total": 0.0, "valid": False, "parts": {}, "stats": {}}
    ref_obj = extract_json(reference)
    plain, conf, reason, rotation = unwrap_answer(model, spec)
    plain = align(plain, ref_obj) if ref_obj is not None else plain
    pred_pairs, ref_pairs = flatten(plain), flatten(ref_obj if ref_obj is not None else {})

    conf_map = dict(_paired(plain, conf)) if spec is not None else {}
    reason_map = dict(_paired(plain, reason)) if spec is not None and spec.reason_key else {}
    field_rewards: list[float] = []
    brier: list[float] = []
    reason_hits: list[float] = []
    stats = {s: {"n": 0, "abstained": 0, "wrong": 0, "nonnull": 0, "correct": 0} for s in STATES}
    confidence_rows: list[tuple[float, int]] = []          # (stated confidence, correct) for AUROC / ECE
    for path, ref_value in ref_pairs.items():
        state = states.get(path, "answerable")
        raw_value = pred_pairs.get(path, "")
        abstained = is_null(raw_value)
        credit = value_credit(raw_value, ref_value)        # 0..1; both null is 1
        stats[state]["n"] += 1
        if abstained:
            stats[state]["abstained"] += 1
        if not is_null(ref_value):                          # exact accuracy: no partial credit, so it cannot be gamed
            stats[state]["nonnull"] += 1                    # the way the fuzzy part of the reward can
            if not abstained and credit >= 1.0:
                stats[state]["correct"] += 1
        if state == "answerable":
            if abstained and not is_null(ref_value):
                reward_field = 0.0                         # a readable field left empty
            elif credit == 0.0:
                reward_field = -LAMBDA
                stats[state]["wrong"] += 1
            else:
                reward_field = credit
        elif state == "partial":
            reward_field = 0.0 if abstained else credit    # no penalty for a hard field, no bonus for abstaining
        else:                                              # must_abstain
            if abstained:
                reward_field = BETA
            else:
                reward_field = -LAMBDA
                stats[state]["wrong"] += 1
        field_rewards.append(reward_field)

        stated = confidence_value(conf_map.get(path), spec) if spec is not None else None
        if stated is not None and not abstained:
            correct = 1.0 if credit >= 0.5 else 0.0
            brier.append((stated - correct) ** 2)
            confidence_rows.append((stated, int(correct)))
        if spec is not None and spec.reason_key:
            given = reason_map.get(path) or "none"
            if state == "answerable":
                reason_hits.append(1.0 if given == "none" else 0.0)
            elif path in tags:
                reason_hits.append(1.0 if given == tags[path] else 0.0)

    parts: dict[str, float] = {}
    if field_rewards:
        mean = sum(field_rewards) / len(field_rewards)
        parts["content"] = (max(-1.0, min(1.0, mean)) + 1.0) / 2.0
    if brier:
        parts["calibration"] = 1.0 - sum(brier) / len(brier)
    if prompt_schema is not None:
        parts["schema"] = schema_score(model, prompt_schema)
    if spec is not None and spec.rotation_key and upright is not None:
        parts["rotation"] = 0.0 if rotation is None else max(0.0, 1.0 - _circular(rotation, upright) / ROT_TOL)
    if reason_hits:
        parts["reason"] = sum(reason_hits) / len(reason_hits)
    total_weight = sum(WEIGHTS[k] for k in parts)
    total = sum(WEIGHTS[k] * v for k, v in parts.items()) / total_weight if total_weight else 0.0
    return {"total": round(total, 4), "valid": True, "parts": {k: round(v, 4) for k, v in parts.items()},
            "stats": stats, "confidence_rows": confidence_rows}


def reward_rl(prediction: str, reference: str, spec: AnswerSpec | dict | None, states: dict[str, str] | None = None,
              upright: float | None = None, tags: dict[str, str] | None = None, prompt_schema: dict | None = None) -> float:
    """The scalar the RL trainer maximises. See `score_rl` for the arguments and the breakdown."""
    return score_rl(prediction, reference, spec, states, upright, tags, prompt_schema)["total"]


def _paired(plain: Any, other: Any, prefix: str = ""):
    """(path, leaf of `other`) for every leaf of `plain`, with paths built exactly as `flatten` builds them, so a
    parallel tree (confidences, reasons) lines up with the plain answer even for lists matched by name."""
    if isinstance(plain, dict):
        for key, item in plain.items():
            yield from _paired(item, other.get(key) if isinstance(other, dict) else None,
                               f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(plain, list):
        for index, item in enumerate(plain):
            child = other[index] if isinstance(other, list) and index < len(other) else None
            yield from _paired(item, child, f"{prefix}[{record_label(item, index)}]")
    else:
        yield prefix, other


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Eval metrics over many `score_rl` results: abstention and hallucination rates per state, calibration (ECE over
    ten bins, the share of confidently wrong and unsure-but-right answers) and AUROC of confidence against being
    right (0.5 = confidence tells nothing; a collapse alarm)."""
    out: dict[str, Any] = {"examples": len(rows), "total": _mean([r["total"] for r in rows]), "per_state": {}}
    for state in STATES:
        n = sum(r["stats"].get(state, {}).get("n", 0) for r in rows if r.get("valid"))
        ab = sum(r["stats"].get(state, {}).get("abstained", 0) for r in rows if r.get("valid"))
        wr = sum(r["stats"].get(state, {}).get("wrong", 0) for r in rows if r.get("valid"))
        nonnull = sum(r["stats"].get(state, {}).get("nonnull", 0) for r in rows if r.get("valid"))
        right = sum(r["stats"].get(state, {}).get("correct", 0) for r in rows if r.get("valid"))
        out["per_state"][state] = {"fields": n, "abstain_rate": round(ab / n, 4) if n else None,
                                   "wrong_rate": round(wr / n, 4) if n else None,
                                   "exact_rate": round(right / nonnull, 4) if nonnull else None}
    pairs = [p for r in rows for p in r.get("confidence_rows", [])]
    if pairs:
        counts = [0] * 10
        hits = [0] * 10
        for stated, correct in pairs:
            b = min(9, int(stated * 10))
            counts[b] += 1
            hits[b] += correct
        n = len(pairs)
        out["ece"] = round(sum(counts[b] / n * abs(hits[b] / counts[b] - (b + 0.5) / 10)
                               for b in range(10) if counts[b]), 4)
        out["high_conf_wrong"] = round(sum(1 for s, c in pairs if s >= 0.8 and not c) / n, 4)
        out["low_conf_right"] = round(sum(1 for s, c in pairs if s < 0.5 and c) / n, 4)
        out["auroc"] = _auroc(pairs)
    return out


def history_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Flat series for the RL checkpoint reports, from the `score_rl` results of one batch of completions:
      reward/<part>        every reward component on its own (content, calibration, schema, rotation, reason)
      reward/valid_json    the share of answers that parse
      behaviour/...        abstention and wrong / invented value rates per field state
      quality/exact_accuracy_<state>   fields answered with exactly the right value (no partial credit): the measure
                           to hold the reward against, since it is not what the reward pays for
      calibration/...      ECE, AUROC, confident-and-wrong, unsure-but-right"""
    if not rows:
        return {}
    out: dict[str, float] = {"reward/valid_json": sum(1 for r in rows if r.get("valid")) / len(rows)}
    for part in WEIGHTS:
        values = [r["parts"][part] for r in rows if r.get("valid") and part in r.get("parts", {})]
        if values:
            out[f"reward/{part}"] = sum(values) / len(values)
    summary = aggregate(rows)
    for state, info in summary["per_state"].items():
        for key, name in (("abstain_rate", "behaviour/abstain_rate_"), ("wrong_rate", "behaviour/wrong_rate_"),
                          ("exact_rate", "quality/exact_accuracy_")):
            if info.get(key) is not None:
                out[f"{name}{state}"] = info[key]
    for key in ("ece", "auroc", "high_conf_wrong", "low_conf_right"):
        if summary.get(key) is not None:
            out[f"calibration/{key}"] = summary[key]
    return out


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _auroc(pairs: list[tuple[float, int]]) -> float | None:
    """Probability that a random right answer is stated more confidently than a random wrong one (ties count half)."""
    pos = [s for s, c in pairs if c]
    neg = [s for s, c in pairs if not c]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return round(wins / (len(pos) * len(neg)), 4)
