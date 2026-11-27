"""RL answer formats: the plain schema answer, or an extended one that also carries a confidence per field, an
optional reason, and the page rotation. SFT never sees these; they are built only for the RL data.

An `AnswerSpec` says which key names and scale the answer uses. It is sampled per example, so the model learns the
idea (value + how sure + why + how the page is turned) and not one fixed spelling. The spec is stored with the
example, and the reward reads the answer through it.

    spec = sample_spec(rng)                         # None = plain format (the normal schema, unchanged)
    schema = wrap_schema(schema, spec)              # what the prompt shows
    prompt = build_prompt(prompt, schema, spec)     # the instruction, with the new schema
    answer = wrap_answer(plain, schema, spec, ...)  # the reference answer for the label
    plain, conf, reason, rotation = unwrap_answer(model_output, spec)
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from typing import Any

from .gen_eval import record_label

VALUE_KEYS = ("value", "val", "answer")
CONFIDENCE_KEYS = ("confidence", "confi", "certainty", "score")
REASON_KEYS = ("reason", "why", "note")
ROTATION_KEYS = ("rotation", "page_rotation", "orientation")
SCALES = ("percent", "unit", "level")
LEVEL_VALUE = {"low": 0.2, "medium": 0.5, "high": 0.9}
REASON_TAGS = ("blurry", "occluded", "cut_off", "rotated", "noisy", "none")
EXTENDED_P = 0.5                # share of RL examples that get an extended format (the rest keep the plain schema)


@dataclass(frozen=True)
class AnswerSpec:
    value_key: str
    confidence_key: str
    scale: str                  # percent (0-100) | unit (0-1) | level (low / medium / high)
    reason_key: str | None      # None = no reason asked
    rotation_key: str | None    # None = no rotation asked

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AnswerSpec | None":
        return None if not data else cls(**data)


def sample_spec(rng: random.Random, p_extended: float = EXTENDED_P) -> AnswerSpec | None:
    """None for the plain format; otherwise one randomly chosen spelling of every part of the extended one."""
    if rng.random() >= p_extended:
        return None
    return AnswerSpec(
        value_key=rng.choice(VALUE_KEYS),
        confidence_key=rng.choice(CONFIDENCE_KEYS),
        scale=rng.choice(SCALES),
        reason_key=rng.choice(REASON_KEYS) if rng.random() < 0.5 else None,
        rotation_key=rng.choice(ROTATION_KEYS) if rng.random() < 0.7 else None,
    )


def _confidence_schema(spec: AnswerSpec) -> dict[str, Any]:
    if spec.scale == "percent":
        return {"type": "number", "minimum": 0, "maximum": 100,
                "description": "how sure you are the value is right, 0 (guess) to 100 (certain)"}
    if spec.scale == "unit":
        return {"type": "number", "minimum": 0, "maximum": 1,
                "description": "how sure you are the value is right, 0 (guess) to 1 (certain)"}
    return {"type": "string", "enum": ["low", "medium", "high"],
            "description": "how sure you are the value is right"}


def _wrap_property(schema: dict[str, Any], spec: AnswerSpec) -> dict[str, Any]:
    if schema.get("type") == "object" and "properties" in schema:
        inner = copy.deepcopy(schema)
        inner["properties"] = {k: _wrap_property(v, spec) for k, v in schema["properties"].items()}
        return inner
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        inner = copy.deepcopy(schema)
        inner["items"] = _wrap_property(schema["items"], spec)
        return inner
    props: dict[str, Any] = {spec.value_key: copy.deepcopy(schema), spec.confidence_key: _confidence_schema(spec)}
    required = [spec.value_key, spec.confidence_key]
    if spec.reason_key:
        props[spec.reason_key] = {"type": "string", "enum": list(REASON_TAGS),
                                  "description": "why it is unsure, or none"}
        required.append(spec.reason_key)
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def wrap_schema(schema: dict[str, Any], spec: AnswerSpec | None) -> dict[str, Any]:
    """The schema the prompt shows. Every scalar becomes {value, confidence[, reason]}; objects and arrays of
    objects are kept and walked; the top level gets the rotation key. None returns the schema unchanged."""
    if spec is None:
        return schema
    out = _wrap_property(schema, spec)
    if spec.rotation_key:
        out["properties"][spec.rotation_key] = {
            "type": "number", "minimum": 0, "maximum": 359.999,
            "description": "clockwise degrees the page must be turned to be upright (0 if it already is)"}
        out.setdefault("required", []).append(spec.rotation_key)
    return out


_CONF_NULL = {"percent": 0, "unit": 0.0, "level": "low"}
_CONF_SURE = {"percent": 100, "unit": 1.0, "level": "high"}


def _wrap_value(plain: Any, schema: dict[str, Any], spec: AnswerSpec, reasons: dict[str, str], path: str) -> Any:
    if schema.get("type") == "object" and "properties" in schema and isinstance(plain, dict):
        return {k: _wrap_value(plain.get(k), schema["properties"][k], spec, reasons, f"{path}.{k}" if path else k)
                for k in schema["properties"] if k in plain or k in schema.get("required", [])}
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict) and isinstance(plain, list):
        return [_wrap_value(item, schema["items"], spec, reasons, f"{path}[{record_label(item, i)}]")
                for i, item in enumerate(plain)]
    if plain is None:
        out = {spec.value_key: None, spec.confidence_key: _CONF_NULL[spec.scale]}
    else:
        out = {spec.value_key: plain, spec.confidence_key: _CONF_SURE[spec.scale]}
    if spec.reason_key:
        out[spec.reason_key] = reasons.get(path, "none")
    return out


def wrap_answer(plain: Any, schema: dict[str, Any], spec: AnswerSpec | None, upright: float | None = None,
                reasons: dict[str, str] | None = None) -> str:
    """The reference answer for the label: values from the plain reference, a sure confidence on every present
    value and a low one on every null, the reason tag from `reasons` (path -> tag), and the upright turn."""
    if spec is None:
        return json.dumps(plain, ensure_ascii=False)
    out = _wrap_value(plain, schema, spec, reasons or {}, "")
    if spec.rotation_key and upright is not None:
        out[spec.rotation_key] = round(float(upright), 3)
    return json.dumps(out, ensure_ascii=False)


def build_prompt(prompt: str, schema: dict[str, Any], spec: AnswerSpec | None) -> str:
    """The prompt with its JSON Schema replaced, plus the instruction for the extended format. The text before
    `JSON Schema:` is kept, so the task wording and any preamble are unchanged."""
    marker = prompt.find("JSON Schema:")
    head = prompt if marker == -1 else prompt[:marker]
    if spec is None:
        return prompt
    rules = [f"For each value, give `{spec.value_key}` (the value, or null) and `{spec.confidence_key}` "
             f"({'0-100' if spec.scale == 'percent' else '0-1' if spec.scale == 'unit' else 'low, medium or high'}). "
             "If you cannot read a value, or you are guessing, use null with a low confidence instead of an invented value."]
    if spec.reason_key:
        rules.append(f"Put the reason in `{spec.reason_key}` when you are unsure (blurry, occluded, cut_off, rotated, "
                     "noisy), otherwise none.")
    if spec.rotation_key:
        rules.append(f"Put in `{spec.rotation_key}` the clockwise degrees the page must be turned to be upright.")
    return f"{head.rstrip()}\n{' '.join(rules)}\n\nJSON Schema:\n{json.dumps(schema, ensure_ascii=False)}"


def _is_wrapped(node: Any, spec: AnswerSpec) -> bool:
    return isinstance(node, dict) and spec.value_key in node and spec.confidence_key in node


def unwrap_answer(model_output: Any, spec: AnswerSpec | None) -> tuple[Any, Any, Any, float | None]:
    """Model output (parsed JSON) -> (plain, confidence tree, reason tree, rotation). The three trees have the same
    shape as the plain answer; a leaf is the raw confidence / reason, or None where the answer had none."""
    if spec is None or not isinstance(model_output, dict):
        return model_output, None, None, None
    rotation = None
    body = dict(model_output)
    if spec.rotation_key and spec.rotation_key in body:
        try:
            rotation = float(body.pop(spec.rotation_key)) % 360
        except (TypeError, ValueError):
            rotation = None

    def walk(node: Any) -> tuple[Any, Any, Any]:
        if _is_wrapped(node, spec):
            reason = node.get(spec.reason_key) if spec.reason_key else None
            return node.get(spec.value_key), node.get(spec.confidence_key), reason
        if isinstance(node, dict):
            parts = {k: walk(v) for k, v in node.items()}
            return ({k: p[0] for k, p in parts.items()}, {k: p[1] for k, p in parts.items()},
                    {k: p[2] for k, p in parts.items()})
        if isinstance(node, list):
            parts = [walk(v) for v in node]
            return [p[0] for p in parts], [p[1] for p in parts], [p[2] for p in parts]
        return node, None, None

    plain, conf, reason = walk(body)
    return plain, conf, reason, rotation


def confidence_value(raw: Any, spec: AnswerSpec) -> float | None:
    """A raw confidence as a number in [0, 1], or None when it is missing or not readable."""
    if raw is None:
        return None
    if spec.scale == "level":
        return LEVEL_VALUE.get(str(raw).strip().lower())
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None
    if spec.scale == "percent":
        number /= 100.0
    return min(1.0, max(0.0, number))
