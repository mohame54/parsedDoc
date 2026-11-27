"""Does the model generate correct data? Generation-based validation.

The validation loss says how well the model predicts the reference text token by token. This runs the model the way
it is used: it gets the page and the prompt only, generates the answer, and the answer is compared with the reference.

    # a saved checkpoint (the adapter you trained), on the validation split
    python -m train.gen_eval --config config/train.yaml --checkpoint checkpoints/lora_run/final_0000612.pt
    python -m train.gen_eval --config config/train.yaml --no-adapter          # the base model, for comparison

During training the same check runs before the first step (the base model) and at the end, and every
`--gen-eval-every` steps if asked: `--gen-eval`.

How an answer is scored (chosen from the reference, so every task is covered):
  * JSON answers (extract, schema, tables, figures, ...): `json_valid` (does it parse), and the F1 of the (path, value)
    pairs against the reference. A value counts fully when it means the same: equal up to case, accents, spacing and
    punctuation, or the same number (`$7,569,000.00` = `7569000`), date (`30/06/2017` = `2017-06-30`), time
    (`7:30pm` = `19:30`), yes/no, or both empty (`null` = `N/A`, a key left out where the reference is null). It
    counts partly when it is at least 85% similar (a one-character slip) or shares at least half of its words; `strict_f1` is the all-or-nothing
    version, `path_f1` ignores the values (right structure, wrong values still count). Lists of records are
    matched by their `name` / `id` and not by position, so a different order is not an error.
  * short answers (classify, a field's value, a chart question): match after case, spacing and punctuation are removed.
  * long text (OCR): character similarity (difflib ratio) of the normalised text.
The score of every example is between 0 and 1; the summary gives the mean per task and overall. Every generated answer
is written to `generations_step_N.jsonl` so you can read them.
"""
from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import torch

log = logging.getLogger("train.gen")

ID_KEYS = ("name", "id", "field", "key", "field_id", "table_id", "figure_id", "element_id", "row_id", "label")
SHORT_ANSWER_WORDS = 12


# ---- scoring ----------------------------------------------------------------------------------------

def normalise(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def extract_json(text: str) -> Any | None:
    """The first JSON value in a model answer, also when it is wrapped in a code fence or has text around it."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except ValueError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except ValueError:
                continue
    return None


GAP = object()                 # a hole left in an aligned list where the model produced no matching item


def record_label(item: Any, index: int) -> str:
    """How a list item is named in a path: its name / id when it has one (so order does not matter), else its index."""
    if isinstance(item, dict):
        for id_key in ID_KEYS:
            if isinstance(item.get(id_key), (str, int)) and str(item[id_key]).strip():
                return f"{id_key}={squash(item[id_key]) or normalise(item[id_key])}"
    return str(index)


def flatten(value: Any, prefix: str = "") -> dict[str, str]:
    """{path: normalised leaf}. A list of records is keyed by each record's name / id, so order does not matter."""
    out: dict[str, str] = {}
    if value is GAP:
        return out
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            out.update(flatten(item, f"{prefix}[{record_label(item, index)}]"))
    else:
        out[prefix] = normalise("" if value is None else value)
    return out


def _f1(predicted: set, reference: set) -> float:
    if not predicted and not reference:
        return 1.0
    hits = len(predicted & reference)
    if not hits:
        return 0.0
    precision, recall = hits / len(predicted), hits / len(reference)
    return 2 * precision * recall / (precision + recall)


SOFT_THRESHOLD = 0.85          # a value at least this similar to the reference earns partial credit
TOKEN_THRESHOLD = 0.5          # ... and so does a value that shares at least this much of its words with it

NULLISH = {"", "null", "none", "n/a", "na", "nan", "nil", "-", "--", "\u2014", "\u2013", "undefined"}
TRUE_WORDS, FALSE_WORDS = {"true", "yes", "y", "ja", "oui", "checked", "1"}, {"false", "no", "n", "nein", "non", "unchecked", "0"}

MONTHS = {}
for _index, _names in enumerate([
        "january jan januar janvier janv", "february feb februar fevrier fevr", "march mar marz mars maerz",
        "april apr avril", "may mai", "june jun juni juin", "july jul juli juillet juil",
        "august aug aout", "september sep sept", "october oct oktober octobre okt",
        "november nov", "december dec dezember decembre dez"], start=1):
    for _name in _names.split():
        MONTHS.setdefault(_name, _index)


def fold(text: Any) -> str:
    """Case-, accent- and width-folded text (`Straße` == `STRASSE`, `é` == `e`, full-width digits are digits)."""
    text = unicodedata.normalize("NFKD", str(text)).casefold()
    return "".join(c for c in text if not unicodedata.combining(c))


def squash(text: Any) -> str:
    """Case, accents, spacing and punctuation removed: `S (238969)` and `S(238969)` are the same value."""
    return re.sub(r"[\W_]+", "", fold(text))


def is_null(text: Any) -> bool:
    return normalise(text) in NULLISH


def parse_number(text: Any) -> float | None:
    """A number written the way documents write them: `$1,234.50`, `1.234,50 EUR`, `(12)`, `15%`, `7569000.0`."""
    s = str(text).strip()
    if not s:
        return None
    negative = s.startswith("(") and s.endswith(")") or s.startswith("-") or s.startswith("\u2212")
    s = re.sub(r"(?i)^[\s(+\-\u2212]*|[)\s]*$", "", s)
    s = re.sub(r"(?i)^(?:[^\d.,]{1,4})\s*(?=\d)", "", s)                  # a leading currency sign or code
    s = re.sub(r"(?i)\s*(?:%|[^\d.,\s]{1,4}|\s)$", "", s)                  # a trailing %, unit or currency code
    s = s.replace("\u00a0", " ").replace("'", "")
    if re.fullmatch(r"\d{1,3}(?: \d{3})+(?:[.,]\d+)?", s):
        s = s.replace(" ", "")
    if not re.fullmatch(r"\d[\d.,]*|[.,]\d+", s):
        return None
    if "." in s and "," in s:                                              # the later mark is the decimal point
        point = "." if s.rfind(".") > s.rfind(",") else ","
        s = s.replace("," if point == "." else ".", "").replace(point, ".")
    elif "," in s:
        s = s.replace(",", "") if re.fullmatch(r"\d{1,3}(?:,\d{3})+", s) else s.replace(",", ".")
    elif s.count(".") > 1:
        if not re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):
            return None
        s = s.replace(".", "")
    try:
        value = float(s)
    except ValueError:
        return None
    return -value if negative else value


def _year(text: str) -> int:
    year = int(text)
    return year if len(text) == 4 else (2000 + year if year <= 69 else 1900 + year)


def parse_dates(text: Any) -> list[tuple[int | None, int, int]]:
    """Every (year, month, day) a date string can mean (a day/month order that is not clear gives both). A date
    without a year has `None` for it. Empty when the text is not a date."""
    s = fold(text).strip()
    s = re.sub(r"\b(mon|tue|wed|thu|fri|sat|sun|montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag)[a-z]*\.?,?\s*", "", s)
    s = re.sub(r"(?<=\d)(st|nd|rd|th|er|e)\b", "", s)
    s = re.sub(r"[,\s]+", " ", s.replace("\u2013", "-")).strip()
    out: list[tuple[int | None, int, int]] = []

    def add(year, month, day):
        if 1 <= month <= 12 and 1 <= day <= 31 and (year is None or 1000 <= year <= 2999):
            out.append((year, month, day))

    if m := re.fullmatch(r"(\d{4})[-/. ](\d{1,2})[-/. ](\d{1,2})(?:[t ]\d{1,2}:\d{2}.*)?", s):
        add(int(m[1]), int(m[2]), int(m[3]))
    elif m := re.fullmatch(r"(\d{1,2})[-/. ](\d{1,2})[-/. ](\d{4}|\d{2})", s):
        add(_year(m[3]), int(m[2]), int(m[1]))                             # day first (the order outside the US)
        add(_year(m[3]), int(m[1]), int(m[2]))
    elif m := re.fullmatch(r"(\d{4})(\d{2})(\d{2})", s):
        add(int(m[1]), int(m[2]), int(m[3]))
    elif m := re.fullmatch(r"(\d{1,2})\.? ?-?([a-z]+)\.?[ -]?(\d{4}|\d{2})", s):
        if m[2] in MONTHS:
            add(_year(m[3]), MONTHS[m[2]], int(m[1]))
    elif m := re.fullmatch(r"([a-z]+)\.? (\d{1,2}) (\d{4}|\d{2})", s):
        if m[1] in MONTHS:
            add(_year(m[3]), MONTHS[m[1]], int(m[2]))
    elif m := re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})", s):                  # no year: `05-05`
        add(None, int(m[2]), int(m[1]))
        add(None, int(m[1]), int(m[2]))
    elif m := re.fullmatch(r"(\d{1,2})\.? ?([a-z]+)\.?", s):
        if m[2] in MONTHS:
            add(None, MONTHS[m[2]], int(m[1]))
    elif m := re.fullmatch(r"([a-z]+)\.? (\d{1,2})", s):
        if m[1] in MONTHS:
            add(None, MONTHS[m[1]], int(m[2]))
    return out


def parse_time(text: Any) -> int | None:
    """Seconds since midnight for `19:30`, `7:30pm`, `1: 45 P. M.`, `19.30 Uhr`; None when it is not a time."""
    s = re.sub(r"\s+", "", fold(text))
    m = re.fullmatch(r"(\d{1,2})[:.h](\d{2})(?:[:.](\d{2}))?(?:([ap])\.?m\.?)?(?:uhr)?", s)
    if not m:
        return None
    hour, minute, second = int(m[1]), int(m[2]), int(m[3] or 0)
    if m[4]:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if m[4] == "p" else 0)
    return hour * 3600 + minute * 60 + second if hour < 24 and minute < 60 and second < 60 else None


def _dates_agree(a: list, b: list) -> bool:
    for (ya, ma, da) in a:
        for (yb, mb, db) in b:
            if (ma, da) == (mb, db) and (ya is None or yb is None or ya == yb):
                return True
    return False


def _token_f1(a: str, b: str) -> float:
    """F1 over the words of two strings, order ignored: `Private Stock, Carlton` against `Carlton` is 0.67."""
    x, y = Counter(re.findall(r"\w+", fold(a))), Counter(re.findall(r"\w+", fold(b)))
    hits = sum((x & y).values())
    if not hits:
        return 0.0
    precision, recall = hits / sum(x.values()), hits / sum(y.values())
    return 2 * precision * recall / (precision + recall)


def value_credit(predicted: str, reference: str) -> float:
    """0..1 for one value. 1 when both mean the same thing: the same text up to case, accents, spacing and
    punctuation, the same number (`7,569,000.00` = `7569000`), the same date (`30/06/2017` = `2017-06-30` =
    `June 30, 2017`), the same time (`7:30pm` = `19:30`), the same yes/no, or both empty (`null` = `` = `N/A`).
    Otherwise the character similarity when it is close (>= SOFT_THRESHOLD), or the word overlap when the value has
    several words and enough of them are right; else 0. A read-off error of one character is not a wrong answer,
    a different value is."""
    if predicted == reference:
        return 1.0
    p_null, r_null = is_null(predicted), is_null(reference)
    if p_null or r_null:
        return 1.0 if p_null and r_null else 0.0
    a, b = squash(predicted), squash(reference)
    if a and a == b:
        return 1.0
    if not a or not b:                                                     # only symbols (a checkbox): exact compare
        return 1.0 if normalise(predicted) == normalise(reference) else 0.0
    x, y = parse_number(predicted), parse_number(reference)
    if x is not None and y is not None:
        return 1.0 if abs(x - y) <= 1e-9 * max(1.0, abs(y)) else 0.0
    p_date, r_date = parse_dates(predicted), parse_dates(reference)
    if p_date and r_date:
        return 1.0 if _dates_agree(p_date, r_date) else 0.0
    p_time, r_time = parse_time(predicted), parse_time(reference)
    if p_time is not None and r_time is not None:
        return 1.0 if p_time == r_time or (p_time % 60 == 0 or r_time % 60 == 0) and p_time // 60 == r_time // 60 else 0.0
    p_word, r_word = normalise(predicted), normalise(reference)
    if (p_word in TRUE_WORDS | FALSE_WORDS) and (r_word in TRUE_WORDS | FALSE_WORDS):
        return 1.0 if (p_word in TRUE_WORDS) == (r_word in TRUE_WORDS) else 0.0
    if x is not None or y is not None or p_date or r_date:                 # a number / date against something else
        return 0.0
    credit = 0.0
    ratio = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    if ratio >= SOFT_THRESHOLD:
        credit = ratio
    if len(re.findall(r"\w+", reference)) > 1 or len(re.findall(r"\w+", predicted)) > 1:
        overlap = _token_f1(predicted, reference)
        if overlap >= TOKEN_THRESHOLD:
            credit = max(credit, overlap)
    return round(credit, 4)


def _match_values(predicted: dict[str, str], reference: dict[str, str]) -> dict[str, str]:
    """A key the model left out where the reference is null is a correct null, not a miss."""
    filled = dict(predicted)
    for path, value in reference.items():
        if path not in filled and is_null(value):
            filled[path] = ""
    return filled


def _strict_key(value: str) -> str:
    number = parse_number(value) if value else None
    return f"#{number!r}" if number is not None else ("" if is_null(value) else value)


def soft_f1(predicted: dict[str, str], reference: dict[str, str]) -> float:
    """F1 over the (path, value) pairs where a value counts by its `value_credit` instead of all or nothing."""
    if not predicted and not reference:
        return 1.0
    credit = sum(value_credit(value, reference[path]) for path, value in predicted.items() if path in reference)
    if not credit:
        return 0.0
    precision, recall = credit / len(predicted), credit / len(reference)
    return 2 * precision * recall / (precision + recall)


def score_example(prediction: str, reference: str) -> dict[str, Any]:
    """Metrics of one answer. `kind` says how it was judged; `score` is the number to average. For JSON `score` is
    the tolerant `value_f1` (spacing, punctuation and one-character slips forgiven); `strict_f1` is the exact one."""
    ref_json = extract_json(reference) if reference.lstrip()[:1] in "{[" else None
    if ref_json is not None:
        pred_json = extract_json(prediction)
        valid = pred_json is not None
        ref_pairs = flatten(ref_json)
        pred_pairs = _match_values(flatten(align(pred_json, ref_json)), ref_pairs) if valid else {}
        strict = _f1({(k, _strict_key(v)) for k, v in pred_pairs.items()},
                     {(k, _strict_key(v)) for k, v in ref_pairs.items()})
        value_f1 = soft_f1(pred_pairs, ref_pairs)
        path_f1 = _f1(set(pred_pairs), set(ref_pairs))
        null_paths = [path for path, value in ref_pairs.items() if is_null(value)]
        invented = sum(1 for path in null_paths if not is_null(pred_pairs.get(path, "")))
        return {"kind": "json", "json_valid": valid, "path_f1": round(path_f1, 4), "value_f1": round(value_f1, 4),
                "strict_f1": round(strict, 4), "_answer": pred_json, "hallucination": round(invented / len(null_paths), 4) if null_paths else 0.0,
                "score": round(value_f1, 4)}
    if len(reference.split()) <= SHORT_ANSWER_WORDS:
        strict = normalise(prediction) == normalise(reference)
        match = strict or value_credit(prediction, reference) == 1.0       # same number / date / time / spelling
        return {"kind": "short", "exact_match": strict, "score": float(match)}
    ratio = difflib.SequenceMatcher(None, normalise(prediction), normalise(reference), autojunk=False).ratio()
    return {"kind": "text", "similarity": round(ratio, 4), "score": round(ratio, 4)}


def _has_id(items: list) -> bool:
    return any(isinstance(i, dict) and any(isinstance(i.get(k), (str, int)) and str(i[k]).strip() for k in ID_KEYS)
               for i in items)


def align(predicted: Any, reference: Any) -> Any:
    """The prediction with every list that has no ids (line items, table rows) put in the reference's order: each
    predicted item is paired with the reference item it matches best, so `[b, a]` against `[a, b]` is not wrong.
    Items nothing matches go after the reference's, and a reference item nothing matches leaves a hole."""
    if isinstance(predicted, dict) and isinstance(reference, dict):
        return {k: align(v, reference[k]) if k in reference else v for k, v in predicted.items()}
    if not (isinstance(predicted, list) and isinstance(reference, list)) or not reference or _has_id(reference):
        return predicted
    scored = []
    for i, p_item in enumerate(predicted):
        p_flat = flatten(align(p_item, reference[min(i, len(reference) - 1)]))
        for j, r_item in enumerate(reference):
            scored.append((soft_f1(p_flat, flatten(r_item)), -abs(i - j), i, j))
    placed: dict[int, Any] = {}
    used: set[int] = set()
    for credit, _, i, j in sorted(scored, reverse=True):
        if credit > 0 and i not in used and j not in placed:
            placed[j] = align(predicted[i], reference[j])
            used.add(i)
    leftover = [align(item, reference[0]) if isinstance(item, (dict, list)) else item
                for i, item in enumerate(predicted) if i not in used]
    # unmatched predicted items fill the reference's holes in order, then go after it
    out = [placed.get(j, GAP) for j in range(len(reference))]
    for j in range(len(reference)):
        if out[j] is GAP and leftover:
            out[j] = leftover.pop(0)
    return out + leftover


def schema_from_prompt(prompt: str) -> dict | None:
    """The JSON Schema the prompt asks the answer to follow (the object after `JSON Schema:`), if there is one."""
    marker = prompt.find("JSON Schema:")
    start = prompt.find("{", marker) if marker != -1 else -1
    if start == -1:
        return None
    try:
        return json.JSONDecoder().raw_decode(prompt[start:])[0]
    except ValueError:
        return None


def schema_score(answer: Any, schema: dict | None) -> float:
    """1.0 when the answer validates against the schema (types, required keys, no extra keys), less for every
    violation. A number written as a string, a missing key or an invented one are all violations."""
    if schema is None:
        return 1.0
    from jsonschema import Draft202012Validator
    try:
        errors = sum(1 for _ in Draft202012Validator(schema).iter_errors(answer))
    except Exception:                                        # a schema the validator cannot read: do not punish
        return 1.0
    return 1.0 if not errors else 1.0 / (1.0 + errors)


# ---- reward for RL training -------------------------------------------------------------------------

# reward = PARSES + SCHEMA * (answer follows the prompt's JSON Schema) + CONTENT * (fields right, minus invention).
# Invalid JSON is 0 (a gate, as in the structured-output RL papers); content carries most of the weight.
PARSES, SCHEMA, CONTENT = 0.05, 0.10, 0.85
HALLUCINATION_PENALTY = 0.5    # taken off for every reference-null field the model filled in, scaled by their share
YEAR_RANGE = (1900, 2035)


def suspicious_reference(reference: str) -> bool:
    """True when a reference value cannot be right (a date in year 2094, a month 13): drop such examples from RL
    training, where the policy would be pushed toward the error. Checks only values that parse as a full date."""
    ref_json = extract_json(reference) if reference.lstrip()[:1] in "{[" else None
    if ref_json is None:
        return False
    for value in flatten(ref_json).values():
        for year, _, _ in parse_dates(value)[:1]:
            if year is not None and not YEAR_RANGE[0] <= year <= YEAR_RANGE[1]:
                return True
    return False


def reward(prediction: str, reference: str, prompt: str | None = None) -> float:
    """Scalar in [0, 1] for RL, built for schema extraction. 0 when the answer is not JSON (or cut off); otherwise
    PARSES + SCHEMA * schema compliance (the schema is read from `prompt` when given) + CONTENT * the tolerant
    field score, minus a penalty for inventing values where the reference is null. Lists without ids (line items,
    rows) are matched by content, not position. Not JSON answers (short, OCR text) score as in `score_example`.
    Keep tracking `strict_f1` next to it: if this rises and strict stays flat, the policy is exploiting the tolerance."""
    result = score_example(prediction, reference)
    answer = result.pop("_answer", None)
    if result["kind"] != "json":
        return float(result["score"])
    if not result["json_valid"] or not isinstance(answer, (dict, list)):
        return 0.0
    content = max(0.0, min(1.0, result["value_f1"] - HALLUCINATION_PENALTY * result["hallucination"]))
    compliance = schema_score(answer, schema_from_prompt(prompt) if prompt else None)
    return round(PARSES + SCHEMA * compliance + CONTENT * content, 4)


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean score overall and per task, and the share of JSON answers that parse."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_task[row["task"]].append(row)

    def mean(rows: list[dict], key: str) -> float | None:
        values = [r[key] for r in rows if key in r]
        return round(sum(float(v) for v in values) / len(values), 4) if values else None

    out: dict[str, Any] = {"examples": len(results), "score": mean(results, "score"),
                           "json_valid": mean(results, "json_valid"), "per_task": {}}
    for task, rows in sorted(by_task.items()):
        out["per_task"][task] = {"n": len(rows), "score": mean(rows, "score"), "json_valid": mean(rows, "json_valid"),
                                 "path_f1": mean(rows, "path_f1"), "strict_f1": mean(rows, "strict_f1"), "hallucination": mean(rows, "hallucination"), "kind": rows[0]["kind"]}
    return out


# ---- generation -------------------------------------------------------------------------------------

class GenerationEvaluator:
    """Generates answers for a fixed set of validation examples and scores them.

    `run(model, step)` is collective under torchrun: every rank generates its share and rank 0 gets everything.
    """

    def __init__(self, root: str | Path, transform, adapter: Callable[[dict], dict] | None, processor=None,
                 split: str = "val", samples_per_task: int = 2, max_new_tokens: int = 2048, batch_size: int = 4,
                 max_documents: int | None = None, precision: str = "auto", out_dir: str | Path | None = None,
                 draws_per_document: int = 6, preset: str = "structured"):
        from .dataset import DocumentDataset
        from .processors import DatasetProcessor

        self.transform, self.adapter = transform, adapter
        self.max_new_tokens, self.batch_size, self.precision = max_new_tokens, batch_size, precision
        self.out_dir = Path(out_dir) if out_dir else None
        dataset = DocumentDataset(root, split, processor or DatasetProcessor(preset=preset),
                                  samples_per_document=draws_per_document, max_documents=max_documents,
                                  fits=transform.fits)
        taken: dict[str, int] = defaultdict(int)
        self.skipped_long = 0
        self.items: list[dict] = []
        for index in range(len(dataset)):
            item = dataset[index]
            if taken[item["task"]] >= samples_per_task:
                continue
            if len(transform.tokenizer.encode(item["label"], add_special_tokens=False)) + 16 > max_new_tokens:
                self.skipped_long += 1                       # the reference is longer than the room to generate
                continue
            taken[item["task"]] += 1
            self.items.append(item)
        # similar lengths together: a batch ends when its longest answer does
        self.items.sort(key=lambda it: len(it["label"]))
        log.info("generation check: %d examples (%s), %d skipped as longer than %d new tokens",
                 len(self.items), ", ".join(f"{t}:{n}" for t, n in sorted(taken.items())), self.skipped_long,
                 max_new_tokens)

    # -- one batch -------------------------------------------------------------------------------
    def _inputs(self, chunk: list[dict]) -> tuple[dict[str, Any], int, int]:
        features = [self.transform(item["images"], item["prompt"], item["label"]) for item in chunk]
        prompt_lengths = [int((f["labels"] != -100).nonzero()[0]) for f in features]   # where the answer starts
        batch = self.transform.make_collator()(features)
        kwargs = self.adapter(batch) if self.adapter is not None else dict(batch)
        pad = self.transform._pad_token_id()
        width = max(prompt_lengths)
        input_ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        attention = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, (feature, length) in enumerate(zip(features, prompt_lengths)):      # left padding for generation
            input_ids[row, width - length:] = feature["input_ids"][:length]
            attention[row, width - length:] = 1
        kwargs.pop("labels", None)
        kwargs.update(input_ids=input_ids, attention_mask=attention)
        longest_answer = max(len(self.transform.tokenizer.encode(item["label"], add_special_tokens=False))
                             for item in chunk)
        return kwargs, width, min(self.max_new_tokens, int(longest_answer * 1.25) + 16)

    @torch.no_grad()
    def generate(self, model, items: list[dict]) -> list[str]:
        from . import distributed as dd
        from .processors.base import to_device
        from .processors.deepseekOcr import EOS_ID

        inner = dd.unwrap(model)
        device = dd.model_device(model)
        amp_dtype, _ = dd.resolve_precision(self.precision, device)
        was_training = model.training
        model.eval()
        answers: list[str] = []
        try:
            for start in range(0, len(items), self.batch_size):
                kwargs, width, budget = self._inputs(items[start:start + self.batch_size])
                kwargs = to_device(kwargs, device)
                with dd.autocast(device, amp_dtype):
                    out = inner.base.generate(**kwargs, max_new_tokens=budget, do_sample=False, use_cache=True,
                                              eos_token_id=EOS_ID, pad_token_id=self.transform._pad_token_id())
                for row in out[:, width:].tolist():
                    row = row[:row.index(EOS_ID)] if EOS_ID in row else row   # up to the end-of-text token
                    answers.append(self.transform.tokenizer.decode(row, skip_special_tokens=True))
        finally:
            model.train(was_training)
        return answers

    # -- the whole check -------------------------------------------------------------------------
    def run(self, model, step: int = 0, logger=None) -> dict[str, Any]:
        import torch.distributed as torch_dist
        from . import distributed as dd

        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()                      # give the training run's cached blocks back for the KV cache
        share = self.items[dd.rank()::dd.world_size()]
        answers = self.generate(model, share) if share else []
        rows = []
        for item, answer in zip(share, answers):
            rows.append({"id": item["id"], "task": item["task"], "reference": item["label"], "prediction": answer,
                         "prompt": item["prompt"], "images": item.get("image_paths", []),
                         **{k: v for k, v in score_example(answer, item["label"]).items() if k != "_answer"}})
        if dd.is_dist():
            gathered: list[Any] = [None] * dd.world_size()
            torch_dist.all_gather_object(gathered, rows)
            rows = [row for part in gathered for row in part]
        summary = summarise(rows)
        if dd.is_main():
            if self.out_dir is not None:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                path = self.out_dir / f"generations_step_{step:07d}.jsonl"
                path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
            scalars = {"gen/score": summary["score"] or 0.0, "gen/json_valid": summary["json_valid"] or 0.0}
            scalars.update({f"gen/{task}": info["score"] for task, info in summary["per_task"].items()})
            if logger is not None:
                logger.scalars(step, scalars)
            log.info("generation check at step %d: score %.3f, json valid %s, %d examples, %.0f s", step,
                     summary["score"] or 0.0, summary["json_valid"], summary["examples"],
                     time.perf_counter() - started)
            for task, info in summary["per_task"].items():
                log.info("  %-18s n=%d score=%.3f%s", task, info["n"], info["score"] or 0.0,
                         "" if info["json_valid"] is None else f" json_valid={info['json_valid']:.2f}")
        return summary


# ---- the command: score a saved checkpoint --------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    from corpus.logs import configure

    from . import distributed as dd
    from .hub import resolve_resume
    from .model import DeepSeekOCR
    from .processors import batch_adapter_for
    from .train import build_transform, parse_args, weights_dtype

    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--checkpoint", help="a .pt checkpoint or hf://user/repo/last/training_state.pt")
    own.add_argument("--no-adapter", action="store_true", help="score the base model without the adapter")
    own.add_argument("--samples-per-task", type=int, default=3)
    own.add_argument("--max-new-tokens", type=int, default=2048)
    own.add_argument("--gen-batch-size", type=int, default=4)
    own.add_argument("--split", default="val")
    known, rest = own.parse_known_args(argv)
    if any(flag in (rest or []) for flag in ("-h", "--help")):
        print(__doc__)
        print("options of this command: --checkpoint FILE | --no-adapter, --samples-per-task N, --max-new-tokens N, "
              "--gen-batch-size N, --split NAME; every other option is the one of `python -m train.train` "
              "(use the same --config as the run)\n")
    args = parse_args(rest)
    configure(args)
    if not known.checkpoint and not known.no_adapter:
        raise SystemExit("give --checkpoint <file>, or --no-adapter to score the base model")

    device = dd.init_distributed()
    dtype, _ = dd.resolve_precision(args.precision, device)
    transform = build_transform(args)
    log.info("loading %s (%s)", args.model_id, str(weights_dtype(args, dtype)).removeprefix("torch."))
    model = DeepSeekOCR.from_pretrained(args.model_id, lang_trained=False, vision_trained=False,
                                        dtype=weights_dtype(args, dtype), device_map={"": str(device)})
    step = 0
    if not known.no_adapter:
        if args.lora:
            model.add_lora(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        meta = model.load_checkpoint(resolve_resume(known.checkpoint, args.hub_token), map_location="cpu")
        step = int(meta.get("step") or 0)
        log.info("loaded the checkpoint from step %d", step)
    evaluator = GenerationEvaluator(
        args.root, transform, batch_adapter_for(args.model_id), split=known.split,
        samples_per_task=known.samples_per_task, max_new_tokens=known.max_new_tokens,
        batch_size=known.gen_batch_size, preset=args.preset, max_documents=args.val_max_documents, precision=args.precision,
        out_dir=Path(args.out_dir) / ("generations_base" if known.no_adapter else "generations"))
    summary = evaluator.run(model, step)
    print(json.dumps(summary, indent=2))
    dd.cleanup()


if __name__ == "__main__":
    main()
