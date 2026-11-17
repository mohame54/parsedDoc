"""Deterministic checks. A document is accepted only if every check passes; otherwise it is excluded with reasons."""
from __future__ import annotations

import math

import difflib
import re

from jsonschema import Draft202012Validator

from .schemas import (
    STAGE1_RESPONSE_SCHEMA,
    STAGE1_V2_RESPONSE_SCHEMA,
    STAGE2_RESPONSE_SCHEMA,
    STAGE2_V3_RESPONSE_SCHEMA,
    stage2_schema,
    CHART_TYPES,
    TREND_CHART_TYPES,
)

BLOCKING_STATUSES = {"uncertain", "illegible"}
ABSENT_STATUSES = {"not_present", "not_applicable"}

# Markup the labeller itself adds to text: [x], [ ], [signature], [handwritten: ...].
_MARKUP = re.compile(r"\[handwritten:\s*|\[signature\]|\[x\]|\[ \]|\]", re.IGNORECASE)
_EDGE_PUNCTUATION = ".,;:!?()[]{}\"'`"


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def strip_markup(s: str) -> str:
    return _MARKUP.sub(" ", s)


def core_text(s: str) -> str:
    """Letters and digits only, lowercase, markup removed. What is left when only real characters count."""
    return re.sub(r"[^a-z0-9]+", "", strip_markup(s).lower())


def _tokens(s: str) -> list[str]:
    """Words for in-source matching: markup removed, lowercase, edge punctuation stripped."""
    words = (word.strip(_EDGE_PUNCTUATION) for word in strip_markup(s).lower().split())
    return [word for word in words if word]


def value_in_source(raw: str, source_text: str, max_gap: int = 0) -> bool:
    """True when the value is found in the source text.

    First an exact match on normalised text. Failing that, every word of the value must appear in the
    source in the same order, with at most max_gap other words between consecutive matches. This accepts
    values that OCR split across columns or that our markup interrupts, and still rejects any value
    containing a word that is not in the source.
    """
    if normalize_text(raw) in normalize_text(source_text):
        return True
    wanted = _tokens(raw)
    source = _tokens(source_text)
    if not wanted:
        return False
    for start, word in enumerate(source):
        if word != wanted[0]:
            continue
        position = start
        for target in wanted[1:]:
            window = source[position + 1:position + 2 + max_gap]
            if target not in window:
                break
            position = position + 1 + window.index(target)
        else:
            return True
    return False


def _schema_errors(instance: object, schema: dict) -> list[str]:
    errors = sorted(Draft202012Validator(schema).iter_errors(instance), key=lambda e: list(e.path))
    return [f"schema:{'/'.join(str(p) for p in e.path) or '$'}:{e.message[:120]}" for e in errors[:5]]


def validate_stage1(obj: object, page_count: int, max_illegible_regions: int) -> list[str]:
    errors = _schema_errors(obj, STAGE1_RESPONSE_SCHEMA)
    if errors:
        return errors
    reasons: list[str] = []
    if not obj["usable"]:
        reasons.append("unusable_document")
    pages = obj["pages"]
    if len(pages) != page_count:
        reasons.append(f"page_count_mismatch:{len(pages)}!={page_count}")
    for index, page in enumerate(pages, start=1):
        if page["page"] != index:
            reasons.append(f"page_order:{index}")
        if page["legibility"] == "unreadable":
            reasons.append(f"unreadable_page:{index}")
        if not page["text"].strip():
            reasons.append(f"empty_page:{index}")
    if obj["illegible_regions"] > max_illegible_regions:
        reasons.append(f"illegible_regions:{obj['illegible_regions']}")
    return reasons


def validate_fields(fields: list[dict], source_text: str, max_gap: int = 0) -> list[str]:
    """Checks that apply to native and Gemini-produced fields alike.

    The hallucination guard: every extracted raw_value must be found in the source text
    (see value_in_source for how column splits and markup are tolerated).
    """
    if not fields:
        return ["no_fields"]
    reasons: list[str] = []
    seen: set[str] = set()
    for field in fields:
        name, status, raw = field["name"], field["status"], field["raw_value"]
        if name in seen:
            reasons.append(f"duplicate_field:{name}")
        seen.add(name)
        if status in BLOCKING_STATUSES:
            reasons.append(f"{status}:{name}")
        elif status == "extracted":
            if not raw.strip():
                reasons.append(f"empty_extracted:{name}")
            elif not value_in_source(raw, source_text, max_gap):
                reasons.append(f"not_in_source:{name}")
        elif status in ABSENT_STATUSES and raw.strip():
            reasons.append(f"value_for_absent_field:{name}")
        if field.get("confidence") == "low":
            reasons.append(f"low_confidence:{name}")
    return reasons


def validate_fill(fields: list[dict], min_ratio: float, min_fields: int) -> list[str]:
    """Backstop for blank forms that have no fillable widgets: too few extracted values means blank.

    Small documents are exempt (min_fields), since a short form can be legitimately sparse.
    """
    if len(fields) < min_fields:
        return []
    extracted = sum(1 for f in fields if f["status"] == "extracted")
    if extracted / len(fields) < min_ratio:
        return [f"blank_form:extracted={extracted}/{len(fields)}"]
    return []


MAX_DESCRIPTION_CHARS = 200


def clean_descriptions(fields: list[dict]) -> int:
    """Blanks field descriptions that cannot be trusted, in place; returns how many were blanked. A blank
    description falls back to a template downstream, so a bad one never costs the document. Blanked: empty,
    too long, or containing the field's own value (a description must describe the field, not leak the data)."""
    blanked = 0
    for field in fields:
        text = " ".join(str(field.get("description", "")).split())
        low = text.lower()
        values = [str(field.get(k) or "").strip().lower() for k in ("raw_value", "normalized_value")]
        leaks = any(len(v) >= 3 and v in low for v in values)
        if not text or len(text) > MAX_DESCRIPTION_CHARS or leaks:
            blanked += bool(field.get("description"))
            field["description"] = ""
        else:
            field["description"] = text
    return blanked


def validate_stage2(obj: object, source_text: str, min_extracted_ratio: float = 0.0,
                    min_fields: int = 0, max_value_gap: int = 0, described: bool = False) -> list[str]:
    errors = _schema_errors(obj, stage2_schema(False, described))
    if errors:
        return errors
    return (validate_fields(obj["fields"], source_text, max_value_gap)
            + validate_fill(obj["fields"], min_extracted_ratio, min_fields))


def validate_stage1_v2(obj: object, ocr_pages: list[dict], page_count: int, max_illegible_regions: int,
                       exclude_blank_forms: bool,
                       max_change_ratio: float
                       ) -> tuple[list[str], list[str], dict[str, float], list[list[dict]]]:
    """Checks the Stage 1 v2 answer and applies its corrections to the OCR draft.

    Returns (reasons, page_texts, metrics, lines). page_texts and lines are empty when reasons is non-empty.
    lines is the corrected draft per page as [{"id", "text"}], keeping the OCR line ids (added lines get
    ids like "added-0"). metrics holds correction_ratio (share of draft lines touched) and change_ratio
    (share of draft characters that really changed, ignoring markup, punctuation and case). The gate uses
    change_ratio.
    """
    metrics = {"correction_ratio": 0.0, "change_ratio": 0.0}
    errors = _schema_errors(obj, STAGE1_V2_RESPONSE_SCHEMA)
    if errors:
        return errors, [], metrics, []

    reasons: list[str] = []
    if not obj["usable"]:
        reasons.append("unusable_document")
    if exclude_blank_forms and obj["fill_level"] == "blank":
        reasons.append("blank_form:stage1")
    if len(obj["pages"]) != page_count or len(ocr_pages) != page_count:
        reasons.append(f"page_count_mismatch:{len(obj['pages'])}!={page_count}")
    for index, page in enumerate(obj["pages"], start=1):
        if page["page"] != index:
            reasons.append(f"page_order:{index}")
        if page["legibility"] == "unreadable":
            reasons.append(f"unreadable_page:{index}")
    if obj["illegible_regions"] > max_illegible_regions:
        reasons.append(f"illegible_regions:{obj['illegible_regions']}")

    # Working copy of the draft: each line is {"id", "text"}; corrections change text, missed adds lines.
    pages = [[{"id": line["id"], "text": line["text"]} for line in page["lines"]] for page in ocr_pages]
    by_id = {line["id"]: line for page in pages for line in page}
    original = {line_id: line["text"] for line_id, line in by_id.items()}
    draft_count = len(by_id)
    draft_chars = sum(len(core_text(text)) for text in original.values())

    changed_chars = 0
    for correction in obj["corrections"]:
        target = by_id.get(correction["line"])
        if target is None:
            reasons.append(f"unknown_line:{correction['line']}")
            continue
        old, new = core_text(original[correction["line"]]), core_text(correction["text"])
        if old != new:
            matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
            changed_chars += max(len(old), len(new)) - sum(block.size for block in matcher.get_matching_blocks())
        target["text"] = correction["text"].strip()

    added = 0
    for missed in obj["missed"]:
        after = missed["after_line"]
        placed = False
        for page in pages:
            positions = [i for i, line in enumerate(page) if line["id"] == after]
            if after and positions:
                page.insert(positions[0] + 1, {"id": f"added-{added}", "text": missed["text"].strip()})
                placed = True
                break
        if not placed and after == "" and pages:
            pages[0].insert(0, {"id": f"added-{added}", "text": missed["text"].strip()})
            placed = True
        if not placed:
            reasons.append(f"unknown_line:{after}")
        else:
            changed_chars += len(core_text(missed["text"]))
        added += 1

    metrics["correction_ratio"] = len(obj["corrections"]) / max(draft_count, 1)
    metrics["change_ratio"] = changed_chars / max(draft_chars, 1)
    if metrics["change_ratio"] > max_change_ratio:
        reasons.append(f"too_many_changes:{metrics['change_ratio']:.2f}")
    if reasons:
        return reasons, [], metrics, []
    page_texts = ["\n".join(line["text"] for line in page if line["text"]) for page in pages]
    return [], page_texts, metrics, [[line for line in page if line["text"]] for page in pages]


TREND_TOLERANCE = 0.05   # a step within 5% of the largest value counts as flat


def trend_of(figure: dict) -> str:
    """The trend of a chart, recomputed from its points. The first series with two or more points is the main one;
    a step up or down beyond the tolerance sets the direction, and both directions make it mixed."""
    if figure.get("figure_type") not in TREND_CHART_TYPES:
        return "not_applicable"
    for series in figure.get("series") or []:
        values = [p["value"] for p in series.get("points") or [] if math.isfinite(p["value"])]
        if len(values) >= 2:
            break
    else:
        return "not_applicable"
    tolerance = TREND_TOLERANCE * max(max(abs(v) for v in values), 1e-9)
    steps = [b - a for a, b in zip(values, values[1:])]
    up = any(step > tolerance for step in steps)
    down = any(step < -tolerance for step in steps)
    if up and down:
        return "mixed"
    if up:
        return "increasing"
    if down:
        return "decreasing"
    return "stable"


def check_figures(figures: list[dict]) -> list[str]:
    """Chart figures need data; values must be finite. The trend is recomputed locally and corrected where it
    disagrees with the points (the figure is marked trend_corrected, so the correction is visible in the data)."""
    reasons = []
    for figure in figures:
        points = [p for series in figure["series"] for p in series["points"]]
        if figure["figure_type"] in CHART_TYPES and not points:
            reasons.append(f"chart_without_data:{figure['figure_id']}")
        if any(not math.isfinite(p["value"]) for p in points):
            reasons.append(f"bad_value:{figure['figure_id']}")
        recomputed = trend_of(figure)
        if figure["trend"] != recomputed:
            figure["trend"] = recomputed
            figure["trend_corrected"] = True
    return reasons


def validate_stage2_v3(obj: object, source_text: str, line_ids: set[str] | None, page_count: int,
                       min_extracted_ratio: float = 0.0, min_fields: int = 0, max_value_gap: int = 0,
                       described: bool = False, figures: bool = False) -> list[str]:
    """Checks a Stage 2 v3 answer. Text must come from the source; ids and references must be consistent.

    Visual properties (colors, highlighting, checkbox state) cannot be checked locally; they are kept as
    Gemini reported them. line_ids is the set of transcription line ids Gemini may cite, or None when the
    transcription had no ids (then evidence is ignored rather than checked).
    """
    errors = _schema_errors(obj, stage2_schema(True, described, figures))
    if errors:
        return errors
    reasons = validate_fields(obj["fields"], source_text, max_value_gap)
    reasons += validate_fill(obj["fields"], min_extracted_ratio, min_fields)
    if figures:
        reasons += check_figures(obj["figures"])

    known_ids: dict[str, str] = {}   # id -> kind

    def register(identifier: str, kind: str) -> None:
        if identifier in known_ids:
            reasons.append(f"duplicate_id:{identifier}")
        known_ids[identifier] = kind

    for field in obj["fields"]:
        register(field["name"], "field")
    for table in obj["tables"]:
        register(table["table_id"], "table")
        for row in table["rows"]:
            register(row["row_id"], "row")
            for cell in row["cells"]:
                if cell["column"] < 1 or cell["row_span"] < 1 or cell["col_span"] < 1:
                    reasons.append(f"bad_cell:{row['row_id']}")
                if cell["text"].strip() and not value_in_source(cell["text"], source_text, max_value_gap):
                    reasons.append(f"table_not_in_source:{table['table_id']}.{row['row_id']}")
    for figure in obj.get("figures") or []:
        register(figure["figure_id"], "figure")
    for element in obj["visual_elements"]:
        register(element["element_id"], "element")
    for item in obj["layout"]:
        register(item["layout_id"], "layout")

    for relationship in obj["relationships"]:
        for end in ("from_id", "to_id"):
            if relationship[end] not in known_ids:
                reasons.append(f"unknown_reference:{relationship[end]}")

    components = [obj["fields"], obj["tables"], obj["visual_elements"], obj["layout"]]
    if figures:
        components.append(obj["figures"])
    for component in components:
        for item in component:
            if not 1 <= item["page"] <= max(page_count, 1):
                reasons.append(f"bad_page:{item['page']}")
            if line_ids is not None:
                reasons += [f"unknown_evidence:{line}" for line in item["evidence"] if line not in line_ids]
    return reasons
