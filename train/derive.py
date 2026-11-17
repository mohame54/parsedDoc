"""Turns one labelled document into many (prompt, answer) training examples.

Each task is a function record -> list[(task_name, prompt_context, answer)] registered in TASKS, and
each has several paraphrased prompts so the model does not overfit to one wording. Everything is
derived from the labels already on disk; no model is called.

To add a task: write `def _my_task(record, rng) -> list[tuple[dict, str]]` returning (format args, answer)
pairs, add its prompt paraphrases to PROMPTS, and register it in TASKS.
"""
from __future__ import annotations

import json
import random
from typing import Callable

from .describe import Describer
from labeller.validate import trend_of  # noqa: E402
from .render import (EXTRACT_SCHEMA, FIGURES_SCHEMA, FORMAT_LABEL, OUTPUT_RULES, SCHEMA_RULES, VISUAL_SCHEMA, VISUAL_TYPES,
                     render_schema, serialise, values)

PROMPTS: dict[str, list[str]] = {
    "ocr": [
        "Free OCR.",   # DeepSeek-OCR's own prompt, so its pre-training prompt stays in the mix
        "Extract the text from this page.",
        "What does this page say? Write out all the text.",
        "Transcribe all text on this page exactly as written, in reading order.",
        "Read the page and output its full text.",
        "OCR this document page. Keep the original line order.",
    ],
    # Canonical serving wording, with the JSON Schema stated (see "extract_to_schema" below for why).
    "extract": [
        "Extract the structured data from this document as a JSON object that conforms to the JSON Schema below. "
        "Group the filled fields by section and leave out fields that are missing or unreadable. "
        "Return only the JSON object.\n\nJSON Schema:\n{schema}",
    ],
    "classify": [
        "Which kind of document is this? Reply with one label.",
        "Give the document category.",
        "What type of document is this? Answer with a short snake_case label.",
        "Classify this document.",
        "Identify the document type.",
    ],
    "field_qa": [
        "Read the {field} from this document.",
        "{field}?",
        "What is the {field} in this document?",
        "Find the value of \"{field}\" ({section}).",
        "Look at the document and tell me the {field}.",
    ],
    "schema": [
        "Produce a JSON object with the keys {fields}; set a key to null when the document lacks it.",
        "From the document, extract: {fields}. Output JSON, null where missing.",
        "Fill in this schema from the document. Use null for anything that is not present.\nFields: {fields}",
        "Return JSON with exactly these keys: {fields}. If a value is missing or unreadable, use null.",
        "Extract only the following fields as JSON (null when absent): {fields}",
    ],
    "key_value": [
        "Write out the document's fields, one `key: value` per line.",
        "Summarise the form as key: value pairs.",
        "List every filled field as `name: value` lines.",
        "Give a key: value list of the information on this document.",
        "Output the form's data as plain `key: value` lines, one per line.",
    ],
    "table": [
        "Give me the page {page} table in markdown.",
        "Turn the table on page {page} into markdown rows.",
        "Extract the table on page {page} as markdown.",
        "Convert the table on page {page} to a markdown table.",
        "Reproduce the page {page} table, headers first, as markdown.",
    ],
    "visual": [
        "List every checkbox, signature, stamp and logo in this document with its state, as a JSON array that "
        "conforms to the JSON Schema below. Return only the JSON array.\n\nJSON Schema:\n{schema}",
    ],
}

PROMPTS.update({
    "figure": [
        "List every chart, diagram, figure and formula in this document as a JSON array that conforms to the JSON "
        "Schema below. Include each series and its points. Return only the JSON array.\n\nJSON Schema:\n{schema}",
    ],
    "chart_qa": [
        "{question}",
        "Answer from the chart: {question}",
        "Look at the chart and answer briefly. {question}",
    ],
    "chart_trend": [
        "{question}",
        "Using only the values shown in the chart, {question}",
    ],

    "propose_schema": [
        "Design a {format} that captures the structured data in this document.",
        "Write the {format} a developer would use to store this kind of document.",
        "Look at the document and produce a suitable {format} for its fields. Output only the code.",
        "Generate a {format} for this document type. Group related fields together.",
        "What {format} would fit this document? Return just the definition.",
    ],
    # One canonical wording: this is the request a vLLM client sends (the JSON Schema is also passed as
    # response_format / structured_outputs for constrained decoding; vLLM does not put it in the prompt
    # itself, so it is stated here). DeepSeek-OCR has a single raw prompt, so there is no separate system turn.
    "extract_to_schema": [
        "Extract the data from this document as a JSON object that conforms to the JSON Schema below. "
        "Use null for any value that is missing or unreadable. Return only the JSON object.\n\n"
        "JSON Schema:\n{schema}",
    ],
    "extract_format": [
        "Extract the document's fields and return them as {format}.",
        "Convert this document to {format}, grouping fields by section. Use empty values for anything missing.",
        "Output the structured data of this document in {format} only.",
        "Give me the form's contents as {format}.",
        "Transcribe the document's data into {format}.",
    ],
})

# Training targets follow what structured-output serving supports: JSON Schema in, JSON out. The other
# renderers (Pydantic, TypeScript, SQL, dataclass, YAML, XML, CSV) stay in train/render.py but are not drawn.
STRUCTURED_SCHEMA_FORMATS = ["json_schema"]
STRUCTURED_OUTPUT_FORMATS = ["json"]

_MAX_QA_PER_DOC = 5
_MAX_CHART_QUESTIONS = 3
_MAX_SCHEMA_FIELDS = 12


def _value(field: dict) -> str | None:
    if field.get("status") != "extracted":
        return None
    value = field.get("normalized_value")
    return value if value not in ("", None) else field.get("raw_value")


def _grouped(fields: list[dict]) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for f in fields:
        value = _value(f)
        if value is not None:
            grouped.setdefault(f["section"], {})[f["name"]] = value
    return grouped


def _human(name: str) -> str:
    return name.replace("_", " ")


def _ocr(record: dict, rng: random.Random) -> list[tuple[dict, str, list[int]]]:
    return [({}, text, [n]) for n, text in enumerate(record["text"]) if text.strip()]


def _schema_text(schema: dict, rng: random.Random) -> str:
    """Clients send the schema either pretty-printed or compact; both appear."""
    return json.dumps(schema, indent=2) if rng.random() < 0.5 else json.dumps(schema, separators=(",", ":"))


def _table_target(table: dict) -> dict:
    """A table as the page shows it. `evidence` (OCR line ids) and `boxes` (pixel coordinates) are dropped:
    the prompt carries neither, so the model could only invent them."""
    rows = []
    for row in table.get("rows") or []:
        visual = row.get("visual") or {}
        rows.append({
            "row_id": row.get("row_id", ""), "row_type": row.get("row_type", "other"),
            "cells": [{"column": int(c.get("column", 0)), "text": c.get("text", ""),
                       "row_span": int(c.get("row_span", 1)), "col_span": int(c.get("col_span", 1))}
                      for c in row.get("cells") or []],
            "visual": {"background_color": visual.get("background_color", ""),
                       "text_color": visual.get("text_color", ""),
                       "highlighted": bool(visual.get("highlighted", False)), "bold": bool(visual.get("bold", False))},
        })
    return {"table_id": table.get("table_id", ""), "page": int(table.get("page") or 1),
            "headers": list(table.get("headers") or []), "rows": rows}


def _element_target(element: dict) -> dict:
    return {"element_id": element.get("element_id", ""), "type": element.get("type", "other"),
            "page": int(element.get("page") or 1), "description": element.get("description", ""),
            "color": element.get("color", ""), "state": element.get("state", "unknown")}


def _figure_target(figure: dict) -> dict:
    """A figure as the training target: values as numbers, evidence ids and boxes dropped (see _table_target)."""
    def axis(a):
        return {"label": a.get("label", ""), "unit": a.get("unit", "")}
    series = [{"name": sr.get("name", ""),
               "points": [{"label": str(p.get("label", "")), "value": float(p["value"]),
                           "estimated": bool(p.get("estimated", False))} for p in sr.get("points") or []]}
              for sr in figure.get("series") or []]
    return {"figure_id": figure.get("figure_id", ""), "figure_type": figure.get("figure_type", "other"),
            "page": int(figure.get("page") or 1), "title": figure.get("title", ""), "caption": figure.get("caption", ""),
            "summary": figure.get("summary", ""), "x_axis": axis(figure.get("x_axis") or {}),
            "y_axis": axis(figure.get("y_axis") or {}), "series": series,
            "trend": figure.get("trend", "not_applicable")}


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"


def _extract(record: dict, rng: random.Random) -> list[tuple[dict, str, list[int]]]:
    target = {
        "document_type": record["document_type"],
        "fields": _grouped(record["fields"]),
        "tables": [_table_target(t) for t in record.get("tables") or []],
        "figures": [_figure_target(f) for f in record.get("figures") or []],
        "visual_elements": [_element_target(e) for e in record.get("visual_elements") or []],
        "relationships": [{"from_id": r["from_id"], "to_id": r["to_id"], "type": r["type"]}
                          for r in record.get("relationships") or []],   # always present, empty when none
    }
    args = {"schema": _schema_text(EXTRACT_SCHEMA, rng)}
    return [(args, json.dumps(target, ensure_ascii=False), list(range(len(record["images"]))))]


def _classify(record: dict, rng: random.Random):
    return [({}, record["document_type"], list(range(len(record["images"]))))] if record.get("document_type") else []


def _field_qa(record: dict, rng: random.Random):
    usable = [f for f in record["fields"] if _value(f) is not None]
    rng.shuffle(usable)
    return [({"field": _human(f["name"]), "section": _human(f["section"])}, _value(f),
             list(range(len(record["images"])))) for f in usable[:_MAX_QA_PER_DOC]]


def _schema(record: dict, rng: random.Random):
    fields = list(record["fields"])
    if not fields:
        return []
    rng.shuffle(fields)
    chosen = fields[:_MAX_SCHEMA_FIELDS]
    answer = {f["name"]: _value(f) for f in chosen}  # null teaches the model to abstain
    return [({"fields": ", ".join(answer)}, json.dumps(answer, ensure_ascii=False),
             list(range(len(record["images"]))))]


def _key_value(record: dict, rng: random.Random):
    lines = [f"{f['name']}: {_value(f)}" for f in record["fields"] if _value(f) is not None]
    return [({}, "\n".join(lines), list(range(len(record["images"]))))] if lines else []


def _markdown(table: dict) -> str:
    headers = table.get("headers") or []
    rows = [[cell["text"] for cell in sorted(row["cells"], key=lambda c: c["column"])]
            for row in table.get("rows", []) if row.get("row_type") != "header"]
    width = max([len(headers)] + [len(r) for r in rows] or [0])
    if not width:
        return ""
    pad = lambda r: (list(r) + [""] * width)[:width]  # noqa: E731
    lines = ["| " + " | ".join(pad(headers)) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(pad(r)) + " |" for r in rows]
    return "\n".join(lines)


def _table(record: dict, rng: random.Random):
    out = []
    for table in record.get("tables") or []:
        md = _markdown(table)
        page = int(table.get("page") or 1)
        if md:
            out.append(({"page": page}, md, [max(page - 1, 0)]))
    return out


def _visual(record: dict, rng: random.Random):
    kept = [{"type": e["type"], "state": e["state"], "description": e["description"], "page": e["page"]}
            for e in record.get("visual_elements") or [] if e["type"] in VISUAL_TYPES]
    args = {"schema": _schema_text(VISUAL_SCHEMA, rng)}
    return [(args, json.dumps(kept, ensure_ascii=False), list(range(len(record["images"]))))] if kept else []


def _propose_schema(record: dict, rng: random.Random):
    if not record["fields"]:
        return []
    fmt = rng.choice(STRUCTURED_SCHEMA_FORMATS)
    args = {"format": FORMAT_LABEL[fmt], "rules": SCHEMA_RULES[fmt]}
    return [(args, render_schema(record, fmt), list(range(len(record["images"]))))]


# Per-example odds of each prompt style for schema-conditioned extraction; they combine independently.
P_DESCRIBED, P_PREAMBLE, P_SUBSET = 0.5, 0.3, 0.3


def _extract_to_schema(record: dict, rng: random.Random, describer: Describer | None = None):
    if not record["fields"]:
        return []
    describer = describer or Describer()
    fields = list(record["fields"])
    if len(fields) > 1 and rng.random() < P_SUBSET:     # ask for fewer fields, in a different order
        rng.shuffle(fields)
        fields = fields[: rng.randint(1, len(fields))]
    document_type = record.get("document_type") or "document"
    described = rng.random() < P_DESCRIBED              # descriptions inside the schema
    schema = json.loads(render_schema(dict(record, fields=fields), "json_schema", describer if described else None))
    args = {"format": FORMAT_LABEL["json_schema"], "schema": _schema_text(schema, rng)}
    if rng.random() < P_PREAMBLE:                       # prose explaining the document and every attribute
        args["preamble"] = describer.preamble(document_type, fields, rng)
    return [(args, serialise(values(fields), "json"), list(range(len(record["images"]))))]


def _extract_format(record: dict, rng: random.Random):
    if not record["fields"]:
        return []
    fmt = rng.choice(STRUCTURED_OUTPUT_FORMATS)
    answer = serialise(values(record["fields"]), fmt, record.get("document_type") or "document")
    args = {"format": FORMAT_LABEL[fmt], "rules": OUTPUT_RULES[fmt]}
    return [(args, answer, list(range(len(record["images"]))))]


def _figure(record: dict, rng: random.Random):
    figures = record.get("figures") or []
    if not figures:
        return []
    args = {"schema": _schema_text(FIGURES_SCHEMA, rng)}
    answer = json.dumps([_figure_target(f) for f in figures], ensure_ascii=False)
    return [(args, answer, list(range(len(record["images"]))))]


def _chart_qa(record: dict, rng: random.Random):
    """ChartQA's own question-answer pairs about the image: the answer is the annotated one."""
    pairs = [q for q in record.get("qa") or [] if q.get("question") and q.get("answers")]
    rng.shuffle(pairs)
    return [({"question": q["question"]}, q["answers"][0], list(range(len(record["images"]))))
            for q in pairs[:_MAX_QA_PER_DOC]]


def _chart_trend(record: dict, rng: random.Random):
    """Questions about the data of charts whose values are all read exactly (no estimated point). The answers are
    computed from those values, so they are correct by construction."""
    out = []
    for figure in record.get("figures") or []:
        if figure.get("figure_type") not in ("bar_chart", "line_chart", "area_chart", "scatter_plot"):
            continue
        main = next((sr for sr in figure.get("series") or [] if len(sr.get("points") or []) >= 2), None)
        if main is None or any(p.get("estimated") for p in main["points"]):
            continue
        points = main["points"]
        where = f"'{figure['title']}'" if figure.get("title") else "this chart"
        values = [float(p["value"]) for p in points]
        top = points[values.index(max(values))]
        out.append(({"question": f"What is the trend of {main.get('name') or 'the values'} in {where}?"},
                    trend_of(figure), list(range(len(record["images"])))))
        out.append(({"question": f"Which category has the highest value in {where}?"},
                     str(top["label"]), list(range(len(record["images"])))))
        pick = rng.choice(points)
        out.append(({"question": f"What is the value for {pick['label']} in {where}?"},
                     _fmt(float(pick["value"])), list(range(len(record["images"])))))
    rng.shuffle(out)
    return out[:_MAX_CHART_QUESTIONS]


TASKS: dict[str, Callable] = {
    "ocr": _ocr, "extract": _extract, "classify": _classify, "field_qa": _field_qa,
    "schema": _schema, "key_value": _key_value, "table": _table, "visual": _visual,
    "propose_schema": _propose_schema, "extract_to_schema": _extract_to_schema, "extract_format": _extract_format,
    "figure": _figure, "chart_qa": _chart_qa, "chart_trend": _chart_trend,
}
_NEEDS_DESCRIBER = {"extract_to_schema"}


def derive(record: dict, rng: random.Random, tasks: list[str] | None = None,
           describer: Describer | None = None) -> list[dict]:
    """Every example this record supports, one list entry per (task, target). All randomness (wording,
    schema format, output format, which fields get a question) comes from `rng`, so the caller controls
    reproducibility and can draw a fresh set each epoch."""
    examples = []
    for name in tasks or list(TASKS):
        made = TASKS[name](record, rng, describer) if name in _NEEDS_DESCRIBER else TASKS[name](record, rng)
        for n, (args, answer, page_idx) in enumerate(made):
            images = [record["images"][i] for i in page_idx if i < len(record["images"])]
            prompt = rng.choice(PROMPTS[name]).format(**args)
            if args.get("rules"):   # the target format's conventions, stated rather than left to guess
                prompt += "\n\n" + args["rules"]
            if args.get("preamble"):   # what the document and its attributes are, as a user would write it
                prompt = args["preamble"] + "\n\n" + prompt
            examples.append({
                "id": f"{record['id']}#{name}{n}",
                "task": name,
                "images": images,
                "messages": [
                    {"role": "user", "content": [{"type": "image"} for _ in images]
                     + [{"type": "text", "text": prompt}]},
                    {"role": "assistant", "content": answer},
                ],
            })
    return examples
