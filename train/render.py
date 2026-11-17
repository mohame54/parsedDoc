"""Renders a document's fields as schemas (JSON Schema, Pydantic, TypeScript, SQL, dataclass) and
serialises extracted values as JSON, YAML, XML or CSV.

Everything here is pure text generation from the canonical export record, so the same document can be
shown to the model as many different "flows" (a schema-first prompt, a code-generation prompt, a
format-conversion prompt) without any extra labelling.
"""
from __future__ import annotations

import csv
import io
import json
import re
from typing import Any
from xml.sax.saxutils import escape

import yaml

SCHEMA_FORMATS = ["json_schema", "pydantic", "typescript", "sql", "dataclass"]
NESTED_FORMATS = ["json_schema", "pydantic", "typescript"]   # keep the section -> field nesting
OUTPUT_FORMATS = ["json", "yaml", "xml", "csv"]

_JSON_TYPE = {"string": "string", "number": "number", "date": "string", "boolean": "boolean", "other": "string"}
_PY_TYPE = {"string": "str", "number": "float", "date": "date", "boolean": "bool", "other": "str"}
_TS_TYPE = {"string": "string", "number": "number", "date": "string", "boolean": "boolean", "other": "string"}
_SQL_TYPE = {"string": "TEXT", "number": "NUMERIC", "date": "DATE", "boolean": "BOOLEAN", "other": "TEXT"}


def _pascal(text: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[^A-Za-z0-9]+", text or "") if part) or "Document"


def _sections(fields: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in fields:
        out.setdefault(f["section"], []).append(f)
    return out


_NUMBER = re.compile(r"[-+]?(0|[1-9]\d*)(\.\d+)?")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_TRUE = {"true", "yes", "y", "[x]", "x", "checked"}
_FALSE = {"false", "no", "n", "[ ]", "unchecked"}


def typed_value(field: dict) -> tuple[str, Any]:
    """(JSON type, value) for one field. The value is coerced to its declared data_type when it is well
    formed (numbers, checkboxes); otherwise it stays the text as written and the type is "string". The
    schema and the answer both come from here, so the answer always validates against its own schema."""
    declared = _JSON_TYPE.get(field["data_type"], "string")
    if field.get("status") != "extracted":
        return declared, None
    text = field.get("normalized_value")
    if text in ("", None):
        text = field.get("raw_value")
    if text is None or not str(text).strip():
        return declared, None
    text = str(text).strip()
    if declared == "number":
        cleaned = text.replace(",", "")
        if _NUMBER.fullmatch(cleaned):
            return "number", float(cleaned) if "." in cleaned else int(cleaned)
    elif declared == "boolean":
        low = text.lower()
        if low in _TRUE:
            return "boolean", True
        if low in _FALSE:
            return "boolean", False
    return "string", text


def values(fields: list[dict]) -> dict[str, dict]:
    """Section -> field -> value, with null for fields that were absent or unreadable."""
    out: dict[str, dict] = {}
    for f in fields:
        out.setdefault(f["section"], {})[f["name"]] = typed_value(f)[1]
    return out


def render_schema(record: dict, fmt: str, describe: Any = None) -> str:
    """`describe` (a train.describe.Describer) adds attribute descriptions; only JSON Schema carries them."""
    fields, title = record["fields"], record.get("document_type") or "document"
    if fmt == "json_schema":
        return _json_schema(_sections(fields), title, describe)
    return _SCHEMA_RENDERERS[fmt](_sections(fields), title)


def _json_schema(sections: dict, title: str, describe: Any = None) -> str:
    """Strict JSON Schema, the form structured-output APIs accept (OpenAI strict, Anthropic, vLLM/xgrammar):
    every object lists all its keys in `required` and forbids extras, so a constrained decoder must emit
    every field (null when missing), which is exactly what `values()` produces. Types follow the values
    (see typed_value). With `describe`, every object and property also carries a `description`."""
    def prop(f):
        json_type, value = typed_value(f)
        node = {"type": [json_type, "null"]}
        if json_type == "string" and f["data_type"] == "date" and (value is None or _ISO_DATE.fullmatch(value)):
            node["format"] = "date"
        if describe is not None:
            node["description"] = describe.field(title, f)
        return node

    def section_node(section, fs):
        node = {"type": "object"}
        if describe is not None:
            node["description"] = describe.section(section)
        node.update({"properties": {f["name"]: prop(f) for f in fs}, "required": [f["name"] for f in fs],
                     "additionalProperties": False})
        return node

    schema = {"title": _pascal(title), "type": "object"}
    if describe is not None:
        schema["description"] = describe.document(title)
    schema.update({"properties": {section: section_node(section, fs) for section, fs in sections.items()},
                   "required": list(sections), "additionalProperties": False})
    return json.dumps(schema, indent=2)


def _pydantic(sections: dict, title: str) -> str:
    uses_date = any(f["data_type"] == "date" for fs in sections.values() for f in fs)
    lines = ["from datetime import date" if uses_date else "", "from typing import Optional", "",
             "from pydantic import BaseModel", "", ""]
    for section, fs in sections.items():
        lines.append(f"class {_pascal(section)}(BaseModel):")
        lines += [f"    {f['name']}: Optional[{_PY_TYPE.get(f['data_type'], 'str')}] = None" for f in fs]
        lines += ["", ""]
    lines.append(f"class {_pascal(title)}(BaseModel):")
    lines += [f"    {section}: Optional[{_pascal(section)}] = None" for section in sections]
    return "\n".join(line for i, line in enumerate(lines) if line or i)


def _typescript(sections: dict, title: str) -> str:
    lines = []
    for section, fs in sections.items():
        lines.append(f"interface {_pascal(section)} {{")
        lines += [f"  {f['name']}: {_TS_TYPE.get(f['data_type'], 'string')} | null;" for f in fs]
        lines += ["}", ""]
    lines.append(f"interface {_pascal(title)} {{")
    lines += [f"  {section}: {_pascal(section)};" for section in sections]
    lines.append("}")
    return "\n".join(lines)


def _sql(sections: dict, title: str) -> str:
    table = re.sub(r"[^a-z0-9]+", "_", (title or "document").lower()).strip("_") or "document"
    cols = [f"  {section}_{f['name']} {_SQL_TYPE.get(f['data_type'], 'TEXT')}" for section, fs in sections.items()
            for f in fs]
    return f"CREATE TABLE {table} (\n  id INTEGER PRIMARY KEY,\n" + ",\n".join(cols) + "\n);"


def _dataclass(sections: dict, title: str) -> str:
    uses_date = any(f["data_type"] == "date" for fs in sections.values() for f in fs)
    lines = ["from dataclasses import dataclass", "from datetime import date" if uses_date else "",
             "from typing import Optional", "", ""]
    for section, fs in sections.items():
        lines += ["@dataclass", f"class {_pascal(section)}:"]
        lines += [f"    {f['name']}: Optional[{_PY_TYPE.get(f['data_type'], 'str')}] = None" for f in fs]
        lines += ["", ""]
    lines += ["@dataclass", f"class {_pascal(title)}:"]
    lines += [f"    {section}: Optional[{_pascal(section)}] = None" for section in sections]
    return "\n".join(line for i, line in enumerate(lines) if line or i)


_SCHEMA_RENDERERS = {"json_schema": _json_schema, "pydantic": _pydantic, "typescript": _typescript,
                     "sql": _sql, "dataclass": _dataclass}

FORMAT_LABEL = {"json_schema": "JSON Schema", "pydantic": "Pydantic models", "typescript": "TypeScript interfaces",
                "sql": "SQL CREATE TABLE statement", "dataclass": "Python dataclasses",
                "json": "JSON", "yaml": "YAML", "xml": "XML", "csv": "CSV"}


# Strict JSON Schemas for the whole-document and visual-element targets. They are shown in the prompt
# (as a vLLM client would send them) and every target is built to validate against them.
from labeller.schemas import ELEMENT_TYPES, FIGURE_TYPES, RELATIONSHIP_TYPES, ROW_TYPES, TRENDS  # noqa: E402

_STR, _INT, _BOOL = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _enum(values) -> dict:
    return {"type": "string", "enum": list(values)}


_ELEMENT_STATES = ["checked", "unchecked", "signed", "present", "unknown"]
VISUAL_TYPES = ["checkbox", "signature", "stamp", "logo"]

# One figure as it is written in the training target (the evidence ids and boxes are not part of it).
FIGURE_SCHEMA = _obj({
    "figure_id": _STR, "figure_type": _enum(FIGURE_TYPES), "page": _INT, "title": _STR, "caption": _STR,
    "summary": _STR, "x_axis": _obj({"label": _STR, "unit": _STR}), "y_axis": _obj({"label": _STR, "unit": _STR}),
    "series": {"type": "array", "items": _obj({"name": _STR, "points": {"type": "array", "items": _obj({
        "label": _STR, "value": {"type": "number"}, "estimated": _BOOL})}})},
    "trend": _enum(TRENDS),
})
FIGURES_SCHEMA = {"type": "array", "items": FIGURE_SCHEMA}

EXTRACT_SCHEMA = _obj({
    "document_type": _STR,
    "fields": {"type": "object", "additionalProperties": {"type": "object", "additionalProperties": _STR}},
    "tables": {"type": "array", "items": _obj({
        "table_id": _STR, "page": _INT, "headers": {"type": "array", "items": _STR},
        "rows": {"type": "array", "items": _obj({
            "row_id": _STR, "row_type": _enum(ROW_TYPES),
            "cells": {"type": "array", "items": _obj({"column": _INT, "text": _STR, "row_span": _INT,
                                                      "col_span": _INT})},
            "visual": _obj({"background_color": _STR, "text_color": _STR, "highlighted": _BOOL, "bold": _BOOL}),
        })},
    })},
    "visual_elements": {"type": "array", "items": _obj({
        "element_id": _STR, "type": _enum(ELEMENT_TYPES), "page": _INT, "description": _STR, "color": _STR,
        "state": _enum(_ELEMENT_STATES)})},
    "figures": FIGURES_SCHEMA,
    "relationships": {"type": "array", "items": _obj({
        "from_id": _STR, "to_id": _STR, "type": _enum(RELATIONSHIP_TYPES)})},
})

VISUAL_SCHEMA = {"type": "array", "items": _obj({
    "type": _enum(VISUAL_TYPES), "state": _enum(_ELEMENT_STATES), "description": _STR, "page": _INT})}


# The conventions each renderer / serialiser applies. They are appended to the matching prompts (train/derive.py)
# so the target format is stated, never left for the model to guess. Keep in sync with the renderers above.
_TYPE_RULE = "Use number types for amounts and quantities, date types for dates and boolean for yes/no."
SCHEMA_RULES = {
    "json_schema": "Group the fields by section: one top-level property per section, each an object with one "
                   "property per field, using the field names found on the document in snake_case. Make every "
                   "field nullable (for example [\"string\", \"null\"]) and use format \"date\" for dates. "
                   "List every key in `required` and set additionalProperties to false on every object. Do not add "
                   "descriptions. " + _TYPE_RULE,
    "pydantic": "Write one BaseModel class per section with Optional fields that default to None, plus a root "
                "model whose Optional attributes are the sections. Use snake_case field names. " + _TYPE_RULE,
    "typescript": "Write one interface per section with `type | null` fields, plus a root interface with one "
                  "property per section. Use snake_case field names. " + _TYPE_RULE,
    "sql": "Write a single CREATE TABLE with an INTEGER PRIMARY KEY id and one column per field, named "
           "section_field in snake_case, with no constraints or foreign keys. Use TEXT, NUMERIC, DATE and "
           "BOOLEAN column types.",
    "dataclass": "Write one @dataclass per section with Optional fields that default to None, plus a root "
                 "dataclass whose Optional attributes are the sections. Use snake_case field names. " + _TYPE_RULE,
}
OUTPUT_RULES = {
    "json": "Nest the values by section as {section: {field: value}}, using null for missing or unreadable values.",
    "yaml": "Nest the values by section as section -> field -> value, using null for missing or unreadable values.",
    "xml": "Use the document type as the root element, one element per section and one element per field inside "
           "it. Leave the element empty for a missing or unreadable value.",
    "csv": "Write flat CSV: a header row of section.field column names and a single data row. Leave a cell empty "
           "for a missing or unreadable value.",
}


def serialise(data: dict[str, dict], fmt: str, root: str = "document") -> str:
    if fmt == "json":
        return json.dumps(data, ensure_ascii=False, indent=2)
    if fmt == "yaml":
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False).rstrip()
    if fmt == "xml":
        tag = lambda s: re.sub(r"[^A-Za-z0-9_]", "_", s) or "item"  # noqa: E731
        body = "".join(
            f"  <{tag(s)}>\n" + "".join(
                f"    <{tag(k)}>{escape(str(v)) if v is not None else ''}</{tag(k)}>\n" for k, v in fs.items())
            + f"  </{tag(s)}>\n" for s, fs in data.items())
        return f"<{tag(root)}>\n{body}</{tag(root)}>"
    if fmt == "csv":
        flat = {f"{s}.{k}": ("" if v is None else v) for s, fs in data.items() for k, v in fs.items()}
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(flat)
        writer.writerow(flat.values())
        return out.getvalue().rstrip()
    raise ValueError(f"unknown output format {fmt!r}")
