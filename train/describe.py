"""Attribute descriptions and instruction text for schema-conditioned extraction prompts.

Real users explain the structured output they want: what the document is, what every attribute means,
how values are formatted. This module produces that text automatically.

    describer = Describer.load("data/descriptions.json")   # optional table from `labeller describe-*`
    describer.field("invoice", field)                      # table entry, else a template from name + type

Descriptions never contain a field's value, and the formatting rules below describe exactly what
`train.render.typed_value` does to the answers, so the text and the target never disagree.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

_TYPE_HINT = {
    "string": "Text copied as written.",
    "other": "Text copied as written.",
    "number": "A plain number without currency symbol or thousands separator, when well formed.",
    "date": "A date, YYYY-MM-DD when well formed.",
    "boolean": "true or false, for checkboxes and yes/no values.",
}

FORMAT_RULES = ("Normalise dates to YYYY-MM-DD and numbers to plain numbers where the value is well formed; "
                "otherwise copy the text as written. Use null for anything that is missing or unreadable.")

_PREAMBLES = [
    "You are extracting data from {a_document}. The attributes are:\n{attributes}\n\n{rules}",
    "The document is {a_document}. Each attribute below is grouped by section:\n{attributes}\n\n{rules}",
    "Task: structured extraction from {a_document}.\nAttributes:\n{attributes}\n\nRules: {rules}",
]


def with_article(noun: str) -> str:
    return ("an " if noun[:1].lower() in "aeiou" else "a ") + noun


def human(name: str) -> str:
    return (name or "").replace("_", " ").strip()


def key(document_type: str, section: str, name: str) -> str:
    return f"{document_type}|{section}|{name}"


class Describer:
    """The field's own description (written by Stage 2 with the document in view), else an entry in the
    optional table, else one built from the field's name and type."""

    def __init__(self, table: dict[str, str] | None = None) -> None:
        self.table = dict(table or {})

    @classmethod
    def load(cls, path: str | Path | None) -> "Describer":
        if path is None:
            return cls()
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def field(self, document_type: str, field: dict) -> str:
        own = " ".join(str(field.get("description") or "").split())
        if own:
            return own
        found = self.table.get(key(document_type, field["section"], field["name"]))
        if found:
            return found
        label = human(field["name"]).capitalize()
        return f"{label}. {_TYPE_HINT.get(field.get('data_type'), _TYPE_HINT['string'])}"

    def section(self, section: str) -> str:
        return f"Fields about {human(section)}."

    def document(self, document_type: str) -> str:
        return f"Data extracted from {with_article(human(document_type) or 'document')} document."

    def preamble(self, document_type: str, fields: list[dict], rng: random.Random) -> str:
        """Prose a user might put before the schema: what the document is, every attribute, the format rules."""
        lines = [f"- {f['section']}.{f['name']}: {self.field(document_type, f)}" for f in fields]
        return rng.choice(_PREAMBLES).format(a_document=with_article(human(document_type) or "document"),
                                             attributes="\n".join(lines), rules=FORMAT_RULES)
