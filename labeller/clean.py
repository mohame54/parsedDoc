"""Keeps the training value of a field faithful to the page.

The labeller stores two values per field: `raw_value` (copied from the transcription, and checked against it by
`validate.validate_fields`) and `normalized_value` (Gemini's cleaned-up form, which nothing checked). Training and
`train.render.typed_value` use the normalised value whenever there is one, so a rewrite such as `Nov.` -> `November`
or `T.G.I.FRIDAY'S` -> `T.G.I. FRIDAY'S` became the answer the model is asked to give, although the page says otherwise.

`clean_field` keeps a normalisation only when it can be derived from the raw text:

    string / other   the raw value is the target; a different normalised string is cleared
    number           kept when it equals the number written in the raw value (`.5` = `0.5`, `1,200.50` = `1200.5`)
    date             kept when it is a real calendar date whose day, month and year are all written in the raw value
    boolean          kept when the raw value is a checkbox mark or a yes/no word that maps to it

A cleared normalisation falls back to the raw value (`typed_value` already does that), and every change is recorded in
the record's `label_notes`, so what was dropped stays available (for instance for training the model to say it is unsure).

    python -m labeller.clean data/export/unverified data/export/clean      # fix an export that already exists
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any

from .validate import value_in_source

TRUE_WORDS = {"true", "yes", "y", "[x]", "x", "checked", "ja", "oui", "si", "sí", "sim", "[✓]", "✓"}
FALSE_WORDS = {"false", "no", "n", "[ ]", "unchecked", "nein", "non", "não", "nao", "[]"}
_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

# three-letter starts of month names (English, German, French, Spanish, Italian, Portuguese)
_MONTH_NAMES = {
    1: ["jan", "gen", "ene"], 2: ["feb", "fev", "fév"], 3: ["mar", "mär", "mrz"], 4: ["apr", "avr", "abr"],
    5: ["may", "mai", "mag"], 6: ["jun", "giu"], 7: ["jul", "lug", "juil"], 8: ["aug", "ago", "aou", "aoû"],
    9: ["sep", "set"], 10: ["oct", "okt", "ott", "out"], 11: ["nov"], 12: ["dec", "déc", "dez", "dic"],
}


def squash(text: Any) -> str:
    """Case, spacing and punctuation removed (letters and digits of any script stay)."""
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", str(text)).casefold())


def _plain(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))


def _months_named(raw: str) -> set[int]:
    words = re.findall(r"[^\W\d_]+", _plain(raw))
    found = set()
    for word in words:
        for month, starts in _MONTH_NAMES.items():
            if any(word.startswith(_plain(s)) for s in starts) and len(word) >= 3:
                found.add(month)
    return found


_ISO_MONTH = re.compile(r"(\d{4})-(\d{2})")
_ISO_YEAR = re.compile(r"\d{4}")


def _take(numbers: list[int], *options: int) -> list[int] | None:
    """`numbers` without the first of `options` it contains, or None when none of them is there."""
    for option in options:
        if option in numbers:
            rest = list(numbers)
            rest.remove(option)
            return rest
    return None


def date_from_raw(normalized: str, raw: str) -> str | None:
    """None when `normalized` (YYYY-MM-DD, YYYY-MM or YYYY) is a good rendering of `raw`, else the reason it is not."""
    text = normalized.strip()
    numbers = [int(n) for n in re.findall(r"\d+", raw)]
    named = _months_named(raw)
    if _ISO.fullmatch(text):
        year, month, day = (int(g) for g in _ISO.fullmatch(text).groups())
        try:
            datetime.date(year, month, day)
        except ValueError:
            return "invalid_date"
        rest = _take(numbers, year, year % 100)               # the year, written in full or with two digits
        rest = _take(rest, day) if rest is not None else None
        if rest is not None and (month in rest or month in named):
            return None
    elif _ISO_MONTH.fullmatch(text):                          # month and year, no day: "March 2017", "2 2019"
        year, month = (int(g) for g in _ISO_MONTH.fullmatch(text).groups())
        if not 1 <= month <= 12:
            return "invalid_date"
        rest = _take(numbers, year, year % 100)
        if rest is not None and (month in rest or month in named):
            return None
    elif _ISO_YEAR.fullmatch(text):
        if int(text) in numbers:
            return None
    return "date_not_derivable"


def _floats(raw: str) -> list[float]:
    """The ways a raw amount can be read: `1,200.50`, `1.200,50`, `.5`, `$ 12`."""
    text = re.sub(r"[^\d.,+-]", "", raw)
    out = []
    for variant in (text.replace(",", ""), text.replace(".", "").replace(",", ".")):
        try:
            out.append(float(variant))
        except ValueError:
            pass
    return out


def number_from_raw(normalized: str, raw: str) -> bool:
    try:
        target = float(str(normalized).replace(",", "").strip())
    except ValueError:
        return False
    return any(abs(v - target) <= 1e-9 * max(1.0, abs(target)) for v in _floats(raw))


_MARK = re.compile(r"^\s*(\[\s*[xX✓✔]\s*\]|\[\s*\])")


def boolean_from_raw(normalized: str, raw: str) -> bool:
    n, r = str(normalized).strip().lower(), str(raw).strip().lower()
    mark = _MARK.match(raw)                      # "[x] AIR" is a checked box with a label next to it
    if mark:
        return (n in TRUE_WORDS) == bool(re.search(r"[xX✓✔]", mark.group(1))) and (n in TRUE_WORDS or n in FALSE_WORDS)
    return (n in TRUE_WORDS and r in TRUE_WORDS) or (n in FALSE_WORDS and r in FALSE_WORDS)


def clean_field(field: dict, source_text: str | None = None) -> tuple[dict, list[dict]]:
    """(the field with an unsupported normalisation removed, the notes of what changed). The input is not modified."""
    out = dict(field)
    notes: list[dict] = []
    if field.get("status") != "extracted":
        return out, notes
    raw, norm = str(field.get("raw_value") or ""), str(field.get("normalized_value") or "")
    kind = field.get("data_type") or "string"

    def drop(action: str) -> None:
        notes.append({"field": f"{field.get('section')}.{field.get('name')}", "action": action, "raw": raw, "was": norm})
        out["normalized_value"] = ""

    if norm and norm != raw:
        if kind == "number":
            if not number_from_raw(norm, raw):
                drop("number_not_in_raw")
        elif kind == "date":
            reason = date_from_raw(norm, raw)
            if reason:
                drop(reason)
        elif kind == "boolean":
            if not boolean_from_raw(norm, raw):
                drop("boolean_not_in_raw")
        elif " ".join(norm.split()) == " ".join(raw.split()):
            pass                   # only the line breaks and runs of spaces were collapsed: still the page text
        elif squash(norm) != squash(raw):
            drop("string_rewrite_dropped")
        else:                      # same letters and digits, other case / spacing / punctuation: the raw text is the target
            drop("string_respaced_dropped")
    if source_text is not None and raw.strip() and not value_in_source(raw, source_text, max_gap=3):
        notes.append({"field": f"{field.get('section')}.{field.get('name')}", "action": "not_in_source",
                      "raw": raw, "was": field.get("status")})
        out["status"] = "uncertain"
    return out, notes


def clean_record(record: dict, check_source: bool = True) -> dict:
    """A copy of a canonical record with every field cleaned and the changes listed in `label_notes`."""
    out = copy.deepcopy(record)
    source = "\n".join(str(p) for p in record.get("text") or []) if check_source else None
    notes = list(out.get("label_notes") or [])
    seen = {json.dumps(n, sort_keys=True) for n in notes}
    fields = []
    for field in record.get("fields") or []:
        cleaned, found = clean_field(field, source or None)
        fields.append(cleaned)
        for note in found:
            key = json.dumps(note, sort_keys=True)
            if key not in seen:
                seen.add(key)
                notes.append(note)
    out["fields"] = fields
    out["label_notes"] = notes
    return out


# ---- fix an existing export ----------------------------------------------------------------------------

def clean_export(src: str | Path, dst: str | Path) -> dict[str, int]:
    """Writes a cleaned copy of an export folder: the canonical split files, and the `sft/extract_*` files rebuilt
    from them. `sft/ocr_*`, `dataset.json` and the other small files are copied; `images/` is linked, not copied."""
    from .export import _extract_example, _write_jsonl

    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("give a different destination folder")
    dst.mkdir(parents=True, exist_ok=True)
    counts = {"records": 0, "fields_changed": 0}
    for split in ("train", "val", "test"):
        path = src / f"{split}.jsonl"
        if not path.is_file():
            continue
        rows = [clean_record(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        counts["records"] += len(rows)
        counts["fields_changed"] += sum(len(r["label_notes"]) for r in rows)
        _write_jsonl(dst / f"{split}.jsonl", rows)
        _write_jsonl(dst / "sft" / f"extract_{split}.jsonl", [_extract_example({"canonical": r}) for r in rows])
    for extra in ("dataset.json", "summary.json", "excluded.jsonl"):
        if (src / extra).is_file():
            shutil.copy2(src / extra, dst / extra)
    ocr_dir = src / "sft"
    if ocr_dir.is_dir():
        for f in ocr_dir.glob("ocr_*.jsonl"):
            (dst / "sft").mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / "sft" / f.name)
    images = dst / "images"
    if (src / "images").is_dir() and not images.exists():
        try:
            os.symlink((src / "images").resolve(), images, target_is_directory=True)
        except OSError:
            shutil.copytree(src / "images", images)
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", help="an existing export folder (the one with train.jsonl)")
    parser.add_argument("dst", help="where the cleaned copy goes")
    args = parser.parse_args(argv)
    counts = clean_export(args.src, args.dst)
    print(f"cleaned {counts['records']} records, {counts['fields_changed']} field changes -> {args.dst}")


if __name__ == "__main__":
    main()
