"""Sizes the label problem in an export: how many extracted values the training target does not take from the page.

    python -m labeller.audit data/export/unverified                 # every split
    python -m labeller.audit data/export --split train --examples 15

Read-only. For every extracted field it asks what `labeller.clean` would change (a rewritten string, a number or date
that is not in the raw text, a value not found in the page text) and reports the shares per dataset and data type, plus
the worst raw -> normalised pairs. Run it before and after `python -m labeller.clean` to see the effect.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .clean import clean_field, squash


def audit(root: str | Path, splits: tuple[str, ...] = ("train", "val", "test"), examples: int = 10) -> dict:
    root = Path(root)
    by_source: dict[str, Counter] = defaultdict(Counter)
    by_type: dict[str, Counter] = defaultdict(Counter)
    pairs: list[tuple[str, str, str, str]] = []
    records = 0
    for split in splits:
        path = root / f"{split}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            records += 1
            source = str(record.get("source") or str(record.get("id", "")).split(":")[0] or "?")
            text = "\n".join(str(p) for p in record.get("text") or [])
            for field in record.get("fields") or []:
                if field.get("status") != "extracted":
                    continue
                kind = field.get("data_type") or "string"
                raw, norm = str(field.get("raw_value") or ""), str(field.get("normalized_value") or "")
                _, notes = clean_field(field, text or None)
                for counter in (by_source[source], by_type[kind]):
                    counter["extracted"] += 1
                    if norm and norm != raw:
                        counter["normalised_differs"] += 1
                    for note in notes:
                        counter[note["action"]] += 1
                    if notes:
                        counter["changed"] += 1
                for note in notes:
                    if note["action"] != "not_in_source":
                        pairs.append((note["action"], source, note["raw"], note["was"]))
    def table(groups: dict[str, Counter]) -> dict:
        out = {}
        for name, c in sorted(groups.items()):
            n = c["extracted"] or 1
            out[name] = {**{k: v for k, v in sorted(c.items())}, "changed_share": round(c["changed"] / n, 4)}
        return out
    return {"records": records, "by_source": table(by_source), "by_data_type": table(by_type),
            "examples": [{"action": a, "source": s, "raw": r, "normalised": w} for a, s, r, w in pairs[:examples]]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--out", default=None, help="also write the report here (JSON)")
    args = parser.parse_args(argv)
    report = audit(args.root, (args.split,) if args.split else ("train", "val", "test"), args.examples)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
