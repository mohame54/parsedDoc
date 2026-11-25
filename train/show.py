"""Look at what the model generated: the page, the request (with its schema), the reference and the model's answer.

    python -m train.show checkpoints/lora_r32/generations/generations_step_0001374.jsonl --root data/export/unverified
    python -m train.show FILE --root ROOT --task extract_to_schema --limit 5 --out look.html
    python -m train.show FILE --text                 # print to the terminal instead of writing an HTML page

The generations file is written by the generation check (`--gen-eval`, or `python -m train.gen_eval`). Files written
before this tool existed hold no prompt or image path; run the check again to get them.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

from .gen_eval import extract_json, flatten


def differences(prediction: str, reference: str, limit: int = 12) -> list[str]:
    """Plain-language list of what the answer got wrong, for JSON answers."""
    ref, pred = extract_json(reference), extract_json(prediction)
    if ref is None:
        return []
    if pred is None:
        return ["the answer is not valid JSON"]
    want, got = flatten(ref), flatten(pred)
    out = [f"{path}: expected {want[path]!r}, got {got[path]!r}" for path in want if path in got and got[path] != want[path]]
    out += [f"{path}: missing (expected {want[path]!r})" for path in want if path not in got]
    out += [f"{path}: not in the reference (got {got[path]!r})" for path in got if path not in want]
    return out[:limit] + ([f"... and {len(out) - limit} more"] if len(out) > limit else [])


def pretty(text: str) -> str:
    value = extract_json(text) if text.lstrip()[:1] in "{[" or "```" in text else None
    return json.dumps(value, indent=2, ensure_ascii=False) if value is not None else text


def _thumb(path: Path, side: int = 900) -> str:
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((side, side))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def load(path: str | Path, task: str | None = None, limit: int | None = None, worst_first: bool = False) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if task:
        rows = [r for r in rows if r["task"] == task]
    if worst_first:
        rows.sort(key=lambda r: r.get("score", 0))
    return rows[:limit] if limit else rows


def render_text(rows: list[dict]) -> str:
    parts = []
    for r in rows:
        parts.append("=" * 100)
        parts.append(f"{r['id']}  [{r['task']}]  score={r.get('score')}  images={r.get('images', '?')}")
        parts.append("--- request ---\n" + r.get("prompt", "(not stored in this file)"))
        parts.append("--- reference ---\n" + pretty(r["reference"]))
        parts.append("--- model ---\n" + pretty(r["prediction"]))
        diffs = differences(r["prediction"], r["reference"])
        if diffs:
            parts.append("--- differences ---\n" + "\n".join(diffs))
    return "\n".join(parts)


def render_html(rows: list[dict], root: Path | None) -> str:
    cards = []
    for r in rows:
        images = ""
        for rel in r.get("images", [])[:2]:
            if root is not None and (root / rel).is_file():
                images += f'<img src="{_thumb(root / rel)}" alt="{html.escape(rel)}">'
        diffs = differences(r["prediction"], r["reference"])
        score = r.get("score", 0) or 0
        cards.append(f"""<section class="card">
<h2>{html.escape(r['task'])} <span class="id">{html.escape(r['id'])}</span> <span class="score {'ok' if score >= .8 else 'mid' if score >= .4 else 'bad'}">{score:.2f}</span></h2>
<div class="grid"><div class="page">{images or '<p class="muted">page image not found (give --root)</p>'}</div>
<div><h3>Request</h3><pre>{html.escape(r.get('prompt', '(not stored in this file)'))}</pre></div>
<div><h3>Reference</h3><pre>{html.escape(pretty(r['reference']))}</pre></div>
<div><h3>Model</h3><pre>{html.escape(pretty(r['prediction']))}</pre></div></div>
{('<h3>Differences</h3><ul>' + ''.join(f'<li>{html.escape(d)}</li>' for d in diffs) + '</ul>') if diffs else ''}
</section>""")
    return f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Generations</title><style>
:root{{--bg:#fff;--fg:#1c1c1e;--card:#f5f5f7;--line:#d8d8de;--ok:#1a7f37;--mid:#9a6700;--bad:#cf222e}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#e8e8ea;--card:#1c1c1f;--line:#38383d;--ok:#3fb950;--mid:#d29922;--bad:#f85149}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:0 auto 18px;max-width:1500px}}
h2{{font-size:16px;margin:0 0 10px}}h3{{font-size:13px;margin:0 0 6px}}.id,.muted{{color:#888;font-weight:400;font-size:12px}}
.score{{float:right;font-weight:700}}.ok{{color:var(--ok)}}.mid{{color:var(--mid)}}.bad{{color:var(--bad)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}}
img{{max-width:100%;border:1px solid var(--line);border-radius:6px;margin-bottom:6px}}
pre{{white-space:pre-wrap;word-break:break-word;margin:0;font:12px/1.4 ui-monospace,monospace;max-height:520px;overflow:auto}}
</style>{''.join(cards)}"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("generations")
    parser.add_argument("--root", help="the export folder (for the page images)")
    parser.add_argument("--task")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--worst-first", action="store_true", help="lowest scores first")
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--out", default="generations.html")
    args = parser.parse_args(argv)
    rows = load(args.generations, args.task, args.limit, args.worst_first)
    if not rows:
        raise SystemExit("no matching rows")
    if args.text:
        print(render_text(rows))
        return
    Path(args.out).write_text(render_html(rows, Path(args.root) if args.root else None), encoding="utf-8")
    print(f"wrote {args.out} ({len(rows)} examples)")


if __name__ == "__main__":
    main()
