"""Builds RL examples from a labelled document: the page is partly made unreadable on purpose, then rotated and
degraded, and every reference field gets a state the reward uses (see train/rl_reward.py).

Order matters: masks are drawn on the ORIGINAL page (so their boxes need no mapping), then the page is turned and
degraded (train/degrade.py), then the result is OCR'd again. A field whose value is still readable anywhere on the
final page is `answerable`, not unreadable: the model must not be punished for reading a value that is there.

    example = build_example(base, rng, ocr_lines=lines_fn, ocr_text=text_fn, eligible=paths)

    base: {"id", "prompt" (with the plain JSON Schema), "label" (the plain reference JSON), "images": [PIL pages]}
    ocr_lines(image) -> [{"text", "box_px": [x0, y0, x1, y1]}]   (RapidOCR; injected so tests need no model)
    ocr_text(image)  -> str                                       (the same OCR, joined; for the readability re-check)

OCR makes mistakes, and a wrong "unreadable" label punishes a correct answer, so every decision that rests on OCR is
guarded: a value is only masked where OCR finds it exactly once (whole tokens, never inside a longer one), the search
uses what is printed (`raw_value`), "still readable" is checked on the masked page before AND after it is turned and
degraded with a forgiving match, and the unmasked fields serve as controls: when OCR cannot read most of those on the
final page, its "not found" means nothing and the unreadable labels of that example become `partial` (no penalty).
"""
from __future__ import annotations

import collections
import difflib
import json
import random
import re
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFilter

from .degrade import plan_example, render, upright_turn
from .gen_eval import flatten, fold, is_null, schema_from_prompt, squash
from .rl_prompts import build_prompt, sample_spec, wrap_answer, wrap_schema

MASKS = ("blur", "downscale", "noise", "bar", "stain")
MASK_STATE = {"blur": "partial", "downscale": "partial", "noise": "must_abstain", "bar": "must_abstain",
              "stain": "must_abstain"}
MASK_TAG = {"blur": "blurry", "downscale": "blurry", "noise": "noisy", "bar": "occluded", "stain": "occluded"}
TARGET_FRACTION = 0.3          # share of the localised fields that get masked on an example
MIN_VALUE_LENGTH = 3           # shorter values are too easy to find again by accident, so they are not masked
DECOYS = (1, 2)                # decoy masks per page: they carry no target, so a mask alone is not a signal
MIN_CONTROLS = 2               # unmasked, localised fields needed to judge how well OCR reads the final page
MIN_RECALL = 0.6               # share of those controls OCR must still read, else its "not found" is not trusted
NEAR_RATIO = 0.85              # similarity at which a value counts as still readable although OCR garbled a character
NEAR_MIN_LENGTH = 5            # shorter values are matched exactly only


def _box(box: list[float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    return max(0, x0), max(0, y0), max(x0 + 1, x1), max(y0 + 1, y1)


def mask_region(image: Image.Image, box: list[float], kind: str, rng: random.Random) -> Image.Image:
    """Makes the box unreadable in the given way, and leaves every other pixel alone."""
    if kind not in MASKS:
        raise ValueError(f"unknown mask {kind!r}")
    out = image.convert("RGB").copy()
    x0, y0, x1, y1 = _box(box)
    region = out.crop((x0, y0, x1, y1))
    if kind == "blur":
        region = region.filter(ImageFilter.GaussianBlur(radius=rng.uniform(3.0, 6.0)))
    elif kind == "downscale":
        w, h = region.size
        factor = rng.choice((4, 6, 8))
        region = region.resize((max(1, w // factor), max(1, h // factor)), Image.Resampling.BILINEAR)
        region = region.resize((w, h), Image.Resampling.NEAREST)
    elif kind == "noise":
        import numpy as np
        gen = np.random.default_rng(rng.getrandbits(32))
        arr = np.asarray(region, dtype=np.float32) + gen.normal(0, 70, size=(region.height, region.width, 3))
        region = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    elif kind == "bar":
        region = Image.new("RGB", region.size, (0, 0, 0))
    else:                                                      # stain: a grey ink blot, not a box, so not a giveaway
        draw = ImageDraw.Draw(region)
        w, h = region.size
        shade = rng.randint(90, 150)
        draw.ellipse((0, 0, w - 1, h - 1), fill=(shade, shade, shade))
    out.paste(region, (x0, y0))
    return out


_ENGINE = None


def rapidocr_lines(image: Image.Image) -> list[dict[str, Any]]:
    """OCR lines with pixel boxes from RapidOCR (already a dependency; the models ship with the package, so it runs
    offline). The polygon is reduced to its enclosing box, which is what the masks need."""
    global _ENGINE
    if _ENGINE is None:
        from rapidocr import RapidOCR
        _ENGINE = RapidOCR()
    result = _ENGINE(image.convert("RGB"))
    if not result or result.txts is None:
        return []
    lines = []
    for text, polygon in zip(result.txts, result.boxes):
        xs, ys = [float(pt[0]) for pt in polygon], [float(pt[1]) for pt in polygon]
        lines.append({"text": text, "box_px": [min(xs), min(ys), max(xs), max(ys)]})
    return lines


def rapidocr_text(image: Image.Image) -> str:
    return " ".join(line["text"] for line in rapidocr_lines(image))


def _tokens(text: Any) -> list[str]:
    return re.findall(r"\w+", fold(text))


def _runs(tokens: list[str], needle: str, line_text: str) -> int:
    """How many times the value is in this line: once when the line is exactly the value, else once per run of its
    word tokens in the line's tokens (so `100` is not in `1100` and `2024` is not in `12024`)."""
    if squash(line_text) == needle:
        return 1
    line, n = _tokens(line_text), len(tokens)
    return sum(1 for k in range(len(line) - n + 1) if line[k:k + n] == tokens) if n else 0


def localise(leaves: dict[str, Any], lines: list[dict[str, Any]],
             stats: collections.Counter | None = None) -> dict[str, list[float]]:
    """Box of every value that OCR finds in exactly one place. A value found more than once (in several lines or
    twice in one) is left out: masking one copy would leave the other readable. So are values OCR did not read as
    written, null ones and very short ones."""
    stats = collections.Counter() if stats is None else stats
    found: dict[str, list[float]] = {}
    for path, value in leaves.items():
        if is_null(value) or len(squash(value)) < MIN_VALUE_LENGTH:
            continue
        needle, tokens = squash(value), _tokens(value)
        counts = [(line, _runs(tokens, needle, line.get("text", ""))) for line in lines]
        hits = [line for line, n in counts if n]
        if not hits:
            stats["not_found"] += 1
        elif len(hits) > 1 or sum(n for _, n in counts) > 1:
            stats["ambiguous"] += 1
        else:
            found[path] = list(hits[0]["box_px"])
            stats["localised"] += 1
    return found


def _near(needle: str, hay: str) -> bool:
    n = len(needle)
    if len(hay) < n - 2:
        return False
    for start in range(0, max(1, len(hay) - n + 1)):
        window = hay[start:start + n]
        matcher = difflib.SequenceMatcher(None, needle, window)
        if matcher.real_quick_ratio() >= NEAR_RATIO and matcher.quick_ratio() >= NEAR_RATIO \
                and matcher.ratio() >= NEAR_RATIO:
            return True
    return False


def still_readable(value: Any, texts: list[str]) -> bool:
    """Is the value on the page, in any of these OCR texts? Exact on the squashed text, or (from `NEAR_MIN_LENGTH`
    characters) nearly: OCR garbling one character must not turn a readable field into an "unreadable" one. Being
    generous is the safe direction, since the harm is a wrong "unreadable" label."""
    needle = squash(value)
    if not needle:
        return False
    for text in texts:
        hay = squash(text)
        if needle in hay or (len(needle) >= NEAR_MIN_LENGTH and _near(needle, hay[:20000])):
            return True
    return False


def _decoys(size: tuple[int, int], taken: list[list[float]], rng: random.Random) -> list[list[float]]:
    width, height = size
    boxes = []
    for _ in range(rng.choice(DECOYS)):
        for _attempt in range(20):
            w = rng.randint(max(4, width // 12), max(5, width // 4))
            h = rng.randint(max(4, height // 40), max(5, height // 12))
            x0, y0 = rng.randint(0, max(0, width - w)), rng.randint(0, max(0, height - h))
            box = [x0, y0, x0 + w, y0 + h]
            if all(box[2] <= t[0] or box[0] >= t[2] or box[3] <= t[1] or box[1] >= t[3] for t in taken):
                boxes.append(box)
                taken = taken + [box]
                break
    return boxes


def build_example(base: dict[str, Any], rng: random.Random, ocr_lines: Callable[[Image.Image], list[dict]],
                  ocr_text: Callable[[Image.Image], str], level: str = "strong", p_degrade: float = 1.0,
                  eligible: set[str] | dict[str, str] | None = None, p_clean: float = 0.25,
                  stats: collections.Counter | None = None) -> dict[str, Any]:
    """One RL example. `eligible` limits the fields that may be masked (e.g. the extracted, high-confidence ones); as
    a dict {path: text as printed on the page} the printed text is what OCR is searched for. `stats` counts what the
    OCR guards did (localised / ambiguous / flipped back to readable / downgraded because OCR was blind).
    `p_clean` is the chance that no field is masked at all (the page may still be turned and degraded): without such
    examples nearly every example holds an unreadable field and "abstain" looks safer than it is.
    Returns the prompt and label for the RL format and the metadata the reward needs."""
    if len(base["images"]) != 1:
        raise ValueError("RL v1 takes single-page examples (rotate and mask one page); filter the others out")
    schema = schema_from_prompt(base["prompt"])
    reference = json.loads(base["label"])
    leaves = {p: v for p, v in flatten(reference).items()}
    pages = [img.convert("RGB") for img in base["images"]]
    lines = ocr_lines(pages[0])
    stats = collections.Counter() if stats is None else stats
    stats["examples"] += 1

    # 1. choose and mask on the original pages
    candidates = [p for p in leaves if (eligible is None or p in eligible)]
    printed = eligible if isinstance(eligible, dict) else {}
    needles = {p: (printed.get(p) or leaves[p]) for p in candidates}      # what OCR should find: the printed text
    boxes = localise(needles, lines, stats)
    clean = rng.random() < p_clean
    chosen = [] if clean else [p for p in sorted(boxes) if rng.random() < TARGET_FRACTION]
    masks: dict[str, str] = {p: rng.choice(MASKS) for p in chosen}
    page = pages[0]
    for path in chosen:
        page = mask_region(page, boxes[path], masks[path], rng)
    for box in _decoys(page.size, [boxes[p] for p in chosen], rng):
        page = mask_region(page, box, rng.choice(MASKS), rng)
    stats["masked"] += len(chosen)
    masked_text = ocr_text(page)                                # the masked page as OCR reads it, before it is turned

    # 2. turn and degrade the masked page
    plan = plan_example(rng, level, 1, p=p_degrade)
    if plan is not None:
        page = render(page, plan[0])
    upright = upright_turn(plan[0]) if plan is not None else 0.0
    pages = [page]

    # 3. state of every field: masked ones by mask kind, unless the value is still readable on either page, and
    #    "unreadable" only when OCR proves it can read this page (the unmasked fields are the controls)
    final_text = masked_text if plan is None else ocr_text(pages[0])
    controls = [p for p in boxes if p not in masks]
    recall = sum(still_readable(needles[p], [final_text]) for p in controls) / len(controls) if controls else None
    blind = recall is None or len(controls) < MIN_CONTROLS or recall < MIN_RECALL
    if masks and blind:
        stats["blind_examples"] += 1
    states: dict[str, str] = {}
    tags: dict[str, str] = {}
    for path, kind in masks.items():
        if still_readable(needles[path], [masked_text, final_text]):
            stats["flipped_readable"] += 1                      # still there: the model should read it, not abstain
            continue
        state = MASK_STATE[kind]
        if state == "must_abstain" and blind:
            state = "partial"                                   # OCR cannot vouch for this page: no penalty either way
            stats["downgraded_blind"] += 1
        states[path] = state
        tags[path] = MASK_TAG[kind]
    for path in leaves:
        states.setdefault(path, "answerable")

    # 4. the RL prompt and label
    spec = sample_spec(rng)
    wrapped = wrap_schema(schema, spec) if schema is not None else None
    prompt = build_prompt(base["prompt"], wrapped, spec) if wrapped is not None else base["prompt"]
    reasons = {p: tags[p] for p in tags if states[p] != "answerable"}
    label = wrap_answer(reference, schema, spec, upright=upright, reasons=reasons) if schema is not None \
        else base["label"]
    stats.update(f"state_{state}" for state in states.values())
    return {"id": base["id"], "prompt": prompt, "label": label, "images": pages, "reference": base["label"],
            "ocr_recall": recall,
            "states": states, "tags": tags, "upright": upright, "spec": spec.to_dict() if spec else None,
            "degradation": plan}
