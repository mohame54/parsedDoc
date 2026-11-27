"""Builds RL examples from a labelled document: the page is partly made unreadable on purpose, then rotated and
degraded, and every reference field gets a state the reward uses (see train/rl_reward.py).

Order matters: masks are drawn on the ORIGINAL page (so their boxes need no mapping), then the page is turned and
degraded (train/degrade.py), then the result is OCR'd again. A field whose value is still readable anywhere on the
final page is `answerable`, not unreadable: the model must not be punished for reading a value that is there.

    example = build_example(base, rng, ocr_lines=lines_fn, ocr_text=text_fn, eligible=paths)

    base: {"id", "prompt" (with the plain JSON Schema), "label" (the plain reference JSON), "images": [PIL pages]}
    ocr_lines(image) -> [{"text", "box_px": [x0, y0, x1, y1]}]   (RapidOCR; injected so tests need no model)
    ocr_text(image)  -> str                                       (the same OCR, joined; for the readability re-check)
"""
from __future__ import annotations

import json
import random
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFilter

from .degrade import plan_example, render, upright_turn
from .gen_eval import flatten, is_null, schema_from_prompt, squash
from .rl_prompts import build_prompt, sample_spec, wrap_answer, wrap_schema

MASKS = ("blur", "downscale", "noise", "bar", "stain")
MASK_STATE = {"blur": "partial", "downscale": "partial", "noise": "must_abstain", "bar": "must_abstain",
              "stain": "must_abstain"}
MASK_TAG = {"blur": "blurry", "downscale": "blurry", "noise": "noisy", "bar": "occluded", "stain": "occluded"}
TARGET_FRACTION = 0.3          # share of the localised fields that get masked on an example
MIN_VALUE_LENGTH = 3           # shorter values are too easy to find again by accident, so they are not masked
DECOYS = (1, 2)                # decoy masks per page: they carry no target, so a mask alone is not a signal


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


def localise(leaves: dict[str, Any], lines: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Box of every leaf value in the OCR lines: the first line that contains the value (case, accents, spacing and
    punctuation ignored). Values that are not on the page as OCR read them are left out."""
    found: dict[str, list[float]] = {}
    for path, value in leaves.items():
        if is_null(value) or len(squash(value)) < MIN_VALUE_LENGTH:
            continue
        needle = squash(value)
        for line in lines:
            if needle and needle in squash(line.get("text", "")):
                found[path] = list(line["box_px"])
                break
    return found


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
                  eligible: set[str] | None = None, p_clean: float = 0.25) -> dict[str, Any]:
    """One RL example. `eligible` limits the fields that may be masked (e.g. the extracted, high-confidence ones).
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

    # 1. choose and mask on the original pages
    candidates = [p for p in leaves if (eligible is None or p in eligible)]
    boxes = localise({p: leaves[p] for p in candidates}, lines)
    clean = rng.random() < p_clean
    chosen = [] if clean else [p for p in sorted(boxes) if rng.random() < TARGET_FRACTION]
    masks: dict[str, str] = {p: rng.choice(MASKS) for p in chosen}
    page = pages[0]
    for path in chosen:
        page = mask_region(page, boxes[path], masks[path], rng)
    for box in _decoys(page.size, [boxes[p] for p in chosen], rng):
        page = mask_region(page, box, rng.choice(MASKS), rng)

    # 2. turn and degrade the masked page
    plan = plan_example(rng, level, 1, p=p_degrade)
    if plan is not None:
        page = render(page, plan[0])
    upright = upright_turn(plan[0]) if plan is not None else 0.0
    pages = [page]

    # 3. state of every field: masked ones by mask kind, unless the value is still readable on the final page
    final_text = ocr_text(pages[0])
    states: dict[str, str] = {}
    tags: dict[str, str] = {}
    for path, kind in masks.items():
        if squash(leaves[path]) in squash(final_text):
            continue                                            # still there: the model should read it, not abstain
        states[path] = MASK_STATE[kind]
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
    return {"id": base["id"], "prompt": prompt, "label": label, "images": pages, "reference": base["label"],
            "states": states, "tags": tags, "upright": upright, "spec": spec.to_dict() if spec else None,
            "degradation": plan}
