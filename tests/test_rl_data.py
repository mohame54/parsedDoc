"""RL examples: masks change only their box, readable values are never marked unreadable, labels validate against the
wrapped schema, and the rotation recorded is the one applied."""
import json
import random

import pytest
from PIL import Image

from train.degrade import upright_turn
from train.gen_eval import schema_from_prompt, schema_score
from train.rl_data import MASKS, build_example, localise, mask_region

SCHEMA = {"type": "object", "properties": {"name": {"type": ["string", "null"]},
                                           "date": {"type": ["string", "null"]},
                                           "total": {"type": ["number", "null"]}},
          "required": ["name", "date", "total"], "additionalProperties": False}
REF = {"name": "Ana Lopez", "date": "2017-06-30", "total": 12.5}
LINES = [{"text": "Ana Lopez", "box_px": [20, 20, 160, 44]},
         {"text": "2017-06-30", "box_px": [20, 70, 160, 94]},
         {"text": "12.5", "box_px": [20, 120, 80, 144]}]
BASE_PROMPT = "Extract the data. Use null when unreadable.\n\nJSON Schema:\n" + json.dumps(SCHEMA)


def page():
    return Image.new("RGB", (300, 200), (255, 255, 255))


def base(**kw):
    return dict({"id": "doc:1", "prompt": BASE_PROMPT, "label": json.dumps(REF), "images": [page()]}, **kw)


def text_lines(image):
    return LINES


def no_text(image):
    return ""


@pytest.mark.parametrize("kind", MASKS)
def test_a_mask_changes_only_its_box(kind):
    box = [40, 40, 120, 80]
    src = Image.new("RGB", (300, 200), (255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(src)
    for y in range(0, 200, 6):                       # text-like stripes, so blur and downscale have something to change
        draw.line([(0, y), (300, y)], fill=(0, 0, 0))
    out = mask_region(src, box, kind, random.Random(1))
    changed = [(x, y) for x in range(300) for y in range(200) if out.getpixel((x, y)) != src.getpixel((x, y))]
    assert changed, kind
    assert all(40 <= x < 120 and 40 <= y < 80 for x, y in changed), kind


def test_localise_finds_readable_values_and_skips_null_and_short_ones():
    found = localise({"name": "Ana Lopez", "date": "2017-06-30", "gap": None, "tiny": "ab"}, LINES)
    assert set(found) == {"name", "date"} and found["name"] == [20, 20, 160, 44]


def test_a_readable_value_is_never_marked_unreadable():
    ex = build_example(base(), random.Random(0), text_lines, lambda img: "Ana Lopez 2017-06-30 12.5",
                       level="off", p_degrade=0.0)
    assert set(ex["states"].values()) == {"answerable"} and ex["tags"] == {}


def test_masked_fields_are_marked_and_the_label_validates_against_the_wrapped_schema():
    masked_seen = set()
    for seed in range(12):
        ex = build_example(base(), random.Random(seed), text_lines, no_text, level="off", p_degrade=0.0, p_clean=0.0)
        masked = [p for p, s in ex["states"].items() if s != "answerable"]
        masked_seen |= set(masked)
        assert set(masked) <= set(REF) and all(p in ex["tags"] for p in masked)
        schema = schema_from_prompt(ex["prompt"])
        assert schema_score(json.loads(ex["label"]), schema) == 1.0
        assert json.loads(ex["reference"]) == REF          # the plain reference is never changed
    assert masked_seen == set(REF), "over the seeds every field gets masked at least once"


def test_the_upright_turn_is_the_one_applied(tmp_path):
    ex = build_example(base(), random.Random(3), text_lines, no_text, level="strong", p_degrade=1.0)
    assert ex["upright"] == upright_turn(ex["degradation"][0])
    answer = json.loads(ex["label"])
    if ex["spec"] and ex["spec"]["rotation_key"]:
        assert answer[ex["spec"]["rotation_key"]] == ex["upright"]


def test_eligible_limits_which_fields_may_be_masked():
    for seed in range(20):
        ex = build_example(base(), random.Random(seed), text_lines, no_text, level="off", p_degrade=0.0,
                           eligible={"total"})
        assert {p for p, s in ex["states"].items() if s != "answerable"} <= {"total"}


def test_multi_page_examples_are_refused_in_v1():
    with pytest.raises(ValueError):
        build_example(base(images=[page(), page()]), random.Random(0), text_lines, no_text)


def test_the_same_seed_gives_the_same_example():
    a = build_example(base(), random.Random("s"), text_lines, no_text)
    b = build_example(base(), random.Random("s"), text_lines, no_text)
    assert a["prompt"] == b["prompt"] and a["label"] == b["label"] and a["states"] == b["states"]
    assert a["images"][0].tobytes() == b["images"][0].tobytes()


def test_rapidocr_reads_a_rendered_line_and_gives_its_box():
    from PIL import ImageDraw, ImageFont

    from train.rl_data import rapidocr_lines
    img = Image.new("RGB", (400, 120), "white")
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text((20, 30), "Ana Lopez 12.5", fill="black", font=font)
    lines = rapidocr_lines(img)
    assert any("Lopez" in line["text"] for line in lines)
    x0, y0, x1, y1 = next(line["box_px"] for line in lines if "Lopez" in line["text"])
    assert 0 <= x0 < x1 <= 400 and 0 <= y0 < y1 <= 120


def test_p_clean_leaves_the_fields_unmasked():
    for seed in range(15):
        ex = build_example(base(), random.Random(seed), text_lines, no_text, level="off", p_degrade=0.0, p_clean=1.0)
        assert set(ex["states"].values()) == {"answerable"}
