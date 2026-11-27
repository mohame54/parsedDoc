"""Synthetic degradation: deterministic per seed, sizes predicted exactly, the upright turn is consistent."""
import random

import pytest
from PIL import Image

from train.degrade import degrade, degraded_size, draw, level_for, rotated_size, upright_turn


def _page(w=120, h=80):
    img = Image.new("RGB", (w, h), (255, 255, 255))
    for y in range(10, h - 10, 12):
        for x in range(10, w - 30):
            img.putpixel((x, y), (0, 0, 0))
    return img


@pytest.mark.parametrize("angle", [0, 3.5, 17, 45, 90, 135, 180, 270, 301.2])
@pytest.mark.parametrize("size", [(120, 80), (64, 200), (333, 97)])
def test_predicted_size_matches_the_real_rotation(angle, size):
    w, h = size
    img = Image.new("RGB", size, (255, 255, 255))
    from train.degrade import _turn
    assert _turn(img, angle).size == rotated_size(w, h, angle)


def test_same_seed_same_degradation_and_different_seed_differs():
    a, info_a = degrade(_page(), random.Random("7:0:3"), "strong")
    b, info_b = degrade(_page(), random.Random("7:0:3"), "strong")
    assert info_a == info_b and a.tobytes() == b.tobytes()
    assert any(degrade(_page(), random.Random(s), "strong")[1] != info_a for s in ("8", "9", "10"))


def test_size_prediction_holds_for_every_level():
    for level in ("off", "soft", "strong"):
        for seed in range(12):
            out, info = degrade(_page(), random.Random(seed), level)
            assert out.size == degraded_size(120, 80, info), (level, seed, info)
            assert out.mode == "RGB"


def test_off_leaves_the_page_untouched():
    page = _page()
    out, info = degrade(page, random.Random(1), "off")
    assert info["angle"] == 0 and info["noise"] == 0 and info["blur"] == 0 and info["jpeg"] == 100
    assert out.tobytes() == page.tobytes()


def test_upright_turn_undoes_the_applied_turn():
    for angle in (0, 90, 180, 270, 33.3):
        info = {"angle": angle}
        assert (angle + upright_turn(info)) % 360 == pytest.approx(0, abs=1e-6) or \
            (angle + upright_turn(info)) % 360 == pytest.approx(360, abs=1e-6)
    assert upright_turn({"angle": 90}) == 270


def test_quarter_turns_are_exact_and_strong_draws_cover_them():
    rng = random.Random(0)
    angles = {draw(rng, level_for("strong"))["angle"] for _ in range(400)}
    assert {90.0, 180.0, 270.0} <= angles
    assert all(0 <= a < 360 for a in angles)


def test_soft_keeps_small_skew_only():
    rng = random.Random(5)
    for _ in range(200):
        p = draw(rng, level_for("soft"))
        assert min(p["angle"], 360 - p["angle"]) <= 4.0 + 1e-6


def test_a_plan_made_from_the_seed_alone_predicts_every_rendered_size():
    from train.degrade import plan_example, render
    for seed in range(40):
        pages = [_page(90 + seed, 60), _page(50, 130)]
        plan = plan_example(random.Random(f"s:{seed}"), "strong", len(pages), p=1.0)
        for image, info in zip(pages, plan):
            assert render(image, info).size == degraded_size(image.width, image.height, info)


def test_plan_is_none_when_the_example_is_left_clean_and_deterministic_otherwise():
    from train.degrade import plan_example
    assert plan_example(random.Random(1), "soft", 1, p=0.0) is None
    a = plan_example(random.Random("x"), "soft", 2, p=1.0)
    assert a == plan_example(random.Random("x"), "soft", 2, p=1.0)


def test_dataset_planning_sees_the_same_sizes_as_the_degraded_items(tmp_path):
    """length_of predicts the page sizes from headers and the seed; __getitem__ must return exactly those sizes."""
    import json

    from train.dataset import DocumentDataset
    from train.processors import DatasetProcessor
    from tests.test_derive import _record
    record = _record()
    rel = record["images"][0]
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (90, 60), (255, 255, 255)).save(tmp_path / rel)
    rows = [dict(record, id=f"doc:{i}") for i in range(3)]
    (tmp_path / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    ds = DocumentDataset(tmp_path, "train", DatasetProcessor(preset="structured"), augment="strong", augment_p=1.0)
    rotated_seen = False
    for index in range(len(ds)):
        item = ds[(0, index)]
        assert "augment" in item
        planned = ds.length_of(0, index, lambda sz, p, l: sum(w * h for w, h in sz))
        assert planned == sum(im.width * im.height for im in item["images"])
        rotated_seen |= item["images"][0].size != (90, 60)
    assert rotated_seen


def test_augment_p_is_the_share_of_examples_that_get_degraded(tmp_path):
    import json

    from train.dataset import DocumentDataset
    from train.processors import DatasetProcessor
    from tests.test_derive import _record
    record = _record()
    rel = record["images"][0]
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), (255, 255, 255)).save(tmp_path / rel)
    (tmp_path / "train.jsonl").write_text("\n".join(json.dumps(dict(record, id=f"doc:{i}")) for i in range(10)))
    for share in (0.0, 0.2, 1.0):
        ds = DocumentDataset(tmp_path, "train", DatasetProcessor(preset="structured"), samples_per_document=60,
                             augment="soft", augment_p=share)
        degraded = sum("augment" in ds[(0, i)] for i in range(len(ds)))
        assert abs(degraded / len(ds) - share) < 0.06, (share, degraded / len(ds))
