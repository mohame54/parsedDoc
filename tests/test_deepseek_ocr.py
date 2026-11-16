import types

import pytest
import torch
from PIL import Image

from train.dataset import TransformCollator, make_loader
from train.processors import (DatasetProcessor, DeepSeekOCRCollator, DeepSeekOCRV1Transform,
                              DeepSeekOCRV2Transform, SequenceTooLong, V1_SPEC, V2_SPEC, create_transform,
                              merge_page_images, model_family, to_device)
from train.processors.deepseekOcr import dynamic_preprocess, image_token_count, tile_grid  # noqa: F401
from train.processors.deepseekOcr import BOS_ID, EOS_ID, IMAGE_TOKEN_ID  # noqa: F401


class StubTokenizer:
    pad_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [10 + ord(c) % 50 for c in text]


def _t(version="v1", **kw):
    cls = DeepSeekOCRV2Transform if version == "v2" else DeepSeekOCRV1Transform
    return cls(tokenizer=StubTokenizer(), **kw)


def _page(w=760, h=1000):
    return Image.new("RGB", (w, h), "white")


def test_token_counts_match_the_models_layout():
    assert [image_token_count(V1_SPEC, m) for m in ("tiny", "small", "base", "large")] == [73, 111, 273, 421]
    assert image_token_count(V1_SPEC, "gundam") == 273 and image_token_count(V1_SPEC, "gundam", (2, 2)) == 273 + 21 * 20


def test_base_mode_layout_masks_and_shapes():
    out = _t(mode="base")([_page()] * 4, "Extract.", "ANSWER")
    ids, labels, mask = out["input_ids"], out["labels"], out["images_seq_mask"]
    assert ids[0] == BOS_ID and mask.sum() == 273 and (ids[mask] == IMAGE_TOKEN_ID).all()
    assert mask[1:274].all() and not mask[274:].any() and not mask[0]
    assert (labels[: 1 + 273] == -100).all()                          # bos, image and prompt are never supervised
    assert labels[labels != -100].tolist()[-1] == EOS_ID
    assert (labels != -100).sum() == len("ANSWER") + 1                  # answer + eos
    patches, global_view = out["images"]
    assert patches.shape == (1, 3, 1024, 1024) and global_view.shape == (1, 3, 1024, 1024)
    assert out["images_spatial_crop"].tolist() == [[1, 1]]
    assert global_view.dtype == torch.bfloat16


def test_gundam_tiles_follow_the_image_and_token_count_adds_up():
    out = _t(mode="gundam", merge_pages="grid")([_page()] * 4, "Extract.", "A")
    patches, _ = out["images"]
    w, h = out["images_spatial_crop"][0].tolist()
    assert patches.shape == (w * h, 3, 640, 640) and 2 <= w * h <= 9
    assert int(out["images_seq_mask"].sum()) == image_token_count(V1_SPEC, "gundam", (w, h))
    small = _t(mode="gundam")(_page(400, 500), "Extract.", "A")      # <= 640 on both sides: no tiles
    assert small["images_spatial_crop"].tolist() == [[1, 1]] and small["images"][0].sum() == 0


def test_dynamic_preprocess_tiles_cover_the_grid():
    tiles, ratio = dynamic_preprocess(Image.new("RGB", (1900, 1000)))
    assert len(tiles) == ratio[0] * ratio[1] and all(t.size == (640, 640) for t in tiles)


def test_fixed_length_and_overflow():
    t = _t(mode="tiny", max_length=200)
    a, b = t(_page(), "Hi", "A"), t(_page(), "A longer prompt here", "BBBB")
    assert a["input_ids"].shape == b["input_ids"].shape == (200,)
    real = int(a["attention_mask"].sum())
    assert (a["input_ids"][real:] == 2).all() and (a["labels"][real:] == -100).all()
    assert not a["images_seq_mask"][real:].any()
    with pytest.raises(SequenceTooLong):
        _t(mode="base", max_length=100)(_page(), "Hi", "A")


def test_needs_exactly_one_image_after_merging():
    with pytest.raises(ValueError, match="one image"):
        _t(merge_pages=None)([_page(), _page()], "Hi", "A")
    with pytest.raises(ValueError):
        _t()(None, "Hi", "A")
    assert _t(merge_pages=None)(_page(), "Hi", "A")["images_seq_mask"].sum() == 273


def test_grid_merge_is_near_square_for_four_pages():
    (grid,) = merge_page_images([_page(100, 200)] * 4, "grid")
    assert grid.size == (208, 408)                                    # 2x2 cells plus 8 px dividers
    (two,) = merge_page_images([_page(100, 200)] * 2, "grid")
    assert two.size == (208, 200)
    assert merge_page_images([_page()], "grid")[0].size == (760, 1000)


def test_collator_keeps_images_as_a_list_of_tuples_and_moves_to_device():
    t = _t(mode="tiny")
    a, b = t(_page(), "Hi", "A"), t(_page(), "A longer prompt", "BBBBBB")
    batch = DeepSeekOCRCollator(2)([a, b])
    n = max(a["input_ids"].shape[0], b["input_ids"].shape[0])
    assert batch["input_ids"].shape == batch["labels"].shape == batch["images_seq_mask"].shape == (2, n)
    assert isinstance(batch["images"], list) and len(batch["images"]) == 2 and len(batch["images"][0]) == 2
    assert batch["images_spatial_crop"].shape == (2, 2)
    moved = to_device(batch, "cpu")
    assert isinstance(moved["images"][1], tuple)


def test_loader_uses_the_deepseek_collator(tmp_path):
    import json
    from tests.test_derive import _record
    _page(40, 40).save(tmp_path / "p.png")
    (tmp_path / "train.jsonl").write_text(json.dumps(_record()))
    transform = _t(mode="tiny", max_length=400)
    loader = make_loader(tmp_path, "train", DatasetProcessor(), transform=transform, batch_size=2,
                         samples_per_document=2, on_too_long="replace")
    assert isinstance(loader.collate_fn.collator, DeepSeekOCRCollator)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (2, 400) and len(batch["images"]) == 2


def test_v2_token_counts_have_no_row_separators():
    assert [image_token_count(V2_SPEC, m) for m in ("small", "base")] == [145, 257]
    assert image_token_count(V2_SPEC, "gundam", (2, 3)) == 257 + 144 * 6
    assert image_token_count(V2_SPEC, "gundam") == 257


def test_version_is_detected_from_the_model_id_and_modes_are_checked():
    assert model_family("deepseek-ai/DeepSeek-OCR-2") == "deepseek-v2"
    assert model_family("deepseek-ai/DeepSeek-OCR") == "deepseek-v1"
    assert model_family("Qwen/Qwen2.5-VL-3B-Instruct") == "instruct"
    assert isinstance(create_transform("deepseek-ai/DeepSeek-OCR-2", tokenizer=StubTokenizer()), DeepSeekOCRV2Transform)
    assert isinstance(create_transform("deepseek-ai/DeepSeek-OCR", tokenizer=StubTokenizer()), DeepSeekOCRV1Transform)
    assert _t(mode="base", version="v2").version == "v2"
    with pytest.raises(ValueError, match="small"):
        _t(mode="large", version="v2")                  # v2 has no 1280 view


def test_v2_uses_768_tiles_max_six_and_matches_its_own_count():
    out = _t(mode="gundam", version="v2", merge_pages="grid")([_page()] * 4, "Extract.", "A")
    patches, global_view = out["images"]
    w, h = out["images_spatial_crop"][0].tolist()
    assert patches.shape == (w * h, 3, 768, 768) and 2 <= w * h <= 6
    assert global_view.shape == (1, 3, 1024, 1024)
    assert int(out["images_seq_mask"].sum()) == 257 + 144 * w * h
    base = _t(mode="base", version="v2")(_page(), "Hi", "A")
    assert int(base["images_seq_mask"].sum()) == 257 and base["images"][0].sum() == 0
    small = _t(mode="small", version="v2")(_page(), "Hi", "A")
    assert small["images"][1].shape == (1, 3, 768, 768) and int(small["images_seq_mask"].sum()) == 145
    tiny_page = _t(mode="gundam", version="v2")(_page(700, 700), "Hi", "A")     # <= 768 both sides: no tiles
    assert tiny_page["images_spatial_crop"].tolist() == [[1, 1]]


def test_merged_size_matches_the_real_merge_without_pixels():
    from train.processors import merged_size
    sizes = [(100, 200), (150, 120), (90, 300), (200, 80)]
    for mode in ("vertical", "horizontal", "grid"):
        for n in (1, 2, 3, 4):
            pages = [_page(w, h) for w, h in sizes[:n]]
            assert merged_size(sizes[:n], mode) == merge_page_images(pages, mode)[0].size, (mode, n)


@pytest.mark.parametrize("version,mode", [("v1", "base"), ("v1", "gundam"), ("v1", "tiny"), ("v2", "gundam"), ("v2", "small")])
def test_token_length_and_fits_agree_exactly_with_the_real_transform(version, mode):
    t = _t(mode=mode, version=version, merge_pages="grid")
    for sizes in ([(760, 1000)], [(760, 1000)] * 4, [(300, 400), (1600, 900)], [(500, 500)] * 2):
        pages = [_page(w, h) for w, h in sizes]
        real = t(pages, "Extract the data.", "A label of some length " * 3)["input_ids"].shape[0]
        assert t.token_length(sizes, "Extract the data.", "A label of some length " * 3) == real
        t.max_length = real
        assert t.fits(sizes, "Extract the data.", "A label of some length " * 3)
        t.max_length = real - 1
        assert not t.fits(sizes, "Extract the data.", "A label of some length " * 3)
        t.max_length = None


def _long_doc(i, n_fields):
    field = lambda k: {"name": f"field_{k}", "section": "s", "data_type": "string", "raw_value": f"value {k}",  # noqa: E731
                       "normalized_value": f"value {k}", "status": "extracted", "confidence": "high"}
    return {"id": f"d:{i}", "images": ["p.png"], "text": ["Hello " * 5], "document_type": "form", "language": "en",
            "fields": [field(k) for k in range(n_fields)], "tables": [], "visual_elements": [], "relationships": []}


def test_over_long_examples_are_dropped_not_raised_and_nothing_longer_than_the_limit_is_returned(tmp_path):
    import json
    _page(40, 40).save(tmp_path / "p.png")
    (tmp_path / "train.jsonl").write_text("\n".join(json.dumps(_long_doc(i, 3 if i % 2 else 40)) for i in range(6)))
    transform = _t(mode="tiny", max_length=330)
    loader = make_loader(tmp_path, "train", DatasetProcessor(), transform=transform, batch_size=3,
                         samples_per_document=8, shuffle=False)
    ds = loader.dataset
    tasks = set()
    for epoch in range(3):
        loader.dataset.set_epoch(epoch)
        for batch in loader:                                          # would raise SequenceTooLong before
            assert batch["input_ids"].shape == (3, 330)
        tasks |= {ds[i]["task"] for i in range(len(ds))}
    assert ds.dropped_examples > 0                                    # the long ones were left out of the mix
    assert "ocr" in tasks and len(tasks) > 2                          # the rest of the mix is untouched
    for i in range(len(ds)):
        item = ds[i]
        assert transform.token_length([im.size for im in item["images"]], item["prompt"], item["label"]) <= 330


def test_document_with_nothing_that_fits_is_skipped_and_impossible_limits_fail_clearly(tmp_path):
    import json
    _page(40, 40).save(tmp_path / "p.png")
    (tmp_path / "train.jsonl").write_text(json.dumps(_long_doc(0, 3)))
    ok = make_loader(tmp_path, "train", DatasetProcessor(), transform=_t(mode="tiny", max_length=330))
    assert ok.dataset[0]["id"].startswith("d:0")
    impossible = make_loader(tmp_path, "train", DatasetProcessor(), transform=_t(mode="tiny", max_length=40))
    with pytest.raises(RuntimeError, match="raise max_length"):
        impossible.dataset[0]
    # a document whose every example is too long (or supports no task) is skipped in favour of the next one
    (tmp_path / "train.jsonl").write_text("\n".join(json.dumps(r) for r in (
        dict(_long_doc(1, 3), text=["x" * 500]), dict(_long_doc(3, 0), text=[""]), _long_doc(2, 3))))
    skip = make_loader(tmp_path, "train", DatasetProcessor(tasks=["ocr"]), transform=_t(mode="tiny", max_length=330))
    ids = {skip.dataset[i]["id"].split("#")[0] for i in range(6)}
    assert ids == {"d:2"} and skip.dataset.skipped_documents > 0


def test_generic_datatransform_fits_matches_its_own_limit():
    from tests.test_train_batching import _transform
    sizes = [(32, 32)]
    t = _transform(max_length=24)
    assert t.fits(sizes, "Hi", "ok") and not t.fits(sizes, "Hi", "x" * 100)
    assert _transform().fits(sizes, "Hi", "x" * 100)                  # no limit -> always fits
