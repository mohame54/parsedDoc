"""Transform + collator + loader, with a stand-in for the Hugging Face processor (no downloads)."""
import json
import types

import pytest
import torch
from PIL import Image

from train.dataset import make_loader
from train.processors import DatasetProcessor, InstructCollator, InstructTransform

from .test_derive import _record


class FakeProcessor:
    """Role token, 4 image tokens per image, one token per character of text; same prefix behaviour as a
    real chat template (the generation prompt is the assistant role token)."""

    def __init__(self):
        self.tokenizer = types.SimpleNamespace(
            chat_template="x", pad_token_id=0, padding_side="right",
            encode=lambda text, add_special_tokens=False: [10 + ord(c) % 50 for c in text])

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict, return_tensors):
        ids, images = [], 0
        for message in messages:
            ids.append(1 if message["role"] == "user" else 2)
            content = message["content"]
            for part in [{"type": "text", "text": content}] if isinstance(content, str) else content:
                if part["type"] == "image":
                    ids += [9] * 4
                    images += 1
                else:
                    ids += [10 + ord(c) % 50 for c in part["text"]]
        if add_generation_prompt:
            ids.append(2)
        out = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        if images:
            out["pixel_values"] = torch.ones(images * 4, 3)
            out["image_grid_thw"] = torch.tensor([[1, 2, 2]] * images)
        return out


def _transform(**kw):
    return InstructTransform(processor=FakeProcessor(), **kw)


def test_transform_supervises_only_the_answer_for_multi_page_input():
    image = Image.new("RGB", (4, 4))
    out = _transform()([image, image], "Read it", "ANSWER")
    labels, ids = out["labels"], out["input_ids"]
    supervised = ids[labels != -100]
    assert len(supervised) == len("ANSWER")                      # prompt, role and 8 image tokens are masked
    assert out["pixel_values"].shape[0] == 8 and out["image_grid_thw"].shape == (2, 3)
    assert (labels[ids == 9] == -100).all()
    assert out["inp_mask"].sum() == len("ANSWER")


def test_transform_rejects_empty_and_overlong_examples():
    image = Image.new("RGB", (4, 4))
    with pytest.raises(ValueError):
        _transform()(image, "Read it", "  ")
    with pytest.raises(ValueError, match="max_length"):
        _transform(max_length=5)(image, "Read it", "ANSWER")


def test_collator_pads_text_and_concatenates_pixels():
    transform = _transform()
    image = Image.new("RGB", (4, 4))
    a, b = transform(image, "Hi", "A"), transform([image, image], "Longer prompt", "BBBB")
    batch = InstructCollator(transform.tokenizer)([a, b])
    longest = max(a["input_ids"].shape[0], b["input_ids"].shape[0])
    assert batch["input_ids"].shape == (2, longest) == batch["labels"].shape
    assert batch["pixel_values"].shape[0] == 4 + 8 and batch["image_grid_thw"].shape == (3, 3)
    assert (batch["labels"][0, a["input_ids"].shape[0]:] == -100).all()    # padding never supervised
    assert (batch["attention_mask"][0, a["input_ids"].shape[0]:] == 0).all()
    assert "messages" not in batch and "images" not in batch               # metadata stays out of forward()


def test_loader_transforms_in_collate_and_redraws(tmp_path):
    Image.new("RGB", (4, 4)).save(tmp_path / "p.png")
    (tmp_path / "train.jsonl").write_text(json.dumps(_record()))
    loader = make_loader(tmp_path, "train", DatasetProcessor(preset="structured"), batch_size=3,
                         transform=_transform(), samples_per_document=3, shuffle=False)
    item = loader.dataset[0]
    assert set(item) == {"id", "task", "prompt", "images", "label"}          # the dataset stays raw
    batch = next(iter(loader))
    assert batch["input_ids"].shape[0] == 3 and (batch["labels"] != -100).any()

    seen = set()
    for epoch in range(12):
        loader.dataset.set_epoch(epoch)
        seen.add(loader.dataset[0]["prompt"])
    assert len(seen) > 3

    plain = make_loader(tmp_path, "train", batch_size=2, samples_per_document=2)
    assert len(next(iter(plain))["prompt"]) == 2                              # no transform -> lists


def _many_docs(tmp_path, n=10):
    Image.new("RGB", (4, 4)).save(tmp_path / "p.png")
    rows = [dict(_record(), id=f"d:{i}") for i in range(n)]
    for split in ("train", "val"):
        (tmp_path / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


def _ids(loader):
    return [i for batch in loader for i in batch["id"]]


def test_ranks_get_disjoint_equal_shards_and_epochs_reshuffle(tmp_path):
    from train.dataset import set_epoch
    _many_docs(tmp_path)
    shards = [make_loader(tmp_path, "train", batch_size=2, rank=r, world_size=3) for r in range(3)]
    docs = lambda loader: [i.split("#")[0] for i in _ids(loader)]  # noqa: E731
    first = [docs(loader) for loader in shards]
    assert len({len(x) for x in first}) == 1                       # same number of samples on every rank
    assert all(len(x) % 2 == 0 for x in first)                     # drop_last: identical batch counts
    assert len(set(sum(first, []))) >= 9                           # shards cover the data (padding repeats <=2)

    for loader in shards:
        set_epoch(loader, 1)
    second = [docs(loader) for loader in shards]
    assert second != first                                         # new shuffle and new prompts per epoch
    again = make_loader(tmp_path, "train", batch_size=2, rank=0, world_size=3)
    set_epoch(again, 1)
    assert docs(again) == second[0]                                # reproducible across restarts


def test_eval_split_is_not_shuffled_and_keeps_every_document(tmp_path):
    _many_docs(tmp_path)
    seen = set()
    for r in range(2):
        loader = make_loader(tmp_path, "val", batch_size=2, rank=r, world_size=2)
        seen |= {i.split("#")[0] for i in _ids(loader)}
    assert seen == {f"d:{i}" for i in range(10)}                   # nothing dropped for evaluation


def test_epoch_reaches_workers_through_the_sampler(tmp_path):
    from train.dataset import set_epoch
    _many_docs(tmp_path, 4)
    loader = make_loader(tmp_path, "train", batch_size=4, num_workers=2, shuffle=False, persistent_workers=True)
    set_epoch(loader, 0)
    a = next(iter(loader))["prompt"]
    set_epoch(loader, 7)
    b = next(iter(loader))["prompt"]
    assert a != b                                                  # persistent workers still see the new epoch


def test_bad_rank_is_rejected(tmp_path):
    _many_docs(tmp_path)
    with pytest.raises(ValueError):
        make_loader(tmp_path, "train", rank=2, world_size=2)


def test_fixed_length_gives_one_shape_for_every_example():
    image = Image.new("RGB", (4, 4))
    transform = _transform(max_length=64)
    a, b = transform(image, "Hi", "A"), transform([image, image], "Longer prompt here", "BBBBBBBB")
    for out in (a, b):
        assert out["input_ids"].shape == out["labels"].shape == out["attention_mask"].shape == (64,)
        assert out["inp_mask"].shape == (64,)
    short = a["input_ids"]
    real = int(a["attention_mask"].sum())
    assert (short[real:] == 0).all() and (a["labels"][real:] == -100).all() and not a["inp_mask"][real:].any()
    assert (a["labels"] != -100).sum() == len("A")                      # padding adds no supervision
    batch = InstructCollator(transform.tokenizer)([a, b])
    assert batch["input_ids"].shape == (2, 64)


def test_fixed_length_off_keeps_variable_length():
    out = _transform(max_length=64, fixed_length=False)(Image.new("RGB", (4, 4)), "Hi", "A")
    assert out["input_ids"].shape[0] < 64


def test_image_size_letterboxes_every_page_to_one_size():
    transform = _transform(image_size=(32, 48))
    wide, tall = Image.new("RGB", (100, 20), "black"), Image.new("RGB", (20, 100), "black")
    out = transform([wide, tall], "Hi", "A")
    assert [img.size for img in out["images"]] == [(32, 48), (32, 48)]
    assert out["images"][0].getpixel((0, 0)) == (255, 255, 255)           # white padding, content centred
    assert out["images"][0].getpixel((16, 24)) == (0, 0, 0)


def test_overlong_examples_raise_or_are_replaced_without_changing_batch_shape(tmp_path):
    from train.dataset import TransformCollator
    from train.processors import SequenceTooLong
    image = Image.new("RGB", (4, 4))
    transform = _transform(max_length=24)
    batch = [{"images": [image], "prompt": "Hi", "label": "ok"},
             {"images": [image], "prompt": "Hi", "label": "x" * 100}]
    with pytest.raises(SequenceTooLong):
        TransformCollator(transform)(batch)
    replacing = TransformCollator(transform, on_too_long="replace")
    out = replacing(batch)
    assert out["input_ids"].shape == (2, 24) and replacing.replaced == 1
    with pytest.raises(SequenceTooLong):
        replacing([batch[1]])


def test_merge_pages_gives_one_image_with_a_divider():
    transform = _transform(merge_pages="vertical", image_size=(32, 64))
    page_a, page_b = Image.new("RGB", (20, 20), "black"), Image.new("RGB", (40, 40), "black")
    out = transform([page_a, page_b], "Hi", "A")
    assert len(out["images"]) == 1 and out["images"][0].size == (32, 64)
    assert out["image_grid_thw"].shape == (1, 3)                       # one image to the vision tower
    unfitted = _transform(merge_pages="vertical")([page_a, page_b], "Hi", "A")["images"][0]
    assert unfitted.size == (40, 40 + 40 + 8)                           # page_a scaled up to width 40, gap 8
    assert unfitted.getpixel((20, 44)) == (200, 200, 200)               # the divider between pages
    assert _transform(merge_pages="horizontal")([page_a, page_b], "Hi", "A")["images"][0].size == (88, 40)
    assert len(_transform()([page_a, page_b], "Hi", "A")["images"]) == 2   # off by default
    with pytest.raises(ValueError):
        _transform(merge_pages="diagonal")


def test_stats_report_and_token_lengths(tmp_path):
    from train.stats import image_report, length_report, merged_size
    assert merged_size([(100, 200), (50, 100)], "vertical") == (100, 200 + 200 + 8)
    _many_docs(tmp_path, 6)
    Image.new("RGB", (100, 100)).save(tmp_path / "p.png")
    report = image_report(tmp_path, "train", "vertical", multiple=4, longest_side=64)
    assert report["pages_per_document"] == {1: 6} and report["suggested_image_size"] == (64, 64)
    capped = image_report(tmp_path, "train", "vertical", multiple=4, longest_side=500)
    assert capped["suggested_image_size"] == (100, 100)               # never upscaled past the source
    lengths = length_report(tmp_path, "train", _transform(), samples=24)
    assert lengths["suggested_max_length"] % 64 == 0
    assert lengths["overall"]["max"] <= lengths["suggested_max_length"] or lengths["overall"]["p99"] <= lengths["suggested_max_length"]


def test_fits_is_cheap_and_always_agrees_with_the_real_transform():
    import random
    from tests.test_train_batching import FakeProcessor
    processor = FakeProcessor()
    calls = {"n": 0}
    real = processor.apply_chat_template

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    processor.apply_chat_template = counting

    def run(transform, sizes, prompt, label):
        return int(transform([Image.new("RGB", s) for s in sizes], prompt, label)["input_ids"].shape[-1])

    rng = random.Random(0)
    for kwargs in ({}, {"merge_pages": "grid"}, {"image_size": (16, 16), "merge_pages": "vertical"}):
        transform = InstructTransform(processor=processor, **kwargs)
        for _ in range(60):
            sizes = [(rng.randint(8, 60), rng.randint(8, 60)) for _ in range(rng.randint(1, 4))]
            prompt, label = "p" * rng.randint(1, 40), "l" * rng.randint(1, 60)
            true_length = run(transform, sizes, prompt, label)
            assert abs(transform.estimate_length(sizes, prompt, label) - true_length) <= transform.ESTIMATE_MARGIN
            for limit in (true_length - 1, true_length, true_length + 1, 10_000, 5):
                transform.max_length, transform.fixed_length = limit, False
                assert transform.fits(sizes, prompt, label) == (true_length <= limit), (kwargs, limit, true_length)
            transform.max_length = None

    cheap = InstructTransform(processor=processor, max_length=500)
    sizes = [(32, 32)]
    cheap.fits(sizes, "p", "l")                         # measures the layout once
    calls["n"] = 0
    for k in range(200):
        cheap.fits(sizes, "p" * (k % 40 + 1), "l" * (k % 25 + 1))
    assert calls["n"] == 0                              # later checks never touch the processor


def test_fits_leaves_the_transforms_limit_and_padding_untouched():
    transform = _transform(max_length=64)
    transform.fits([(8, 8)], "Hi", "x" * 100)
    assert transform.max_length == 64 and transform.fixed_length
    assert transform(Image.new("RGB", (8, 8)), "Hi", "A")["input_ids"].shape == (64,)
