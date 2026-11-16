"""The shared base: the template, the length rules, and the contract every model transform must satisfy."""
import types

import pytest
import torch
from PIL import Image

from train.processors import (BaseCollator, BaseTransform, DeepSeekOCRV1Transform, DeepSeekOCRV2Transform,
                              InstructTransform, SequenceTooLong, V1_SPEC, create_transform, model_family, to_device)
from train.processors.base import PAD_TOKEN


class StubTokenizer:
    pad_token_id = 2
    padding_side = "right"
    chat_template = "x"

    def encode(self, text, add_special_tokens=False):
        return [10 + ord(c) % 50 for c in text]


class ToyTransform(BaseTransform):
    """One token per character of the label, plus a fixed image token. Declares a custom padded key."""
    SEQUENCE_FILLS = {**BaseTransform.SEQUENCE_FILLS, "extra_ids": 7}

    def _encode(self, pages, prompt, target):
        ids = [5] * len(pages) + [10 + ord(c) % 50 for c in prompt + target]
        n = len(ids)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor([-100] * (n - len(target)) + ids[-len(target):]),
                "attention_mask": torch.ones(n, dtype=torch.long), "inp_mask": torch.ones(n, dtype=torch.bool),
                "extra_ids": torch.arange(n), "untouched": torch.zeros(3), "pages": len(pages)}

    def make_collator(self):
        return ToyCollator(pad_token_id=2)


class ToyCollator(BaseCollator):
    def collate_extras(self, features, batch):
        batch["pages"] = [f["pages"] for f in features]
        return batch


def test_template_validates_prompt_and_label_before_any_work():
    t = ToyTransform(StubTokenizer())
    with pytest.raises(ValueError, match="Prompt"):
        t(Image.new("RGB", (4, 4)), "  ", "x")
    with pytest.raises(ValueError, match="None"):
        t(Image.new("RGB", (4, 4)), "p", None)
    with pytest.raises(ValueError, match="empty"):
        t(Image.new("RGB", (4, 4)), "p", "   ")


def test_template_runs_hooks_and_squeezes_to_one_dimension():
    t = ToyTransform(StubTokenizer(), merge_pages="vertical")
    out = t([Image.new("RGB", (4, 4)), Image.new("RGB", (4, 6))], "p", "ab")
    assert out["input_ids"].ndim == 1 and out["labels"].ndim == 1 and out["attention_mask"].ndim == 1
    assert out["pages"] == 1                                       # merge happened before _encode


def test_fixed_length_pads_declared_keys_only_and_keeps_others():
    t = ToyTransform(StubTokenizer(), max_length=20)
    out = t(Image.new("RGB", (4, 4)), "p", "ab")
    n = int(out["attention_mask"].sum())
    assert out["input_ids"].shape == out["labels"].shape == out["extra_ids"].shape == (20,)
    assert (out["input_ids"][n:] == 2).all()                       # tokenizer pad id
    assert (out["labels"][n:] == -100).all() and (out["extra_ids"][n:] == 7).all()
    assert out["untouched"].shape == (3,)                          # not a sequence field: never padded


def test_overflow_raises_and_is_never_truncated():
    with pytest.raises(SequenceTooLong):
        ToyTransform(StubTokenizer(), max_length=5)(Image.new("RGB", (4, 4)), "prompt", "a long answer")


def test_fits_matches_the_real_call_at_the_limit():
    t = ToyTransform(StubTokenizer())
    sizes, prompt, label = [(4, 4)], "prompt", "answer"
    true = int(t(Image.new("RGB", (4, 4)), prompt, label)["input_ids"].shape[-1])
    for limit, expected in ((true - 1, False), (true, True), (true + 1, True)):
        t.max_length, t.fixed_length = limit, False
        assert t.fits(sizes, prompt, label) is expected


def test_missing_pad_token_is_reported_when_padding_needs_it():
    class NoPad(StubTokenizer):
        pad_token_id = None
    t = ToyTransform(NoPad(), max_length=30)
    with pytest.raises(ValueError, match="pad_token_id"):
        t(Image.new("RGB", (4, 4)), "p", "a")


def test_abstract_hooks_must_be_implemented():
    class Incomplete(BaseTransform):
        pass
    with pytest.raises(TypeError):
        Incomplete(StubTokenizer())


def test_collator_pads_to_longest_and_to_multiple():
    features = [{"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 2, 3]), "pages": 1},
                {"input_ids": torch.tensor([4]), "labels": torch.tensor([9]), "pages": 2}]
    batch = ToyCollator(pad_token_id=0, pad_to_multiple_of=4)(features)
    assert batch["input_ids"].tolist() == [[1, 2, 3, 0], [4, 0, 0, 0]]
    assert batch["labels"][1].tolist() == [9, -100, -100, -100]
    assert batch["pages"] == [1, 2]
    left = ToyCollator(pad_token_id=0, padding_side="left")(features)
    assert left["input_ids"][1].tolist() == [0, 0, 4]          # padded to the longest example (3)


def test_to_device_keeps_structure_and_non_tensors():
    batch = {"a": torch.ones(2), "b": [(torch.ones(1), torch.ones(1))], "c": "text", "d": 3}
    moved = to_device(batch, "cpu")
    assert isinstance(moved["b"][0], tuple) and moved["c"] == "text" and moved["d"] == 3


# ---------------------------------------------------------------------------------------------------
# The contract every model transform must pass. A new model joins by adding its factory to CONTRACTS.
# ---------------------------------------------------------------------------------------------------
def _page(w=60, h=80):
    return Image.new("RGB", (w, h), "white")


def _instruct(**kwargs):
    from tests.test_train_batching import FakeProcessor
    return InstructTransform(processor=FakeProcessor(), **kwargs)


CONTRACTS = {
    "instruct": lambda **kw: _instruct(**kw),
    "deepseek-v1": lambda **kw: DeepSeekOCRV1Transform(tokenizer=StubTokenizer(), mode="small", **kw),
    "deepseek-v2": lambda **kw: DeepSeekOCRV2Transform(tokenizer=StubTokenizer(), mode="small", **kw),
}


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_contract_outputs_are_one_dimensional_with_required_keys(name):
    out = CONTRACTS[name](merge_pages="grid")([_page(), _page()], "Extract.", "A short answer")
    for key in ("input_ids", "labels", "attention_mask"):
        assert out[key].ndim == 1, (name, key)
    assert out["input_ids"].shape == out["labels"].shape == out["attention_mask"].shape


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_contract_fixed_length_gives_one_shape_and_unsupervised_padding(name):
    t = CONTRACTS[name](max_length=2000)
    for pages, prompt, label in (([_page()], "Hi", "A"), ([_page(), _page()], "Longer prompt here", "BBBB")):
        out = t(pages, prompt, label)
        assert out["input_ids"].shape == (2000,)
        real = int(out["attention_mask"].sum())
        assert (out["labels"][real:] == -100).all()
        assert (out["attention_mask"][real:] == 0).all()


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_contract_fits_agrees_with_the_real_call(name):
    t = CONTRACTS[name]()
    pages, prompt, label = [_page(), _page()], "Extract the data.", "answer text " * 3
    true = int(t(pages, prompt, label)["input_ids"].shape[-1])
    assert t.token_length([(60, 80), (60, 80)], prompt, label) == true
    for limit, expected in ((true - 1, False), (true, True), (true + 1, True)):
        t.max_length, t.fixed_length = limit, False
        assert t.fits([(60, 80), (60, 80)], prompt, label) is expected, (name, limit)


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_contract_collator_batches_fixed_length_examples(name):
    t = CONTRACTS[name](max_length=400)
    batch = t.make_collator()([t(_page(), "Hi", "A"), t(_page(), "Hi", "B")])
    assert batch["input_ids"].shape == batch["labels"].shape == (2, 400)
    moved = to_device(batch, "cpu")
    assert moved["input_ids"].device.type == "cpu"


def test_create_transform_picks_the_right_class_for_each_model_family():
    assert model_family("deepseek-ai/DeepSeek-OCR-2") == "deepseek-v2"
    assert model_family("deepseek-ai/DeepSeek-OCR") == "deepseek-v1"
    assert model_family("deepseek_ocr_2") == "deepseek-v2"
    assert model_family("Qwen/Qwen2.5-VL-3B-Instruct") == "instruct"
    tok = StubTokenizer()
    assert isinstance(create_transform("deepseek-ai/DeepSeek-OCR-2", tokenizer=tok), DeepSeekOCRV2Transform)
    assert isinstance(create_transform("deepseek-ai/DeepSeek-OCR", tokenizer=tok), DeepSeekOCRV1Transform)
    assert isinstance(create_transform("Qwen/Qwen2.5-VL", processor=types.SimpleNamespace(tokenizer=tok,
                      apply_chat_template=None)), InstructTransform)
    with pytest.raises(TypeError):
        from train.processors import DeepSeekOCRTransform
        DeepSeekOCRTransform(tokenizer=tok)


def test_pad_sentinel_is_resolved_by_the_tokenizer_not_stored():
    assert PAD_TOKEN is not None and V1_SPEC.name == "v1"
