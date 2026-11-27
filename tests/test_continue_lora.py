"""Continuing from an adapter with a new LoRA configuration: the old adapter (and the ones it sits on) is merged into
the base, then a fresh adapter is attached. Checked on a toy model by comparing outputs at every stage."""
import copy

import pytest
import torch
from torch import nn

peft = pytest.importorskip("peft")

from train.model import DeepSeekOCR  # noqa: E402


class _LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(6, 6)

    def forward(self, x):
        return torch.tanh(self.proj(x))


class _Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LM()
        self.multi_modal_projector = nn.Linear(6, 6)
        self.vision_tower = nn.Linear(6, 6)
        self.view_separator = nn.Parameter(torch.zeros(6))


class Inner(nn.Module):
    """Has the attribute paths DeepSeekOCR expects from a HF DeepseekOcr model."""

    def __init__(self):
        super().__init__()
        self.model = _Core()
        self.lm_head = nn.Linear(6, 3)

    def forward(self, x):
        return self.lm_head(self.model.language_model(x))


@pytest.fixture()
def base():
    torch.manual_seed(0)
    return Inner()


X = torch.randn(4, 6)


def _wrapper(base):
    return DeepSeekOCR(copy.deepcopy(base), lang_trained=True, vision_trained=False)


def _randomise_adapter(wrapper):
    with torch.no_grad():
        for name, param in wrapper.named_parameters():
            if "lora_" in name:
                param.copy_(torch.randn_like(param) * 0.5)


def _stage(base, tmp_path, name, previous=None, r=2, alpha=4):
    wrapper = _wrapper(base)
    if previous is None:
        wrapper.add_lora(r=r, alpha=alpha, dropout=0.0, target_modules=["proj"])
    else:
        wrapper.continue_from(previous, r=r, alpha=alpha, dropout=0.0, target_modules=["proj"])
    _randomise_adapter(wrapper)
    path = tmp_path / f"{name}.pt"
    wrapper.save_checkpoint(path, step=7, model_state=wrapper.trainable_state_dict())
    with torch.no_grad():
        return wrapper, path, wrapper(X)


def test_the_new_stage_starts_exactly_where_the_old_one_ended(base, tmp_path):
    _, first, out_first = _stage(base, tmp_path, "sft", r=2, alpha=4)
    wrapper = _wrapper(base)
    meta = wrapper.continue_from(first, r=8, alpha=16, dropout=0.0, target_modules=["proj"])
    with torch.no_grad():
        assert torch.allclose(wrapper(X), out_first, atol=1e-5)          # merged old adapter, new B is zero
    assert meta["step"] == 7 and wrapper.lora_enabled
    assert wrapper.lora_config_dict()["r"] == 8                             # the new shape, not the old one
    assert [e["r"] for e in wrapper.lineage] == [2]
    assert {n for n, p in wrapper.named_parameters() if p.requires_grad} and all(
        "lora_" in n for n, p in wrapper.named_parameters() if p.requires_grad)


def test_a_third_stage_replays_the_whole_chain(base, tmp_path):
    _, first, _ = _stage(base, tmp_path, "sft", r=2, alpha=4)
    _, second, out_second = _stage(base, tmp_path, "rl", previous=first, r=4, alpha=8)
    third = _wrapper(base)
    third.continue_from(second, r=2, alpha=4, dropout=0.0, target_modules=["proj"])
    with torch.no_grad():
        assert torch.allclose(third(X), out_second, atol=1e-5)               # base + A + B, rebuilt from two files
    assert [e["r"] for e in third.lineage] == [2, 4]


def test_a_missing_earlier_checkpoint_is_a_clear_error_not_a_silently_wrong_model(base, tmp_path):
    _, first, _ = _stage(base, tmp_path, "sft")
    _, second, _ = _stage(base, tmp_path, "rl", previous=first, r=4, alpha=8)
    first.unlink()
    with pytest.raises(FileNotFoundError, match="lineage"):
        _wrapper(base).continue_from(second, r=2, alpha=4, target_modules=["proj"])


def test_a_checkpoint_without_a_stored_lora_shape_needs_the_old_r_and_alpha(base, tmp_path):
    _, first, out_first = _stage(base, tmp_path, "sft", r=2, alpha=4)
    old = torch.load(first, weights_only=False)
    old.pop("lora_config")
    old.pop("lineage")
    torch.save(old, first)
    with pytest.raises(ValueError, match="init-lora-r"):
        _wrapper(base).continue_from(first, r=4, alpha=8, target_modules=["proj"])
    wrapper = _wrapper(base)
    wrapper.continue_from(first, r=4, alpha=8, dropout=0.0, target_modules=["proj"], old_r=2, old_alpha=4,
                          old_target_modules=["proj"])
    with torch.no_grad():
        assert torch.allclose(wrapper(X), out_first, atol=1e-5)


def test_a_quantized_base_cannot_merge_and_adapters_cannot_be_attached_twice(base, tmp_path):
    _, first, _ = _stage(base, tmp_path, "sft")
    quantized = _wrapper(base)
    quantized.quantization = "4bit"
    with pytest.raises(RuntimeError, match="quantized"):
        quantized.continue_from(first)
    attached = _wrapper(base)
    attached.add_lora(r=2, alpha=4, target_modules=["proj"])
    with pytest.raises(RuntimeError, match="before add_lora"):
        attached.continue_from(first)
