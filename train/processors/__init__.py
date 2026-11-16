"""Model-specific input transforms and collators, with one shared base.

    from train.processors import create_transform, DatasetProcessor
    transform = create_transform("deepseek-ai/DeepSeek-OCR-2", tokenizer=tok, mode="base", max_length=1280)

Layout:
    base.py          BaseTransform / BaseCollator: the shared template, length rules, padding, to_device
    images.py        page merging and size arithmetic
    mixture.py       DatasetProcessor: task mixing (model-agnostic)
    instruct.py      InstructTransform / InstructCollator: generic chat-template vision-language models
    deepseekOcrV2.py DeepSeek-OCR v1 and v2 (one shared base, one spec per version)

To add a model: subclass BaseTransform, implement `_encode` and `make_collator`, and pass the shared contract
tests in tests/test_processor_base.py.
"""
from __future__ import annotations

from typing import Any

from .base import BaseCollator, BaseTransform, NoFittingExample, PAD_TOKEN, SequenceTooLong, to_device
from .deepseekOcr import (DeepSeekOCRCollator, DeepSeekOCRTransform, DeepSeekOCRV1Transform,
                            DeepSeekOCRV2Transform, DeepSeekSpec, V1_SPEC, V2_SPEC)
from .images import MERGE_MODES, as_rgb_page, letterbox, merge_page_images, merged_size
from .instruct import InstructCollator, InstructTransform
from .mixture import PRESETS, DatasetProcessor


def model_family(model_id: str | None) -> str:
    """'deepseek-v2', 'deepseek-v1' or 'instruct', from the model id."""
    name = (model_id or "").lower().replace("_", "-")
    if any(marker in name for marker in V2_SPEC.markers):
        return "deepseek-v2"
    if any(marker in name for marker in V1_SPEC.markers):
        return "deepseek-v1"
    return "instruct"


def create_transform(model_id: str | None, **kwargs: Any) -> BaseTransform:
    """The transform for a model id. kwargs go to the chosen class (tokenizer / processor, mode, max_length, ...)."""
    family = model_family(model_id)
    if family == "deepseek-v2":
        return DeepSeekOCRV2Transform(model_id=model_id, **kwargs)
    if family == "deepseek-v1":
        return DeepSeekOCRV1Transform(model_id=model_id, **kwargs)
    return InstructTransform(model_id=model_id, **kwargs)


__all__ = [
    "BaseTransform", "BaseCollator", "NoFittingExample", "SequenceTooLong", "PAD_TOKEN", "to_device",
    "InstructTransform", "InstructCollator",
    "DeepSeekOCRTransform", "DeepSeekOCRV1Transform", "DeepSeekOCRV2Transform", "DeepSeekOCRCollator",
    "DeepSeekSpec", "V1_SPEC", "V2_SPEC",
    "DatasetProcessor", "PRESETS",
    "MERGE_MODES", "as_rgb_page", "letterbox", "merge_page_images", "merged_size",
    "create_transform", "model_family",
]
