"""Shared machinery for every model's transform and collator.

Everything that must behave the same across models lives here: the drop-or-fit length rules, fixed-length
padding, the order of checks in `__call__`, and the page preparation. A model supplies only what is specific
to it, through three hooks: `_encode` (tokens and image inputs), `make_collator` (batching), and optionally
`token_length` / `estimate_length` (fast length checks) and `_finalize_page` / `_check_pages` (page rules).

Contract every transform satisfies, which is what `fits` and the loader rely on:
  * `__call__(image, prompt, label)` returns 1-D `input_ids`, `labels` and `attention_mask` (plus model extras).
  * an example longer than `max_length` raises SequenceTooLong (never truncated).
  * with `fixed_length` every sequence field is right-padded to exactly `max_length`.
  * `fits(page_sizes, prompt, label)` answers exactly what `__call__` would do, without pixel processing.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch
from PIL import Image

from .images import MERGE_MODES, as_rgb_page, merge_page_images


class SequenceTooLong(ValueError):
    """The tokenised example does not fit in max_length."""


class NoFittingExample(ValueError):
    """None of a document's candidate examples fits the length limit."""


class _PadToken:
    """Sentinel fill value meaning 'the tokenizer's pad token id'."""

    def __repr__(self) -> str:
        return "PAD_TOKEN"


PAD_TOKEN = _PadToken()


class BaseTransform(ABC):
    # Fill value per sequence-aligned key, used by fixed-length padding. Subclasses extend it.
    SEQUENCE_FILLS: ClassVar[dict[str, Any]] = {"input_ids": PAD_TOKEN, "attention_mask": 0, "labels": -100,
                                                "inp_mask": False}
    # Estimates within this many tokens of max_length are settled by the exact measurement.
    # 0 means the estimate is exact, so no measurement is needed.
    ESTIMATE_MARGIN: ClassVar[int] = 0
    # Sequence fields that are trimmed from a leading batch dimension of 1 to 1-D.
    _SEQUENCE_KEYS: ClassVar[tuple[str, ...]] = ("input_ids", "attention_mask", "labels", "inp_mask")

    def __init__(self, tokenizer: Any, max_length: int | None = None, fixed_length: bool = True,
                 merge_pages: str | None = None):
        if merge_pages is not None and merge_pages not in MERGE_MODES:
            raise ValueError(f"merge_pages must be None or one of {MERGE_MODES}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fixed_length = fixed_length and max_length is not None
        self.merge_pages = merge_pages
        self.exact_checks = 0   # how often fits() needed the exact measurement

    # ------------------------------------------------------------------------------------------
    # The template: every model goes through these steps in this order.
    # ------------------------------------------------------------------------------------------
    def __call__(self, image: Any, prompt: str, label: Any) -> dict[str, Any]:
        target = self._validate(prompt, label)
        pages = self._prepare_pages(image)
        result = self._encode(pages, prompt, target)
        for key in self._SEQUENCE_KEYS:
            value = result.get(key)
            if isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[0] == 1:
                result[key] = value[0]
        length = int(result["input_ids"].shape[-1])
        if self.max_length is not None and length > self.max_length:
            raise SequenceTooLong(
                f"Sequence length {length} exceeds max_length={self.max_length}. "
                "Truncating could remove part of the image or target."
            )
        if self.fixed_length:
            self._pad_to_max_length(result, length)
        return result

    def _validate(self, prompt: str, label: Any) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")
        if label is None:
            raise ValueError("Label cannot be None.")
        target = self._serialize_label(label)
        if not target.strip():
            raise ValueError("Label cannot be empty.")
        return target

    def _prepare_pages(self, image: Any) -> list[Image.Image]:
        """Accepts one image, None, or a list (one per page). Returns the final list of RGB images."""
        raw = list(image) if isinstance(image, (list, tuple)) else [image]
        pages = [page for page in map(as_rgb_page, raw) if page is not None]
        merged = merge_page_images(pages, self.merge_pages)
        finalized = [self._finalize_page(page) for page in merged]
        self._check_pages(finalized)
        return finalized

    def _finalize_page(self, image: Image.Image) -> Image.Image:
        """Hook: per-page resizing after merging. Identity by default."""
        return image

    def _check_pages(self, pages: list[Image.Image]) -> None:
        """Hook: reject page layouts the model cannot take. Accepts anything by default."""

    @staticmethod
    def _serialize_label(label: Any) -> str:
        if isinstance(label, (dict, list)):
            return json.dumps(label, ensure_ascii=False, separators=(",", ":"))
        if isinstance(label, str):
            return label
        return json.dumps(label, ensure_ascii=False)

    def _pad_token_id(self) -> int | None:
        """Hook: the id used for padding. Models with a fallback id override this."""
        return self.tokenizer.pad_token_id

    def _fill_value(self, fill: Any) -> Any:
        if fill is PAD_TOKEN:
            pad_id = self._pad_token_id()
            if pad_id is None:
                raise ValueError("Fixed-length padding needs the tokenizer to have a pad_token_id.")
            return pad_id
        return fill

    def _pad_to_max_length(self, result: dict[str, Any], length: int) -> None:
        """Right-pads every declared sequence field to max_length. Other fields are left alone."""
        extra = self.max_length - length
        if extra == 0:
            return
        for key, fill in self.SEQUENCE_FILLS.items():
            tensor = result.get(key)
            if not isinstance(tensor, torch.Tensor) or tensor.shape[-1] != length:
                continue
            pad = torch.full((*tensor.shape[:-1], extra), self._fill_value(fill), dtype=tensor.dtype)
            result[key] = torch.cat([tensor, pad], dim=-1)

    # ------------------------------------------------------------------------------------------
    # Per-model hooks.
    # ------------------------------------------------------------------------------------------
    @abstractmethod
    def _encode(self, pages: list[Image.Image], prompt: str, target: str) -> dict[str, Any]:
        """Builds the model inputs for one example. Returns at least input_ids, labels, attention_mask."""

    @abstractmethod
    def make_collator(self):
        """The collator that batches this transform's outputs."""

    def token_length(self, page_sizes: list[tuple[int, int]], prompt: str, label: str) -> int:
        """Exact sequence length of an example. Default: run the real transform on blank pages (token counts
        depend on the sizes, not the pixels). Models with a closed-form count override this."""
        blanks = [Image.new("RGB", (w, h), "white") for w, h in page_sizes]
        saved = self.max_length, self.fixed_length
        self.max_length, self.fixed_length = None, False   # measure only: no limit, no padding
        try:
            return int(self(blanks, prompt, label)["input_ids"].shape[-1])
        finally:
            self.max_length, self.fixed_length = saved

    def estimate_length(self, page_sizes: list[tuple[int, int]], prompt: str, label: str) -> int:
        """Cheap length estimate. Default: the exact length. Override together with ESTIMATE_MARGIN."""
        return self.token_length(page_sizes, prompt, label)

    def fits(self, page_sizes: list[tuple[int, int]], prompt: str, label: str) -> bool:
        """Would this example fit max_length? Decided from the estimate unless it lands within
        ESTIMATE_MARGIN tokens of the limit, in which case the exact length settles it."""
        if self.max_length is None:
            return True
        estimate = self.estimate_length(page_sizes, prompt, label)
        if estimate > self.max_length + self.ESTIMATE_MARGIN:
            return False
        if estimate <= self.max_length - self.ESTIMATE_MARGIN:
            return True
        self.exact_checks += 1
        return self.token_length(page_sizes, prompt, label) <= self.max_length


class BaseCollator(ABC):
    """Pads a batch of transform outputs to one length and hands the rest to `collate_extras`."""

    SEQUENCE_FILLS: ClassVar[dict[str, Any]] = {"input_ids": PAD_TOKEN, "attention_mask": 0, "labels": -100}
    REQUIRED: ClassVar[tuple[str, ...]] = ("input_ids", "labels")

    def __init__(self, pad_token_id: int, padding_side: str = "right", pad_to_multiple_of: int | None = None):
        if padding_side not in ("left", "right"):
            raise ValueError("padding_side must be 'left' or 'right'.")
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty batch.")
        for i, feature in enumerate(features):
            for key in self.REQUIRED:
                if key not in feature:
                    raise ValueError(f"Example {i} is missing {key}.")
        length = self._target_length(features)
        batch = self.collate_sequences(features, length)
        return self.collate_extras(features, batch)

    def collate_sequences(self, features: list[dict[str, Any]], length: int) -> dict[str, Any]:
        """Pads every declared sequence field present in all features. Subclasses may extend this."""
        batch: dict[str, Any] = {}
        for key, fill in self.SEQUENCE_FILLS.items():
            if all(key in feature for feature in features):
                value = self.pad_token_id if fill is PAD_TOKEN else fill
                batch[key] = self._pad(features, key, value, length)
        return batch

    @abstractmethod
    def collate_extras(self, features: list[dict[str, Any]], batch: dict[str, Any]) -> dict[str, Any]:
        """Adds the non-sequence parts (images, visual tensors, ...) to the batch."""

    @staticmethod
    def _normalize(tensor: torch.Tensor, key: str) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{key} must be a torch.Tensor.")
        if tensor.ndim == 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 1:
            raise ValueError(f"{key} must have shape [L] or [1, L], got {tuple(tensor.shape)}.")
        return tensor

    def _target_length(self, features: list[dict[str, Any]]) -> int:
        length = max(self._normalize(f["input_ids"], "input_ids").shape[0] for f in features)
        if self.pad_to_multiple_of:
            length = -(-length // self.pad_to_multiple_of) * self.pad_to_multiple_of
        return length

    def _pad(self, features: list[dict[str, Any]], key: str, value: Any, length: int) -> torch.Tensor:
        sequences = [self._normalize(f[key], key) for f in features]
        output = torch.full((len(sequences), length), fill_value=value, dtype=sequences[0].dtype)
        for i, sequence in enumerate(sequences):
            n = sequence.shape[0]
            if n > length:
                raise ValueError(f"{key} length exceeds target padding length.")
            if self.padding_side == "right":
                output[i, :n] = sequence
            else:
                output[i, length - n:] = sequence
        return output


def to_device(value: Any, device: Any) -> Any:
    """Moves every tensor in a (nested) batch to a device. Lists, tuples and dicts keep their structure."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    return value
