"""DeepSeek-OCR training transforms: DeepSeek-OCR v1 (deepseek-ai/DeepSeek-OCR) and v2 (DeepSeek-OCR-2).

Both versions share the tokenizer, the placeholder layout and the tile geometry, so that machinery lives in
DeepSeekOCRTransform. What differs between versions is described by a DeepSeekSpec (resolution modes, tile
limit, separator tokens); V1 and V2 are the two specs. The preprocessing follows `infer` in each model's
repository (dynamic_preprocess and find_closest_aspect_ratio are copied from it), so training sees the same
pixels and token layout as inference.

The model's forward pass takes one image per sample, so multi-page documents are merged first (merge_pages).
Per sample the model expects:
    input_ids            [bos] + <image> placeholders + prompt + answer + [eos]
    images_seq_mask      True at the placeholder positions, where the vision features are scattered
    images               (patches, global_view), one tuple per sample, never concatenated
    images_spatial_crop  [width_tiles, height_tiles]

This file also holds V1, because the two versions share the geometry; the spec is the only difference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from PIL import Image, ImageOps

from .base import BaseCollator, BaseTransform, PAD_TOKEN
from .images import merged_size

IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 128815
BOS_ID, EOS_ID = 0, 1
PATCH_SIZE, DOWNSAMPLE = 16, 4
MEAN = STD = 0.5


@dataclass(frozen=True)
class DeepSeekSpec:
    """What differs between DeepSeek-OCR versions. modes: name -> (base_size, image_size, crop_mode)."""
    name: str
    modes: dict
    max_tiles: int          # dynamic_preprocess max_num
    row_tokens: bool        # v1 adds a separator token after every feature row
    resize_below: int       # non-crop views up to this size are resized, not just padded, to the square
    markers: tuple[str, ...]  # substrings of a model id that identify this version


V1_SPEC = DeepSeekSpec(
    name="v1",
    modes={"tiny": (512, 512, False), "small": (640, 640, False), "base": (1024, 1024, False),
           "large": (1280, 1280, False), "gundam": (1024, 640, True)},
    max_tiles=9, row_tokens=True, resize_below=640, markers=("deepseek-ocr",),
)
V2_SPEC = DeepSeekSpec(
    name="v2",
    modes={"small": (768, 768, False), "base": (1024, 1024, False), "gundam": (1024, 768, True)},
    max_tiles=6, row_tokens=False, resize_below=768, markers=("ocr-2", "ocr2", "ocr-v2", "ocrv2"),
)


def _queries(size: int) -> int:
    return math.ceil((size // PATCH_SIZE) / DOWNSAMPLE)


def image_token_count(spec: DeepSeekSpec, mode: str, tiles: tuple[int, int] = (1, 1)) -> int:
    """Number of <image> placeholder tokens the model expects for one image."""
    base, size, crop = spec.modes[mode]
    extra = 1 if spec.row_tokens else 0
    if not crop:
        q = _queries(size)
        return (q + extra) * q + 1
    q_base, q = _queries(base), _queries(size)
    total = (q_base + extra) * q_base + 1
    if tiles[0] > 1 or tiles[1] > 1:
        total += (q * tiles[0] + extra) * (q * tiles[1])
    return total


def _closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff, best = float("inf"), (1, 1)
    area = width * height
    for ratio in target_ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best = ratio
    return best


def tile_grid(width: int, height: int, min_num: int = 2, max_num: int = 9, image_size: int = 640):
    """The (columns, rows) tile grid dynamic_preprocess would pick for an image of this size."""
    ratios = sorted({(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
                     for j in range(1, n + 1) if min_num <= i * j <= max_num}, key=lambda r: r[0] * r[1])
    return _closest_aspect_ratio(width / height, ratios, width, height, image_size)


def dynamic_preprocess(image: Image.Image, min_num: int = 2, max_num: int = 9, image_size: int = 640):
    """Splits the image into image_size tiles on the grid whose aspect ratio is closest to the image's."""
    ratio = tile_grid(image.width, image.height, min_num, max_num, image_size)
    resized = image.resize((image_size * ratio[0], image_size * ratio[1]))
    tiles = []
    for i in range(ratio[0] * ratio[1]):
        col, row = i % ratio[0], i // ratio[0]
        tiles.append(resized.crop((col * image_size, row * image_size, (col + 1) * image_size,
                                   (row + 1) * image_size)))
    return tiles, ratio


def _to_tensor(image: Image.Image, dtype: torch.dtype) -> torch.Tensor:
    data = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8)
    data = data.view(image.height, image.width, 3).permute(2, 0, 1).float() / 255.0
    return ((data - MEAN) / STD).to(dtype)


class DeepSeekOCRTransform(BaseTransform):
    """Shared DeepSeek-OCR transform. Subclasses set SPEC (V1 or V2)."""

    SPEC: ClassVar[DeepSeekSpec | None] = None
    SEQUENCE_FILLS: ClassVar[dict[str, Any]] = {"input_ids": PAD_TOKEN, "attention_mask": 0, "labels": -100,
                                                "images_seq_mask": False}
    ESTIMATE_MARGIN: ClassVar[int] = 0   # the closed-form length is exact

    def __init__(self, model_id: str | None = None, tokenizer: Any = None, mode: str = "base",
                 max_length: int | None = None, fixed_length: bool = True, merge_pages: str | None = "grid",
                 dtype: torch.dtype = torch.bfloat16):
        """mode: a resolution mode of this version (see the module docstring). max_length: with fixed_length
        every sample is right-padded to exactly this many tokens. merge_pages: how a multi-page document becomes
        the single image the model needs ("grid" keeps the canvas near square, which suits the square global
        view)."""
        if self.SPEC is None:
            raise TypeError("Use DeepSeekOCRV1Transform or DeepSeekOCRV2Transform (or create_transform).")
        if tokenizer is None:
            if model_id is None:
                raise ValueError("Provide model_id or tokenizer.")
            # Plain tokenizer class: AutoTokenizer would offer to run the repository's custom model code.
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
        super().__init__(tokenizer, max_length=max_length, fixed_length=fixed_length, merge_pages=merge_pages)
        if mode not in self.SPEC.modes:
            raise ValueError(f"mode must be one of {sorted(self.SPEC.modes)} for DeepSeek-OCR {self.SPEC.name}")
        self.version = self.SPEC.name
        self.mode = mode
        self.base_size, self.image_size, self.crop_mode = self.SPEC.modes[mode]
        self.dtype = dtype

    # -- validation and pages ---------------------------------------------------------------------
    def _validate(self, prompt: str, label: Any) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Label must be a non-empty string.")
        return label

    def _check_pages(self, pages: list[Image.Image]) -> None:
        if not pages:
            raise ValueError("DeepSeek-OCR training needs an image for every sample.")
        if len(pages) != 1:
            raise ValueError(f"DeepSeek-OCR takes one image per sample; got {len(pages)} pages. "
                             "Set merge_pages to 'grid', 'vertical' or 'horizontal'.")

    def _pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 2

    # -- one image ---------------------------------------------------------------------------------
    def _encode_image(self, image: Image.Image):
        """Returns (patches, global_view, [tiles_w, tiles_h], placeholder_count), as `infer` builds them."""
        fill = (int(MEAN * 255),) * 3
        if self.crop_mode:
            tiles, ratio = None, (1, 1)
            if image.width > self.image_size or image.height > self.image_size:
                tiles, ratio = dynamic_preprocess(image, max_num=self.SPEC.max_tiles, image_size=self.image_size)
            global_view = ImageOps.pad(image, (self.base_size, self.base_size), color=fill)
            if ratio[0] > 1 or ratio[1] > 1:
                patches = torch.stack([_to_tensor(t, self.dtype) for t in tiles])
            else:
                patches = torch.zeros((1, 3, self.base_size, self.base_size), dtype=self.dtype)
        else:
            if self.image_size <= self.SPEC.resize_below:
                image = image.resize((self.image_size, self.image_size))
            global_view = ImageOps.pad(image, (self.image_size, self.image_size), color=fill)
            ratio = (1, 1)
            patches = torch.zeros((1, 3, self.base_size, self.base_size), dtype=self.dtype)
        return patches, _to_tensor(global_view, self.dtype)[None], list(ratio), \
            image_token_count(self.SPEC, self.mode, ratio)

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    # -- the example -------------------------------------------------------------------------------
    def _encode(self, pages: list[Image.Image], prompt: str, target: str) -> dict[str, Any]:
        patches, global_view, crop, n_image_tokens = self._encode_image(pages[0])
        prompt_ids = self._encode_text("\n" + prompt.strip())   # the <image> token is followed by a newline
        answer_ids = self._encode_text(target.strip()) + [EOS_ID]

        input_ids = [BOS_ID] + [IMAGE_TOKEN_ID] * n_image_tokens + prompt_ids + answer_ids
        seq_mask = [False] + [True] * n_image_tokens + [False] * (len(prompt_ids) + len(answer_ids))
        labels = [-100] * (1 + n_image_tokens + len(prompt_ids)) + answer_ids   # supervise the answer and eos only
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "images_seq_mask": torch.tensor(seq_mask, dtype=torch.bool),
            "images": (patches, global_view),
            "images_spatial_crop": torch.tensor([crop], dtype=torch.long),
        }

    # -- lengths without pixels ----------------------------------------------------------------------
    def token_length(self, page_sizes: list[tuple[int, int]], prompt: str, label: str) -> int:
        """Exact length from page sizes and text alone: the closed-form image count plus the text tokens."""
        width, height = merged_size(list(page_sizes), self.merge_pages)
        tiles = (1, 1)
        if self.crop_mode and (width > self.image_size or height > self.image_size):
            tiles = tile_grid(width, height, max_num=self.SPEC.max_tiles, image_size=self.image_size)
        return (1 + image_token_count(self.SPEC, self.mode, tiles)
                + len(self._encode_text("\n" + prompt.strip())) + len(self._encode_text(label.strip())) + 1)

    def make_collator(self) -> "DeepSeekOCRCollator":
        return DeepSeekOCRCollator(self._pad_token_id())


class DeepSeekOCRV1Transform(DeepSeekOCRTransform):
    """deepseek-ai/DeepSeek-OCR."""
    SPEC = V1_SPEC


class DeepSeekOCRV2Transform(DeepSeekOCRTransform):
    """deepseek-ai/DeepSeek-OCR-2."""
    SPEC = V2_SPEC


class DeepSeekOCRCollator(BaseCollator):
    """Pads the token sequences and keeps `images` as a list of per-sample (patches, global_view) tuples."""

    SEQUENCE_FILLS: ClassVar[dict[str, Any]] = {"input_ids": PAD_TOKEN, "attention_mask": 0, "labels": -100,
                                                "images_seq_mask": False}

    def __init__(self, pad_token_id: int = 2, pad_to_multiple_of: int | None = None):
        super().__init__(pad_token_id, padding_side="right", pad_to_multiple_of=pad_to_multiple_of)

    def collate_extras(self, features: list[dict[str, Any]], batch: dict[str, Any]) -> dict[str, Any]:
        batch["images"] = [f["images"] for f in features]
        batch["images_spatial_crop"] = torch.cat([f["images_spatial_crop"] for f in features], dim=0)
        return batch
