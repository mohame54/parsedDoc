"""Generic transform for instruction-tuned vision-language models with a Hugging Face chat template
(Qwen2.5-VL style). The chat template and the model's processor decide the tokens; this module only builds
the conversation, supervises the answer, and batches the results.
"""
from __future__ import annotations

from typing import Any, ClassVar

import torch
from PIL import Image

from .base import BaseCollator, BaseTransform
from .images import letterbox, merged_size


class InstructTransform(BaseTransform):
    SEQUENCE_FILLS: ClassVar[dict[str, Any]] = {**BaseTransform.SEQUENCE_FILLS, "token_type_ids": 0,
                                                "position_ids": 0}
    ESTIMATE_MARGIN: ClassVar[int] = 8   # the overhead estimate is within a few tokens of the exact count

    def __init__(
        self,
        model_id: str | None = None,
        max_length: int | None = None,
        add_generation_prompt: bool = True,
        processor: Any = None,
        fixed_length: bool = True,
        image_size: tuple[int, int] | None = None,
        merge_pages: str | None = None,
    ):
        """model_id loads the Hugging Face processor; an already-built `processor` works too (tests, or sharing
        one processor with the model). image_size (width, height) letterboxes every page to one size, so the
        vision tower also sees one input shape. Without it the processor's own resizing decides."""
        if processor is None:
            if model_id is None:
                raise ValueError("Provide model_id or processor.")
            from transformers import AutoProcessor   # imported here so the data code works without it
            processor = AutoProcessor.from_pretrained(model_id)
        self.processor = processor
        self.model_id = model_id or type(processor).__name__
        super().__init__(processor.tokenizer, max_length=max_length, fixed_length=fixed_length,
                         merge_pages=merge_pages)
        self.image_size = tuple(image_size) if image_size else None
        self.add_generation_prompt = add_generation_prompt
        if not self.tokenizer.chat_template:
            raise ValueError(f"{self.model_id} does not provide a chat template.")

    # -- pages -----------------------------------------------------------------------------------
    def _finalize_page(self, image: Image.Image) -> Image.Image:
        return letterbox(image, self.image_size) if self.image_size is not None else image

    # -- the conversation ------------------------------------------------------------------------
    @staticmethod
    def _build_user_message(prompt: str, images: list[Image.Image]) -> dict:
        """One image block per page, then the prompt."""
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt.strip()})
        return {"role": "user", "content": content}

    def _apply_template(self, messages: list[dict], add_generation_prompt: bool):
        """The model-specific chat template and multimodal processor. The processor expands image tokens."""
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )

    def _get_input_ids(self, processed: dict) -> torch.Tensor:
        input_ids = processed["input_ids"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Expected a single-example batch with shape [1, seq_len].")
        return input_ids

    def _encode(self, pages: list[Image.Image], prompt: str, target: str) -> dict[str, Any]:
        user_message = self._build_user_message(prompt=prompt, images=pages)
        messages = [user_message, {"role": "assistant", "content": target}]

        # The prefix (images, prompt, generation prefix) must produce the same tokens as the start of the full
        # conversation. Nothing in the prefix contributes to the loss.
        prefix_ids = self._get_input_ids(self._apply_template([user_message], self.add_generation_prompt))
        model_inputs = self._apply_template(messages, add_generation_prompt=False)
        input_ids = self._get_input_ids(model_inputs)

        prefix_length = prefix_ids.shape[1]
        sequence_length = input_ids.shape[1]
        if prefix_length > sequence_length:
            raise RuntimeError("The input prefix is longer than the complete sequence.")
        if not torch.equal(prefix_ids, input_ids[:, :prefix_length]):
            raise RuntimeError(
                "Prefix token alignment failed. The processor generated "
                "different input tokens for the prefix and full conversation. "
                "A model-specific masking implementation is required."
            )

        inp_mask = torch.zeros(sequence_length, dtype=torch.bool)
        inp_mask[prefix_length:] = True
        labels = input_ids.clone()
        labels[:, ~inp_mask] = -100   # image and prompt tokens are never supervised

        result = dict(model_inputs)
        result.update({"labels": labels, "inp_mask": inp_mask, "messages": messages, "images": pages})
        return result

    # -- lengths ---------------------------------------------------------------------------------
    def _image_layout(self, page_sizes: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
        """The image sizes the processor will actually see: merged and/or letterboxed, from sizes alone."""
        sizes = [tuple(size) for size in page_sizes]
        if self.merge_pages and len(sizes) > 1:
            sizes = [merged_size(sizes, self.merge_pages)]
        if self.image_size is not None:
            sizes = [self.image_size] * len(sizes)
        return tuple(sizes)

    def _overhead(self, page_sizes: list[tuple[int, int]]) -> int:
        """Tokens an example costs besides its prompt and label text (image tokens, template, specials).
        Measured once per distinct image layout and cached; documents share a few layouts."""
        cache = self.__dict__.setdefault("_overhead_cache", {})
        key = self._image_layout(page_sizes)
        if key not in cache:
            total = self.token_length(page_sizes, "a", "b")
            cache[key] = total - self._text_tokens("a") - self._text_tokens("b")
        return cache[key]

    def _text_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text.strip(), add_special_tokens=False))

    def estimate_length(self, page_sizes: list[tuple[int, int]], prompt: str, label: str) -> int:
        return self._overhead(page_sizes) + self._text_tokens(prompt) + \
            self._text_tokens(self._serialize_label(label))

    def make_collator(self) -> "InstructCollator":
        return InstructCollator(self.tokenizer)


class InstructCollator(BaseCollator):
    """Pads text fields to the batch length and concatenates the visual tensors of the batch."""

    VISUAL_KEYS = {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"}
    METADATA_KEYS = {"messages", "images"}
    SEQUENCE_KEYS = {"input_ids", "attention_mask", "labels", "inp_mask", "token_type_ids", "position_ids"}

    def __init__(self, tokenizer, padding_side: str | None = None, pad_to_multiple_of: int | None = None,
                 include_inp_mask: bool = False):
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id. Configure one before creating the collator.")
        super().__init__(tokenizer.pad_token_id, padding_side or tokenizer.padding_side, pad_to_multiple_of)
        self.tokenizer = tokenizer
        self.include_inp_mask = include_inp_mask

    def collate_sequences(self, features: list[dict[str, Any]], length: int) -> dict[str, Any]:
        pad = self.pad_token_id
        batch: dict[str, Any] = {"input_ids": self._pad(features, "input_ids", pad, length)}
        if all("attention_mask" in f for f in features):
            batch["attention_mask"] = self._pad(features, "attention_mask", 0, length)
        else:   # no mask supplied: derive it from the padded ids
            batch["attention_mask"] = (self._pad(features, "input_ids", 0, length) != pad).long()
        batch["labels"] = self._pad(features, "labels", -100, length)
        if self.include_inp_mask and all("inp_mask" in f for f in features):
            batch["inp_mask"] = self._pad(features, "inp_mask", False, length)

        # Other sequence-aligned tensors (token_type_ids, position_ids, ...) are padded with 0.
        excluded = self.SEQUENCE_KEYS | self.VISUAL_KEYS | self.METADATA_KEYS
        common = set.intersection(*(set(f) for f in features))
        for key in common - excluded:
            values = [f[key] for f in features]
            if not all(isinstance(v, torch.Tensor) for v in values):
                continue
            if all(v.ndim in (1, 2) and v.shape[-1] == self._normalize(f["input_ids"], "input_ids").shape[0]
                   for v, f in zip(values, features)):
                batch[key] = self._pad(features, key, 0, length)
        return batch

    def collate_extras(self, features: list[dict[str, Any]], batch: dict[str, Any]) -> dict[str, Any]:
        visual_keys = set(features[0]) & self.VISUAL_KEYS
        for i, feature in enumerate(features):
            if (set(feature) & self.VISUAL_KEYS) != visual_keys:
                raise ValueError(
                    f"Example {i} has different visual fields. Expected {visual_keys}, "
                    f"got {set(feature) & self.VISUAL_KEYS}. "
                    "Mixed visual/no-visual batches require model-specific handling."
                )
        for key in visual_keys:
            values = [feature[key] for feature in features]
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise TypeError(f"Visual field {key} must contain tensors.")
            try:
                batch[key] = torch.cat(values, dim=0)   # processor outputs are concatenated, not padded
            except RuntimeError as exc:
                raise ValueError(
                    f"Cannot concatenate visual field '{key}'. The processor outputs have incompatible shapes."
                ) from exc
        return batch
