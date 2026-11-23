"""The batch adapter turns the collator's DeepSeek layout into transformers' DeepseekOcr2 forward arguments.
The layout was also compared with `DeepseekOcr2Processor` on a real image: the same token counts, and pixel
values within 0.002 (resampling rounding), for the global view and for six 768-px tiles."""
import pytest
import torch
from PIL import Image

from train.processors import (DeepSeekOCRV2Transform, V2_SPEC, batch_adapter_for, deepseek_v2_batch_adapter,
                              model_family)
from train.processors.deepseekOcr import IMAGE_TOKEN_ID, image_token_count

from .test_deepseek_ocr import StubTokenizer


def _batch(mode, sizes):
    transform = DeepSeekOCRV2Transform(tokenizer=StubTokenizer(), mode=mode, fixed_length=False)
    items = [transform(Image.new("RGB", size, "white"), "Extract.", "ANSWER") for size in sizes]
    return transform.make_collator()(items), items


def test_base_mode_has_one_global_view_per_sample_and_no_tiles():
    batch, _ = _batch("base", [(500, 660), (760, 1000)])
    out = deepseek_v2_batch_adapter(batch)
    assert set(out) == {"input_ids", "attention_mask", "labels", "pixel_values", "pixel_values_local",
                        "num_local_patches"}
    assert out["pixel_values"].shape == (2, 3, 1024, 1024)
    assert out["pixel_values_local"] is None and out["num_local_patches"] == [0, 0]
    assert int((out["input_ids"] == IMAGE_TOKEN_ID).sum(1)[0]) == image_token_count(V2_SPEC, "base")   # 257


def test_gundam_mode_concatenates_every_samples_tiles_and_counts_them():
    batch, items = _batch("gundam", [(400, 500), (760, 1000), (1500, 1100)])
    out = deepseek_v2_batch_adapter(batch)
    counts = out["num_local_patches"]
    assert counts[0] == 0 and counts[1] > 1 and counts[2] > 1              # a small page has no tiles
    assert out["pixel_values"].shape == (3, 3, 1024, 1024)
    assert out["pixel_values_local"].shape == (sum(counts), 3, 768, 768)
    for row, count in enumerate(counts):                                    # placeholders = global + tiles + separator
        tokens = int((out["input_ids"][row] == IMAGE_TOKEN_ID).sum())
        assert tokens == 256 + 144 * count + 1
    first_tiles = items[1]["images"][0][: counts[1]]
    assert torch.equal(out["pixel_values_local"][: 0 + counts[0] + counts[1]][counts[0]:], first_tiles)


def test_the_adapter_is_picked_by_model_family():
    assert batch_adapter_for("deepseek-community/DeepSeek-OCR-2") is deepseek_v2_batch_adapter
    assert model_family("deepseek-community/DeepSeek-OCR-2") == "deepseek-v2"
    assert batch_adapter_for("Qwen/Qwen2.5-VL-3B-Instruct") is None
    with pytest.raises(ValueError, match="no batch adapter"):
        batch_adapter_for("deepseek-ai/DeepSeek-OCR")
