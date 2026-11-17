"""Inspect the export to choose a fixed image size and max_length before training.

    uv run --extra train python -m train.stats data/export/unverified --merge vertical
    uv run --extra train python -m train.stats data/export/unverified --model-id Qwen/Qwen2.5-VL-3B-Instruct \\
        --merge vertical --image-size 896x1176

Without --model-id it reports the images only (no tokenizer needed). With a model it also tokenises
a few hundred real training examples per task, reports the length percentiles and how many would not
fit a given max_length, and suggests one (the 99th percentile rounded up to a multiple of 64).
"""
from __future__ import annotations

import sys

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from PIL import Image
from corpus.progress import Progress

from .processors import create_transform, merged_size, model_family


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def _round_up(value: float, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def image_report(root: Path, split: str, merge: str | None, multiple: int = 28, longest_side: int = 1344) -> dict:
    with (root / f"{split}.jsonl").open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    pages, final = Counter(), []
    for record in Progress(records, desc="measuring images", unit="doc", leave=False):
        sizes = []
        for rel in record["images"]:
            with Image.open(root / rel) as image:
                sizes.append(image.size)
        pages[len(sizes)] += 1
        if sizes:
            final.append(merged_size(sizes, merge))
    widths, heights = [w for w, _ in final], [h for _, h in final]
    aspect = statistics.median(w / h for w, h in final)
    # Never suggest more resolution than the data has: upscaling only adds image tokens, not detail.
    longest_side = min(longest_side, _round_up(statistics.median(max(w, h) for w, h in final), multiple))
    # Suggested canvas: the median aspect ratio, longest side capped, both sides a multiple of the patch grid.
    if aspect <= 1:
        height, width = longest_side, longest_side * aspect
    else:
        width, height = longest_side, longest_side / aspect
    return {
        "documents": len(records),
        "pages_per_document": dict(sorted(pages.items())),
        "image_after_merge": {"width_p50": statistics.median(widths), "width_p95": _pct(widths, 0.95),
                              "height_p50": statistics.median(heights), "height_p95": _pct(heights, 0.95),
                              "aspect_p50": round(aspect, 3),
                              "aspect_min": round(min(w / h for w, h in final), 3),
                              "aspect_max": round(max(w / h for w, h in final), 3)},
        "suggested_image_size": (_round_up(width, multiple), _round_up(height, multiple)),
        "note": "Capped at the median source resolution. Merging more pages into one canvas shrinks text.",
    }


def length_report(root: Path, split: str, transform, samples: int = 300, max_length: int | None = None) -> dict:
    """Tokenised length (image tokens included) of real examples drawn the way training draws them."""
    from .dataset import DocumentDataset
    from .processors import DatasetProcessor
    transform.fixed_length, transform.max_length = False, None   # measure, do not pad or reject
    dataset = DocumentDataset(root, split, DatasetProcessor(), samples_per_document=max(1, samples // 8))
    by_task: dict[str, list[int]] = {}
    for i in Progress(range(min(samples, len(dataset))), desc="tokenising examples", unit="example", leave=False):
        item = dataset[i]
        out = transform(item["images"], item["prompt"], item["label"])
        by_task.setdefault(item["task"], []).append(int(out["input_ids"].shape[-1]))
    everything = [n for lengths in by_task.values() for n in lengths]
    suggestion = _round_up(_pct(everything, 0.99), 64)
    limit = max_length or suggestion
    return {
        "examples": len(everything),
        "per_task": {t: {"n": len(v), "p50": _pct(v, 0.5), "p95": _pct(v, 0.95), "max": max(v)}
                     for t, v in sorted(by_task.items())},
        "overall": {"p50": _pct(everything, 0.5), "p95": _pct(everything, 0.95), "p99": _pct(everything, 0.99),
                    "max": max(everything)},
        "suggested_max_length": suggestion,
        f"share_over_{limit}": round(sum(n > limit for n in everything) / len(everything), 4),
        "padding_waste_at_suggestion": round(1 - statistics.mean(everything) / suggestion, 3),
    }


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)   # lines appear at once in notebooks and pipes
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    parser.add_argument("--split", default="train")
    parser.add_argument("--merge", choices=["vertical", "horizontal", "grid"], default=None)
    parser.add_argument("--ocr-mode", choices=["tiny", "small", "base", "large", "gundam"], default="base",
                        help="DeepSeek-OCR resolution mode (v2 has only small, base, gundam)")
    parser.add_argument("--image-size", help="WxH to measure token lengths with, e.g. 896x1176")
    parser.add_argument("--model-id", help="Hugging Face model whose processor tokenises the examples")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--samples", type=int, default=300)
    args = parser.parse_args(argv)

    root = Path(args.root)
    report = {"images": image_report(root, args.split, args.merge)}
    if args.model_id:
        if model_family(args.model_id).startswith("deepseek"):
            transform = create_transform(args.model_id, mode=args.ocr_mode, merge_pages=args.merge or "grid")
            report["tokens"] = length_report(root, args.split, transform, args.samples, args.max_length)
            report["tokens"]["measured_with"] = {"version": transform.version, "mode": args.ocr_mode,
                                                 "merge_pages": transform.merge_pages}
        else:
            size = tuple(int(x) for x in args.image_size.lower().split("x")) if args.image_size else \
                report["images"]["suggested_image_size"]
            transform = create_transform(args.model_id, image_size=size, merge_pages=args.merge)
            report["tokens"] = length_report(root, args.split, transform, args.samples, args.max_length)
            report["tokens"]["measured_with_image_size"] = size
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
