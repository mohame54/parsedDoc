"""Page-image helpers shared by every transform: merging a document's pages into one image, and the
size arithmetic that predicts the merged size without touching pixels."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

MERGE_MODES = ("vertical", "horizontal", "grid")


def merge_page_images(images: list[Image.Image], mode: str | None, gap: int = 8) -> list[Image.Image]:
    """Joins a document's pages into ONE image (a list with one item), with a grey divider between pages.

    vertical / horizontal: pages scaled to a common width / height, stacked in a line.
    grid: two columns, as many rows as needed (2 pages side by side, 3-4 pages in a 2x2 grid). Each page
        is fitted into a cell the size of the largest page. Keeps the canvas close to square, which suits
        models whose image view is square (DeepSeek-OCR)."""
    if not mode or len(images) < 2:
        return images
    grey = (200, 200, 200)
    if mode == "grid":
        cols = 2
        rows = -(-len(images) // cols)
        cell_w, cell_h = max(i.width for i in images), max(i.height for i in images)
        canvas = Image.new("RGB", (cols * cell_w + gap * (cols - 1), rows * cell_h + gap * (rows - 1)), grey)
        for n, image in enumerate(images):
            scale = min(cell_w / image.width, cell_h / image.height)
            fitted = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                                  Image.Resampling.LANCZOS)
            cell = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
            cell.paste(fitted, ((cell_w - fitted.width) // 2, (cell_h - fitted.height) // 2))
            canvas.paste(cell, ((n % cols) * (cell_w + gap), (n // cols) * (cell_h + gap)))
        return [canvas]
    vertical = mode == "vertical"
    common = max(i.width for i in images) if vertical else max(i.height for i in images)
    scaled = []
    for image in images:
        ratio = common / (image.width if vertical else image.height)
        size = (common, round(image.height * ratio)) if vertical else (round(image.width * ratio), common)
        scaled.append(image.resize((max(1, size[0]), max(1, size[1])), Image.Resampling.LANCZOS))
    span = sum((i.height if vertical else i.width) for i in scaled) + gap * (len(scaled) - 1)
    canvas = Image.new("RGB", (common, span) if vertical else (span, common), grey)
    offset = 0
    for image in scaled:
        canvas.paste(image, (0, offset) if vertical else (offset, 0))
        offset += (image.height if vertical else image.width) + gap
    return [canvas]


def merged_size(sizes: list[tuple[int, int]], mode: str | None, gap: int = 8) -> tuple[int, int]:
    """Size of the image merge_page_images would build from pages of these (width, height) sizes, computed
    without touching any pixels."""
    if not mode or len(sizes) < 2:
        return tuple(sizes[0]) if len(sizes) == 1 else (0, 0)
    if mode == "grid":
        cols, rows = 2, -(-len(sizes) // 2)
        cell_w, cell_h = max(w for w, _ in sizes), max(h for _, h in sizes)
        return cols * cell_w + gap * (cols - 1), rows * cell_h + gap * (rows - 1)
    if mode == "vertical":
        common = max(w for w, _ in sizes)
        return common, sum(max(1, round(h * common / w)) for w, h in sizes) + gap * (len(sizes) - 1)
    common = max(h for _, h in sizes)
    return sum(max(1, round(w * common / h)) for w, h in sizes) + gap * (len(sizes) - 1), common


def as_rgb_page(image: Any) -> Image.Image | None:
    """Normalises one page input (PIL image, path, or None) to an RGB PIL image. None stays None."""
    if image is None:
        return None
    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            return img.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image).__name__}. Expected PIL.Image, str, Path, or None.")


def letterbox(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scales to fit `size`, keeps the aspect ratio and centres on a white canvas. Never distorts text."""
    width, height = size
    scale = min(width / image.width, height / image.height)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                           Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas
