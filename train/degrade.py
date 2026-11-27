"""Synthetic degradation of page images: rotation, skew, noise, blur, JPEG and contrast.

Used by both pipelines with different levels (`soft` for SFT, where the label must stay true; `strong` for RL, where
the model is also asked to report the rotation and to abstain on unreadable fields). Everything is drawn from a
`random.Random` the caller seeds, so the same example always gets the same degradation.

Conventions (used by the RL reward too, so define them once here):
  * `info["angle"]` is the clockwise turn applied to the page to degrade it (a degree in [0, 360)).
  * `upright_turn(info)` is the clockwise turn that puts the degraded page back upright: `(360 - angle) % 360`.
    That is the value an RL answer reports as its rotation.
  * Only rotation changes the image size (`expand=True`), so `degraded_size` is exact for everything else.
  * `plan_example` draws every parameter from a seed without touching pixels; `render` applies them.
"""
from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

WHITE = (255, 255, 255)


@dataclass(frozen=True)
class Level:
    """Ranges one degradation level draws from. `quarter_p` / `arbitrary_p` are the chances of a 90/180/270 turn or
    of a free angle; a page can get neither (it stays upright apart from skew)."""
    quarter_p: float = 0.0
    arbitrary_p: float = 0.0
    skew_max: float = 0.0          # |angle| in degrees for a small skew (drawn when neither turn is picked)
    arbitrary_max: float = 45.0    # |angle| for a free turn (before it is folded to [0, 360))
    noise_max: float = 0.0         # Gaussian noise sigma, in 0-255 units
    blur_max: float = 0.0          # Gaussian blur radius in pixels
    jpeg_min: int = 100            # lowest JPEG quality drawn (100 = no compression)
    contrast_min: float = 1.0      # lowest contrast factor drawn (1 = unchanged)


LEVELS: dict[str, Level] = {
    "off": Level(),
    "soft": Level(skew_max=4.0, noise_max=6.0, blur_max=0.6, jpeg_min=60, contrast_min=0.8),
    "strong": Level(quarter_p=0.25, arbitrary_p=0.35, skew_max=8.0, arbitrary_max=45.0, noise_max=25.0,
                    blur_max=2.0, jpeg_min=20, contrast_min=0.4),
}


def level_for(name: str) -> Level:
    if name not in LEVELS:
        raise ValueError(f"unknown degradation level {name!r}; expected one of {sorted(LEVELS)}")
    return LEVELS[name]


def draw(rng: random.Random, level: Level) -> dict[str, Any]:
    """The degradation parameters for one page. Always consumes the same number of draws (the pixel seed last), so
    a plan made without pixels matches what `render` does with them."""
    roll = rng.random()
    if roll < level.quarter_p:
        angle = float(rng.choice((90, 180, 270)))
    elif roll < level.quarter_p + level.arbitrary_p:
        angle = rng.uniform(-level.arbitrary_max, level.arbitrary_max) % 360
    else:
        angle = rng.uniform(-level.skew_max, level.skew_max) % 360 if level.skew_max else 0.0
    noise = rng.uniform(0, level.noise_max)
    blur = rng.uniform(0, level.blur_max)
    jpeg_roll = rng.random()
    jpeg = rng.randint(min(level.jpeg_min, 95), 95)
    contrast = rng.uniform(level.contrast_min, 1.0)
    return {
        "angle": round(angle % 360, 3),
        "noise": round(noise, 2) if level.noise_max else 0.0,
        "blur": round(blur, 3) if level.blur_max else 0.0,
        "jpeg": jpeg if level.jpeg_min < 95 and jpeg_roll < 0.5 else 100,
        "contrast": round(contrast, 3) if level.contrast_min < 1.0 else 1.0,
        "seed": rng.getrandbits(32),
    }


def plan_example(rng: random.Random, level: str | Level, n_pages: int, p: float = 1.0) -> list[dict[str, Any]] | None:
    """Parameters for every page of one example, or None when the example is left clean (chance `p`). Consumes
    `rng` in a fixed order, so the plan can be made from the seed alone, without images."""
    lvl = level_for(level) if isinstance(level, str) else level
    if rng.random() >= p:
        return None
    return [draw(rng, lvl) for _ in range(n_pages)]


def upright_turn(info: dict[str, Any]) -> float:
    """Clockwise degrees that bring the degraded page back upright, in [0, 360)."""
    return round((360 - info.get("angle", 0.0)) % 360, 3)


def rotated_size(width: int, height: int, angle: float) -> tuple[int, int]:
    """Size PIL gives `Image.rotate(..., expand=True)` for a turn of `angle` degrees, without rotating. Mirrors
    PIL's arithmetic: matrix entries rounded to 15 decimals, corners taken about the centre, then ceil(max) -
    floor(min). Checked against real rotations in tests/test_degrade.py."""
    angle %= 360
    if angle % 90 == 0:
        return (height, width) if angle % 180 == 90 else (width, height)
    t = math.radians(angle)
    cos, sin = round(math.cos(t), 15), round(math.sin(t), 15)
    cx, cy = width / 2.0, height / 2.0
    xs, ys = [], []
    for x, y in ((0, 0), (width, 0), (width, height), (0, height)):
        dx, dy = x - cx, y - cy
        xs.append(cos * dx + sin * dy + cx)
        ys.append(-sin * dx + cos * dy + cy)
    return math.ceil(max(xs)) - math.floor(min(xs)), math.ceil(max(ys)) - math.floor(min(ys))


def degraded_size(width: int, height: int, info: dict[str, Any]) -> tuple[int, int]:
    """Size of the image `degrade` returns for a page of this size and these parameters."""
    return rotated_size(width, height, info.get("angle", 0.0))


def _turn(image: Image.Image, angle: float) -> Image.Image:
    """Clockwise turn (PIL's rotate is counter-clockwise, hence the minus)."""
    if angle % 360 == 0:
        return image
    return image.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=WHITE)


def render(image: Image.Image, info: dict[str, Any]) -> Image.Image:
    """Applies one page's planned degradation. Deterministic: the same `info` always gives the same pixels."""
    out = _turn(image.convert("RGB"), info["angle"])
    if info["contrast"] != 1.0:
        out = Image.fromarray(np.clip(
            (np.asarray(out, dtype=np.float32) - 128) * info["contrast"] + 128, 0, 255).astype(np.uint8))
    if info["noise"]:
        gen = np.random.default_rng(info["seed"])
        arr = np.asarray(out, dtype=np.float32) + gen.normal(0, info["noise"], size=(out.height, out.width, 3))
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if info["blur"]:
        out = out.filter(ImageFilter.GaussianBlur(radius=info["blur"]))
    if info["jpeg"] < 100:
        buffer = io.BytesIO()
        out.save(buffer, format="JPEG", quality=info["jpeg"])
        buffer.seek(0)
        with Image.open(buffer) as reread:
            out = reread.convert("RGB")
    return out


def degrade(image: Image.Image, rng: random.Random, level: str | Level = "soft") -> tuple[Image.Image, dict[str, Any]]:
    """One page: draw its parameters and render them. Returns the image and `info` (with `upright`)."""
    info = draw(rng, level_for(level) if isinstance(level, str) else level)
    return render(image, info), {**info, "upright": upright_turn(info)}
