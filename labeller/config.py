from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import prompts


@dataclass(frozen=True)
class StageConfig:
    model: str
    prompt_version: str
    max_output_tokens: int
    send_images: bool = True


@dataclass(frozen=True)
class OcrConfig:
    enabled: bool = False
    engine: str = "rapidocr"
    model_type: str = "server"
    ocr_version: str = "PP-OCRv5"
    min_mean_confidence: float = 0.60
    min_chars_per_page: int = 20
    max_side: int = 1600          # longest image side given to OCR; larger photos are shrunk first (0 = never)


VISUAL_MODES = ("never", "auto", "always")
STAGE1_FLAGS = ("has_handwriting", "has_checkboxes", "has_signature", "has_table")


@dataclass(frozen=True)
class VisualConfig:
    """Decides per document whether Stage 2 also gets the page images (multimodal) or reads text only.

    never: always text only. always: always multimodal. auto: multimodal only for documents with visual
    content, found by free local analysis (and, for scans, by Stage 1's layout flags).
    Thresholds are shares of the page area. They were set from real scans and are untuned for your data.
    """
    mode: str = "never"
    min_image_area: float = 0.02      # an embedded picture (not a full-page scan) covering this share of a page
    min_color_area: float = 0.004     # coloured pixels (stamps, highlights, coloured rows or logos)
    min_shaded_area: float = 0.02     # light shaded blocks such as table-row or header backgrounds
    min_blob_area: float = 0.015      # large solid dark regions: logos, photos, halftone headers (edge bands ignored)
    color_saturation: int = 40        # 0-255; paper tint stays below this
    color_value: int = 100
    pdf_annotations: bool = True      # PDF highlights, stamps and ink count as visual content
    pdf_checkboxes: bool = True       # PDF checkbox / radio form fields count as visual content
    stage1_flags: tuple[str, ...] = ("has_table", "has_checkboxes", "has_signature")


UNKNOWN_POLICIES = ("reject", "accept")
MIXED_POLICIES = ("accept_if_primary_allowed", "accept_if_all_allowed", "reject_mixed")


@dataclass(frozen=True)
class LanguageConfig:
    """Language filtering. Disabled unless a `language:` section is present in the config file."""
    enabled: bool = False
    allowed: frozenset[str] = frozenset()
    unknown_policy: str = "reject"
    mixed_language_policy: str = "accept_if_primary_allowed"
    min_chars: int = 20           # letters needed before a language is decided (CJK letters count triple)
    min_confidence: float = 0.50  # below this the language counts as unknown
    mixed_min_share: float = 0.10  # another language with at least this share of the text makes it mixed


@dataclass(frozen=True)
class AnnotationConfig:
    project_root: Path
    data_root: Path
    max_pages: int
    min_text_chars_per_page: int
    batch_size: int
    max_attempts: int
    tokens_per_page_estimate: int
    stage1: StageConfig
    stage2: StageConfig
    ocr: OcrConfig
    language: LanguageConfig
    visual: VisualConfig
    max_illegible_regions: int
    exclude_blank_forms: bool
    blank_form_min_widgets: int
    blank_form_max_fill_ratio: float
    min_extracted_ratio: float
    min_fields_for_ratio: int
    max_change_ratio: float
    max_value_gap: int
    phi_allowed_sources: frozenset[str]
    train_ratio: float
    val_ratio: float
    seed: int
    upload_workers: int = 8         # page images uploaded to Gemini at the same time
    upload_max_side: int = 2000     # longest side of an uploaded image in px; larger ones are shrunk (0 = send as is)

    @property
    def manifest_path(self) -> Path:
        return self.data_root / "manifest.jsonl"


def _visual(raw: dict[str, Any] | None) -> VisualConfig:
    """No `visual:` section means text-only Stage 2. With a section the mode defaults to auto."""
    if not raw:
        return VisualConfig()
    defaults = VisualConfig()
    flags = tuple(raw.get("stage1_flags", defaults.stage1_flags) or ())
    config = VisualConfig(
        mode=str(raw.get("mode", "auto")),
        min_image_area=float(raw.get("min_image_area", defaults.min_image_area)),
        min_color_area=float(raw.get("min_color_area", defaults.min_color_area)),
        min_shaded_area=float(raw.get("min_shaded_area", defaults.min_shaded_area)),
        min_blob_area=float(raw.get("min_blob_area", defaults.min_blob_area)),
        color_saturation=int(raw.get("color_saturation", defaults.color_saturation)),
        color_value=int(raw.get("color_value", defaults.color_value)),
        pdf_annotations=bool(raw.get("pdf_annotations", defaults.pdf_annotations)),
        pdf_checkboxes=bool(raw.get("pdf_checkboxes", defaults.pdf_checkboxes)),
        stage1_flags=flags,
    )
    if config.mode not in VISUAL_MODES:
        raise ValueError(f"visual.mode must be one of {VISUAL_MODES}")
    bad = [flag for flag in flags if flag not in STAGE1_FLAGS]
    if bad:
        raise ValueError(f"visual.stage1_flags has unknown flags {bad}; use any of {STAGE1_FLAGS}")
    for name in ("min_image_area", "min_color_area", "min_shaded_area", "min_blob_area"):
        if not 0 < getattr(config, name) <= 1:
            raise ValueError(f"visual.{name} must be a share of the page between 0 and 1")
    return config


def _language(raw: dict[str, Any] | None) -> LanguageConfig:
    """No `language:` section means no filtering. With a section, filtering is on unless enabled: false."""
    if not raw:
        return LanguageConfig()
    allowed = frozenset(str(code).strip().lower() for code in (raw.get("allowed") or []))
    config = LanguageConfig(
        enabled=bool(raw.get("enabled", True)),
        allowed=allowed,
        unknown_policy=str(raw.get("unknown_policy", "reject")),
        mixed_language_policy=str(raw.get("mixed_language_policy", "accept_if_primary_allowed")),
        min_chars=int(raw.get("min_chars", 20)),
        min_confidence=float(raw.get("min_confidence", 0.50)),
        mixed_min_share=float(raw.get("mixed_min_share", 0.10)),
    )
    if not config.enabled:
        return config
    if not allowed:
        raise ValueError("language.allowed must list at least one language code (for example: en)")
    bad = sorted(code for code in allowed if not (2 <= len(code) <= 3 and code.isalpha()))
    if bad:
        raise ValueError(f"language.allowed has invalid codes {bad}; use ISO 639-1 codes such as en, zh, ja")
    if config.unknown_policy not in UNKNOWN_POLICIES:
        raise ValueError(f"language.unknown_policy must be one of {UNKNOWN_POLICIES}")
    if config.mixed_language_policy not in MIXED_POLICIES:
        raise ValueError(f"language.mixed_language_policy must be one of {MIXED_POLICIES}")
    return config


def load_dotenv(path: Path) -> None:
    """Minimal .env reader. Existing environment variables win; values are never printed."""
    import os

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.removeprefix("export ").partition("=")
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def _stage(raw: dict[str, Any], name: str) -> StageConfig:
    stage = StageConfig(
        model=str(raw["model"]),
        prompt_version=str(raw["prompt_version"]),
        max_output_tokens=int(raw.get("max_output_tokens", 4096)),
        # Stage 1 sends images by default. Stage 2 is text only unless asked, because images are the costly part.
        send_images=bool(raw.get("send_images", name == "stage1")),
    )
    if stage.prompt_version not in prompts.PROMPTS:
        raise ValueError(f"{name}: unknown prompt_version {stage.prompt_version!r}")
    return stage


def load_config(path: str | Path) -> AnnotationConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    project_root = path.parent.parent
    train = float(raw["export"]["train"])
    val = float(raw["export"]["val"])
    if not (0 < train and 0 <= val and train + val < 1):
        raise ValueError("export.train and export.val must satisfy 0 < train, 0 <= val, train + val < 1")

    ocr_raw = raw.get("ocr") or {}
    ocr = OcrConfig(
        enabled=bool(ocr_raw.get("enabled", False)),
        engine=str(ocr_raw.get("engine", "rapidocr")),
        model_type=str(ocr_raw.get("model_type", "server")),
        ocr_version=str(ocr_raw.get("ocr_version", "PP-OCRv5")),
        min_mean_confidence=float(ocr_raw.get("min_mean_confidence", 0.60)),
        min_chars_per_page=int(ocr_raw.get("min_chars_per_page", 20)),
        max_side=int(ocr_raw.get("max_side", 1600)),
    )
    language = _language(raw.get("language"))
    visual = _visual(raw.get("visual"))
    stage1 = _stage(raw["stage1"], "stage1")
    # Stage 1 v2 takes the OCR draft as input, so it only makes sense when OCR runs first.
    if stage1.prompt_version in prompts.OCR_STAGE1_PROMPTS and not ocr.enabled:
        raise ValueError(f"{stage1.prompt_version} needs the local OCR draft: set ocr.enabled: true")
    if stage1.prompt_version not in prompts.OCR_STAGE1_PROMPTS and ocr.enabled:
        raise ValueError(f"ocr.enabled needs an OCR-aware Stage 1 prompt, not {stage1.prompt_version}")

    gates = raw.get("gates") or {}
    return AnnotationConfig(
        project_root=project_root,
        data_root=(project_root / raw.get("data_root", "./data")).resolve(),
        max_pages=int(raw["max_pages"]),
        min_text_chars_per_page=int(raw["min_text_chars_per_page"]),
        batch_size=int(raw["batch_size"]),
        upload_workers=max(1, int(raw.get("upload_workers", 8))),
        upload_max_side=int(raw.get("upload_max_side", 2000)),
        max_attempts=int(raw["max_attempts"]),
        tokens_per_page_estimate=int(raw.get("tokens_per_page_estimate", 560)),
        stage1=stage1,
        stage2=_stage(raw["stage2"], "stage2"),
        ocr=ocr,
        language=language,
        visual=visual,
        max_illegible_regions=int(gates.get("max_illegible_regions", 0)),
        exclude_blank_forms=bool(gates.get("exclude_blank_forms", True)),
        blank_form_min_widgets=int(gates.get("blank_form_min_widgets", 5)),
        blank_form_max_fill_ratio=float(gates.get("blank_form_max_fill_ratio", 0.1)),
        min_extracted_ratio=float(gates.get("min_extracted_ratio", 0.25)),
        min_fields_for_ratio=int(gates.get("min_fields_for_ratio", 8)),
        max_change_ratio=float(gates.get("max_change_ratio", 0.30)),
        max_value_gap=int(gates.get("max_value_gap", 6)),
        phi_allowed_sources=frozenset(raw.get("phi_allowed_sources") or []),
        train_ratio=train,
        val_ratio=val,
        seed=int(raw["export"]["seed"]),
    )
