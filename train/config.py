"""YAML configs for training, so a run is one file instead of a long command.

    uv run --extra train python -m train.train --config config/train.yaml
    uv run --extra train python -m train.train --config train_qlora --batch-tokens 4800 --epochs 3

Precedence, lowest to highest: the defaults in `train/train.py`, the YAML file (and the files it `extends`),
then flags on the command line. So a file sets the baseline and a flag changes one thing for one run.

The file groups settings in sections; every key is the matching flag without its dashes:

    extends: train.yaml        # optional: start from another file in the same folder, then override
    model:       id, ocr_mode, train_vision, quant
    lora:        enabled, r, alpha, dropout, target_modules
    data:        root, preset, descriptions, max_length, merge_pages, workers, samples_per_document,
                 max_documents
    batching:    batch_size, batch_tokens, max_batch_size, pad_multiple, pad_to_max, loss_weighting
    optimizer:   name, lr, muon_lr, muon_adjust, weight_decay, max_grad_norm, scheduler, warmup_ratio, min_lr_ratio
    training:    strategy, epochs, grad_accum, precision, compile, max_steps
    checkpoints: out_dir, save_every, log_every, tensorboard, resume
    hub:         repo, private, every_save           (the token is never read from a file: use HF_TOKEN)

An unknown section or key is an error with a suggestion, so a typo cannot silently fall back to a default.
"""
from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# section -> {yaml key: argparse destination}
SECTIONS: dict[str, dict[str, str]] = {
    "model": {"id": "model_id", "ocr_mode": "ocr_mode", "train_vision": "train_vision", "quant": "quant"},
    "lora": {"enabled": "lora", "r": "lora_r", "alpha": "lora_alpha", "dropout": "lora_dropout",
             "target_modules": "lora_targets"},
    "data": {"root": "root", "preset": "preset", "descriptions": "descriptions", "max_length": "max_length",
             "merge_pages": "merge_pages", "workers": "workers",
             "samples_per_document": "samples_per_document", "max_documents": "max_documents"},
    "batching": {"batch_size": "batch_size", "batch_tokens": "batch_tokens", "max_batch_size": "max_batch_size",
                 "pad_multiple": "pad_multiple", "pad_to_max": "pad_to_max", "loss_weighting": "loss_weighting"},
    "optimizer": {"name": "optimizer", "lr": "lr", "muon_lr": "muon_lr", "muon_adjust": "muon_adjust",
                  "weight_decay": "weight_decay", "max_grad_norm": "max_grad_norm", "scheduler": "scheduler",
                  "warmup_ratio": "warmup_ratio", "min_lr_ratio": "min_lr_ratio"},
    "training": {"strategy": "strategy", "epochs": "epochs", "grad_accum": "grad_accum", "precision": "precision",
                 "compile": "compile", "max_steps": "max_steps"},
    "checkpoints": {"out_dir": "out_dir", "save_every": "save_every", "log_every": "log_every",
                    "tensorboard": "tensorboard", "resume": "resume"},
    "hub": {"repo": "hub_repo", "private": "hub_private", "every_save": "hub_every_save"},
}


def find_config(name: str | Path) -> Path:
    """`config/train.yaml`, `train.yaml` or `train` (a file in the config folder) -> the path."""
    path = Path(name)
    candidates = [path, path.with_suffix(".yaml"), CONFIG_DIR / path, CONFIG_DIR / path.with_suffix(".yaml")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no training config {str(name)!r} (looked in the current folder and {CONFIG_DIR})")


def _merge(base: dict, over: dict) -> dict:
    out = {key: dict(value) if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def _read(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"`extends` loops back to {path.name}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name}: the top level must be a mapping of sections")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    beside = path.parent / parent
    return _merge(_read(beside if beside.is_file() else find_config(parent), seen + (path,)), raw)


def _suggest(word: str, options: list[str]) -> str:
    close = difflib.get_close_matches(word, options, n=1)
    return f" Did you mean {close[0]!r}?" if close else f" Known: {', '.join(options)}."


def load_config(name: str | Path) -> dict[str, Any]:
    """Reads a config (following `extends`) and returns {argparse destination: value}."""
    path = find_config(name)
    merged = _read(path)
    flat: dict[str, Any] = {}
    for section, values in merged.items():
        if section not in SECTIONS:
            raise ValueError(f"{path.name}: unknown section {section!r}." + _suggest(section, list(SECTIONS)))
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError(f"{path.name}: section {section!r} must be a mapping")
        for key, value in values.items():
            if key not in SECTIONS[section]:
                raise ValueError(f"{path.name}: unknown key {section}.{key}." + _suggest(key, list(SECTIONS[section])))
            flat[SECTIONS[section][key]] = value
    return flat


def check_choices(parser, args) -> None:
    """argparse does not check `choices` for values that came from a config file, so do it here."""
    for action in parser._actions:
        value = getattr(args, action.dest, None)
        if action.choices is not None and value is not None and value not in action.choices:
            parser.error(f"{action.dest}={value!r} is not one of {list(action.choices)} (set in the config file)")
