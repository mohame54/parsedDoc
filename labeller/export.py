"""Writes a training-ready dataset from accepted annotations.

Layout of the export folder (every path inside a record is relative to that folder):

    images/<doc>/page-N.<ext>     copies of the page images, so the folder can be moved or uploaded as is
    {train,val,test}.jsonl        one canonical record per document (images, text, fields, tables, ...)
    sft/ocr_{split}.jsonl         chat format, one example per page: page image -> page text
    sft/extract_{split}.jsonl     chat format, one example per document: page images -> structured JSON

The many-task mix (schema generation, schema-conditioned extraction, format conversion, QA, ...) is built
at training time from the canonical files by train/processor.py, so it can change every epoch.
    dataset.json                  counts, formats and a loading snippet


Splits are grouped by family_id (the document type) so that near-identical layouts never land in
both train and test. Documents whose licence is not VERIFIED go to export/unverified/ unless the
caller asks otherwise, so nothing reaches the training set by accident.
"""
from __future__ import annotations

import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from .config import AnnotationConfig
from .pipeline import page_paths
from .store import Store, read_json, safe_name, write_json
from tqdm import tqdm


def export(cfg: AnnotationConfig, store: Store, docs: dict[str, dict], include_unverified: bool = False,
           copy_images: bool = True, progress: bool = False) -> dict:
    export_root = store.root / "export"
    statuses = store.all_statuses()
    target = export_root / "unverified" if include_unverified else export_root
    target.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    accepted = [st for st in statuses.values() if st["state"] == "accepted"]
    for st in tqdm(accepted, desc="export", unit="doc", leave=False, disable=not progress):
        doc = docs.get(st["doc_id"])
        annotation = read_json(store.ann_dir(st["doc_id"]) / "annotation.json")
        if doc is None or annotation is None:
            continue
        verified = doc.get("license_review_status") == "VERIFIED"
        if not verified and not include_unverified:
            continue
        records.append(_record(store, st["doc_id"], doc, annotation, copy_images, target))

    splits = _split(records, cfg)
    for name, rows in splits.items():
        _write_jsonl(target / f"{name}.jsonl", [dict(r["canonical"], split=name) for r in rows])
        _write_jsonl(target / "sft" / f"ocr_{name}.jsonl", [e for r in rows for e in _ocr_examples(r)])
        _write_jsonl(target / "sft" / f"extract_{name}.jsonl", [_extract_example(r) for r in rows])

    excluded_rows = [
        {"doc_id": st["doc_id"], "reason": st.get("reason"), "route": st.get("route")}
        for st in statuses.values() if st["state"] == "excluded"
    ]
    with (export_root / "excluded.jsonl").open("w", encoding="utf-8") as handle:
        for row in excluded_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "target": str(target),
        "accepted_written": len(records),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "excluded": len(excluded_rows),
        "by_state": _count_states(statuses),
    }
    write_json(export_root / "summary.json", summary)
    write_json(target / "dataset.json", _dataset_card(summary, target))
    return summary


def _qa_path(image: Path) -> Path:
    return image.with_name(image.stem + ".qa.json")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _place_image(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _record(store: Store, doc_id: str, doc: dict, annotation: dict, copy_images: bool, target: Path) -> dict:
    """Canonical record plus the pieces the chat formats are built from."""
    images: list[str] = []
    for n, rel in enumerate(page_paths(doc), start=1):
        src = store.root / rel
        if copy_images:
            dest_rel = f"images/{safe_name(doc_id)}/page-{n}{Path(rel).suffix.lower()}"
            _place_image(src, target / dest_rel)
            images.append(dest_rel)
        else:
            images.append(rel)
    source = read_json(store.ann_dir(doc_id) / "source_text.json") or {}
    pages = list(source.get("pages") or [])
    components = {key: annotation.get(key) or [] for key in ("tables", "visual_elements", "layout", "relationships",
                                                              "figures")}
    qa = read_json(_qa_path(store.root / doc["original_path"])) or {}
    canonical = {
        "id": doc_id,
        "source": doc.get("source_id"),
        "images": images,
        "page_count": len(images),
        "text": pages,
        "text_origin": source.get("origin"),
        "document_type": annotation.get("document_type"),
        "family_id": annotation.get("family_id"),
        "language": (annotation.get("language") or {}).get("primary") if isinstance(annotation.get("language"), dict)
        else annotation.get("language"),
        "fields": annotation.get("fields") or [],
        **components,
        "qa": qa.get("qa") or [],          # question-answer pairs about the image (ChartQA), read from its sidecar
        "visual_routing": annotation.get("visual_routing"),
        "meta": {
            "category": doc.get("category"),
            "license": doc.get("license_declared"),
            "license_review_status": doc.get("license_review_status"),
            "source_url": doc.get("source_url"),
            "origin": annotation.get("origin"),
            "schema_fingerprint": annotation.get("schema_fingerprint"),
            "provenance": annotation.get("provenance"),
        },
    }
    return {"family_id": annotation["family_id"], "canonical": canonical}


OCR_PROMPT = "Transcribe all text on this page exactly as written, in reading order."
EXTRACT_PROMPT = ("Extract the structured data from this document. Return JSON with document_type, "
                  "fields (grouped by section), tables, visual_elements and relationships.")


def _user_turn(images: list[str], prompt: str) -> dict:
    return {"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]}


def _ocr_examples(record: dict) -> list[dict]:
    c = record["canonical"]
    out = []
    for n, (image, text) in enumerate(zip(c["images"], c["text"]), start=1):
        if not text.strip():
            continue
        out.append({
            "id": f"{c['id']}#p{n}", "task": "ocr", "images": [image],
            "messages": [_user_turn([image], OCR_PROMPT), {"role": "assistant", "content": text}],
        })
    return out


def _extract_example(record: dict) -> dict:
    c = record["canonical"]
    grouped: dict[str, dict] = {}
    for f in c["fields"]:
        if f.get("status") != "extracted":
            continue  # absent, illegible or uncertain values are not training targets
        value = f.get("normalized_value")
        grouped.setdefault(f["section"], {})[f["name"]] = value if value not in ("", None) else f.get("raw_value")
    target = {"document_type": c["document_type"], "fields": grouped}
    for key in ("tables", "visual_elements", "relationships", "figures"):
        if c.get(key):
            target[key] = c[key]
    return {
        "id": c["id"], "task": "extract", "images": c["images"],
        "messages": [_user_turn(c["images"], EXTRACT_PROMPT),
                     {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}],
    }


def _dataset_card(summary: dict, target: Path) -> dict:
    return {
        "root": ".",
        "note": "Every path in the JSONL files is relative to this folder.",
        "splits": summary["splits"],
        "files": {
            "canonical": "{split}.jsonl",
            "sft_ocr": "sft/ocr_{split}.jsonl",
            "sft_extract": "sft/extract_{split}.jsonl",
        },
        "load": ("from datasets import load_dataset\n"
                 "ds = load_dataset('json', data_files={'train': 'sft/extract_train.jsonl'})  # run from this folder\n"
                 "# many-task training: see train/dataset.py\n"
                 "# images are paths in row['images']; open with PIL.Image.open(path)"),
    }


def _split(records: list[dict], cfg: AnnotationConfig) -> dict[str, list[dict]]:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_family[record["family_id"]].append(record)
    families = sorted(by_family)
    random.Random(cfg.seed).shuffle(families)

    total = len(records)
    train_cut = total * cfg.train_ratio
    val_cut = total * (cfg.train_ratio + cfg.val_ratio)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    seen = 0
    for family in families:
        name = "train" if seen < train_cut else ("val" if seen < val_cut else "test")
        splits[name].extend(by_family[family])
        seen += len(by_family[family])
    return splits


def _count_states(statuses: dict[str, dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for st in statuses.values():
        counts[st["state"]] += 1
    return dict(counts)
