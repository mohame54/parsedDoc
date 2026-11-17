from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .datasets import (
    download_grouped_qa,
    download_huggingface,
    download_zenodo
)
from .labels import attach_xfund
from .remote_zip import download_remote_zip

from .utils import (
    download_stream,
    discover_files,
    safe_extract_zip,
    bytes_to_gb,
    sha256_file,
    classify,
    render_pdf,
    pdf_page_count,
    pdf_form_fill,
    is_blank_form,
    Budget
)

import yaml
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DOC_EXTS = IMAGE_EXTS | {".pdf"}



def main() -> None:
    parser = argparse.ArgumentParser(description="Download, slice and catalogue real-world document datasets.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", action="append", help="Source ID; repeat to select multiple. Default: all enabled.")
    parser.add_argument("--dry-run", action="store_true", help="Print enabled source plan without downloading.")
    parser.add_argument("--max-total-gb", type=float, help="Override global storage budget.")
    parser.add_argument("--max-documents", type=int, help="Override per-source document limit.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    project = cfg["project"]
    root = Path(project["output_dir"]).resolve()
    raw_root, extracted_root, page_root = root / "raw", root / "extracted", root / "pages"
    manifest_path = root / "manifest.jsonl"
    max_total_gb = args.max_total_gb or float(project.get("max_total_gb", 20))
    budget = Budget(root, max_total_gb)

    sources = cfg.get("sources", [])
    selected = set(args.source or [])
    sources = [s for s in sources if (s.get("enabled") or s["id"] in selected)
               and (not selected or s["id"] in selected)]
    if args.dry_run:
        for s in sources:
            print(f'{s["id"]}: kind={s["kind"]}, category={s.get("category")}, '
                  f'max_docs={args.max_documents or s.get("max_documents")}, '
                  f'max_gb={s.get("max_download_gb")}')
        return

    rules = cfg.get("categories", {}).get("rules", {})
    allowed = set(project.get("allowed_categories", []))
    # Merge with the existing manifest so re-runs are incremental and never drop earlier documents.
    rows_by_id: dict[str, dict[str, Any]] = {}
    rejected_rows: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows_by_id[row["document_id"]] = row
    seen_hashes: set[str] = {row["sha256"] for row in rows_by_id.values()}
    for s_ in sources:
        if s_.get("visual_mode") not in (None, "auto", "always", "never"):
            raise SystemExit(f"{s_['id']}: visual_mode must be auto, always or never")
    source_bar = tqdm(sources, desc="sources", unit="source", position=0)
    for source in source_bar:
        sid = source["id"]
        source_bar.set_postfix(now=sid, refresh=True)
        tqdm.write(f"\n=== {sid} ===")
        source_raw = raw_root / sid
        source_extracted = extracted_root / sid
        source_pages = page_root / sid
        source_raw.mkdir(parents=True, exist_ok=True)
        kind = source["kind"]
        visual_mode = source.get("visual_mode")   # auto (default), always or never: per-document Stage 2 images
        max_gb = float(source.get("max_download_gb", 2))
        max_bytes = int(max_gb * 1024**3)

        try:
            if kind == "manual":
                tqdm.write("[manual] " + source.get("instructions", "See source documentation"))
                continue
            if kind == "direct":
                url = source.get("url", "")
                if not url:
                    tqdm.write("[skip] No direct URL configured; see source_url in config.")
                    continue
                filename = Path(urlparse(url).path).name or f"{sid}.download"
                archive = source_raw / filename
                if not archive.exists():
                    download_stream(url, archive, budget, max_bytes)
                downloaded = [archive]
            elif kind == "zenodo":
                downloaded = download_zenodo(str(source["record_id"]), source_raw, budget, max_bytes)
            elif kind == "huggingface":
                loader = download_grouped_qa if source.get("group_by_image") else download_huggingface
                downloaded = loader(
                    source, source_extracted, args.max_documents or int(source.get("max_documents", 1000)),
                    budget=budget, max_bytes=max_bytes,
                )
            elif kind == "remote_zip":
                # Reads the archive's table of contents with HTTP Range requests and fetches only a sample.
                downloaded = download_remote_zip(
                    source, source_extracted, budget, max_bytes,
                    args.max_documents or int(source.get("max_documents", 1000)),
                    seed=int(project.get("random_seed", 42)),
                )
                if source.get("annotations_url"):
                    # The zip holds images only; the text labels are a separate (small) JSON asset.
                    ann = source_raw / Path(urlparse(source["annotations_url"]).path).name
                    if not ann.exists():
                        download_stream(source["annotations_url"], ann, budget, max_bytes)
                    tqdm.write(f"[labels] {sid}: {attach_xfund(ann, downloaded, sid)} of {len(downloaded)} "
                               "documents have annotation text (they skip Stage 1)")
            else:
                raise ValueError(f"Unsupported source kind: {kind}")

            # Extract supported archives, safely. Keep originals in raw/ for provenance.
            # (huggingface and remote_zip already wrote their files under source_extracted.)
            if kind not in ("huggingface", "remote_zip"):
                for archive in downloaded:
                    if archive.suffix.lower() == ".zip":
                        target = source_extracted / archive.stem
                        if not target.exists():
                            safe_extract_zip(archive, target)
                    elif archive.suffix.lower() in DOC_EXTS:
                        source_extracted.mkdir(parents=True, exist_ok=True)
                        target = source_extracted / archive.name
                        if archive.resolve() != target.resolve() and not target.exists():
                            shutil.copy2(archive, target)

            # Count extracted payloads too: ZIP expansion can be much larger than the archive.
            # This keeps the configured total storage budget meaningful after decompression.
            budget.rescan()
            budget.check()

            candidates = discover_files(source_extracted, int(project.get("max_files_per_source", 20000)))
            # For word-crop corpora, every image is an individual sample; for PDFs, one PDF is one document.
            doc_limit = args.max_documents or int(source.get("max_documents", project.get("max_documents_per_source", 5000)))
            if project.get("sampling", "random") == "random":
                rng = random.Random(int(project.get("random_seed", 42)))
                rng.shuffle(candidates)
            candidates = candidates[:doc_limit]

            accepted = 0
            catalogue = tqdm(candidates, desc=f"{sid} catalogue", unit="doc", position=1, leave=False)
            for path in catalogue:
                if project.get("max_document_mb", 0) and path.stat().st_size > float(project["max_document_mb"]) * 1024**2:
                    continue
                digest = sha256_file(path)
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                category, category_method = classify(path, source.get("category", "other"), rules, allowed)
                form_fill = pdf_form_fill(path) if path.suffix.lower() == ".pdf" else None
                if form_fill and is_blank_form(form_fill, project):
                    reason = f"blank_form:widgets={form_fill['widgets']},filled={form_fill['filled']}"
                    rejected_rows[str(path.relative_to(root))] = {
                        "source_id": sid, "original_path": str(path.relative_to(root)),
                        "sha256": digest, "reason": reason,
                    }
                    tqdm.write(f"[reject] {path.name}: {reason}")
                    rows_by_id.pop(f"{sid}:{digest[:16]}", None)  # drop any row from an earlier run
                    continue
                pages = []
                if path.suffix.lower() == ".pdf" and project.get("render_pdf_pages", True):
                    pages = render_pdf(
                        path, source_pages / path.stem, int(project.get("pdf_dpi", 180)),
                        int(project.get("max_pages_per_document", 20))
                    )
                    for page in pages:
                        budget.check(page.stat().st_size)
                elif path.suffix.lower() in IMAGE_EXTS:
                    pages = [path]

                # True page count (rendering may be capped), used by the labeller's page gate.
                page_count = pdf_page_count(path) if path.suffix.lower() == ".pdf" else 1
                rows_by_id[f"{sid}:{digest[:16]}"] = {
                    "document_id": f"{sid}:{digest[:16]}",
                    "source_id": sid,
                    "processor": source.get("processor", "generic"),
                    "visual_mode": visual_mode,
                    "phi_risk": bool(source.get("phi_risk", False)),
                    "source_url": source.get("source_url"),
                    "source_description": source.get("description"),
                    "license_declared": source.get("license", "UNKNOWN - REVIEW REQUIRED"),
                    "license_review_status": "UNVERIFIED",
                    "category": category,
                    "category_method": category_method,
                    "original_path": str(path.relative_to(root)),
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                    "file_type": path.suffix.lower().lstrip("."),
                    "page_count": page_count,
                    "form_fill": form_fill,
                    "page_paths": [str(p.relative_to(root)) for p in pages],
                    "annotation_status": "unlabeled",
                    "real_world_status": "source_claim_not_independently_verified"
                }
                accepted += 1
                catalogue.set_postfix(accepted=accepted, refresh=False)
                if accepted >= doc_limit:
                    break
            catalogue.close()
            tqdm.write(f"[done] {sid}: discovered={len(candidates)} accepted={accepted}")
        except Exception as exc:
            tqdm.write(f"[error] {sid}: {exc}")

    source_bar.close()

    # Re-apply the blank-form rule to rows from earlier runs (they are skipped as seen duplicates above).
    for doc_id, row in list(rows_by_id.items()):
        if row.get("file_type") != "pdf":
            continue
        if "form_fill" not in row:  # manifests written before form_fill existed
            row["form_fill"] = pdf_form_fill(root / row["original_path"])
        if row["form_fill"] and is_blank_form(row["form_fill"], project):
            ff = row["form_fill"]
            reason = f"blank_form:widgets={ff['widgets']},filled={ff['filled']}"
            rejected_rows[row["original_path"]] = {
                "source_id": row["source_id"], "original_path": row["original_path"],
                "sha256": row["sha256"], "reason": reason,
            }
            rows_by_id.pop(doc_id)
            print(f"[reject] {Path(row['original_path']).name}: {reason}")

    rows = list(rows_by_id.values())
    root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Rejections are kept in their own log so every dropped document has a recorded reason.
    rejected_path = root / "rejected.jsonl"
    merged_rejected: dict[str, dict[str, Any]] = {}
    if rejected_path.exists():
        for line in rejected_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                merged_rejected[entry["original_path"]] = entry
    merged_rejected.update(rejected_rows)
    with rejected_path.open("w", encoding="utf-8") as f:
        for entry in merged_rejected.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Simple category/source summary for quick corpus inspection.
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        summary.setdefault(row["category"], {})
        summary[row["category"]][row["source_id"]] = summary[row["category"]].get(row["source_id"], 0) + 1
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nManifest: {manifest_path}")
    print(f"Documents: {len(rows)}")
    print(f"Disk usage: {bytes_to_gb(budget.used()):.3f} GB / {max_total_gb:.3f} GB")


if __name__ == "__main__":
    main()
