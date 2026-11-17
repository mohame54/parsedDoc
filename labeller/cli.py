from __future__ import annotations

import argparse
import sys
from collections import Counter

from pathlib import Path

from . import describe
from .config import load_config, load_dotenv
from .export import export
from .gemini import BatchApi
from .ocr import RapidOcrEngine
from tqdm import tqdm

from .pipeline import (
    collect,
    load_manifest,
    ocr_step,
    prepare,
    reevaluate,
    refresh_visual,
    run_all,
    submit,
    watch,
)
from .store import Store


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="labeller", description="Gemini Batch labeller for the document corpus.")
    parser.add_argument("--config", default="config/annotation.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan", help="Route documents and estimate cost. Writes nothing.")
    sub.add_parser("prepare", help="Route documents locally (free). Safe to re-run.")
    sub.add_parser("status", help="Counts per state and top exclusion reasons.")
    sub.add_parser("ocr", help="Run local OCR for documents that need it (free, cached per document).")
    p_run = sub.add_parser("run", help="Whole pipeline with progress: prepare, OCR, Stage 1, Stage 2, export.")
    p_run.add_argument("--limit", type=int, help="Only submit the first N pending documents per stage.")
    p_run.add_argument("--interval", type=int, default=30, help="Seconds between polls while waiting on Gemini.")
    p_run.add_argument("--include-unverified", action="store_true")
    p_submit = sub.add_parser("submit", help="Upload and submit a Gemini Batch job (costs money).")
    p_submit.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p_submit.add_argument("--limit", type=int, help="Only submit the first N pending documents (pilot runs).")
    p_collect = sub.add_parser("collect", help="Poll jobs, then validate and accept or exclude results.")
    p_collect.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p_collect.add_argument("--watch", action="store_true", help="Keep polling until every batch has finished.")
    p_collect.add_argument("--interval", type=int, default=30, help="Seconds between polls with --watch.")
    sub.add_parser("reevaluate", help="Re-run the gates on saved Gemini outputs after changing a gate (no API calls).")
    p_visual = sub.add_parser("visual", help="Show each document's visual signals and routing (free).")
    p_visual.add_argument("--apply", action="store_true",
                          help="Save the result for documents that have not reached Stage 2 yet.")
    p_export = sub.add_parser("export", help="Write train/val/test JSONL from accepted documents.")
    p_export.add_argument("--include-unverified", action="store_true",
                          help="Also export documents whose licence is not VERIFIED (written to export/unverified/).")
    p_export.add_argument("--no-copy-images", action="store_true",
                          help="Keep image paths pointing into data/ instead of copying images into the export folder.")
    p_dsub = sub.add_parser("describe-submit", help="Optional: ask Gemini for attribute descriptions (costs money).")
    p_dsub.add_argument("--records", nargs="+", required=True, help="exported JSONL files, e.g. data/export/train.jsonl")
    p_dsub.add_argument("--out-dir", default="data/descriptions")
    p_dcol = sub.add_parser("describe-collect", help="Write descriptions.json once the describe job has finished.")
    p_dcol.add_argument("--out-dir", default="data/descriptions")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    load_dotenv(cfg.project_root / ".env")
    store = Store(cfg)
    try:
        docs = load_manifest(cfg)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    if args.cmd == "plan":
        report = prepare(cfg, store, docs, dry_run=True, progress=True)
        _print_plan(cfg, report)
    elif args.cmd == "prepare":
        report = prepare(cfg, store, docs, progress=True)
        print(f"Routed: {report['counts']}")
    elif args.cmd == "status":
        _print_status(store)
    elif args.cmd == "ocr":
        result = ocr_step(cfg, store, _engine_factory(cfg), docs, out=tqdm.write, progress=True)
        print(f"OCR: {dict(result)}")
    elif args.cmd == "submit":
        api = BatchApi()
        count = submit(cfg, store, api, args.stage, docs, limit=args.limit, out=tqdm.write, progress=True)
        print(f"Submitted {count} requests for stage {args.stage}.")
    elif args.cmd == "collect":
        api = BatchApi()
        if args.watch:
            result = watch(cfg, store, api, args.stage, docs, interval=args.interval, out=tqdm.write, progress=True)
        else:
            result = collect(cfg, store, api, args.stage, docs, out=tqdm.write, progress=True)
        print(f"Collect stage {args.stage}: {dict(result)}")
    elif args.cmd == "run":
        summary = run_all(cfg, store, docs, _engine_factory(cfg), BatchApi, limit=args.limit,
                          interval=args.interval, out=print, progress=True)
        export_summary = export(cfg, store, docs, include_unverified=args.include_unverified,
                         copy_images=not args.no_copy_images, progress=True)
        print(f"Run finished. Steps: {summary}")
        print(f"Export: {export_summary}")
    elif args.cmd == "visual":
        rows = refresh_visual(cfg, store, docs, apply=args.apply, progress=True)
        _print_visual(cfg, rows)
    elif args.cmd == "reevaluate":
        result = reevaluate(cfg, store, docs)
        print(f"Reevaluated: {dict(result)}")
        if result.get("stage1_stage1_ok"):
            print("Documents that now pass Stage 1 are waiting for Stage 2. Run `labeller run` to label them.")
    elif args.cmd == "describe-submit":
        count = describe.submit(cfg.stage2.model, BatchApi(), [Path(p) for p in args.records], Path(args.out_dir))
        print(f"Submitted descriptions for {count} document types. Then run: labeller describe-collect")
    elif args.cmd == "describe-collect":
        print(describe.collect(BatchApi(), Path(args.out_dir)))
    elif args.cmd == "export":
        summary = export(cfg, store, docs, include_unverified=args.include_unverified,
                         copy_images=not args.no_copy_images, progress=True)
        print(summary)


def _engine_factory(cfg):
    def make_engine():
        print(f"Loading RapidOCR ({cfg.ocr.ocr_version}, {cfg.ocr.model_type}); first run downloads the models.")
        return RapidOcrEngine(model_type=cfg.ocr.model_type, ocr_version=cfg.ocr.ocr_version)

    return make_engine


def _print_plan(cfg, report: dict) -> None:
    print(f"Routing if prepared now: {report['counts']}")
    pages = report["stage1_pages"]
    chars = report["stage2_chars"]
    print(f"Local OCR:        {report['ocr_pages']} pages  (free, runs on this machine)")
    print(f"Stage 1 (vision): {pages} pages  ~ {pages * cfg.tokens_per_page_estimate:,} input tokens"
          f"{' (images attached)' if cfg.stage1.send_images else ' (text only)'}")
    print(f"Stage 2 (text):   {chars:,} chars  ~ {chars // 4:,} input tokens (+ prompt overhead)")
    if report.get("multimodal_docs"):
        extra = report["multimodal_pages"] * cfg.tokens_per_page_estimate
        print(f"Stage 2 images:   {report['multimodal_docs']} documents / {report['multimodal_pages']} pages go "
              f"multimodal (visual routing)  ~ {extra:,} more input tokens, plus larger outputs")
    print(f"Visual routing:   mode = {cfg.visual.mode}")
    print("Output tokens are capped per request by max_output_tokens. Prices are not hard-coded; "
          "check ai.google.dev/pricing for the models in config/annotation.yaml.")


def _print_visual(cfg, rows: list[dict]) -> None:
    print(f"Visual routing (mode: {cfg.visual.mode}). multimodal = Stage 2 gets the page images.")
    print(f"{'document':26}{'state':14}{'colour':>7}{'shade':>7}{'dark':>7}  route")
    for row in rows:
        s = row["signals"]
        route = "MULTIMODAL " + ",".join(row["reasons"]) if row["multimodal"] else "text only"
        print(f"{row['doc_id'][-24:]:26}{row['state']:14}{s.get('color', 0):>7.3f}{s.get('shaded', 0):>7.3f}"
              f"{s.get('blob', 0):>7.3f}  {route}")
    multimodal = sum(1 for row in rows if row["multimodal"])
    print(f"\n{multimodal} of {len(rows)} documents would go multimodal; {len(rows) - multimodal} text only.")
    if cfg.visual.mode != "auto":
        print(f"Note: visual.mode is '{cfg.visual.mode}', so this routing is not used until it is 'auto'.")


def _print_status(store: Store) -> None:
    statuses = store.all_statuses()
    states = Counter(st["state"] for st in statuses.values())
    reasons = Counter()
    for st in statuses.values():
        if st["state"] == "excluded":
            reasons[(st.get("reason") or "unknown").split(";")[0].split(":")[0]] += 1
    print(f"Documents tracked: {len(statuses)}")
    for state, count in sorted(states.items()):
        print(f"  {state}: {count}")
    if reasons:
        print("Top exclusion reasons:")
        for reason, count in reasons.most_common(10):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    sys.exit(main())
