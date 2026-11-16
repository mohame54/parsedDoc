"""prepare (local, free) -> submit (Gemini Batch) -> collect (validate, accept or exclude).

Every step reads and writes status.json, so any step can be re-run after a crash without
repeating finished work. Submitting is idempotent: the batch display name is derived from the
documents and attempt counts, and an existing non-failed job with that name is reattached.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from . import gemini, ocr, prompts, schemas
from .evidence import attach_boxes, build_line_index
from .language import detect_language, from_code, language_reason
from .visual import add_stage1_flags
from .visual import analyze as visual_analyze
from .progress import RunProgress, inner_bar
from .config import AnnotationConfig
from .processors import get_processor
from .processors.generic import pdf_text_pages
from .store import Store, read_json, utc_now, write_json
from .validate import (
    validate_fields,
    validate_fill,
    validate_stage1,
    validate_stage1_v2,
    validate_stage2,
    validate_stage2_v3,
    clean_descriptions,
)

Out = Callable[[str], None]


def load_manifest(cfg: AnnotationConfig) -> dict[str, dict]:
    if not cfg.manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {cfg.manifest_path}. Run the corpus builder first.")
    docs: dict[str, dict] = {}
    for line in cfg.manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            docs[row["document_id"]] = row
    return docs


def page_paths(doc: dict) -> list[str]:
    return doc.get("page_paths") or [doc["original_path"]]


def _excluded(reason: str) -> dict:
    return {"state": "excluded", "reason": reason}


def check_language(cfg: AnnotationConfig, text: str) -> tuple[dict | None, str | None]:
    """Detects the text's language and applies the language policy. Returns (result for the record, reason).
    With filtering off nothing is detected, so documents are never slowed or excluded by it."""
    if not cfg.language.enabled:
        return None, None
    result = detect_language(text, cfg.language)
    return result.to_dict(), language_reason(result, cfg.language)


def _blank_widgets(form_fill: dict, cfg: AnnotationConfig) -> bool:
    """Labeller-side blank-form rule (the corpus applies the same rule when it builds the manifest)."""
    widgets = int(form_fill.get("widgets", 0))
    if widgets < cfg.blank_form_min_widgets:
        return False
    return int(form_fill.get("filled", 0)) / widgets < cfg.blank_form_max_fill_ratio


def decide(doc: dict, cfg: AnnotationConfig) -> dict:
    """Pure decision for one document. Does not write anything."""
    root = cfg.data_root
    original = root / doc["original_path"]
    if not original.exists():
        return _excluded("missing_file")
    page_count = int(doc.get("page_count") or 1)
    if page_count > cfg.max_pages:
        return _excluded(f"too_many_pages:{page_count}>{cfg.max_pages}")
    if doc.get("file_type") == "pdf" and len(page_paths(doc)) < page_count:
        return _excluded("incomplete_render")
    form_fill = doc.get("form_fill")
    if cfg.exclude_blank_forms and form_fill and _blank_widgets(form_fill, cfg):
        return _excluded(f"blank_form:widgets={form_fill['widgets']},filled={form_fill['filled']}")

    labels = get_processor(doc.get("processor", "generic")).load(doc, root)
    if labels.route == "skip":
        return _excluded(labels.reason or "skipped_by_processor")

    if labels.route == "native_full":
        text = "\n".join(labels.text_pages or [])
        reasons = validate_fields(labels.fields or [], text, cfg.max_value_gap) + validate_fill(
            labels.fields or [], cfg.min_extracted_ratio, cfg.min_fields_for_ratio)
        if reasons:
            return _excluded("; ".join(reasons))
        language, reason = check_language(cfg, text)
        if reason:
            return _excluded(reason)
        return {
            "state": "accepted", "route": "native_full", "origin": "native_labels",
            "document_type": labels.document_type, "source_text": labels.text_pages, "fields": labels.fields,
            "language": language,
        }

    text_pages: list[str] | None = None
    origin = "gemini_stage1"
    route = "vision"
    if labels.route == "native_text" and labels.text_pages:
        text_pages, origin, route = labels.text_pages, "native_text", "native_text"
    elif doc.get("file_type") == "pdf":
        text_pages = pdf_text_pages(original, cfg.min_text_chars_per_page, cfg.max_pages)
        if text_pages:
            origin, route = "pdf_text_layer", "native_text"

    # Anything below sends document content to Gemini, so the PHI policy applies to all of it.
    if doc.get("phi_risk") and doc["source_id"] not in cfg.phi_allowed_sources:
        return _excluded("phi_gate_unauthorized_source")
    # Free local look at the pages: does Stage 2 need the images, or is the text enough?
    visual = (visual_analyze(doc, root, cfg.visual, cfg.max_pages) if cfg.visual.mode == "auto" else None)
    if text_pages:
        language, reason = check_language(cfg, "\n".join(text_pages))
        if reason:
            return _excluded(reason)
        return {"state": "needs_stage2", "route": route, "origin": origin,
                "document_type": labels.document_type, "source_text": text_pages,
                "language": language, "visual": visual}
    state = "needs_ocr" if cfg.ocr.enabled else "needs_stage1"
    return {"state": state, "route": "vision", "origin": "gemini_stage1", "source_text": None, "visual": visual}


def make_annotation(doc: dict, document_type: str, fields: list[dict], origin: str, provenance: dict,
                    components: dict[str, list] | None = None, language: dict | None = None,
                    visual_routing: dict | None = None) -> dict:
    """components holds the Stage 2 v3 parts (tables, visual_elements, layout, relationships). They stay
    separate from fields so each can train a different kind of model."""
    items = sorted((f["section"], f["name"], f["data_type"]) for f in fields)
    annotation = {
        "document_id": doc["document_id"],
        "source_id": doc["source_id"],
        "document_type": document_type,
        "family_id": re.sub(r"[^a-z0-9]+", "_", (document_type or "unknown").lower()).strip("_"),
        "schema_fingerprint": hashlib.sha256(json.dumps(items).encode()).hexdigest()[:16],
        "language": language,
        "visual_routing": visual_routing,
        "fields": fields,
        "origin": origin,
        "provenance": provenance,
    }
    annotation.update(components or {})
    return annotation


def prepare(cfg: AnnotationConfig, store: Store, docs: dict[str, dict], dry_run: bool = False) -> dict[str, Any]:
    """Routes every undecided document. Returns counts and the cost drivers for `plan`."""
    counts: Counter[str] = Counter()
    stage1_pages = 0
    ocr_pages = 0
    stage2_chars = 0
    multimodal_docs = 0
    multimodal_pages = 0
    for doc_id, doc in docs.items():
        if store.status(doc_id) is not None:
            counts["already_decided"] += 1
            continue
        decision = decide(doc, cfg)
        counts[decision["state"]] += 1
        if dry_run:
            if decision["state"] == "needs_stage1":
                stage1_pages += len(page_paths(doc))
            elif decision["state"] == "needs_ocr":
                ocr_pages += len(page_paths(doc))
            elif decision["state"] == "needs_stage2":
                stage2_chars += sum(len(p) for p in decision["source_text"])
            if decision["state"] != "excluded" and stage2_uses_images(cfg, {"visual": decision.get("visual")}):
                multimodal_docs += 1
                multimodal_pages += len(page_paths(doc))
            continue

        base = {"page_count": int(doc.get("page_count") or 1), "attempts": {"stage1": 0, "stage2": 0}, "jobs": {}}
        if decision["state"] == "excluded":
            store.set_status(doc_id, "excluded", reason=decision["reason"], **base)
        elif decision["state"] == "accepted":
            annotation = make_annotation(doc, decision["document_type"], decision["fields"], decision["origin"],
                                         {"route": decision["route"]}, language=decision.get("language"))
            write_json(store.ann_dir(doc_id) / "annotation.json", annotation)
            store.set_status(doc_id, "accepted", route=decision["route"], origin=decision["origin"],
                             language=decision.get("language"), **base)
        else:
            if decision["source_text"] is not None:
                write_json(store.ann_dir(doc_id) / "source_text.json",
                           {"origin": decision["origin"], "pages": decision["source_text"]})
            store.set_status(doc_id, decision["state"], route=decision["route"], origin=decision["origin"],
                             document_type=decision.get("document_type"), language=decision.get("language"),
                             visual=decision.get("visual"), **base)
    return {"counts": dict(counts), "stage1_pages": stage1_pages, "ocr_pages": ocr_pages,
            "stage2_chars": stage2_chars, "multimodal_docs": multimodal_docs,
            "multimodal_pages": multimodal_pages}


def refresh_visual(cfg: AnnotationConfig, store: Store, docs: dict[str, dict],
                   apply: bool = False) -> list[dict]:
    """Visual routing for every tracked document that is not excluded. With apply=True the result is saved
    for documents that have not reached Stage 2 yet (accepted ones keep the routing they were labelled with)."""
    rows = []
    for doc_id, status in sorted(store.all_statuses().items()):
        doc = docs.get(doc_id)
        if doc is None or status["state"] == "excluded":
            continue
        result = visual_analyze(doc, cfg.data_root, cfg.visual, cfg.max_pages)
        if status.get("visual") and "stage1:" in ",".join(status["visual"].get("reasons") or []):
            result = add_stage1_flags(result, read_json(store.ann_dir(doc_id) / "stage1.json") or {}, cfg.visual)
        editable = status["state"] in ("needs_ocr", "needs_stage1", "needs_stage2")
        if apply and editable:
            store.set_status(doc_id, status["state"], visual=result)
        rows.append({"doc_id": doc_id, "state": status["state"], "saved": bool(apply and editable), **result})
    return rows


def ocr_step(cfg: AnnotationConfig, store: Store, engine_factory: Callable[[], Any], docs: dict[str, dict],
             out: Out = print, progress: bool = False) -> Counter[str]:
    """Runs local OCR for every needs_ocr document (free). The engine is built only if there is work."""
    outcome: Counter[str] = Counter()
    pending = sorted((s for s in store.all_statuses().values() if s["state"] == "needs_ocr"),
                     key=lambda s: s["doc_id"])
    if not pending:
        return outcome
    engine = engine_factory()
    doc_bar = inner_bar(len(pending), "OCR documents", position=1) if progress else None
    for status in pending:
        doc = docs.get(status["doc_id"])
        if doc is None:
            outcome["missing_doc"] += 1
            if doc_bar: doc_bar.update(1)
            continue
        page_bar = inner_bar(len(page_paths(doc)), f"  pages of {doc['document_id'][-16:]}", position=2, unit="page") if progress else None
        try:
            reason = ocr.run_ocr(store, engine, doc, cfg.ocr, on_page=page_bar.update if page_bar else None)
        except Exception as exc:  # left in needs_ocr, so the next run retries it
            _say(out, progress, f"[error] {doc['document_id']}: ocr failed: {exc}")
            outcome["ocr_error"] += 1
            if page_bar: page_bar.close()
            if doc_bar: doc_bar.update(1)
            continue
        if page_bar: page_bar.close()
        if doc_bar: doc_bar.update(1)
        language = None
        if not reason:
            cached = read_json(store.ann_dir(doc["document_id"]) / "ocr.json")
            text = "\n".join(line["text"] for page in cached["pages"] for line in page["lines"])
            language, reason = check_language(cfg, text)
        if reason:
            store.set_status(doc["document_id"], "excluded", reason=reason, language=language)
            _say(out, progress, f"  [skip] {doc['document_id']}  {reason}")
            outcome["excluded"] += 1
        else:
            store.set_status(doc["document_id"], "needs_stage1", language=language)
            outcome["ocr_ok"] += 1
    if doc_bar:
        doc_bar.close()
    return outcome


def _runkey(stage: int, model: str, prompt_version: str, attempts: dict[str, int]) -> str:
    payload = {"stage": stage, "model": model, "prompt": prompt_version, "attempts": sorted(attempts.items())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def request_key(doc_id: str, stage: int, prompt_version: str) -> str:
    return f"{doc_id}:stage{stage}:{prompt_version}"


def _join_pages(pages: list[str]) -> str:
    return "\n\n".join(f"--- page {i} ---\n{text}" for i, text in enumerate(pages, start=1))


def _draft_text(ocr_result: dict) -> str:
    lines = []
    for page in ocr_result["pages"]:
        lines.append(f"Page {page['page']}:")
        lines += [f"{line['id']}: {line['text']}" for line in page["lines"]]
    return "\n".join(lines)


def stage2_uses_images(cfg: AnnotationConfig, status: dict) -> bool:
    """Does this document's Stage 2 get the page images (multimodal) or only the text?

    stage2.send_images: true or visual.mode: always -> every document. visual.mode: auto -> the documents the
    visual analysis marked multimodal. Otherwise text only.
    """
    if cfg.stage2.send_images or cfg.visual.mode == "always":
        return True
    if cfg.visual.mode == "auto":
        return bool((status.get("visual") or {}).get("multimodal"))
    return False


def prompt_version_for(stage: int, cfg: AnnotationConfig, status: dict) -> str:
    """Stage 1 uses the configured prompt. Stage 2 uses the multimodal prompt when images are sent, the
    OCR-aware prompt when the text came from OCR, and the configured prompt otherwise."""
    if stage == 1:
        return cfg.stage1.prompt_version
    if stage2_uses_images(cfg, status):
        return prompts.MULTIMODAL_STAGE2_PROMPT
    if status.get("origin") == "gemini_stage1":
        return prompts.OCR_STAGE2_PROMPT
    return cfg.stage2.prompt_version


def _version_used(stage: int, cfg: AnnotationConfig, status: dict) -> str:
    """The prompt version a document's batch actually ran, as recorded when it was submitted. Falls back to
    what the config would choose now, for documents submitted before that was recorded."""
    recorded = (status.get("models") or {}).get(f"stage{stage}") or {}
    return recorded.get("prompt_version") or prompt_version_for(stage, cfg, status)


def _transcription(source: dict) -> str:
    """The text Stage 2 reads. Lines carry their ids when known, so Gemini can cite them as evidence."""
    lines = source.get("lines")
    if not lines:
        return _join_pages(source["pages"])
    return "\n\n".join(
        f"--- page {number} ---\n" + "\n".join(f"{line['id']}: {line['text']}" for line in page)
        for number, page in enumerate(lines, start=1)
    )


def _line_ids(source: dict) -> set[str] | None:
    lines = source.get("lines")
    return {line["id"] for page in lines for line in page} if lines else None


def _build_line(stage: int, cfg: AnnotationConfig, store: Store, api: Any, doc: dict, version: str,
                on_page: Callable[[], None] | None = None) -> dict:
    scfg = cfg.stage1 if stage == 1 else cfg.stage2
    doc_id = doc["document_id"]
    key = request_key(doc_id, stage, version)
    prompt = prompts.get_prompt(version)
    if stage == 1 and version in prompts.OCR_STAGE1_PROMPTS:
        ocr_result = read_json(store.ann_dir(doc_id) / "ocr.json")
        if ocr_result is None:
            raise FileNotFoundError("ocr.json missing: run the ocr step first")
        parts: list[dict] = [gemini.text_part(prompt + "\n\nOCR DRAFT:\n" + _draft_text(ocr_result))]
        if scfg.send_images:
            scratch = store.ann_dir(doc_id) / "input"
            for index, rel in enumerate(page_paths(doc), start=1):
                path, mime = gemini.prepare_image(store.resolve(rel), scratch)
                uri = api.upload_file(path, mime)
                parts += [gemini.text_part(f"Page {index} image:"), gemini.file_part(uri, mime)]
                if on_page: on_page()
        schema = schemas.STAGE1_V2_RESPONSE_SCHEMA
    elif stage == 1:
        parts = [gemini.text_part(prompt)]
        scratch = store.ann_dir(doc_id) / "input"
        for index, rel in enumerate(page_paths(doc), start=1):
            path, mime = gemini.prepare_image(store.resolve(rel), scratch)
            uri = api.upload_file(path, mime)
            parts += [gemini.text_part(f"Page {index}:"), gemini.file_part(uri, mime)]
            if on_page: on_page()
        schema = schemas.STAGE1_RESPONSE_SCHEMA
    else:
        source = read_json(store.ann_dir(doc_id) / "source_text.json")
        if source is None:
            raise FileNotFoundError("source_text.json missing")
        parts = [gemini.text_part(prompt + "\n\nTRANSCRIPTION:\n" + _transcription(source))]
        multimodal = version in prompts.MULTIMODAL_STAGE2_PROMPTS
        if multimodal:
            scratch = store.ann_dir(doc_id) / "input"
            for index, rel in enumerate(page_paths(doc), start=1):
                path, mime = gemini.prepare_image(store.resolve(rel), scratch)
                uri = api.upload_file(path, mime)
                parts += [gemini.text_part(f"Page {index} image:"), gemini.file_part(uri, mime)]
                if on_page: on_page()
        schema = schemas.stage2_schema(multimodal, version in prompts.DESCRIBED_STAGE2_PROMPTS)
    return gemini.build_request_line(key, parts, schema, scfg.max_output_tokens)


def submit(cfg: AnnotationConfig, store: Store, api: Any, stage: int, docs: dict[str, dict],
           limit: int | None = None, out: Out = print, progress: bool = False) -> int:
    scfg = cfg.stage1 if stage == 1 else cfg.stage2
    pending = sorted((s for s in store.all_statuses().values() if s["state"] == f"needs_stage{stage}"),
                     key=lambda s: s["doc_id"])
    if limit is not None:
        pending = pending[:limit]
    # One batch holds one prompt version, so documents are grouped by the prompt they need.
    by_version: dict[str, list[dict]] = {}
    for status in pending:
        by_version.setdefault(prompt_version_for(stage, cfg, status), []).append(status)
    groups = [(version, group[start:start + cfg.batch_size])
              for version, group in sorted(by_version.items())
              for start in range(0, len(group), cfg.batch_size)]
    submitted = 0
    for version, chunk in groups:
        attempts = {s["doc_id"]: int(s.get("attempts", {}).get(f"stage{stage}", 0)) for s in chunk}
        runkey = _runkey(stage, scfg.model, version, attempts)
        bdir = store.batch_dir(stage, runkey)
        job_path = bdir / "job.json"
        existing = read_json(job_path)
        if existing and existing.get("job_name") and not existing.get("closed"):
            out(f"reattached {bdir.name} ({len(existing['doc_ids'])} docs)")
            _mark_awaiting(store, stage, existing)
            continue

        bdir.mkdir(parents=True, exist_ok=True)
        lines, built = [], []
        # Stage 1 uploads every page, so it gets a page bar nested under the document bar.
        page_total = sum(len(page_paths(docs[s["doc_id"]])) for s in chunk
                         if stage == 1 and s["doc_id"] in docs) if progress else 0
        doc_bar = inner_bar(len(chunk), f"Stage {stage} prepare {bdir.name}", position=1) if progress else None
        page_bar = inner_bar(page_total, "  uploading pages", position=2, unit="page") if progress and stage == 1 else None
        for status in chunk:
            doc = docs.get(status["doc_id"])
            if doc is None:
                _say(out, progress, f"[skip] {status['doc_id']}: not in manifest")
                if doc_bar: doc_bar.update(1)
                continue
            try:
                lines.append(_build_line(stage, cfg, store, api, doc, version,
                                         on_page=page_bar.update if page_bar else None))
                built.append(doc["document_id"])
            except Exception as exc:  # one bad document must not stop the chunk
                _say(out, progress, f"[skip] {doc['document_id']}: {exc}")
            if doc_bar: doc_bar.update(1)
        if page_bar: page_bar.close()
        if doc_bar: doc_bar.close()
        if not lines:
            continue

        gemini.write_jsonl(bdir / "input.jsonl", lines)
        src_name = api.upload_jsonl(bdir / "input.jsonl")
        job_name = api.find_or_create(display_name=f"labeller-{runkey}", model=scfg.model, src_name=src_name)
        job = {
            "job_name": job_name, "stage": stage, "model": scfg.model, "prompt_version": version,
            "runkey": runkey, "doc_ids": built, "attempts": attempts, "closed": False, "submitted_at": utc_now(),
        }
        write_json(job_path, job)
        _mark_awaiting(store, stage, job)
        out(f"submitted {bdir.name}: {len(built)} requests -> {job_name}")
        submitted += len(built)
    return submitted


def _mark_awaiting(store: Store, stage: int, job: dict) -> None:
    for doc_id in job["doc_ids"]:
        st = store.status(doc_id) or {}
        jobs = dict(st.get("jobs") or {})
        jobs[f"stage{stage}"] = job["job_name"]
        # Record what actually runs, so provenance never depends on the config as it is later.
        models = dict(st.get("models") or {})
        models[f"stage{stage}"] = {"model": job["model"], "prompt_version": job["prompt_version"]}
        store.set_status(doc_id, f"awaiting_stage{stage}", jobs=jobs, models=models)


def _say(out: Out, progress: bool, message: str) -> None:
    """Print without breaking an active tqdm bar."""
    if progress:
        tqdm.write(message)
    else:
        out(message)


def _elapsed(submitted_at: str | None) -> str:
    if not submitted_at:
        return "?"
    seconds = int((datetime.now(timezone.utc) - datetime.fromisoformat(submitted_at)).total_seconds())
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _batch_line(name: str, info: dict, job: dict) -> str:
    total = len(job["doc_ids"])
    done = [info.get("succeeded"), info.get("failed")]
    if any(value is not None for value in done):
        finished = sum(value or 0 for value in done)
        counts = f"{finished}/{total} requests done ({info.get('failed') or 0} failed)"
    else:
        counts = f"{total} requests (Gemini reports no counters yet)"
    return f"{name}: {info['state']}  {counts}  elapsed {_elapsed(job.get('submitted_at'))}"


def _doc_line(store: Store, doc_id: str, outcome: str) -> str:
    st = store.status(doc_id) or {}
    reason = st.get("reason") or st.get("last_error") or ""
    mark = {"accepted": "ok  ", "stage1_ok": "ok  ", "excluded": "skip", "retry": "retry"}.get(outcome, outcome)
    return f"  [{mark}] {doc_id}" + (f"  {reason}" if reason and outcome in ("excluded", "retry") else "")


def collect(cfg: AnnotationConfig, store: Store, api: Any, stage: int, docs: dict[str, dict],
            out: Out = print, progress: bool = False, bars: dict[str, tqdm] | None = None) -> Counter[str]:
    """Polls every open batch of this stage and processes the finished ones.

    With progress=True each batch gets a live bar (documents done, state, elapsed). The bars dict is
    shared across polls, so the same bar updates in place instead of printing a new line each time.
    """
    outcome: Counter[str] = Counter()
    bars = bars if bars is not None else {}
    for bdir in store.batch_dirs(stage):
        job_path = bdir / "job.json"
        job = read_json(job_path)
        if not job or job.get("closed") or not job.get("job_name"):
            continue
        info = api.progress(job["job_name"]) if progress else {"state": api.state(job["job_name"])}
        state = info["state"]
        total = len(job["doc_ids"])
        bar = None
        if progress:
            bar = bars.get(bdir.name)
            if bar is None:
                bar = tqdm(total=total, desc=f"Stage {stage} batch {bdir.name}", position=1, leave=True, unit="doc")
                bars[bdir.name] = bar
            finished = [v for v in (info.get("succeeded"), info.get("failed")) if v is not None]
            if finished:
                bar.n = min(sum(finished), total)
            failed = info.get("failed") or 0
            label = state.replace("JOB_STATE_", "").lower()
            bar.set_postfix_str(f"{label} · {_elapsed(job.get('submitted_at'))}" + (f" · {failed} failed" if failed else ""))
            bar.refresh()
        if state not in gemini.TERMINAL_STATES:
            if not progress:
                out(f"{bdir.name}: {state}")
            outcome["still_running"] += 1
            continue
        if not progress:
            out(f"{bdir.name}: {state}")
        if state == gemini.SUCCEEDED:
            output_path = bdir / "output.jsonl"
            if not output_path.exists():
                output_path.write_text(api.download_output(job["job_name"]), encoding="utf-8")
            results = gemini.parse_output_text(output_path.read_text(encoding="utf-8"))
            for doc_id in job["doc_ids"]:
                result = _process_result(cfg, store, stage, doc_id, results, docs)
                outcome[result] += 1
                if progress:
                    out(_doc_line(store, doc_id, result))
                    bar.update(1)
        else:
            for doc_id in job["doc_ids"]:
                result = _fail(cfg, store, stage, doc_id, f"batch_{state.lower()}")
                outcome[result] += 1
                if progress:
                    out(_doc_line(store, doc_id, result))
                    bar.update(1)
        if bar is not None:
            bar.n = total
            bar.set_postfix_str(f"{state.replace('JOB_STATE_', '').lower()} · done")
            bar.close()
            bars.pop(bdir.name, None)
        job["closed"] = True
        job["final_state"] = state
        write_json(job_path, job)
    return outcome


def watch(cfg: AnnotationConfig, store: Store, api: Any, stage: int, docs: dict[str, dict],
          interval: int = 30, out: Out = print, progress: bool = False) -> Counter[str]:
    """Polls until every open batch for this stage has finished. Safe to interrupt and re-run."""
    total: Counter[str] = Counter()
    bars: dict[str, tqdm] = {}
    try:
        while True:
            outcome = collect(cfg, store, api, stage, docs, out=out, progress=progress, bars=bars)
            running = outcome.pop("still_running", 0)
            total.update(outcome)
            if not running:
                return total
            if not progress:
                out(f"  ... {running} batch(es) still running, next check in {interval}s (Ctrl+C is safe)")
            time.sleep(interval)
    except KeyboardInterrupt:
        for bar in bars.values():
            bar.close()
        out("Stopped watching. Jobs keep running on Gemini; re-run collect to resume.")
        return total


def run_all(cfg: AnnotationConfig, store: Store, docs: dict[str, dict], engine_factory: Callable[[], Any],
            api_factory: Callable[[], Any], limit: int | None = None, interval: int = 30,
            out: Out = print, progress: bool = False) -> dict[str, Any]:
    """Runs prepare, OCR, Stage 1 and Stage 2 end to end, with a run bar showing the current step.

    Export is left to the caller. The Gemini client is created only when the first submit runs,
    so a run with nothing to send never needs the API key.
    """
    say = tqdm.write if progress else out
    run = RunProgress(progress)
    summary: dict[str, Any] = {}
    api_holder: dict[str, Any] = {}

    def api() -> Any:
        if "api" not in api_holder:
            api_holder["api"] = api_factory()
        return api_holder["api"]

    def step(name: str, fn: Callable[[], Any]) -> None:
        run.begin(name)
        summary[name] = fn()
        run.done()

    try:
        step("prepare", lambda: prepare(cfg, store, docs)["counts"])
        step("OCR", lambda: dict(ocr_step(cfg, store, engine_factory, docs, out=say, progress=progress)))
        step("Stage 1 submit", lambda: submit(cfg, store, api(), 1, docs, limit=limit, out=say, progress=progress))
        step("Stage 1 wait", lambda: dict(watch(cfg, store, api(), 1, docs, interval=interval, out=say, progress=progress)))
        step("Stage 2 submit", lambda: submit(cfg, store, api(), 2, docs, limit=limit, out=say, progress=progress))
        step("Stage 2 wait", lambda: dict(watch(cfg, store, api(), 2, docs, interval=interval, out=say, progress=progress)))
    finally:
        run.close()
    return summary


def _process_result(cfg: AnnotationConfig, store: Store, stage: int, doc_id: str,
                    results: dict[str, tuple[str | None, str | None]], docs: dict[str, dict]) -> str:
    st = store.status(doc_id)
    if not st or st["state"] != f"awaiting_stage{stage}":
        return "skipped"
    scfg = cfg.stage1 if stage == 1 else cfg.stage2
    version = _version_used(stage, cfg, st)
    key = request_key(doc_id, stage, version)
    if key not in results:
        return _fail(cfg, store, stage, doc_id, "missing_result")
    text, error = results[key]
    if error or text is None:
        return _fail(cfg, store, stage, doc_id, f"gemini_error:{error}")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return _fail(cfg, store, stage, doc_id, "invalid_json")

    ann_dir = store.ann_dir(doc_id)
    if stage == 1 and version in prompts.OCR_STAGE1_PROMPTS:
        ocr_result = read_json(ann_dir / "ocr.json")
        if ocr_result is None:
            return _fail(cfg, store, stage, doc_id, "ocr_missing")
        reasons, page_texts, metrics, lines = validate_stage1_v2(
            obj, ocr_result["pages"], int(st.get("page_count") or 1), cfg.max_illegible_regions,
            cfg.exclude_blank_forms, cfg.max_change_ratio,
        )
        if reasons:
            store.set_status(doc_id, "excluded", reason="; ".join(reasons), **_round_metrics(metrics))
            return "excluded"
        # Gemini's own language code is a second opinion. An unusable code is ignored: the local
        # detection already applied the unknown-language policy.
        language = st.get("language")
        if cfg.language.enabled:
            said = from_code(obj.get("language"))
            if said.primary is not None:
                reason = language_reason(said, cfg.language)
                if reason:
                    store.set_status(doc_id, "excluded", reason=f"{reason} (stage1)", language=said.to_dict())
                    return "excluded"
                if not language or language.get("primary") is None:
                    language = said.to_dict()
        visual = st.get("visual")
        if cfg.visual.mode == "auto":
            visual = add_stage1_flags(visual, obj, cfg.visual)   # Stage 1's layout flags can only add reasons
        write_json(ann_dir / "stage1.json", obj)
        write_json(ann_dir / "source_text.json",
                   {"origin": "gemini_stage1", "pages": page_texts, "lines": lines, **_round_metrics(metrics)})
        store.set_status(doc_id, "needs_stage2", origin="gemini_stage1", document_type=obj["document_type"],
                         reason=None, language=language, visual=visual, **_round_metrics(metrics))
        return "stage1_ok"
    if stage == 1:
        reasons = validate_stage1(obj, int(st.get("page_count") or 1), cfg.max_illegible_regions)
        if reasons:
            store.set_status(doc_id, "excluded", reason="; ".join(reasons))
            return "excluded"
        write_json(ann_dir / "stage1.json", obj)
        write_json(ann_dir / "source_text.json",
                   {"origin": "gemini_stage1", "pages": [p["text"] for p in obj["pages"]]})
        store.set_status(doc_id, "needs_stage2", origin="gemini_stage1", document_type=obj["document_type"])
        return "stage1_ok"

    source = read_json(ann_dir / "source_text.json")
    if source is None:
        return _fail(cfg, store, stage, doc_id, "source_text_missing")
    source_text = "\n".join(source["pages"])
    multimodal = version in prompts.MULTIMODAL_STAGE2_PROMPTS
    described = version in prompts.DESCRIBED_STAGE2_PROMPTS
    if multimodal:
        reasons = validate_stage2_v3(obj, source_text, _line_ids(source), int(st.get("page_count") or 1),
                                     cfg.min_extracted_ratio, cfg.min_fields_for_ratio, cfg.max_value_gap,
                                     described=described)
    else:
        reasons = validate_stage2(obj, source_text, cfg.min_extracted_ratio, cfg.min_fields_for_ratio,
                                  cfg.max_value_gap, described=described)
    if reasons:
        store.set_status(doc_id, "excluded", reason="; ".join(reasons))
        return "excluded"
    if described:
        clean_descriptions(obj["fields"])   # untrustworthy descriptions are blanked; templates cover them
    doc = docs[doc_id]
    provenance = _provenance(cfg, st, version)
    components = None
    if multimodal:
        # Spatial grounding: the OCR's own boxes for the lines Gemini cited. Gemini produces no coordinates.
        ocr_result = read_json(ann_dir / "ocr.json")
        index = build_line_index(ocr_result["pages"]) if ocr_result else {}
        if _line_ids(source) is None:
            _clear_evidence(obj)  # no line ids were given, so any evidence would be invented
        attach_boxes(obj, index)
        components = {key: obj[key] for key in ("tables", "visual_elements", "layout", "relationships")}
    annotation = make_annotation(doc, obj["document_type"], obj["fields"], st.get("origin", "gemini_stage1"),
                                 provenance, components, language=st.get("language"),
                                 visual_routing=st.get("visual"))
    write_json(ann_dir / "stage2.json", obj)
    write_json(ann_dir / "annotation.json", annotation)
    store.set_status(doc_id, "accepted", document_type=obj["document_type"], reason=None)
    return "accepted"


def _clear_evidence(obj: dict) -> None:
    for key in ("fields", "tables", "visual_elements", "layout"):
        for item in obj[key]:
            item["evidence"] = []


def _provenance(cfg: AnnotationConfig, status: dict, stage2_version: str) -> dict:
    """Which models and prompts produced a label. Uses what the batch jobs actually ran, as recorded in
    status.json; falls back to the current config only for documents labelled before that was recorded."""
    models = status.get("models") or {}
    jobs = status.get("jobs") or {}
    stage1 = models.get("stage1") or {"model": cfg.stage1.model, "prompt_version": cfg.stage1.prompt_version}
    stage2 = models.get("stage2") or {"model": cfg.stage2.model, "prompt_version": stage2_version}
    return {
        "stage1": {**stage1, "job": jobs.get("stage1")} if status.get("origin") == "gemini_stage1" else None,
        "stage2": {**stage2, "job": jobs.get("stage2")},
    }


def repair_provenance(store: Store, out: Out = print) -> int:
    """Fixes annotations labelled before models were recorded: reads the model each batch job really ran
    from data/batches/*/job.json. Returns the number of annotations corrected."""
    job_info: dict[str, dict] = {}
    for stage in (1, 2):
        for bdir in store.batch_dirs(stage):
            job = read_json(bdir / "job.json")
            if job and job.get("job_name"):
                job_info[job["job_name"]] = {"model": job["model"], "prompt_version": job["prompt_version"]}
    fixed = 0
    for doc_id, status in store.all_statuses().items():
        jobs = status.get("jobs") or {}
        recorded = {f"stage{n}": job_info[jobs[f"stage{n}"]] for n in (1, 2) if jobs.get(f"stage{n}") in job_info}
        if not recorded:
            continue
        if status.get("models") != recorded:
            store.set_status(doc_id, status["state"], models=recorded)
        path = store.ann_dir(doc_id) / "annotation.json"
        annotation = read_json(path)
        if annotation is None:
            continue
        prov = annotation.get("provenance") or {}
        changed = False
        for key in ("stage1", "stage2"):
            info = recorded.get(key)
            if info and prov.get(key) is not None:
                for field in ("model", "prompt_version"):
                    if prov[key].get(field) != info[field]:
                        out(f"  provenance {doc_id[-6:]} {key}.{field}: {prov[key].get(field)} -> {info[field]}")
                        prov[key][field] = info[field]
                        changed = True
        if changed:
            annotation["provenance"] = prov
            write_json(path, annotation)
            fixed += 1
    return fixed


def _round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {name: round(value, 4) for name, value in metrics.items()}


# Exclusion reasons that depend on settings in config/annotation.yaml, so they can change when the
# settings do. Other reasons (unusable, unreadable, uncertain, ...) come from the model and stay as they are.
STAGE1_SETTING_REASONS = ("too_many_corrections", "too_many_changes", "illegible_regions", "blank_form:stage1")
STAGE2_SETTING_REASONS = ("not_in_source", "table_not_in_source", "blank_form:extracted")


def _saved_results(store: Store, stage: int) -> dict[str, tuple[str | None, str | None]]:
    results: dict[str, tuple[str | None, str | None]] = {}
    for bdir in store.batch_dirs(stage):
        path = bdir / "output.jsonl"
        if path.exists():
            results.update(gemini.parse_output_text(path.read_text(encoding="utf-8")))
    return results


def reevaluate(cfg: AnnotationConfig, store: Store, docs: dict[str, dict], out: Out = print) -> Counter[str]:
    """Re-runs the gates on Gemini outputs already saved in data/batches/. Makes no API calls.

    Use it after changing a gate in config/annotation.yaml. A gate change can loosen or tighten, so
    every document is held to the current gates of both stages:
      - excluded by a setting-dependent gate: re-checked, and may now pass;
      - past Stage 1 (needs_stage2, accepted, or excluded at Stage 2): re-checked against the Stage 1
        gate and excluded if it no longer passes, never moved forward by this step;
      - accepted: re-checked against the Stage 2 gates too.
    Documents that now pass Stage 1 wait in needs_stage2 (run `labeller run` to label them).
    """
    outcome: Counter[str] = Counter()
    repaired = repair_provenance(store, out)
    if repaired:
        outcome["provenance_repaired"] = repaired
    _recheck_stage1(cfg, store, docs, _saved_results(store, 1), out, outcome)
    _recheck_stage2(cfg, store, docs, _saved_results(store, 2), out, outcome)
    return outcome


def _recheck_stage1(cfg: AnnotationConfig, store: Store, docs: dict[str, dict],
                    results: dict[str, tuple[str | None, str | None]], out: Out, outcome: Counter[str]) -> None:
    for doc_id, status in sorted(store.all_statuses().items()):
        if status.get("origin") != "gemini_stage1" or doc_id not in docs:
            continue  # documents labelled from native text never went through Stage 1
        state, reason = status["state"], status.get("reason") or ""
        stage1_excluded = state == "excluded" and reason.startswith(STAGE1_SETTING_REASONS)
        past_stage1 = state in ("needs_stage2", "accepted") or (
            state == "excluded" and reason.startswith(STAGE2_SETTING_REASONS))
        if not (stage1_excluded or past_stage1):
            continue
        key = request_key(doc_id, 1, _version_used(1, cfg, status))
        if key not in results or results[key][0] is None:
            outcome["stage1_no_saved_result"] += 1
            continue
        if stage1_excluded:
            # Reuse the normal result handling by putting the document back in its awaiting state.
            store.set_status(doc_id, "awaiting_stage1", reason=None)
            result = _process_result(cfg, store, 1, doc_id, results, docs)
            outcome[f"stage1_{result}"] += 1
            out(f"  stage 1: {_doc_line(store, doc_id, result).strip()}")
            continue
        ocr_result = read_json(store.ann_dir(doc_id) / "ocr.json")
        try:
            obj = json.loads(results[key][0])
        except json.JSONDecodeError:
            continue
        if ocr_result is None:
            continue
        reasons, _, metrics, _ = validate_stage1_v2(
            obj, ocr_result["pages"], int(status.get("page_count") or 1), cfg.max_illegible_regions,
            cfg.exclude_blank_forms, cfg.max_change_ratio)
        if reasons:
            store.set_status(doc_id, "excluded", reason="; ".join(reasons), **_round_metrics(metrics))
            outcome["stage1_now_excluded"] += 1
            out(f"  stage 1: [skip] {doc_id}  {'; '.join(reasons)}  (passed the earlier gate)")


def _recheck_stage2(cfg: AnnotationConfig, store: Store, docs: dict[str, dict],
                    results: dict[str, tuple[str | None, str | None]], out: Out, outcome: Counter[str]) -> None:
    for doc_id, status in sorted(store.all_statuses().items()):
        state, reason = status["state"], status.get("reason") or ""
        if not (state == "accepted" or (state == "excluded" and reason.startswith(STAGE2_SETTING_REASONS))):
            continue
        key = request_key(doc_id, 2, _version_used(2, cfg, status))
        if key not in results or doc_id not in docs:
            if state == "excluded":
                outcome["stage2_no_saved_result"] += 1
            continue
        before = state
        store.set_status(doc_id, "awaiting_stage2", reason=None)
        result = _process_result(cfg, store, 2, doc_id, results, docs)
        if before == "accepted" and result == "accepted":
            continue  # unchanged
        outcome[f"stage2_{result}"] += 1
        out(f"  stage 2: {_doc_line(store, doc_id, result).strip()}")


def _fail(cfg: AnnotationConfig, store: Store, stage: int, doc_id: str, reason: str) -> str:
    st = store.status(doc_id) or {}
    attempts = dict(st.get("attempts") or {})
    count = int(attempts.get(f"stage{stage}", 0)) + 1
    attempts[f"stage{stage}"] = count
    if count >= cfg.max_attempts:
        store.set_status(doc_id, "excluded", reason=f"{reason}; attempts={count}", attempts=attempts)
        return "excluded"
    store.set_status(doc_id, f"needs_stage{stage}", attempts=attempts, last_error=reason)
    return "retry"
