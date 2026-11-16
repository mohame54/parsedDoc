"""End-to-end pipeline against a fake Batch API. No network, no billing."""
import json
from pathlib import Path

import pytest
from PIL import Image

from labeller import gemini
from labeller.config import load_config
from labeller.export import export
from labeller.gemini import BatchApi
from labeller.pipeline import collect, load_manifest, prepare, request_key, submit
from labeller.store import Store

CONFIG = """
data_root: ./data
max_pages: 4
min_text_chars_per_page: 80
batch_size: 500
max_attempts: 2
tokens_per_page_estimate: 560
stage1: {model: test-flash-lite, prompt_version: stage1_v1, max_output_tokens: 4096}
stage2: {model: test-flash, prompt_version: stage2_v1, max_output_tokens: 4096}
gates: {max_illegible_regions: 0}
phi_allowed_sources: []
export: {train: 0.8, val: 0.1, seed: 42}
"""


class FakeApi(BatchApi):
    """Implements the BatchApi surface in memory. `responder(request)` returns a Gemini response dict."""

    def __init__(self, responder):
        self.client = None
        self.responder = responder
        self.inputs: dict[str, str] = {}
        self.jobs: dict[str, dict] = {}
        self.created = 0

    def upload_file(self, path, mime_type):
        return f"files/{path.name}"

    def upload_jsonl(self, path):
        name = f"files/input-{len(self.inputs)}"
        self.inputs[name] = path.read_text(encoding="utf-8")
        return name

    def find_job(self, display_name):
        for name, job in self.jobs.items():
            if job["display"] == display_name and job["state"] not in gemini.FAILED_STATES:
                return name
        return None

    def create_job(self, model, src_name, display_name):
        self.created += 1
        name = f"batches/{self.created}"
        self.jobs[name] = {"display": display_name, "src": src_name, "state": gemini.SUCCEEDED}
        return name

    def state(self, job_name):
        return self.jobs[job_name]["state"]

    def progress(self, job_name):
        job = self.jobs[job_name]
        done = job["state"] == gemini.SUCCEEDED
        return {"state": job["state"], "succeeded": 1 if done else 0, "failed": 0, "incomplete": 0}

    def download_output(self, job_name):
        lines = []
        for line in self.inputs[self.jobs[job_name]["src"]].splitlines():
            request = json.loads(line)
            lines.append(json.dumps({"key": request["key"], "response": self.responder(request["request"])}))
        return "\n".join(lines) + "\n"


def gemini_text(obj):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}


def stage1_response(_request):
    return gemini_text({
        "usable": True, "document_type": "form", "illegible_regions": 0,
        "pages": [{"page": 1, "legibility": "good", "text": "Name: John Smith\nDate: 2020-01-02"}],
    })


def stage2_response(request):
    return gemini_text({
        "document_type": "form",
        "fields": [
            {"name": "patient_name", "section": "patient", "data_type": "string", "raw_value": "John Smith",
             "normalized_value": "John Smith", "status": "extracted", "confidence": "high", "description": "What this field holds."},
            {"name": "visit_date", "section": "visit", "data_type": "date", "raw_value": "2020-01-02",
             "normalized_value": "2020-01-02", "status": "extracted", "confidence": "high", "description": "What this field holds."},
        ],
    })


def responder(request):
    schema_keys = request["generation_config"]["response_schema"]["properties"]
    return stage1_response(request) if "pages" in schema_keys else stage2_response(request)


@pytest.fixture
def project(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "annotation.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    data = tmp_path / "data"

    img_dir = data / "extracted" / "imgsrc"
    img_dir.mkdir(parents=True)
    Image.new("RGB", (40, 40), "white").save(img_dir / "doc_img.png")
    (img_dir / "big.pdf").write_bytes(b"%PDF-1.4 placeholder")

    funsd_img = data / "extracted" / "funsd" / "dataset" / "training_data" / "images"
    funsd_ann = data / "extracted" / "funsd" / "dataset" / "training_data" / "annotations"
    funsd_img.mkdir(parents=True)
    funsd_ann.mkdir(parents=True)
    Image.new("RGB", (40, 40), "white").save(funsd_img / "00001.png")
    (funsd_ann / "00001.json").write_text(json.dumps({"form": [
        {"id": 0, "text": "Name:", "box": [10, 10, 50, 20]},
        {"id": 1, "text": "John Smith", "box": [60, 10, 120, 20]},
        {"id": 2, "text": "Date:", "box": [10, 40, 50, 50]},
        {"id": 3, "text": "2020-01-02", "box": [60, 40, 120, 50]},
    ]}))

    rows = [
        {"document_id": "imgsrc:aaa", "source_id": "imgsrc", "processor": "generic", "phi_risk": False,
         "original_path": "extracted/imgsrc/doc_img.png", "page_paths": [], "file_type": "png",
         "page_count": 1, "license_review_status": "UNVERIFIED", "license_declared": "test", "category": "form",
         "source_url": None},
        {"document_id": "funsd:bbb", "source_id": "funsd", "processor": "funsd", "phi_risk": False,
         "original_path": "extracted/funsd/dataset/training_data/images/00001.png", "page_paths": [],
         "file_type": "png", "page_count": 1, "license_review_status": "UNVERIFIED", "license_declared": "test",
         "category": "form", "source_url": None},
        {"document_id": "imgsrc:ccc", "source_id": "imgsrc", "processor": "generic", "phi_risk": False,
         "original_path": "extracted/imgsrc/big.pdf", "page_paths": [], "file_type": "pdf",
         "page_count": 6, "license_review_status": "UNVERIFIED", "license_declared": "test", "category": "form",
         "source_url": None},
    ]
    (data / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    cfg = load_config(config_path)
    return cfg, Store(cfg), load_manifest(cfg)


def test_prepare_routes_each_document(project):
    cfg, store, docs = project
    report = prepare(cfg, store, docs)
    assert report["counts"] == {"needs_stage1": 1, "needs_stage2": 1, "excluded": 1}
    assert store.status("imgsrc:ccc")["reason"].startswith("too_many_pages")
    assert store.status("funsd:bbb")["route"] == "native_text"


def test_full_pipeline_accepts_grounded_annotations(project):
    cfg, store, docs = project
    prepare(cfg, store, docs)
    api = FakeApi(responder)

    assert submit(cfg, store, api, 1, docs, out=lambda _: None) == 1
    assert collect(cfg, store, api, 1, docs, out=lambda _: None)["stage1_ok"] == 1
    assert store.status("imgsrc:aaa")["state"] == "needs_stage2"

    assert submit(cfg, store, api, 2, docs, out=lambda _: None) == 2
    assert collect(cfg, store, api, 2, docs, out=lambda _: None)["accepted"] == 2
    assert store.status("imgsrc:aaa")["state"] == "accepted"
    assert store.status("funsd:bbb")["state"] == "accepted"

    summary = export(cfg, store, docs, include_unverified=True)
    assert summary["accepted_written"] == 2
    assert summary["target"].endswith("export/unverified")
    assert sum(summary["splits"].values()) == 2

    target = Path(summary["target"])
    rows = [json.loads(line) for name in ("train", "val", "test")
            for line in (target / f"{name}.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert all((target / image).exists() for image in row["images"])  # relative, self-contained
        assert row["text"] and row["fields"] and row["split"] in ("train", "val", "test")
    extract = [json.loads(line) for name in ("train", "val", "test")
               for line in (target / "sft" / f"extract_{name}.jsonl").read_text().splitlines()]
    assert len(extract) == 2
    assistant = json.loads(extract[0]["messages"][1]["content"])
    assert assistant["fields"] and extract[0]["messages"][0]["content"][0] == {"type": "image"}
    ocr = [json.loads(line) for name in ("train", "val", "test")
           for line in (target / "sft" / f"ocr_{name}.jsonl").read_text().splitlines()]
    assert ocr and all(len(e["images"]) == 1 for e in ocr)
    assert (target / "dataset.json").exists()



def test_resubmit_after_crash_reattaches_instead_of_duplicating(project):
    cfg, store, docs = project
    prepare(cfg, store, docs)
    api = FakeApi(responder)
    submit(cfg, store, api, 1, docs, out=lambda _: None)
    assert api.created == 1

    # Simulate a crash after the batch was created but before the documents were marked awaiting.
    store.set_status("imgsrc:aaa", "needs_stage1")
    submit(cfg, store, api, 1, docs, out=lambda _: None)
    assert api.created == 1
    assert store.status("imgsrc:aaa")["state"] == "awaiting_stage1"


def test_failed_batch_retries_then_excludes(project):
    cfg, store, docs = project
    prepare(cfg, store, docs)
    api = FakeApi(responder)
    submit(cfg, store, api, 1, docs, limit=1, out=lambda _: None)
    job_name = next(iter(api.jobs))
    api.jobs[job_name]["state"] = "JOB_STATE_EXPIRED"
    collect(cfg, store, api, 1, docs, out=lambda _: None)
    assert store.status("imgsrc:aaa")["state"] == "needs_stage1"
    assert store.status("imgsrc:aaa")["attempts"]["stage1"] == 1

    # A second failure reaches max_attempts (2) and the document is excluded, not retried forever.
    submit(cfg, store, api, 1, docs, limit=1, out=lambda _: None)
    job_name = max(api.jobs)
    api.jobs[job_name]["state"] = "JOB_STATE_FAILED"
    collect(cfg, store, api, 1, docs, out=lambda _: None)
    assert store.status("imgsrc:aaa")["state"] == "excluded"


def test_hallucinated_value_excludes_document(project):
    cfg, store, docs = project

    def bad_stage2(request):
        if "pages" in request["generation_config"]["response_schema"]["properties"]:
            return stage1_response(request)
        return gemini_text({
            "document_type": "form",
            "fields": [{"name": "patient_name", "section": "patient", "data_type": "string",
                        "raw_value": "Jane Doe", "normalized_value": "Jane Doe", "status": "extracted",
                        "confidence": "high", "description": "What this field holds."}],
        })

    prepare(cfg, store, docs)
    api = FakeApi(bad_stage2)
    submit(cfg, store, api, 1, docs, out=lambda _: None)
    collect(cfg, store, api, 1, docs, out=lambda _: None)
    submit(cfg, store, api, 2, docs, out=lambda _: None)
    collect(cfg, store, api, 2, docs, out=lambda _: None)
    assert store.status("imgsrc:aaa")["state"] == "excluded"
    assert "not_in_source:patient_name" in store.status("imgsrc:aaa")["reason"]


def test_stage1_gate_failure_excludes_without_stage2(project):
    cfg, store, docs = project
    prepare(cfg, store, docs)

    def unreadable(request):
        obj = json.loads(json.dumps(stage1_response(request)))
        payload = json.loads(obj["candidates"][0]["content"]["parts"][0]["text"])
        payload["pages"][0]["legibility"] = "unreadable"
        return gemini_text(payload)

    api = FakeApi(unreadable)
    submit(cfg, store, api, 1, docs, out=lambda _: None)
    collect(cfg, store, api, 1, docs, out=lambda _: None)
    assert store.status("imgsrc:aaa")["state"] == "excluded"
    assert "unreadable_page:1" in store.status("imgsrc:aaa")["reason"]


def test_phi_risk_source_is_never_sent_to_gemini(project):
    cfg, store, docs = project
    docs["imgsrc:aaa"]["phi_risk"] = True
    report = prepare(cfg, store, docs)
    assert store.status("imgsrc:aaa") is None or store.status("imgsrc:aaa")["state"] == "excluded"
    assert report["counts"].get("excluded", 0) >= 1
    assert store.status("imgsrc:aaa")["reason"] == "phi_gate_unauthorized_source"


def test_collect_progress_reports_each_batch_and_each_document(project):
    cfg, store, docs = project
    prepare(cfg, store, docs)
    api = FakeApi(responder)
    submit(cfg, store, api, 1, docs, out=lambda _: None, progress=True)
    lines: list[str] = []
    collect(cfg, store, api, 1, docs, out=lines.append, progress=True)
    # Batch progress is a live bar now; the per-document result line still goes through `out`.
    assert any("[ok  ] imgsrc:aaa" in line for line in lines)
    assert store.status("imgsrc:aaa")["state"] == "needs_stage2"


def test_watch_polls_until_the_batch_finishes(project, monkeypatch):
    cfg, store, docs = project
    prepare(cfg, store, docs)
    api = FakeApi(responder)
    submit(cfg, store, api, 1, docs, out=lambda _: None)
    job_name = next(iter(api.jobs))
    api.jobs[job_name]["state"] = "JOB_STATE_RUNNING"

    def finish_after_first_sleep(_seconds):
        api.jobs[job_name]["state"] = gemini.SUCCEEDED

    monkeypatch.setattr("labeller.pipeline.time.sleep", finish_after_first_sleep)
    lines: list[str] = []
    from labeller.pipeline import watch
    result = watch(cfg, store, api, 1, docs, interval=1, out=lines.append)
    assert result["stage1_ok"] == 1
    assert any("JOB_STATE_RUNNING" in line for line in lines)
    assert any("still running" in line for line in lines)


def test_request_key_round_trips_through_result_parsing():
    key = request_key("funsd:0123456789abcdef", 2, "stage2_v1")
    assert key == "funsd:0123456789abcdef:stage2:stage2_v1"
    assert key.rsplit(":", 2)[0] == "funsd:0123456789abcdef"
