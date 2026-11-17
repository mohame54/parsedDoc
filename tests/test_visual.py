"""Visual routing: which documents get the multimodal Stage 2 and which stay text only."""
import dataclasses
import json

import fitz
import pytest
from PIL import Image, ImageDraw

from labeller.config import VisualConfig, load_config
from labeller.pipeline import (
    collect,
    ocr_step,
    prepare,
    prompt_version_for,
    refresh_visual,
    stage2_uses_images,
    submit,
)
from labeller.store import Store
from labeller.visual import add_stage1_flags, analyze, page_signals, pdf_structure
from tests.test_pipeline import FakeApi, gemini_text, stage2_response
from tests.test_pipeline_ocr import CONFIG_OCR, FakeOcr
from tests.test_stage2_v3 import v3

CFG = VisualConfig(mode="auto")


def page(draw_extra=None, size=(850, 1100)):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for line in range(30):
        draw.text((60, 60 + line * 30), "Patient name John Smith date of birth 1980 address Main Street", fill="black")
    if draw_extra:
        draw_extra(draw)
    return image


def routed(image):
    signals = page_signals(image, CFG)
    return signals


def test_plain_text_and_thin_lines_are_not_visual_content():
    plain = routed(page())
    assert max(plain.values()) == 0.0

    def form_lines(draw):
        draw.line((60, 400, 790, 400), fill="black", width=2)
        draw.rectangle((60, 420, 790, 520), outline="black", width=2)

    assert max(routed(page(form_lines)).values()) < 0.005


def test_coloured_and_shaded_rows_are_detected():
    yellow = routed(page(lambda d: d.rectangle((60, 300, 790, 360), fill=(255, 250, 170))))
    gray = routed(page(lambda d: d.rectangle((60, 300, 790, 360), fill=(225, 225, 225))))
    assert yellow["color"] >= CFG.min_color_area and yellow["shaded"] >= CFG.min_shaded_area
    assert gray["color"] == 0.0 and gray["shaded"] >= CFG.min_shaded_area


def test_large_dark_regions_are_detected_but_edge_bands_are_not():
    logo = routed(page(lambda d: d.ellipse((500, 700, 760, 960), fill=(30, 30, 30))))
    assert logo["blob"] >= CFG.min_blob_area

    def microfilm_bands(draw):
        draw.rectangle((0, 0, 850, 70), fill="black")        # scanner borders touch the page edge
        draw.rectangle((0, 1030, 850, 1100), fill="black")

    assert routed(page(microfilm_bands))["blob"] < CFG.min_blob_area


def test_a_coloured_stamp_is_detected():
    def stamp(draw):
        draw.rectangle((560, 800, 800, 920), outline=(200, 30, 30), width=8)
        draw.text((600, 850), "APPROVED", fill=(200, 30, 30))

    assert routed(page(stamp))["color"] >= CFG.min_color_area


def make_doc(tmp_path, name, image):
    path = tmp_path / "data" / "extracted" / "src" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return {"document_id": f"src:{name}", "source_id": "src", "original_path": f"extracted/src/{name}.png",
            "page_paths": [], "file_type": "png", "page_count": 1}


def test_analyze_reports_reasons_and_signals(tmp_path):
    doc = make_doc(tmp_path, "shaded", page(lambda d: d.rectangle((60, 300, 790, 360), fill=(225, 225, 225))))
    result = analyze(doc, tmp_path / "data", CFG, max_pages=4)
    assert result["multimodal"] and result["source"] == "local"
    assert any(reason.startswith("shading:") for reason in result["reasons"])
    plain = analyze(make_doc(tmp_path, "plain", page()), tmp_path / "data", CFG, max_pages=4)
    assert not plain["multimodal"] and plain["reasons"] == []


def write_pdf(path, build):
    pdf = fitz.open()
    build(pdf.new_page())
    pdf.save(path)
    pdf.close()
    return path


def test_pdf_structure_finds_pictures_annotations_and_checkboxes(tmp_path):
    def build(pdf_page):
        pdf_page.insert_text((72, 72), "Quarterly report with a chart and a stamp")
        pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100), False)
        pixmap.set_rect(pixmap.irect, (30, 90, 200))
        pdf_page.insert_image(fitz.Rect(72, 120, 300, 300), pixmap=pixmap)
        pdf_page.add_highlight_annot(pdf_page.search_for("report")[0])
        widget = fitz.Widget()
        widget.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
        widget.field_name = "agree"
        widget.rect = fitz.Rect(72, 400, 90, 418)
        pdf_page.add_widget(widget)

    structure = pdf_structure(write_pdf(tmp_path / "rich.pdf", build), CFG, max_pages=4)
    assert structure["image_area"] > 0.02
    assert structure["annotations"] == 1
    assert structure["checkboxes"] == 1


def test_a_full_page_scan_is_not_counted_as_an_embedded_picture(tmp_path):
    def build(pdf_page):
        pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 140), False)
        pixmap.set_rect(pixmap.irect, (250, 250, 250))
        pdf_page.insert_image(pdf_page.rect, pixmap=pixmap)

    structure = pdf_structure(write_pdf(tmp_path / "scan.pdf", build), CFG, max_pages=4)
    assert structure["scan_pages"] == 1 and structure["image_area"] == 0.0


def test_a_plain_text_pdf_has_no_structure_signals(tmp_path):
    structure = pdf_structure(write_pdf(tmp_path / "text.pdf", lambda p: p.insert_text((72, 72), "Just text.")),
                              CFG, max_pages=4)
    assert (structure["image_area"], structure["annotations"], structure["checkboxes"]) == (0.0, 0, 0)


def test_unreadable_pdf_gives_no_signal_instead_of_an_error(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    assert pdf_structure(broken, CFG, max_pages=4)["annotations"] == 0


def test_stage1_flags_only_add_reasons():
    plain = {"multimodal": False, "reasons": [], "signals": {}, "source": "local"}
    stage1 = {"pages": [{"has_table": True, "has_checkboxes": False, "has_signature": False,
                         "has_handwriting": True}]}
    merged = add_stage1_flags(plain, stage1, CFG)
    assert merged["multimodal"] and merged["reasons"] == ["stage1:has_table"] and merged["source"] == "local+stage1"
    # has_handwriting is not in the default flag list, and nothing is ever removed.
    only_hand = add_stage1_flags(plain, {"pages": [{"has_handwriting": True}]}, CFG)
    assert not only_hand["multimodal"]
    local = {"multimodal": True, "reasons": ["shading:0.050"], "signals": {}, "source": "local"}
    assert add_stage1_flags(local, {"pages": [{}]}, CFG)["multimodal"]


def test_config_defaults_and_validation(tmp_path):
    path = tmp_path / "config" / "annotation.yaml"
    path.parent.mkdir()
    path.write_text(CONFIG_OCR, encoding="utf-8")
    assert load_config(path).visual.mode == "never"                      # no section: text only

    path.write_text(CONFIG_OCR + "\nvisual:\n  min_blob_area: 0.05\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.visual.mode == "auto" and cfg.visual.min_blob_area == 0.05  # a section defaults to auto

    for block, message in [("mode: sometimes", "visual.mode"), ("stage1_flags: [has_pizza]", "unknown flags"),
                           ("min_color_area: 3", "share of the page")]:
        path.write_text(CONFIG_OCR + f"\nvisual:\n  {block}\n", encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_config(path)


def test_modes_decide_which_documents_get_images():
    class Cfg:
        stage1 = type("S", (), {"prompt_version": "stage1_v3"})
        stage2 = type("S", (), {"prompt_version": "stage2_v1", "send_images": False})
        visual = VisualConfig(mode="auto")

    visual = {"visual": {"multimodal": True}}
    assert stage2_uses_images(Cfg, visual) and not stage2_uses_images(Cfg, {"visual": {"multimodal": False}})
    assert not stage2_uses_images(Cfg, {})                                # not analysed: text only
    Cfg.visual = VisualConfig(mode="never")
    assert not stage2_uses_images(Cfg, visual)
    Cfg.visual = VisualConfig(mode="always")
    assert stage2_uses_images(Cfg, {})
    Cfg.visual = VisualConfig(mode="never")
    Cfg.stage2 = type("S", (), {"prompt_version": "stage2_v1", "send_images": True})
    assert stage2_uses_images(Cfg, {}) and prompt_version_for(2, Cfg, {}) == "stage2_v7"


# --- end to end ---------------------------------------------------------------------------------------

def stage1_response(flag=False):
    return gemini_text({
        "usable": True, "document_type": "form", "language": "en", "fill_level": "filled",
        "illegible_regions": 0,
        "pages": [{"page": 1, "legibility": "good", "has_handwriting": False, "has_checkboxes": False,
                   "has_signature": False, "has_table": flag}],
        "corrections": [{"line": "P1-L2", "text": "Date: 2020-01-02"}], "missed": []})


def make_responder(table_flag_for=()):
    def responder(request):
        props = request["generation_config"]["response_schema"]["properties"]
        if "corrections" in props:
            text = json.dumps(request)
            return stage1_response(flag=any(token in text for token in table_flag_for))
        return gemini_text(v3()) if "tables" in props else stage2_response(request)
    return responder


@pytest.fixture
def two_docs(tmp_path):
    config = tmp_path / "config" / "annotation.yaml"
    config.parent.mkdir()
    config.write_text(CONFIG_OCR + "\nvisual:\n  mode: auto\n  stage1_flags: [has_table]\n", encoding="utf-8")
    cfg = load_config(config)
    rows = [
        {**make_doc(tmp_path, "plain", page()), "processor": "generic", "phi_risk": False,
         "license_review_status": "UNVERIFIED", "license_declared": "t", "category": "form", "source_url": None},
        {**make_doc(tmp_path, "shaded", page(lambda d: d.rectangle((60, 300, 790, 360), fill=(225, 225, 225)))),
         "processor": "generic", "phi_risk": False, "license_review_status": "UNVERIFIED",
         "license_declared": "t", "category": "form", "source_url": None},
    ]
    (tmp_path / "data" / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    docs = {row["document_id"]: row for row in rows}
    return cfg, Store(cfg), docs


def run_to_stage2_requests(cfg, store, docs, responder):
    prepare(cfg, store, docs)
    ocr_step(cfg, store, FakeOcr, docs, out=lambda _: None)
    api = FakeApi(responder)
    submit(cfg, store, api, 1, docs, out=lambda _: None)
    collect(cfg, store, api, 1, docs, out=lambda _: None)
    before = set(api.inputs)
    submit(cfg, store, api, 2, docs, out=lambda _: None)
    requests = [json.loads(line) for name in set(api.inputs) - before for line in api.inputs[name].splitlines()]
    collect(cfg, store, api, 2, docs, out=lambda _: None)
    return {request["key"].split(":stage2:")[0]: request for request in requests}


def test_each_document_is_routed_by_its_own_visual_content(two_docs):
    cfg, store, docs = two_docs
    requests = run_to_stage2_requests(cfg, store, docs, make_responder())
    plain, shaded = requests["src:plain"], requests["src:shaded"]

    assert plain["key"].endswith("stage2_v5") and "file_data" not in json.dumps(plain)
    assert "tables" not in plain["request"]["generation_config"]["response_schema"]["properties"]

    assert shaded["key"].endswith("stage2_v7") and "file_data" in json.dumps(shaded)
    assert "tables" in shaded["request"]["generation_config"]["response_schema"]["properties"]

    assert store.status("src:plain")["visual"]["multimodal"] is False
    assert any(r.startswith("shading:") for r in store.status("src:shaded")["visual"]["reasons"])
    annotation = json.loads((store.ann_dir("src:shaded") / "annotation.json").read_text())
    assert annotation["tables"] and annotation["visual_routing"]["multimodal"] is True
    plain_annotation = json.loads((store.ann_dir("src:plain") / "annotation.json").read_text())
    assert "tables" not in plain_annotation and plain_annotation["visual_routing"]["reasons"] == []


def test_a_stage1_table_flag_upgrades_a_text_only_scan_to_multimodal(two_docs):
    cfg, store, docs = two_docs
    # Stage 1 says the plain scan has a table (its request carries the plain image's path).
    requests = run_to_stage2_requests(cfg, store, docs, make_responder(table_flag_for=("plain",)))
    assert requests["src:plain"]["key"].endswith("stage2_v7")
    assert "stage1:has_table" in store.status("src:plain")["visual"]["reasons"]


def test_never_mode_keeps_everything_text_only(two_docs):
    cfg, store, docs = two_docs
    cfg = dataclasses.replace(cfg, visual=VisualConfig(mode="never"))
    requests = run_to_stage2_requests(cfg, store, docs, make_responder())
    assert all(request["key"].endswith("stage2_v5") for request in requests.values())
    assert store.status("src:shaded")["visual"] is None                  # not even analysed


def test_plan_and_visual_command_report_the_routing(two_docs):
    cfg, store, docs = two_docs
    report = prepare(cfg, store, docs, dry_run=True)
    assert report["multimodal_docs"] == 1 and report["multimodal_pages"] == 1
    prepare(cfg, store, docs)
    rows = {row["doc_id"]: row for row in refresh_visual(cfg, store, docs)}
    assert rows["src:shaded"]["multimodal"] and not rows["src:plain"]["multimodal"]
    # Loosening the shading threshold in the config changes the routing once applied.
    loose = dataclasses.replace(cfg, visual=VisualConfig(mode="auto", min_shaded_area=0.5))
    refresh_visual(loose, store, docs, apply=True)
    assert store.status("src:shaded")["visual"]["multimodal"] is False


def test_fast_opening_matches_pillow_away_from_the_borders_and_is_much_faster():
    import time

    import numpy as np
    from PIL import Image, ImageFilter

    from labeller.visual import _opening

    rng = np.random.default_rng(3)
    mask = np.zeros((300, 400), dtype=np.uint8)
    mask[40:140, 60:200] = 255                       # a solid block: must survive
    mask[200:204, 50:350] = 255                      # a thin rule: must be removed
    mask[rng.random(mask.shape) > 0.995] = 255       # speckle: must be removed
    image = Image.fromarray(mask)
    for size in (7, 25):
        start = time.perf_counter()
        fast = np.asarray(_opening(image, size))
        t_fast = time.perf_counter() - start
        start = time.perf_counter()
        slow = np.asarray(image.filter(ImageFilter.MinFilter(size)).filter(ImageFilter.MaxFilter(size)))
        t_slow = time.perf_counter() - start
        r = size  # compare away from the border band, where the two treat missing neighbours differently
        assert (fast[r:-r, r:-r] == slow[r:-r, r:-r]).all(), size
        assert fast[90, 100] == 255 and fast[202, 100] == 0     # block kept, thin rule gone
        assert t_fast < t_slow
