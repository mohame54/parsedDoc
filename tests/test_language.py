import dataclasses
import json

import pytest

from labeller.config import LanguageConfig, load_config
from labeller.language import detect_language, from_code, language_reason, normalize_code
from labeller.pipeline import collect, ocr_step, prepare, submit
from tests.test_pipeline import FakeApi, gemini_text
from tests.test_pipeline_ocr import CONFIG_OCR, project  # noqa: F401  (fixture)

ALLOWED = frozenset({"en", "es", "fr", "de", "it", "pt", "zh", "ja"})
CFG = LanguageConfig(enabled=True, allowed=ALLOWED)

TEXTS = {
    "en": "Patient name: John Smith. Date of birth: 12 May 1980. Please sign below and return this form.",
    "es": "Nombre del paciente: Juan Pérez. Fecha de nacimiento: 12 de mayo de 1980. Dirección: Calle Mayor 5",
    "fr": "Nom du patient: Jean Dupont. Date de naissance: 3 mars 1975. Adresse: 12 rue de la Paix, Paris",
    "it": "Nome del paziente: Mario Rossi. Data di nascita: 12 maggio 1980. Indirizzo: Via Roma 5, Milano",
    "pt": "Nome do paciente: João Silva. Data de nascimento: 12 de maio de 1980. Endereço: Rua Augusta 5",
    "zh": "患者姓名：王小明 出生日期：1980年5月12日 地址：北京市朝阳区",
    "ja": "患者氏名：山田太郎 生年月日：1980年5月12日 住所：東京都新宿区のクリニックです",
    "nl": "Naam van de patiënt: Jan de Vries. Geboortedatum: 12 mei 1980. Adres: Hoofdstraat 5, Amsterdam",
    "bn": "রোগীর নাম: রহিম উদ্দিন বয়স: ৩৫ বছর ঠিকানা: ঢাকা",
    "ar": "اسم المريض: أحمد محمد. تاريخ الميلاد: ١٢ مايو ١٩٨٠. العنوان: القاهرة شارع التحرير",
}


@pytest.mark.parametrize("code", ["en", "es", "fr", "it", "pt", "zh", "ja", "nl", "bn", "ar"])
def test_detects_each_language(code):
    result = detect_language(TEXTS[code], CFG)
    assert result.primary == code
    assert not result.mixed


def test_allowed_languages_pass_and_others_are_excluded():
    assert language_reason(detect_language(TEXTS["es"], CFG), CFG) is None
    assert language_reason(detect_language(TEXTS["zh"], CFG), CFG) is None
    assert language_reason(detect_language(TEXTS["nl"], CFG), CFG) == "language:nl"
    assert language_reason(detect_language(TEXTS["bn"], CFG), CFG) == "language:bn"


def test_japanese_with_kana_is_not_taken_for_chinese():
    assert detect_language(TEXTS["ja"], CFG).primary == "ja"
    assert detect_language(TEXTS["zh"], CFG).primary == "zh"


def test_ocr_noise_markup_and_numbers_do_not_confuse_the_detector():
    noisy = "Pat1ent name: J0hn Sm1th. Dat3 of b1rth: 12 May 1980. Please s1gn bel0w and return th1s f0rm."
    markup = "Name: [handwritten: John Smith] [x] Yes [ ] No  Please sign below and return this form [signature]"
    assert detect_language(noisy, CFG).primary == "en"
    assert detect_language(markup, CFG).primary == "en"


def test_too_little_text_is_unknown_and_follows_the_policy():
    for text in ("Name John", "2020 01 02 415 898 1555 $26,880.00", ""):
        assert detect_language(text, CFG).primary is None
    result = detect_language("Name John", CFG)
    assert language_reason(result, CFG) == "language_unknown"
    lenient = dataclasses.replace(CFG, unknown_policy="accept")
    assert language_reason(result, lenient) is None


def test_mixed_language_policies():
    mixed = detect_language(TEXTS["en"] + " " + "患者姓名：王小明 出生日期：1980年5月12日 地址：北京市朝阳区长安街五号 联系电话", CFG)
    assert mixed.primary == "en" and mixed.mixed
    assert language_reason(mixed, CFG) is None                                    # primary allowed
    strict = dataclasses.replace(CFG, mixed_language_policy="reject_mixed")
    assert language_reason(mixed, strict).startswith("language_mixed:")
    all_allowed = dataclasses.replace(CFG, mixed_language_policy="accept_if_all_allowed")
    assert language_reason(mixed, all_allowed) is None                            # en and zh are both allowed
    en_only = dataclasses.replace(all_allowed, allowed=frozenset({"en"}))
    assert language_reason(mixed, en_only).startswith("language_mixed:")


def test_disabled_filter_never_excludes():
    off = LanguageConfig()
    assert language_reason(detect_language(TEXTS["nl"], CFG), off) is None


def test_gemini_language_codes_are_normalised():
    assert normalize_code("en-US") == "en" and normalize_code("zh_CN") == "zh"
    for bad in ("unknown", "", None, "und", "Spanish", "mul"):
        assert normalize_code(bad) is None
    assert from_code("en-GB").primary == "en"
    assert from_code("unknown").primary is None


def write_config(tmp_path, language_block):
    path = tmp_path / "config" / "annotation.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(CONFIG_OCR + language_block, encoding="utf-8")
    return path


def test_config_parses_the_language_section(tmp_path):
    cfg = load_config(write_config(tmp_path, """
language:
  allowed: [en, es, zh]
  unknown_policy: reject
  mixed_language_policy: accept_if_primary_allowed
"""))
    assert cfg.language.enabled and cfg.language.allowed == {"en", "es", "zh"}
    assert not load_config(write_config(tmp_path, "")).language.enabled   # no section, no filtering


@pytest.mark.parametrize("block, message", [
    ("\nlanguage:\n  allowed: []\n", "at least one language"),
    ("\nlanguage:\n  allowed: [english]\n", "invalid codes"),
    ("\nlanguage:\n  allowed: [en]\n  unknown_policy: maybe\n", "unknown_policy"),
    ("\nlanguage:\n  allowed: [en]\n  mixed_language_policy: sometimes\n", "mixed_language_policy"),
])
def test_config_rejects_bad_language_settings(tmp_path, block, message):
    with pytest.raises(ValueError, match=message):
        load_config(write_config(tmp_path, block))


class TextOcr:
    def __init__(self, text):
        self.text = text

    def __call__(self):
        return self

    def recognise(self, image):
        return [{"text": line, "score": 0.95, "box": [[10, 10 + 30 * i], [200, 10 + 30 * i],
                                                      [200, 20 + 30 * i], [10, 20 + 30 * i]]}
                for i, line in enumerate(self.text.split("\n"))]


def with_language(cfg, **overrides):
    return dataclasses.replace(cfg, language=LanguageConfig(enabled=True, allowed=ALLOWED, **overrides))


def test_wrong_language_is_excluded_after_ocr_before_any_gemini_call(project):  # noqa: F811
    cfg, store, docs = project
    cfg = with_language(cfg)
    prepare(cfg, store, docs)
    ocr_step(cfg, store, TextOcr(TEXTS["nl"].replace(". ", "\n")), docs, out=lambda _: None)
    status = store.status("imgsrc:aaa")
    assert status["state"] == "excluded" and status["reason"] == "language:nl"
    assert status["language"]["primary"] == "nl"


def test_allowed_language_continues_and_is_recorded_on_the_annotation(project):  # noqa: F811
    cfg, store, docs = project
    cfg = with_language(cfg)
    prepare(cfg, store, docs)
    ocr_step(cfg, store, TextOcr("Name: John Smith\nDate: 2020-01-02\nPlease indicate the capacity in which "
                                 "you are executing this document"), docs, out=lambda _: None)
    assert store.status("imgsrc:aaa")["state"] == "needs_stage1"

    def responder(request):
        props = request["generation_config"]["response_schema"]["properties"]
        if "corrections" in props:
            return gemini_text({
                "usable": True, "document_type": "form", "language": "en", "fill_level": "filled",
                "illegible_regions": 0,
                "pages": [{"page": 1, "legibility": "good", "has_handwriting": False, "has_checkboxes": False,
                           "has_signature": False, "has_table": False}],
                "corrections": [], "missed": []})
        return gemini_text({"document_type": "form", "fields": [
            {"name": "patient_name", "section": "s", "data_type": "string", "raw_value": "John Smith",
             "normalized_value": "", "status": "extracted", "confidence": "high", "description": "What this field holds."}]})

    api = FakeApi(responder)
    submit(cfg, store, api, 1, docs, out=lambda _: None)
    collect(cfg, store, api, 1, docs, out=lambda _: None)
    submit(cfg, store, api, 2, docs, out=lambda _: None)
    collect(cfg, store, api, 2, docs, out=lambda _: None)
    annotation = json.loads((store.ann_dir("imgsrc:aaa") / "annotation.json").read_text())
    assert annotation["language"]["primary"] == "en"


def test_gemini_disagreeing_about_the_language_excludes_at_stage1(project):  # noqa: F811
    cfg, store, docs = project
    cfg = with_language(cfg)
    prepare(cfg, store, docs)
    ocr_step(cfg, store, TextOcr("Name: John Smith\nDate: 2020-01-02\nPlease indicate the capacity in which "
                                 "you are executing this document"), docs, out=lambda _: None)

    def responder(request):
        return gemini_text({
            "usable": True, "document_type": "form", "language": "ru", "fill_level": "filled",
            "illegible_regions": 0,
            "pages": [{"page": 1, "legibility": "good", "has_handwriting": False, "has_checkboxes": False,
                       "has_signature": False, "has_table": False}],
            "corrections": [], "missed": []})

    api = FakeApi(responder)
    submit(cfg, store, api, 1, docs, out=lambda _: None)
    collect(cfg, store, api, 1, docs, out=lambda _: None)
    assert store.status("imgsrc:aaa")["reason"] == "language:ru (stage1)"


def test_without_a_language_section_nothing_is_detected(project):  # noqa: F811
    cfg, store, docs = project
    assert not cfg.language.enabled
    prepare(cfg, store, docs)
    ocr_step(cfg, store, TextOcr(TEXTS["nl"].replace(". ", "\n")), docs, out=lambda _: None)
    assert store.status("imgsrc:aaa")["state"] == "needs_stage1"
    assert store.status("imgsrc:aaa")["language"] is None
