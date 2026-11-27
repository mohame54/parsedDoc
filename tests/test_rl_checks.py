"""The preflight checks of train/rl.py: they pass on good data and fail on each risk they are meant to catch."""
import copy
import dataclasses
import json
import types

import pytest
import torch

from tests.test_rl import build, item
from train import rl_checks
from train.rl import reward_function


@pytest.fixture()
def rows(tmp_path):
    return build(tmp_path, [item(i) for i in range(40)]), tmp_path


def by_name(checks):
    return {c.name: c for c in checks}


def test_good_data_passes_the_integrity_and_reward_checks(rows):
    data, path = rows
    checks = by_name(rl_checks.check_data(data, path, 1536))
    for name in ("data rows", "reward ordering", "answer schema", "answer length"):
        assert checks[name].status == "pass", checks[name].line()
    assert any("must_abstain" in r["states"] and '"must_abstain"' in r["states"] for r in data)   # the data has masks


def test_a_missing_image_a_bad_json_column_and_an_old_build_are_failures(rows):
    data, path = rows
    (path / data[0]["image"]).unlink()
    assert rl_checks.check_data(data, path, 1536)[0].status == "fail"
    broken = copy.deepcopy(data)
    broken[1]["states"] = "{not json"
    assert "unusable" in rl_checks.check_data(broken[1:2], path, 1536)[0].detail
    old = [{k: v for k, v in data[2].items() if k != "base_schema"}]
    result = rl_checks.check_data(old, path, 1536)[0]
    assert result.status == "fail" and "rebuild" in result.detail


def test_the_ordering_check_fails_when_the_reward_stops_preferring_the_best_answer(rows, monkeypatch):
    data, path = rows
    monkeypatch.setattr(rl_checks, "score_rl", lambda *a, **k: {"total": 0.5, "parts": {}})
    assert by_name(rl_checks.check_data(data, path, 1536))["reward ordering"].status == "fail"


def test_answers_longer_than_the_budget_are_flagged(rows):
    data, path = rows
    assert by_name(rl_checks.check_data(data, path, 5))["answer length"].status == "fail"


def test_the_balance_checks_warn_about_collapse_and_about_too_little_abstention(rows):
    data, _ = rows
    everything_masked = []
    for row in data[:10]:
        row = dict(row, states=json.dumps({p: "must_abstain" for p in json.loads(row["states"])}))
        everything_masked.append(row)
    assert by_name(rl_checks._balance(everything_masked))["abstention balance"].status == "warn"
    nothing_masked = [dict(r, states=json.dumps({p: "answerable" for p in json.loads(r["states"])})) for r in data[:10]]
    out = by_name(rl_checks._balance(nothing_masked))
    assert out["abstention balance"].status == "warn" and out["clean examples"].status == "pass"
    assert by_name(rl_checks._balance(everything_masked))["clean examples"].status == "warn"


def test_batching_needs_whole_generation_groups():
    assert rl_checks.check_batching(8, 2, 8).status == "pass"
    assert rl_checks.check_batching(6, 1, 8).status == "fail"
    assert rl_checks.check_batching(4, 2, 8, world_size=2).status == "pass"
    assert rl_checks.check_batching(8, 1, 1).status == "fail"


def _fake_trl(config_fields=("num_generations", "max_completion_length", "beta"),
              trainer_args=("model", "reward_funcs", "args", "train_dataset", "processing_class")):
    config = dataclasses.make_dataclass("GRPOConfig", [(f, int, 0) for f in config_fields])
    namespace = {"__init__": eval(f"lambda self, {', '.join(trainer_args)}: None")}      # noqa: S307 - test fixture
    return types.SimpleNamespace(__version__="9.9", GRPOConfig=config, GRPOTrainer=type("GRPOTrainer", (), namespace))


def test_trl_checks_catch_a_missing_setting_a_changed_constructor_and_a_bad_reward_signature():
    ok = by_name(rl_checks.check_trl(reward_function, _fake_trl()))
    assert ok["GRPOTrainer"].status == "pass" and ok["reward function"].status == "pass"
    assert ok["GRPOConfig"].status == "warn"                                  # beta exists, the other settings do not
    assert by_name(rl_checks.check_trl(None, _fake_trl(config_fields=("beta",))))["GRPOConfig"].status == "fail"
    assert by_name(rl_checks.check_trl(None, _fake_trl(trainer_args=("model", "args"))))["GRPOTrainer"].status == "fail"
    assert by_name(rl_checks.check_trl(lambda completions: [], _fake_trl()))["reward function"].status == "fail"


def test_trl_not_installed_is_a_skip_not_a_failure(monkeypatch):
    import builtins
    real = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", lambda name, *a, **k: (_ for _ in ()).throw(ImportError())
                        if name == "trl" else real(name, *a, **k))
    assert rl_checks.check_trl()[0].status == "skip"


# ---- processor parity, with a fake Hugging Face processor so no download is needed ----------------------

def _patch_processor(monkeypatch, extra_image_tokens=0, view_shape=None, tiles=True):
    import transformers

    from tests.test_deepseek_ocr import StubTokenizer
    from train import processors
    from train.processors import DeepSeekOCRV2Transform, batch_adapter_for
    from train.processors.deepseekOcr import IMAGE_TOKEN_ID

    transform = DeepSeekOCRV2Transform(tokenizer=StubTokenizer(), mode="base", fixed_length=False)
    image = __import__("PIL.Image", fromlist=["Image"]).new("RGB", (1100, 1500), "white")
    batch = transform.make_collator()([transform([image], "Extract.", "x")])
    ours = batch_adapter_for("deepseek-community/DeepSeek-OCR-2")(batch)
    n = int((batch["input_ids"] == IMAGE_TOKEN_ID).sum()) + extra_image_tokens

    class FakeProcessor:
        def apply_chat_template(self, *a, **k):
            return "text"

        def __call__(self, **k):
            out = {"input_ids": torch.tensor([[IMAGE_TOKEN_ID] * n + [5, 6]]),
                   "pixel_values": torch.zeros(view_shape or tuple(ours["pixel_values"].shape))}
            if ours.get("pixel_values_local") is not None and tiles:
                out["pixel_values_local"] = torch.zeros(tuple(ours["pixel_values_local"].shape))
            return out
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(lambda *a, **k: FakeProcessor()))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        staticmethod(lambda *a, **k: types.SimpleNamespace(image_token_id=IMAGE_TOKEN_ID)))
    monkeypatch.setattr(processors, "create_transform", lambda *a, **k: transform)


def test_processor_parity_passes_when_the_inputs_agree(monkeypatch):
    _patch_processor(monkeypatch)
    assert rl_checks.check_processor("deepseek-community/DeepSeek-OCR-2").status == "pass"


def test_processor_parity_fails_on_a_different_token_count_or_view_shape(monkeypatch):
    _patch_processor(monkeypatch, extra_image_tokens=3)
    result = rl_checks.check_processor("deepseek-community/DeepSeek-OCR-2")
    assert result.status == "fail" and "image tokens" in result.detail
    _patch_processor(monkeypatch, view_shape=(1, 3, 10, 10))
    assert "global view" in rl_checks.check_processor("deepseek-community/DeepSeek-OCR-2").detail


def test_processor_parity_skips_when_the_files_cannot_be_loaded(monkeypatch):
    import transformers

    def fail(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(fail))
    result = rl_checks.check_processor("deepseek-community/DeepSeek-OCR-2")
    assert result.status == "skip" and "offline" in result.detail


def test_the_report_and_the_failed_flag():
    checks = [rl_checks.Check("a", "pass", "ok"), rl_checks.Check("b", "skip", "no gpu")]
    assert not rl_checks.failed(checks) and "[SKIP] b: no gpu" in rl_checks.report(checks)
    assert rl_checks.failed(checks + [rl_checks.Check("c", "fail", "bad")])


def test_the_format_mix_counts_the_plain_format_written_as_null(rows):
    data, _ = rows
    plain = [dict(r, spec="null") for r in data[:10]]            # how `build` writes a row in the plain format
    assert "0% of examples use the extended" in by_name(rl_checks._balance(plain))["format mix"].detail
