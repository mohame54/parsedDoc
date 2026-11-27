"""The CPU-testable parts of train/rl.py: the data rows, the reward adapter and the config filter. The TRL trainer
itself is not exercised here."""
import dataclasses
import json
import random

import pytest

from PIL import Image

from train.rl import (build_rows, completion_text, load_rows, reward_function, supported_kwargs, to_dataset_row,
                      write_rows)
from train.rl_prompts import wrap_answer
from train.gen_eval import schema_from_prompt

SCHEMA = {"type": "object", "properties": {"name": {"type": ["string", "null"]}, "total": {"type": ["number", "null"]}},
          "required": ["name", "total"], "additionalProperties": False}
REF = {"name": "Ana Lopez", "total": 12.5}
PROMPT = "Extract the data.\n\nJSON Schema:\n" + json.dumps(SCHEMA)
LINES = [{"text": "Ana Lopez", "box_px": [20, 20, 160, 44]}, {"text": "12.5", "box_px": [20, 70, 80, 94]}]


def item(i=0, pages=1, label=None):
    return {"id": f"doc:{i}#extract_to_schema0", "prompt": PROMPT, "label": label or json.dumps(REF),
            "images": [Image.new("RGB", (300, 200), (255, 255, 255)) for _ in range(pages)]}


def build(tmp_path, items, **kw):
    return build_rows(items, tmp_path, ocr_lines=lambda im: LINES, ocr_text=lambda im: "", level="strong", **kw)


def test_build_rows_saves_images_and_the_reward_metadata(tmp_path):
    rows = build(tmp_path, [item(0), item(1)])
    assert len(rows) == 2 and (tmp_path / rows[0]["image"]).exists()
    row = rows[0]
    assert json.loads(row["reference"]) == REF and isinstance(row["upright"], float)
    assert set(json.loads(row["states"])) == {"name", "total"}
    assert schema_from_prompt(row["prompt"]) == json.loads(row["prompt_schema"])


def test_multi_page_and_implausible_references_are_left_out(tmp_path):
    bad = json.dumps({"name": "x", "total": 1, "date": "2094-04-28"})
    rows = build(tmp_path, [item(0, pages=2), item(1, label=bad), item(2)])
    assert [r["id"] for r in rows] == ["doc:2#extract_to_schema0"]


def test_rows_round_trip_and_become_trl_rows(tmp_path):
    rows = build(tmp_path, [item(0)])
    write_rows(rows, tmp_path)
    loaded = load_rows(tmp_path)
    assert loaded == rows
    out = to_dataset_row(loaded[0], tmp_path)
    assert out["prompt"][0]["content"][0] == {"type": "image"} and out["image"].endswith(".png")
    assert "id" not in out and out["reference"] == loaded[0]["reference"]


def test_completion_text_handles_strings_and_chat_messages():
    assert completion_text("abc") == "abc"
    assert completion_text([{"role": "assistant", "content": "abc"}]) == "abc"
    assert completion_text([{"role": "assistant", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]) == "ab"
    assert completion_text(None) == ""


def test_the_reward_function_scores_each_completion_with_its_own_row():
    from train.rl_prompts import AnswerSpec, wrap_schema
    spec = AnswerSpec("value", "confidence", "percent", None, "rotation")
    schema = wrap_schema(SCHEMA, spec)
    states = json.dumps({"name": "answerable", "total": "answerable"})
    good = wrap_answer(REF, SCHEMA, spec, upright=90)
    wrong = wrap_answer({"name": "Zed", "total": 99}, SCHEMA, spec, upright=90)
    cols = {"reference": [json.dumps(REF)] * 4, "spec": [json.dumps(spec.to_dict())] * 3 + [""],
            "states": [states] * 4, "tags": ["{}"] * 4, "upright": [90.0] * 4,
            "prompt_schema": [json.dumps(schema)] * 3 + [""]}
    scores = reward_function(completions=[[{"role": "assistant", "content": good}], wrong, "not json",
                                          json.dumps(REF)], **cols)
    assert scores[0] == 1.0                       # exact values, sure confidence, right rotation, valid schema
    assert 0.0 < scores[1] < scores[0]            # valid JSON but wrong values
    assert scores[2] == 0.0                       # not JSON
    assert scores[3] == 1.0                       # the same row in the plain format (spec "" -> no extended terms)


def test_unknown_config_settings_are_dropped_not_fatal():
    @dataclasses.dataclass
    class Config:
        lr: float = 1.0
        beta: float = 0.0
    assert supported_kwargs(Config, {"lr": 2.0, "loss_type": "dapo", "beta": 0.1}) == {"lr": 2.0, "beta": 0.1}


# ---- tracking, reports and checkpoints of the RL run ------------------------------------------------------

def _columns(rows):
    return {k: [r[k] for r in rows] for k in ("reference", "spec", "states", "tags", "upright", "prompt_schema")}


def _plain_rows(n=3):
    states = json.dumps({"name": "answerable", "total": "answerable"})
    return {"reference": [json.dumps(REF)] * n, "spec": [""] * n, "states": [states] * n, "tags": ["{}"] * n,
            "upright": [0.0] * n, "prompt_schema": [""] * n}


def test_the_tracker_scores_like_the_plain_function_and_keeps_every_component():
    from train.rl import RewardTracker
    tracker = RewardTracker()
    cols = _plain_rows(3)
    completions = [json.dumps(REF), json.dumps({"name": "Ana Lopes", "total": 12.5}), "not json"]
    scores = tracker(completions=completions, **cols)
    assert scores == reward_function(completions=completions, **cols) and tracker.__name__ == "reward_rl"
    metrics = tracker.drain()
    assert metrics["reward/valid_json"] == pytest.approx(2 / 3) and "reward/content" in metrics
    assert metrics["quality/exact_accuracy_answerable"] == pytest.approx(0.75)      # 3 of 4 readable fields exactly right
    assert tracker.drain() == {}                                                     # drained: starts over


class FakeHub:
    def __init__(self, fail=False):
        self.calls, self.closed, self.fail = [], False, fail

    def push_folder(self, step, folder, final=False, report_dir=None):
        if self.fail:
            raise RuntimeError("network")
        self.calls.append((step, Path(folder).name, final, Path(report_dir).name if report_dir else None,
                           sorted(p.name for p in Path(report_dir).iterdir()) if report_dir else []))

    def close(self):
        self.closed = True


from pathlib import Path  # noqa: E402


def test_the_recorder_writes_history_reports_the_adapter_state_and_pushes_each_save(tmp_path):
    from train.rl import RewardTracker, RLCheckpointRecorder
    tracker, hub, saved = RewardTracker(), FakeHub(), []
    recorder = RLCheckpointRecorder(tmp_path / "out", prefix="rl", tracker=tracker, hub=hub,
                                    run_config={"lr": 5e-6, "hub_token": "no"}, lineage=lambda: [{"path": "sft.pt", "r": 16, "alpha": 32}],
                                    save_adapter=lambda path, step: saved.append((path.name, step)))
    checkpoint = tmp_path / "out" / "checkpoint-4"
    checkpoint.mkdir(parents=True)
    for step in range(1, 5):
        tracker(completions=[json.dumps(REF)], **_plain_rows(1))
        recorder.on_log(step, {"reward": 0.5 + 0.1 * step, "reward_std": 0.1, "kl": 0.01, "note": "text", "loss": 0.2})
    recorder.on_save(4, checkpoint)
    final = tmp_path / "out" / "final"
    final.mkdir()
    recorder.on_save(4, final, final=True)
    recorder.close()
    assert saved == [("adapter_state.pt", 4), ("adapter_state.pt", 4)]
    assert [(c[0], c[1], c[2], c[3]) for c in hub.calls] == [(4, "checkpoint-4", False, "rl_step_0000004"),
                                                              (4, "final", True, "rl_final_0000004")]
    assert {"metrics.json", "report.md", "reward.png", "reward_components.png", "hacking_watch.png"} <= set(hub.calls[0][4])
    report = json.loads((tmp_path / "out" / "reports" / "rl_step_0000004" / "metrics.json").read_text())
    assert report["kind"] == "rl" and report["lineage"][0]["path"] == "sft.pt" and "hub_token" not in report["run"]
    assert (tmp_path / "out" / "reports" / "index.md").exists() and hub.closed
    from train.ckpt_report import load_history
    history = load_history(tmp_path / "out" / "metrics.jsonl")
    assert "train/reward" in history and "reward/content" in history and "train/note" not in history


def test_a_failing_upload_or_report_never_stops_the_run(tmp_path):
    from train.rl import RLCheckpointRecorder
    checkpoint = tmp_path / "c"
    checkpoint.mkdir()
    for hub, reports in ((FakeHub(fail=True), True), (None, True)):
        recorder = RLCheckpointRecorder(tmp_path / "o", hub=hub, reports=reports,
                                        save_adapter=lambda path, step: (_ for _ in ()).throw(OSError("disk")))
        recorder.on_log(1, {"reward": 1.0})
        recorder.on_save(1, checkpoint)                      # must not raise
        recorder.close()


def test_the_trainer_callback_feeds_the_recorder_on_the_main_process_only(tmp_path):
    import types

    from train.rl import make_trainer_callback

    class Spy:
        def __init__(self):
            self.logs, self.saves = [], []

        def on_log(self, step, logs):
            self.logs.append((step, logs))

        def on_save(self, step, folder, final=False):
            self.saves.append((step, Path(folder).name))
    spy = Spy()
    callback = make_trainer_callback(spy)
    args = types.SimpleNamespace(output_dir=str(tmp_path))
    main, other = types.SimpleNamespace(global_step=7, is_world_process_zero=True), types.SimpleNamespace(
        global_step=7, is_world_process_zero=False)
    callback.on_log(args, main, None, logs={"reward": 1.0})
    callback.on_save(args, main, None)
    callback.on_log(args, other, None, logs={"reward": 2.0})
    callback.on_save(args, other, None)
    assert spy.logs == [(7, {"reward": 1.0})] and spy.saves == [(7, "checkpoint-7")]


def test_load_wrapper_merges_the_earlier_adapter_and_attaches_a_new_lora(monkeypatch, tmp_path):
    import copy

    import torch

    from tests.test_continue_lora import Inner, X, _randomise_adapter, _wrapper
    from train import rl
    from train.model import DeepSeekOCR

    torch.manual_seed(0)
    base = Inner()
    sft = _wrapper(base)
    sft.add_lora(r=2, alpha=4, dropout=0.0, target_modules=["proj"])
    _randomise_adapter(sft)
    ckpt = tmp_path / "sft.pt"
    sft.save_checkpoint(ckpt, step=9, model_state=sft.trainable_state_dict())
    with torch.no_grad():
        expected = sft(X)
    monkeypatch.setattr(DeepSeekOCR, "from_pretrained", classmethod(lambda cls, *a, **k: cls(copy.deepcopy(base))))
    args = rl.parse_args(["train", "--data", "d", "--sft-checkpoint", str(ckpt), "--lora-r", "8", "--lora-alpha", "16",
                          "--lora-targets", ".*proj"])
    wrapper = rl.load_wrapper(args, torch.float32)
    with torch.no_grad():
        assert torch.allclose(wrapper(X), expected, atol=1e-5)          # starts exactly at the SFT model
    assert wrapper.lora_config_dict()["r"] == 8 and len(wrapper.lineage) == 1
    same = rl.parse_args(["train", "--data", "d", "--sft-checkpoint", str(ckpt), "--lora-r", "2", "--lora-alpha", "4",
                          "--lora-targets", ".*proj", "--lora-dropout", "0", "--same-adapter"])
    kept = rl.load_wrapper(same, torch.float32)
    with torch.no_grad():
        assert torch.allclose(kept(X), expected, atol=1e-5)
    assert kept.lineage == [] and kept.lora_config_dict()["r"] == 2


def test_the_rl_commands_default_to_per_save_folders_and_an_rl_prefix():
    from train import rl
    args = rl.parse_args(["train", "--data", "d"])
    assert args.state_prefix is None and args.reports is True and args.hub_every_save is True and args.same_adapter is False
    assert rl.parse_args(["check", "--data", "d", "--lora-r", "4"]).lora_r == 4


def test_epochs_replace_the_step_limit():
    from train import rl
    assert rl.trl_max_steps(rl.parse_args(["train", "--data", "d", "--max-steps", "30"])) == 30
    five = rl.parse_args(["train", "--data", "d", "--epochs", "5"])
    assert five.epochs == 5 and rl.trl_max_steps(five) == -1
