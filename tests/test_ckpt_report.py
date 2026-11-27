"""Per-checkpoint reports: the right metrics for SFT and for RL, graphs written, and the comparison across checkpoints."""
import json
import math
from pathlib import Path

import pytest

from train.ckpt_report import build_checkpoint_report, load_history, summarise, write_index
from train.metrics import MetricLogger


def sft_history(steps=60):
    h = {"train/loss": [(s, 2.0 * math.exp(-s / 30) + 0.2) for s in range(1, steps + 1)],
         "train/lr": [(s, 1e-4 * (1 - s / 100)) for s in range(1, steps + 1)],
         "train/grad_norm": [(s, 1.0 / s ** 0.3) for s in range(1, steps + 1)],
         "val/loss": [(0, 2.5), (20, 1.4), (40, 0.9), (60, 0.95)],
         "gen/score": [(0, 0.1), (20, 0.4), (40, 0.7), (60, 0.66)],
         "gen/json_valid": [(0, 0.5), (20, 0.9), (40, 1.0), (60, 1.0)],
         "gen/extract_to_schema": [(0, 0.1), (20, 0.5), (40, 0.8), (60, 0.7)],
         "gen_hallucination/extract_to_schema": [(20, 0.3), (40, 0.1), (60, 0.12)]}
    return h


def rl_history(steps=40):
    return {"train/reward": [(s, 0.3 + 0.01 * s) for s in range(1, steps + 1)],
            "train/reward_std": [(s, 0.2) for s in range(1, steps + 1)],
            "train/kl": [(s, 0.001 * s) for s in range(1, steps + 1)],
            "train/learning_rate": [(s, 5e-6) for s in range(1, steps + 1)],
            "reward/content": [(s, 0.5 + 0.01 * s) for s in range(1, steps + 1)],
            "behaviour/abstain_rate_must_abstain": [(s, 0.2 + 0.01 * s) for s in range(1, steps + 1)],
            "behaviour/abstain_rate_answerable": [(s, 0.05) for s in range(1, steps + 1)],
            "behaviour/wrong_rate_must_abstain": [(s, 0.6 - 0.01 * s) for s in range(1, steps + 1)],
            "calibration/ece": [(s, 0.3 - 0.005 * s) for s in range(1, steps + 1)],
            "calibration/auroc": [(s, 0.6 + 0.005 * s) for s in range(1, steps + 1)]}


def test_an_sft_report_has_loss_generation_and_optimisation_graphs(tmp_path):
    out = build_checkpoint_report(sft_history(), 40, tmp_path / "step_0000040", kind="sft",
                                  run_config={"lr": 1e-4, "hub_token": "secret", "lora_r": 16}, extra={"epoch": 1})
    for name in ("metrics.json", "report.md", "loss.png", "generation.png", "optimization.png"):
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    assert not (out / "reward.png").exists()                                # an SFT report has no RL graphs
    metrics = json.loads((out / "metrics.json").read_text())
    h = metrics["highlights"]
    assert h["val_loss"] == 0.9 and h["val_loss_measured_at"] == 40 and h["is_best_val"] is True
    assert h["best_val_step"] == 40 and h["gen_score"] == 0.7
    assert metrics["run"] == {"lr": 1e-4, "lora_r": 16}                     # the token is never written
    report = (out / "report.md").read_text()
    assert "SFT checkpoint" in report and "this checkpoint" in report and "loss.png" in report


def test_only_the_history_up_to_the_checkpoint_is_used(tmp_path):
    early = summarise(sft_history(), 20, "sft")["highlights"]
    assert early["val_loss"] == 1.4 and early["gen_score"] == 0.4 and early["best_val_step"] == 20
    late = summarise(sft_history(), 60, "sft")["highlights"]
    assert late["is_best_val"] is False and late["best_val_step"] == 40      # step 60 got worse: not the best


def test_an_rl_report_has_reward_behaviour_and_calibration_graphs(tmp_path):
    out = build_checkpoint_report(rl_history(), 30, tmp_path / "rl_step_0000030", kind="rl")
    for name in ("reward.png", "behaviour.png", "calibration.png", "optimization.png"):
        assert (out / name).exists(), name
    assert not (out / "loss.png").exists()
    h = json.loads((out / "metrics.json").read_text())["highlights"]
    assert h["hallucination"] == pytest.approx(sum(0.6 - 0.01 * s for s in range(11, 31)) / 20)
    assert {"reward", "ece", "auroc", "abstain_when_unreadable"} <= set(h)
    assert "RL checkpoint" in (out / "report.md").read_text()


def test_missing_series_are_skipped_not_errors(tmp_path):
    out = build_checkpoint_report({"train/loss": [(1, 1.0), (2, 0.9)]}, 2, tmp_path / "x", kind="sft")
    assert (out / "loss.png").exists() and not (out / "generation.png").exists()
    assert build_checkpoint_report({}, 0, tmp_path / "y", kind="rl").joinpath("report.md").exists()


def test_the_index_ranks_checkpoints_and_marks_the_best_per_kind(tmp_path):
    root = tmp_path / "reports"
    for step in (20, 40, 60):
        build_checkpoint_report(sft_history(), step, root / f"sft_step_{step:07d}", kind="sft")
    build_checkpoint_report(rl_history(), 30, root / "rl_step_0000030", kind="rl")
    index = write_index(root)
    text = index.read_text()
    assert "sft_step_0000040](sft_step_0000040/report.md) ★" in text           # best val loss
    assert "rl_step_0000030](rl_step_0000030/report.md) ★" in text             # the only RL one is its own best
    sft_table, rl_table = text.split("## SFT")[1], text.split("## RL")[1].split("## SFT")[0]
    assert "reward" not in sft_table and "ece" not in sft_table               # each kind's table has its own columns
    assert "val_loss" not in rl_table and "gen_score" not in rl_table
    assert (root / "compare.png").exists() and (root / "checkpoints.csv").read_text().count("\n") == 5
    assert write_index(tmp_path / "empty") is None


def test_the_logger_keeps_a_history_that_survives_a_resume(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = MetricLogger(None, is_main=True, log_every=1, tensorboard=False, history_path=path)
    for step in (1, 2, 3):
        logger.log(step, {"loss": 1.0 / step})
    logger.scalars(2, {"val/loss": 0.5})
    logger.close()
    assert load_history(path)["train/loss"] == [(1, 1.0), (2, 0.5), (3, 1 / 3)]
    resumed = MetricLogger(None, is_main=True, log_every=1, tensorboard=False, history_path=path, initial_step=2)
    assert resumed.history["train/loss"] == [(1, 1.0), (2, 0.5)]                # step 3 is being redone
    resumed.log(3, {"loss": 0.1})
    resumed.close()
    assert load_history(path)["train/loss"][-1] == (3, 0.1)
    other_rank = MetricLogger(None, is_main=False, tensorboard=False, history_path=tmp_path / "no.jsonl")
    other_rank.log(1, {"loss": 1.0})
    assert not (tmp_path / "no.jsonl").exists()


def hack_history(steps=60, accuracy_slope=-0.004, abstain_slope=0.006, reward_slope=0.01):
    """The reward climbs; what it should go with does or does not, depending on the slopes."""
    rng = range(1, steps + 1)
    return {"train/reward": [(s, 0.3 + reward_slope * s) for s in rng],
            "train/reward_std": [(s, 0.1) for s in rng],
            "reward/content": [(s, 0.5 + 0.005 * s) for s in rng],
            "reward/calibration": [(s, 0.6 + 0.002 * s) for s in rng],
            "reward/schema": [(s, 0.95) for s in rng],
            "reward/rotation": [(s, 0.4 + 0.004 * s) for s in rng],
            "reward/reason": [(s, 0.5) for s in rng],
            "reward/valid_json": [(s, 0.98) for s in rng],
            "quality/exact_accuracy_answerable": [(s, 0.8 + accuracy_slope * s) for s in rng],
            "behaviour/abstain_rate_answerable": [(s, 0.05 + abstain_slope * s) for s in rng],
            "behaviour/wrong_rate_must_abstain": [(s, 0.4) for s in rng],
            "train/completion_length": [(s, 300.0) for s in rng]}


def test_rl_reports_have_a_panel_per_reward_component_and_a_graph_for_the_final_reward(tmp_path):
    out = build_checkpoint_report(hack_history(), 60, tmp_path / "rl_step_0000060", kind="rl")
    for name in ("reward.png", "reward_components.png", "hacking_watch.png", "behaviour.png"):
        assert (out / name).exists(), name
    from train.rl_reward import WEIGHTS

    import matplotlib.image as mpimg
    components = mpimg.imread(out / "reward_components.png")
    total = mpimg.imread(out / "reward.png")
    assert components.shape[0] > 2.5 * total.shape[0]              # one tall figure: several panels vs a single one
    assert set(WEIGHTS) == {"content", "calibration", "schema", "rotation", "reason"}


def test_a_rising_reward_with_falling_accuracy_and_more_nulls_is_flagged(tmp_path):
    from train.ckpt_report import reward_hacking_flags
    flags = reward_hacking_flags(hack_history(), 60)
    assert any("exact accuracy" in f for f in flags) and any("nulls on readable fields" in f for f in flags)
    out = build_checkpoint_report(hack_history(), 60, tmp_path / "x", kind="rl")
    assert "**Warnings**" in (out / "report.md").read_text()
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["flags"] and metrics["highlights"]["hacking_flags"] == len(metrics["flags"])


def test_no_flag_when_the_quality_measures_rise_with_the_reward(tmp_path):
    from train.ckpt_report import reward_hacking_flags
    healthy = hack_history(accuracy_slope=0.002, abstain_slope=-0.0005)
    assert reward_hacking_flags(healthy, 60) == []
    out = build_checkpoint_report(healthy, 60, tmp_path / "ok", kind="rl")
    assert "No warning" in (out / "report.md").read_text()


def test_the_watch_only_judges_a_rising_reward_and_needs_enough_points():
    from train.ckpt_report import reward_hacking_flags
    assert reward_hacking_flags(hack_history(reward_slope=-0.002), 60) == []      # a falling reward is not being gamed
    assert reward_hacking_flags(hack_history(reward_slope=0.0), 60) == []
    assert reward_hacking_flags(hack_history(steps=6), 6) == []                    # too little history to say
    shrinking = hack_history(accuracy_slope=0.0, abstain_slope=0.0)
    shrinking["train/completion_length"] = [(s, 300.0 if s <= 30 else 80.0) for s in range(1, 61)]
    assert any("shrank" in f for f in reward_hacking_flags(shrinking, 60))


def test_the_index_does_not_star_a_flagged_checkpoint_while_a_clean_one_exists(tmp_path):
    root = tmp_path / "reports"
    build_checkpoint_report(hack_history(), 60, root / "rl_step_0000060", kind="rl")                  # highest reward, flagged
    healthy = hack_history(accuracy_slope=0.002, abstain_slope=-0.0005, reward_slope=0.006)
    build_checkpoint_report(healthy, 40, root / "rl_step_0000040", kind="rl")                          # lower reward, clean
    text = write_index(root).read_text()
    assert "rl_step_0000040](rl_step_0000040/report.md) ★" in text and "rl_step_0000060](rl_step_0000060/report.md) ★" not in text
    assert "hacking_flags" in text


def test_exact_accuracy_is_measured_without_partial_credit():
    """A near miss earns some reward through partial credit; the exact-accuracy series must not count it."""
    from train.rl_reward import history_metrics, score_rl
    reference = json.dumps({"name": "Ana Lopez", "total": 12.5})
    near = json.dumps({"name": "Ana Lopes", "total": 12.5})
    exact = json.dumps({"name": "Ana Lopez", "total": 12.5})
    rows_near = [score_rl(near, reference, None)]
    rows_exact = [score_rl(exact, reference, None)]
    assert history_metrics(rows_exact)["quality/exact_accuracy_answerable"] == 1.0
    assert history_metrics(rows_near)["quality/exact_accuracy_answerable"] == 0.5          # 1 of 2 fields exactly right
    assert 0 < rows_near[0]["parts"]["content"] < rows_exact[0]["parts"]["content"]        # but the reward gave partial credit
    metrics = history_metrics(rows_exact + [score_rl("not json", reference, None)])
    assert metrics["reward/valid_json"] == 0.5 and "reward/content" in metrics
