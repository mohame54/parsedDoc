"""Reading a training log back: the points, the validation losses, the summary and the plot."""
import pytest

from train.report import load, main, parse_log, plot, summarise

LOG = """\
23:00:00 INFO validation loss 1.2000 at step 0 (30 batches, 20 s)
23:00:10 INFO step 10/60 (17%) eta 0:01:00 loss=1.0 grad_norm=3.7 lr=0.0001 skipped_steps=1 batch_size=1.5 padded_tokens=998.4 peak_mem_gb=9.97 loss_scale=3.277e+04 tokens_per_s=1325
23:00:12 INFO sft: 14/60 step (23%) | 0:00:10 | 1.4 step/s | loss=1.0101, grad_norm=3.7141, lr=0.0001, loss_scale=32768.0 | still working
23:00:20 INFO step 20/60 (33%) eta 0:00:50 loss=0.8 grad_norm=nan lr=0.0001 skipped_steps=2 batch_size=2 padded_tokens=1300 peak_mem_gb=10.5 tokens_per_s=2000
23:00:30 INFO step 30/60 (50%) eta 0:00:30 loss=0.6 grad_norm=2.0 lr=9e-05 skipped_steps=2 batch_size=2 padded_tokens=1300 peak_mem_gb=11.0 tokens_per_s=2100
23:00:35 INFO validation loss 0.9000 at step 30 (30 batches, 21 s)
23:00:40 INFO step 40/60 (66%) eta 0:00:20 loss=0.5 grad_norm=1.5 lr=5e-05 skipped_steps=2 batch_size=2 padded_tokens=1200 peak_mem_gb=10.9 tokens_per_s=2000
23:00:50 INFO step 50/60 (83%) eta 0:00:10 loss=0.4 grad_norm=1.2 lr=2e-05 skipped_steps=2 batch_size=2 padded_tokens=1200 peak_mem_gb=10.2 tokens_per_s=1900
23:01:00 INFO step 60/60 (100%) eta 0:00:00 loss=0.3 grad_norm=1.0 lr=1e-05 skipped_steps=2 batch_size=2 padded_tokens=1100 peak_mem_gb=10.0 tokens_per_s=1900
23:01:05 INFO validation loss 0.8000 at step 60 (30 batches, 21 s)
"""


def test_the_log_is_parsed_into_points_and_validation_losses():
    steps, vals = parse_log(LOG)
    assert [int(s["step"]) for s in steps] == [10, 20, 30, 40, 50, 60]          # the `sft:` bar line is not a point
    assert steps[0]["loss"] == 1.0 and steps[0]["peak_mem_gb"] == 9.97 and steps[0]["loss_scale"] == 32770.0
    assert steps[1]["grad_norm"] != steps[1]["grad_norm"]                       # nan survives as nan
    assert vals == [(0, 1.2), (30, 0.9), (60, 0.8)]


def test_a_resumed_run_keeps_the_last_point_for_each_step():
    text = LOG + "23:05:00 INFO step 60/60 (100%) eta 0:00:00 loss=0.25 grad_norm=1.0 lr=1e-05\n"
    steps, _ = parse_log(text)
    assert [s["loss"] for s in steps if s["step"] == 60] == [0.25]


def test_the_summary_reports_the_change_the_best_validation_and_warnings():
    out = summarise(*parse_log(LOG))
    assert out["last_step"] == 60 and out["val_first"] == 1.2 and out["val_best"] == 0.8 and out["val_best_step"] == 60
    assert out["train_loss_change"].startswith("-")                              # the loss fell
    assert out["skipped_steps"] == 2 and out["peak_mem_gb_max"] == 11.0

    worse = LOG.replace("0.8000 at step 60", "1.1900 at step 60")
    assert any("did not generalise" in n or "above its best" in n for n in summarise(*parse_log(worse))["notes"])


def test_rising_validation_loss_with_falling_training_loss_is_called_overfitting():
    text = LOG.replace("0.9000 at step 30", "0.7000 at step 30").replace("0.8000 at step 60", "0.9500 at step 60")
    text += "23:01:10 INFO validation loss 1.1000 at step 70 (30 batches, 21 s)\n"
    notes = summarise(*parse_log(text))["notes"]
    assert any("overfitting" in n for n in notes) or any("above its best" in n for n in notes)


def test_the_command_writes_a_plot_and_prints_the_summary(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    log = tmp_path / "train.log"
    log.write_text(LOG)
    main([str(log), "--out", str(tmp_path / "r.png")])
    assert (tmp_path / "r.png").stat().st_size > 5000
    assert "val_best" in capsys.readouterr().out
    steps, vals = load(log)
    assert plot(steps, vals, tmp_path / "again.png", smooth=3).exists()


def test_a_log_without_training_lines_says_so(tmp_path):
    log = tmp_path / "empty.log"
    log.write_text("nothing here\n")
    with pytest.raises(SystemExit, match="no training lines"):
        main([str(log)])
