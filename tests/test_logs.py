import logging
import pytest
import subprocess
import sys
import textwrap

from corpus.logs import setup_logging


def _ours():
    return [h for h in logging.getLogger().handlers if getattr(h, "_corpus_logging", False)]


def test_setup_logging_is_idempotent_and_sets_the_level(tmp_path):
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    try:
        setup_logging("INFO")
        setup_logging("DEBUG")
        assert len(_ours()) == 1 and root.level == logging.DEBUG
        setup_logging("INFO", str(tmp_path / "run.log"))
        setup_logging("INFO", str(tmp_path / "run.log"))
        assert len(_ours()) == 2
        logging.getLogger("progress").info("hello file")
        assert "hello file" in (tmp_path / "run.log").read_text()
    finally:
        root.handlers[:] = before
        root.setLevel(level)


def test_progress_lines_reach_a_pipe_at_once_without_unbuffered_python():
    code = textwrap.dedent("""
        import time
        from corpus.logs import setup_logging
        from corpus.progress import Progress
        setup_logging()
        bar = Progress(total=4, desc="job", unit="doc")
        print("RESULT")
        time.sleep(30)
    """)
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = proc.stderr.readline()          # arrives while the child is still sleeping: not buffered
        assert "job: starting (4 doc)" in line and line[2] == ":"      # carries a HH:MM:SS timestamp
    finally:
        proc.kill()
        proc.wait()


def test_chatty_library_lines_are_dropped_unless_debug(tmp_path):
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    try:
        setup_logging("INFO", str(tmp_path / "q.log"))
        logging.getLogger("httpx").setLevel(logging.INFO)         # a library resetting its own level
        logging.getLogger("httpx").info("HTTP Request: GET https://example")
        logging.getLogger("httpx2").info("HTTP Request: GET https://hub")       # huggingface_hub's own name
        logging.getLogger("labeller").info("kept line")
        text = (tmp_path / "q.log").read_text()
        assert "kept line" in text and "HTTP Request" not in text
    finally:
        root.handlers[:] = before
        root.setLevel(level)


def test_progress_modes_choose_between_log_lines_and_plain_bars(caplog):
    import logging as _logging
    from corpus import progress
    from corpus.progress import Progress, set_mode

    caplog.set_level(_logging.INFO)
    try:
        set_mode("lines")
        assert Progress(total=4, desc="a").bar is None                       # no terminal: log lines
        set_mode("bars")
        bar = Progress(total=4, desc="b", position=2, leave=False)
        assert bar.bar is not None and bar.bar.pos == 0 and bar.bar.leave   # flat bar, kept on screen
        bar.close()
        with pytest.raises(ValueError):
            set_mode("sometimes")
    finally:
        set_mode("auto")


def test_a_plain_tqdm_bar_counts_each_update_once():
    from corpus.progress import Progress, set_mode

    try:
        set_mode("bars")
        bar = Progress(total=8, desc="sft", unit="step")
        for _ in range(8):
            bar.update(1)
        assert bar.n == 8 and bar.bar.n == 8                  # it used to read 16
        bar.close()
    finally:
        set_mode("auto")


def test_a_progress_total_that_was_an_estimate_grows_instead_of_passing_100_percent(caplog):
    import logging as _logging
    from corpus.progress import Progress, set_mode

    caplog.set_level(_logging.INFO)
    set_mode("lines")
    bar = Progress(total=4, desc="est", unit="step")
    for _ in range(6):
        bar.update(1)
    bar.close()
    assert bar.total == 6
    assert not any("(1" in r.getMessage() and "%" in r.getMessage() and "(150%" in r.getMessage() for r in caplog.records)
    assert "6/6 step (100%)" in caplog.records[-1].getMessage()


def test_under_torchrun_only_rank_zero_prints_info_lines(monkeypatch, capsys):
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    try:
        root.handlers[:] = []
        monkeypatch.setenv("RANK", "1")
        setup_logging("INFO")
        logging.getLogger("train.sft").info("only rank 0 says this")
        logging.getLogger("train.sft").warning("every rank says this")
        err = capsys.readouterr().err
        assert "only rank 0" not in err and "every rank says this" in err
    finally:
        root.handlers[:] = before
        root.setLevel(level)


def test_other_ranks_get_no_download_bars_or_repeated_warnings(monkeypatch):
    import warnings
    from huggingface_hub.utils import are_progress_bars_disabled, enable_progress_bars
    from transformers.utils import logging as hf_logging

    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    saved = list(warnings.filters)
    try:
        root.handlers[:] = []
        enable_progress_bars()
        hf_logging.enable_progress_bar()
        monkeypatch.setenv("RANK", "1")
        setup_logging("INFO")
        assert are_progress_bars_disabled() and not hf_logging.is_progress_bar_enabled()
    finally:
        root.handlers[:] = before
        root.setLevel(level)
        warnings.filters[:] = saved
        enable_progress_bars()
        hf_logging.enable_progress_bar()
