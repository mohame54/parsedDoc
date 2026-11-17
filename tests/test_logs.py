import logging
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
