"""Hugging Face loader (config name, row filter, caps) and the progress reporting of the download and train steps."""
import io
import logging
import types

import pytest
from PIL import Image

from corpus import datasets as corpus_datasets
from corpus.utils import Budget, safe_extract_zip
from train.metrics import MetricLogger


def _jpeg(color=(200, 10, 10), size=(40, 40)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, "JPEG")
    return buffer.getvalue()


class FakeStream:
    def __init__(self, rows):
        self.rows = rows

    def cast_column(self, name, feature):
        return self

    def __iter__(self):
        return iter(self.rows)


@pytest.fixture
def fake_hub(monkeypatch):
    calls = {}
    rows = [{"image": {"bytes": _jpeg(), "path": None}, "form_id": "a" if i % 2 else "b"} for i in range(10)]

    def load_dataset(name, config=None, split=None, streaming=None):
        calls.update(name=name, config=config, split=split, streaming=streaming)
        return FakeStream(rows)

    monkeypatch.setitem(__import__("sys").modules, "datasets", types.SimpleNamespace(
        load_dataset=load_dataset, Image=lambda decode: None))
    return calls


def test_huggingface_passes_the_config_name_and_keeps_original_jpeg_bytes(tmp_path, fake_hub):
    source = {"id": "x", "dataset_id": "o/d", "config_name": "French", "split": "train"}
    files = corpus_datasets.download_huggingface(source, tmp_path, max_documents=3)
    assert fake_hub["config"] == "French"                       # was silently dropped before
    assert len(files) == 3 and all(f.suffix == ".jpg" for f in files)
    assert files[0].read_bytes() == _jpeg()                     # not re-encoded to a larger PNG


def test_huggingface_applies_the_row_filter_and_counts_only_kept_rows(tmp_path, fake_hub):
    source = {"id": "x", "dataset_id": "o/d", "form_id_filter": "a"}
    files = corpus_datasets.download_huggingface(source, tmp_path, max_documents=100)
    assert len(files) == 5                                      # half of the 10 rows
    with pytest.raises(RuntimeError, match="filter column"):
        corpus_datasets.download_huggingface(dict(source, filter_column="missing"), tmp_path / "y", 5)


def test_huggingface_stops_at_the_source_byte_cap_and_the_total_budget(tmp_path, fake_hub):
    size = len(_jpeg())
    source = {"id": "x", "dataset_id": "o/d"}
    messages = []
    files = corpus_datasets.download_huggingface(source, tmp_path / "a", 100, max_bytes=size * 3 + 1,
                                                 out=messages.append)
    assert len(files) == 3 and any("source cap" in m for m in messages)
    budget = Budget(tmp_path / "b", max_total_gb=(size * 2 + 1) / 1024**3)
    with pytest.raises(RuntimeError, match="budget"):
        corpus_datasets.download_huggingface(source, tmp_path / "b", 100, budget=budget)


def test_zip_extraction_reports_progress_and_still_blocks_path_traversal(tmp_path, capsys):
    import zipfile

    good = tmp_path / "ok.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("a/one.txt", "hello")
    safe_extract_zip(good, tmp_path / "out")
    assert (tmp_path / "out/a/one.txt").read_text() == "hello"
    assert "extract ok.zip" in capsys.readouterr().out         # no terminal here: a progress line is printed

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract_zip(bad, tmp_path / "out2")
    assert not (tmp_path / "evil.txt").exists()


def test_training_log_lines_show_step_total_percent_and_eta(caplog):
    logger = MetricLogger(log_dir=None, is_main=True, log_every=10, total_steps=100, tensorboard=False)
    with caplog.at_level(logging.INFO, logger="train.sft"):
        logger.log(10, {"loss": 1.5, "grad_norm": 0.7})
        logger.log(20, {"loss": 1.2, "grad_norm": 0.6})
    logger.close()
    assert "step 10/100 (10%)" in caplog.text and "step 20/100 (20%)" in caplog.text and "eta" in caplog.text
    assert "loss=1.2" in caplog.text and "grad_norm=0.6" in caplog.text


def test_progress_prints_plain_lines_when_no_terminal_is_attached(capsys):
    from corpus.progress import Progress

    bar = Progress(total=100, desc="work", unit="doc")
    for _ in range(100):
        bar.update(1)
    bar.close()
    out = capsys.readouterr().out
    assert "work: 11/100" not in out                     # one line per tenth of the work, not per item
    assert "work: 10/100 doc (10%)" in out and "work: 100/100 doc (100%)" in out

    items = list(Progress(["a", "b", "c"], desc="items", unit="it"))
    assert items == ["a", "b", "c"]                       # iterating counts the items and keeps them
    quiet = Progress(total=4, desc="silent", disable=True)
    quiet.update(4)
    assert "silent" not in capsys.readouterr().out


def test_progress_bar_counts_can_be_set_directly_like_the_labeller_does(capsys):
    from corpus.progress import Progress

    bar = Progress(total=6, desc="stage", unit="doc")
    bar.n = 4
    bar.set_postfix_str("running")
    bar.refresh()
    assert bar.n == 4
    bar.close()
    assert "stage: 4/6" in capsys.readouterr().out


def test_progress_without_a_terminal_announces_itself_and_never_goes_quiet_for_long(capsys, monkeypatch):
    from corpus import progress as progress_module
    from corpus.progress import Progress

    clock = {"t": 0.0}
    monkeypatch.setattr(progress_module.time, "monotonic", lambda: clock["t"])
    bar = Progress(total=1000, desc="slow", unit="doc")           # a tenth of the work is 100 documents
    assert "slow: starting (1000 doc)" in capsys.readouterr().out    # visible before any item finishes
    clock["t"] = 3.0
    bar.update(5)
    assert capsys.readouterr().out == ""                           # neither a tenth nor ten seconds yet
    clock["t"] = 11.0
    bar.update(5)
    out = capsys.readouterr().out
    assert "slow: 10/1000 doc (1%)" in out and "doc/s" in out and "eta" in out and "0:00:11" in out
    clock["t"] = 12.0
    bar.update(1)
    assert capsys.readouterr().out == ""                           # the next line waits for the interval again


def test_a_polled_bar_reports_through_refresh_even_when_only_its_count_and_text_change(capsys, monkeypatch):
    """The Gemini batch wait sets `n`, a status text and calls refresh(); it never calls update()."""
    from corpus import progress as progress_module
    from corpus.progress import Progress

    clock = {"t": 0.0}
    monkeypatch.setattr(progress_module.time, "monotonic", lambda: clock["t"])
    bar = Progress(total=2, desc="Stage 2 batch", unit="doc", leave=True)
    capsys.readouterr()
    clock["t"] = 4.0
    bar.set_postfix_str("running")
    bar.refresh()
    assert capsys.readouterr().out == ""                           # polled again too soon: no line
    clock["t"] = 31.0
    bar.set_postfix_str("running · 0:00:31")
    bar.refresh()
    assert "Stage 2 batch: 0/2 doc (0%)" in capsys.readouterr().out
    bar.n = 2
    clock["t"] = 32.0
    bar.refresh()
    assert "Stage 2 batch: 2/2 doc (100%)" in capsys.readouterr().out
