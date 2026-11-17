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
    assert "extract ok.zip" in capsys.readouterr().err         # a bar was drawn

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
