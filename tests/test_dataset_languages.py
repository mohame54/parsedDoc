import json

from train.dataset import DocumentDataset


def test_exclude_languages(tmp_path):
    rows = [{"id": str(i), "language": lang} for i, lang in enumerate(["de", "en", None, "DE", "fr"])]
    (tmp_path / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    ds = DocumentDataset(tmp_path, exclude_languages={"de"})
    assert [r["id"] for r in ds.records] == ["1", "2", "4"]
