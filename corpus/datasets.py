from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from tqdm import tqdm

from .labels import entities_from_row, parse_answers, write_qa_sidecar, write_sidecar
from .utils import Budget, download_stream


def download_zenodo(record_id: str, destination_dir: Path, budget: Budget,
                    max_bytes: int) -> list[Path]:
    api = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(api, timeout=30)
    response.raise_for_status()
    record = response.json()
    files = record.get("files", [])
    if not files:
        raise RuntimeError(f"No public files listed for Zenodo record {record_id}")
    downloaded, total = [], 0
    destination_dir.mkdir(parents=True, exist_ok=True)
    for item in files:
        key = item.get("key") or item.get("filename")
        size = int(item.get("size", 0))
        if size and total + size > max_bytes:
            print(f"[skip] {key}: source download cap reached")
            continue
        links = item.get("links", {})
        url = links.get("self") or links.get("download")
        if not url:
            continue
        target = destination_dir / Path(key).name
        if not target.exists():
            download_stream(url, target, budget, max_bytes - total)
        total += target.stat().st_size
        downloaded.append(target)
    return downloaded


_SUFFIX_BY_FORMAT = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif", "BMP": ".bmp", "WEBP": ".webp", "GIF": ".gif"}


def _image_bytes(image: Any, fallback_suffix: str = ".png") -> tuple[bytes, str] | None:
    """(file bytes, suffix) for a dataset image: the original encoded bytes when available (a re-encode to PNG
    inflates JPEGs several times and would break the byte budget), else a PNG encoded from the PIL image."""
    if isinstance(image, dict):
        data = image.get("bytes")
        path = image.get("path")
        if data:
            try:
                with Image.open(io.BytesIO(data)) as probe:
                    return data, _SUFFIX_BY_FORMAT.get(probe.format or "", fallback_suffix)
            except Exception:
                return None
        if path and Path(path).exists():
            return Path(path).read_bytes(), Path(path).suffix.lower() or fallback_suffix
        return None
    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "PNG")
        return buffer.getvalue(), ".png"
    return None


def _open_stream(source: dict[str, Any]):
    """The dataset as a stream of rows, with images left as raw bytes (no decode, no re-encode)."""
    try:
        from datasets import Image as ImageFeature
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install optional dependency: pip install datasets huggingface_hub") from exc
    ds = load_dataset(source["dataset_id"], source.get("config_name"), split=source.get("split", "train"),
                      streaming=True)
    try:
        ds = ds.cast_column(source.get("image_column", "image"), ImageFeature(decode=False))
    except Exception:
        pass
    return ds


def download_grouped_qa(source: dict[str, Any], destination: Path, max_documents: int,
                        budget: Budget | None = None, max_bytes: int | None = None,
                        out=tqdm.write) -> list[Path]:
    """Datasets with one row per question (ChartQA): each distinct image is saved once, and every question about
    it goes into `<stem>.qa.json`. Images are de-duplicated by the hash of their bytes, so a chart seen again on a
    later row adds its question to the existing file instead of a new document. `max_documents` counts images."""
    ds = _open_stream(source)
    image_col = source.get("image_column", "image")
    columns = source.get("qa_columns") or {"question": "query", "answers": "label"}
    out_dir = destination / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_digest: dict[str, dict[str, Any]] = {}      # digest -> {"path": Path, "qa": [...]}
    written = scanned = 0
    bar = tqdm(total=max_documents, desc=f"{source['id']} download", unit="img", leave=False)
    try:
        for idx, row in enumerate(ds):
            scanned += 1
            encoded = _image_bytes(row.get(image_col))
            if encoded is None:
                continue
            data, suffix = encoded
            digest = hashlib.sha256(data).hexdigest()
            question = str(row.get(columns["question"], "") or "").strip()
            pair = {"question": question, "answers": parse_answers(row.get(columns["answers"])),
                    "annotator": row.get("human_or_machine")}
            if digest in by_digest:
                if question:
                    by_digest[digest]["qa"].append(pair)
                continue
            if len(by_digest) >= max_documents:
                break
            if max_bytes is not None and written + len(data) > max_bytes:
                out(f"[huggingface] {source['id']}: stopped at the {max_bytes / 1e9:.2f} GB source cap "
                    f"after {len(by_digest)} images")
                break
            if budget is not None:
                budget.check(len(data))
            target = out_dir / f"{idx:08d}{suffix}"
            target.write_bytes(data)
            written += len(data)
            by_digest[digest] = {"path": target, "qa": [pair] if question else []}
            bar.update(1)
            bar.set_postfix(MB=round(written / 1e6, 1), rows=scanned, refresh=False)
    finally:
        bar.close()
    result = []
    for entry in by_digest.values():
        write_qa_sidecar(entry["path"], entry["qa"], source["id"])
        result.append(entry["path"])
    return result


def download_huggingface(source: dict[str, Any], destination: Path, max_documents: int,
                         budget: Budget | None = None, max_bytes: int | None = None,
                         out=tqdm.write) -> list[Path]:
    """Streams images from a Hugging Face dataset. Honours `config_name`, an optional row filter
    (`form_id_filter` on column `filter_column`, default "form_id"), the document cap and the byte cap."""
    try:
        from datasets import Image as ImageFeature
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install optional dependency: pip install datasets huggingface_hub") from exc

    ds = _open_stream(source)
    image_col = source.get("image_column", "image")
    labels_spec = source.get("labels")        # where the dataset keeps its own text, if it has any
    filter_value = source.get("form_id_filter")
    filter_column = source.get("filter_column", "form_id")
    out_dir = destination / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    result: list[Path] = []
    written = scanned = 0
    bar = tqdm(total=max_documents, desc=f"{source['id']} download", unit="doc", leave=False)
    try:
        for idx, row in enumerate(ds):
            if len(result) >= max_documents:
                break
            scanned += 1
            if filter_value is not None:
                if filter_column not in row:
                    raise RuntimeError(f"{source['id']}: filter column {filter_column!r} not in {sorted(row)}")
                if row[filter_column] != filter_value:
                    bar.set_postfix(scanned=scanned, refresh=False)   # filtered-out rows still cost time
                    continue
            encoded = _image_bytes(row.get(image_col))
            if encoded is None:
                continue
            data, suffix = encoded
            if max_bytes is not None and written + len(data) > max_bytes:
                out(f"[huggingface] {source['id']}: stopped at the {max_bytes / 1e9:.2f} GB source cap "
                    f"after {len(result)} documents")
                break
            if budget is not None:
                budget.check(len(data))
            target = out_dir / f"{idx:08d}{suffix}"
            target.write_bytes(data)
            written += len(data)
            if labels_spec:
                write_sidecar(target, entities_from_row(labels_spec, row), source["id"])
            result.append(target)
            bar.update(1)
            bar.set_postfix(MB=round(written / 1e6, 1), scanned=scanned, refresh=False)
    finally:
        bar.close()
    return result
