from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import requests
from PIL import Image
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


def download_huggingface(source: dict[str, Any], destination: Path,
                         max_documents: int) -> list[Path]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install optional dependency: pip install datasets huggingface_hub") from exc

    ds = load_dataset(source["dataset_id"], split=source.get("split", "train"), streaming=True)
    image_col = source.get("image_column", "image")
    out = destination / "images"
    out.mkdir(parents=True, exist_ok=True)
    result = []
    for idx, row in enumerate(ds):
        if idx >= max_documents:
            break
        image = row.get(image_col)
        if image is None:
            continue
        if isinstance(image, dict) and image.get("path"):
            src = Path(image["path"])
            if src.exists():
                target = out / f"{idx:08d}{src.suffix.lower() or '.png'}"
                shutil.copy2(src, target)
                result.append(target)
                continue
        pil = image if isinstance(image, Image.Image) else None
        if pil is None and isinstance(image, dict) and image.get("bytes"):
            import io
            pil = Image.open(io.BytesIO(image["bytes"]))
        if pil is None:
            continue
        target = out / f"{idx:08d}.png"
        pil.convert("RGB").save(target)
        result.append(target)
    return result
