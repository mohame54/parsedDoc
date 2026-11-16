"""File-backed state. One folder per document under annotations/, one per batch under batches/.

Nothing here talks to Gemini. Resuming is just reading status.json files back.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_name(doc_id: str) -> str:
    return doc_id.replace(":", "__").replace("/", "_")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class Store:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.root: Path = cfg.data_root

    def ann_dir(self, doc_id: str) -> Path:
        return self.root / "annotations" / safe_name(doc_id)

    def status(self, doc_id: str) -> dict | None:
        return read_json(self.ann_dir(doc_id) / "status.json")

    def set_status(self, doc_id: str, state: str, **fields: Any) -> dict:
        current = self.status(doc_id) or {"doc_id": doc_id}
        current.update(fields)
        current["state"] = state
        current["updated_at"] = utc_now()
        write_json(self.ann_dir(doc_id) / "status.json", current)
        return current

    def all_statuses(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        base = self.root / "annotations"
        if base.exists():
            for path in base.glob("*/status.json"):
                st = read_json(path)
                if st:
                    out[st["doc_id"]] = st
        return out

    def resolve(self, rel: str) -> Path:
        return self.root / rel

    def batch_dir(self, stage: int, runkey: str) -> Path:
        return self.root / "batches" / f"stage{stage}-{runkey}"

    def batch_dirs(self, stage: int) -> list[Path]:
        base = self.root / "batches"
        return sorted(base.glob(f"stage{stage}-*")) if base.exists() else []
