"""Download a slice of a huge remote ZIP without downloading the archive.

A ZIP keeps its table of contents at the end of the file, and each member is stored on its own. With HTTP
Range requests we read the table of contents, choose a deterministic sample of members, and fetch only those.
For a 30 GB archive this means a few megabytes for the listing plus the members actually wanted.

The server must support Range requests (HTTP 206); anything else is refused with a clear error.
"""
from __future__ import annotations

import fnmatch
import io
import random
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import requests

from .utils import Budget

DEFAULT_BLOCK = 256 * 1024
USER_AGENT = "real-document-corpus-builder/0.1"


class RangeNotSupported(RuntimeError):
    pass


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file over an HTTP URL, fetched with Range requests.

    Small reads are served from cached blocks; reads of a block or more go straight to the server, which is
    what the ZIP reader does for its table of contents.
    """

    def __init__(self, url: str, *, session: requests.Session | None = None, block_size: int = DEFAULT_BLOCK,
                 max_cached_blocks: int = 16, timeout: tuple[float, float] = (20, 120), retries: int = 3) -> None:
        super().__init__()
        self.url = url
        self.session = session or requests.Session()
        self.block_size = block_size
        self.max_cached_blocks = max_cached_blocks
        self.timeout = timeout
        self.retries = retries
        self.pos = 0
        self.bytes_fetched = 0
        self.requests_made = 0
        self._blocks: dict[int, bytes] = {}
        self.size = self._probe()

    def _probe(self) -> int:
        """Reads the first byte to learn the total size and to prove the server honours Range requests."""
        response = self.session.get(self.url, headers={"Range": "bytes=0-0", "User-Agent": USER_AGENT},
                                    timeout=self.timeout)
        if response.status_code != 206:
            raise RangeNotSupported(
                f"{self.url} answered HTTP {response.status_code} to a Range request; "
                "a partial download needs a server that supports byte ranges (HTTP 206)"
            )
        total = response.headers.get("Content-Range", "").rpartition("/")[2]
        if not total.isdigit():
            raise RangeNotSupported(f"{self.url} did not report its size in Content-Range")
        return int(total)

    def _fetch(self, start: int, end: int) -> bytes:
        """Fetches bytes start..end inclusive, retrying transient failures."""
        expected = end - start + 1
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(
                    self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT},
                    timeout=self.timeout)
                if response.status_code == 206 and len(response.content) == expected:
                    self.requests_made += 1
                    self.bytes_fetched += expected
                    return response.content
                last = RuntimeError(f"HTTP {response.status_code}, {len(response.content)} of {expected} bytes")
            except requests.RequestException as exc:
                last = exc
            time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"range {start}-{end} of {self.url} failed after {self.retries} attempts: {last}")

    def _block(self, index: int) -> bytes:
        cached = self._blocks.get(index)
        if cached is not None:
            return cached
        start = index * self.block_size
        data = self._fetch(start, min(start + self.block_size, self.size) - 1)
        if len(self._blocks) >= self.max_cached_blocks:
            self._blocks.pop(next(iter(self._blocks)))
        self._blocks[index] = data
        return data

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        self.pos = max(self.pos, 0)
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0 or self.pos + size > self.size:
            size = self.size - self.pos
        if size <= 0:
            return b""
        if size >= self.block_size:
            data = self._fetch(self.pos, self.pos + size - 1)
        else:
            parts, position, remaining = [], self.pos, size
            while remaining > 0:
                index, offset = divmod(position, self.block_size)
                chunk = self._block(index)[offset:offset + remaining]
                parts.append(chunk)
                position += len(chunk)
                remaining -= len(chunk)
            data = b"".join(parts)
        self.pos += len(data)
        return data

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def safe_member_path(destination: Path, member: str) -> Path | None:
    """Where a ZIP member may be written, or None if its name tries to escape the destination."""
    pure = PurePosixPath(member)
    if pure.is_absolute() or ".." in pure.parts or not pure.name or "\\" in member:
        return None
    target = (destination / pure).resolve()
    base = destination.resolve()
    return target if base in target.parents else None


def select_members(names: list[str], include: list[str], limit: int, seed: int) -> list[str]:
    """A deterministic sample of up to `limit` member names matching any of the glob patterns."""
    matching = sorted(name for name in names if any(fnmatch.fnmatch(name, pattern) for pattern in include))
    if limit and len(matching) > limit:
        return sorted(random.Random(seed).sample(matching, limit))
    return matching


def download_remote_zip(source: dict[str, Any], destination: Path, budget: Budget, max_bytes: int,
                        max_documents: int, seed: int = 42, session: requests.Session | None = None,
                        out=print) -> list[Path]:
    """Fetches a sample of members (and their sidecar files) from a remote ZIP into `destination`.

    source keys: url, include (glob patterns for the documents), sidecars (optional templates such as
    "JSON/{stem}.json", copied next to each document when the archive has them).
    Stops when max_documents is reached or the bytes fetched would exceed max_bytes. Files already
    present with the right size are kept, so an interrupted run resumes.
    """
    include = list(source.get("include") or ["*"])
    sidecars = list(source.get("sidecars") or [])
    destination.mkdir(parents=True, exist_ok=True)

    remote = HttpRangeFile(source["url"], session=session)
    written: list[Path] = []
    try:
        with zipfile.ZipFile(remote) as archive:
            infos = {info.filename: info for info in archive.infolist() if not info.is_dir()}
            chosen = select_members(list(infos), include, max_documents, seed)
            out(f"[remote_zip] {len(infos):,} members, {len(chosen):,} selected; "
                f"listing cost {remote.bytes_fetched / 1e6:.1f} MB of {remote.size / 1e9:.1f} GB")
            for name in chosen:
                stem = PurePosixPath(name).stem
                wanted = [name] + [t.format(stem=stem) for t in sidecars if t.format(stem=stem) in infos]
                fetch_cost = sum(infos[w].compress_size for w in wanted)
                if remote.bytes_fetched + fetch_cost > max_bytes:
                    out(f"[remote_zip] stopped: the next document would exceed the {max_bytes / 1e9:.2f} GB source cap")
                    break
                for member in wanted:
                    target = safe_member_path(destination, member)
                    if target is None:
                        out(f"[remote_zip] skipped unsafe member name {member!r}")
                        continue
                    info = infos[member]
                    if not (target.exists() and target.stat().st_size == info.file_size):
                        budget.check(info.file_size)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        tmp = target.with_name(target.name + ".part")
                        with archive.open(info) as src, tmp.open("wb") as dst:
                            while chunk := src.read(1024 * 1024):
                                dst.write(chunk)
                        tmp.replace(target)
                    if member == name:
                        written.append(target)
            out(f"[remote_zip] fetched {remote.bytes_fetched / 1e6:.1f} MB in {remote.requests_made} requests "
                f"for {len(written)} documents")
    finally:
        remote.close()
    return written
