"""Partial downloads from a remote ZIP, tested against a local HTTP server that supports Range requests."""
import http.server
import io
import random
import threading
import zipfile
from pathlib import Path

import pytest

from corpus.remote_zip import HttpRangeFile, RangeNotSupported, download_remote_zip, safe_member_path, select_members
from corpus.utils import Budget


def make_zip(path: Path, members: dict[str, bytes], extra=None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        if extra:
            extra(archive)
    path.write_bytes(buffer.getvalue())
    return buffer.getvalue()


def serve(payload: bytes, honour_ranges: bool = True):
    """Starts a throwaway server for `payload`. Returns (url, stats, stop)."""
    stats = {"requests": 0, "bytes": 0, "ranges": []}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            stats["requests"] += 1
            header = self.headers.get("Range")
            if header and honour_ranges:
                start, _, end = header.removeprefix("bytes=").partition("-")
                start, end = int(start), min(int(end), len(payload) - 1)
                body = payload[start:end + 1]
                stats["ranges"].append((start, end))
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            else:
                body = payload
                self.send_response(200)
            stats["bytes"] += len(body)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}/archive.zip", stats, server.shutdown


@pytest.fixture
def archive(tmp_path):
    members = {}
    for i in range(40):
        members[f"PDF/{i:03d}.pdf"] = (f"%PDF page {i} ".encode()) * 300
        members[f"JSON/{i:03d}.json"] = f'{{"page": {i}}}'.encode()
    members["__MACOSX/PDF/._000.pdf"] = b"junk"
    # 3 MB of incompressible data that must never be downloaded.
    members["BIG/huge.bin"] = random.Random(0).randbytes(3_000_000)
    payload = make_zip(tmp_path / "a.zip", members)
    url, stats, stop = serve(payload)
    yield url, stats, payload
    stop()


def budget(tmp_path):
    return Budget(tmp_path / "data", 5)


def test_http_range_file_reads_exactly_the_requested_bytes(archive):
    url, stats, payload = archive
    remote = HttpRangeFile(url, block_size=1024)
    assert remote.size == len(payload)
    remote.seek(100)
    assert remote.read(50) == payload[100:150]
    remote.seek(-30, 2)
    assert remote.read() == payload[-30:]
    assert remote.bytes_fetched < 4096        # only a couple of small blocks, never the whole archive


def test_downloads_only_the_sampled_members_and_their_sidecars(archive, tmp_path):
    url, stats, payload = archive
    source = {"url": url, "include": ["PDF/*.pdf"], "sidecars": ["JSON/{stem}.json"]}
    dest = tmp_path / "out"
    written = download_remote_zip(source, dest, budget(tmp_path), 50_000_000, max_documents=5, seed=1,
                                  out=lambda _: None)
    assert len(written) == 5 and all(p.suffix == ".pdf" and p.exists() for p in written)
    assert {p.stem for p in written} == {p.stem for p in (dest / "JSON").glob("*.json")}   # a sidecar for each
    assert not (dest / "BIG").exists() and not (dest / "__MACOSX").exists()
    assert len(payload) > 3_000_000
    assert stats["bytes"] < 1_000_000    # two 256 KB blocks (members, table of contents), not the 3 MB member
    assert (dest / written[0].relative_to(dest)).read_bytes().startswith(b"%PDF page")


def test_selection_is_deterministic_and_seeded():
    names = [f"PDF/{i:03d}.pdf" for i in range(100)] + ["JSON/x.json"]
    first = select_members(names, ["PDF/*.pdf"], 10, seed=42)
    assert first == select_members(names, ["PDF/*.pdf"], 10, seed=42)
    assert first != select_members(names, ["PDF/*.pdf"], 10, seed=43)
    assert len(select_members(names, ["PDF/*.pdf"], 0, seed=1)) == 100          # 0 = no limit
    assert all(n.startswith("PDF/") for n in first)


def test_a_second_run_reuses_files_instead_of_fetching_again(archive, tmp_path):
    url, stats, _ = archive
    source = {"url": url, "include": ["PDF/*.pdf"]}
    dest = tmp_path / "out"
    download_remote_zip(source, dest, budget(tmp_path), 50_000_000, 4, out=lambda _: None)
    first_run = len(stats["ranges"])
    download_remote_zip(source, dest, budget(tmp_path), 50_000_000, 4, out=lambda _: None)
    second_run = stats["ranges"][first_run:]
    # The second run re-reads the table of contents at the end of the archive, but no member data
    # (the members live in the first block, which starts at byte 0).
    assert second_run and all(start > 0 for start, _ in second_run if _ > 0)


def test_the_source_byte_cap_stops_the_download(archive, tmp_path):
    url, _, _ = archive
    messages = []
    written = download_remote_zip({"url": url, "include": ["PDF/*.pdf"]}, tmp_path / "out", budget(tmp_path),
                                  max_bytes=3000, max_documents=30, out=messages.append)
    assert len(written) < 30
    assert any("exceed" in m for m in messages)


def test_unsafe_member_names_are_never_written(tmp_path):
    def evil(archive):
        archive.writestr("../escape.pdf", b"x")
        archive.writestr("/abs.pdf", b"x")
        archive.writestr("PDF/ok.pdf", b"fine")

    payload = make_zip(tmp_path / "evil.zip", {}, extra=evil)
    url, _, stop = serve(payload)
    dest = tmp_path / "out" / "inner"
    try:
        written = download_remote_zip({"url": url, "include": ["*.pdf"]}, dest, budget(tmp_path), 10_000_000,
                                      max_documents=10, out=lambda _: None)
    finally:
        stop()
    assert [p.name for p in written] == ["ok.pdf"]
    assert not (tmp_path / "out" / "escape.pdf").exists() and not Path("/abs.pdf").exists()
    assert safe_member_path(dest, "../x.pdf") is None and safe_member_path(dest, "a\\b.pdf") is None


def test_a_server_without_range_support_is_refused_clearly(tmp_path):
    payload = make_zip(tmp_path / "a.zip", {"PDF/a.pdf": b"x"})
    url, _, stop = serve(payload, honour_ranges=False)
    try:
        with pytest.raises(RangeNotSupported, match="byte ranges"):
            HttpRangeFile(url)
    finally:
        stop()
