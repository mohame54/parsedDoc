"""One logging setup for the corpus builder, the labeller and training.

Log lines (progress, step banners, per-document skips and errors) go to stderr through `logging`, which is never
held back by a pipe or notebook the way stdout is. Command results (a plan, a status table, a summary) stay on
stdout with `print`, so they can still be piped.

    setup_logging()                            # INFO to stderr
    setup_logging("DEBUG", "labeller.log")     # also a file
"""
from __future__ import annotations

import logging
import sys

from tqdm import tqdm

FORMAT = "%(asctime)s %(levelname)s %(message)s"
DATE_FORMAT = "%H:%M:%S"
QUIET_LIBRARIES = ("rapidocr", "urllib3", "httpx", "httpcore", "filelock", "fsspec")  # INFO lines are noise
_MARK = "_corpus_logging"


class TqdmHandler(logging.StreamHandler):
    """Writes through tqdm so a live progress bar is not torn by a log line (terminals only)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:
            self.handleError(record)


def setup_logging(level: str | int = "INFO", log_file: str | None = None) -> None:
    """Idempotent: calling it again only changes the level and the file."""
    root = logging.getLogger()
    root.setLevel(level.upper() if isinstance(level, str) else level)
    handlers = [h for h in root.handlers if getattr(h, _MARK, False)]
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in handlers):
        stream = sys.stderr
        console: logging.Handler = TqdmHandler(stream) if stream.isatty() else logging.StreamHandler(stream)
        console.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
        setattr(console, _MARK, True)
        root.addHandler(console)
    if log_file and not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith(log_file) for h in handlers):
        out = logging.FileHandler(log_file, encoding="utf-8")
        out.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
        setattr(out, _MARK, True)
        root.addHandler(out)
    for name in QUIET_LIBRARIES:                       # `--log-level DEBUG` brings their lines back
        logging.getLogger(name).setLevel(logging.WARNING if root.level > logging.DEBUG else logging.DEBUG)


def add_arguments(parser) -> None:
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Log level for progress and events (stderr).")
    parser.add_argument("--log-file", help="Also write the log lines to this file.")


def configure(args) -> None:
    setup_logging(getattr(args, "log_level", "INFO"), getattr(args, "log_file", None))
