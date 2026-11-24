"""One logging setup for the corpus builder, the labeller and training.

Log lines (progress, step banners, per-document skips and errors) go to stderr through `logging`, which is never
held back by a pipe or notebook the way stdout is. Command results (a plan, a status table, a summary) stay on
stdout with `print`, so they can still be piped.

    setup_logging()                            # INFO to stderr
    setup_logging("DEBUG", "labeller.log")     # also a file
"""
from __future__ import annotations

import logging
import os
import sys

from tqdm import tqdm

FORMAT = "%(asctime)s %(levelname)s %(message)s"
DATE_FORMAT = "%H:%M:%S"
QUIET_LIBRARIES = ("rapidocr", "urllib3", "httpx", "httpcore", "filelock", "fsspec", "huggingface_hub", "datasets",
                   "hf_xet")  # INFO lines are noise
_MARK = "_corpus_logging"


class QuietLibraries(logging.Filter):
    """Drops INFO lines from chatty libraries (every HTTP request of a dataset download). A filter on the handler
    holds even when a library resets its own logger level."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING or logging.getLogger().level <= logging.DEBUG:
            return True
        return not record.name.split(".")[0].startswith(QUIET_LIBRARIES)      # also httpx2, httpcore.*, ...


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
        console.addFilter(QuietLibraries())
        if os.environ.get("RANK", "0") not in ("0", ""):
            console.setLevel(logging.WARNING)       # under torchrun only rank 0 prints the INFO lines: no doubled log
            _silence_other_ranks()
        setattr(console, _MARK, True)
        root.addHandler(console)
    if log_file and not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith(log_file) for h in handlers):
        out = logging.FileHandler(log_file, encoding="utf-8")
        out.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
        out.addFilter(QuietLibraries())
        setattr(out, _MARK, True)
        root.addHandler(out)
    for name in QUIET_LIBRARIES:                       # `--log-level DEBUG` brings their lines back
        logging.getLogger(name).setLevel(logging.WARNING if root.level > logging.DEBUG else logging.DEBUG)


def _silence_other_ranks() -> None:
    """Under torchrun only rank 0 shows anything: no download bars, no repeated Python warnings. Errors still
    raise and print their traceback from any rank."""
    import warnings

    warnings.filterwarnings("ignore")
    if "huggingface_hub" in sys.modules:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    if "transformers" in sys.modules:
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()


def add_arguments(parser) -> None:
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Log level for progress and events (stderr).")
    parser.add_argument("--log-file", help="Also write the log lines to this file.")
    parser.add_argument("--progress", default=None, choices=["auto", "bars", "lines"],
                        help="auto: live bars on a terminal, log lines elsewhere; bars: tqdm bars even in a "
                             "notebook cell or pipe; lines: always log lines (env PROGRESS_MODE sets the default)")


def configure(args) -> None:
    setup_logging(getattr(args, "log_level", "INFO"), getattr(args, "log_file", None))
    if getattr(args, "progress", None):
        from .progress import set_mode
        set_mode(args.progress)
