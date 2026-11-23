"""Progress bars that also show up where no terminal is attached (notebook `!` commands, pipes, log files).

tqdm redraws its bar with carriage returns. A pipe or notebook output does not render those as lines, so the
bar looks frozen or is missing altogether. On a terminal, Progress is a normal tqdm bar. Anywhere else it prints
one plain log line (logger "progress", see corpus/logs.py) every tenth of the work or every few seconds.

    with Progress(total=300, desc="nova download", unit="doc") as bar:
        bar.update(1)
    for row in Progress(rows, desc="catalogue", unit="doc"):
        ...
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Any, Iterable, Iterator

from tqdm import tqdm

log = logging.getLogger("progress")

# auto: a live bar on a terminal, log lines everywhere else.  bars: a plain tqdm bar even without a terminal
# (a notebook `!` cell that does show carriage-return bars, like Hugging Face's download bars).  lines: log lines always.
MODES = ("auto", "bars", "lines")
_mode = os.environ.get("PROGRESS_MODE", "auto")


def set_mode(mode: str) -> None:
    global _mode
    if mode not in MODES:
        raise ValueError(f"progress mode must be one of {MODES}")
    _mode = mode

TIME_STEP_SECONDS = 5.0       # without a terminal, a line at least this often so a long step never looks hung


class Progress:
    def __init__(self, iterable: Iterable | None = None, total: int | None = None, desc: str = "",
                 unit: str = "it", initial: int = 0, leave: bool = True, position: int | None = None,
                 disable: bool | None = False, **kwargs: Any) -> None:
        if total is None and iterable is not None and hasattr(iterable, "__len__"):
            total = len(iterable)
        self.desc, self.unit, self.total = desc, unit, total
        self.disabled = bool(disable)
        self._n = initial
        self._iterable = iterable
        self._postfix = ""
        self._step = max(1, total // 10) if total else 0
        self._initial = initial
        self._last_line_n = initial
        self._start = self._last_line_t = time.monotonic()
        self._printed = False
        self.bar = None
        self._stop = None
        terminal = sys.stderr.isatty()
        if not self.disabled and (_mode == "bars" or (_mode == "auto" and terminal)):
            if not terminal:           # a pipe or notebook cell: flat bars (no cursor moves), a frame every few seconds
                position, leave = None, True
                kwargs.setdefault("mininterval", 3.0)
                kwargs.setdefault("file", sys.stderr)
            self.bar = tqdm(iterable, total=total, desc=desc, unit=unit, initial=initial, leave=leave,
                            position=position, **kwargs)
        elif not self.disabled:
            log.info("%s: starting%s", desc, f" ({total} {unit})" if total else "")
            self._stop = threading.Event()
            threading.Thread(target=self._heartbeat, daemon=True).start()

    # ---- counting -------------------------------------------------------------------------------
    @property
    def n(self) -> int:
        return self.bar.n if self.bar is not None else self._n

    @n.setter
    def n(self, value: int) -> None:
        if self.bar is not None:
            self.bar.n = value
        else:
            self._n = value

    def refresh(self) -> None:
        """With a terminal: redraw. Without one: print a line if one is due, which is how a bar whose count is set
        directly (a Gemini batch being polled) still reports every few seconds."""
        if self.bar is not None:
            self.bar.refresh()
        elif not self.disabled:
            now = time.monotonic()
            if self._due(now):
                self._line(now)

    def _heartbeat(self) -> None:
        """Without a terminal, prints a line while one step takes long (a slow OCR page), so the run never looks hung."""
        while not self._stop.wait(TIME_STEP_SECONDS):
            now = time.monotonic()
            if now - self._last_line_t >= TIME_STEP_SECONDS:
                self._line(now, waiting=True)

    def _due(self, now: float) -> bool:
        finished = bool(self.total) and self.n >= self.total and self._last_line_n < self.total
        return (finished or now - self._last_line_t >= TIME_STEP_SECONDS
                or bool(self._step) and self.n - self._last_line_n >= self._step)

    def update(self, n: int = 1) -> None:
        if self.bar is not None:
            self.bar.update(n)         # tqdm counts it; adding to self.n as well would count every step twice
            if self.total and self.bar.n > self.total:      # the total was an estimate: grow it instead of showing 102%
                self.total = self.bar.total = self.bar.n
            return
        self._n += n
        if self.total and self._n > self.total:
            self.total = self._n
        if self.disabled:
            return
        now = time.monotonic()
        if self._due(now):
            self._line(now)

    def __iter__(self) -> Iterator:
        if self._iterable is None:
            raise TypeError("Progress was not given an iterable")
        if self.bar is not None:
            yield from self.bar          # tqdm counts the items itself
            return
        try:
            for item in self._iterable:
                yield item
                self.update(1)
        finally:
            self.close()

    # ---- display --------------------------------------------------------------------------------
    def set_postfix(self, ordered_dict: dict | None = None, refresh: bool = True, **kwargs: Any) -> None:
        if self.bar is not None:
            self.bar.set_postfix(ordered_dict, refresh=refresh, **kwargs)
            return
        values = {**(ordered_dict or {}), **kwargs}
        self._postfix = ", ".join(f"{k}={v}" for k, v in values.items())

    def set_postfix_str(self, text: str, refresh: bool = True) -> None:
        if self.bar is not None:
            self.bar.set_postfix_str(text, refresh=refresh)
            return
        self._postfix = text

    @staticmethod
    def _clock(seconds: float) -> str:
        hours, rest = divmod(max(int(seconds), 0), 3600)
        return f"{hours}:{rest // 60:02d}:{rest % 60:02d}"

    def _line(self, now: float, waiting: bool = False) -> None:
        count = f"{self.n}/{self.total}" if self.total else f"{self.n}"
        share = f" ({self.n / self.total:.0%})" if self.total else ""
        elapsed = now - self._start
        done = self.n - self._initial
        timing = f" | {self._clock(elapsed)}"
        if done > 0 and elapsed > 0:
            rate = done / elapsed
            timing += f" | {rate:.1f} {self.unit}/s" if rate >= 0.1 else f" | {1 / rate:.0f} s/{self.unit}"
            if self.total and self.n < self.total:
                timing += f" | eta {self._clock((self.total - self.n) / rate)}"
        tail = f" | {self._postfix}" if self._postfix else ""
        if waiting:
            tail += " | still working"
        log.info("%s: %s %s%s%s%s", self.desc, count, self.unit, share, timing, tail)
        self._printed = True
        self._last_line_n, self._last_line_t = self.n, now

    def close(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self.bar is not None:
            self.bar.close()
            return
        if not self.disabled and self.n != self._last_line_n:
            self._line(time.monotonic())

    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
