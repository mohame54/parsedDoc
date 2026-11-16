"""Rank-0 training metrics: tqdm progress bar, a logging line, and TensorBoard scalars.

Loss and grad norm are reported together on every log step. The grad norm is the pre-clip total norm,
so it shows what clipping is doing. Under fp16 the grad scaler's loss scale is logged as well.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

log = logging.getLogger("train.sft")


class MetricLogger:
    def __init__(
        self,
        log_dir: str | Path | None = None,
        is_main: bool = True,
        log_every: int = 10,
        total_steps: int | None = None,
        tensorboard: bool = True,
    ) -> None:
        self.enabled = is_main
        self.log_every = max(1, log_every)
        self.writer = None
        self.bar = None
        self._last = time.perf_counter()
        if not self.enabled:
            return
        # disable=None: no bar when output is a file or pipe (background runs), so logs stay readable
        self.bar = tqdm(total=total_steps, desc="sft", dynamic_ncols=True, disable=None)
        if tensorboard and log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(Path(log_dir)))

    def should_log(self, step: int) -> bool:
        return self.enabled and step % self.log_every == 0

    def advance(self) -> None:
        """Call once per optimizer step on every rank (rank 0 moves the bar)."""
        if self.bar is not None:
            self.bar.update(1)

    def log(self, step: int, metrics: dict[str, float], tokens: float | None = None) -> None:
        """Call on log steps only, with already-reduced python floats."""
        if not self.enabled:
            return
        now = time.perf_counter()
        extra: dict[str, float] = {}
        if tokens is not None and now > self._last:
            extra["tokens_per_s"] = tokens / (now - self._last)
        self._last = now
        values = {**metrics, **extra}

        shown = {k: round(v, 4) for k, v in values.items() if k in ("loss", "grad_norm", "lr", "loss_scale")}
        if self.bar is not None:
            self.bar.set_postfix(shown, refresh=False)
        log.info("step %d %s", step, " ".join(f"{k}={v:.4g}" for k, v in values.items()))
        if self.writer is not None:
            for key, value in values.items():
                self.writer.add_scalar(f"train/{key}", value, step)

    def scalars(self, step: int, values: dict[str, Any]) -> None:
        """Write arbitrary scalars (e.g. epoch-level or validation numbers) to TensorBoard."""
        if self.writer is not None:
            for key, value in values.items():
                self.writer.add_scalar(key, float(value), step)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
