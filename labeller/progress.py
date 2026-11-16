"""Progress display.

Layout while `labeller run` is going:
  labeller: 2/7 [now: OCR] ████░░░░ 0:01:12        <- run bar (position 0)
  Stage 1 · stage1-ab12cd34  ███░░░ 1/3 running    <- batch bar (position 1)
    uploading pages          ██░░░ 4/9              <- inner bar (position 2, cleared when done)

Step banners and per-document lines go through tqdm.write so they never break a bar.
"""
from __future__ import annotations

from tqdm import tqdm

RUN_STEPS = [
    "prepare",
    "OCR",
    "Stage 1 submit",
    "Stage 1 wait",
    "Stage 2 submit",
    "Stage 2 wait",
]


class RunProgress:
    def __init__(self, enabled: bool, steps: list[str] | None = None) -> None:
        self.enabled = enabled
        self.total = len(steps or RUN_STEPS)
        self.bar = (
            tqdm(total=self.total, position=0, desc="labeller", unit="step",
                 bar_format="{desc}: {n_fmt}/{total_fmt} {bar} {postfix} {elapsed}")
            if enabled else None
        )

    def begin(self, name: str) -> None:
        if self.bar is not None:
            self.bar.set_postfix_str(f"now: {name}")
            tqdm.write(f"── {name} ──")

    def done(self) -> None:
        if self.bar is not None:
            self.bar.update(1)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.set_postfix_str("finished")
            self.bar.close()


def inner_bar(total: int, desc: str, position: int, unit: str = "doc") -> tqdm:
    """A bar that disappears when finished, so the terminal keeps only the run bar and batch bars."""
    return tqdm(total=total, desc=desc, position=position, unit=unit, leave=False)
