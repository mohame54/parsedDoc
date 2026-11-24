"""Loss curves and the other metrics of a finished (or running) training run, from its log or its TensorBoard files.

    python -m train.report train.log                              # prints a summary, writes report.png
    python -m train.report train.log --out run1.png --smooth 10
    python -m train.report --tb checkpoints/lora_run/tb           # from TensorBoard's own event files

In a notebook:

    from train.report import load, summarise, plot
    steps, vals = load("train.log")
    print(summarise(steps, vals)); plot(steps, vals, "report.png")

It needs no TensorBoard server (so it works where ports are not reachable, like Kaggle). The log is parsed with the
standard library; plotting needs matplotlib, and `--tb` needs tensorboard.
"""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path
from typing import Any

STEP_LINE = re.compile(r"\bstep (\d+)/\d+ .*")
PAIR = re.compile(r"(\w+)=(-?[\d.]+(?:[eE][-+]?\d+)?|nan|inf)")
VAL_LINE = re.compile(r"validation loss (-?[\d.]+(?:[eE][-+]?\d+)?) at step (\d+)")


def parse_log(text: str) -> tuple[list[dict[str, float]], list[tuple[int, float]]]:
    """(training points, validation points). A training point is {"step": n, "loss": ..., "grad_norm": ..., ...}."""
    steps: list[dict[str, float]] = []
    vals: list[tuple[int, float]] = []
    for line in text.splitlines():
        found = VAL_LINE.search(line)
        if found:
            vals.append((int(found.group(2)), float(found.group(1))))
            continue
        head = STEP_LINE.search(line)
        if head and " loss=" in line:
            row = {key: float(value) for key, value in PAIR.findall(line)}
            row["step"] = float(head.group(1))
            steps.append(row)
    # a resumed run repeats steps in one log: keep the last point for each step
    unique = {int(row["step"]): row for row in steps}
    return [unique[s] for s in sorted(unique)], sorted(dict(vals).items())


def parse_events(tb_dir: str | Path) -> tuple[list[dict[str, float]], list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    rows: dict[int, dict[str, float]] = {}
    for tag in tags:
        if tag.startswith("train/"):
            for event in acc.Scalars(tag):
                rows.setdefault(event.step, {"step": float(event.step)})[tag.split("/", 1)[1]] = event.value
    vals = [(e.step, e.value) for e in acc.Scalars("val/loss")] if "val/loss" in tags else []
    return [rows[s] for s in sorted(rows)], vals


def load(path: str | Path | None = None, tb: str | Path | None = None):
    if tb is not None:
        return parse_events(tb)
    return parse_log(Path(path).read_text(encoding="utf-8", errors="replace"))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def summarise(steps: list[dict[str, float]], vals: list[tuple[int, float]]) -> dict[str, Any]:
    """Numbers worth reading after a run, with a few plain-language warnings."""
    out: dict[str, Any] = {"logged_points": len(steps), "last_step": int(steps[-1]["step"]) if steps else 0}
    notes: list[str] = []
    losses = [row["loss"] for row in steps if "loss" in row]
    if len(losses) >= 6:
        first, last = _mean(losses[:3]), _mean(losses[-3:])
        out["train_loss_first3"], out["train_loss_last3"] = round(first, 4), round(last, 4)
        out["train_loss_change"] = f"{(last - first) / first:+.0%}"
    if vals:
        best_step, best = min(vals, key=lambda p: p[1])
        out.update(val_first=round(vals[0][1], 4), val_last=round(vals[-1][1], 4), val_best=round(best, 4),
                   val_best_step=best_step, val_points=len(vals))
        if len(vals) >= 3 and vals[-1][1] > best * 1.03 and vals[-1][0] != best_step:
            notes.append(f"validation loss is {vals[-1][1] / best - 1:.0%} above its best (step {best_step}): "
                         "the last checkpoint may be past the best one, so consider an earlier one")
        if len(vals) >= 3 and vals[-1][1] > vals[-2][1] > vals[-3][1] and len(losses) >= 6 and out["train_loss_last3"] < out["train_loss_first3"]:
            notes.append("validation loss rose over the last evaluations while the training loss fell: overfitting")
        if vals[-1][1] >= vals[0][1] * 0.98:
            notes.append("validation loss ended no lower than where it started: the adapters did not generalise")
    for key, label in (("grad_norm", "grad_norm"), ("tokens_per_s", "tokens_per_s"), ("peak_mem_gb", "peak_mem_gb")):
        series = [row[key] for row in steps if key in row]
        if series:
            out[f"{label}_median"] = round(statistics.median(series), 3)
            out[f"{label}_max"] = round(max(series), 3)
    if any("skipped_steps" in row for row in steps):
        out["skipped_steps"] = int(max(row.get("skipped_steps", 0) for row in steps))
    clip = [row["grad_norm"] for row in steps if "grad_norm" in row]
    if clip and sum(g > 1.0 for g in clip) > 0.9 * len(clip):
        notes.append("almost every gradient norm is above the clip value 1.0, so clipping is limiting every step: "
                     "a lower learning rate may train more smoothly")
    if any(row.get("loss") != row.get("loss") for row in steps):
        notes.append("the loss contains nan")
    out["notes"] = notes or ["nothing unusual"]
    return out


def plot(steps: list[dict[str, float]], vals: list[tuple[int, float]], out: str | Path = "report.png",
         smooth: int = 5) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(key: str):
        pairs = [(row["step"], row[key]) for row in steps if key in row]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def rolling(values: list[float]) -> list[float]:
        return [_mean(values[max(0, i - smooth + 1): i + 1]) for i in range(len(values))]

    panels = [("loss", "loss"), ("grad_norm", "gradient norm (before clipping)"), ("lr", "learning rate"),
              ("peak_mem_gb", "peak GPU memory (GB)"), ("tokens_per_s", "tokens / second"),
              ("batch_size", "samples per micro-batch")]
    shown = [(key, title) for key, title in panels if series(key)[0]]
    columns = 2
    rows = max(1, -(-len(shown) // columns))
    fig, axes = plt.subplots(rows, columns, figsize=(11, 3.2 * rows), squeeze=False)
    for axis, (key, title) in zip(axes.flat, shown):
        x, y = series(key)
        axis.plot(x, y, color="tab:blue", alpha=0.35 if key == "loss" else 0.9, label="logged")
        if key == "loss":
            axis.plot(x, rolling(y), color="tab:blue", linewidth=2, label=f"train, mean of {smooth}")
            if vals:
                axis.plot([v[0] for v in vals], [v[1] for v in vals], color="tab:orange", marker="o", linewidth=2,
                          label="validation")
            axis.legend()
        axis.set_title(title)
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.3)
    for axis in list(axes.flat)[len(shown):]:
        axis.axis("off")
    fig.tight_layout()
    path = Path(out)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", help="the training log (stdout/stderr of the run)")
    parser.add_argument("--tb", help="a TensorBoard folder (<out_dir>/tb) instead of the log")
    parser.add_argument("--out", default="report.png")
    parser.add_argument("--smooth", type=int, default=5, help="points in the moving average of the training loss")
    args = parser.parse_args(argv)
    if not args.log and not args.tb:
        parser.error("give the log file or --tb <folder>")
    steps, vals = load(args.log, args.tb)
    if not steps:
        raise SystemExit("no training lines found (looking for `step N/M ... loss=...`)")
    for key, value in summarise(steps, vals).items():
        print(f"{key:22} {value}")
    try:
        print("plot written to", plot(steps, vals, args.out, args.smooth))
    except ImportError:
        print("matplotlib is not installed here, so no plot was drawn (pip install matplotlib)")


if __name__ == "__main__":
    main()
