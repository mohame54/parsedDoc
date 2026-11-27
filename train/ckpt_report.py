"""A metrics report with graphs for every checkpoint, so a long run can be reviewed and the best checkpoint chosen.

    build_checkpoint_report(history, step, out_dir, kind="sft")     # one checkpoint's folder
    write_index(reports_dir)                                        # the comparison across all of them

`history` is {series name: [(step, value), ...]} (train/metrics.py `MetricLogger.history`, or `load_history` of its
`metrics.jsonl`). `kind` picks what is reported:

  sft  train loss, validation loss, generation scores per task (and strict F1 / hallucination), learning rate, grad norm
  rl   reward and its parts, abstention / hallucination rates per field state, calibration (ECE, AUROC, confident-wrong),
       KL to the starting model, completion length

Each checkpoint folder gets `metrics.json` (every series at this step, plus the highlights), `report.md`, and the PNG
graphs of the run up to this step with a marker on this step. `write_index` then writes `index.md`, `checkpoints.csv`
and `compare.png` next to the folders, one row per checkpoint, the best by the selection metric marked.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

History = dict[str, list[tuple[int, float]]]
WINDOW = 20                                     # train values averaged for "the loss / reward now"

# what each kind reports -----------------------------------------------------------------------------------------
# highlights: label -> (series candidates tried in order, how). how: latest | mean_last | best_min | best_max
KINDS: dict[str, dict[str, Any]] = {
    "sft": {
        "highlights": {
            "train_loss": (("train/loss",), "mean_last"),
            "val_loss": (("val/loss",), "latest"),
            "gen_score": (("gen/score",), "latest"),
            "json_valid": (("gen/json_valid",), "latest"),
        },
        "select": ("val_loss", "min"),          # the metric that ranks checkpoints, and which direction is better
        "compare": ("val_loss", "gen_score", "json_valid"),
        "plots": {
            "loss.png": [
                {"title": "training loss", "series": [("train/loss", "loss")], "smooth": ["train/loss"]},
                {"title": "validation loss", "series": [("val/loss", "val loss")]}],
            "generation.png": [
                {"title": "generation score (higher is better)", "prefix": "gen/", "skip": ("gen/hallucination",)},
                {"title": "strict F1 per task", "prefix": "gen_strict/"},
                {"title": "hallucination per task (lower is better)", "prefix": "gen_hallucination/"}],
            "optimization.png": [
                {"title": "learning rate", "series": [("train/lr", "lr")]},
                {"title": "gradient norm (before clipping)", "series": [("train/grad_norm", "grad norm")]}],
        },
    },
    "rl": {
        "highlights": {
            "reward": (("train/reward", "reward"), "mean_last"),
            "reward_std": (("train/reward_std", "reward_std"), "mean_last"),
            "exact_accuracy": (("quality/exact_accuracy_answerable",), "mean_last"),
            "valid_json": (("reward/valid_json",), "mean_last"),
            "kl": (("train/kl", "kl"), "mean_last"),
            "completion_length": (("train/completion_length", "completion_length"), "mean_last"),
            "hallucination": (("behaviour/wrong_rate_must_abstain",), "mean_last"),
            "abstain_when_unreadable": (("behaviour/abstain_rate_must_abstain",), "mean_last"),
            "abstain_when_readable": (("behaviour/abstain_rate_answerable",), "mean_last"),
            "ece": (("calibration/ece",), "mean_last"),
            "auroc": (("calibration/auroc",), "mean_last"),
            "high_conf_wrong": (("calibration/high_conf_wrong",), "mean_last"),
        },
        "select": ("reward", "max"),
        "compare": ("reward", "exact_accuracy", "hallucination", "abstain_when_readable", "ece"),
        "plots": {
            "reward.png": [
                {"title": "final reward: the number GRPO maximises (mean over the group, band = ± std)",
                 "series": [("train/reward", "reward")], "smooth": ["train/reward"],
                 "band": ("train/reward", "train/reward_std")}],
            "reward_components.png": "components",         # one panel per reward component, built in `_rl_components`
            "hacking_watch.png": [
                {"title": "1. the reward (what is optimised)", "series": [("train/reward", "reward")],
                 "smooth": ["train/reward"]},
                {"title": "2. exact accuracy on readable fields (higher is better; not paid for directly)",
                 "series": [("quality/exact_accuracy_answerable", "readable fields"),
                            ("quality/exact_accuracy_partial", "hard-to-read fields")]},
                {"title": "3. invented values on unreadable fields (lower is better)",
                 "series": [("behaviour/wrong_rate_must_abstain", "invented")]},
                {"title": "4. nulls on readable fields (should stay low: abstaining is not a safe answer)",
                 "series": [("behaviour/abstain_rate_answerable", "abstained")]},
                {"title": "5. answers that parse as JSON (higher is better)", "series": [("reward/valid_json", "valid")]},
                {"title": "6. completion length in tokens (a sudden collapse is a warning)",
                 "series": [("train/completion_length", "length"), ("train/completions/mean_length", "length")]}],
            "behaviour.png": [
                {"title": "abstain rate by field state", "prefix": "behaviour/abstain_rate_"},
                {"title": "wrong / invented value rate by field state (lower is better)",
                 "prefix": "behaviour/wrong_rate_"},
                {"title": "exact accuracy by field state (higher is better)", "prefix": "quality/exact_accuracy_"}],
            "calibration.png": [
                {"title": "calibration error (lower is better)", "series": [("calibration/ece", "ECE")]},
                {"title": "does confidence separate right from wrong? (AUROC, 0.5 = no)",
                 "series": [("calibration/auroc", "AUROC")]},
                {"title": "confident-and-wrong / unsure-and-right share",
                 "series": [("calibration/high_conf_wrong", "confident and wrong"),
                            ("calibration/low_conf_right", "unsure but right")]}],
            "optimization.png": [
                {"title": "KL to the starting model", "series": [("train/kl", "kl"), ("kl", "kl")]},
                {"title": "learning rate", "series": [("train/learning_rate", "lr"), ("train/lr", "lr")]},
                {"title": "gradient norm", "series": [("train/grad_norm", "grad norm")]}],
        },
    },
}
RUN_KEYS = ("model_id", "kind", "lr", "epochs", "max_steps", "batch_tokens", "grad_accum", "lora", "lora_r", "lora_alpha",
            "init_from", "preset", "augment", "augment_p", "extended_p", "max_length", "num_generations", "beta",
            "loss_type", "max_completion_tokens", "state_prefix")


def _rl_components() -> list[dict[str, Any]]:
    """One panel per reward component, each on its own scale, so no term hides inside the average. The weights are
    the reward's own (train/rl_reward.py)."""
    from .rl_reward import WEIGHTS
    meaning = {"content": "right values; abstaining on unreadable fields; no invented values",
               "calibration": "stated confidence matches being right", "schema": "answer follows the prompt's schema",
               "rotation": "reported page rotation is right", "reason": "reason tag matches the degradation"}
    return [{"title": f"{part}: {meaning.get(part, part)} (weight {weight:.0%} of the reward)",
             "series": [(f"reward/{part}", part)], "smooth": [f"reward/{part}"]} for part, weight in WEIGHTS.items()]


def reward_hacking_flags(history: History, step: int) -> list[str]:
    """Warnings when the reward is rising while something it should go with is not. Compares the mean of the latest
    WINDOW points with the WINDOW before them, so it needs about 2 x WINDOW points to speak, and it only judges while
    the reward is rising (a falling reward is not being gamed). The thresholds are deliberately plain."""
    def moved(names: tuple[str, ...]) -> float | None:
        series = _pick(history, names, step)
        if len(series) < 10:
            return None
        k = min(WINDOW, len(series) // 2)
        return _mean_last(series, k) - _mean_last(series[:-k], k)

    rise = moved(("train/reward", "reward"))
    if rise is None or rise < 0.02:
        return []
    flags = []
    checks = [
        (("quality/exact_accuracy_answerable",), -0.02, "exact accuracy on readable fields fell by {d:.3f}"),
        (("behaviour/wrong_rate_must_abstain",), 0.02, "invented values on unreadable fields rose by {d:.3f}"),
        (("behaviour/abstain_rate_answerable",), 0.05, "nulls on readable fields rose by {d:.3f} (the model may have "
                                                       "learned that abstaining is safe)"),
        (("reward/valid_json",), -0.05, "valid JSON fell by {d:.3f}"),
        (("calibration/auroc",), -0.03, "confidence separates right from wrong worse (AUROC down {d:.3f})"),
        (("calibration/ece",), 0.03, "calibration error rose by {d:.3f}"),
    ]
    for names, limit, text in checks:
        delta = moved(names)
        if delta is not None and ((limit < 0 and delta < limit) or (limit > 0 and delta > limit)):
            flags.append(f"reward rose by {rise:.3f} but " + text.format(d=abs(delta)))
    length = _pick(history, ("train/completion_length", "train/completions/mean_length"), step)
    if len(length) >= 10:
        k = min(WINDOW, len(length) // 2)
        before, now = _mean_last(length[:-k], k), _mean_last(length, k)
        if before > 0 and now < 0.6 * before:
            flags.append(f"reward rose by {rise:.3f} but completions shrank from {before:.0f} to {now:.0f} tokens "
                         "(short answers may be skipping fields)")
    content, calibration = moved(("reward/content",)), moved(("reward/calibration",))
    if content is not None and calibration is not None and calibration > 0.05 and content < -0.02:
        flags.append(f"calibration score rose by {calibration:.3f} while content fell by {abs(content):.3f} "
                     "(confidence is being tuned instead of the answers)")
    return flags


# ---- reading -------------------------------------------------------------------------------------------------

def load_history(path: str | Path) -> History:
    history: History = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        history.setdefault(row["name"], []).append((row["step"], row["value"]))
    return history


def _upto(history: History, name: str, step: int) -> list[tuple[int, float]]:
    return [(s, v) for s, v in history.get(name, []) if s <= step]


def _pick(history: History, names: tuple[str, ...], step: int) -> list[tuple[int, float]]:
    for name in names:
        series = _upto(history, name, step)
        if series:
            return series
    return []


def _mean_last(series: list[tuple[int, float]], n: int = WINDOW) -> float:
    tail = [v for _, v in series[-n:]]
    return sum(tail) / len(tail)


def summarise(history: History, step: int, kind: str) -> dict[str, Any]:
    """Every series at `step` (latest value and the step it was measured at, mean of the last WINDOW, min, max) and the
    kind's highlights."""
    series_summary: dict[str, dict[str, float]] = {}
    for name in sorted(history):
        series = _upto(history, name, step)
        if series:
            values = [v for _, v in series]
            series_summary[name] = {"latest": series[-1][1], "at": series[-1][0], f"mean_last_{WINDOW}": _mean_last(series),
                                    "min": min(values), "max": max(values), "n": len(values)}
    highlights: dict[str, Any] = {}
    flags = reward_hacking_flags(history, step) if kind == "rl" else []
    for label, (names, how) in KINDS[kind]["highlights"].items():
        series = _pick(history, names, step)
        if not series:
            continue
        highlights[label] = _mean_last(series) if how == "mean_last" else series[-1][1]
        if how == "latest":
            highlights[f"{label}_measured_at"] = series[-1][0]
    if kind == "sft":
        val = _upto(history, "val/loss", step)
        if val:
            best_step, best = min(val, key=lambda sv: sv[1])
            highlights.update(best_val_loss=best, best_val_step=best_step,
                              is_best_val=bool(val[-1][0] == step and val[-1][1] <= best))
    if kind == "rl":
        highlights["hacking_flags"] = len(flags)
    return {"series": series_summary, "highlights": highlights, "flags": flags}


# ---- drawing -------------------------------------------------------------------------------------------------

def _moving_average(values: list[float], window: int = 10) -> list[float]:
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1): i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _plot(path: Path, panels: list[dict[str, Any]], history: History, step: int, title: str) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drawn: list[tuple[dict[str, Any], list[tuple[str, list[tuple[int, float]]]]]] = []
    for panel in panels:
        lines: list[tuple[str, list[tuple[int, float]]]] = []
        if "prefix" in panel:
            for name in sorted(history):
                if name.startswith(panel["prefix"]) and name not in panel.get("skip", ()):
                    series = _upto(history, name, step)
                    if series:
                        lines.append((name[len(panel["prefix"]):], series))
        else:
            used = set()
            for name, label in panel["series"]:
                series = _upto(history, name, step)
                if series and label not in used:
                    lines.append((label, series))
                    used.add(label)
        if lines:
            drawn.append((panel, lines))
    if not drawn:
        return False
    fig, axes = plt.subplots(len(drawn), 1, figsize=(9, 3.1 * len(drawn)), squeeze=False)
    for ax, (panel, lines) in zip(axes[:, 0], drawn):
        for label, series in lines:
            xs, ys = [s for s, _ in series], [v for _, v in series]
            smooth = any(label == lbl and name in panel.get("smooth", ()) for name, lbl in panel.get("series", ()))
            if smooth and len(ys) > 3:
                ax.plot(xs, ys, alpha=0.3, linewidth=1, label=f"{label} (each logged step)")
                ax.plot(xs, _moving_average(ys), linewidth=1.8, label=f"{label} (moving average)")
            else:
                ax.plot(xs, ys, marker="o" if len(xs) < 40 else None, markersize=3.5, linewidth=1.5, label=label)
        if "band" in panel:
            main, spread = (_upto(history, n, step) for n in panel["band"])
            if main and spread and len(main) == len(spread):
                xs = [s for s, _ in main]
                ax.fill_between(xs, [m - d for (_, m), (_, d) in zip(main, spread)],
                                [m + d for (_, m), (_, d) in zip(main, spread)], alpha=0.12, label="± std")
        ax.axvline(step, color="crimson", linestyle="--", linewidth=1, label=f"this checkpoint (step {step})")
        ax.set_title(panel["title"], fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.34 / (3.1 * len(drawn)) - 0.01))     # room for the title above the first panel
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


# ---- one checkpoint ------------------------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def build_checkpoint_report(history: History, step: int, out_dir: str | Path, kind: str = "sft", name: str | None = None,
                            run_config: dict[str, Any] | None = None, extra: dict[str, Any] | None = None,
                            generations: str | Path | None = None, final: bool = False,
                            lineage: list[dict[str, Any]] | None = None) -> Path:
    """Writes metrics.json, report.md and the graphs into `out_dir`; returns it."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = name or out.name
    summary = summarise(history, step, kind)
    config = {k: v for k, v in (run_config or {}).items() if k in RUN_KEYS and v is not None}
    metrics = {"name": name, "kind": kind, "step": step, "final": final, "epoch": (extra or {}).get("epoch"),
               "highlights": summary["highlights"], "flags": summary["flags"], "series": summary["series"], "run": config,
               "lineage": lineage or []}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    images = [fname for fname, panels in KINDS[kind]["plots"].items()
              if _plot(out / fname, _rl_components() if panels == "components" else panels, history, step,
                       f"{name} ({kind})")]
    if generations and Path(generations).exists():
        shutil.copy2(generations, out / "generations.jsonl")

    lines = [f"# {name}", "", f"**{kind.upper()} checkpoint** at step {step}" + (" (end of the run)" if final else "") +
             (f", epoch {metrics['epoch']}" if metrics["epoch"] is not None else ""), "",
             "## Highlights", "", "| metric | value |", "|---|---|"]
    for key, value in summary["highlights"].items():
        lines.append(f"| {key.replace('_', ' ')} | {_fmt(value)} |")
    if kind == "sft" and "best_val_step" in summary["highlights"]:
        h = summary["highlights"]
        lines += ["", f"The best validation loss so far is {_fmt(h['best_val_loss'])} at step {h['best_val_step']}" +
                  (" - **this checkpoint**." if h["is_best_val"] else ".")]
    gen = {n: s for n, s in summary["series"].items() if n.startswith("gen/") and n not in ("gen/score", "gen/json_valid")}
    if gen:
        lines += ["", "## Generation score per task (at the latest evaluation)", "", "| task | score | measured at step |",
                  "|---|---|---|"]
        lines += [f"| {n[4:]} | {_fmt(s['latest'])} | {s['at']} |" for n, s in gen.items()]
    behaviour = {n: s for n, s in summary["series"].items() if n.startswith(("behaviour/", "reward/", "calibration/"))}
    if behaviour:
        lines += ["", "## Reward parts and behaviour (mean of the last evaluations)", "", "| series | recent mean | latest |",
                  "|---|---|---|"]
        lines += [f"| {n} | {_fmt(s[f'mean_last_{WINDOW}'])} | {_fmt(s['latest'])} |" for n, s in behaviour.items()]
    if kind == "rl":
        lines += ["", "## Reward-hacking watch", ""]
        if summary["flags"]:
            lines += ["**Warnings** (the reward rose while something it should go with did not):", ""] + \
                     [f"- {f}" for f in summary["flags"]] + \
                     ["", "Look at `hacking_watch.png` and `reward_components.png` before using this checkpoint."]
        else:
            lines += ["No warning. The check compares the latest evaluations with the ones before them; it needs about "
                      f"{2 * WINDOW} logged steps to say anything, and only judges while the reward is rising. "
                      "`hacking_watch.png` puts the reward next to exact accuracy, invented values, nulls on readable "
                      "fields, valid JSON and completion length: read them together."]
    if images:
        lines += ["", "## Graphs", ""] + [f"![{img}]({img})" for img in images]
    if lineage:
        lines += ["", "## Built on", ""] + [f"- {e['path']} (LoRA r={e['r']}, alpha={e['alpha']}), merged into the base"
                                             for e in lineage]
    if config:
        lines += ["", "## Run settings", "", "| setting | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in config.items()]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ---- across checkpoints --------------------------------------------------------------------------------------

def write_index(reports_dir: str | Path) -> Path | None:
    """index.md, checkpoints.csv and compare.png over every `<folder>/metrics.json` below `reports_dir`."""
    root = Path(reports_dir)
    rows = []
    for path in sorted(root.glob("*/metrics.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        rows.append({"folder": path.parent.name, "step": data["step"], "kind": data["kind"], "final": data.get("final", False),
                     **data["highlights"]})
    if not rows:
        return None
    rows.sort(key=lambda r: (r["step"], r["folder"]))
    kinds = {r["kind"] for r in rows}
    columns = ["folder", "step", "kind"] + sorted({k for r in rows for k in r if k not in ("folder", "step", "kind", "final")
                                                    and not k.endswith("_measured_at")})
    best: dict[str, str] = {}
    for kind in kinds:
        metric, direction = KINDS[kind]["select"]
        pool = [r for r in rows if r["kind"] == kind and metric in r]
        clean = [r for r in pool if not r.get("hacking_flags")]
        pool = clean or pool                                  # a flagged checkpoint is starred only if nothing else is
        if pool:
            pick = (min if direction == "min" else max)(pool, key=lambda r: r[metric])
            best[kind] = pick["folder"]
    with (root / "checkpoints.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Checkpoints", "",
             "One table per kind of training. Each folder has its own `report.md` with graphs. The starred row is the best "
             "by the selection metric (an RL checkpoint with reward-hacking flags is skipped while a clean one exists); "
             "read the other columns before choosing."]
    for kind in sorted(kinds):
        metric, direction = KINDS[kind]["select"]
        shown = [k for k in KINDS[kind]["highlights"] if any(k in r for r in rows if r["kind"] == kind)]
        if kind == "sft" and any("best_val_step" in r for r in rows if r["kind"] == kind):
            shown += ["best_val_step"]
        if kind == "rl":
            shown += ["hacking_flags"]
        lines += ["", f"## {kind.upper()}  (selection metric: {metric}, {'lower' if direction == 'min' else 'higher'} is better)",
                  "", "| checkpoint | step | " + " | ".join(shown) + " |", "|" + "---|" * (len(shown) + 2)]
        for r in (r for r in rows if r["kind"] == kind):
            mark = " ★" if best.get(kind) == r["folder"] else ""
            lines.append(f"| [{r['folder']}]({r['folder']}/report.md){mark} | {r['step']} | "
                         + " | ".join(_fmt(r.get(c, "")) for c in shown) + " |")
    (root / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _plot_compare(root / "compare.png", rows, kinds)
    return root / "index.md"


def _plot_compare(path: Path, rows: list[dict[str, Any]], kinds: set[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [(kind, m) for kind in sorted(kinds) for m in KINDS[kind]["compare"]
               if any(m in r and r["kind"] == kind for r in rows)]
    if not metrics:
        return
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 2.8 * len(metrics)), squeeze=False)
    for ax, (kind, metric) in zip(axes[:, 0], metrics):
        pts = [(r["step"], r[metric]) for r in rows if r["kind"] == kind and metric in r]
        ax.plot([s for s, _ in pts], [v for _, v in pts], marker="o", linewidth=1.5)
        for s, v in pts:
            ax.annotate(str(s), (s, v), textcoords="offset points", xytext=(0, 5), fontsize=6, ha="center")
        ax.set_title(f"{kind}: {metric.replace('_', ' ')} by checkpoint", fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)
