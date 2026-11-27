"""GRPO with TRL on the extended schema-extraction format (train/rl_prompts.py), rewarded by train/rl_reward.py.

    # 1. build the RL data once: masked + rotated + degraded pages, with the state of every field (runs OCR, slow)
    python -m train.rl build --root data/export/unverified --out data/rl --samples-per-document 4

    # 2. check the risks of the run (add --processor and --with-model when the files / weights are there)
    python -m train.rl check --data data/rl

    # 3. train from the SFT checkpoint
    python -m train.rl train --data data/rl --sft-checkpoint checkpoints/lora_run/final_0000612.pt --out-dir checkpoints/rl

Curriculum (easy to hard): build two folders (`--level soft --p-degrade 0.3`, then `--level strong --p-degrade 0.8`)
and run `train` on the first, then on the second with `--sft-checkpoint` pointing at the first run's checkpoint.

STATUS: written against the TRL GRPO documentation and not run. The parts most likely to need adjusting for the
installed TRL version are marked `# TRL:`. The wiring to the DeepSeek-OCR-2 model assumes its Hugging Face processor
(`AutoProcessor`) builds the same image inputs the model accepts for generation; the SFT code builds them with its own
transform (train/processors), so this is the first thing to check on the first run.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import html
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterable

from .gen_eval import schema_from_prompt, suspicious_reference
from .hub import state_name
from .metrics import HistoryLog
from .rl_data import build_example, rapidocr_lines, rapidocr_text
from .rl_reward import history_metrics, score_rl

log = logging.getLogger("train.rl")

DEFAULT_MODEL = "deepseek-community/DeepSeek-OCR-2"
JSON_COLUMNS = ("spec", "states", "tags", "prompt_schema")      # dicts are stored as JSON strings (Arrow-safe)


# ---- the data -------------------------------------------------------------------------------------------

def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)[:120]


def eligible_anchors(record: dict[str, Any]) -> dict[str, str]:
    """{label path: the text as printed on the page} for the fields it is safe to mask: extracted with high confidence.
    Paths are `section.name`, as `flatten` names them; the printed text is `raw_value` (what OCR will see)."""
    return {f"{f['section']}.{f['name']}": str(f.get("raw_value") or "") for f in record.get("fields", [])
            if f.get("status") == "extracted" and f.get("confidence") == "high" and f.get("raw_value")}


def build_rows(items: Iterable[dict[str, Any]], out_dir: Path, ocr_lines: Callable = rapidocr_lines,
               ocr_text: Callable = rapidocr_text, level: str = "strong", p_degrade: float = 0.8,
               seed: int = 0, p_clean: float = 0.25, eligible: Callable[[str], Any] | None = None,
               stats: collections.Counter | None = None) -> list[dict[str, Any]]:
    """One row per usable item: the image is saved, the reward's metadata is stored next to the prompt.
    Multi-page items, items without a JSON schema in the prompt, and implausible references are left out.
    `eligible(item_id)` gives the fields that may be masked (see `eligible_anchors`); `stats` counts what the OCR
    guards did (see rl_data.build_example)."""
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for n, item in enumerate(items):
        schema = schema_from_prompt(item["prompt"])
        if len(item["images"]) != 1 or schema is None or suspicious_reference(item["label"]):
            continue
        example = build_example(item, random.Random(f"{seed}:{n}:{item['id']}"), ocr_lines, ocr_text,
                                level=level, p_degrade=p_degrade, p_clean=p_clean,
                                eligible=eligible(item["id"]) if eligible else None, stats=stats)
        image_rel = f"images/{safe_name(item['id'])}.png"
        example["images"][0].save(out_dir / image_rel)
        spec = example["spec"]
        wrapped = schema_from_prompt(example["prompt"])
        rows.append({"id": item["id"], "prompt": example["prompt"], "image": image_rel,
                     "reference": example["reference"], "upright": float(example["upright"]),
                     "base_schema": json.dumps(schema),
                     "spec": json.dumps(spec), "states": json.dumps(example["states"]),
                     "tags": json.dumps(example["tags"]), "prompt_schema": json.dumps(wrapped)})
    return rows


def write_rows(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "rl.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return path


def load_rows(data_dir: Path) -> list[dict[str, Any]]:
    with (data_dir / "rl.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_dataset_row(row: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    """The row TRL reads: a conversational `prompt` with an image placeholder, the `image`, and every other column
    forwarded to the reward function as a keyword argument."""
    out = {k: v for k, v in row.items() if k not in ("prompt", "image", "id")}
    out["prompt"] = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["prompt"]}]}]
    out["image"] = str(data_dir / row["image"])
    return out


# ---- the reward ----------------------------------------------------------------------------------------

def completion_text(completion: Any) -> str:
    """A completion as TRL hands it over: a string, or a list of chat messages (the answer is the last one)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        content = completion[-1].get("content", "")
        if isinstance(content, list):                           # [{"type": "text", "text": ...}]
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return str(content)
    return ""


def _load(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) and value else None


def score_batch(completions: Any, columns: dict[str, Any]) -> list[dict[str, Any]]:
    """`score_rl` for every completion, each with its own row of the dataset columns (they arrive as lists)."""
    return [score_rl(completion_text(completion), columns["reference"][i], _load(columns["spec"][i]),
                     states=_load(columns["states"][i]) or {}, upright=columns["upright"][i],
                     tags=_load(columns["tags"][i]) or {}, prompt_schema=_load(columns["prompt_schema"][i]))
            for i, completion in enumerate(completions or [])]


def reward_function(prompts=None, completions=None, **columns) -> list[float]:  # TRL: reward_funcs signature
    """The reward of every completion. The dataset columns arrive as lists, one entry per completion."""
    return [r["total"] for r in score_batch(completions, columns)]


class RewardTracker:
    """The reward function that also remembers every component of every score, so the report can draw each one on its
    own and hold the total against measures it does not pay for (exact accuracy, invented values, nulls on readable
    fields). `drain()` returns the series for the steps since the last call and starts over."""

    __name__ = "reward_rl"                                         # TRL logs the mean reward under this name

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def __call__(self, prompts=None, completions=None, **columns) -> list[float]:
        results = score_batch(completions, columns)
        self.rows.extend(results)
        return [r["total"] for r in results]

    def drain(self) -> dict[str, float]:
        metrics, self.rows = history_metrics(self.rows), []
        return metrics


class RLCheckpointRecorder:
    """What happens at TRL's log and save events: the numbers go to the history, and every save gets the RL metrics
    report (with the reward components, the hacking watch and the rest) and its own folder in the Hub repo. A failure
    here only costs the report or the upload, never the run."""

    def __init__(self, out_dir: str | Path, prefix: str = "rl", tracker: RewardTracker | None = None, hub: Any = None,
                 run_config: dict[str, Any] | None = None, reports: bool = True, save_adapter: Callable | None = None,
                 lineage: Callable[[], list] | None = None, initial_step: int = 0) -> None:
        self.out_dir, self.prefix, self.tracker, self.hub = Path(out_dir), prefix, tracker, hub
        self.run_config, self.reports, self.save_adapter, self.lineage = run_config, reports, save_adapter, lineage
        self.history = HistoryLog(self.out_dir / "metrics.jsonl", initial_step)

    def on_log(self, step: int, logs: dict[str, Any]) -> None:
        for key, value in (logs or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.history.record(step, f"train/{key}", value)
        if self.tracker is not None:
            for key, value in self.tracker.drain().items():
                self.history.record(step, key, value)

    def on_save(self, step: int, checkpoint_dir: str | Path, final: bool = False) -> None:
        name = state_name(self.prefix, step, final)
        checkpoint_dir = Path(checkpoint_dir)
        try:
            if self.save_adapter is not None and checkpoint_dir.is_dir():
                self.save_adapter(checkpoint_dir / "adapter_state.pt", step)    # what `--init-from` loads later
            report = None
            if self.reports:
                from .ckpt_report import build_checkpoint_report, write_index
                report = build_checkpoint_report(
                    self.history.history, step, self.out_dir / "reports" / name, kind="rl", name=name,
                    run_config=self.run_config, final=final, lineage=self.lineage() if self.lineage else None)
                write_index(self.out_dir / "reports")
            if self.hub is not None:
                self.hub.push_folder(step, checkpoint_dir, final=final, report_dir=report)
        except Exception as error:                                              # noqa: BLE001
            log.warning("could not record the checkpoint of step %d: %s", step, error)

    def close(self) -> None:
        self.history.close()
        if self.hub is not None:
            self.hub.close()


def make_trainer_callback(recorder: RLCheckpointRecorder):
    """A transformers TrainerCallback that feeds the recorder (on the main process only)."""
    from transformers import TrainerCallback

    class RecorderCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero:
                recorder.on_log(state.global_step, logs or {})

        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                recorder.on_save(state.global_step, Path(args.output_dir) / f"checkpoint-{state.global_step}")

    return RecorderCallback()


def load_wrapper(args: argparse.Namespace, dtype: Any):
    """The model for GRPO. With `--sft-checkpoint` the earlier adapter (and the ones it sits on) is merged into the base
    and a NEW LoRA (--lora-r/--lora-alpha) is attached for this stage; `--same-adapter` instead keeps training the
    same adapter, which needs the same --lora-r as the checkpoint."""
    from .hub import resolve_resume
    from .model import DeepSeekOCR, parse_targets

    wrapper = DeepSeekOCR.from_pretrained(args.model_id, lang_trained=False, vision_trained=False, dtype=dtype)
    new = {} if args.lora_targets is None else {"target_modules": parse_targets(args.lora_targets)}
    if args.sft_checkpoint and not args.same_adapter:
        old = args.init_lora_targets
        wrapper.continue_from(args.sft_checkpoint, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
                              old_r=args.init_lora_r, old_alpha=args.init_lora_alpha,
                              old_target_modules=None if old is None else parse_targets(old), token=args.hub_token, **new)
    else:
        wrapper.add_lora(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, **new)
        if args.sft_checkpoint:
            wrapper.load_checkpoint(resolve_resume(args.sft_checkpoint, args.hub_token), map_location="cpu")
    return wrapper


def trl_max_steps(args: argparse.Namespace) -> int:
    """TRL stops at max_steps when it is positive, so a run measured in epochs passes -1."""
    return -1 if args.epochs else args.max_steps


def supported_kwargs(config_class: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drops settings the installed TRL version does not have (its config class changes between releases)."""
    known = {f.name for f in dataclasses.fields(config_class)}
    unknown = sorted(set(kwargs) - known)
    if unknown:
        log.warning("this TRL version has no %s; leaving them out", ", ".join(unknown))
    return {k: v for k, v in kwargs.items() if k in known}


# ---- the commands --------------------------------------------------------------------------------------

def cmd_build(args: argparse.Namespace) -> None:
    from .dataset import DocumentDataset
    from .processors import DatasetProcessor

    dataset = DocumentDataset(args.root, args.split, DatasetProcessor(tasks=["extract_to_schema"]),
                              samples_per_document=args.samples_per_document, max_documents=args.max_documents,
                              seed=args.seed)
    out_dir = Path(args.out)
    by_id = {record["id"]: record for record in dataset.records}
    stats: collections.Counter = collections.Counter()
    rows = build_rows((dataset[i] for i in range(len(dataset))), out_dir, level=args.level,
                      p_degrade=args.p_degrade, seed=args.seed, p_clean=args.p_clean, stats=stats,
                      eligible=lambda item_id: eligible_anchors(by_id.get(item_id.split("#")[0], {})))
    path = write_rows(rows, out_dir)
    (out_dir / STATS_FILE).write_text(json.dumps(dict(stats), indent=2, sort_keys=True), encoding="utf-8")
    states = [s for r in rows for s in json.loads(r["states"]).values()]
    log.info("wrote %d examples to %s (fields: %d answerable, %d partial, %d must_abstain)", len(rows), path,
             states.count("answerable"), states.count("partial"), states.count("must_abstain"))
    log.info("OCR guards: %s", ", ".join(f"{k}={v}" for k, v in sorted(stats.items()) if not k.startswith("state_")))


STATS_FILE = "build_stats.json"
_COLORS = {"answerable": "#d4edda", "partial": "#fff3cd", "must_abstain": "#f8d7da"}


def write_preview(data_dir: Path, out: Path, n: int = 20, seed: int = 0) -> Path:
    """A page of `n` random examples to look at before spending GPU time: the image, and every field with its state
    (green readable, yellow partial, red unreadable) and the reason tag, next to the build counters."""
    rows = load_rows(data_dir)
    random.Random(seed).shuffle(rows)
    out.mkdir(parents=True, exist_ok=True)
    stats_path = data_dir / STATS_FILE
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    parts = ["<!doctype html><meta charset='utf-8'><title>RL preview</title><style>body{font:14px sans-serif;margin:16px}"
             ".ex{display:flex;gap:16px;margin:0 0 24px;border-top:1px solid #ccc;padding-top:12px}"
             ".ex img{max-width:420px;max-height:560px}td{padding:2px 8px}</style>",
             "<h1>RL preview</h1><pre>" + html.escape(json.dumps(stats, indent=1, sort_keys=True), quote=False) + "</pre>"]
    for row in rows[:n]:
        states, tags = json.loads(row["states"]), json.loads(row.get("tags") or "{}")
        reference = json.loads(row["reference"])
        cells = "".join(
            f"<tr style='background:{_COLORS.get(state, '#eee')}'><td>{html.escape(path)}</td><td>{state}</td>"
            f"<td>{html.escape(tags.get(path, ''))}</td><td>{html.escape(str(_leaf(reference, path)))}</td></tr>"
            for path, state in sorted(states.items()))
        parts.append(f"<div class='ex'><img src='{html.escape(str((data_dir / row['image']).resolve()))}'>"
                     f"<div><b>{html.escape(row['id'])}</b> upright turn {row['upright']:.0f}&deg;"
                     f"<table><tr><th>field</th><th>state</th><th>tag</th><th>reference</th></tr>{cells}</table></div></div>")
    path = out / "preview.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _leaf(tree: Any, path: str) -> Any:
    """The value at a `flatten` path in the plain reference (best effort: list items are shown as their path)."""
    from .gen_eval import flatten
    return flatten(tree).get(path, "")


def cmd_preview(args: argparse.Namespace) -> None:
    print(write_preview(Path(args.data), Path(args.out), args.n, args.seed))


def _dtype(args: argparse.Namespace):
    import torch
    return torch.bfloat16 if args.precision == "bf16" else torch.float32


def cmd_check(args: argparse.Namespace) -> None:
    from . import rl_checks

    data_dir = Path(args.data)
    rows = load_rows(data_dir) if (data_dir / "rl.jsonl").exists() else []
    checks = rl_checks.preflight(rows, data_dir, args.max_completion_tokens, args.batch_size, args.grad_accum,
                                 args.num_generations, reward_function,
                                 model_id=args.model_id if args.processor else None, ocr_mode=args.ocr_mode)
    if args.with_model and rows:
        checks.extend(rl_checks.check_model(args.model_id, lambda: load_wrapper(args, _dtype(args)), rows[0], data_dir))
    print(rl_checks.report(checks))
    raise SystemExit(1 if rl_checks.failed(checks) else 0)


def cmd_train(args: argparse.Namespace) -> None:
    from . import rl_checks
    rows = load_rows(Path(args.data))
    if not args.skip_checks:
        checks = rl_checks.preflight(rows, Path(args.data), args.max_completion_tokens, args.batch_size,
                                     args.grad_accum, args.num_generations, reward_function,
                                     model_id=args.model_id, ocr_mode=args.ocr_mode)
        print(rl_checks.report(checks))
        if rl_checks.failed(checks):
            raise SystemExit("preflight failed (fix the lines marked FAIL, or pass --skip-checks)")
    from datasets import Dataset, Image as HFImage
    from transformers import AutoProcessor
    from trl import GRPOConfig, GRPOTrainer                         # TRL: imported here so the rest runs without it

    from .hub import HubPusher

    data_dir = Path(args.data)
    dataset = Dataset.from_list([to_dataset_row(r, data_dir) for r in rows])
    dataset = dataset.cast_column("image", HFImage())

    wrapper = load_wrapper(args, _dtype(args))
    processor = AutoProcessor.from_pretrained(args.model_id)

    prefix = "rl" if args.state_prefix is None else args.state_prefix
    run_config = {k: v for k, v in vars(args).items() if k != "hub_token"}
    hub = None
    if args.hub_repo:
        hub = HubPusher(args.hub_repo, token=args.hub_token, private=args.hub_private, every_save=args.hub_every_save,
                        run_config=run_config, stage_dir=Path(args.out_dir) / "hub_stage", prefix=prefix,
                        keep_last=args.hub_keep_last, state=args.hub_state)
    tracker = RewardTracker()
    recorder = RLCheckpointRecorder(
        args.out_dir, prefix=prefix, tracker=tracker, hub=hub, run_config=run_config, reports=args.reports,
        lineage=lambda: wrapper.lineage,
        save_adapter=lambda path, step: wrapper.save_checkpoint(
            path, step=step, extra={"kind": "rl"}, model_state=wrapper.trainable_state_dict()))

    config = GRPOConfig(**supported_kwargs(GRPOConfig, {                                  # TRL: field names
        "output_dir": args.out_dir, "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_tokens, "learning_rate": args.lr, "beta": args.beta,
        "loss_type": args.loss_type, "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum, "max_steps": trl_max_steps(args),
        "num_train_epochs": args.epochs if args.epochs else 1.0, "bf16": args.precision == "bf16",
        "logging_steps": 1, "save_steps": args.save_steps, "report_to": ["tensorboard"],
        "remove_unused_columns": False, "seed": args.seed}))
    trainer = GRPOTrainer(model=wrapper.model, reward_funcs=[tracker], args=config, train_dataset=dataset,
                          processing_class=processor, callbacks=[make_trainer_callback(recorder)])   # TRL: constructor
    try:
        trainer.train()
        trainer.save_model(args.out_dir)
        recorder.on_save(trainer.state.global_step, args.out_dir, final=True)
    finally:
        recorder.close()               # waits for the uploads still running


def _model_args(p: argparse.ArgumentParser) -> None:
    """How the model is built: shared by `train` and `check` so the check loads exactly what the run will."""
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--sft-checkpoint", default=None,
                   help="start from this earlier adapter (.pt, or hf://user/repo/sft_step_0000200/adapter_state.pt): it is "
                        "merged into the base and a NEW LoRA (--lora-r ...) is attached for this stage")
    p.add_argument("--same-adapter", action="store_true",
                   help="instead keep training the checkpoint's own adapter (--lora-r must equal its rank)")
    p.add_argument("--lora-r", type=int, default=16, help="rank of the LoRA of THIS stage")
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", default=None,
                   help="regex or comma-separated module names for THIS stage's LoRA (default: the language model's projections)")
    p.add_argument("--init-lora-r", type=int, default=None,
                   help="the earlier adapter's rank, only for checkpoints that predate recording it")
    p.add_argument("--init-lora-alpha", type=int, default=None, help="its alpha (old checkpoints only)")
    p.add_argument("--init-lora-targets", default=None, help="its target modules, comma separated (old checkpoints only)")
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--hub-token", default=None, help="Hugging Face token (default: HF_TOKEN or the cached login)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser("preview", help="write preview.html: sample examples with each field's state, to look at before training")
    v.add_argument("--data", required=True, help="the folder `build` wrote")
    v.add_argument("--out", default="preview")
    v.add_argument("--n", type=int, default=20)
    v.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("build", help="build the RL data (masks, rotation, degradation, field states)")
    b.add_argument("--root", required=True, help="the export folder with train.jsonl")
    b.add_argument("--out", required=True, help="where the RL data is written")
    b.add_argument("--split", default="train")
    b.add_argument("--level", choices=["soft", "strong"], default="strong", help="degradation level (train/degrade.py)")
    b.add_argument("--p-degrade", type=float, default=0.8, help="share of examples that get rotation / noise / blur")
    b.add_argument("--p-clean", type=float, default=0.25,
                   help="share of examples with no field masked (the page may still be turned and degraded)")
    b.add_argument("--samples-per-document", type=int, default=4)
    b.add_argument("--max-documents", type=int, default=None)
    b.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train", help="GRPO with TRL (written against the docs; not run)")
    t.add_argument("--data", required=True, help="the folder `build` wrote")
    t.add_argument("--out-dir", default="checkpoints/rl")
    _model_args(t)
    t.add_argument("--num-generations", type=int, default=8, help="completions sampled per prompt (the GRPO group)")
    t.add_argument("--max-completion-tokens", type=int, default=1536)
    t.add_argument("--lr", type=float, default=5e-6)
    t.add_argument("--beta", type=float, default=0.0, help="KL weight to the SFT model; 0 drops the reference model")
    t.add_argument("--loss-type", default="dapo")
    t.add_argument("--batch-size", type=int, default=8, help="per device; TRL needs it to fit the generation group")
    t.add_argument("--grad-accum", type=int, default=2)
    t.add_argument("--max-steps", type=int, default=500, help="optimizer steps; ignored when --epochs is given")
    t.add_argument("--epochs", type=float, default=None,
                   help="passes over the RL data (TRL num_train_epochs); when set, --max-steps no longer stops the run")
    t.add_argument("--save-steps", type=int, default=100)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--ocr-mode", default="base", help="the DeepSeek-OCR resolution mode the SFT run used")
    t.add_argument("--skip-checks", action="store_true", help="do not run the preflight checks first")
    t.add_argument("--state-prefix", default=None,
                   help="put before every checkpoint, report and hub folder (rl_step_0000100); default rl, '' for none")
    t.add_argument("--reports", action=argparse.BooleanOptionalAction, default=True,
                   help="a metrics report with graphs (reward components, hacking watch, calibration) at every save")
    t.add_argument("--hub-repo", default=None, help="push every save to this Hugging Face repo, one folder each")
    t.add_argument("--hub-private", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--hub-every-save", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--hub-keep-last", action="store_true")
    t.add_argument("--hub-state", action=argparse.BooleanOptionalAction, default=True,
                   help="include the trainer's optimizer files in each hub folder (--no-hub-state: much smaller)")
    c = sub.add_parser("check", help="check the risks of the run: data, reward sanity, batching, TRL, processor, model")
    c.add_argument("--data", required=True)
    _model_args(c)
    c.add_argument("--ocr-mode", default="base", help="the DeepSeek-OCR resolution mode the SFT run used")
    c.add_argument("--processor", action="store_true", help="compare the HF processor with the SFT transform (downloads)")
    c.add_argument("--with-model", action="store_true", help="load the model and generate once (needs weights)")
    c.add_argument("--max-completion-tokens", type=int, default=1536)
    c.add_argument("--num-generations", type=int, default=8)
    c.add_argument("--batch-size", type=int, default=8)
    c.add_argument("--grad-accum", type=int, default=2)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args(argv)
    {"build": cmd_build, "train": cmd_train, "check": cmd_check, "preview": cmd_preview}[args.command](args)


if __name__ == "__main__":
    main()
