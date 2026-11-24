"""Does the model generate correct data? Generation-based validation.

The validation loss says how well the model predicts the reference text token by token. This runs the model the way
it is used: it gets the page and the prompt only, generates the answer, and the answer is compared with the reference.

    # a saved checkpoint (the adapter you trained), on the validation split
    python -m train.gen_eval --config config/train.yaml --checkpoint checkpoints/lora_run/final_0000612.pt
    python -m train.gen_eval --config config/train.yaml --no-adapter          # the base model, for comparison

During training the same check runs before the first step (the base model) and at the end, and every
`--gen-eval-every` steps if asked: `--gen-eval`.

How an answer is scored (chosen from the reference, so every task is covered):
  * JSON answers (extract, schema, tables, figures, ...): `json_valid` (does it parse), and the F1 of the (path, value)
    pairs against the reference, plus `path_f1` (right structure, wrong values still count). Lists of records are
    matched by their `name` / `id` and not by position, so a different order is not an error.
  * short answers (classify, a field's value, a chart question): normalised exact match.
  * long text (OCR): character similarity (difflib ratio) of the normalised text.
The score of every example is between 0 and 1; the summary gives the mean per task and overall. Every generated answer
is written to `generations_step_N.jsonl` so you can read them.
"""
from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch

log = logging.getLogger("train.gen")

ID_KEYS = ("name", "id", "field", "key", "field_id", "table_id", "figure_id", "element_id", "row_id", "label")
SHORT_ANSWER_WORDS = 12


# ---- scoring ----------------------------------------------------------------------------------------

def normalise(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def extract_json(text: str) -> Any | None:
    """The first JSON value in a model answer, also when it is wrapped in a code fence or has text around it."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except ValueError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except ValueError:
                continue
    return None


def flatten(value: Any, prefix: str = "") -> dict[str, str]:
    """{path: normalised leaf}. A list of records is keyed by each record's name / id, so order does not matter."""
    out: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            label = str(index)
            if isinstance(item, dict):
                for id_key in ID_KEYS:
                    if isinstance(item.get(id_key), (str, int)) and str(item[id_key]).strip():
                        label = f"{id_key}={normalise(item[id_key])}"
                        break
            out.update(flatten(item, f"{prefix}[{label}]"))
    else:
        out[prefix] = normalise("" if value is None else value)
    return out


def _f1(predicted: set, reference: set) -> float:
    if not predicted and not reference:
        return 1.0
    hits = len(predicted & reference)
    if not hits:
        return 0.0
    precision, recall = hits / len(predicted), hits / len(reference)
    return 2 * precision * recall / (precision + recall)


def score_example(prediction: str, reference: str) -> dict[str, Any]:
    """Metrics of one answer. `kind` says how it was judged; `score` is the number to average."""
    ref_json = extract_json(reference) if reference.lstrip()[:1] in "{[" else None
    if ref_json is not None:
        pred_json = extract_json(prediction)
        valid = pred_json is not None
        ref_pairs = flatten(ref_json)
        pred_pairs = flatten(pred_json) if valid else {}
        value_f1 = _f1(set(pred_pairs.items()), set(ref_pairs.items()))
        path_f1 = _f1(set(pred_pairs), set(ref_pairs))
        return {"kind": "json", "json_valid": valid, "path_f1": round(path_f1, 4), "value_f1": round(value_f1, 4),
                "score": round(value_f1, 4)}
    if len(reference.split()) <= SHORT_ANSWER_WORDS:
        match = normalise(prediction) == normalise(reference)
        return {"kind": "short", "exact_match": match, "score": float(match)}
    ratio = difflib.SequenceMatcher(None, normalise(prediction), normalise(reference), autojunk=False).ratio()
    return {"kind": "text", "similarity": round(ratio, 4), "score": round(ratio, 4)}


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean score overall and per task, and the share of JSON answers that parse."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_task[row["task"]].append(row)

    def mean(rows: list[dict], key: str) -> float | None:
        values = [r[key] for r in rows if key in r]
        return round(sum(float(v) for v in values) / len(values), 4) if values else None

    out: dict[str, Any] = {"examples": len(results), "score": mean(results, "score"),
                           "json_valid": mean(results, "json_valid"), "per_task": {}}
    for task, rows in sorted(by_task.items()):
        out["per_task"][task] = {"n": len(rows), "score": mean(rows, "score"), "json_valid": mean(rows, "json_valid"),
                                 "path_f1": mean(rows, "path_f1"), "kind": rows[0]["kind"]}
    return out


# ---- generation -------------------------------------------------------------------------------------

class GenerationEvaluator:
    """Generates answers for a fixed set of validation examples and scores them.

    `run(model, step)` is collective under torchrun: every rank generates its share and rank 0 gets everything.
    """

    def __init__(self, root: str | Path, transform, adapter: Callable[[dict], dict] | None, processor=None,
                 split: str = "val", samples_per_task: int = 2, max_new_tokens: int = 2048, batch_size: int = 4,
                 max_documents: int | None = None, precision: str = "auto", out_dir: str | Path | None = None,
                 draws_per_document: int = 6, preset: str = "structured"):
        from .dataset import DocumentDataset
        from .processors import DatasetProcessor

        self.transform, self.adapter = transform, adapter
        self.max_new_tokens, self.batch_size, self.precision = max_new_tokens, batch_size, precision
        self.out_dir = Path(out_dir) if out_dir else None
        dataset = DocumentDataset(root, split, processor or DatasetProcessor(preset=preset),
                                  samples_per_document=draws_per_document, max_documents=max_documents,
                                  fits=transform.fits)
        taken: dict[str, int] = defaultdict(int)
        self.skipped_long = 0
        self.items: list[dict] = []
        for index in range(len(dataset)):
            item = dataset[index]
            if taken[item["task"]] >= samples_per_task:
                continue
            if len(transform.tokenizer.encode(item["label"], add_special_tokens=False)) + 16 > max_new_tokens:
                self.skipped_long += 1                       # the reference is longer than the room to generate
                continue
            taken[item["task"]] += 1
            self.items.append(item)
        # similar lengths together: a batch ends when its longest answer does
        self.items.sort(key=lambda it: len(it["label"]))
        log.info("generation check: %d examples (%s), %d skipped as longer than %d new tokens",
                 len(self.items), ", ".join(f"{t}:{n}" for t, n in sorted(taken.items())), self.skipped_long,
                 max_new_tokens)

    # -- one batch -------------------------------------------------------------------------------
    def _inputs(self, chunk: list[dict]) -> tuple[dict[str, Any], int, int]:
        features = [self.transform(item["images"], item["prompt"], item["label"]) for item in chunk]
        prompt_lengths = [int((f["labels"] != -100).nonzero()[0]) for f in features]   # where the answer starts
        batch = self.transform.make_collator()(features)
        kwargs = self.adapter(batch) if self.adapter is not None else dict(batch)
        pad = self.transform._pad_token_id()
        width = max(prompt_lengths)
        input_ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        attention = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, (feature, length) in enumerate(zip(features, prompt_lengths)):      # left padding for generation
            input_ids[row, width - length:] = feature["input_ids"][:length]
            attention[row, width - length:] = 1
        kwargs.pop("labels", None)
        kwargs.update(input_ids=input_ids, attention_mask=attention)
        longest_answer = max(len(self.transform.tokenizer.encode(item["label"], add_special_tokens=False))
                             for item in chunk)
        return kwargs, width, min(self.max_new_tokens, int(longest_answer * 1.25) + 16)

    @torch.no_grad()
    def generate(self, model, items: list[dict]) -> list[str]:
        from . import distributed as dd
        from .processors.base import to_device
        from .processors.deepseekOcr import EOS_ID

        inner = dd.unwrap(model)
        device = dd.model_device(model)
        amp_dtype, _ = dd.resolve_precision(self.precision, device)
        was_training = model.training
        model.eval()
        answers: list[str] = []
        try:
            for start in range(0, len(items), self.batch_size):
                kwargs, width, budget = self._inputs(items[start:start + self.batch_size])
                kwargs = to_device(kwargs, device)
                with dd.autocast(device, amp_dtype):
                    out = inner.base.generate(**kwargs, max_new_tokens=budget, do_sample=False, use_cache=True,
                                              eos_token_id=EOS_ID, pad_token_id=self.transform._pad_token_id())
                for row in out[:, width:].tolist():
                    row = row[:row.index(EOS_ID)] if EOS_ID in row else row   # up to the end-of-text token
                    answers.append(self.transform.tokenizer.decode(row, skip_special_tokens=True))
        finally:
            model.train(was_training)
        return answers

    # -- the whole check -------------------------------------------------------------------------
    def run(self, model, step: int = 0, logger=None) -> dict[str, Any]:
        import torch.distributed as torch_dist
        from . import distributed as dd

        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()                      # give the training run's cached blocks back for the KV cache
        share = self.items[dd.rank()::dd.world_size()]
        answers = self.generate(model, share) if share else []
        rows = []
        for item, answer in zip(share, answers):
            rows.append({"id": item["id"], "task": item["task"], "reference": item["label"], "prediction": answer,
                         **score_example(answer, item["label"])})
        if dd.is_dist():
            gathered: list[Any] = [None] * dd.world_size()
            torch_dist.all_gather_object(gathered, rows)
            rows = [row for part in gathered for row in part]
        summary = summarise(rows)
        if dd.is_main():
            if self.out_dir is not None:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                path = self.out_dir / f"generations_step_{step:07d}.jsonl"
                path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
            scalars = {"gen/score": summary["score"] or 0.0, "gen/json_valid": summary["json_valid"] or 0.0}
            scalars.update({f"gen/{task}": info["score"] for task, info in summary["per_task"].items()})
            if logger is not None:
                logger.scalars(step, scalars)
            log.info("generation check at step %d: score %.3f, json valid %s, %d examples, %.0f s", step,
                     summary["score"] or 0.0, summary["json_valid"], summary["examples"],
                     time.perf_counter() - started)
            for task, info in summary["per_task"].items():
                log.info("  %-18s n=%d score=%.3f%s", task, info["n"], info["score"] or 0.0,
                         "" if info["json_valid"] is None else f" json_valid={info['json_valid']:.2f}")
        return summary


# ---- the command: score a saved checkpoint --------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    from corpus.logs import configure

    from . import distributed as dd
    from .hub import resolve_resume
    from .model import DeepSeekOCR
    from .processors import batch_adapter_for
    from .train import build_transform, parse_args, weights_dtype

    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--checkpoint", help="a .pt checkpoint or hf://user/repo/last/training_state.pt")
    own.add_argument("--no-adapter", action="store_true", help="score the base model without the adapter")
    own.add_argument("--samples-per-task", type=int, default=3)
    own.add_argument("--max-new-tokens", type=int, default=2048)
    own.add_argument("--gen-batch-size", type=int, default=4)
    own.add_argument("--split", default="val")
    known, rest = own.parse_known_args(argv)
    if any(flag in (rest or []) for flag in ("-h", "--help")):
        print(__doc__)
        print("options of this command: --checkpoint FILE | --no-adapter, --samples-per-task N, --max-new-tokens N, "
              "--gen-batch-size N, --split NAME; every other option is the one of `python -m train.train` "
              "(use the same --config as the run)\n")
    args = parse_args(rest)
    configure(args)
    if not known.checkpoint and not known.no_adapter:
        raise SystemExit("give --checkpoint <file>, or --no-adapter to score the base model")

    device = dd.init_distributed()
    dtype, _ = dd.resolve_precision(args.precision, device)
    transform = build_transform(args)
    log.info("loading %s (%s)", args.model_id, str(weights_dtype(args, dtype)).removeprefix("torch."))
    model = DeepSeekOCR.from_pretrained(args.model_id, lang_trained=False, vision_trained=False,
                                        dtype=weights_dtype(args, dtype), device_map={"": str(device)})
    step = 0
    if not known.no_adapter:
        if args.lora:
            model.add_lora(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        meta = model.load_checkpoint(resolve_resume(known.checkpoint, args.hub_token), map_location="cpu")
        step = int(meta.get("step") or 0)
        log.info("loaded the checkpoint from step %d", step)
    evaluator = GenerationEvaluator(
        args.root, transform, batch_adapter_for(args.model_id), split=known.split,
        samples_per_task=known.samples_per_task, max_new_tokens=known.max_new_tokens,
        batch_size=known.gen_batch_size, preset=args.preset, max_documents=args.val_max_documents, precision=args.precision,
        out_dir=Path(args.out_dir) / ("generations_base" if known.no_adapter else "generations"))
    summary = evaluator.run(model, step)
    print(json.dumps(summary, indent=2))
    dd.cleanup()


if __name__ == "__main__":
    main()
