"""Checks for the risks of the GRPO run (train/rl.py). Each returns a `Check` with a status:
  pass  verified.   warn  a risk worth reading, the run can go on.   fail  the run would be wasted or break.
  skip  could not be checked here (no network, no GPU, TRL not installed), with the reason.

    python -m train.rl check --data data/rl                  # data, reward sanity, batching, TRL API
    python -m train.rl check --data data/rl --processor      # + the processor against the SFT transform (downloads)
    python -m train.rl check --data data/rl --with-model --sft-checkpoint X.pt   # + a real generation (needs weights)

`train` runs the cheap ones first and stops on a fail (`--skip-checks` to override).
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .gen_eval import flatten, is_null, record_label
from .rl_prompts import AnswerSpec, wrap_answer
from .rl_reward import reward_rl, score_rl

REQUIRED_COLUMNS = ("prompt", "image", "reference", "upright", "spec", "states", "tags", "prompt_schema", "base_schema")
GRPO_CONFIG_NEEDED = ("num_generations", "max_completion_length")                 # without these the run cannot work
GRPO_CONFIG_USED = ("learning_rate", "beta", "loss_type", "per_device_train_batch_size", "gradient_accumulation_steps",
                    "max_steps", "bf16", "remove_unused_columns", "report_to", "save_steps", "logging_steps", "seed")
GRPO_TRAINER_ARGS = ("model", "reward_funcs", "args", "train_dataset", "processing_class")
GRPO_TRAINER_OPTIONAL = ("callbacks",)       # the reports and hub uploads hang on it; training itself does not
CHARS_PER_TOKEN = 3.0           # rough, for JSON answers; only used to flag answers that cannot fit the budget


@dataclass
class Check:
    name: str
    status: str                 # pass | warn | fail | skip
    detail: str

    def line(self) -> str:
        return f"[{self.status.upper():4}] {self.name}: {self.detail}"


def report(checks: list[Check]) -> str:
    return "\n".join(c.line() for c in checks)


def failed(checks: list[Check]) -> bool:
    return any(c.status == "fail" for c in checks)


# ---- the answers a row can be judged by ----------------------------------------------------------------

def _null_out(node: Any, paths: set[str], prefix: str = "") -> Any:
    """The reference with the leaves at `paths` set to null (paths are `flatten` paths)."""
    if isinstance(node, dict):
        return {k: _null_out(v, paths, f"{prefix}.{k}" if prefix else str(k)) for k, v in node.items()}
    if isinstance(node, list):
        return [_null_out(v, paths, f"{prefix}[{record_label(v, i)}]") for i, v in enumerate(node)]
    return None if prefix in paths else node


def _row_parts(row: dict[str, Any]):
    spec = AnswerSpec.from_dict(json.loads(row["spec"]) if row["spec"] else None)
    return (json.loads(row["reference"]), spec, json.loads(row["states"]), json.loads(row["tags"]),
            json.loads(row["prompt_schema"]) if row["prompt_schema"] else None, json.loads(row["base_schema"]))


def _answer(row: dict[str, Any], null_paths: set[str]) -> str:
    reference, spec, states, tags, _, base = _row_parts(row)
    values = _null_out(reference, null_paths)
    if spec is None:
        return json.dumps(values)
    reasons = {p: t for p, t in tags.items() if states.get(p) != "answerable"}
    return wrap_answer(values, base, spec, upright=row["upright"], reasons=reasons)


def oracle_answer(row: dict[str, Any]) -> str:
    """The best answer: every value, except null on the fields the page makes unreadable."""
    states = json.loads(row["states"])
    return _answer(row, {p for p, s in states.items() if s == "must_abstain"})


def guess_answer(row: dict[str, Any]) -> str:
    """Invents a value for the unreadable fields (the reference value, confidently)."""
    return _answer(row, set())


def blank_answer(row: dict[str, Any]) -> str:
    """Abstains on everything."""
    return _answer(row, set(flatten(json.loads(row["reference"]))))


def _score(row: dict[str, Any], answer: str) -> dict[str, Any]:
    reference, spec, states, tags, schema, _ = _row_parts(row)
    return score_rl(answer, row["reference"], spec, states, row["upright"], tags, schema)


# ---- data ------------------------------------------------------------------------------------------------

def check_data(rows: list[dict[str, Any]], data_dir: Path, max_completion_tokens: int) -> list[Check]:
    out: list[Check] = []
    if not rows:
        return [Check("data rows", "fail", "no rows; run `python -m train.rl build` first")]
    broken = []
    for row in rows:
        missing = [c for c in REQUIRED_COLUMNS if c not in row]
        if missing:
            broken.append(f"{row.get('id')}: missing {missing}")
            continue
        try:
            with Image.open(data_dir / row["image"]) as image:
                image.load()
            for column in ("spec", "states", "tags", "prompt_schema", "base_schema", "reference"):
                if row[column]:
                    json.loads(row[column])
        except Exception as error:                                   # noqa: BLE001 - report, do not stop
            broken.append(f"{row.get('id')}: {type(error).__name__}: {error}")
    if broken:
        extra = " (rebuild the data: older builds lack `base_schema`)" if any("base_schema" in b for b in broken) else ""
        return [Check("data rows", "fail", f"{len(broken)} of {len(rows)} rows unusable, e.g. {broken[0]}{extra}")]
    out.append(Check("data rows", "pass", f"{len(rows)} rows, every image opens and every JSON column parses"))

    # reward sanity: the best answer must beat inventing values on unreadable fields, and beat abstaining on all
    bad_order, bad_schema, longest = [], [], 0
    for row in rows:
        oracle_text = oracle_answer(row)
        longest = max(longest, len(oracle_text))
        oracle, guess, blank = (_score(row, t) for t in (oracle_text, guess_answer(row), blank_answer(row)))
        states = json.loads(row["states"])
        has_masked = "must_abstain" in states.values()
        # abstaining on everything is only a different answer when something is readable (else it IS the best answer)
        has_readable = any(states.get(p) != "must_abstain" and not is_null(v)
                           for p, v in flatten(json.loads(row["reference"])).items())
        if (has_masked and not oracle["total"] > guess["total"]) or (has_readable and not oracle["total"] > blank["total"]):
            bad_order.append(row["id"])
        if oracle["parts"].get("schema", 1.0) != 1.0:
            bad_schema.append(row["id"])
    out.append(Check("reward ordering", "fail" if bad_order else "pass",
                     f"{len(bad_order)} rows where the best answer does not beat guessing / abstaining, e.g. {bad_order[0]}"
                     if bad_order else "on every row the best answer beats inventing values and abstaining on all"))
    out.append(Check("answer schema", "fail" if bad_schema else "pass",
                     f"{len(bad_schema)} best answers do not validate against their own prompt schema, e.g. {bad_schema[0]}"
                     if bad_schema else "every best answer validates against the schema its prompt shows"))

    budget_chars = max_completion_tokens * CHARS_PER_TOKEN
    too_long = sum(1 for row in rows if len(oracle_answer(row)) > budget_chars)
    if too_long:
        out.append(Check("answer length", "fail" if too_long > len(rows) * 0.2 else "warn",
                         f"{too_long} of {len(rows)} best answers (~{longest / CHARS_PER_TOKEN:.0f} tokens at most) exceed "
                         f"--max-completion-tokens {max_completion_tokens}; a cut-off answer scores 0. Raise the budget "
                         "or drop the long examples (the estimate is {:.0f} characters per token)".format(CHARS_PER_TOKEN)))
    else:
        out.append(Check("answer length", "pass", f"longest best answer ~{longest / CHARS_PER_TOKEN:.0f} tokens, "
                                                  f"budget {max_completion_tokens}"))
    out.extend(_balance(rows))
    out.extend(_build_audit(data_dir))
    return out


BLIND_WARN = 0.3          # share of masked examples whose OCR could not read the unmasked fields
FLIPPED_WARN = 0.5        # share of masks that turned out to hide nothing (the value was still readable)


def _build_audit(data_dir: Path) -> list[Check]:
    """What the OCR guards did while the data was built (build_stats.json): many examples where OCR was blind, or many
    masks that hid nothing, mean the unreadable labels are only as good as an OCR that is not working here."""
    path = Path(data_dir) / "build_stats.json"
    if not path.exists():
        return [Check("build audit", "warn", "no build_stats.json next to the data (built by an older version): the "
                                             "OCR guards cannot be audited; rebuild to get it")]
    stats = json.loads(path.read_text())
    masked, examples_masked = stats.get("masked", 0), max(1, stats.get("examples", 0))
    if not masked:
        return [Check("build audit", "warn", "no field was masked at all: nothing teaches abstention (OCR found too few "
                                             "fields it could place uniquely, or --p-clean is too high)")]
    blind = stats.get("blind_examples", 0) / examples_masked
    flipped = stats.get("flipped_readable", 0) / masked
    detail = (f"{stats.get('localised', 0)} fields placed, {stats.get('ambiguous', 0)} skipped as ambiguous, "
              f"{masked} masked, {flipped:.0%} of masks still readable, {blind:.0%} of examples with blind OCR "
              f"({stats.get('downgraded_blind', 0)} labels downgraded to partial)")
    if blind > BLIND_WARN:
        return [Check("build audit", "warn", detail + ": OCR often cannot read the degraded page, so many unreadable "
                      "labels fell back to partial; use --level soft or a lower --p-degrade")]
    if flipped > FLIPPED_WARN:
        return [Check("build audit", "warn", detail + ": most masks hide nothing OCR can see; check the preview")]
    return [Check("build audit", "pass", detail)]


def _balance(rows: list[dict[str, Any]]) -> list[Check]:
    states = [s for r in rows for s in json.loads(r["states"]).values()]
    n = len(states) or 1
    unreadable, partial = states.count("must_abstain") / n, states.count("partial") / n
    out = []
    if unreadable > 0.5:
        out.append(Check("abstention balance", "warn", f"{unreadable:.0%} of fields are unreadable: abstaining looks "
                         "better than it is and training can collapse into null answers; mask fewer fields"))
    elif unreadable < 0.05:
        out.append(Check("abstention balance", "warn", f"only {unreadable:.0%} of fields are unreadable: there is "
                         "little to learn abstention from"))
    else:
        out.append(Check("abstention balance", "pass", f"{unreadable:.0%} unreadable, {partial:.0%} partial, "
                                                       f"{1 - unreadable - partial:.0%} readable fields"))
    clean = sum(1 for r in rows if "must_abstain" not in json.loads(r["states"]).values()) / len(rows)
    out.append(Check("clean examples", "pass" if clean >= 0.1 else "warn",
                     f"{clean:.0%} of examples have no unreadable field" + ("" if clean >= 0.1 else
                     "; with so few, answering everything is rarely right on any example (raise --p-clean)")))
    extended = sum(1 for r in rows if r["spec"] and json.loads(r["spec"]) is not None) / len(rows)  # "null" = plain
    out.append(Check("format mix", "pass" if 0.2 <= extended <= 0.8 else "warn",
                     f"{extended:.0%} of examples use the extended format, the rest the plain schema"))
    return out


# ---- batching and TRL ---------------------------------------------------------------------------------

def check_batching(batch_size: int, grad_accum: int, num_generations: int, world_size: int = 1) -> Check:
    """TRL's documented rule: one generation round (per-device batch x devices x accumulation steps) must hold whole
    groups of `num_generations` completions, and a group needs at least two to have an advantage."""
    if num_generations < 2:
        return Check("batching", "fail", "num_generations must be at least 2: GRPO compares completions of one prompt")
    total = batch_size * grad_accum * world_size
    if total % num_generations:
        return Check("batching", "fail", f"batch {batch_size} x accumulation {grad_accum} x {world_size} device(s) = "
                     f"{total} completions per round, not a multiple of num_generations {num_generations}")
    return Check("batching", "pass", f"{total} completions per round = {total // num_generations} prompts x "
                                     f"{num_generations} generations")


def check_trl(reward_function: Any = None, trl_module: Any = None) -> list[Check]:
    """The installed TRL has what `train/rl.py` passes to it. Version differences are the risk here."""
    if trl_module is None:
        try:
            import trl as trl_module
        except ImportError:
            return [Check("trl", "skip", "TRL is not installed (pip install '.[rl]')")]
    out = [Check("trl version", "pass", str(getattr(trl_module, "__version__", "unknown")))]
    config_fields = {f.name for f in dataclass_fields(trl_module.GRPOConfig)}
    missing = [f for f in GRPO_CONFIG_NEEDED if f not in config_fields]
    absent = [f for f in GRPO_CONFIG_USED if f not in config_fields]
    if missing:
        out.append(Check("GRPOConfig", "fail", f"this TRL has no {missing}"))
    elif absent:
        out.append(Check("GRPOConfig", "warn", f"settings this TRL does not have, they will be dropped: {absent}"))
    else:
        out.append(Check("GRPOConfig", "pass", "every setting train/rl.py passes exists"))
    params = inspect.signature(trl_module.GRPOTrainer.__init__).parameters
    lacking = [p for p in GRPO_TRAINER_ARGS if p not in params]
    out.append(Check("GRPOTrainer", "fail" if lacking else "pass",
                     f"its constructor has no {lacking}" if lacking else "the constructor takes what train/rl.py passes"))
    no_callbacks = [p for p in GRPO_TRAINER_OPTIONAL if p not in params]
    if no_callbacks:
        out.append(Check("GRPOTrainer callbacks", "warn", "this TRL takes no `callbacks`: the metrics reports and the hub "
                         "uploads of each save will not happen"))
    if reward_function is not None:
        sig = inspect.signature(reward_function).parameters.values()
        ok = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig) and "completions" in {p.name for p in sig}
        out.append(Check("reward function", "pass" if ok else "fail",
                         "takes completions and the dataset columns as keywords" if ok
                         else "must take `completions` and **kwargs so the dataset columns reach it"))
    return out


# ---- the processor against the SFT transform ----------------------------------------------------------

def check_processor(model_id: str, ocr_mode: str = "base") -> Check:
    """TRL builds the model inputs with the Hugging Face processor; SFT builds them with the repo's transform plus the
    batch adapter. If the two disagree, RL would train on inputs the SFT model never saw. Compares, for one page-sized
    image: the global view, the tiles, and the number of image tokens."""
    try:
        from transformers import AutoConfig, AutoProcessor

        from .processors import batch_adapter_for, create_transform
        from .processors.deepseekOcr import IMAGE_TOKEN_ID
        processor = AutoProcessor.from_pretrained(model_id)
        image_token_id = AutoConfig.from_pretrained(model_id).image_token_id
        transform = create_transform(model_id, mode=ocr_mode, fixed_length=False)
    except Exception as error:                                       # noqa: BLE001 - offline, no cache, wrong id
        return Check("processor parity", "skip", f"could not load {model_id}: {type(error).__name__}: {error}")
    image = Image.new("RGB", (1100, 1500), (255, 255, 255))
    batch = transform.make_collator()([transform([image], "Extract.", "x")])
    ours = batch_adapter_for(model_id)(batch)
    ours_tokens = int((batch["input_ids"] == IMAGE_TOKEN_ID).sum())
    text = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"},
                                                                       {"type": "text", "text": "Extract."}]}],
                                         add_generation_prompt=True, tokenize=False)
    theirs = processor(images=[image], text=[text], return_tensors="pt")
    their_tokens = int((theirs["input_ids"] == image_token_id).sum())
    problems = []
    if tuple(ours["pixel_values"].shape) != tuple(theirs["pixel_values"].shape):
        problems.append(f"global view {tuple(ours['pixel_values'].shape)} vs {tuple(theirs['pixel_values'].shape)}")
    local_ours, local_theirs = ours.get("pixel_values_local"), theirs.get("pixel_values_local")
    if (local_ours is None) != (local_theirs is None) or (
            local_ours is not None and tuple(local_ours.shape) != tuple(local_theirs.shape)):
        problems.append(f"tiles {None if local_ours is None else tuple(local_ours.shape)} vs "
                        f"{None if local_theirs is None else tuple(local_theirs.shape)}")
    if ours_tokens != their_tokens:
        problems.append(f"image tokens {ours_tokens} vs {their_tokens}")
    if problems:
        return Check("processor parity", "fail", f"mode {ocr_mode!r}: " + "; ".join(problems) +
                     ". RL would see different image inputs than SFT did; set the SFT mode or processor options to match")
    return Check("processor parity", "pass", f"mode {ocr_mode!r}: same global view, tiles and {ours_tokens} image tokens "
                                             "as the Hugging Face processor")


# ---- the model (needs weights; run with --with-model) -------------------------------------------------

def check_model(model_id: str, load_wrapper: Callable[[], Any], row: dict[str, Any], data_dir: Path) -> list[Check]:
    """Builds the model the way `train` does (`load_wrapper` is train/rl.py's, merge + new LoRA included) and generates a
    few tokens for one row."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        wrapper = load_wrapper()
        wrapper.model.to(device)
    except Exception as error:                                       # noqa: BLE001
        return [Check("model load", "fail" if isinstance(error, (RuntimeError, KeyError, FileNotFoundError, ValueError))
                      else "skip", f"{type(error).__name__}: {error}")]
    merged = len(getattr(wrapper, "lineage", []) or [])
    out = [Check("model load", "pass", f"loaded on {device}" + (f", {merged} earlier adapter(s) merged into the base, then a "
                                                                f"new LoRA of rank {wrapper.lora_config_dict()['r']}"
                                                                if merged else ", no earlier adapter merged"))]
    trainable = [n for n, p in wrapper.model.named_parameters() if p.requires_grad]
    odd = [n for n in trainable if "lora_" not in n]
    out.append(Check("trainable weights", "fail" if (not trainable or odd) else "pass",
                     "nothing is trainable" if not trainable else f"non-adapter weights are trainable: {odd[:2]}" if odd
                     else f"{len(trainable)} adapter tensors train, the base is frozen"))
    out.append(Check("peft wrapper", "pass" if isinstance(wrapper.model, PeftModel) else "fail",
                     "TRL receives a PeftModel and trains its adapters" if isinstance(wrapper.model, PeftModel)
                     else "the model handed to TRL is not a PeftModel"))
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        text = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"},
                                                                           {"type": "text", "text": row["prompt"]}]}],
                                             add_generation_prompt=True, tokenize=False)
        inputs = processor(images=[Image.open(data_dir / row["image"]).convert("RGB")], text=[text],
                           return_tensors="pt").to(device)
        with torch.no_grad():
            generated = wrapper.model.generate(**inputs, max_new_tokens=16, do_sample=False)
        decoded = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        reward = reward_rl(decoded, row["reference"], json.loads(row["spec"]) if row["spec"] else None,
                           json.loads(row["states"]), row["upright"], json.loads(row["tags"]),
                           json.loads(row["prompt_schema"]) if row["prompt_schema"] else None)
        out.append(Check("generation", "pass", f"generated 16 tokens from the processor's inputs and scored them "
                                               f"({reward:.2f}); the chain model -> text -> reward runs"))
    except Exception as error:                                       # noqa: BLE001
        out.append(Check("generation", "fail", f"{type(error).__name__}: {error}"))
    return out


def preflight(rows: list[dict[str, Any]], data_dir: Path, max_completion_tokens: int, batch_size: int, grad_accum: int,
              num_generations: int, reward_function: Any, world_size: int = 1, model_id: str | None = None,
              ocr_mode: str | None = None) -> list[Check]:
    """The checks that are cheap enough to run before every training run."""
    checks = check_data(rows, data_dir, max_completion_tokens)
    checks.append(check_batching(batch_size, grad_accum, num_generations, world_size))
    checks.extend(check_trl(reward_function))
    if model_id is not None:
        checks.append(check_processor(model_id, ocr_mode or "base"))
    return checks


__all__ = ["Check", "report", "failed", "check_data", "check_batching", "check_trl", "check_processor", "check_model",
           "preflight", "oracle_answer", "guess_answer", "blank_answer"]
