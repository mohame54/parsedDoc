"""Command line for SFT: single device, DDP or FSDP2 (LoRA / QLoRA optional).

    python -m train.train --root data --model-id deepseek-community/DeepSeek-OCR-2 --strategy single --lora
    torchrun --nproc_per_node=4 -m train.train --root data --model-id deepseek-community/DeepSeek-OCR-2 --strategy fsdp2

Note: the collator's keys are not yet mapped to the Hugging Face forward signature. Until a batch
adapter is supplied for the target model, pass one through SFTConfig.batch_adapter (see train/sft.py).
"""
from __future__ import annotations

import sys

import argparse
import math
from pathlib import Path
import logging

import torch
from transformers import AutoProcessor

from corpus.logs import add_arguments, configure

from . import distributed as dd
from .config import check_choices, load_config
from .hub import HubPusher, resolve_resume
from .dataset import make_loader
from .model import DeepSeekOCR, parse_targets
from .optim import build_optimizer, build_scheduler
from .processors import DatasetProcessor, batch_adapter_for, create_transform, model_family
log = logging.getLogger("train.sft")
from .sft import SFTConfig, train_ddp, train_fsdp2, train_single


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Supervised fine-tuning for DeepSeek-OCR. Settings come from the built-in defaults, then "
                    "--config (a YAML file, see train/config.py), then these flags.")
    p.add_argument("--config", default=None,
                   help="YAML file: a path, or a name in the config folder (config/train.yaml, train_qlora, ...)")
    p.add_argument("--root", default=None, help="dataset root (see train/dataset.py)")
    p.add_argument("--model-id", default=None)
    p.add_argument("--strategy", choices=["single", "ddp", "fsdp2"], default="single")
    p.add_argument("--preset", default=None,
                   help="task mix (train/processors/mixture.py). Default: `visual` with --train-vision, else `structured`")
    p.add_argument("--descriptions", default=None,
                   help="attribute descriptions JSON from `labeller describe-collect` (templates are used without it)")
    p.add_argument("--max-length", type=int, default=None,
                   help="drop examples longer than this many tokens (images included); see `python -m train.stats`")
    p.add_argument("--merge-pages", choices=["vertical", "horizontal", "grid"], default=None,
                   help="how a multi-page document becomes one image (DeepSeek-OCR defaults to grid)")
    p.add_argument("--ocr-mode", choices=["tiny", "small", "base", "large", "gundam"], default="base",
                   help="DeepSeek-OCR resolution mode (v2 has only small, base, gundam)")
    p.add_argument("--pad-to-max", action=argparse.BooleanOptionalAction, default=False,
                   help="pad every sample to --max-length (default: pad each batch to its longest sample)")
    p.add_argument("--batch-size", type=int, default=2, help="per rank; ignored with --batch-tokens")
    p.add_argument("--batch-tokens", type=int, default=None,
                   help="length-bucketed batches: each holds as many similar-length examples as fit this many "
                        "padded tokens per rank (see the memory probe in the README). --max-length is the longest "
                        "sample allowed; it must fit one batch")
    p.add_argument("--max-batch-size", type=int, default=16, help="largest batch with --batch-tokens")
    p.add_argument("--pad-multiple", type=int, default=64, help="batches are padded to a multiple of this")
    p.add_argument("--loss-weighting", choices=["token", "batch"], default=None,
                   help="token: every target token counts equally over the accumulation window; batch: mean of the "
                        "micro-batch losses. Default: token with --batch-tokens, else batch")
    p.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=False,
                   help="recompute activations in backward: 3-5x less activation memory, about 30%% more compute. "
                        "Needed for long samples (--max-length 8192) with --train-vision on one 80 GB card")
    p.add_argument("--chunked-loss", action=argparse.BooleanOptionalAction, default=True,
                   help="compute the loss over the supervised positions only, in chunks, without the full logits "
                        "(a lot less memory on long batches; the same loss)")
    p.add_argument("--eval", action=argparse.BooleanOptionalAction, default=True,
                   help="validate on val.jsonl: before training, every --eval-every steps and at the end of each "
                        "epoch (the val/loss curve in TensorBoard)")
    p.add_argument("--eval-every", type=int, default=0,
                   help="also validate every N optimizer steps (0 = only at the end of each epoch)")
    p.add_argument("--val-max-documents", type=int, default=None,
                   help="validate on the first N documents of val.jsonl only (default: all)")
    p.add_argument("--gen-eval", action=argparse.BooleanOptionalAction, default=False,
                   help="generation check: generate answers for validation examples (page + prompt only) and score "
                        "them, before the first step and at the end. Slower than the validation loss")
    p.add_argument("--gen-eval-every", type=int, default=0, help="also run the generation check every N steps")
    p.add_argument("--gen-samples-per-task", type=int, default=2, help="validation examples per task to generate")
    p.add_argument("--gen-max-new-tokens", type=int, default=2048,
                   help="most tokens to generate; examples whose reference is longer are left out")
    p.add_argument("--gen-batch-size", type=int, default=4, help="examples generated at the same time")
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (memory probes)")
    p.add_argument("--hub-repo", default=None, help="push checkpoints to this Hugging Face repo (user/name)")
    p.add_argument("--hub-token", default=None, help="Hugging Face token (default: HF_TOKEN or the cached login)")
    p.add_argument("--hub-every-save", action=argparse.BooleanOptionalAction, default=True,
                   help="push every periodic save, each into its own folder (--no-hub-every-save: only the final one)")
    p.add_argument("--hub-keep-last", action="store_true",
                   help="also keep a `last/` copy of the newest save in the repo (every save has its own folder anyway)")
    p.add_argument("--hub-state", action=argparse.BooleanOptionalAction, default=True,
                   help="put training_state.pt (optimizer etc.) in every hub folder; --no-hub-state makes the folders "
                        "much smaller but they can no longer be resumed from")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon",
                   help="muon: Muon for 2D hidden matrices, AdamW for the rest; adamw: AdamW everywhere")
    p.add_argument("--lr", type=float, default=2e-5, help="AdamW learning rate")
    p.add_argument("--muon-lr", type=float, default=0.02, help="Muon learning rate (muon only)")
    p.add_argument("--muon-adjust", choices=["original", "match_rms_adamw", "spectral_unclamped"],
                   default="original",
                   help="torch Muon update scaling; match_rms_adamw lets AdamW-tuned lrs carry over")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--scheduler", choices=["cosine", "linear", "constant"], default="cosine",
                   help="learning-rate curve after the warmup")
    p.add_argument("--warmup-ratio", type=float, default=0.05, help="share of the steps spent warming up linearly")
    p.add_argument("--min-lr-ratio", type=float, default=0.1, help="the curve ends at this fraction of the base lr")
    p.add_argument("--exclude-languages", nargs="*", default=None, metavar="CODE",
                   help="leave documents of these languages out of training and validation (ISO 639-1, e.g. de)")
    p.add_argument("--max-documents", type=int, default=None,
                   help="use only the first N documents of the split (a quick overfit test)")
    p.add_argument("--samples-per-document", type=int, default=1,
                   help="examples drawn from each document per epoch (a different task / format each time)")
    p.add_argument("--augment", choices=["off", "soft", "strong"], default="soft",
                   help="how hard a degraded TRAINING page is hit (train/degrade.py): soft = small skew, noise, blur, "
                        "JPEG; strong = quarter and free turns too. Validation is never degraded.")
    p.add_argument("--augment-p", type=float, default=0.0,
                   help="share of training examples that get degraded, 0-1 (0.2 = a fifth of them, the rest stay "
                        "clean). 0 turns it off.")
    p.add_argument("--extended-p", type=float, default=0.0,
                   help="share of schema extractions written in the extended format (a confidence per field, "
                        "optionally a reason and the page rotation, varied key names); the rest keep the plain schema")
    p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--train-vision", action=argparse.BooleanOptionalAction, default=False,
                   help="also train the vision tower and projector")
    p.add_argument("--lora", action=argparse.BooleanOptionalAction, default=False, help="train LoRA adapters")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", default=None,
                   help="regex or list of module names to adapt (default: the language model's attention and MLP "
                        "projections, see train/model.py)")
    p.add_argument("--quant", choices=["4bit", "8bit"], default=None, help="QLoRA base weights (ddp or single only)")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                   help="torch.compile the language model (off by default: varying batch shapes make it recompile)")
    p.add_argument("--out-dir", default="checkpoints/sft")
    p.add_argument("--resume", default=None,
                   help="carry on THIS run: a checkpoint .pt, or hf://user/repo/step_0000200/training_state.pt (same "
                        "LoRA configuration, optimizer and step restored)")
    p.add_argument("--init-from", default=None,
                   help="start a NEW run from an earlier adapter with a new LoRA configuration: the old adapter (and the "
                        "ones it was built on) is loaded and merged into the base, then a fresh LoRA "
                        "(--lora-r/--lora-alpha/--lora-targets) is attached. Optimizer and step start from zero. Needs "
                        "--lora and an unquantized base. A .pt, or hf://user/repo/sft_step_0000200/adapter_state.pt")
    p.add_argument("--init-lora-r", type=int, default=None,
                   help="the rank of the --init-from adapter, only for checkpoints that predate recording it")
    p.add_argument("--init-lora-alpha", type=int, default=None, help="its alpha (same: old checkpoints only)")
    p.add_argument("--init-lora-targets", default=None, help="its target modules (same: old checkpoints only)")
    p.add_argument("--state-prefix", default=None,
                   help="put before every checkpoint, report and hub folder name, to tell runs apart (sft_step_0000200, "
                        "rl_step_...). Default: the kind of training (sft); give '' for none")
    p.add_argument("--reports", action=argparse.BooleanOptionalAction, default=True,
                   help="write a metrics report with graphs next to every checkpoint (train/ckpt_report.py)")
    p.add_argument("--gen-eval-at-save", action=argparse.BooleanOptionalAction, default=True,
                   help="validate and generate at every checkpoint, so each report has numbers from its own step")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hub-private", action=argparse.BooleanOptionalAction, default=False)
    add_arguments(p)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    config_path = pre.parse_known_args(argv)[0].config
    if config_path:
        p.set_defaults(**load_config(config_path))     # a file only moves the defaults; flags still win
    args = p.parse_args(argv)
    check_choices(p, args)
    for name, flag in (("root", "--root"), ("model_id", "--model-id")):
        if getattr(args, name) is None:
            p.error(f"{flag} is required (on the command line or in the config file)")
    if isinstance(args.lora_targets, str) and args.lora_targets.lstrip().startswith("["):
        p.error("--lora-targets takes a regex or comma-separated names; give a list in the YAML file")
    if args.init_from:
        if args.resume:
            p.error("--init-from starts a new run and --resume carries on an old one; give one of them")
        if not args.lora:
            p.error("--init-from attaches a new LoRA on top of the merged old adapter, so it needs --lora")
        if args.quant:
            p.error("--init-from merges the old adapter into the base, which a quantized (QLoRA) base cannot take")
    if args.preset is None:
        args.preset = "visual" if args.train_vision else "structured"
        log.info("task mix: the `%s` preset (%s)", args.preset,
                 "vision is trained, so visual reasoning tasks are upweighted" if args.train_vision else "default")
    return args


def _targets(value):
    """A regex string stays one; "q_proj,v_proj" or a YAML list becomes a list of module names."""
    return parse_targets(value)


def save_resolved(args: argparse.Namespace) -> None:
    """The settings this run really used (file + flags), next to the checkpoints, for reproducing it."""
    import yaml

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    resolved = {k: v for k, v in sorted(vars(args).items()) if k != "hub_token"}
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")


def weights_dtype(args: argparse.Namespace, compute_dtype: torch.dtype) -> torch.dtype:
    """The dtype the pretrained weights are loaded in.

    fp32 master weights are needed only where the base weights themselves train (full fine-tuning, or
    --train-vision). With LoRA or QLoRA the base is frozen, so it loads in the compute dtype: for this 3.4B-parameter
    model that is 6.8 GB instead of 13.6 GB, the difference between fitting a 16 GB card and not. PEFT keeps the
    adapter weights themselves in fp32, so the optimizer and the fp16 loss scaler still see fp32 parameters."""
    if args.quant is not None or (args.lora and not args.train_vision):
        return compute_dtype
    return torch.float32


def build_transform(args: argparse.Namespace):
    """The same transform settings `python -m train.stats` measures with, so its numbers apply here."""
    model_id = args.model_id
    if model_family(model_id) == "instruct":
        return create_transform(model_id, processor=AutoProcessor.from_pretrained(model_id, trust_remote_code=True),
                                max_length=args.max_length, fixed_length=args.pad_to_max,
                                merge_pages=args.merge_pages)
    kwargs = {"mode": args.ocr_mode, "max_length": args.max_length, "fixed_length": args.pad_to_max}
    if args.merge_pages is not None:       # otherwise DeepSeek-OCR's own default ("grid")
        kwargs["merge_pages"] = args.merge_pages
    return create_transform(model_id, **kwargs)     # loads its own tokenizer, without running remote code


def main(argv: list[str] | None = None) -> dict[str, int] | None:
    args = parse_args(argv)
    configure(args)
    if args.config:
        log.info("settings from %s, then the command line", args.config)
    if dd.is_main():
        save_resolved(args)
    device = dd.init_distributed()          # first: the loader shards by the process group
    try:
        dtype, _ = dd.resolve_precision(args.precision, device)
        if args.quant is not None and args.strategy == "fsdp2":
            raise SystemExit("--quant is not compatible with --strategy fsdp2; use ddp or single")

        log.info("loading tokenizer and transform (%s, mode %s)", args.model_id, args.ocr_mode)
        transform = build_transform(args)
        log.info("building the training loader from %s", args.root)
        if args.batch_tokens and args.pad_to_max:
            raise SystemExit("--pad-to-max pads every sample to --max-length, which defeats --batch-tokens; use one")
        loader = make_loader(
            args.root, split="train", processor=DatasetProcessor(preset=args.preset, descriptions=args.descriptions),
            transform=transform, batch_size=args.batch_size, num_workers=args.workers,
            batch_tokens=args.batch_tokens, max_batch_size=args.max_batch_size, pad_multiple=args.pad_multiple,
            samples_per_document=args.samples_per_document, max_documents=args.max_documents, exclude_languages=args.exclude_languages,
            augment=args.augment, augment_p=args.augment_p, extended_p=args.extended_p,
        )
        if args.augment != "off" and args.augment_p > 0:
            log.info("synthetic degradation (%s) on %.0f%% of the training examples; validation stays clean",
                     args.augment, 100 * args.augment_p)
        if args.batch_tokens:
            log.info("length-bucketed batches: up to %d padded tokens and %d samples per micro-batch, "
                     "samples longer than %s tokens are left out", args.batch_tokens, args.max_batch_size,
                     transform.max_length)

        val_loader = None
        if args.eval and (Path(args.root) / "val.jsonl").exists():
            val_loader = make_loader(
                args.root, split="val", processor=DatasetProcessor(preset=args.preset, descriptions=args.descriptions),
                transform=transform, batch_size=args.batch_size, num_workers=args.workers,
                batch_tokens=args.batch_tokens, max_batch_size=args.max_batch_size, pad_multiple=args.pad_multiple,
                max_documents=args.val_max_documents, exclude_languages=args.exclude_languages,
            )
            log.info("validation: %d batches from %s/val.jsonl, loss logged as val/loss", len(val_loader), args.root)
        elif args.eval:
            log.warning("no val.jsonl in %s: training without a validation loss", args.root)

        gen_eval = None
        if args.gen_eval and (Path(args.root) / "val.jsonl").exists():
            from .gen_eval import GenerationEvaluator
            gen_eval = GenerationEvaluator(
                args.root, transform, batch_adapter_for(args.model_id), samples_per_task=args.gen_samples_per_task,
                preset=args.preset,
                max_new_tokens=args.gen_max_new_tokens, batch_size=args.gen_batch_size,
                max_documents=args.val_max_documents, precision=args.precision, out_dir=Path(args.out_dir) / "generations",
                exclude_languages=args.exclude_languages)

        log.info("loading the model %s (the first run downloads its weights)", args.model_id)
        load_dtype = weights_dtype(args, dtype)
        log.info("weights are loaded as %s (compute dtype %s)", str(load_dtype).removeprefix("torch."),
                 str(dtype).removeprefix("torch."))
        # single and ddp load every tensor straight onto this process's GPU (no staging copy in CPU RAM); fsdp2 loads on
        # the CPU and shards afterwards
        extra = {"device_map": {"": str(device)}} if args.quant is not None or args.strategy != "fsdp2" else {}
        model = DeepSeekOCR.from_pretrained(
            args.model_id, lang_trained=True, vision_trained=args.train_vision,
            dtype=load_dtype, quantization=args.quant, **extra,
        )
        model.chunked_loss = args.chunked_loss
        if args.lora:
            kwargs = {} if args.lora_targets is None else {"target_modules": _targets(args.lora_targets)}
            if args.init_from:
                log.info("continuing from %s: merging its adapter chain into the base, then a new LoRA "
                         "(r=%d, alpha=%d, dropout=%g)", args.init_from, args.lora_r, args.lora_alpha, args.lora_dropout)
                meta = model.continue_from(
                    args.init_from, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
                    old_r=args.init_lora_r, old_alpha=args.init_lora_alpha,
                    old_target_modules=None if args.init_lora_targets is None else _targets(args.init_lora_targets),
                    token=args.hub_token, **kwargs)
                log.info("merged %d earlier adapter(s) (the last from step %s)", len(model.lineage), meta.get("step"))
            else:
                log.info("adding LoRA adapters (r=%d, alpha=%d, dropout=%g)", args.lora_r, args.lora_alpha,
                         args.lora_dropout)
                model.add_lora(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, **kwargs)
        if args.grad_checkpointing:
            n_ckpt = model.enable_grad_checkpointing()
            if not n_ckpt:
                raise SystemExit("--grad-checkpointing: this model has no layer that supports it")
            log.info("gradient checkpointing on (%d module(s)): less activation memory, about a third more compute",
                     n_ckpt)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log.info("trainable parameters: %s of %s (%.2f%%)%s", f"{trainable:,}", f"{total:,}", 100 * trainable / total,
                 " - LoRA adapters only" if args.lora else "")
        log.info("starting training: strategy %s, %d epoch(s)", args.strategy, args.epochs)

        def make_optimizer(named_params):
            if args.optimizer == "adamw":
                return torch.optim.AdamW([p for _, p in named_params], lr=args.lr, weight_decay=args.weight_decay)
            return build_optimizer(named_params, muon_lr=args.muon_lr, adamw_lr=args.lr,
                                   weight_decay=args.weight_decay, adjust_lr_fn=args.muon_adjust)

        steps = math.ceil(len(loader) / args.grad_accum) * args.epochs      # an estimate when batches are bucketed
        if args.max_steps:
            steps = min(steps, args.max_steps)

        def make_scheduler(opt):
            return build_scheduler(opt, steps, args.scheduler, args.warmup_ratio, args.min_lr_ratio)

        loss_weighting = args.loss_weighting or ("token" if args.batch_tokens else "batch")
        state_prefix = "sft" if args.state_prefix is None else args.state_prefix
        hub = None
        if args.hub_repo and dd.is_main():
            hub = HubPusher(args.hub_repo, token=args.hub_token, private=args.hub_private,
                            every_save=args.hub_every_save, run_config=vars(args),
                            stage_dir=Path(args.out_dir) / "hub_stage", prefix=state_prefix,
                            keep_last=args.hub_keep_last, state=args.hub_state)
        cfg = SFTConfig(
            epochs=args.epochs, grad_accum=args.grad_accum, max_grad_norm=args.max_grad_norm,
            precision=args.precision, log_every=args.log_every, save_every_steps=args.save_every,
            out_dir=args.out_dir, resume=resolve_resume(args.resume, args.hub_token), tensorboard=args.tensorboard,
            compile=args.compile, loss_weighting=loss_weighting, max_steps=args.max_steps, hub=hub,
            kind="sft", state_prefix=state_prefix, reports=args.reports, gen_eval_at_save=args.gen_eval_at_save,
            run_config={k: v for k, v in vars(args).items() if k != "hub_token"},
            batch_adapter=batch_adapter_for(args.model_id), val_loader=val_loader, eval_every=args.eval_every,
            gen_eval=gen_eval, gen_eval_every=args.gen_eval_every,
        )
        runner = {"single": train_single, "ddp": train_ddp, "fsdp2": train_fsdp2}[args.strategy]
        result = runner(model, loader, make_optimizer, make_scheduler, cfg)
        if dd.is_main():
            print(result)
        return result
    finally:
        dd.cleanup()


if __name__ == "__main__":
    main()
