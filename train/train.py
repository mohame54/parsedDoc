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
from .model import DeepSeekOCR
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
    p.add_argument("--preset", default="structured", help="DatasetProcessor preset")
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
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (memory probes)")
    p.add_argument("--hub-repo", default=None, help="push checkpoints to this Hugging Face repo (user/name)")
    p.add_argument("--hub-token", default=None, help="Hugging Face token (default: HF_TOKEN or the cached login)")
    p.add_argument("--hub-every-save", action="store_true",
                   help="push at every periodic save (default: only the final checkpoint)")
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
    p.add_argument("--samples-per-document", type=int, default=1,
                   help="examples drawn from each document per epoch (a different task / format each time)")
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
    p.add_argument("--resume", default=None, help="a checkpoint .pt, or hf://user/repo/last/training_state.pt")
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
    return args


def _targets(value):
    """A regex string stays one; "q_proj,v_proj" or a YAML list becomes a list of module names."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [part.strip() for part in value.split(",")] if "," in value and "(" not in value else value


def save_resolved(args: argparse.Namespace) -> None:
    """The settings this run really used (file + flags), next to the checkpoints, for reproducing it."""
    import yaml

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    resolved = {k: v for k, v in sorted(vars(args).items()) if k != "hub_token"}
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")


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
            samples_per_document=args.samples_per_document,
        )
        if args.batch_tokens:
            log.info("length-bucketed batches: up to %d padded tokens and %d samples per micro-batch, "
                     "samples longer than %s tokens are left out", args.batch_tokens, args.max_batch_size,
                     transform.max_length)

        log.info("loading the model %s (the first run downloads its weights)", args.model_id)
        # Master weights stay fp32 (mixed precision casts them); quantized weights use the compute dtype.
        load_dtype = dtype if args.quant is not None else torch.float32
        extra = {"device_map": {"": str(device)}} if args.quant is not None else {}
        model = DeepSeekOCR.from_pretrained(
            args.model_id, lang_trained=True, vision_trained=args.train_vision,
            dtype=load_dtype, quantization=args.quant, **extra,
        )
        if args.lora:
            log.info("adding LoRA adapters (r=%d, alpha=%d, dropout=%g)", args.lora_r, args.lora_alpha,
                     args.lora_dropout)
            kwargs = {} if args.lora_targets is None else {"target_modules": _targets(args.lora_targets)}
            model.add_lora(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, **kwargs)
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
        hub = None
        if args.hub_repo and dd.is_main():
            hub = HubPusher(args.hub_repo, token=args.hub_token, private=args.hub_private,
                            every_save=args.hub_every_save, run_config=vars(args),
                            stage_dir=Path(args.out_dir) / "hub_stage")
        cfg = SFTConfig(
            epochs=args.epochs, grad_accum=args.grad_accum, max_grad_norm=args.max_grad_norm,
            precision=args.precision, log_every=args.log_every, save_every_steps=args.save_every,
            out_dir=args.out_dir, resume=resolve_resume(args.resume, args.hub_token), tensorboard=args.tensorboard,
            compile=args.compile, loss_weighting=loss_weighting, max_steps=args.max_steps, hub=hub,
            batch_adapter=batch_adapter_for(args.model_id),
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
