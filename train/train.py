"""Command line for SFT: single device, DDP or FSDP2 (LoRA / QLoRA optional).

    python -m train.train --root data --model-id deepseek-ai/DeepSeek-OCR-2 --strategy single --lora
    torchrun --nproc_per_node=4 -m train.train --root data --model-id deepseek-ai/DeepSeek-OCR-2 --strategy fsdp2

Note: the collator's keys are not yet mapped to the Hugging Face forward signature. Until a batch
adapter is supplied for the target model, pass one through SFTConfig.batch_adapter (see train/sft.py).
"""
from __future__ import annotations

import sys

import argparse
import logging

import torch
from transformers import AutoProcessor

from corpus.logs import add_arguments, configure

from . import distributed as dd
from .dataset import make_loader
from .model import DeepSeekOCR
from .optim import build_optimizer
from .processors import DatasetProcessor, create_transform, model_family
log = logging.getLogger("train.sft")
from .sft import SFTConfig, train_ddp, train_fsdp2, train_single


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervised fine-tuning for DeepSeek-OCR")
    p.add_argument("--root", required=True, help="dataset root (see train/dataset.py)")
    p.add_argument("--model-id", required=True)
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
    p.add_argument("--pad-to-max", action="store_true",
                   help="pad every sample to --max-length (default: pad each batch to its longest sample)")
    p.add_argument("--batch-size", type=int, default=2, help="per rank")
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
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--train-vision", action="store_true", help="also train the vision tower and projector")
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--quant", choices=["4bit", "8bit"], default=None, help="QLoRA base weights (ddp or single only)")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--out-dir", default="checkpoints/sft")
    p.add_argument("--resume", default=None)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--no-tensorboard", action="store_true")
    add_arguments(p)
    return p.parse_args(argv)


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
    device = dd.init_distributed()          # first: the loader shards by the process group
    try:
        dtype, _ = dd.resolve_precision(args.precision, device)
        if args.quant is not None and args.strategy == "fsdp2":
            raise SystemExit("--quant is not compatible with --strategy fsdp2; use ddp or single")

        log.info("loading tokenizer and transform (%s, mode %s)", args.model_id, args.ocr_mode)
        transform = build_transform(args)
        log.info("building the training loader from %s", args.root)
        loader = make_loader(
            args.root, split="train", processor=DatasetProcessor(preset=args.preset, descriptions=args.descriptions),
            transform=transform, batch_size=args.batch_size, num_workers=args.workers,
        )

        log.info("loading the model %s (the first run downloads its weights)", args.model_id)
        # Master weights stay fp32 (mixed precision casts them); quantized weights use the compute dtype.
        load_dtype = dtype if args.quant is not None else torch.float32
        extra = {"device_map": {"": str(device)}} if args.quant is not None else {}
        model = DeepSeekOCR.from_pretrained(
            args.model_id, lang_trained=True, vision_trained=args.train_vision,
            dtype=load_dtype, quantization=args.quant, **extra,
        )
        if args.lora:
            log.info("adding LoRA adapters (r=%d)", args.lora_r)
            model.add_lora(r=args.lora_r)
        log.info("starting training: strategy %s, %d epoch(s)", args.strategy, args.epochs)

        def make_optimizer(named_params):
            if args.optimizer == "adamw":
                return torch.optim.AdamW([p for _, p in named_params], lr=args.lr)
            return build_optimizer(named_params, muon_lr=args.muon_lr, adamw_lr=args.lr,
                                   adjust_lr_fn=args.muon_adjust)

        cfg = SFTConfig(
            epochs=args.epochs, grad_accum=args.grad_accum, max_grad_norm=args.max_grad_norm,
            precision=args.precision, log_every=args.log_every, save_every_steps=args.save_every,
            out_dir=args.out_dir, resume=args.resume, tensorboard=not args.no_tensorboard,
            compile=args.compile,
        )
        runner = {"single": train_single, "ddp": train_ddp, "fsdp2": train_fsdp2}[args.strategy]
        result = runner(model, loader, make_optimizer, None, cfg)
        if dd.is_main():
            print(result)
        return result
    finally:
        dd.cleanup()


if __name__ == "__main__":
    main()
