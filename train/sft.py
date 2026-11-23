"""Supervised fine-tuning loop for DeepSeekOCR (single device, DDP or FSDP2).

Typical use (under torchrun for ddp / fsdp2):

    device = distributed.init_distributed()          # before building the loader (sharding reads it)
    loader = make_loader(root, processor=..., transform=..., batch_size=4)
    model = DeepSeekOCR.from_pretrained(model_id, quantization=None, dtype=torch.bfloat16)
    model.add_lora()                                 # optional; before the optimizer exists
    train_fsdp2(model, loader,
                make_optimizer=lambda named: build_optimizer(named, adamw_lr=2e-5),  # Muon + AdamW by default
                make_scheduler=lambda opt: get_cosine_schedule_with_warmup(opt, 100, total),
                cfg=SFTConfig(epochs=2, grad_accum=8, out_dir="runs/sft1"))

The optimizer is built *after* wrapping, because FSDP2 replaces parameters with shards in place.
"""
from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader
from corpus.progress import Progress

from . import distributed as dd
from .dataset import set_epoch
from .metrics import MetricLogger
from .processors.base import to_device

log = logging.getLogger("train.sft")


@dataclass
class SFTConfig:
    epochs: int = 1
    grad_accum: int = 1
    max_grad_norm: float | None = 1.0          # None disables clipping (the norm is still logged)
    precision: str = "auto"                    # auto | bf16 | fp16 | fp32
    strategy: str = "single"                   # single | ddp | fsdp2 (set by the entry point)
    log_every: int = 10                        # optimizer steps between log lines / TensorBoard points
    save_every_steps: int = 200                # 0 disables periodic checkpoints (a final one is still written)
    out_dir: str = "checkpoints/sft"
    resume: str | None = None                  # a checkpoint written by this loop
    tensorboard: bool = True
    compile: bool = False                      # torch.compile the language model before wrapping
    batch_adapter: Callable[[dict], dict] | None = None   # maps collator keys to the model's forward kwargs
    val_loader: DataLoader | None = None
    loss_weighting: str = "batch"              # batch: mean of the micro-batch losses | token: every target token equal
    max_steps: int = 0                         # stop after this many optimizer steps (0 = run all epochs); for memory probes
    hub: Any = None                            # train.hub.HubPusher on rank 0 (checkpoints go to a Hugging Face repo)


def train_sft(
    epochs: int,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    loader: DataLoader,
    scheduler=None,
    save_every_steps: int = 200,
    cfg: SFTConfig | None = None,
) -> dict[str, int]:
    """Run the loop. ``model`` may be the bare DeepSeekOCR wrapper or a DDP / FSDP2 wrapped one.

    ``epochs`` and ``save_every_steps`` override the matching fields of ``cfg``.
    Returns {"step": optimizer steps taken, "skipped": steps dropped for non-finite gradients}.
    """
    cfg = replace(cfg or SFTConfig(), epochs=epochs, save_every_steps=save_every_steps)
    device = dd.model_device(model)
    amp_dtype, use_scaler = dd.resolve_precision(cfg.precision, device)
    scaler = _make_scaler(use_scaler, device, dd.is_fsdp(model))
    params = [p for p in model.parameters() if p.requires_grad]
    out_dir = Path(cfg.out_dir)

    global_step, skipped = 0, 0
    start_epoch, start_batch = 0, 0
    if cfg.resume:
        start_epoch, start_batch, global_step = _resume(cfg.resume, model, opt, scheduler)

    steps_per_epoch = math.ceil(len(loader) / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
    if cfg.loss_weighting not in ("batch", "token"):
        raise ValueError("loss_weighting must be 'batch' or 'token'")
    log.info("loss weighting: %s", "per target token over each accumulation window" if cfg.loss_weighting == "token"
             else "mean of the micro-batch losses")
    logger = MetricLogger(out_dir / "tb", is_main=dd.is_main(), log_every=cfg.log_every,
                          total_steps=total_steps, tensorboard=cfg.tensorboard,
                          initial_step=global_step)

    # Running sums over the current logging window; reduced across ranks only on log steps.
    loss_sum = torch.zeros((), device=device)          # sum of loss x target tokens (or x 1 per micro-batch)
    loss_count = torch.zeros((), device=device)
    token_sum = torch.zeros((), device=device)
    window_samples = window_padded = window_batches = 0

    model.train()
    stop = False
    try:
        for epoch in range(start_epoch, cfg.epochs):
            if stop:
                break
            set_epoch(loader, epoch)
            _log_plan(loader, epoch)
            first = start_batch if epoch == start_epoch else 0
            n_batches = len(loader)
            batches = itertools.islice(loader, first, None)
            consumed = first
            while True:
                # One accumulation window at a time: with token weighting the whole window's target-token count
                # is needed before the first backward, and the window boundary is where checkpoints fall.
                window = list(itertools.islice(batches, cfg.grad_accum))
                if not window:
                    break
                window = [_prepare(batch, cfg.batch_adapter, device) for batch in window]
                consumed += len(window)
                last = consumed == n_batches
                weights = _window_weights(window, cfg.loss_weighting, device)

                for position, batch in enumerate(window):
                    with dd.grad_sync(model, sync=position == len(window) - 1):
                        with dd.autocast(device, amp_dtype):
                            out = model(**batch)
                        loss = out.loss
                        scaled = loss * weights[position]
                        (scaler.scale(scaled) if scaler is not None else scaled).backward()

                    targets = weights.targets[position]
                    loss_sum += loss.detach().float() * targets
                    loss_count += targets
                    if "attention_mask" in batch:
                        token_sum += batch["attention_mask"].sum()
                    ids = batch.get("input_ids")
                    if ids is not None:
                        window_samples += ids.shape[0]
                        window_padded += ids.numel()
                        window_batches += 1

                # ---- optimizer step on the accumulation boundary ----
                scale_before = scaler.get_scale() if scaler is not None else None
                if scaler is not None:
                    scaler.unscale_(opt)
                grad_norm = _clip(params, cfg.max_grad_norm)   # pre-clip total norm
                if scaler is not None:
                    scaler.step(opt)           # skips internally on inf / nan
                    scaler.update()
                    if scaler.get_scale() < scale_before:
                        skipped += 1
                elif torch.isfinite(grad_norm):
                    opt.step()
                else:
                    skipped += 1
                if scheduler is not None:
                    scheduler.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                logger.advance()

                # Collectives run on every rank at the same step; only rank 0 writes.
                if global_step % cfg.log_every == 0:
                    sums = dd.all_reduce_sum(torch.stack([loss_sum, loss_count.float(), grad_norm.float()]))
                    tokens = float(dd.all_reduce_sum(token_sum))
                    metrics: dict[str, float] = {
                        "loss": float(sums[0] / sums[1].clamp(min=1)),
                        "grad_norm": float(sums[2] / dd.world_size()),
                        "lr": float(opt.param_groups[0]["lr"]),
                        "skipped_steps": float(skipped),
                        "batch_size": window_samples / max(window_batches, 1),
                        "padded_tokens": window_padded / max(window_batches, 1),
                    }
                    if device.type == "cuda":
                        metrics["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
                        torch.cuda.reset_peak_memory_stats(device)
                    if scaler is not None:
                        metrics["loss_scale"] = float(scaler.get_scale())
                    logger.log(global_step, metrics, tokens=tokens)
                    loss_sum.zero_()
                    loss_count.zero_()
                    token_sum.zero_()
                    window_samples = window_padded = window_batches = 0

                done = bool(cfg.max_steps) and global_step >= cfg.max_steps
                if (cfg.save_every_steps and global_step % cfg.save_every_steps == 0) and not done:
                    next_epoch, next_batch = (epoch + 1, 0) if last else (epoch, consumed)
                    _save(model, opt, scheduler, global_step, out_dir,
                          extra={"epoch": next_epoch, "next_batch": next_batch}, hub=cfg.hub)
                if done:
                    stop = True
                    break

            if cfg.val_loader is not None and not stop:
                val = evaluate(model, cfg.val_loader, device, amp_dtype, cfg.batch_adapter)
                logger.scalars(global_step, {"val/loss": val})

        _save(model, opt, scheduler, global_step, out_dir,
              extra={"epoch": cfg.epochs, "next_batch": 0}, tag="final", hub=cfg.hub, final=True)
    finally:
        logger.close()
        if cfg.hub is not None:
            cfg.hub.close()           # the last uploads finish in the background; wait for them
    return {"step": global_step, "skipped": skipped}


# ---- helpers -----------------------------------------------------------------------------------

class _Weights:
    """How much each micro-batch of a window contributes to the loss, and how many target tokens it holds."""

    def __init__(self, factors: list[torch.Tensor], targets: list[torch.Tensor]):
        self.factors, self.targets = factors, targets

    def __getitem__(self, position: int) -> torch.Tensor:
        return self.factors[position]


def _window_weights(window: list[dict], mode: str, device: torch.device) -> _Weights:
    """Loss factors for one accumulation window.

    "batch": 1 / micro-batches, the plain mean (what a fixed batch size gives).
    "token": n_i / N x world_size, where n_i is the micro-batch's target-token count and N the window's total
    over every rank. The model's loss is the mean over a micro-batch's target tokens, so multiplying by n_i
    turns it back into a sum; dividing by N makes every target token count equally whatever the batch shape;
    DDP and FSDP2 average gradients over ranks, which the world_size factor undoes."""
    ones = torch.ones((), device=device)
    if mode == "batch" or any("labels" not in batch for batch in window):
        if mode == "token":
            log.warning("the batch has no 'labels' key, so the loss falls back to the plain batch mean")
        return _Weights([ones / len(window)] * len(window), [ones] * len(window))
    counts = [(batch["labels"][..., 1:] != -100).sum().float() for batch in window]
    total = dd.all_reduce_sum(torch.stack(counts).sum()).clamp(min=1)
    return _Weights([n / total * dd.world_size() for n in counts], counts)


def _log_plan(loader: DataLoader, epoch: int) -> None:
    """One line about the epoch's batches (length-bucketed loaders) and the examples left out for being too long."""
    summary = getattr(loader.batch_sampler, "summary", None)
    if summary is not None:
        info = summary()
        log.info("epoch %d: %d batches, mean batch size %.1f, padding waste %.0f%%, longest %d tokens",
                 epoch, info["batches"], info["mean_batch_size"], 100 * info["padding_waste"], info["longest"])
    dropped = getattr(loader.dataset, "dropped_examples", 0)
    skipped = getattr(loader.dataset, "skipped_documents", 0)
    if dropped or skipped:
        log.info("left out for being longer than max_length: %d example(s), %d document(s) without any fitting example",
                 dropped, skipped)


def _lora_config(model: nn.Module):
    inner = dd.unwrap(model)
    if not getattr(inner, "lora_enabled", False):
        return None
    configs = getattr(inner.model, "peft_config", None) or {}
    return next(iter(configs.values()), None)


def _prepare(batch: dict, adapter: Callable[[dict], dict] | None, device: torch.device) -> dict:
    if adapter is not None:
        batch = adapter(batch)
    return to_device(batch, device)


def _make_scaler(use_scaler: bool, device: torch.device, fsdp: bool):
    if not use_scaler:
        return None
    if fsdp:
        try:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        except ImportError:  # newer torch: the plain scaler works with DTensor grads
            return torch.amp.GradScaler(device.type)
        return ShardedGradScaler()
    return torch.amp.GradScaler(device.type)


def _clip(params: Iterable[nn.Parameter], max_norm: float | None) -> torch.Tensor:
    """Clip to max_norm and return the pre-clip total norm (infinite max_norm only measures)."""
    return torch.nn.utils.clip_grad_norm_(list(params), max_norm if max_norm is not None else math.inf)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp_dtype: torch.dtype,
             adapter: Callable[[dict], dict] | None = None) -> float:
    """Mean loss over a loader (all ranks). Restores the previous train/eval mode afterwards."""
    was_training = model.training
    model.eval()
    total = torch.zeros((), device=device)
    count = 0
    try:
        for batch in Progress(loader, desc="eval", unit="batch", leave=False, disable=not dd.is_main()):
            batch = _prepare(batch, adapter, device)
            with dd.autocast(device, amp_dtype):
                total += model(**batch).loss.float()
            count += 1
    finally:
        model.train(was_training)
    return float(dd.all_reduce_mean(total / max(count, 1)))


def _save(model: nn.Module, opt, scheduler, step: int, out_dir: Path, extra: dict[str, Any],
          tag: str | None = None, hub: Any = None, final: bool = False) -> None:
    # Gathering is collective (FSDP2): every rank calls it, only rank 0 writes.
    model_state = dd.gather_trainable_state(model)
    opt_state = dd.gather_optimizer_state(model, opt)
    if dd.is_main():
        name = f"{tag or 'step'}_{step:07d}.pt"
        path = dd.unwrap(model).save_checkpoint(
            out_dir / name, scheduler=scheduler, step=step, extra=extra,
            model_state=model_state, optimizer_state=opt_state)
        if hub is not None:
            hub.push(step, model_state, training_state=path, final=final, lora_config=_lora_config(model))
    dd.barrier()


def _resume(path: str, model: nn.Module, opt, scheduler) -> tuple[int, int, int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if bool(ckpt.get("lora", False)) != dd.unwrap(model).lora_enabled:
        raise RuntimeError("checkpoint LoRA state does not match the model; attach adapters first")
    dd.load_model_state(model, ckpt["model"])
    if ckpt.get("optimizer") is not None:
        dd.load_optimizer_state(model, opt, ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    extra = ckpt.get("extra") or {}
    return int(extra.get("epoch", 0)), int(extra.get("next_batch", 0)), int(ckpt.get("step") or 0)


# ---- strategy entry points ---------------------------------------------------------------------

def _run(strategy: str, model: nn.Module, loader: DataLoader,
         make_optimizer: Callable[[list[tuple[str, nn.Parameter]]], torch.optim.Optimizer],
         make_scheduler: Callable[[torch.optim.Optimizer], Any] | None,
         cfg: SFTConfig | None) -> dict[str, int]:
    cfg = replace(cfg or SFTConfig(), strategy=strategy)
    device = dd.init_distributed()
    dtype, _ = dd.resolve_precision(cfg.precision, device)
    if cfg.compile:
        dd.unwrap(model).compile()
    if dd.unwrap(model).quantization is None:
        model.to(device)           # bitsandbytes weights are placed by from_pretrained(device_map=...)

    if strategy == "single":
        wrapped = model
    elif strategy == "ddp":
        wrapped = dd.wrap_ddp(model, device)
    elif strategy == "fsdp2":
        wrapped = dd.wrap_fsdp2(model, dtype)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    # Named (name, param) pairs: the hybrid optimizer routes parameters by name as well as by shape.
    opt = make_optimizer([(n, p) for n, p in wrapped.named_parameters() if p.requires_grad])
    scheduler = make_scheduler(opt) if make_scheduler is not None else None
    return train_sft(cfg.epochs, wrapped, opt, loader, scheduler, cfg.save_every_steps, cfg=cfg)


def train_single(model, loader, make_optimizer, make_scheduler=None, cfg=None):
    """One device, no wrapper."""
    return _run("single", model, loader, make_optimizer, make_scheduler, cfg)


def train_ddp(model, loader, make_optimizer, make_scheduler=None, cfg=None):
    """Full replica per rank (torchrun). Works with LoRA and QLoRA."""
    return _run("ddp", model, loader, make_optimizer, make_scheduler, cfg)


def train_fsdp2(model, loader, make_optimizer, make_scheduler=None, cfg=None):
    """Per-parameter sharding with FSDP2 (torchrun). Not compatible with QLoRA."""
    return _run("fsdp2", model, loader, make_optimizer, make_scheduler, cfg)
