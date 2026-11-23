"""Hybrid optimizer built from torch's own implementations: `torch.optim.Muon` for 2D hidden weight
matrices and `torch.optim.AdamW` for everything else (embeddings, output head, norms, biases, ...).

    opt = build_optimizer(named_trainable_params, muon_lr=0.02, adamw_lr=2e-5)

`MuonAdamW` is only a thin container: it exposes both optimizers' param groups as one list (the very
same dicts, so LR schedulers act on both), and steps / zero-grads / checkpoints them together.

Limitation: `torch.optim.Muon` works on plain tensors. Under FSDP2 the parameters are sharded
DTensors and orthogonalising a local shard is wrong, so `build_optimizer` warns and uses AdamW
for every trainable parameter in that case (use DDP / single device to get Muon).
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Iterable

import torch
from torch import nn
from torch.optim import AdamW, Muon, Optimizer
from torch.optim.lr_scheduler import LambdaLR

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover - older torch
    DTensor = ()

# 2D tensors whose names contain these fragments stay on AdamW: Muon is meant for hidden layers,
# not for the input embedding or the output head.
ADAMW_NAMES = ("embed", "lm_head")


class MuonAdamW(Optimizer):
    """Steps a `torch.optim.Muon` and a `torch.optim.AdamW` as one optimizer."""

    def __init__(self, muon: Muon | None, adamw: AdamW | None) -> None:
        self.optimizers: list[Optimizer] = [o for o in (muon, adamw) if o is not None]
        if not self.optimizers:
            raise ValueError("no optimizers to combine")
        # Deliberately not calling Optimizer.__init__: the sub-optimizers own their groups and state.
        self.defaults: dict[str, Any] = {}
        self.param_groups = [g for o in self.optimizers for g in o.param_groups]

    @property
    def state(self):  # type: ignore[override]
        merged: dict[Any, Any] = {}
        for o in self.optimizers:
            merged.update(o.state)
        return merged

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for o in self.optimizers:
            o.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {"optimizers": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        states = state_dict["optimizers"]
        if len(states) != len(self.optimizers):
            raise ValueError("optimizer checkpoint has a different Muon/AdamW layout")
        for o, s in zip(self.optimizers, states):
            o.load_state_dict(s)
        self.param_groups = [g for o in self.optimizers for g in o.param_groups]


def split_params(
    named_params: Iterable[tuple[str, nn.Parameter]],
    adamw_names: tuple[str, ...] = ADAMW_NAMES,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """(muon, adamw): Muon takes trainable 2D tensors whose name does not match ``adamw_names``."""
    muon, adamw = [], []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        if param.ndim == 2 and not any(fragment in name for fragment in adamw_names):
            muon.append(param)
        else:
            adamw.append(param)
    return muon, adamw


def build_optimizer(
    named_params: Iterable[tuple[str, nn.Parameter]],
    muon_lr: float = 0.02,
    adamw_lr: float = 2e-5,
    weight_decay: float = 0.01,
    muon_momentum: float = 0.95,
    ns_steps: int = 5,
    adjust_lr_fn: str | None = None,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> MuonAdamW:
    """Default optimizer for SFT. ``named_params`` are the trainable (name, param) pairs of the model.

    ``adjust_lr_fn`` is torch Muon's update scaling: None/"original" (Keller), "match_rms_adamw"
    (update RMS matches AdamW, so AdamW-tuned lr / weight decay carry over to Muon: handy for
    fine-tuning with muon_lr ~ adamw_lr) or "spectral_unclamped".
    """
    muon_params, adamw_params = split_params(named_params)
    if any(isinstance(p, DTensor) for p in muon_params):
        warnings.warn(
            "torch.optim.Muon does not support sharded (FSDP2) parameters; falling back to AdamW "
            "for all trainable parameters (use ddp or single to get Muon).",
            stacklevel=2,
        )
        muon_params, adamw_params = [], muon_params + adamw_params
    muon = (Muon(muon_params, lr=muon_lr, weight_decay=weight_decay, momentum=muon_momentum,
                 ns_steps=ns_steps, adjust_lr_fn=adjust_lr_fn) if muon_params else None)
    adamw = (AdamW(adamw_params, lr=adamw_lr, weight_decay=weight_decay, betas=betas, eps=eps)
             if adamw_params else None)
    return MuonAdamW(muon, adamw)


def build_scheduler(optimizer: Optimizer, total_steps: int, kind: str = "cosine", warmup_ratio: float = 0.05,
                    min_lr_ratio: float = 0.1) -> LambdaLR:
    """Linear warmup over `warmup_ratio` of the steps, then cosine / linear decay to `min_lr_ratio` x the base
    learning rate (or flat for "constant"). It scales every param group, so Muon and AdamW follow the same curve.

    `total_steps` is an estimate when batches are length-bucketed (the count varies a little per epoch): the
    curve stops at its floor if the run is longer and ends slightly above it if shorter.
    """
    if kind not in ("cosine", "linear", "constant"):
        raise ValueError("scheduler must be cosine, linear or constant")
    total = max(int(total_steps), 1)
    warmup = max(1, round(total * warmup_ratio)) if warmup_ratio > 0 else 0

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if kind == "constant":
            return 1.0
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        decay = 1.0 - progress if kind == "linear" else 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay

    return LambdaLR(optimizer, factor)
