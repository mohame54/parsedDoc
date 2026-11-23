"""Process-group setup, precision choice and the DDP / FSDP2 specific wrapping for SFT.

Everything strategy-specific lives here so `train/sft.py` stays one loop:
    single  -> model on one device, no wrapper
    ddp     -> DistributedDataParallel (full replica per rank; works with QLoRA)
    fsdp2   -> torch.distributed.fsdp.fully_shard (per-parameter shards; not for QLoRA)
"""
from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


# ---- process group ---------------------------------------------------------------------------

def init_distributed() -> torch.device:
    """Initialise the process group when launched by torchrun; returns this rank's device."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            dist.init_process_group("nccl")
            return torch.device("cuda", local_rank)
        dist.init_process_group("gloo")
        return torch.device("cpu")
    return default_device()


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return rank() == 0


def barrier() -> None:
    if is_dist():
        dist.barrier()


def all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    """Sum over ranks of a tensor (returns a new tensor; identity on one process)."""
    value = value.detach().clone()
    if is_dist():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    """Mean over ranks of a tensor (returns a new tensor; identity on one process)."""
    value = value.detach().clone()
    if is_dist():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size()
    return value


# ---- precision -------------------------------------------------------------------------------

def resolve_precision(pref: str = "auto", device: torch.device | str = "cpu") -> tuple[torch.dtype, bool]:
    """Return (autocast/compute dtype, use_grad_scaler).

    auto: CUDA with bf16 support -> bf16, no scaler; other CUDA -> fp16 + scaler;
          CPU and MPS -> fp32 (no autocast).
    Explicit "bf16" / "fp16" / "fp32" are honoured, with a clear error when the device cannot run them.
    """
    device_type = torch.device(device).type
    if pref == "auto":
        if device_type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16, False
            return torch.float16, True
        return torch.float32, False
    if pref == "bf16":
        if device_type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested but this CUDA device does not support it")
        return torch.bfloat16, False
    if pref == "fp16":
        if device_type != "cuda":
            raise RuntimeError("fp16 autocast with a grad scaler needs CUDA")
        return torch.float16, True
    if pref == "fp32":
        return torch.float32, False
    raise ValueError("precision must be 'auto', 'bf16', 'fp16' or 'fp32'")


def autocast(device: torch.device | str, dtype: torch.dtype):
    """Autocast context; a no-op for fp32."""
    if dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype)


# ---- wrapping --------------------------------------------------------------------------------

def wrap_ddp(model: nn.Module, device: torch.device) -> DDP:
    """DDP replica per rank. Frozen groups have no grads, so unused-parameter search is off."""
    device_ids = [device.index] if device.type == "cuda" else None
    return DDP(
        model,
        device_ids=device_ids,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )


def wrap_fsdp2(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    """Shard the language decoder layers, then the root, with FSDP2 `fully_shard`.

    Per-parameter sharding lets frozen groups (vision tower, base weights under LoRA) and
    trainable ones coexist. Parameters are kept in fp32 with bf16/fp16 compute and fp32 reduce.
    """
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    if getattr(model, "quantization", None) is not None:
        raise RuntimeError("QLoRA (4/8-bit bitsandbytes weights) cannot be sharded with FSDP2; use ddp or single")
    inner = unwrap(model)
    layers = getattr(inner._resolve_from_hf("model.language_model"), "layers", None)
    if layers is None:
        raise RuntimeError("language model has no `layers` attribute; cannot shard per decoder layer")
    policy = MixedPrecisionPolicy(
        param_dtype=None if dtype == torch.float32 else dtype,
        reduce_dtype=torch.float32,
    )
    for layer in layers:
        fully_shard(layer, mp_policy=policy)
    return fully_shard(model, mp_policy=policy)


def is_fsdp(model: nn.Module) -> bool:
    from torch.distributed.fsdp import FSDPModule

    return isinstance(model, FSDPModule)


def unwrap(model: nn.Module) -> Any:
    """The DeepSeekOCR wrapper under DDP (FSDP2 wraps in place, so it is already the wrapper)."""
    return model.module if isinstance(model, DDP) else model


def model_device(model: nn.Module) -> torch.device:
    return next(unwrap(model).parameters()).device


@contextlib.contextmanager
def grad_sync(model: nn.Module, sync: bool) -> Iterator[None]:
    """Gradient all-reduce control for gradient accumulation.

    sync=False on every micro-batch except the last of an accumulation window. DDP uses no_sync();
    FSDP2 toggles requires-gradient-sync on the root (the reduce-scatter is then deferred).
    """
    if isinstance(model, DDP) and not sync:
        with model.no_sync():
            yield
        return
    if hasattr(model, "set_requires_gradient_sync"):
        model.set_requires_gradient_sync(sync)
    yield


# ---- state gather / load (rank-collective for FSDP2; call on every rank) ---------------------

def gather_trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Full trainable weights on CPU. Under FSDP2 every rank must call this (it all-gathers)."""
    inner = unwrap(model)
    if not is_fsdp(model):
        return inner.trainable_state_dict()
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    trainable = {name for name, p in inner.named_parameters() if p.requires_grad}
    full = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    return {name: tensor for name, tensor in full.items() if name in trainable}


def gather_optimizer_state(model: nn.Module, opt: torch.optim.Optimizer) -> dict[str, Any]:
    if not is_fsdp(model):
        return opt.state_dict()
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

    return get_optimizer_state_dict(
        model, opt, options=StateDictOptions(full_state_dict=True, cpu_offload=True))


def load_model_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load a trainable-only (or full) state dict into the model, sharded or not."""
    if not is_fsdp(model):
        result = unwrap(model).load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"checkpoint has unexpected keys: {result.unexpected_keys[:5]}")
        return
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    set_model_state_dict(model, state, options=StateDictOptions(full_state_dict=True, strict=False))


def load_optimizer_state(model: nn.Module, opt: torch.optim.Optimizer, state: dict[str, Any]) -> None:
    if not is_fsdp(model):
        opt.load_state_dict(state)
        return
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict

    set_optimizer_state_dict(model, opt, state, options=StateDictOptions(full_state_dict=True))
