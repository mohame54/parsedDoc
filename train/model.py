from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForImageTextToText
from typing_extensions import Self

LOSS_CHUNK = 512          # supervised positions whose logits exist at the same time in the chunked loss

LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def _cross_entropy_sum(lm_head: nn.Module, states: torch.Tensor, wanted: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(lm_head(states).float(), wanted, reduction="sum")


def parse_targets(value: Any) -> Any:
    """LoRA target modules from a flag or YAML value: a regex string stays one; "q_proj,v_proj" or a list becomes a
    list of module names."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [part.strip() for part in value.split(",")] if "," in value and "(" not in value else value


class DeepSeekOCR(nn.Module):
    """Thin wrapper around a HF DeepseekOcr*ForConditionalGeneration model that lets
    the language side and the vision side be trained (or frozen) independently.

    A "frozen" group has ``requires_grad=False`` on all its parameters and is kept in
    eval mode (no dropout) even while the wrapper is in train mode.

    LoRA / QLoRA: ``add_lora`` wraps ``self.model`` with PEFT. Chains are then resolved
    through the PEFT-unwrapped base model. Under LoRA only the ``lora_`` adapter tensors
    inside a trained group are trainable; the base weights stay frozen.
    """

    LANG_MODULES = ("model.language_model", "lm_head")
    VISION_MODULES = (
        "model.multi_modal_projector",
        "model.vision_tower",
        "model.view_separator",  # nn.Parameter, not an nn.Module
    )

    def __init__(
        self,
        model: nn.Module,
        lang_trained: bool = True,
        vision_trained: bool = False,
        quantization: str | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.lang_trained = lang_trained
        self.vision_trained = vision_trained
        self.quantization = quantization
        self.lora_enabled = False
        self.lineage: list[dict[str, Any]] = []   # adapters merged into the base before the current one (continue_from)
        self.chunked_loss = True
        self.apply_training()

    # ---- module resolution -------------------------------------------------------------

    @property
    def base(self) -> nn.Module:
        """The HF model with any PEFT wrapper removed (names match the HF checkpoint)."""
        return self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model

    def _resolve_from_hf(self, chain: str) -> nn.Module | nn.Parameter:
        # Chains are relative to the (un-PEFT-wrapped) HF model, e.g. "model.language_model", "lm_head".
        target: Any = self.base
        for name in chain.split("."):
            target = getattr(target, name, None)
            if target is None:
                raise AttributeError(f"'{name}' (from '{chain}') not found in {type(self.base).__name__}")
        return target

    # ---- freezing ----------------------------------------------------------------------

    def set_train_chain(self, chains: tuple[str, ...] | list[str], training: bool) -> None:
        """Set requires_grad and train/eval mode for every module/parameter on ``chains``."""
        for chain in chains:
            target = self._resolve_from_hf(chain)
            if isinstance(target, nn.Parameter):
                target.requires_grad_(training)
                continue
            if self.lora_enabled:
                # Only the adapters train; base weights (and lm_head) stay frozen.
                for name, param in target.named_parameters():
                    param.requires_grad_(training and "lora_" in name)
            else:
                target.requires_grad_(training)
            # Train mode only when this wrapper is in train mode too.
            target.train(training and self.training)

    def apply_training(
        self,
        lang_trained: bool | None = None,
        vision_trained: bool | None = None,
    ) -> None:
        if lang_trained is not None:
            self.lang_trained = lang_trained
        if vision_trained is not None:
            self.vision_trained = vision_trained

        self.set_train_chain(self.LANG_MODULES, self.lang_trained)
        self.set_train_chain(self.VISION_MODULES, self.vision_trained)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)  # recursively sets every submodule to `mode`
        self.apply_training()  # re-freeze the groups that must stay in eval mode
        return self

    # ---- construction ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        lang_trained: bool = True,
        vision_trained: bool = False,
        dtype: torch.dtype | None = None,
        quantization: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """``quantization``: None, "4bit" (NF4 + double quant) or "8bit" (bitsandbytes).
        ``dtype`` is the compute dtype for quantized weights and the load dtype otherwise."""
        if quantization not in (None, "4bit", "8bit"):
            raise ValueError("quantization must be None, '4bit' or '8bit'")
        if quantization is not None:
            from transformers import BitsAndBytesConfig

            if quantization == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype or torch.bfloat16,
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        if dtype is not None:
            kwargs["dtype"] = dtype
        base_model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        return cls(base_model, lang_trained=lang_trained, vision_trained=vision_trained,
                   quantization=quantization)

    # ---- LoRA / QLoRA ------------------------------------------------------------------

    def add_lora(
        self,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: str | list[str] = LORA_TARGETS,
    ) -> None:
        """Wrap the language model with LoRA adapters. Call once, before optimizer creation.

        On a quantized model the k-bit preparation (fp32 norms, input grads) runs first.
        Vision-side LoRA is not added; the vision group stays frozen unless trained fully.
        """
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if self.lora_enabled:
            raise RuntimeError("LoRA adapters are already attached")
        model = self.model
        if self.quantization is not None:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                            target_modules=target_modules)
        self.model = get_peft_model(model, config)
        self.lora_enabled = True
        self.apply_training()

    def lora_config_dict(self) -> dict[str, Any] | None:
        """The attached adapter's shape (r, alpha, dropout, target modules), or None without LoRA."""
        if not self.lora_enabled:
            return None
        cfg = self.model.peft_config["default"]
        targets = cfg.target_modules
        return {"r": cfg.r, "alpha": cfg.lora_alpha, "dropout": cfg.lora_dropout,
                "target_modules": sorted(targets) if isinstance(targets, (set, list, tuple)) else targets}

    def continue_from(self, checkpoint: str | os.PathLike, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                      target_modules: str | list[str] | None = None, old_r: int | None = None,
                      old_alpha: int | None = None, old_target_modules: str | list[str] | None = None,
                      token: str | None = None) -> dict[str, Any]:
        """Start a new stage from an earlier adapter, with a LoRA configuration of its own.

        The earlier adapter is loaded and MERGED into the base weights, then a fresh adapter (`r`, `alpha`, ...) is
        attached for the new stage. A checkpoint's adapter is relative to the base plus every adapter merged before
        it, so the whole chain recorded in the checkpoint (`lineage`) is replayed first: load, merge, repeat. The
        checkpoint's own adapter shape is read from it; `old_r` / `old_alpha` / `old_target_modules` are only for
        checkpoints written before that was stored (they default to the standard targets). Needs an unquantized base (a quantized one cannot be merged into).

        Returns the checkpoint's metadata. Optimizer, scheduler and step are NOT restored: this is a new run (use
        --resume to carry on the same one)."""
        from .hub import resolve_resume
        if self.lora_enabled:
            raise RuntimeError("continue_from must come before add_lora: it merges the old adapter first")
        if self.quantization is not None:
            raise RuntimeError("cannot merge an earlier adapter into a quantized base; continue from an unquantized "
                               "model (the new stage can still be LoRA)")
        path = resolve_resume(str(checkpoint), token)
        head = torch.load(path, map_location="cpu", weights_only=False)
        chain = list(head.get("lineage") or [])
        stored = head.get("lora_config") or {}
        this = {"path": str(checkpoint), "r": stored.get("r", old_r), "alpha": stored.get("alpha", old_alpha),
                "target_modules": stored.get("target_modules") or old_target_modules}
        if bool(head.get("lora", False)):
            if this["r"] is None or this["alpha"] is None:
                raise ValueError("this checkpoint does not record its LoRA shape (it predates that); pass the old "
                                 "r and alpha (--init-lora-r / --init-lora-alpha)")
            chain.append(this)
            for entry in chain:                                  # base + A, then + B, ...
                self._merge_checkpoint(entry, token)
        else:                                                    # a full fine-tune: its weights simply replace the base's
            self.load_checkpoint(path)
        self.lineage = chain
        kwargs = {} if target_modules is None else {"target_modules": target_modules}
        self.add_lora(r=r, alpha=alpha, dropout=dropout, **kwargs)
        return {k: head.get(k) for k in ("step", "extra")}

    def _merge_checkpoint(self, entry: dict[str, Any], token: str | None = None) -> None:
        from .hub import resolve_resume
        try:
            path = resolve_resume(entry["path"], token)
            if not path or not Path(path).exists():
                raise FileNotFoundError(path)
        except Exception as error:
            raise FileNotFoundError(
                f"cannot replay the adapter chain: {entry['path']} is missing ({error}). Every earlier checkpoint in "
                "the lineage must still be reachable at the path it had when the next stage started.") from error
        kwargs = {} if not entry.get("target_modules") else {"target_modules": entry["target_modules"]}
        self.add_lora(r=entry["r"], alpha=entry["alpha"], dropout=0.0, **kwargs)
        self.load_checkpoint(path)
        self.merge_and_unload()

    def save_lora(self, save_dir: str | os.PathLike) -> None:
        """Adapter-only export (small); reload with ``load_lora`` on a freshly built base."""
        if not self.lora_enabled:
            raise RuntimeError("no LoRA adapters attached")
        self.model.save_pretrained(save_dir)

    def load_lora(self, adapter_dir: str | os.PathLike, is_trainable: bool = True) -> None:
        from peft import PeftModel

        if self.lora_enabled:
            raise RuntimeError("LoRA adapters are already attached")
        self.model = PeftModel.from_pretrained(self.base, adapter_dir, is_trainable=is_trainable)
        self.lora_enabled = True
        self.apply_training()

    def merge_and_unload(self) -> None:
        """Fold the adapters into the base weights for inference export (not for quantized models)."""
        if not self.lora_enabled:
            raise RuntimeError("no LoRA adapters attached")
        if self.quantization is not None:
            raise RuntimeError("cannot merge LoRA into a quantized base; export the adapter instead")
        self.model = self.model.merge_and_unload()
        self.lora_enabled = False
        self.apply_training()

    # ---- forward / compile -------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any):
        if self.chunked_loss and not args and kwargs.get("labels") is not None:
            return self._chunked_forward(dict(kwargs))
        return self.model(*args, **kwargs)

    def _chunked_forward(self, kwargs: dict[str, Any]):
        """The same loss as the HF forward, without the full logits.

        The HF model builds logits of shape (batch, length, vocabulary) for every position and then upcasts them to
        fp32 for the loss. With a 129,280-token vocabulary that is about 2 MB per token and most of what a long batch
        costs. The loss only needs the supervised positions (the answer, a fraction of the length), so this runs the
        network up to the last hidden state, keeps the positions that have a label, and applies the output layer
        and the cross-entropy to them `LOSS_CHUNK` at a time under activation checkpointing: a chunk's logits are
        dropped after use and recomputed during backward. The value and the gradients equal the HF loss (the
        shift, the -100 ignore index and the mean over supervised tokens are the same).
        """
        labels = kwargs.pop("labels")
        kwargs.pop("logits_to_keep", None)
        kwargs.setdefault("use_cache", False)
        hf = self.base
        hidden = hf.model(**kwargs)[0][:, :-1]                  # position t predicts token t + 1
        target = labels[:, 1:].to(hidden.device)
        keep = target != -100
        states, wanted = hidden[keep], target[keep]
        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, states.shape[0], LOSS_CHUNK):
            total = total + checkpoint(_cross_entropy_sum, hf.lm_head, states[start:start + LOSS_CHUNK],
                                       wanted[start:start + LOSS_CHUNK], use_reentrant=False)
        return SimpleNamespace(loss=total / keep.sum().clamp(min=1), logits=None)

    def enable_grad_checkpointing(self) -> int:
        """Recompute activations during backward instead of storing them (about 3-5x less activation memory for
        roughly a third more compute). Covers the decoder layers and, when it trains, the vision tower: the HF
        model's layers checkpoint themselves while in train mode, so evaluation and generation are unaffected.
        Non-reentrant, so frozen inputs (LoRA over a frozen base) need no extra hooks. Returns how many layers do it."""
        self.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.base.config.use_cache = False
        return sum(1 for m in self.base.modules() if getattr(m, "gradient_checkpointing", False))

    def compile(self, **compile_kwargs: Any) -> None:
        """Compile only the language model and lm_head (not the whole wrapper).

        The vision side has dynamic tiling shapes, so it is left eager. ``compile_kwargs``
        are the usual ``torch.compile`` options (``mode``, ``fullgraph``, ``dynamic``, ...).
        Compilation is in place, so state_dict keys stay free of ``_orig_mod.``.
        """
        for chain in self.LANG_MODULES:
            self._resolve_from_hf(chain).compile(**compile_kwargs)

    # ---- state dicts / checkpoints -------------------------------------------------

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """State dict restricted to parameters with ``requires_grad=True`` (detached, on CPU)."""
        return {
            name: param.detach().cpu()
            for name, param in self.named_parameters()
            if param.requires_grad
        }

    def save_checkpoint(
        self,
        path: str | os.PathLike,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        step: int | None = None,
        trainable_only: bool = True,
        extra: dict[str, Any] | None = None,
        model_state: dict[str, torch.Tensor] | None = None,
        optimizer_state: dict[str, Any] | None = None,
    ) -> Path:
        """Write a resumable training checkpoint atomically (tmp file + rename).

        ``trainable_only=True`` stores just the parameters being trained, which keeps
        checkpoints small when the vision side is frozen; the frozen weights are
        recovered from the base model on load. ``model_state`` / ``optimizer_state``
        accept pre-gathered dicts (used by the sharded FSDP2 path, where only rank 0 writes).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if model_state is None:
            model_state = self.trainable_state_dict() if trainable_only else self.state_dict()
        if optimizer_state is None and optimizer is not None:
            optimizer_state = optimizer.state_dict()
        ckpt = {
            "model": model_state,
            "trainable_only": trainable_only,
            "lora": self.lora_enabled,
            "lora_config": self.lora_config_dict(),
            "lineage": self.lineage,
            "lang_trained": self.lang_trained,
            "vision_trained": self.vision_trained,
            "step": step,
            "optimizer": optimizer_state,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "extra": extra,
        }
        tmp = path.with_name(path.name + ".tmp")
        torch.save(ckpt, tmp)
        os.replace(tmp, path)
        return path

    def load_checkpoint(
        self,
        path: str | os.PathLike,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Restore a checkpoint from :meth:`save_checkpoint`; returns its metadata
        (``step``, ``extra``, ...). Unexpected keys always raise; missing keys raise
        only for full (non ``trainable_only``) checkpoints. Requires the same LoRA state
        as when the checkpoint was written (attach adapters first)."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if bool(ckpt.get("lora", False)) != self.lora_enabled:
            raise RuntimeError("checkpoint LoRA state does not match the model; attach adapters first")
        result = self.load_state_dict(ckpt["model"], strict=False)
        if result.unexpected_keys or (not ckpt["trainable_only"] and result.missing_keys):
            raise RuntimeError(
                f"Checkpoint mismatch: missing={result.missing_keys[:5]} "
                f"unexpected={result.unexpected_keys[:5]}"
            )
        self.apply_training(ckpt["lang_trained"], ckpt["vision_trained"])
        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        return {k: ckpt.get(k) for k in ("step", "extra", "lang_trained", "vision_trained", "lora_config", "lineage")}

    def save_pretrained(self, save_dir: str | os.PathLike, **kwargs: Any) -> None:
        """Export the wrapped HF model. With LoRA attached this writes the adapter only."""
        self.model.save_pretrained(save_dir, **kwargs)
