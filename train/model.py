from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForImageTextToText
from typing_extensions import Self

LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


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
        return self.model(*args, **kwargs)

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
        return {k: ckpt.get(k) for k in ("step", "extra", "lang_trained", "vision_trained")}

    def save_pretrained(self, save_dir: str | os.PathLike, **kwargs: Any) -> None:
        """Export the wrapped HF model. With LoRA attached this writes the adapter only."""
        self.model.save_pretrained(save_dir, **kwargs)
