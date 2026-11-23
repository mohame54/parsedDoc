"""Push training checkpoints to a Hugging Face repo, without ever stopping training.

    pusher = HubPusher("user/ocr-sft", private=True, every_save=True, run_config=vars(args))
    pusher.push(step=200, model_state=state, training_state="checkpoints/sft/step_0000200.pt", final=False)
    pusher.close()                      # waits for the uploads that are still running

One push writes a folder into the repo (`last/` for periodic saves, `final/` at the end):

    trainable.safetensors      the trained weights (any strategy: single, DDP, FSDP2)
    adapter_model.safetensors  with LoRA: a PEFT adapter,           } PeftModel.from_pretrained(repo,
    adapter_config.json        its configuration                    }   subfolder="last") loads it
    training_state.pt          weights + optimizer + scheduler + epoch / next batch, for `--resume hf://...`
    run_config.json            the command line, the step and the model id

Uploads run in the background (`run_as_future`), and a failed upload is logged as a warning, never raised: a
network problem must not kill a training run. Only rank 0 creates a pusher.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger("train.hub")

HF_PREFIX = "hf://"


def adapter_state(model_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The LoRA tensors of a trainable state dict, named the way PEFT saves them: without the wrapper's
    leading `model.` and without the adapter name (`lora_A.default.weight` -> `lora_A.weight`)."""
    out = {}
    for name, tensor in model_state.items():
        if "lora_" not in name:
            continue
        key = name[len("model."):] if name.startswith("model.") else name
        out[re.sub(r"\.(lora_[AB]|lora_embedding_[AB])\.[^.]+\.weight$", r".\1.weight", key)] = tensor
    return out


def _safe(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}


class HubPusher:
    def __init__(self, repo_id: str, token: str | None = None, private: bool = True, every_save: bool = False,
                 run_config: dict[str, Any] | None = None, stage_dir: str | Path = "checkpoints/hub_stage",
                 api: Any = None, create: bool = True):
        self.repo_id = repo_id
        self.every_save = every_save
        self.run_config = dict(run_config or {})
        self.run_config.pop("hub_token", None)
        self.stage_dir = Path(stage_dir)
        self._futures: list[Any] = []
        self.failures = 0
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi(token=token or os.environ.get("HF_TOKEN"))
        self.api = api
        if create:
            self.api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
            log.info("pushing checkpoints to https://huggingface.co/%s (%s)", repo_id,
                     "every save and at the end" if every_save else "at the end")

    def wants(self, final: bool) -> bool:
        return final or self.every_save

    def push(self, step: int, model_state: dict[str, torch.Tensor], training_state: str | Path | None = None,
             final: bool = False, lora_config: Any = None) -> None:
        """Stage a folder and upload it in the background. `lora_config` is the PEFT LoraConfig (or None)."""
        if not self.wants(final):
            return
        try:
            self._push(step, model_state, training_state, final, lora_config)
        except Exception as exc:                       # never stop training for the network
            self.failures += 1
            log.warning("hub push at step %d failed: %s", step, exc)

    def _push(self, step, model_state, training_state, final, lora_config) -> None:
        from safetensors.torch import save_file

        folder = self.stage_dir / f"step_{step:07d}{'_final' if final else ''}"
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        save_file(_safe(model_state), str(folder / "trainable.safetensors"))
        adapter = adapter_state(model_state)
        if adapter and lora_config is not None:
            save_file(_safe(adapter), str(folder / "adapter_model.safetensors"))
            lora_config.save_pretrained(str(folder))
        if training_state is not None and Path(training_state).exists():
            try:
                os.link(training_state, folder / "training_state.pt")     # no second copy on the same disk
            except OSError:
                shutil.copy2(training_state, folder / "training_state.pt")
        (folder / "run_config.json").write_text(
            json.dumps({**self.run_config, "step": step, "final": final}, indent=2, default=str), encoding="utf-8")

        targets = ["final", "last"] if final else ["last"]
        for target in targets:
            future = self.api.upload_folder(folder_path=str(folder), path_in_repo=target, repo_id=self.repo_id,
                                            repo_type="model", commit_message=f"step {step} ({target})",
                                            run_as_future=True)
            self._track(future, folder if target == targets[-1] else None, step)

    def _track(self, future: Any, cleanup: Path | None, step: int) -> None:
        self._futures.append(future)

        def done(f: Any) -> None:
            try:
                f.result()
                log.info("hub: uploaded step %d to %s", step, self.repo_id)
            except Exception as exc:
                self.failures += 1
                log.warning("hub upload for step %d failed: %s", step, exc)
            if cleanup is not None:
                shutil.rmtree(cleanup, ignore_errors=True)

        if hasattr(future, "add_done_callback"):
            future.add_done_callback(done)

    def close(self, timeout: float = 1800.0) -> None:
        """Wait for the uploads still running (call once, at the end of the run)."""
        for future in self._futures:
            try:
                future.result(timeout=timeout)
            except Exception as exc:
                log.warning("hub upload did not finish: %s", exc)
        self._futures.clear()


def resolve_resume(path: str | None, token: str | None = None) -> str | None:
    """`hf://user/repo/last/training_state.pt` is downloaded and its local path returned; other paths pass."""
    if not path or not path.startswith(HF_PREFIX):
        return path
    from huggingface_hub import hf_hub_download

    parts = path[len(HF_PREFIX):].split("/")
    if len(parts) < 3:
        raise ValueError("expected hf://<user>/<repo>/<path inside the repo>, e.g. hf://me/ocr/last/training_state.pt")
    repo_id, filename = "/".join(parts[:2]), "/".join(parts[2:])
    log.info("downloading the checkpoint %s from %s", filename, repo_id)
    return hf_hub_download(repo_id=repo_id, filename=filename, token=token or os.environ.get("HF_TOKEN"))
