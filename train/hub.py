"""Push training checkpoints to a Hugging Face repo, without ever stopping training.

    pusher = HubPusher("user/ocr-sft", private=True, every_save=True, run_config=vars(args))
    pusher.push(step=200, model_state=state, training_state="checkpoints/sft/step_0000200.pt", final=False)
    pusher.close()                      # waits for the uploads that are still running

Every save gets a folder of its own, so a long run can be reviewed afterwards and the best checkpoint picked:

    step_0000200/              a periodic save           (final_0000612/ is the end of the run)
    sft_step_0000200/          the same with --state-prefix sft; rl_step_... for RL, so the two never mix
    summary/                   index.md, checkpoints.csv, compare.png: every checkpoint side by side
    last/                      only with keep_last=True: a copy of the newest save

and each checkpoint folder holds:

    trainable.safetensors      the trained weights (any strategy: single, DDP, FSDP2)
    adapter_model.safetensors  with LoRA: a PEFT adapter,           } PeftModel.from_pretrained(repo,
    adapter_config.json        its configuration                    }   subfolder="step_0000200") loads it
    adapter_state.pt           LoRA runs: weights + the adapter's shape + the chain of adapters it sits on, no optimizer.
                               Small. This is what `--init-from hf://user/repo/step_0000200/adapter_state.pt` loads
    training_state.pt          weights + optimizer + scheduler + epoch / next batch, for `--resume hf://...`
                               (left out with state=False: smaller folders, but no resume from them)
    run_config.json            the command line, the step and the model id
    metrics.json, report.md    the metrics report of this checkpoint (train/ckpt_report.py), with its graphs
    loss.png, ...              (the graph files depend on the kind of training: SFT or RL)

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
STATE_FILES = ("training_state.pt", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")


def state_name(prefix: str, step: int, final: bool = False) -> str:
    """The name of a save: step_0000200, final_0000612, and with a prefix sft_step_0000200 / rl_final_0000612. Used for
    the local checkpoint file, its report folder and the folder in the Hub repo, so all three match."""
    base = f"{'final' if final else 'step'}_{step:07d}"
    prefix = (prefix or "").strip("_")
    return f"{prefix}_{base}" if prefix else base


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
                 api: Any = None, create: bool = True, prefix: str = "", keep_last: bool = False, state: bool = True):
        """prefix: put before every folder name (`sft` -> `sft_step_0000200`). keep_last: also keep a `last/` copy of
        the newest save. state: include training_state.pt (optimizer etc.) in every folder."""
        self.repo_id = repo_id
        self.every_save = every_save
        self.prefix = prefix.strip("_")
        self.keep_last = keep_last
        self.state = state
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

    def name(self, step: int, final: bool = False) -> str:
        """The folder name of a save in the repo (and in the staging area): step_0000200, final_0000612, with the prefix."""
        return state_name(self.prefix, step, final)

    def push(self, step: int, model_state: dict[str, torch.Tensor], training_state: str | Path | None = None,
             final: bool = False, lora_config: Any = None, report_dir: str | Path | None = None) -> None:
        """Stage a folder and upload it in the background. `lora_config` is the PEFT LoraConfig (or None).
        `report_dir`: the checkpoint's metrics report (metrics.json, report.md, graphs); its files go into the folder,
        and the comparison files next to it (index.md, checkpoints.csv, compare.png) into `summary/`."""
        if not self.wants(final):
            return
        try:
            self._push(step, model_state, training_state, final, lora_config, report_dir)
        except Exception as exc:                       # never stop training for the network
            self.failures += 1
            log.warning("hub push at step %d failed: %s", step, exc)

    def push_folder(self, step: int, folder: str | Path, final: bool = False, report_dir: str | Path | None = None) -> None:
        """Upload a folder somebody else already wrote (a TRL checkpoint, say) under this save's name, with the report."""
        if not self.wants(final):
            return
        try:
            staged = self._fresh_stage(step, final)
            for item in Path(folder).iterdir():
                if item.is_file() and (self.state or item.name not in STATE_FILES):
                    shutil.copy2(item, staged / item.name)
            self._finish(step, final, staged, report_dir)
        except Exception as exc:
            self.failures += 1
            log.warning("hub push at step %d failed: %s", step, exc)

    def _fresh_stage(self, step: int, final: bool) -> Path:
        folder = self.stage_dir / self.name(step, final)
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        return folder

    def _push(self, step, model_state, training_state, final, lora_config, report_dir=None) -> None:
        from safetensors.torch import save_file

        folder = self._fresh_stage(step, final)
        save_file(_safe(model_state), str(folder / "trainable.safetensors"))
        adapter = adapter_state(model_state)
        if adapter and lora_config is not None:
            save_file(_safe(adapter), str(folder / "adapter_model.safetensors"))
            lora_config.save_pretrained(str(folder))
        if training_state is not None and Path(training_state).exists():
            self._write_adapter_state(Path(training_state), folder / "adapter_state.pt")
            if self.state:
                try:
                    os.link(training_state, folder / "training_state.pt")     # no second copy on the same disk
                except OSError:
                    shutil.copy2(training_state, folder / "training_state.pt")
        self._finish(step, final, folder, report_dir)

    @staticmethod
    def _write_adapter_state(source: Path, target: Path) -> None:
        """The checkpoint without optimizer and scheduler, for LoRA runs only (a full fine-tune's weights are already
        in trainable.safetensors). Never raises: it is an extra."""
        try:
            ckpt = torch.load(source, map_location="cpu", weights_only=False)
            if ckpt.get("lora"):
                torch.save({k: v for k, v in ckpt.items() if k not in ("optimizer", "scheduler")}, target)
        except Exception as exc:                                               # noqa: BLE001
            log.warning("could not write adapter_state.pt from %s: %s", source, exc)

    def _finish(self, step: int, final: bool, folder: Path, report_dir: str | Path | None) -> None:
        """Adds the run config and the report to the staged folder, then uploads it (and the summary)."""
        (folder / "run_config.json").write_text(
            json.dumps({**self.run_config, "step": step, "final": final}, indent=2, default=str), encoding="utf-8")
        summary: Path | None = None
        if report_dir is not None and Path(report_dir).is_dir():
            for item in Path(report_dir).iterdir():
                if item.is_file():
                    shutil.copy2(item, folder / item.name)
            index = Path(report_dir).parent
            wanted = [index / n for n in ("index.md", "checkpoints.csv", "compare.png") if (index / n).exists()]
            if wanted:
                summary = self.stage_dir / (f"{self.prefix}_summary" if self.prefix else "summary")
                if summary.exists():
                    shutil.rmtree(summary)
                summary.mkdir(parents=True)
                for item in wanted:
                    shutil.copy2(item, summary / item.name)
        name = self.name(step, final)
        targets = [(name, folder)] + ([(f"{self.prefix}_last" if self.prefix else "last", folder)] if self.keep_last else [])
        if summary is not None:
            targets.append((summary.name, summary))
        for number, (target, source) in enumerate(targets):
            future = self.api.upload_folder(folder_path=str(source), path_in_repo=target, repo_id=self.repo_id,
                                            repo_type="model", commit_message=f"step {step} ({target})",
                                            run_as_future=True)
            owners = [t for t in targets if t[1] == source]                    # clean a staged folder after its last upload
            self._track(future, source if (target, source) == owners[-1] else None, step)

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
