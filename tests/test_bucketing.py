"""Length-bucketed batches, per-token loss weighting and the Hugging Face push. CPU only, no downloads."""
import json
from concurrent.futures import Future
from pathlib import Path

import pytest
import torch
from PIL import Image

from train import distributed as dd
from train.dataset import LengthBucketBatchSampler, make_loader, set_epoch
from train.hub import HubPusher, adapter_state, resolve_resume
from train.processors import DatasetProcessor
from train.sft import SFTConfig, _window_weights, train_sft

from .test_derive import _record
from .test_train_batching import _transform


def _docs(tmp_path: Path, n: int = 24) -> Path:
    Image.new("RGB", (4, 4)).save(tmp_path / "p.png")
    rows = [dict(_record(), id=f"d:{i}") for i in range(n)]
    for split in ("train", "val"):
        (tmp_path / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return tmp_path


def _loader(root, **kw):
    kw.setdefault("batch_tokens", 600)
    kw.setdefault("max_batch_size", 8)
    return make_loader(root, "train", DatasetProcessor(preset="structured"), transform=_transform(), **kw)


def _plan(loader):
    return [[index for _, index in batch] for batch in loader.batch_sampler]


def test_batches_respect_the_token_budget_and_cover_every_example_once(tmp_path):
    loader = _loader(_docs(tmp_path), pad_multiple=16)
    sampler = loader.batch_sampler
    lengths = {i: -(-loader.dataset.length_of(0, i, sampler.token_length) // 16) * 16
               for i in range(len(loader.dataset))}
    plan = _plan(loader)
    assert sorted(i for batch in plan for i in batch) == list(range(len(loader.dataset)))
    for batch in plan:
        assert len(batch) <= 8
        assert len(batch) * max(lengths[i] for i in batch) <= 600
    batch = next(iter(loader))
    assert batch["input_ids"].shape[1] % 16 == 0                        # the collator pads to the multiple
    assert batch["input_ids"].shape[0] * batch["input_ids"].shape[1] <= 600


def test_the_plan_is_reproducible_per_epoch_and_changes_between_epochs(tmp_path):
    root = _docs(tmp_path)
    a, b = _loader(root), _loader(root)
    assert _plan(a) == _plan(b)
    first = _plan(a)
    set_epoch(a, 1)
    assert _plan(a) != first
    set_epoch(b, 1)
    assert _plan(a) == _plan(b)
    assert len(a) == len(a.batch_sampler._plan) > 1


def test_bucketing_cuts_padding_against_random_fixed_size_batches(tmp_path):
    root = _docs(tmp_path, 48)
    bucketed = _loader(root, batch_tokens=800, max_batch_size=16)
    info = bucketed.batch_sampler.summary()
    sizes = {i: bucketed.dataset.length_of(0, i, bucketed.batch_sampler.token_length) for i in range(48)}
    size = max(2, round(info["mean_batch_size"]))
    order = list(range(48))
    import random
    random.Random(0).shuffle(order)
    batches = [order[i:i + size] for i in range(0, 48 - size + 1, size)]
    padded = sum(len(b) * max(sizes[i] for i in b) for b in batches)
    waste_random = 1 - sum(sizes[i] for b in batches for i in b) / padded
    assert info["padding_waste"] < waste_random


def test_ranks_run_the_same_number_of_steps_on_disjoint_batches(tmp_path):
    root = _docs(tmp_path, 30)
    plans = [_plan(_loader(root, rank=r, world_size=2)) for r in range(2)]
    assert len(plans[0]) == len(plans[1]) > 0
    flat = [sum(p, []) for p in plans]
    assert not set(flat[0]) & set(flat[1])


def test_max_length_must_fit_one_batch_and_defaults_to_the_batch(tmp_path):
    root = _docs(tmp_path)
    transform = _transform(max_length=700)
    with pytest.raises(ValueError, match="does not fit one batch"):
        make_loader(root, "train", DatasetProcessor(preset="structured"), transform=transform, batch_tokens=600)
    open_ended = _transform()
    make_loader(root, "train", DatasetProcessor(preset="structured"), transform=open_ended, batch_tokens=600,
                pad_multiple=64)
    assert open_ended.max_length == 576                                   # 600 rounded down to a multiple of 64


# ---- per-token loss weighting ------------------------------------------------------------------------

def _micro_loss(weight, tokens, targets):
    logits = tokens @ weight
    return torch.nn.functional.cross_entropy(logits, targets)            # mean over the micro-batch's tokens


def test_token_weighting_equals_one_big_batch_and_undoes_the_rank_average(monkeypatch):
    torch.manual_seed(0)
    sizes = [[3, 7, 2], [5, 1, 9]]                                       # target tokens of two ranks' windows
    data = [[(torch.randn(n, 4), torch.randint(0, 3, (n,))) for n in rank] for rank in sizes]
    counts = [[float(n) for n in rank] for rank in sizes]

    def grads(weight, rank):
        monkeypatch.setattr(dd, "all_reduce_sum", lambda t: torch.tensor(sum(map(sum, counts))))
        monkeypatch.setattr(dd, "world_size", lambda: 2)
        window = [{"labels": torch.cat([torch.zeros(n, 1, dtype=torch.long), targets.view(n, 1)], 1)}
                  for (x, targets), n in zip(data[rank], sizes[rank])]
        # labels[..., 1:] counts one target per row, so rows stand for the tokens here
        for batch, (x, t) in zip(window, data[rank]):
            batch["labels"] = torch.cat([torch.full((1, 1), -100), t.view(1, -1)], 1)
        weights = _window_weights(window, "token", torch.device("cpu"))
        total = sum(_micro_loss(weight, x, t) * weights[i] for i, (x, t) in enumerate(data[rank]))
        return torch.autograd.grad(total, weight)[0]

    w = torch.randn(4, 3, requires_grad=True)
    averaged = (grads(w, 0) + grads(w, 1)) / 2                           # what DDP's gradient average gives
    everything_x = torch.cat([x for rank in data for x, _ in rank])
    everything_t = torch.cat([t for rank in data for _, t in rank])
    big = torch.autograd.grad(_micro_loss(w, everything_x, everything_t), w)[0]
    assert torch.allclose(averaged, big, atol=1e-6)


def test_batch_weighting_is_the_plain_mean():
    window = [{"labels": torch.zeros(1, 4, dtype=torch.long)}] * 4
    weights = _window_weights(window, "batch", torch.device("cpu"))
    assert [float(weights[i]) for i in range(4)] == [0.25] * 4


# ---- Hugging Face push -------------------------------------------------------------------------------

class FakeApi:
    def __init__(self, fail=False):
        self.created, self.uploads, self.fail = [], [], fail

    def create_repo(self, **kw):
        self.created.append(kw)

    def upload_folder(self, folder_path, path_in_repo, repo_id, **kw):
        if self.fail:
            raise RuntimeError("network down")
        self.uploads.append((path_in_repo, sorted(p.name for p in Path(folder_path).iterdir()), kw["commit_message"]))
        future = Future()
        future.set_result("ok")
        return future


def _state():
    return {"model.base_model.model.layer.lora_A.default.weight": torch.ones(2, 3),
            "model.base_model.model.layer.lora_B.default.weight": torch.ones(3, 2),
            "model.head.weight": torch.ones(2)}


def test_adapter_keys_follow_the_peft_convention():
    keys = set(adapter_state(_state()))
    assert keys == {"base_model.model.layer.lora_A.weight", "base_model.model.layer.lora_B.weight"}


def test_push_stages_a_folder_and_final_goes_to_final_and_last(tmp_path):
    api = FakeApi()
    pusher = HubPusher("me/ocr", every_save=False, api=api, stage_dir=tmp_path / "stage", run_config={"hub_token": "x", "lr": 1})
    ckpt = tmp_path / "step.pt"
    ckpt.write_bytes(b"state")
    pusher.push(10, _state(), ckpt, final=False)
    assert api.uploads == []                                              # periodic pushes are off
    pusher.push(20, _state(), ckpt, final=True)
    pusher.close()
    targets = [u[0] for u in api.uploads]
    assert targets == ["final", "last"]
    files = api.uploads[0][1]
    assert {"trainable.safetensors", "training_state.pt", "run_config.json"} <= set(files)
    assert "adapter_model.safetensors" not in files                       # no LoRA config given
    assert api.created[0]["repo_id"] == "me/ocr" and api.created[0]["exist_ok"]


def test_every_save_pushes_last_and_a_failing_upload_never_raises(tmp_path):
    api = FakeApi()
    pusher = HubPusher("me/ocr", every_save=True, api=api, stage_dir=tmp_path / "s")
    pusher.push(5, _state(), None, final=False)
    assert [u[0] for u in api.uploads] == ["last"]
    broken = HubPusher("me/ocr", every_save=True, api=FakeApi(fail=True), stage_dir=tmp_path / "b")
    broken.push(5, _state(), None, final=False)                           # must not raise
    assert broken.failures == 1


def test_resume_paths_pass_through_unless_they_point_at_the_hub():
    assert resolve_resume("checkpoints/a.pt") == "checkpoints/a.pt" and resolve_resume(None) is None
    with pytest.raises(ValueError, match="hf://"):
        resolve_resume("hf://onlyone")


def test_lora_adapter_round_trips_through_peft(tmp_path):
    peft = pytest.importorskip("peft")
    from safetensors.torch import load_file

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

    class Wrapper(torch.nn.Module):          # like DeepSeekOCR: the PEFT model is the attribute `model`
        def __init__(self, model):
            super().__init__()
            self.model = model

    config = peft.LoraConfig(r=2, target_modules=["proj"])
    wrapper = Wrapper(peft.get_peft_model(Toy(), config))
    with torch.no_grad():                      # lora_B starts at zero; make every adapter tensor distinctive
        for name, param in wrapper.named_parameters():
            if "lora_" in name:
                param.copy_(torch.randn_like(param))
    state = {k: v for k, v in wrapper.state_dict().items() if "lora_" in k}
    pusher = HubPusher("me/ocr", every_save=True, api=FakeApi(), stage_dir=tmp_path / "s", create=False)
    pusher.stage_dir.mkdir()
    folder = pusher.stage_dir / "out"
    folder.mkdir()
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in adapter_state(state).items()}, str(folder / "adapter_model.safetensors"))
    config.save_pretrained(str(folder))
    loaded = peft.PeftModel.from_pretrained(Toy(), str(folder))
    assert loaded.peft_config["default"].r == 2
    for name, tensor in state.items():         # the weights really arrived, not just the keys
        assert torch.equal(loaded.state_dict()[name.removeprefix("model.")], tensor), name


# ---- the loop ------------------------------------------------------------------------------------------

class StubModel(torch.nn.Module):
    """A tiny causal LM with the surface train_sft needs (checkpoint saving, LoRA flag)."""
    lora_enabled = False
    quantization = None

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(128, 8)
        self.head = torch.nn.Linear(8, 128)

    def forward(self, input_ids, labels, attention_mask=None, **_):
        logits = self.head(self.embed(input_ids))
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 128), labels[:, 1:].reshape(-1),
                                                 ignore_index=-100)
        return type("Out", (), {"loss": loss})()

    def trainable_state_dict(self):
        return {k: v.detach().cpu() for k, v in self.named_parameters()}

    def save_checkpoint(self, path, scheduler=None, step=None, extra=None, model_state=None, optimizer_state=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model_state, "optimizer": optimizer_state, "step": step, "extra": extra}, path)
        return path


@pytest.mark.parametrize("weighting", ["token", "batch"])
def test_the_loop_trains_on_bucketed_batches_and_pushes_checkpoints(tmp_path, weighting):
    loader = _loader(_docs(tmp_path), batch_tokens=400, max_batch_size=4)
    model = StubModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    api = FakeApi()
    hub = HubPusher("me/ocr", every_save=True, api=api, stage_dir=tmp_path / "stage")
    cfg = SFTConfig(epochs=2, grad_accum=2, precision="fp32", log_every=1, save_every_steps=2, tensorboard=False,
                    out_dir=str(tmp_path / "out"), loss_weighting=weighting, hub=hub,
                    batch_adapter=lambda b: {k: b[k] for k in ("input_ids", "attention_mask", "labels")})
    expected = 0
    for epoch in range(2):                       # each epoch is planned again, so its batch count can differ
        set_epoch(loader, epoch)
        expected += -(-len(loader) // 2)
    set_epoch(loader, 0)
    result = train_sft(2, model, opt, loader, None, 2, cfg=cfg)
    assert result["step"] == expected and result["skipped"] == 0
    assert list((tmp_path / "out").glob("final_*.pt"))
    pushed = [u[0] for u in api.uploads]
    assert pushed[-2:] == ["final", "last"] and "last" in pushed[:-2]


def test_max_steps_stops_early_and_still_writes_the_final_checkpoint(tmp_path):
    loader = _loader(_docs(tmp_path), batch_tokens=300, max_batch_size=2)
    model = StubModel()
    cfg = SFTConfig(epochs=3, grad_accum=1, precision="fp32", tensorboard=False, out_dir=str(tmp_path / "o"),
                    max_steps=3, save_every_steps=0, loss_weighting="token",
                    batch_adapter=lambda b: {k: b[k] for k in ("input_ids", "attention_mask", "labels")})
    result = train_sft(3, model, torch.optim.SGD(model.parameters(), lr=0.01), loader, None, 0, cfg=cfg)
    assert result["step"] == 3
    assert list((tmp_path / "o").glob("final_0000003.pt"))


def test_validation_loss_is_logged_before_training_every_n_steps_and_each_epoch(tmp_path, caplog):
    import logging
    caplog.set_level(logging.INFO)
    root = _docs(tmp_path)
    loader = _loader(root, batch_tokens=400, max_batch_size=4)
    val_loader = make_loader(root, "val", DatasetProcessor(preset="structured"), transform=_transform(),
                             batch_tokens=400, max_batch_size=4)
    model = StubModel()
    cfg = SFTConfig(epochs=2, grad_accum=2, precision="fp32", log_every=100, save_every_steps=0, tensorboard=False,
                    out_dir=str(tmp_path / "o"), val_loader=val_loader, eval_every=3,
                    batch_adapter=lambda b: {k: b[k] for k in ("input_ids", "attention_mask", "labels")})
    result = train_sft(2, model, torch.optim.SGD(model.parameters(), lr=0.01), loader, None, 0, cfg=cfg)
    steps = [int(m.group(1)) for r in caplog.records if (m := __import__("re").search(r"validation loss [\d.]+ at step (\d+)", r.getMessage()))]
    assert steps[0] == 0                                          # the starting point
    assert len(steps) == len(set(steps)) and steps == sorted(steps)         # never twice for one step
    every_third = set(range(3, result["step"] + 1, 3))
    assert every_third <= set(steps)                              # every N steps ...
    assert result["step"] in steps                                # ... and the end of the last epoch


def test_validation_loss_weights_every_target_token_equally():
    from train.sft import evaluate

    class Fixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, labels, **_):                             # batch loss = 1.0 for 1 token, 3.0 for 3 tokens
            return type("O", (), {"loss": self.w.sum() + (1.0 if labels.shape[0] == 1 else 3.0)})()

    batches = [{"labels": torch.tensor([[0, 5]])}, {"labels": torch.tensor([[0, 5], [0, 5], [0, 5]])}]
    value = evaluate(Fixed(), batches, torch.device("cpu"), torch.float32)
    assert value == pytest.approx((1.0 * 1 + 3.0 * 3) / 4)          # not the plain mean of the two batch losses (2.0)
