"""YAML training configs: precedence, `extends`, typo detection and the files shipped in config/."""
import pytest
import yaml

from train.config import CONFIG_DIR, SECTIONS, load_config
from train.train import parse_args


def _write(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return str(path)


BASE = ["--root", "data", "--model-id", "m"]


def test_the_shipped_configs_load_and_only_use_known_settings():
    for name in ("train", "train_probe", "train_qlora", "train_full"):
        flat = load_config(name)
        assert flat["model_id"] == "deepseek-community/DeepSeek-OCR-2" and flat["root"]
        args = parse_args(["--config", name])                   # no flags needed: the file is complete
        assert args.batch_tokens == 3200 and args.max_length == 3200
    assert load_config("train_qlora")["quant"] == "4bit" and load_config("train_full")["lora"] is False
    assert load_config("train_probe")["max_steps"] == 8 and load_config("train_probe")["grad_accum"] == 1
    assert (CONFIG_DIR / "train.yaml").is_file()


def test_every_yaml_key_is_a_real_flag():
    args = parse_args(BASE)
    for section in SECTIONS.values():
        for dest in section.values():
            assert hasattr(args, dest), dest


def test_flags_beat_the_file_and_the_file_beats_the_defaults(tmp_path):
    cfg = _write(tmp_path, "run.yaml", {"training": {"epochs": 5, "grad_accum": 3}, "optimizer": {"lr": "2e-5"}})
    args = parse_args(["--config", cfg, *BASE, "--epochs", "2"])
    assert args.epochs == 2                                     # flag
    assert args.grad_accum == 3                                 # file
    assert args.batch_size == 2                                 # built-in default
    assert args.lr == pytest.approx(2e-5)                       # "2e-5" is a string in YAML; it is converted


def test_a_file_value_can_be_turned_off_from_the_command_line(tmp_path):
    cfg = _write(tmp_path, "run.yaml", {"lora": {"enabled": True}, "checkpoints": {"tensorboard": True}})
    on = parse_args(["--config", cfg, *BASE])
    off = parse_args(["--config", cfg, *BASE, "--no-lora", "--no-tensorboard"])
    assert on.lora and on.tensorboard and not off.lora and not off.tensorboard


def test_extends_merges_section_by_section_and_detects_loops(tmp_path):
    _write(tmp_path, "base.yaml", {"model": {"id": "m", "ocr_mode": "small"}, "data": {"root": "d"},
                                   "training": {"epochs": 4, "grad_accum": 2}})
    child = _write(tmp_path, "child.yaml", {"extends": "base.yaml", "training": {"epochs": 1}})
    flat = load_config(child)
    assert flat["epochs"] == 1 and flat["grad_accum"] == 2 and flat["ocr_mode"] == "small" and flat["root"] == "d"
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="loops"):
        load_config(tmp_path / "a.yaml")


def test_typos_are_errors_with_a_suggestion(tmp_path):
    with pytest.raises(ValueError, match="Did you mean 'batching'"):
        load_config(_write(tmp_path, "a.yaml", {"batchin": {"batch_tokens": 1}}))
    with pytest.raises(ValueError, match="Did you mean 'grad_accum'"):
        load_config(_write(tmp_path, "b.yaml", {"training": {"grad_acum": 4}}))
    with pytest.raises(FileNotFoundError):
        load_config("no_such_config")


def test_bad_choices_and_missing_required_settings_stop_with_a_message(tmp_path, capsys):
    cfg = _write(tmp_path, "run.yaml", {"training": {"strategy": "ddp2"}})
    with pytest.raises(SystemExit):
        parse_args(["--config", cfg, *BASE])
    assert "strategy" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        parse_args([])
    assert "--root is required" in capsys.readouterr().err


def test_lora_targets_may_be_a_list_in_the_file(tmp_path):
    cfg = _write(tmp_path, "run.yaml", {"lora": {"target_modules": ["q_proj", "v_proj"]}})
    assert parse_args(["--config", cfg, *BASE]).lora_targets == ["q_proj", "v_proj"]


def test_the_resolved_config_is_written_without_the_token(tmp_path):
    from train.train import save_resolved
    args = parse_args([*BASE, "--out-dir", str(tmp_path / "o"), "--hub-token", "secret"])
    save_resolved(args)
    text = (tmp_path / "o" / "resolved_config.yaml").read_text()
    assert "model_id: m" in text and "secret" not in text


def test_shipped_defaults_follow_the_documented_recipe():
    base, qlora, full = (load_config(n) for n in ("train", "train_qlora", "train_full"))
    assert base["optimizer"] == "adamw" and base["lora"] and base["lr"] == 1e-4          # adapters: AdamW, LoRA rate
    assert qlora["lr"] == 2e-4 and qlora["quant"] == "4bit" and qlora["lora_dropout"] == 0.1
    assert full["lora"] is False and full["lr"] == base["lr"] / 10 and full["strategy"] == "fsdp2"
    assert base["scheduler"] == "cosine" and 0 < base["warmup_ratio"] < 0.2 and base["min_lr_ratio"] < 1
    assert base["max_length"] <= base["batch_tokens"]                                    # one sample fits a batch


def test_the_scheduler_warms_up_decays_and_moves_every_param_group():
    import torch
    from train.optim import build_optimizer, build_scheduler

    layer = torch.nn.Linear(4, 4)
    named = [("layer.weight", layer.weight), ("layer.bias", layer.bias)]
    opt = build_optimizer(named, muon_lr=0.02, adamw_lr=1e-3)         # one Muon and one AdamW group
    sched = build_scheduler(opt, total_steps=100, kind="cosine", warmup_ratio=0.1, min_lr_ratio=0.1)
    lrs = []
    for _ in range(100):
        lrs.append([g["lr"] for g in opt.param_groups])
        opt.step()
        sched.step()
    muon, adamw = zip(*lrs)
    assert muon[0] < muon[5] < muon[9] and muon[9] == pytest.approx(0.02 * 1.0, rel=0.2)   # warmup
    assert max(adamw) == pytest.approx(1e-3, rel=0.05) and min(adamw) >= 1e-4 * 0.99        # floor 10% of the peak
    assert adamw[-1] < adamw[50] < adamw[10]                                                # decays
    assert all(abs(m / 0.02 - a / 1e-3) < 1e-9 for m, a in lrs)                            # same curve for both
    flat_opt = build_optimizer(named, muon_lr=0.02, adamw_lr=1e-3)
    flat = build_scheduler(flat_opt, 10, kind="constant", warmup_ratio=0)
    for _ in range(5):
        flat_opt.step()
        flat.step()
    assert flat.get_last_lr() == [0.02, 1e-3]                          # constant without warmup never moves


def test_a_frozen_base_loads_in_the_compute_dtype_and_a_trained_one_in_fp32():
    import torch
    from train.train import weights_dtype

    half = torch.float16
    assert weights_dtype(parse_args([*BASE, "--lora"]), half) == half                   # LoRA: frozen base
    assert weights_dtype(parse_args([*BASE, "--lora", "--quant", "4bit"]), half) == half
    assert weights_dtype(parse_args([*BASE, "--no-lora"]), half) == torch.float32       # full fine-tuning
    assert weights_dtype(parse_args([*BASE, "--lora", "--train-vision"]), half) == torch.float32


def test_peft_keeps_adapters_in_fp32_on_a_half_precision_base():
    peft = pytest.importorskip("peft")
    import torch

    base = torch.nn.Sequential(torch.nn.Linear(8, 8)).half()
    model = peft.get_peft_model(base, peft.LoraConfig(r=2, target_modules=["0"]))
    dtypes = {p.dtype for n, p in model.named_parameters() if "lora_" in n}
    assert dtypes == {torch.float32}                              # the fp16 loss scaler needs fp32 trainable params


def test_the_large_gpu_config_loads_and_keeps_both_training_settings():
    flat = load_config("train_large_gpu")
    assert flat["precision"] == "bf16" and flat["batch_tokens"] == 16000 and flat["eval_every"] == 100
    assert flat["grad_accum"] == 1 and flat["val_max_documents"] is None and flat["workers"] == 4
    assert load_config("train")["eval"] is True and load_config("train_probe")["eval"] is False


def test_init_from_starts_a_new_run_with_a_new_lora_and_conflicts_are_refused(capsys):
    args = parse_args(BASE + ["--lora", "--lora-r", "8", "--init-from", "old.pt", "--init-lora-r", "16"])
    assert args.init_from == "old.pt" and args.lora_r == 8 and args.init_lora_r == 16
    for bad in (["--init-from", "old.pt"],                                   # no --lora
                ["--lora", "--init-from", "old.pt", "--resume", "run.pt"],   # a new run and an old one
                ["--lora", "--init-from", "old.pt", "--quant", "4bit"]):     # cannot merge into a quantized base
        with pytest.raises(SystemExit):
            parse_args(BASE + bad)
    assert "--init-from" in capsys.readouterr().err


def test_checkpoint_folder_defaults_keep_every_save_and_name_them_by_kind():
    args = parse_args(BASE)
    assert args.hub_every_save is True and args.hub_keep_last is False and args.hub_state is True
    assert args.state_prefix is None and args.reports is True and args.gen_eval_at_save is True
    assert parse_args(BASE + ["--state-prefix", "", "--no-hub-state", "--hub-keep-last"]).state_prefix == ""
