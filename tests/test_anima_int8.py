"""The Anima int8 base's surface: which file supplies the forward copies, the plans it
refuses, the LoRA stamps and the flags. The GEMMs themselves: ``test_moss_int8.py``."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from library import downloads
from library.anima.int8 import (
    CONVERT_SUFFIXES,
    stream_blocks_,
    ROUND_TO_NEAREST,
    check_plan,
    ckpt_keys,
    convert_for_training,
    convert_targets,
    forward_file,
)
from library.anima.trainer.cli import setup_parser
from library.anima.trainer.trainer import AnimaTrainer
from library.int8.linear import _int8_flops
from library.runtime.dynamo import _flash_flops
from library.runtime.harness import _apply_activation_memory_budget


def args(**over):
    base = dict(
        int8="full",
        int8_file=None,
        blocks_to_swap=0,
        base_weights=None,
        pretrained_model_name_or_path="/nowhere/dit.safetensors",
    )
    return SimpleNamespace(**{**base, **over})


def test_forward_file_order(tmp_path):
    dit = tmp_path / "dit.safetensors"
    assert forward_file(None, str(dit)) == ""
    beside = tmp_path / "dit_int8_gptq.safetensors"
    beside.touch()
    assert forward_file(None, str(dit)) == str(beside)
    assert forward_file(ROUND_TO_NEAREST, str(dit)) == ""
    other = tmp_path / "other.safetensors"
    assert forward_file(str(other), str(dit)) == str(other)


def test_catalog_dit_fetches_its_forward_file(monkeypatch, tmp_path):
    fetched = []
    target = tmp_path / downloads.ANIMA_DIT_INT8_FILE
    asset = dataclasses.replace(downloads.by_id()["anima_dit_int8"], dest=tmp_path)
    monkeypatch.setattr(downloads, "by_id", lambda: {"anima_dit_int8": asset})
    monkeypatch.setattr(downloads, "fetch", lambda a: fetched.append(a))

    dit = str(tmp_path / downloads.ANIMA_DIT_FILE)
    assert forward_file(None, dit) == str(target) and fetched == [asset]
    assert forward_file(ROUND_TO_NEAREST, dit) == "" and len(fetched) == 1
    target.touch()
    assert forward_file(None, dit) == str(target) and len(fetched) == 1


def test_check_plan_refuses_swap():
    check_plan(args())
    with pytest.raises(SystemExit, match="blocks_to_swap"):
        check_plan(args(blocks_to_swap=20))


def test_base_weights_need_round_to_nearest(tmp_path):
    (tmp_path / "dit_int8_gptq.safetensors").touch()
    dit = str(tmp_path / "dit.safetensors")
    plan = dict(base_weights=["x"], pretrained_model_name_or_path=dit)
    with pytest.raises(SystemExit, match="base_weights"):
        check_plan(args(**plan))
    check_plan(args(int8_file=ROUND_TO_NEAREST, **plan))


def test_only_lora_networks_train_on_int8():
    with pytest.raises(SystemExit, match="not a LoRANetwork"):
        convert_for_training(nn.Identity(), object(), args())


def test_targets_are_the_block_linears():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.layer1 = nn.Linear(4, 4, bias=False)
            self.adaln = nn.Linear(4, 4, bias=False)

    dit = nn.Module()
    dit.blocks = nn.ModuleList([Block()])
    dit.layer1 = nn.Linear(4, 4, bias=False)
    assert [n for n, _ in convert_targets(dit)] == ["blocks.0.mlp.layer1"]


def test_qkv_splits_back_to_checkpoint_keys():
    keys = ckpt_keys("blocks.3.self_attn.qkv_proj")
    assert len(keys) == 3 and all(k.startswith("net.blocks.3.") for k in keys)
    assert ckpt_keys("blocks.3.mlp.layer1") == ["net.blocks.3.mlp.layer1.weight"]


def stamp(**over):
    meta = {}
    fields = dict(
        weighting_scheme="none",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
        timestep_sampling="sigma",
        sigmoid_scale=1.0,
        discrete_flow_shift=1.0,
    )
    AnimaTrainer._update_metadata(meta, SimpleNamespace(**fields, **over))
    return meta


def test_lora_stamps():
    assert "ss_int8_base" not in stamp(int8="off")
    meta = stamp(int8="full", int8_base_id=ROUND_TO_NEAREST)
    assert (meta["ss_int8_base"], meta["ss_int8_base_id"]) == ("full", ROUND_TO_NEAREST)


def test_flags_default_off():
    parsed = setup_parser().parse_args([])
    assert (parsed.int8, parsed.int8_file) == ("off", None)
    with pytest.raises(SystemExit):
        setup_parser().parse_args(["--int8", "half"])


def test_stream_groups_each_block_by_suffix(monkeypatch):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            for path in CONVERT_SUFFIXES:
                head, tail = path.split(".")
                if not hasattr(self, head):
                    setattr(self, head, nn.Module())
                setattr(getattr(self, head), tail, nn.Linear(4, 4, bias=False))

    dit = nn.Module()
    dit.blocks = nn.ModuleList([Block(), Block()])
    seen = []
    monkeypatch.setattr("library.anima.int8.stream_compacted_", seen.extend)
    stream_blocks_(dit)
    assert [len(b) for b in seen] == [len(CONVERT_SUFFIXES)] * 2
    assert seen[1][0] is dit.blocks[1].self_attn.qkv_proj


def test_stream_flag_parses():
    assert setup_parser().parse_args(["--int8", "stream"]).int8 == "stream"


def test_int8_flops_count_tokens_on_fake_5d():
    wq = torch.empty(2048, 2048, dtype=torch.int8)
    flat = _int8_flops(torch.empty(1, 4096, 2048), wq)
    assert _int8_flops(torch.empty(1, 1, 4096, 1, 2048), wq) == flat
    assert flat == int(0.5 * 4096 * 2048 * 2048)


def test_fa2_flops_read_the_key_length():
    q, k = (1, 4096, 16, 128), (1, 512, 16, 128)
    assert _flash_flops(q, k) == 2 * 16 * 128 * 4096 * 512


@pytest.mark.parametrize("budget, ckpt, on", [(0.4, False, True), (0.4, True, False)])
def test_budget_joins_custom_ops_to_the_knapsack(monkeypatch, budget, ckpt, on):
    import torch._functorch.config as functorch_config

    monkeypatch.setattr(functorch_config, "activation_memory_budget", 1.0)
    monkeypatch.setattr(functorch_config, "is_non_builtin_to_include", False)
    _apply_activation_memory_budget(budget, grad_ckpt=ckpt)
    assert functorch_config.is_non_builtin_to_include is on
    assert functorch_config.activation_memory_budget == (budget if on else 1.0)
