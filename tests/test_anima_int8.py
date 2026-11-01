"""The Anima int8 base's surface: which file supplies the forward copies, the plan it
refuses, the LoRA stamps, and dX on block swap. The GEMMs: ``test_moss_int8.py``."""

from __future__ import annotations

import dataclasses
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from library import downloads
from library.anima.int8 import (
    stream_blocks_,
    ROUND_TO_NEAREST,
    check_plan,
    convert_dit_,
    convert_for_turbo,
    convert_targets,
    dequantize_checkpoint_,
    forward_file,
)
from library.anima.models import Anima
from library.anima.stream import stream_fp32_
from library.anima.trainer.trainer import AnimaTrainer
from library.anima.weights import is_int8_checkpoint
from library.int8.linear import _int8_flops, close_stream, quantize
from library.runtime.dynamo import _flash_flops
from library.runtime.harness import (
    _apply_activation_memory_budget,
    place_dit_for_training,
)
from scripts.distill_turbo.steps import selective_block_grad_ckpt


def args(**over):
    base = dict(
        int8="full",
        int8_file=None,
        blocks_to_swap=0,
        base_weights=None,
        pretrained_model_name_or_path="/nowhere/dit.safetensors",
    )
    return SimpleNamespace(**{**base, **over})


@pytest.mark.core
def test_forward_file_order(tmp_path):
    """Which file supplies the int8 forward copies; the wrong one renders wrong, silently."""
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


def tiny_dit(num_blocks: int = 6) -> Anima:
    torch.manual_seed(0)
    model = Anima(
        max_img_h=256,
        max_img_w=256,
        max_frames=4,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=256,
        num_blocks=num_blocks,
        num_heads=4,
        mlp_ratio=2.0,
        crossattn_emb_channels=64,
        use_adaln_lora=True,
        adaln_lora_dim=16,
        use_llm_adapter=False,
        attn_mode="torch",
    )
    for block in model.blocks:
        block.reset_parameters()
    for p in model.parameters():
        nn.init.normal_(p, std=0.05)
    return model.to(torch.bfloat16).train().requires_grad_(False)


def convert_tiny(model: Anima, mode: str) -> None:
    if mode == "stream":
        model.to("cuda")
    convert_dit_(model, "full" if mode == "stream" else mode)
    if mode == "stream":
        stream_blocks_(model)


def swapped_step(model: Anima):
    """One forward + backward from the swap start layout; ``(output, dX)``."""
    model.prepare_block_swap_before_forward()
    torch.manual_seed(1)
    x = torch.randn(1, 16, 1, 8, 8, device="cuda", dtype=torch.bfloat16)
    x.requires_grad_()
    t = torch.tensor([0.5], device="cuda")
    c = torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = model.forward_mini_train_dit(x, t, c)
    y.square().mean().backward()
    return y.detach(), x.grad


@pytest.mark.core
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a card")
@pytest.mark.parametrize(
    "order, mode, stream, swap",
    [
        ("train", "stream", "fp32", 3),
        ("turbo", "stream", "fp32", 4),
        ("turbo", "stream", "bf16", 3),
        ("train", "fwd", "bf16", 4),
        ("turbo", "fwd", "fp32", 3),
        ("train", "full", "fp32", 4),
        ("turbo", "full", "bf16", 3),
    ],
)
def test_swapped_step_is_the_resident_step(order, mode, stream, swap):
    """Both places the swapper attaches: train.py before the conversion, turbo after.
    The resident reference is the same model with the swapper off, so the int8 copies
    are the same bytes."""
    device = torch.device("cuda")
    model = tiny_dit()
    if order == "train":
        model.enable_block_swap(swap, device)
        convert_tiny(model, mode)
    else:
        place_dit_for_training(model, device, blocks_to_swap=swap)
        convert_for_turbo(model, mode, ROUND_TO_NEAREST, "/nowhere/dit.safetensors")
    if stream == "fp32":
        stream_fp32_(model)
    model.to(dtype=torch.bfloat16)
    model.move_to_device_except_swap_blocks(device)
    model.prepare_block_swap_before_forward()

    tail = model.blocks[-1].mlp.layer1
    assert tail.wq.device.type == "cpu" and tail.s.device.type == "cuda"
    steps = [swapped_step(model) for _ in range(3)]

    model.pause_block_swap()
    for handle in model.offloader.remove_handles:
        handle.remove()
    assert tail.wq.device.type == "cuda"
    y_ref, dx_ref = swapped_step(model)
    assert dx_ref.abs().sum() > 0
    for y, dx in steps:
        assert torch.equal(y, y_ref) and torch.equal(dx, dx_ref)
    close_stream()


def turbo_step(model: Anima, cdm: bool) -> list[torch.Tensor]:
    """A student forward whose graph outlives teacher and fake forwards, then under
    ``cdm`` a second student forward, checkpointed, backwarded first; every ``dX``."""

    def forward(x, no_grad=False):
        model.rewind_block_swap()
        grad = torch.no_grad() if no_grad else torch.enable_grad()
        with grad, torch.autocast("cuda", dtype=torch.bfloat16):
            return model.forward_mini_train_dit(x, t, c)

    torch.manual_seed(1)
    x = torch.randn(1, 16, 1, 8, 8, device="cuda", dtype=torch.bfloat16)
    x.requires_grad_()
    t = torch.tensor([0.5], device="cuda")
    c = torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16)
    y = forward(x)
    forward(x.detach(), no_grad=True)
    fake = forward(x.detach(), no_grad=True)
    leaves = [x]
    if cdm:
        x_off = (x.detach() + 0.1 * fake).requires_grad_()
        with selective_block_grad_ckpt(model):
            y_off = forward(x_off)
        forward(x_off.detach(), no_grad=True)
        y_off.square().mean().backward()
        leaves.append(x_off)
    y.square().mean().backward()
    return [leaf.grad for leaf in leaves]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a card")
@pytest.mark.parametrize("cdm", [False, True])
@pytest.mark.parametrize("mode", ["off", "fwd", "full", "stream"])
def test_turbo_step_on_swap_is_the_resident_step(mode, cdm):
    device = torch.device("cuda")
    model = tiny_dit()
    model.enable_block_swap(3, device)
    if mode != "off":
        convert_tiny(model, mode)
    model.to(dtype=torch.bfloat16)
    model.move_to_device_except_swap_blocks(device)
    model.prepare_block_swap_before_forward()
    got = turbo_step(model, cdm)
    assert model.offloader.moved == 0

    model.pause_block_swap()
    for handle in model.offloader.remove_handles:
        handle.remove()
    for dx, dx_ref in zip(got, turbo_step(model, cdm), strict=True):
        assert dx_ref.abs().sum() > 0 and torch.equal(dx, dx_ref)
    close_stream()


def gan_step(model: Anima, tap: int, ckpt: bool, between: bool) -> list[torch.Tensor]:
    """A tapped teacher forward on the student's output, backwarded with the student's
    graph (``between``: a full forward in the way), then the disc's tapped forwards
    around a fake update; every ``dX``."""

    def forward(x, no_grad=False, **tapped):
        model.rewind_block_swap()
        grad = torch.no_grad() if no_grad else torch.enable_grad()
        with grad, torch.autocast("cuda", dtype=torch.bfloat16):
            return model.forward_mini_train_dit(x, t, c, **tapped)

    tapped = dict(return_block_features={tap}, return_features_early=True)
    torch.manual_seed(1)
    x = torch.randn(1, 16, 1, 8, 8, device="cuda", dtype=torch.bfloat16)
    x.requires_grad_()
    t = torch.tensor([0.5], device="cuda")
    c = torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16)
    y = forward(x)
    forward(x.detach(), no_grad=True)
    with selective_block_grad_ckpt(model) if ckpt else nullcontext():
        feats = forward(0.9 * y, **tapped)
    if between:
        forward(x.detach(), no_grad=True)
    (y.square().mean() + feats[tap].square().mean()).backward()

    x_fake = y.detach().requires_grad_()
    forward(x_fake).square().mean().backward()
    disc_x = forward(x_fake.detach(), no_grad=True, **tapped)[tap]
    disc_real = forward(x.detach(), no_grad=True, **tapped)[tap]
    return [x.grad, x_fake.grad, disc_x, disc_real]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a card")
@pytest.mark.parametrize(
    "mode, tap, swap, ckpt, between",
    [
        ("off", 1, 4, True, False),
        ("stream", 1, 3, False, True),
        ("off", 2, 4, False, False),
        ("stream", 2, 3, True, True),
        ("off", 3, 3, True, True),
        ("stream", 3, 4, False, False),
        ("off", 4, 3, False, False),
        ("stream", 4, 4, True, True),
        ("off", 5, 3, False, True),
        ("stream", 5, 4, True, False),
    ],
)
def test_gan_step_on_swap_is_the_resident_step(mode, tap, swap, ckpt, between):
    device = torch.device("cuda")
    model = tiny_dit()
    model.enable_block_swap(swap, device)
    if mode != "off":
        convert_tiny(model, mode)
    model.to(dtype=torch.bfloat16)
    model.move_to_device_except_swap_blocks(device)
    model.prepare_block_swap_before_forward()
    got = gan_step(model, tap, ckpt, between)

    model.pause_block_swap()
    for handle in model.offloader.remove_handles:
        handle.remove()
    ref = gan_step(model, tap, ckpt, between)
    for a, b in zip(got, ref, strict=True):
        assert b.abs().sum() > 0 and torch.equal(a, b)
    close_stream()


def test_base_weights_need_round_to_nearest(tmp_path):
    (tmp_path / "dit_int8_gptq.safetensors").touch()
    dit = str(tmp_path / "dit.safetensors")
    plan = dict(base_weights=["x"], pretrained_model_name_or_path=dit)
    with pytest.raises(SystemExit, match="base_weights"):
        check_plan(args(**plan))
    check_plan(args(int8_file=ROUND_TO_NEAREST, **plan))


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


@pytest.mark.core
def test_lora_stamps():
    assert "ss_int8_base" not in stamp(int8="off")
    meta = stamp(int8="full", int8_base_id=ROUND_TO_NEAREST)
    assert (meta["ss_int8_base"], meta["ss_int8_base_id"]) == ("full", ROUND_TO_NEAREST)


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


def _write_int8_checkpoint(path, w):
    wq, s = quantize(w)
    save_file(
        {
            "net.blocks.0.mlp.layer1.weight": wq,
            "net.blocks.0.mlp.layer1.weight_scale": s,
            "net.blocks.0.mlp.layer1.comfy_quant": torch.zeros(4, dtype=torch.uint8),
        },
        str(path),
        metadata={"int8_modules": json.dumps(["blocks.0.mlp.layer1"])},
    )


@pytest.mark.core
def test_an_int8_checkpoint_is_its_own_forward_file(tmp_path):
    dit = tmp_path / "dit.safetensors"
    _write_int8_checkpoint(dit, torch.randn(8, 256))
    (tmp_path / "dit_int8_gptq.safetensors").touch()
    assert is_int8_checkpoint(str(dit))
    assert forward_file(None, str(dit)) == str(dit)
    assert forward_file(ROUND_TO_NEAREST, str(dit)) == ""


@pytest.mark.core
def test_dequantize_rebuilds_the_weight_and_drops_side_keys(tmp_path):
    dit = tmp_path / "dit.safetensors"
    w = torch.randn(8, 256)
    _write_int8_checkpoint(dit, w)
    key = "blocks.0.mlp.layer1"
    sd = {
        f"{key}.weight": torch.zeros(8, 256, dtype=torch.bfloat16),
        f"{key}.weight_scale": torch.zeros(8, 1),
        f"{key}.comfy_quant": torch.zeros(4),
    }
    dequantize_checkpoint_(sd, str(dit))
    assert list(sd) == [f"{key}.weight"]
    assert sd[f"{key}.weight"].dtype == torch.bfloat16
    assert (sd[f"{key}.weight"].float() - w).norm() / w.norm() < 0.03
