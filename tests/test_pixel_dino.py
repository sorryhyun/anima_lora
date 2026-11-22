"""The DINO feature term of pixel-head training (``DinoFeat``, ``X1Trainer``,
``X2Trainer``): the distance, where each trainer applies it, and the RNG streams a run
without the term shares. CPU only, on a randomly initialised DINOv3 ViT (no download)."""

import random

import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from library.anima.pixel_head import PATCH, PixelHead, PixelHeadX2, drop_x1_output
from library.anima.pixel_head_train import (
    DinoFeat,
    Window,
    X2TrainConfig,
    X2Trainer,
    pick_window,
)
from library.anima.pixel_head_train_x1 import X1TrainConfig, X1Trainer

D_FEAT = 32
GH = GW = 6  # 96-px images: one tile is the whole image


def _dino() -> DinoFeat:
    torch.manual_seed(1)
    cfg = DINOv3ViTConfig(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=64,
        patch_size=16,
        num_register_tokens=4,
        image_size=64,
    )
    return DinoFeat("cpu", layers=(1, 3), net=DINOv3ViTModel(cfg))


def _trainer(dino=None, **kw) -> X1Trainer:
    torch.manual_seed(0)
    head = PixelHead(D_FEAT, 64, width=16, depth=3, ls_init=1.0, use_skip=False, up=1)
    cfg = X1TrainConfig(rollouts=12, opt_steps=2, replay=4, warmup=1, **kw)
    return X1Trainer(head, cfg, dino=dino)


def _rollouts(n: int):
    g = torch.Generator().manual_seed(7)
    for r in range(n):
        f = torch.randn(GH, GW, D_FEAT, generator=g)
        lat = torch.randn(GH, GW, 64, generator=g)
        tgt = torch.rand(3, GH * PATCH, GW * PATCH, generator=g) * 2 - 1
        yield f, lat, tgt, r % 2 == 0  # the points in turn: every other one is :real


def test_distance_pairs_tiles_at_the_same_place():
    dn = _dino()
    g = torch.Generator().manual_seed(3)
    a = torch.rand(3, 320, 272, generator=g) * 2 - 1
    b = torch.rand(3, 320, 272, generator=g) * 2 - 1
    # the same image at the same places: nothing to compare
    assert dn.term(a, a, random.Random(0), n=3, compiled=False).item() < 1e-10
    assert dn.image(a, a) < 1e-10
    assert dn.image(a, b) > 1e-4
    # the gradient reaches the first image only inside the drawn tiles
    a.requires_grad_(True)
    rng = random.Random(5)
    dn.term(a, b, rng, n=1, compiled=False).backward()
    rng2 = random.Random(5)
    y, x = rng2.randint(0, 320 - 224), rng2.randint(0, 272 - 224)
    touched = a.grad.abs().sum(0) > 0
    assert touched[y : y + 224, x : x + 224].any()
    touched[y : y + 224, x : x + 224] = False
    assert not touched.any()


def test_convnext_distance_reads_the_stage_maps():
    from transformers import DINOv3ConvNextConfig, DINOv3ConvNextModel

    torch.manual_seed(2)
    cfg = DINOv3ConvNextConfig(hidden_sizes=[8, 16, 32, 64], depths=[1, 1, 2, 1])
    dn = DinoFeat("cpu", net=DINOv3ConvNextModel(cfg))
    assert dn.layers == (0, 1, 2, 3) and dn.patch == 32
    g = torch.Generator().manual_seed(3)
    a = torch.rand(3, 320, 272, generator=g) * 2 - 1
    b = a.clone()
    assert dn.image(a, b) < 1e-10
    # a change inside one 4-px cell moves the distance: the stride-4 stage sees it
    b[:, 100:104, 100:104] += 0.5
    assert dn.image(a, b) > 1e-6
    a.requires_grad_(True)
    dn.term(a, b, random.Random(0), n=2, compiled=False).backward()
    assert a.grad.abs().sum() > 0


def test_x1_term_on_real_samples_and_control_replay_draws():
    arm = _trainer(_dino(), lambda_dino=300.0)
    ctl = _trainer(None, lambda_dino=0.0)
    for tr in (arm, ctl):
        rolls = _rollouts(6)
        f, lat, tgt, _ = next(rolls)
        tr.push(f, lat, tgt, False)  # a point without :real: no DINO term
        tr.update()
        assert tr.dino_ema is None
        for f, lat, tgt, real in rolls:
            tr.push(f, lat, tgt, real)
            tr.update()
    assert arm.dino_ema is not None and ctl.dino_ema is None
    # the term's tiles have their own stream: the replay draws are the control's
    assert arm._replay_rng.getstate() == ctl._replay_rng.getstate()


def _x2_trainer(dino=None, **kw) -> X2Trainer:
    torch.manual_seed(0)
    base = PixelHead(D_FEAT, 64, width=16, depth=3, ls_init=1.0, use_skip=False, up=1)
    head = PixelHeadX2(base, depth2=1, width2=4)
    drop_x1_output(head)
    cfg = X2TrainConfig(
        rollouts=12, opt_steps=2, replay=4, warmup=1, window=4, margin=1,
        pegan_ratio=0.0, dc_block=8, **kw,
    )  # fmt: skip
    return X2Trainer(head, cfg, dino=dino)


def _x2_rollouts(n: int):
    g = torch.Generator().manual_seed(7)
    wr = random.Random(4)
    for _ in range(n):
        f = torch.randn(8, 8, D_FEAT, generator=g)
        lat = torch.randn(8, 8, 64, generator=g)
        tgt = torch.rand(3, 16 * PATCH, 16 * PATCH, generator=g) * 2 - 1
        win = pick_window(8, 8, 4, 1, wr)
        assert isinstance(win, Window)
        yield f, lat, tgt, win


def test_x2_term_and_control_replay_draws():
    arm = _x2_trainer(_dino(), lambda_dino=245.2)
    ctl = _x2_trainer(None, lambda_dino=0.0)
    logs = []
    for tr in (arm, ctl):
        for r, (f, lat, tgt, win) in enumerate(_x2_rollouts(4)):
            tr.push(f, lat, tgt, win)
            logs.append(tr.update(r))
    assert all("dino" in log for log in logs[:4])
    assert not any("dino" in log for log in logs[4:])
    assert arm._replay_rng.getstate() == ctl._replay_rng.getstate()
