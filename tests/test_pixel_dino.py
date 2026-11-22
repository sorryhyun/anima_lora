"""The DINO feature term of pixel-head training (``DinoFeat``, ``X1Trainer``,
``X2Trainer``): the distance, its weight rule and the RNG streams an LPIPS control
shares. CPU only, on a randomly initialised DINOv3 ViT (no download)."""

import math
import random

import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from library.anima.pixel_head import PATCH, PixelHead, PixelHeadX2, drop_x1_output
from library.anima.pixel_head_train import (
    DinoFeat,
    Window,
    X2TrainConfig,
    X2Trainer,
    _grad_norm,
    pick_window,
    rec_loss,
)
from library.anima.pixel_head_train_x1 import DINO_MATCH_N, X1TrainConfig, X1Trainer

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


class _StubLpips:
    def term(self, a, b, compiled=True):
        return (a - b).abs().mean()


def _trainer(dino=None, **kw) -> X1Trainer:
    torch.manual_seed(0)
    head = PixelHead(D_FEAT, 64, width=16, depth=3, ls_init=1.0, use_skip=False, up=1)
    cfg = X1TrainConfig(rollouts=12, opt_steps=2, replay=4, warmup=1, **kw)
    return X1Trainer(head, cfg, lpips=_StubLpips(), dino=dino)


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


def test_weight_rule_matches_lpips_gradient_norm():
    tr = _trainer(_dino(), lambda_lpips=0.0, dino_match_lpips=0.05)
    assert tr.lam_dino == 0.0
    lam_at = []
    for f, lat, tgt, real in _rollouts(2 * DINO_MATCH_N + 4):
        tr.push(f, lat, tgt, real)
        lam_at.append(tr.lam_dino)
        tr.update()
    log = tr.dino_log
    assert log["match_n"] == DINO_MATCH_N
    # λ · mean‖∇ DINO‖ = 0.05 · mean‖∇ LPIPS‖ over the samples the rule read
    assert math.isclose(
        log["lam_dino"] * log["g_dino"], 0.05 * log["g_lpips"], rel_tol=1e-6
    )
    # set from the first :real rollout on, fixed once the rule has its samples
    assert lam_at[0] > 0
    assert lam_at[2 * DINO_MATCH_N - 2] == lam_at[-1] == tr.lam_dino
    assert tr.dino_ema is not None and tr.lpips_ema is None

    # the rule's norms are the eager gradient norms of the unweighted terms
    tr2 = _trainer(_dino(), lambda_lpips=0.0, dino_match_lpips=0.05)
    f, lat, tgt, _ = next(_rollouts(1))
    tr2.push(f, lat, tgt, True)
    params = [q for q in tr2.head.parameters() if q.requires_grad]
    pred = tr2._predict(f.to(torch.bfloat16), lat, compiled=False)
    gs = torch.autograd.grad(
        rec_loss(pred, tgt.half().float())[0], params, allow_unused=True
    )
    assert math.isclose(tr2.dino_log["g_rec"], float(_grad_norm(gs)), rel_tol=1e-4)


def test_dino_arm_draws_the_control_replay_samples():
    arm = _trainer(_dino(), lambda_lpips=0.0, dino_match_lpips=0.05)
    ctl = _trainer(None, lambda_lpips=0.05)
    for tr in (arm, ctl):
        for f, lat, tgt, real in _rollouts(6):
            tr.push(f, lat, tgt, real)
            tr.update()
    assert arm._replay_rng.getstate() == ctl._replay_rng.getstate()
    assert ctl.lam_dino == 0.0 and ctl.dino_ema is None and not ctl.dino_log


def _x2_trainer(dino=None, **kw) -> X2Trainer:
    torch.manual_seed(0)
    base = PixelHead(D_FEAT, 64, width=16, depth=3, ls_init=1.0, use_skip=False, up=1)
    head = PixelHeadX2(base, depth2=1, width2=4)
    drop_x1_output(head)
    cfg = X2TrainConfig(
        rollouts=12, opt_steps=2, replay=4, warmup=1, window=4, margin=1,
        pegan_ratio=0.0, dc_block=8, **kw,
    )  # fmt: skip
    return X2Trainer(head, cfg, lpips=_StubLpips(), dino=dino)


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


def test_x2_weight_rule_and_control_replay_draws():
    arm = _x2_trainer(_dino(), lambda_lpips=0.0, dino_match_lpips=0.1)
    ctl = _x2_trainer(None, lambda_lpips=0.1)
    logs = []
    for tr in (arm, ctl):
        for r, (f, lat, tgt, win) in enumerate(_x2_rollouts(DINO_MATCH_N + 3)):
            tr.push(f, lat, tgt, win)
            logs.append(tr.update(r))
    log = arm.dino_log
    assert log["match_n"] == DINO_MATCH_N
    assert math.isclose(
        log["lam_dino"] * log["g_dino"], 0.1 * log["g_lpips"], rel_tol=1e-6
    )
    assert log["g_rec2"] > 0 and log["g_dc"] > 0
    # the arm trains on the DINO term and not on LPIPS; the control the reverse
    assert "dino" in logs[0] and "lpips" not in logs[0]
    assert "lpips" in logs[-1] and "dino" not in logs[-1]
    assert arm._replay_rng.getstate() == ctl._replay_rng.getstate()
    assert ctl.lam_dino == 0.0 and not ctl.dino_log
