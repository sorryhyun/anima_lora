"""Invariants for turbo DMAD Phase 0 (docs/proposal/turbo_dmad.md § Phase 0).

* config — ``[dmad]`` default-off, TOML/CLI precedence, every guard.
* ``normalize_signal_rms`` — per-sample RMS == ``signal_rms``, direction kept,
  a zero gradient stays zero.
* ``DmadDisc`` on a toy CPU backbone (``ctx.forward`` stubbed): the disc loss
  is BCE_T + BCE_R, the window accumulates n pairs into one optimizer step, a
  head with λ = 0 adds no branch, and the student signal is the normalized
  ``−(λ_T ∇h_T + λ_R ∇h_R)`` with the disc left untouched.
* ``TurboDMDNetwork(build_fake=False)`` — no critic stack is built.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from networks.methods.turbo_dmd import TurboDMDNetwork, gan_loss_discriminator
from scripts.distill_turbo.config import build_argparser, resolve_config
from scripts.distill_turbo.dmad import (
    _GEN_SEED_OFFSET,
    DmadDisc,
    disc_view,
    gen_rand_tau,
    gen_randn_like,
    normalize_signal_rms,
)
from scripts.distill_turbo.metrics import TurboMetrics
from scripts.distill_turbo.primitives import renoise

ON = {"dmad": {"enabled": True}}


def _resolve(cli: list[str] | None = None, cfg: dict | None = None):
    args = build_argparser().parse_args(cli or [])
    return resolve_config(args, cfg or {})


# --- config -------------------------------------------------------------------


def test_off_by_default():
    c = _resolve()
    assert c.dmad is False
    assert (c.dmad_lambda_t, c.dmad_lambda_r) == (1.0, 1.0)
    assert c.dmad_signal_rms == pytest.approx(0.18)
    assert c.dmad_lr == pytest.approx(4e-5)
    assert c.dmad_grad_clip == 0.0
    assert c.dmad_window == 4
    assert c.dmad_disc_warmup_steps == 50
    assert c.dmad_feature_block_idx == -1


def test_toml_and_cli_precedence():
    toml = {
        "dmad": {
            "enabled": True,
            "lambda_r": 0.5,
            "signal_rms": 0.09,
            "window": 2,
            "feature_block_idx": 5,
        }
    }
    c = _resolve(cfg=toml)
    assert c.dmad is True
    assert c.dmad_lambda_r == pytest.approx(0.5)
    assert c.dmad_signal_rms == pytest.approx(0.09)
    assert c.dmad_window == 2 and c.dmad_feature_block_idx == 5
    c = _resolve(
        [
            "--dmad_lambda_r",
            "0",
            "--dmad_window",
            "8",
            "--dmad_lr",
            "1e-4",
            "--dmad_grad_clip",
            "5",
            "--dmad_disc_warmup_steps",
            "0",
            "--dmad_feature_block_idx",
            "-1",
        ],
        cfg=toml,
    )
    assert c.dmad_lambda_r == 0.0 and c.dmad_window == 8
    assert c.dmad_lr == pytest.approx(1e-4) and c.dmad_grad_clip == 5.0
    assert c.dmad_disc_warmup_steps == 0
    assert c.dmad_feature_block_idx == -1  # -1 is a value here, not "unset"
    assert _resolve(["--dmad"]).dmad is True


@pytest.mark.parametrize(
    ("cli", "cfg", "match"),
    [
        (["--base_loss", "dmd", "--student_steps", "4"], {}, "dpdmd"),
        ([], {"gan": {"weight_gen": 0.03}}, "weight_gen"),
        ([], {"cdm": {"weight": 1.0}}, "cdm.weight"),
        (["--f_div", "kl"], {}, "f_distill"),
        (["--dmad_probe"], {}, "dmad_probe"),
        (["--fake_tau_banks", "2"], {}, "fake_tau_banks"),
        (["--blocks_to_swap", "4"], {}, "blocks_to_swap"),
        (["--grad_ckpt"], {}, "grad_ckpt"),
        (["--resume", "auto"], {}, "resume"),
        (["--dmad_lambda_t", "0", "--dmad_lambda_r", "0"], {}, "both 0"),
        (["--dmad_lambda_t", "-0.5"], {}, "lambda_t"),  # -1 is the CLI sentinel
        (["--dmad_signal_rms", "0"], {}, "signal_rms"),
        (["--dmad_grad_clip=-1e-3"], {}, "grad_clip"),
        (["--dmad_window", "0"], {}, "window"),
    ],
)
def test_guards(cli, cfg, match):
    with pytest.raises(ValueError, match=match):
        _resolve(cli, cfg={**ON, **cfg})


def test_guards_silent_when_off():
    c = _resolve(cfg={"gan": {"weight_gen": 0.03}, "cdm": {"weight": 1.0}})
    assert c.dmad is False


# --- normalization --------------------------------------------------------------


def test_normalization_rms_direction_zero():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 4, 6, 6, generator=g)
    x[0] *= 1e-7
    x[1] *= 1e3
    x[2] = 0.0
    out = normalize_signal_rms(x, 0.18)
    rms = out.pow(2).mean(dim=(1, 2, 3)).sqrt()
    assert rms[0] == pytest.approx(0.18, rel=1e-5)
    assert rms[1] == pytest.approx(0.18, rel=1e-5)
    assert torch.equal(out[2], torch.zeros_like(out[2]))
    for i in (0, 1):
        cos = torch.nn.functional.cosine_similarity(
            out[i].flatten(), x[i].flatten(), dim=0
        )
        assert cos == pytest.approx(1.0, abs=1e-6)


# --- toy disc -------------------------------------------------------------------

C, D, B = 4, 8, 2


class _Stack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(C, D)
        self.enabled = False

    def set_enabled(self, flag: bool) -> None:
        self.enabled = flag


class _Turbo:
    def __init__(self) -> None:
        self.stack = _Stack()

    def make_aux_stack(self):
        return self.stack

    def set_view(self, view: str) -> None:
        self.view = view


class _Ctx:
    """``ctx.forward`` stub: the stack's Linear over the latent's tokens."""

    def __init__(self) -> None:
        self.turbo = _Turbo()
        self.model = SimpleNamespace(
            num_blocks=4,
            model_channels=D,
            blocks=[
                SimpleNamespace(
                    gradient_checkpointing=False, unsloth_offload_checkpointing=False
                )
            ],
        )
        self.n_forward = 0

    def forward(
        self, view, x, t, c, *, no_grad, return_block_features, return_features_early
    ):
        assert view == "teacher" and return_features_early
        assert self.turbo.stack.enabled
        self.n_forward += 1
        (tap,) = return_block_features
        with torch.no_grad() if no_grad else torch.enable_grad():
            tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
            f = torch.tanh(self.turbo.stack.proj(tokens)) * (1.0 + t.view(-1, 1, 1))
            f = f + c.mean(dim=(1, 2)).view(-1, 1, 1)
        return {tap: f}


def _cfg(**over):
    base = dict(
        dmad_lambda_t=1.0,
        dmad_lambda_r=1.0,
        dmad_signal_rms=0.18,
        dmad_grad_clip=0.0,
        dmad_feature_block_idx=-1,
        dmad_lr=1e-3,
        dmad_window=4,
        seed=0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _disc(**over):
    torch.manual_seed(0)
    ctx = _Ctx()
    disc = DmadDisc(
        _cfg(**over),
        turbo=ctx.turbo,
        model=ctx.model,
        device="cpu",
        dtype=torch.float32,
    )
    return ctx, disc


def _pair(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "x_teacher": torch.randn(B, C, 3, 3, generator=g),
        "x_student": torch.randn(B, C, 3, 3, generator=g),
        "x_real": torch.randn(B, C, 3, 3, generator=g),
        "crossattn_emb": torch.randn(B, 5, 6, generator=g),
    }


def test_disc_loss_is_sum_of_balanced_bces():
    ctx, disc = _disc()
    assert disc.tap == 2
    p = _pair(1)
    stats = disc.accumulate(ctx, **p)
    got = {id(q): q.grad.clone() for q in disc.params}

    gen = torch.Generator().manual_seed(_GEN_SEED_OFFSET)
    tau = gen_rand_tau(gen, B, "cpu", torch.float32)
    eps = gen_randn_like(gen, p["x_student"])
    disc.opt.zero_grad(set_to_none=True)

    def h(head, x):
        f = ctx.forward(
            "teacher",
            renoise(x, tau, eps),
            tau,
            p["crossattn_emb"],
            no_grad=False,
            return_block_features={disc.tap},
            return_features_early=True,
        )[disc.tap]
        return head([f]).mean(dim=1, keepdim=True)

    with disc_view(ctx.turbo, disc.stack):
        loss = gan_loss_discriminator(
            h(disc.head_t, p["x_teacher"]), h(disc.head_t, p["x_student"])
        ) + gan_loss_discriminator(
            h(disc.head_r, p["x_real"]), h(disc.head_r, p["x_student"])
        )
    loss.backward()
    assert float(stats["loss"]) == pytest.approx(float(loss.detach()), rel=1e-5)
    for q in disc.params:
        torch.testing.assert_close(got[id(q)], q.grad)
    assert float(stats["bce_t"]) + float(stats["bce_r"]) == pytest.approx(
        float(loss), rel=1e-5
    )
    assert set(stats) >= {"rank_acc_t", "rank_acc_r", "margin_t", "gap_r"}


def test_window_accumulates_into_one_step():
    ctx, disc = _disc(dmad_window=3)
    n_steps = 0
    real_step = disc.opt.step

    def counting_step(*a, **k):
        nonlocal n_steps
        n_steps += 1
        return real_step(*a, **k)

    disc.opt.step = counting_step
    for i in range(5):
        ctx.n_forward = 0
        stats = disc.update(ctx, **_pair(i))
        n = min(i + 1, 3)
        assert float(stats["n_pairs"]) == n
        assert ctx.n_forward == 3 * n  # teacher + real + one student pass per pair
        assert n_steps == i + 1
    assert all(q.grad is None for q in disc.params)


def test_zero_lambda_drops_the_branch():
    ctx, disc = _disc(dmad_lambda_r=0.0)
    assert disc.head_r is None
    p = _pair(0)
    p["x_real"] = None
    disc.update(ctx, **p)
    ctx.n_forward = 0
    stats = disc.update(ctx, **_pair(1) | {"x_real": None})
    assert ctx.n_forward == 2 * 2
    assert not any(k.endswith("_r") for k in stats)

    ctx, disc = _disc(dmad_lambda_t=0.0)
    assert disc.head_t is None
    stats = disc.update(ctx, **_pair(0) | {"x_teacher": None})
    assert ctx.n_forward == 2
    assert "rank_acc_r" in stats and "rank_acc_t" not in stats


def test_student_signal_is_normalized_head_mix():
    ctx, disc = _disc(dmad_lambda_r=0.5)
    p = _pair(3)
    tau = torch.tensor([0.3, 0.8])
    eps = torch.randn(B, C, 3, 3, generator=torch.Generator().manual_seed(9))
    signal, stats = disc.student_signal(
        ctx, p["x_student"], tau, eps, p["crossattn_emb"]
    )
    assert all(q.requires_grad and q.grad is None for q in disc.params)

    x = p["x_student"].clone().requires_grad_()
    with disc_view(ctx.turbo, disc.stack):
        f = ctx.forward(
            "teacher",
            renoise(x, tau, eps),
            tau,
            p["crossattn_emb"],
            no_grad=False,
            return_block_features={disc.tap},
            return_features_early=True,
        )[disc.tap]
        h_t = disc.head_t([f]).mean(dim=1)
        h_r = disc.head_r([f]).mean(dim=1)
    (g,) = torch.autograd.grad(-(h_t + 0.5 * h_r).sum(), x)
    torch.testing.assert_close(signal, normalize_signal_rms(g, 0.18))
    assert set(stats) == {"g_t_rms", "g_r_rms", "cos_tr"}


def test_metrics_tolerate_missing_dm_terms():
    m = TurboMetrics(torch.device("cpu"))
    x = torch.randn(1, 4, 3, 3)
    m.accumulate_per_step(
        fake_loss_mean_t=torch.zeros(()),
        grad_signal=x,
        delta_dm=None,
        x_pred=x,
        v_student=x,
        tau_dm_e=torch.ones(1, 1, 1, 1),
        v_real_cond_dm=None,
        v_fake_cond_dm=None,
    )
    out = m.flush(1)
    assert out.dm == 0.0 and out.cos == 0.0 and out.grad > 0.0


# --- no critic under DMAD -------------------------------------------------------


class _SelfAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = nn.Linear(8, 18, bias=False)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _SelfAttn()
        self.mlp_layer1 = nn.Linear(8, 6, bias=False)


class _TinyDiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block()])


def test_build_fake_false_builds_no_critic():
    torch.manual_seed(0)
    turbo = TurboDMDNetwork(
        unet=_TinyDiT(), student_rank=4, fake_rank=4, build_fake=False
    )
    assert turbo.fake is None and turbo.fake_banks == []
    assert turbo.fake_params() == []
    turbo.freeze_dit()
    turbo.set_view("student")
    with pytest.raises(RuntimeError, match="build_fake"):
        turbo.set_view("fake")
    assert len(turbo.make_aux_stack().unet_loras) == len(turbo.student.unet_loras)
    with pytest.raises(ValueError, match="fake_tau_banks"):
        TurboDMDNetwork(
            unet=_TinyDiT(),
            student_rank=4,
            fake_rank=4,
            build_fake=False,
            fake_tau_banks=2,
        )
