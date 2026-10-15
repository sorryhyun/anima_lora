"""Invariant tests for the τ-split critic (turbo_tau_split_critic Phase 1).

Tier-1.5 guards for `fake_tau_banks`:

  * banks=1 constructs NO second network and leaves the update/query paths
    byte-identical (sample_t_routed resolves to the plain sample_t call on the
    same RNG stream; set_view touches only the single bank);
  * degenerate boundary=1.0 with both banks warm-started from the same weights
    is functionally equal to a single critic (bank-lo answers everything);
  * routing: τ < boundary → bank 0 trains/answers, else bank 1 — for updates
    AND queries;
  * config validation: banks=2 requires batch_size=1; boundary ∈ (0, 1];
  * resume: bank-count mismatches are REFUSED (the shared fake optimizer state
    cannot align), banks=2 round-trips both banks.

No GPU, no DiT: a one-Block tiny module satisfies LoRANetwork's target
discovery (class name "Block" is what ANIMA_TARGET_REPLACE_MODULE matches), so
the real chain/enabled/routing machinery is exercised end to end.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from networks.methods.turbo_dmd import TurboDMDNetwork
from scripts.distill_turbo.primitives import sample_t_routed
from scripts.distill_turbo.resume import (
    ResumeMismatchError,
    apply_resume_state,
    load_resume_state,
    resume_path_for,
    save_resume_state,
)

IN, HEAD = 8, 6
RANK = 4


class _SelfAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = nn.Linear(IN, 3 * HEAD, bias=False)


# Class name must be exactly "Block" — LoRANetwork.ANIMA_TARGET_REPLACE_MODULE
# matches on __class__.__name__.
class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _SelfAttn()
        self.mlp_layer1 = nn.Linear(IN, HEAD, bias=False)


class _TinyDiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block()])


def _make_turbo(banks: int, seed: int = 0) -> tuple[TurboDMDNetwork, _TinyDiT]:
    torch.manual_seed(seed)
    dit = _TinyDiT()
    turbo = TurboDMDNetwork(
        unet=dit,
        student_rank=RANK,
        fake_rank=RANK,
        fake_tau_banks=banks,
    )
    return turbo, dit


def _randomize(net) -> None:
    """Give a LoRA stack a nonzero ΔW (lora_up is zero-init by convention)."""
    for p in net.parameters():
        nn.init.normal_(p, std=0.1)


def _fake_out(turbo, dit, x) -> torch.Tensor:
    turbo.set_view("fake")
    with torch.no_grad():
        return dit.blocks[0].self_attn.qkv_proj(x)


# --- banks=1: byte-identical shipped loop -----------------------------------


def test_banks1_constructs_no_second_network():
    turbo, _ = _make_turbo(banks=1)
    assert turbo.fake_hi is None
    assert turbo.fake_banks == [turbo.fake]
    assert turbo.fake_bank == 0
    # fake_params is exactly the single bank's trainable params, same order.
    assert turbo.fake_params() == [
        p for p in turbo.fake.parameters() if p.requires_grad
    ]


def test_banks1_sample_t_routed_is_plain_sample_t():
    """Same RNG stream, same value, no turbo interaction → byte-identical."""

    class _Boom:
        def set_fake_bank(self, i):  # pragma: no cover - must never fire
            raise AssertionError("banks=1 must not touch the bank selector")

    torch.manual_seed(123)
    routed = sample_t_routed(
        4,
        turbo=_Boom(),
        fake_tau_banks=1,
        fake_tau_boundary=0.5,
        distribution="uniform",
        sigmoid_scale=1.0,
        device="cpu",
        dtype=torch.float32,
    )
    torch.manual_seed(123)
    plain = torch.rand(4, dtype=torch.float32)
    torch.testing.assert_close(routed, plain)


def test_banks1_set_fake_bank_out_of_range_raises():
    turbo, _ = _make_turbo(banks=1)
    with pytest.raises(ValueError, match="1 fake bank"):
        turbo.set_fake_bank(1)


# --- banks=2: exclusivity, routing, degenerate equality ----------------------


def test_banks2_fake_view_enables_exactly_one_bank():
    turbo, _ = _make_turbo(banks=2)
    lo, hi = turbo.fake_banks

    def flags():
        return (
            all(m.enabled for m in lo.unet_loras),
            all(m.enabled for m in hi.unet_loras),
        )

    def none_enabled(bank):
        return not any(m.enabled for m in bank.unet_loras)

    assert none_enabled(lo) and none_enabled(hi)  # teacher view at init
    turbo.set_view("fake")
    assert flags() == (True, False)
    turbo.set_fake_bank(1)  # mid-view switch flips both
    assert flags() == (False, True)
    turbo.set_view("teacher")
    assert none_enabled(lo) and none_enabled(hi)
    turbo.set_view("fake")  # re-enter: the remembered bank comes back
    assert flags() == (False, True)
    turbo.set_view("student")
    turbo.set_fake_bank(0)  # off-view switch defers the flip to set_view
    assert none_enabled(lo) and none_enabled(hi)
    turbo.set_view("fake")
    assert flags() == (True, False)


def test_banks2_routing_by_boundary():
    calls: list[int] = []

    class _Spy:
        def set_fake_bank(self, i):
            calls.append(i)

    torch.manual_seed(7)
    taus = []
    for _ in range(64):
        taus.append(
            float(
                sample_t_routed(
                    1,
                    turbo=_Spy(),
                    fake_tau_banks=2,
                    fake_tau_boundary=0.5,
                    distribution="uniform",
                    sigmoid_scale=1.0,
                    device="cpu",
                    dtype=torch.float32,
                )
            )
        )
    assert calls == [0 if t < 0.5 else 1 for t in taus]
    assert {0, 1} <= set(calls)  # both banks actually exercised


def test_banks2_degenerate_boundary_matches_single_critic():
    """boundary=1.0 → bank 0 answers everything; with both banks holding the
    same weights the fake view must match a banks=1 critic bit-for-bit.
    (Functional equality, not RNG-stream equality — constructing the second
    network advances the init RNG, so the fake weights are copied in
    explicitly. Same seed → identical BASE DiT weights, which are drawn before
    any LoRA construction consumes the stream.)"""
    turbo1, dit1 = _make_turbo(banks=1, seed=0)
    turbo2, dit2 = _make_turbo(banks=2, seed=0)
    _randomize(turbo1.fake)
    # Same file seeds both banks (the warm-start contract).
    sd = turbo1.fake.state_dict()
    turbo2.fake.load_state_dict(sd)
    turbo2.fake_hi.load_state_dict(sd)
    # Students differ across the two constructions — silence them by zeroing.
    for p in list(turbo1.student.parameters()) + list(turbo2.student.parameters()):
        nn.init.zeros_(p)

    x = torch.randn(2, IN)
    # boundary=1.0: uniform τ < 1.0 always → bank 0. Route a few draws.
    for _ in range(4):
        sample_t_routed(
            1,
            turbo=turbo2,
            fake_tau_banks=2,
            fake_tau_boundary=1.0,
            distribution="uniform",
            sigmoid_scale=1.0,
            device="cpu",
            dtype=torch.float32,
        )
        assert turbo2.fake_bank == 0
    torch.testing.assert_close(_fake_out(turbo2, dit2, x), _fake_out(turbo1, dit1, x))


def test_banks2_banks_differ_when_weights_differ():
    """The two banks are real alternatives: different weights → different fake
    view output, and each bank's output matches its own weights applied alone."""
    turbo, dit = _make_turbo(banks=2)
    _randomize(turbo.fake)
    _randomize(turbo.fake_hi)
    for p in turbo.student.parameters():
        nn.init.zeros_(p)

    x = torch.randn(2, IN)
    turbo.set_fake_bank(0)
    out_lo = _fake_out(turbo, dit, x)
    turbo.set_fake_bank(1)
    out_hi = _fake_out(turbo, dit, x)
    assert not torch.allclose(out_lo, out_hi)

    # Teacher view is clean (no bank leaks through).
    turbo.set_view("teacher")
    with torch.no_grad():
        base = dit.blocks[0].self_attn.qkv_proj(x)
    assert not torch.allclose(base, out_lo) and not torch.allclose(base, out_hi)


def test_banks2_fake_params_spans_both_banks_in_order():
    turbo, _ = _make_turbo(banks=2)
    want = [p for p in turbo.fake.parameters() if p.requires_grad] + [
        p for p in turbo.fake_hi.parameters() if p.requires_grad
    ]
    assert turbo.fake_params() == want


def test_invalid_bank_count_raises():
    with pytest.raises(ValueError, match="fake_tau_banks"):
        _make_turbo(banks=3)


# --- config validation --------------------------------------------------------


def _resolve(cfg_overrides: dict, argv: list[str] | None = None):
    from scripts.distill_turbo.config import build_argparser, resolve_config

    args = build_argparser().parse_args(argv or [])
    cfg = {
        "iterations": 10,
        "output_name": "toy",
        **cfg_overrides,
    }
    return resolve_config(args, cfg)


def test_config_banks2_requires_batch_size_1():
    with pytest.raises(ValueError, match="requires batch_size=1"):
        _resolve({"batch_size": 2, "network": {"fake_tau_banks": 2}})
    # batch_size=1 resolves fine.
    c = _resolve({"batch_size": 1, "network": {"fake_tau_banks": 2}})
    assert c.fake_tau_banks == 2 and c.fake_tau_boundary == 0.5


def test_config_boundary_and_banks_validation():
    with pytest.raises(ValueError, match="fake_tau_boundary"):
        _resolve({"network": {"fake_tau_banks": 2, "fake_tau_boundary": 0.0}})
    with pytest.raises(ValueError, match="fake_tau_boundary"):
        _resolve({"network": {"fake_tau_banks": 2, "fake_tau_boundary": 1.5}})
    with pytest.raises(ValueError, match="fake_tau_banks"):
        _resolve({"network": {"fake_tau_banks": 4}})
    # Defaults: off.
    c = _resolve({})
    assert c.fake_tau_banks == 1


# --- resume: bank topology is strict ------------------------------------------


@dataclass
class _Cfg:
    """Only the fields resume.py reads (mirrors tests/test_turbo_resume.py)."""

    output_dir: str
    output_name: str = "toy_turbo"
    student_rank: int = RANK
    fake_rank: int = RANK
    fake_tau_banks: int = 1
    fake_tau_boundary: float = 0.5
    gan_disc_hidden: int = 0
    gan_feature_block_idx: int = -1
    channel_scaling_alpha: float = 0.0
    base_loss: str = "dpdmd"
    iterations: int = 100
    student_lr: float = 5e-5
    fake_lr: float = 5e-5


class _ToyTurbo:
    """resume.py surface: .student/.fake/.fake_hi/.disc state_dict holders."""

    def __init__(self, banks: int = 1):
        torch.manual_seed(0)
        self.student = nn.Linear(IN, IN)
        self.fake = nn.Linear(IN, IN)
        self.fake_hi = nn.Linear(IN, IN) if banks == 2 else None
        self.disc = None


def _toy_build(cfg: _Cfg, banks: int):
    turbo = _ToyTurbo(banks=banks)
    params = list(turbo.fake.parameters()) + (
        list(turbo.fake_hi.parameters()) if turbo.fake_hi is not None else []
    )
    s_opt = torch.optim.AdamW(turbo.student.parameters(), lr=cfg.student_lr)
    f_opt = torch.optim.AdamW(params, lr=cfg.fake_lr)
    s_sch = torch.optim.lr_scheduler.CosineAnnealingLR(s_opt, T_max=cfg.iterations)
    f_sch = torch.optim.lr_scheduler.CosineAnnealingLR(f_opt, T_max=cfg.iterations)
    return turbo, s_opt, f_opt, s_sch, f_sch


def _toy_save(tmp_path, cfg, turbo, s_opt, f_opt, s_sch, f_sch, step=5):
    path = resume_path_for(str(tmp_path), cfg.output_name)
    save_resume_state(
        path,
        step=step,
        cfg=cfg,
        turbo=turbo,
        student_opt=s_opt,
        fake_opt=f_opt,
        disc_opt=None,
        student_sched=s_sch,
        fake_sched=f_sch,
        disc_sched=None,
        fdistill_bins=None,
    )
    return path


def _toy_apply(state, cfg, turbo, s_opt, f_opt, s_sch, f_sch):
    return apply_resume_state(
        state,
        cfg=cfg,
        turbo=turbo,
        student_opt=s_opt,
        fake_opt=f_opt,
        disc_opt=None,
        student_sched=s_sch,
        fake_sched=f_sch,
        disc_sched=None,
        fdistill_bins=None,
    )


def test_resume_banks2_round_trips_both_banks(tmp_path):
    cfg = _Cfg(output_dir=str(tmp_path), fake_tau_banks=2)
    turbo, *rest = _toy_build(cfg, banks=2)
    with torch.no_grad():
        turbo.fake.weight.add_(1.0)
        turbo.fake_hi.weight.add_(2.0)
    path = _toy_save(tmp_path, cfg, turbo, *rest)

    fresh, *frest = _toy_build(cfg, banks=2)
    step = _toy_apply(load_resume_state(path), cfg, fresh, *frest)
    assert step == 5
    torch.testing.assert_close(fresh.fake.weight, turbo.fake.weight)
    torch.testing.assert_close(fresh.fake_hi.weight, turbo.fake_hi.weight)


def test_resume_single_critic_bundle_refused_at_banks2(tmp_path):
    cfg1 = _Cfg(output_dir=str(tmp_path), fake_tau_banks=1)
    turbo, *rest = _toy_build(cfg1, banks=1)
    path = _toy_save(tmp_path, cfg1, turbo, *rest)

    cfg2 = _Cfg(output_dir=str(tmp_path), fake_tau_banks=2)
    fresh, *frest = _toy_build(cfg2, banks=2)
    # The fingerprint (fake_tau_banks strict) refuses before any half-load.
    with pytest.raises(ResumeMismatchError):
        _toy_apply(load_resume_state(path), cfg2, fresh, *frest)


def test_resume_banks2_bundle_refused_at_banks1(tmp_path):
    cfg2 = _Cfg(output_dir=str(tmp_path), fake_tau_banks=2)
    turbo, *rest = _toy_build(cfg2, banks=2)
    path = _toy_save(tmp_path, cfg2, turbo, *rest)

    cfg1 = _Cfg(output_dir=str(tmp_path), fake_tau_banks=1)
    fresh, *frest = _toy_build(cfg1, banks=1)
    with pytest.raises(ResumeMismatchError):
        _toy_apply(load_resume_state(path), cfg1, fresh, *frest)


def test_resume_old_bundle_without_banks_field_still_loads_at_banks1(tmp_path):
    """Pre-bank bundles (superturbo_B/C) have neither the fingerprint field nor
    the fake_hi key — they must keep resuming a banks=1 run untouched."""
    cfg = _Cfg(output_dir=str(tmp_path))
    turbo, *rest = _toy_build(cfg, banks=1)
    path = _toy_save(tmp_path, cfg, turbo, *rest)
    state = load_resume_state(path)
    # Simulate a pre-bank bundle: strip the new fingerprint field + state key.
    state["fingerprint"].pop("fake_tau_banks", None)
    state["fingerprint"].pop("fake_tau_boundary", None)
    state.pop("fake_hi", None)

    fresh, *frest = _toy_build(cfg, banks=1)
    assert _toy_apply(state, cfg, fresh, *frest) == 5
