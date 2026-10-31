"""Invariant tests for dual-pool gradient routing (turbo_dual_pool_grad_routing).

Tier-1.5 guards for the `dual_pool` student:

  * single-pool (dual_pool=false) constructs NO second stack and every path is
    byte-identical to the shipped loop (student_pools == [student], the routing
    handles are no-ops);
  * pool A is built at div_pool_rank, plain-LoRA only, zero-init up (ΔW=0 at
    init) — at step 0 the merged student IS the warm-started pool B;
  * set_view("student") enables BOTH pools; teacher/fake disable both;
  * ROUTING CORRECTNESS: after a student-view backward, route_div() lands grad on
    pool A alone (pool B grad None) and route_quality() on pool B alone;
    route_all_on() restores both;
  * MERGE EQUIVALENCE: the saved concat LoRA reproduces ΔW_A + ΔW_B per module
    exactly, and div_scale dials pool A (ΔW_B + α·ΔW_A);
  * config validation: dual_pool requires dpdmd + detach_after_first, is mutually
    exclusive with per_step_expert, div_pool_rank >= 1;
  * resume: pool A round-trips; a single↔dual presence mismatch is REFUSED.

No GPU, no real DiT: a one-Block tiny module satisfies LoRANetwork's target
discovery (class name "Block" is what ANIMA_TARGET_REPLACE_MODULE matches), same
stub as tests/test_turbo_tau_critic.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn
from safetensors.torch import load_file

from networks.methods.turbo_dmd import TurboDMDNetwork
from scripts.distill_turbo.config import build_argparser, resolve_config
from scripts.distill_turbo.resume import (
    ResumeMismatchError,
    apply_resume_state,
    load_resume_state,
    resume_path_for,
    save_resume_state,
)

IN, HEAD = 8, 6
RANK_B, RANK_A = 4, 2


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


def _make_turbo(dual_pool: bool, div_pool_rank: int = RANK_A, seed: int = 0):
    torch.manual_seed(seed)
    dit = _TinyDiT()
    turbo = TurboDMDNetwork(
        unet=dit,
        student_rank=RANK_B,
        fake_rank=RANK_B,
        dual_pool=dual_pool,
        div_pool_rank=div_pool_rank,
    )
    return turbo, dit


def _randomize(net) -> None:
    """Give a stack a nonzero ΔW (lora_up is zero-init by convention)."""
    for p in net.parameters():
        nn.init.normal_(p, std=0.1)


def _student_forward_sum(turbo, dit, x):
    turbo.set_view("student")
    return dit.blocks[0].self_attn.qkv_proj(x).sum() + dit.blocks[0].mlp_layer1(x).sum()


def _delta(sd: dict, prefix: str) -> torch.Tensor:
    """ΔW for one plain-LoRA module from its on-disk (down, up, alpha)."""
    down = sd[prefix + ".lora_down.weight"].float()
    up = sd[prefix + ".lora_up.weight"].float()
    r = down.shape[0]
    alpha = float(sd[prefix + ".alpha"]) if prefix + ".alpha" in sd else float(r)
    return (alpha / r) * (up @ down)


# --- single-pool: byte-identical shipped loop -------------------------------


def test_single_pool_constructs_no_second_stack():
    turbo, _ = _make_turbo(dual_pool=False)
    assert turbo.student_div is None
    assert turbo.student_pools == [turbo.student]
    # student_params is exactly the sole stack's trainable params.
    assert turbo.student_params() == [
        p for p in turbo.student.parameters() if p.requires_grad
    ]


def test_single_pool_routing_handles_are_noops():
    turbo, dit = _make_turbo(dual_pool=False)
    before = [p.requires_grad for p in turbo.student.parameters()]
    turbo.route_div()
    turbo.route_quality()
    turbo.route_all_on()
    after = [p.requires_grad for p in turbo.student.parameters()]
    assert before == after  # untouched


def test_single_pool_param_groups_is_one_group():
    turbo, _ = _make_turbo(dual_pool=False)
    groups = turbo.student_param_groups(5e-5)
    assert len(groups) == 1
    assert groups[0]["lr"] == 5e-5


# --- dual-pool construction --------------------------------------------------


def test_dual_pool_builds_pool_a():
    turbo, _ = _make_turbo(dual_pool=True, div_pool_rank=RANK_A)
    assert turbo.student_div is not None
    assert turbo.student_pools == [turbo.student, turbo.student_div]
    # Pool A rank = div_pool_rank on every module.
    for m in turbo.student_div.unet_loras:
        assert m.lora_down.weight.shape[0] == RANK_A


def test_dual_pool_pool_a_is_zero_init():
    """Zero-init up ⇒ ΔW_A = 0 at init ⇒ merged student == warm-started pool B."""
    turbo, _ = _make_turbo(dual_pool=True)
    for m in turbo.student_div.unet_loras:
        assert torch.count_nonzero(m.lora_up.weight) == 0


def test_dual_pool_rejects_per_step_expert():
    dit = _TinyDiT()
    with pytest.raises(ValueError, match="per-step-expert"):
        TurboDMDNetwork(
            unet=dit,
            student_rank=RANK_B,
            fake_rank=RANK_B,
            dual_pool=True,
            student_step_expert_K=2,
        )


# --- set_view toggles both pools --------------------------------------------


def test_set_view_toggles_both_pools():
    turbo, _ = _make_turbo(dual_pool=True)
    turbo.set_view("student")
    # Probe via the module `enabled` flag directly (set_enabled writes it).
    assert all(m.enabled for m in turbo.student.unet_loras)
    assert all(m.enabled for m in turbo.student_div.unet_loras)
    turbo.set_view("fake")
    assert not any(m.enabled for m in turbo.student.unet_loras)
    assert not any(m.enabled for m in turbo.student_div.unet_loras)


# --- routing correctness (the whole point) ----------------------------------


def _grad_present(net) -> bool:
    return all(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


def _grad_absent(net) -> bool:
    return all(p.grad is None for p in net.parameters())


def test_route_div_grads_pool_a_only():
    turbo, dit = _make_turbo(dual_pool=True)
    _randomize(turbo.student)
    _randomize(turbo.student_div)
    x = torch.randn(2, IN)

    turbo.route_div()
    _student_forward_sum(turbo, dit, x).backward()
    assert _grad_present(turbo.student_div)  # pool A got the diversity grad
    assert _grad_absent(turbo.student)  # pool B untouched


def test_route_quality_grads_pool_b_only():
    turbo, dit = _make_turbo(dual_pool=True)
    _randomize(turbo.student)
    _randomize(turbo.student_div)
    x = torch.randn(2, IN)

    turbo.route_quality()
    _student_forward_sum(turbo, dit, x).backward()
    assert _grad_present(turbo.student)  # pool B got the refinement grad
    assert _grad_absent(turbo.student_div)  # pool A untouched


def test_route_all_on_restores_both():
    turbo, _ = _make_turbo(dual_pool=True)
    turbo.route_div()
    assert not all(p.requires_grad for p in turbo.student.parameters())
    turbo.route_all_on()
    assert all(p.requires_grad for p in turbo.student.parameters())
    assert all(p.requires_grad for p in turbo.student_div.parameters())


def test_param_groups_two_groups_with_lrs():
    turbo, _ = _make_turbo(dual_pool=True)
    groups = turbo.student_param_groups(5e-5, div_pool_lr=2e-4)
    assert len(groups) == 2
    assert groups[0]["lr"] == 5e-5  # pool B
    assert groups[1]["lr"] == 2e-4  # pool A
    # div_pool_lr=0 inherits student_lr.
    assert turbo.student_param_groups(5e-5, div_pool_lr=0.0)[1]["lr"] == 5e-5


# --- merge equivalence: saved concat LoRA == ΔW_A + ΔW_B --------------------


def test_merge_reproduces_sum_of_deltas(tmp_path):
    turbo, _ = _make_turbo(dual_pool=True)
    _randomize(turbo.student)
    _randomize(turbo.student_div)

    sd_b = TurboDMDNetwork._finalize_pool_state_dict(turbo.student, None)[0]
    sd_a = TurboDMDNetwork._finalize_pool_state_dict(turbo.student_div, None)[0]

    out = tmp_path / "merged.safetensors"
    turbo.save_student(str(out), dtype=torch.float32)
    merged = load_file(str(out))

    # Every pool-A module (⊆ pool B) merges to ΔW_A + ΔW_B exactly.
    prefixes = [
        k[: -len(".lora_down.weight")] for k in sd_a if k.endswith(".lora_down.weight")
    ]
    assert prefixes  # sanity: the merge actually covered modules
    for pfx in prefixes:
        expected = _delta(sd_a, pfx) + _delta(sd_b, pfx)
        got = _delta(merged, pfx)
        assert got.shape == expected.shape
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)
    # Merged rank = r_B + r_A on a merged module.
    any_pfx = prefixes[0]
    assert merged[any_pfx + ".lora_down.weight"].shape[0] == RANK_B + RANK_A


def test_div_scale_dials_pool_a(tmp_path):
    turbo, _ = _make_turbo(dual_pool=True)
    _randomize(turbo.student)
    _randomize(turbo.student_div)
    sd_b = TurboDMDNetwork._finalize_pool_state_dict(turbo.student, None)[0]
    sd_a = TurboDMDNetwork._finalize_pool_state_dict(turbo.student_div, None)[0]
    pfx = next(
        k[: -len(".lora_down.weight")] for k in sd_a if k.endswith(".lora_down.weight")
    )

    for scale in (0.0, 0.5):
        out = tmp_path / f"merged_{scale}.safetensors"
        turbo.save_student(str(out), dtype=torch.float32, div_scale=scale)
        merged = load_file(str(out))
        expected = _delta(sd_b, pfx) + scale * _delta(sd_a, pfx)
        torch.testing.assert_close(_delta(merged, pfx), expected, rtol=1e-4, atol=1e-5)


# --- config validation -------------------------------------------------------


def _resolve(cli: list[str] | None = None, cfg: dict | None = None):
    args = build_argparser().parse_args(cli or [])
    return resolve_config(args, cfg or {})


def test_dual_pool_default_off():
    c = _resolve()
    assert c.dual_pool is False
    assert c.div_pool_rank == 16  # documented default


def test_dual_pool_cli_and_toml():
    assert _resolve(["--dual_pool"]).dual_pool is True
    assert _resolve(cfg={"network": {"dual_pool": True}}).dual_pool is True
    assert (
        _resolve(
            ["--div_pool_rank", "24"], cfg={"network": {"div_pool_rank": 8}}
        ).div_pool_rank
        == 24  # CLI wins
    )
    assert _resolve(cfg={"network": {"div_pool_rank": 8}}).div_pool_rank == 8


def test_dual_pool_requires_dpdmd():
    with pytest.raises(ValueError, match="dpdmd"):
        _resolve(["--dual_pool", "--base_loss", "dmd"])


def test_dual_pool_requires_detach():
    with pytest.raises(ValueError, match="detach_after_first"):
        _resolve(["--dual_pool", "--no_detach_after_first"])


def test_dual_pool_excludes_per_step_expert():
    with pytest.raises(ValueError, match="per_step_expert"):
        _resolve(["--dual_pool", "--per_step_expert"])


def test_div_pool_rank_must_be_positive():
    with pytest.raises(ValueError, match="div_pool_rank"):
        _resolve(["--dual_pool", "--div_pool_rank", "0"])


# --- resume ------------------------------------------------------------------


@dataclass
class _Cfg:
    """Only the fields resume.py reads."""

    output_dir: str
    output_name: str = "toy_dual"
    student_rank: int = RANK_B
    fake_rank: int = RANK_B
    fake_tau_banks: int = 1
    step_expert_K: int = 0
    dual_pool: bool = False
    div_pool_rank: int = RANK_A
    gan_disc_hidden: int = 0
    gan_feature_block_idx: int = -1
    channel_scaling_alpha: float = 0.0
    base_loss: str = "dpdmd"
    iterations: int = 100
    student_lr: float = 5e-5
    fake_lr: float = 5e-5


class _ToyTurbo:
    """resume.py surface: .student/.student_div/.fake/.fake_hi/.disc holders."""

    def __init__(self, dual_pool: bool = False):
        torch.manual_seed(0)
        self.student = nn.Linear(IN, IN)
        self.student_div = nn.Linear(IN, IN) if dual_pool else None
        self.fake = nn.Linear(IN, IN)
        self.fake_hi = None
        self.disc = None


def _toy_build(cfg: _Cfg, dual_pool: bool):
    turbo = _ToyTurbo(dual_pool=dual_pool)
    s_params = list(turbo.student.parameters()) + (
        list(turbo.student_div.parameters()) if turbo.student_div is not None else []
    )
    s_opt = torch.optim.AdamW(s_params, lr=cfg.student_lr)
    f_opt = torch.optim.AdamW(turbo.fake.parameters(), lr=cfg.fake_lr)
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


def test_resume_dual_pool_round_trips_pool_a(tmp_path):
    cfg = _Cfg(output_dir=str(tmp_path), dual_pool=True)
    turbo, *rest = _toy_build(cfg, dual_pool=True)
    with torch.no_grad():
        turbo.student_div.weight.add_(3.0)
    path = _toy_save(tmp_path, cfg, turbo, *rest)

    fresh, *frest = _toy_build(cfg, dual_pool=True)
    step = _toy_apply(load_resume_state(path), cfg, fresh, *frest)
    assert step == 5
    torch.testing.assert_close(fresh.student_div.weight, turbo.student_div.weight)


def test_resume_single_bundle_refused_at_dual(tmp_path):
    cfg1 = _Cfg(output_dir=str(tmp_path), dual_pool=False)
    turbo, *rest = _toy_build(cfg1, dual_pool=False)
    path = _toy_save(tmp_path, cfg1, turbo, *rest)

    cfg2 = _Cfg(output_dir=str(tmp_path), dual_pool=True)
    fresh, *frest = _toy_build(cfg2, dual_pool=True)
    with pytest.raises(ResumeMismatchError):
        _toy_apply(load_resume_state(path), cfg2, fresh, *frest)


def test_resume_dual_bundle_refused_at_single(tmp_path):
    cfg2 = _Cfg(output_dir=str(tmp_path), dual_pool=True)
    turbo, *rest = _toy_build(cfg2, dual_pool=True)
    path = _toy_save(tmp_path, cfg2, turbo, *rest)

    cfg1 = _Cfg(output_dir=str(tmp_path), dual_pool=False)
    fresh, *frest = _toy_build(cfg1, dual_pool=False)
    with pytest.raises(ResumeMismatchError):
        _toy_apply(load_resume_state(path), cfg1, fresh, *frest)


def test_before_apply_runs_with_stacks_built_and_unpatched():
    dit = _TinyDiT()
    target = dit.blocks[0].mlp_layer1
    original = target.forward
    seen = []

    class Converted(nn.Linear):
        def forward(self, x):
            seen.append("converted")
            return super().forward(x)

    def convert() -> None:
        assert target.forward == original  # no apply_to yet
        target.__class__ = Converted

    turbo = TurboDMDNetwork(
        unet=dit, student_rank=RANK_B, fake_rank=RANK_B, before_apply=convert
    )
    assert len(turbo.student.unet_loras) > 0
    target(torch.zeros(1, IN))
    assert seen == ["converted"]


def test_int8_config_and_its_refusals():
    assert _resolve().int8 == "off"
    assert _resolve(["--int8", "stream"]).int8 == "stream"
    assert _resolve(cfg={"int8": "full"}).int8 == "full"
    with pytest.raises(ValueError, match="blocks_to_swap=0"):
        _resolve(["--int8", "stream", "--blocks_to_swap", "4"])
    with pytest.raises(ValueError, match="dmad"):
        _resolve(["--int8", "full"], {"dmad_probe": {"enabled": True}})
