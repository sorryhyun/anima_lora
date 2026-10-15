"""Invariant tests for the turbo crash-resume bundle (scripts/distill_turbo/resume.py).

The bundle exists because ``save_student`` ships ONLY the student LoRA — a crash
(the Xid-8 GPU hang that killed anima_turbo_T2 at step 3155) otherwise threw away
the critic, both AdamW moment sets and the LR schedule position. These tests pin
the properties that make a resume a *continuation* rather than a fresh run with
warm weights:

  * every mutable object round-trips bit-exact (weights, optimizer moments,
    scheduler position, f-distill EMA, step counter);
  * the restored LR is the schedule's value AT the resumed step, not its peak;
  * a topology change (rank, disc width, objective) is REFUSED, not silently
    loaded into a mismatched stack.

No GPU, no DiT: the resume path only ever touches state_dicts, so toy modules
exercise it exactly as the real stack does.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from scripts.distill_turbo.resume import (
    ResumeMismatchError,
    apply_resume_state,
    load_resume_state,
    resolve_resume_arg,
    resume_path_for,
    save_resume_state,
)


@dataclass
class _Cfg:
    """Only the fields resume.py reads (fingerprint + output naming)."""

    output_dir: str
    output_name: str = "toy_turbo"
    resume: str = ""
    # strict (shape / topology)
    student_rank: int = 96
    fake_rank: int = 96
    gan_disc_hidden: int = 0
    gan_feature_block_idx: int = -1
    channel_scaling_alpha: float = 0.0
    base_loss: str = "dpdmd"
    # warn (objective / schedule)
    iterations: int = 8000
    student_lr: float = 5e-5
    fake_lr: float = 5e-5
    gan_disc_lr: float = 1e-5
    student_alpha: float = 212.0
    fake_alpha: float = 212.0
    student_steps: int = 4
    div_weight: float = 0.05
    gan_loss_weight_gen: float = 0.03
    f_div: str = "rkl"
    fake_steps_per_student_step: int = 4
    teacher_cfg: float = 4.0


class _Turbo:
    """Stand-in for TurboDMDNetwork: resume.py only uses .student/.fake/.disc."""

    def __init__(self, *, width: int = 8, disc: bool = True):
        torch.manual_seed(0)
        self.student = nn.Linear(width, width)
        self.fake = nn.Linear(width, width)
        self.disc = nn.Linear(width, 1) if disc else None


def _build(cfg: _Cfg, *, width: int = 8, disc: bool = True):
    """A toy turbo stack + the three optimizer/scheduler pairs the real loop has."""
    turbo = _Turbo(width=width, disc=disc)
    s_opt = torch.optim.AdamW(turbo.student.parameters(), lr=cfg.student_lr)
    f_opt = torch.optim.AdamW(turbo.fake.parameters(), lr=cfg.fake_lr)
    d_opt = (
        torch.optim.AdamW(turbo.disc.parameters(), lr=cfg.gan_disc_lr)
        if turbo.disc is not None
        else None
    )
    # Cosine over the full run — the thing a naive "restart from the student ckpt"
    # resets to its peak, and the reason the step-1000 critic runaway happened.
    s_sched = torch.optim.lr_scheduler.CosineAnnealingLR(s_opt, T_max=cfg.iterations)
    f_sched = torch.optim.lr_scheduler.CosineAnnealingLR(f_opt, T_max=cfg.iterations)
    d_sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(d_opt, T_max=cfg.iterations)
        if d_opt is not None
        else None
    )
    return turbo, s_opt, f_opt, d_opt, s_sched, f_sched, d_sched


def _spin(turbo, opts, scheds, steps: int):
    """Run `steps` real optimizer steps so moments/schedules are non-trivial."""
    for _ in range(steps):
        for mod, opt in zip((turbo.student, turbo.fake, turbo.disc), opts):
            if mod is None or opt is None:
                continue
            opt.zero_grad()
            mod(torch.randn(4, mod.in_features)).sum().backward()
            opt.step()
        for sch in scheds:
            if sch is not None:
                sch.step()


def _save(tmp_path, cfg, step, turbo, opts, scheds, bins=None):
    path = resume_path_for(str(tmp_path), cfg.output_name)
    save_resume_state(
        path,
        step=step,
        cfg=cfg,
        turbo=turbo,
        student_opt=opts[0],
        fake_opt=opts[1],
        disc_opt=opts[2],
        student_sched=scheds[0],
        fake_sched=scheds[1],
        disc_sched=scheds[2],
        fdistill_bins=bins,
    )
    return path


def _apply(state, cfg, turbo, opts, scheds, bins=None):
    return apply_resume_state(
        state,
        cfg=cfg,
        turbo=turbo,
        student_opt=opts[0],
        fake_opt=opts[1],
        disc_opt=opts[2],
        student_sched=scheds[0],
        fake_sched=scheds[1],
        disc_sched=scheds[2],
        fdistill_bins=bins,
    )


def test_resume_round_trips_every_mutable_object(tmp_path):
    """Weights, AdamW moments, schedule position and step counter all come back."""
    cfg = _Cfg(output_dir=str(tmp_path))
    turbo, s_opt, f_opt, d_opt, s_sch, f_sch, d_sch = _build(cfg)
    opts, scheds = (s_opt, f_opt, d_opt), (s_sch, f_sch, d_sch)

    _spin(turbo, opts, scheds, steps=3155)  # where T2 actually died
    bins = torch.rand(10)
    path = _save(tmp_path, cfg, 3155, turbo, opts, scheds, bins)

    want_w = {k: v.clone() for k, v in turbo.student.state_dict().items()}
    want_disc = {k: v.clone() for k, v in turbo.disc.state_dict().items()}
    want_exp_avg = copy.deepcopy(
        s_opt.state[next(iter(s_opt.state))]["exp_avg"]
    ).clone()
    want_lr = s_sch.get_last_lr()[0]

    # Fresh stack, as a relaunched process would build it.
    fresh, *rest = _build(cfg)
    f_opts, f_scheds = tuple(rest[:3]), tuple(rest[3:])
    fresh_bins = torch.ones(10)
    step = _apply(load_resume_state(path), cfg, fresh, f_opts, f_scheds, fresh_bins)

    assert step == 3155
    for k, v in fresh.student.state_dict().items():
        torch.testing.assert_close(v, want_w[k])
    for k, v in fresh.disc.state_dict().items():
        torch.testing.assert_close(v, want_disc[k])
    # Optimizer moments — the part a warm-start restart silently zeroes.
    got_exp_avg = f_opts[0].state[next(iter(f_opts[0].state))]["exp_avg"]
    torch.testing.assert_close(got_exp_avg, want_exp_avg)
    torch.testing.assert_close(fresh_bins, bins)
    assert f_scheds[0].get_last_lr()[0] == pytest.approx(want_lr)


def test_resumed_lr_is_mid_schedule_not_peak(tmp_path):
    """The whole point: resume lands on the decayed LR, not back at the peak.

    A restart via --student_init_weights rebuilds the scheduler at step 0, i.e.
    peak LR against a trained student — the regime the T2 critic runaway lives in.
    """
    cfg = _Cfg(output_dir=str(tmp_path))
    turbo, *rest = _build(cfg)
    opts, scheds = tuple(rest[:3]), tuple(rest[3:])
    _spin(turbo, opts, scheds, steps=3155)
    path = _save(tmp_path, cfg, 3155, turbo, opts, scheds)

    fresh, *rest2 = _build(cfg)
    f_opts, f_scheds = tuple(rest2[:3]), tuple(rest2[3:])
    peak = f_scheds[0].get_last_lr()[0]  # freshly-built scheduler == peak LR
    _apply(load_resume_state(path), cfg, fresh, f_opts, f_scheds)
    resumed = f_scheds[0].get_last_lr()[0]

    assert resumed < peak * 0.95, (
        f"resumed LR {resumed:.3g} should sit well below the peak {peak:.3g} "
        "— the cosine must continue, not restart"
    )


def test_extending_iterations_resumes_onto_new_curve(tmp_path):
    """Resume a fully-annealed 4k run with iterations=8000 → LR climbs back to
    the NEW curve's mid-schedule value, it does not step the OLD cosine past its
    T_max (which drives the recursive update NEGATIVE — the superturbo_B
    extension bug this pins).

    Uses the real make_scheduler shape (warmup → cosine SequentialLR), because
    the failure lives in the nested schedulers' state_dicts.
    """
    from scripts.distill_turbo.primitives import make_scheduler

    def _real_scheds(cfg, opts):
        return tuple(
            make_scheduler(opt, cfg.iterations, lr) if opt is not None else None
            for opt, lr in zip(opts, (cfg.student_lr, cfg.fake_lr, cfg.gan_disc_lr))
        )

    cfg = _Cfg(output_dir=str(tmp_path), iterations=4000)
    turbo, *rest = _build(cfg)
    opts = tuple(rest[:3])
    scheds = _real_scheds(cfg, opts)
    _spin(turbo, opts, scheds, steps=4000)  # the cosine is fully annealed
    path = _save(tmp_path, cfg, 4000, turbo, opts, scheds)

    longer = _Cfg(output_dir=str(tmp_path), iterations=8000)
    fresh, *r = _build(longer)
    f_opts = tuple(r[:3])
    f_scheds = _real_scheds(longer, f_opts)
    _apply(load_resume_state(path), longer, fresh, f_opts, f_scheds)

    # Reference: what an 8000-sized curve reads at step 4000 when stepped there
    # directly. The resumed scheduler must sit on that value, not the old floor.
    ref_opt = torch.optim.AdamW(nn.Linear(8, 8).parameters(), lr=longer.student_lr)
    ref = make_scheduler(ref_opt, longer.iterations, longer.student_lr)
    for _ in range(4000):
        ref_opt.step()
        ref.step()
    resumed = f_scheds[0].get_last_lr()[0]
    assert resumed == pytest.approx(ref.get_last_lr()[0])
    assert resumed > 0.4 * longer.student_lr  # mid-curve, well off the 0.1·lr floor

    # And running the extension out must keep the LR sane the whole way.
    for _ in range(4000):
        f_opts[0].step()
        f_scheds[0].step()
    end = f_scheds[0].get_last_lr()[0]
    assert 0 < end == pytest.approx(0.1 * longer.student_lr, rel=1e-3)


def test_topology_change_is_refused(tmp_path):
    """A rank/objective/disc-width change must raise, never load into a mismatch."""
    cfg = _Cfg(output_dir=str(tmp_path))
    turbo, *rest = _build(cfg)
    opts, scheds = tuple(rest[:3]), tuple(rest[3:])
    _spin(turbo, opts, scheds, steps=5)
    path = _save(tmp_path, cfg, 5, turbo, opts, scheds)
    state = load_resume_state(path)

    for field, bad in (
        ("student_rank", 64),
        ("fake_rank", 32),
        ("base_loss", "dmd"),
        ("gan_disc_hidden", 512),
    ):
        drifted = _Cfg(output_dir=str(tmp_path), **{field: bad})
        fresh, *r = _build(drifted)
        with pytest.raises(ResumeMismatchError, match="different model topology"):
            _apply(state, drifted, fresh, tuple(r[:3]), tuple(r[3:]))


def test_schedule_reshape_is_allowed_with_warning(tmp_path, caplog):
    """Shortening `iterations` (the fix for T2's stretched cosine) must still resume."""
    cfg = _Cfg(output_dir=str(tmp_path), iterations=8000)
    turbo, *rest = _build(cfg)
    opts, scheds = tuple(rest[:3]), tuple(rest[3:])
    _spin(turbo, opts, scheds, steps=100)
    path = _save(tmp_path, cfg, 100, turbo, opts, scheds)

    shorter = _Cfg(output_dir=str(tmp_path), iterations=5000)
    fresh, *r = _build(shorter)
    with caplog.at_level("WARNING"):
        step = _apply(
            load_resume_state(path), shorter, fresh, tuple(r[:3]), tuple(r[3:])
        )
    assert step == 100
    assert any("iterations" in m for m in caplog.messages), (
        "a schedule reshape must warn — it is legitimate but never silent"
    )


def test_gan_off_bundle_refused_when_gan_on(tmp_path):
    """Bundle written with no disc cannot resume a GAN-on run."""
    cfg = _Cfg(output_dir=str(tmp_path))
    turbo, *rest = _build(cfg, disc=False)
    opts, scheds = tuple(rest[:3]), tuple(rest[3:])
    _spin(turbo, opts, scheds, steps=5)
    path = _save(tmp_path, cfg, 5, turbo, opts, scheds)

    fresh, *r = _build(cfg, disc=True)
    with pytest.raises(ResumeMismatchError, match="no disc state"):
        _apply(load_resume_state(path), cfg, fresh, tuple(r[:3]), tuple(r[3:]))


def test_auto_starts_fresh_when_no_bundle(tmp_path):
    """`--resume auto` is safe to bake into a restart wrapper: fresh on first launch."""
    arg = resolve_resume_arg("auto", str(tmp_path), "toy_turbo")
    assert arg.path is None and not arg.explicit

    cfg = _Cfg(output_dir=str(tmp_path))
    turbo, *rest = _build(cfg)
    _save(tmp_path, cfg, 7, turbo, tuple(rest[:3]), tuple(rest[3:]))

    arg = resolve_resume_arg("auto", str(tmp_path), "toy_turbo")
    assert arg.path is not None and arg.path.exists()


def test_explicit_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_resume_arg(str(tmp_path / "nope.pt"), str(tmp_path), "toy_turbo")


def test_partial_write_does_not_destroy_previous_bundle(tmp_path):
    """Atomic write: the bundle on disk is always a complete one.

    The failure this guards is the nasty one — an Xid hang *during* the save would,
    with a naive torch.save straight to the final path, leave a truncated bundle and
    take the resume down with the run.
    """
    cfg = _Cfg(output_dir=str(tmp_path))
    turbo, *rest = _build(cfg)
    opts, scheds = tuple(rest[:3]), tuple(rest[3:])
    _spin(turbo, opts, scheds, steps=10)
    path = _save(tmp_path, cfg, 10, turbo, opts, scheds)

    _spin(turbo, opts, scheds, steps=10)
    with pytest.raises(RuntimeError):
        # Simulate the process dying inside the second save.
        import scripts.distill_turbo.resume as R

        real_save = torch.save
        try:
            torch.save = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
            R.save_resume_state(
                path,
                step=20,
                cfg=cfg,
                turbo=turbo,
                student_opt=opts[0],
                fake_opt=opts[1],
                disc_opt=opts[2],
                student_sched=scheds[0],
                fake_sched=scheds[1],
                disc_sched=scheds[2],
                fdistill_bins=None,
            )
        finally:
            torch.save = real_save

    # The step-10 bundle must still be intact and loadable.
    assert load_resume_state(path)["step"] == 10
