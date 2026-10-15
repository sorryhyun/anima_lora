"""Crash-resume state for the turbo distillation loop.

``save_student`` ships only the student LoRA — the critic, discriminator, and
all three optimizer/scheduler pairs are training scaffolding that only ever
lived in RAM. A mid-run death loses all of it: restarting from a step-tagged
checkpoint would re-warm a *cold* critic against an *already trained* student
and reset the LR cosine to its peak — the exact regime the critic runaway lives
in. This module checkpoints the rest of the loop so a restart continues the run
instead of re-opening it.

The bundle is a single rolling ``{output_name}_resume.pt`` next to the
step-tagged students, overwritten each ``save_every`` and written atomically
(tmp + ``os.replace``, so a crash mid-write leaves the previous bundle intact).
Only the latest is kept — it's ~10x a student checkpoint's size (two AdamW
moments in fp32 over both LoRA stacks), and the step-tagged students already
preserve the trajectory.

Not restored: the dataloader's position within its epoch (perturbs sample
order, not the optimizer/critic/schedule state that actually carries the run).
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1

# Fields whose value determines a tensor SHAPE or the module topology of the
# stack we are loading into. A mismatch means the bundle's tensors cannot be
# loaded at all (or would load into a silently different model), so we refuse.
_STRICT_FIELDS = (
    "student_rank",
    "fake_rank",
    "fake_tau_banks",
    "gan_disc_hidden",
    "gan_feature_block_idx",
    "channel_scaling_alpha",
    "base_loss",
)

# Fields that change the objective or the schedule but not the shapes. Resuming
# across a change here is legitimate (e.g. extending `iterations` past a run
# whose cosine already annealed, or lowering peak LR after a critic runaway) —
# the schedulers restore their step counter onto the NEW curve (see
# ``_restore_scheduler``). We warn so the change is never silent.
_WARN_FIELDS = (
    "fake_tau_boundary",
    "iterations",
    "student_lr",
    "fake_lr",
    "gan_disc_lr",
    # Head granularity changes the objective, not the disc's parameters (same
    # MLP applied pooled vs per-token) — bundles load across a switch.
    "gan_disc_head",
    "student_alpha",
    "fake_alpha",
    "student_steps",
    "div_weight",
    "gan_loss_weight_gen",
    # Delay/warmup reshape the generator-side λ ramp around the resume step —
    # legitimate (e.g. adding a ramp after a collapsed cold start), never silent.
    "gan_delay_steps",
    "gan_warmup_steps",
    "f_div",
    "fake_steps_per_student_step",
    "teacher_cfg",
    "dynamic_schedule",
    "cdm_weight",
)


class ResumeMismatchError(RuntimeError):
    """A resume bundle is incompatible with the config being resumed into."""


def resume_path_for(output_dir: str, output_name: str) -> Path:
    """Canonical rolling-bundle path for a run (sibling of its step-tagged ckpts)."""
    return Path(output_dir) / output_name / f"{output_name}_resume.pt"


def _fingerprint(cfg) -> dict[str, Any]:
    fields = _STRICT_FIELDS + _WARN_FIELDS
    return {k: getattr(cfg, k) for k in fields if hasattr(cfg, k)}


def save_resume_state(
    path: str | Path,
    *,
    step: int,
    cfg,
    turbo,
    student_opt,
    fake_opt,
    disc_opt,
    student_sched,
    fake_sched,
    disc_sched,
    fdistill_bins: torch.Tensor | None,
) -> None:
    """Atomically write the full mutable loop state after ``step`` completed steps."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "format": FORMAT_VERSION,
        "step": int(step),
        "output_name": cfg.output_name,
        "fingerprint": _fingerprint(cfg),
        "student": turbo.student.state_dict(),
        "fake": turbo.fake.state_dict(),
        # τ-split critic second bank (None on the shipped single-critic loop;
        # getattr keeps pre-bank TurboDMDNetwork stand-ins loadable).
        "fake_hi": (
            turbo.fake_hi.state_dict()
            if getattr(turbo, "fake_hi", None) is not None
            else None
        ),
        "disc": turbo.disc.state_dict() if turbo.disc is not None else None,
        "student_opt": student_opt.state_dict(),
        "fake_opt": fake_opt.state_dict(),
        "disc_opt": disc_opt.state_dict() if disc_opt is not None else None,
        "student_sched": student_sched.state_dict(),
        "fake_sched": fake_sched.state_dict(),
        "disc_sched": disc_sched.state_dict() if disc_sched is not None else None,
        "fdistill_bins": fdistill_bins.detach().cpu()
        if fdistill_bins is not None
        else None,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        },
    }

    # tmp + replace: a crash mid-write must not shred the previous bundle. Same
    # directory so the rename stays on one filesystem (os.replace is atomic there).
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_resume_state(path: str | Path) -> dict[str, Any]:
    """Read a bundle written by :func:`save_resume_state`."""
    # weights_only=False: the bundle carries optimizer/scheduler state and RNG
    # blobs (numpy/random tuples), not just tensors. It is our own file, written
    # by this module into our own output dir.
    state = torch.load(path, map_location="cpu", weights_only=False)
    fmt = state.get("format")
    if fmt != FORMAT_VERSION:
        raise ResumeMismatchError(
            f"resume bundle {path} has format {fmt!r}, expected {FORMAT_VERSION}"
        )
    return state


@dataclass
class ResumeArg:
    """A resolved ``--resume`` request."""

    path: Path | None  # None → start fresh (only legal for `auto` with no bundle)
    explicit: bool  # user named a path (vs `auto`)


def resolve_resume_arg(value: str, output_dir: str, output_name: str) -> ResumeArg:
    """Turn the ``--resume`` string into a concrete bundle path.

    ``auto`` → the run's canonical rolling bundle, or *start fresh* when it does
    not exist yet. That makes ``--resume auto`` safe to bake into a supervisor /
    restart wrapper: the first launch trains from scratch, every relaunch after a
    crash picks the run back up. An explicit path must exist.
    """
    if value == "auto":
        p = resume_path_for(output_dir, output_name)
        return ResumeArg(path=p if p.exists() else None, explicit=False)
    p = Path(value)
    if not p.exists():
        raise FileNotFoundError(f"--resume: {p} not found")
    return ResumeArg(path=p, explicit=True)


def _check_fingerprint(state: dict[str, Any], cfg) -> None:
    saved = state.get("fingerprint", {})
    now = _fingerprint(cfg)

    broken = [
        (k, saved[k], now[k])
        for k in _STRICT_FIELDS
        if k in saved and k in now and saved[k] != now[k]
    ]
    if broken:
        detail = "\n".join(f"  {k}: bundle={s!r} → config={n!r}" for k, s, n in broken)
        raise ResumeMismatchError(
            "resume bundle was written under a different model topology, so its "
            "tensors do not fit this stack:\n"
            f"{detail}\n"
            "Re-run with the original values, or start a fresh run."
        )

    drifted = [
        (k, saved[k], now[k])
        for k in _WARN_FIELDS
        if k in saved and k in now and saved[k] != now[k]
    ]
    for k, s, n in drifted:
        logger.warning(
            f"resume: {k} changed since the bundle was written: {s!r} → {n!r}"
        )
    if drifted:
        logger.warning(
            "resume: the objective/schedule differs from the interrupted run. The "
            "schedulers resume their step counter onto the NEW curve — intended when "
            "you are deliberately reshaping the run, a mistake otherwise."
        )


def _restore_scheduler(sched, saved: dict[str, Any]) -> None:
    """Put a freshly-built scheduler at the bundle's step count on the NEW curve.

    ``load_state_dict`` would drag the OLD curve along (``T_max``/``eta_min``/
    ``base_lrs``/``SequentialLR`` milestones are all in the state dict) — a
    resume that changes ``iterations``/``student_lr`` would silently keep
    training on the bundle's schedule, and *extending* ``iterations`` steps the
    restored cosine past its old ``T_max``, driving the LR negative. Replaying
    ``step()`` on the new instance instead lands the counter at the same
    position on the CURRENT config's curve (the contract ``_WARN_FIELDS``
    documents).
    """
    import warnings

    # The optimizer's load_state_dict (which runs before this) overwrote each
    # param group's "lr" with the bundle's saved value, and the cosine's
    # recursive step() reads it — reset to the fresh scheduler's own position
    # so the replay walks the new curve from its true start.
    for group, lr in zip(sched.optimizer.param_groups, sched.get_last_lr()):
        group["lr"] = lr

    replay = int(saved.get("last_epoch", 0)) - sched.last_epoch
    with warnings.catch_warnings():
        # The "lr_scheduler.step() before optimizer.step()" order warning is a
        # false positive here: the optimizer's state was already load_state_dict'd.
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(replay):
            sched.step()


def apply_resume_state(
    state: dict[str, Any],
    *,
    cfg,
    turbo,
    student_opt,
    fake_opt,
    disc_opt,
    student_sched,
    fake_sched,
    disc_sched,
    fdistill_bins: torch.Tensor | None,
) -> int:
    """Restore every mutable piece of the loop in-place. Returns the start step.

    The returned step is the number of student steps ALREADY completed, i.e. the
    index the main loop should resume at.
    """
    _check_fingerprint(state, cfg)

    turbo.student.load_state_dict(state["student"])
    turbo.fake.load_state_dict(state["fake"])
    # τ-split critic bank count must match EXACTLY — unlike the disc (whose
    # optimizer is separate, so an extra disc state can be dropped with a
    # warning), both fake banks share ONE optimizer whose state is keyed by the
    # combined param list; a bank-count mismatch would half-load it silently.
    fake_hi = getattr(turbo, "fake_hi", None)
    if fake_hi is not None:
        if state.get("fake_hi") is None:
            raise ResumeMismatchError(
                "config sets fake_tau_banks=2 but the resume bundle carries no "
                "second fake bank (it was written by a single-critic run). "
                "Start fresh, or resume with fake_tau_banks=1."
            )
        fake_hi.load_state_dict(state["fake_hi"])
    elif state.get("fake_hi") is not None:
        raise ResumeMismatchError(
            "resume bundle carries a second fake bank but fake_tau_banks=1 — "
            "the shared fake optimizer state cannot align. Resume with "
            "fake_tau_banks=2."
        )
    if turbo.disc is not None:
        if state.get("disc") is None:
            raise ResumeMismatchError(
                "config enables the GAN disc but the resume bundle has no disc state "
                "(it was written by a run with the GAN off)."
            )
        turbo.disc.load_state_dict(state["disc"])
    elif state.get("disc") is not None:
        logger.warning(
            "resume: bundle carries disc state but the GAN is off — dropped."
        )

    student_opt.load_state_dict(state["student_opt"])
    fake_opt.load_state_dict(state["fake_opt"])
    _restore_scheduler(student_sched, state["student_sched"])
    _restore_scheduler(fake_sched, state["fake_sched"])
    if disc_opt is not None and state.get("disc_opt") is not None:
        disc_opt.load_state_dict(state["disc_opt"])
    if disc_sched is not None and state.get("disc_sched") is not None:
        _restore_scheduler(disc_sched, state["disc_sched"])

    # f-distill's per-τ EMA ratio histogram: a live running statistic, so it has
    # to come back or the first post-resume steps normalize against ones().
    saved_bins = state.get("fdistill_bins")
    if fdistill_bins is not None and saved_bins is not None:
        fdistill_bins.copy_(saved_bins.to(fdistill_bins.device))

    rng = state.get("rng") or {}
    try:
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda"):
            torch.cuda.set_rng_state_all(rng["cuda"])
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
    except Exception as e:  # noqa: BLE001 — RNG restore is best-effort, never fatal
        logger.warning(f"resume: could not restore RNG state ({e}); continuing.")

    return int(state["step"])
