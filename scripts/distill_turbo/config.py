"""Turbo distillation config: one field table → argparser, TOML resolver, snapshot.

Each ``TurboConfig`` field is declared once with :func:`knob`: its dotted key in the
sectioned TOML (``configs/methods/turbo.toml``), its default and its CLI help. Every
field is a ``--<name>`` flag; precedence is CLI → TOML → default. Knob docs:
``docs/anima/methods/turbo.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from library.config.io import toml_get
from library.config.resolved import dataclass_snapshot_toml, dataclass_tb_text

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/methods/turbo.toml"


def knob(toml: str | None, default: Any, help: str | None = None, *, choices=None):
    """A config field read from dotted TOML key ``toml`` (None: CLI only)."""
    return field(
        default=default, metadata={"toml": toml, "help": help, "choices": choices}
    )


@dataclass(frozen=True)
class TurboConfig:
    # Paths / IO
    dit_path: str = knob(
        "dit_path", "models/diffusion_models/anima-base-v1.0.safetensors"
    )
    dit_nonquant_path: str | None = knob(
        "dit_nonquant_path",
        None,
        "The bf16 DiT an int8 --dit_path was quantized from (the float weights wq_t "
        "is built from); unset, they are rebuilt from the int8 copies.",
    )
    data_dir: str = knob("data_dir", "post_image_dataset/lora")
    output_dir: str = knob("output_dir", "output/ckpt")
    output_name: str = knob("output_name", "anima_turbo")
    log_dir: str = knob("io.log_dir", "output/logs/turbo")
    save_every: int = knob("io.save_every", 1000)
    resume: str = knob(
        None,
        "",
        "Crash-resume bundle (nets, optimizers, schedulers, RNG). 'auto' uses "
        "output/ckpt/<name>/<name>_resume.pt if present, else starts fresh; a named "
        "path must exist. Written every --save_every.",
    )
    log_interval: int = knob("io.log_interval", 2)
    no_log: bool = knob(None, False)

    # Diversity validation (DAVE same-prompt probe)
    validate_every_n_steps: int = knob(
        "io.validate_every_n_steps", 0, "Diversity probe every N steps (0 off)."
    )
    val_diversity_seeds: int = knob(
        "io.val_diversity_seeds", 8, "Seeds the diversity probe rolls (>=2)."
    )
    val_prompt_idx: int = knob(
        "io.val_prompt_idx",
        -1,
        "Held-out sample the probe conditions on (-1: the last one, not "
        "--single_prompt_idx).",
    )

    # Run shape
    iterations: int = knob("iterations", 20000)
    batch_size: int = knob("batch_size", 1)
    seed: int = knob("seed", 42)
    sample_ratio: float = knob(None, 1.0)
    single_prompt_idx: int | None = knob(
        None, None, "Overfit mode: pin the dataloader to one (latent, text) pair."
    )

    # LoRA stacks
    student_rank: int = knob("network.student_rank", 48)
    fake_rank: int = knob("network.fake_rank", 48)
    student_alpha: float | None = knob(
        "network.student_alpha", None, "Unset: student_rank."
    )
    fake_alpha: float | None = knob("network.fake_alpha", None, "Unset: fake_rank.")
    attn_mode: str = knob("network.attn_mode", "flash")
    channel_scaling_alpha: float = knob(
        "channel_scaling_alpha",
        0.0,
        "Per-input-channel rebalance absorbed into lora_down (student and fake); "
        "0 off, 0.5 sqrt-balance. Merges out at save.",
    )
    student_down_init: str = knob(
        "network.student_down_init", "kaiming", choices=("kaiming", "weight_svd")
    )
    fake_down_init: str = knob(
        "network.fake_down_init", "kaiming", choices=("kaiming", "weight_svd")
    )
    student_init_weights: str = knob(
        "network.student_init_weights",
        "",
        "Plain LoRA file seeding the student's ΔW (SVD-truncated to its rank).",
    )
    fake_init_weights: str = knob("network.fake_init_weights", "")
    train_adaln: bool = knob(
        "network.train_adaln",
        False,
        "adaln_up_{branch} targets; the student ships them in the ComfyUI layout.",
    )
    fake_adaln: bool | None = knob(
        "network.fake_adaln",
        None,
        "Unset: train_adaln. False drops the fake's 84 adaln modules (~0.8 GB at "
        "rank 96; unbenched).",
    )
    adaln_rank: int = knob(
        "network.adaln_rank", 0, "Own rank for the adaln modules (0 = network's)."
    )
    adaln_alpha: float = knob(
        "network.adaln_alpha",
        0.0,
        "Own alpha for the adaln modules (0 = network's). Scale-preserving: "
        "student_alpha × adaln_rank / student_rank.",
    )
    partitioner_recompute_views: bool = knob("partitioner_recompute_views", False)
    partitioner_aggressive_recomputation: bool = knob(
        "partitioner_aggressive_recomputation", False
    )

    # DP-DMD (arXiv:2602.03139)
    k_anchor: int = knob(
        "dpdmd.k_anchor",
        7,
        "Teacher steps rolled to the anchor (their K). The target is the teacher's "
        "mean velocity over [σ_K, 1]; step 0 applies it over [σ_1, 1], so K should "
        "end on the student's σ_1 (4 steps at shift 3: 7 of 28). A mismatch is "
        "logged as a warning.",
    )
    teacher_anchor_steps: int = knob(
        "dpdmd.teacher_anchor_steps", 28, "Teacher σ grid the K anchor is counted on."
    )
    div_weight: float = knob(
        "dpdmd.div_weight",
        5e-2,
        "λ on the step-0 anchor loss; 0 skips the teacher anchor.",
    )
    anchor_only: bool = knob(
        "dpdmd.anchor_only",
        False,
        "Train step 0 on the anchor regression alone: no rollout, no DM, no fake "
        "update or warmup.",
    )
    anchor_cache: str = knob(
        "dpdmd.anchor_cache",
        "",
        "Anchor target cache dir: train on the cached samples of data_dir, reading "
        "v_target and ε from the cache. Built by "
        "`python -m scripts.distill_turbo.anchor_cache`.",
    )
    flow_shift: float = knob(
        "sampling.flow_shift",
        3.0,
        "σ shift of the student and teacher Euler grids (as at inference).",
    )

    # DMD core
    student_steps: int = knob(
        "dmd.student_steps", 4, "Sampler step count baked into the student."
    )
    teacher_cfg: float = knob("dmd.teacher_cfg", 4.0)
    dm_x0_norm: bool = knob(
        "dmd.dm_x0_norm",
        False,
        "Replace τ-damping with per-sample x0-space normalisation: "
        "τ·Δ / clamp(τ·mean|v_real|, norm_floor), so τ cancels.",
    )
    norm_floor: float = knob(
        "dmd.norm_floor", 0.05, "Floor of the --dm_x0_norm denominator."
    )

    # Optimizer + scheduler
    student_lr: float = knob("optim.student_lr", 1e-5)
    fake_lr: float = knob("optim.fake_lr", 1e-5)
    fake_steps_per_student_step: int = knob(
        "optim.fake_steps_per_student_step",
        1,
        "Fake updates per student step, each on fresh (τ, ε).",
    )
    fake_warmup_steps: int = knob(
        "optim.fake_warmup_steps",
        0,
        "Fake-only head-start updates before the main loop. Without it the "
        "warm-start fake is miscalibrated when the student reaches full LR "
        "(grad_signal_rms spike near step 50).",
    )
    grad_accum: int = knob(
        "optim.grad_accum",
        1,
        "Loop iterations per student optimizer step; each iteration's gradient is "
        "clipped on its own and their mean applied. The fake updates every "
        "iteration; iterations, save_every and the logs count iterations.",
    )
    weight_decay: float = knob("optim.weight_decay", 0.0)
    grad_clip: float = knob("optim.grad_clip", 1.0)
    lr_schedule: str = knob(
        "optim.lr_schedule",
        "cosine",
        "Shape after the 2%% linear warmup, for student and fake: cosine to "
        "0.1·lr, or constant.",
        choices=("cosine", "constant"),
    )
    ema_window: float = knob(
        "ema.window",
        0.0,
        "Student EMA span as a fraction of the run (0 off); saves a "
        "*_ema.safetensors beside each checkpoint.",
    )

    # Fake τ distribution
    t_distribution: str = knob(
        "sampling.t_distribution", "uniform", choices=("uniform", "sigmoid")
    )
    sigmoid_scale: float = knob("sampling.sigmoid_scale", 1.0)

    # Runtime
    blocks_to_swap: int = knob(None, 0)
    grad_ckpt: bool = knob(None, False)
    torch_compile: bool = knob(None, True, "Compile the DiT blocks.")
    compile_dynamic_seq: bool = knob(
        "compile_dynamic_seq",
        True,
        "One block graph with a dynamic sequence axis instead of one per token count.",
    )
    target_res: list[int] | None = knob(
        "target_res",
        None,
        "Tier edges that size the dynamic-seq bound and the dynamo budget; unset, "
        "they come from the token counts in the cached pool.",
    )
    dynamo_recompile_limit: int = knob("dynamo_recompile_limit", 64)
    activation_memory_budget: float = knob(
        "activation_memory_budget",
        1.0,
        "Compile partitioner saved-activation fraction (<1 recomputes in backward). "
        "Ignored under --grad_ckpt.",
    )
    int8: str = knob(
        "int8",
        "off",
        "Frozen DiT linears as int8 GEMMs, teacher included "
        "(docs/anima/optimizations/int8.md).",
        choices=("off", "fwd", "full", "stream"),
    )
    int8_file: str | None = knob(
        "int8_file",
        None,
        "GPTQ forward copies; unset: <dit>_int8_gptq.safetensors beside the DiT, "
        "else round to nearest; 'rtn' forces round to nearest.",
    )
    stream: str = knob(
        "stream",
        "fp32",
        "Residual stream precision, teacher included "
        "(docs/anima/optimizations/fp32_stream.md).",
        choices=("bf16", "fp32"),
    )


_FIELDS = {f.name: f for f in dataclasses.fields(TurboConfig)}


def _scalar_type(f: dataclasses.Field) -> type:
    base = str(f.type).split("|")[0].strip()
    return {"int": int, "float": float, "bool": bool, "list[int]": int}.get(base, str)


def build_argparser() -> argparse.ArgumentParser:
    """``--config`` plus one flag per field; an absent flag reads None (→ TOML)."""
    parser = argparse.ArgumentParser(
        description="Turbo Anima — DP-DMD distillation (CLI flags override the TOML)."
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    for name, f in _FIELDS.items():
        kind = _scalar_type(f)
        kw: dict[str, Any] = {"default": None, "help": f.metadata["help"]}
        if kind is bool:
            kw["action"] = argparse.BooleanOptionalAction
        else:
            kw["type"] = kind
            kw["choices"] = f.metadata["choices"]
            if str(f.type).startswith("list"):
                kw["nargs"] = "+"
        parser.add_argument(f"--{name}", **kw)
    # Older spelling of the negation.
    parser.add_argument(
        "--no_grad_ckpt", dest="grad_ckpt", action="store_false", default=None
    )
    return parser


# Keys of removed features → the only value the loop still runs; any other value
# is refused rather than silently ignored.
_REMOVED_KEYS = {
    "base_loss": "dpdmd",
    "dmd.grad_step": "random",
    "network.per_step_expert": False,
    "network.dual_pool": False,
    "network.fake_tau_banks": 1,
    "use_masked_loss": False,
    "softrank.weight": 0.0,
    "f_distill.f_div": "rkl",
    "dmad.enabled": False,
    "dmad_probe.enabled": False,
    "use_custom_down_autograd": False,
    "network.fake_transition": False,
    "dpdmd.detach_after_first": True,
    "dpdmd.dm_first_step": False,
    "dpdmd.rollout_grad": False,
    "dpdmd.anchor_draws": 1,
    "dpdmd.fm_weight": 0.0,
    "dpdmd.teacher_handoff": False,
    "dmd.dm_draws": 1,
    "dmd.dm_tau_distribution": "uniform",
    "dmd.dynamic_schedule": False,
    "cdm.weight": 0.0,
    "mfd.dtau": 0.0,
    "gan.weight_gen": 0.0,
}


def _coerce(f: dataclasses.Field, value: Any) -> Any:
    if value is None:
        return None
    if str(f.type).startswith("list"):
        return [int(v) for v in value]
    return _scalar_type(f)(value)


def resolve_config(args: argparse.Namespace, cfg: dict) -> TurboConfig:
    """CLI → TOML → field default, then derived defaults and validation."""
    for key, allowed in _REMOVED_KEYS.items():
        value = toml_get(cfg, key, allowed)
        if value != allowed:
            raise ValueError(
                f"{key} = {value!r}: this feature was removed from the turbo loop "
                f"(only {allowed!r} runs). Drop the key."
            )

    values: dict[str, Any] = {}
    for name, f in _FIELDS.items():
        cli = getattr(args, name, None)
        if cli is not None:
            value = cli
        elif f.metadata["toml"] is not None:
            value = toml_get(cfg, f.metadata["toml"], f.default)
        else:
            value = f.default
        values[name] = _coerce(f, value)

    if values["student_alpha"] is None:
        values["student_alpha"] = float(values["student_rank"])
    if values["fake_alpha"] is None:
        values["fake_alpha"] = float(values["fake_rank"])
    if values["fake_adaln"] is None:
        values["fake_adaln"] = values["train_adaln"]
    if values["anchor_only"] and values["fake_warmup_steps"]:
        logger.info(
            f"dpdmd.anchor_only: fake_warmup_steps {values['fake_warmup_steps']} → 0 "
            "(the fake is never queried)."
        )
        values["fake_warmup_steps"] = 0

    c = TurboConfig(**values)
    _validate(c)
    _log_summary(c)
    return c


def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


def _validate(c: TurboConfig) -> None:
    for name, f in _FIELDS.items():
        choices = f.metadata["choices"]
        value = getattr(c, name)
        _require(
            choices is None or value in choices,
            f"{f.metadata['toml'] or name}={value!r}: expected one of {choices}",
        )

    for name in ("student", "fake"):
        path = getattr(c, f"{name}_init_weights")
        _require(
            not path or getattr(c, f"{name}_down_init") == "kaiming",
            f"network.{name}_init_weights is incompatible with {name}_down_init="
            "'weight_svd': the warm start overwrites what it seeds.",
        )
        _require(
            not path or os.path.exists(path),
            f"network.{name}_init_weights: {path!r} not found.",
        )
    _require(
        c.adaln_rank >= 0 and c.adaln_alpha >= 0,
        "network.adaln_rank / adaln_alpha must be >= 0",
    )
    _require(
        c.train_adaln or (c.adaln_rank == 0 and c.adaln_alpha == 0),
        "network.adaln_rank / adaln_alpha > 0 require network.train_adaln = true",
    )

    _require(
        c.student_steps >= 2,
        f"dmd.student_steps must be >= 2 (got {c.student_steps}): the grad-step "
        "draw assumes a refinement step after step 0.",
    )
    _require(
        1 <= c.k_anchor < c.teacher_anchor_steps,
        f"dpdmd.k_anchor={c.k_anchor} must satisfy 1 <= k_anchor < "
        f"teacher_anchor_steps={c.teacher_anchor_steps}.",
    )
    _require(c.div_weight >= 0.0, f"dpdmd.div_weight={c.div_weight}: must be >= 0")
    _require(
        not c.anchor_cache or c.div_weight > 0.0,
        "dpdmd.anchor_cache needs div_weight > 0 (div_weight 0 skips the anchor).",
    )
    _require(
        not c.anchor_only or c.div_weight > 0.0,
        "dpdmd.anchor_only needs div_weight > 0 (the anchor is computed only then; "
        "its value is not used).",
    )
    _require(c.flow_shift > 0.0, f"sampling.flow_shift={c.flow_shift}: must be > 0")

    _require(
        c.norm_floor > 0.0, f"dmd.norm_floor={c.norm_floor}: must be > 0 (latent scale)"
    )
    _require(c.grad_accum >= 1, f"optim.grad_accum={c.grad_accum}: must be >= 1")
    _require(
        c.fake_steps_per_student_step >= 1,
        f"optim.fake_steps_per_student_step={c.fake_steps_per_student_step}: "
        "must be >= 1",
    )
    _require(
        0.0 <= c.ema_window <= 1.0, f"ema.window={c.ema_window}: must be in [0, 1]"
    )
    _require(
        c.single_prompt_idx is None or c.batch_size == 1,
        f"--single_prompt_idx requires batch_size=1 (got {c.batch_size}): a "
        "one-sample dataset with drop_last yields zero batches.",
    )


def _log_summary(c: TurboConfig) -> None:
    if c.fake_rank < c.student_rank:
        logger.warning(
            f"fake_rank={c.fake_rank} < student_rank={c.student_rank}: the fake has "
            "less capacity than the student it tracks."
        )
    if c.div_weight == 0.0:
        logger.warning("dpdmd.div_weight=0: step 0 gets no loss and stays at its init.")
    logger.info(
        "DM gradient policy: "
        + (
            f"x0-norm, norm_floor={c.norm_floor} (τ cancels)"
            if c.dm_x0_norm
            else "τ-damping"
        )
    )
    logger.info(
        (
            f"DP-DMD: step-0 anchor k_anchor={c.k_anchor}/"
            f"{c.teacher_anchor_steps} teacher steps, div_weight={c.div_weight}, "
            if c.div_weight > 0.0
            else "teacher anchor OFF (div_weight=0), "
        )
        + ("anchor only (no DM), " if c.anchor_only else "grad step g ~ U{1..N-1}, ")
        + f"student N={c.student_steps} @ flow_shift={c.flow_shift}, "
        f"teacher_cfg={c.teacher_cfg}."
    )


def snapshot_toml_text(c: TurboConfig, *, source_config: str | None = None) -> str:
    """Every resolved field as a TOML snapshot (train.py's ``.snapshot.toml``)."""
    return dataclass_snapshot_toml(
        c,
        title="Anima turbo distillation — resolved config snapshot",
        source_config=source_config,
    )


_TB_KEYS = (
    "k_anchor",
    "teacher_anchor_steps",
    "div_weight",
    "anchor_only",
    "anchor_cache",
    "flow_shift",
    "student_rank",
    "fake_rank",
    "channel_scaling_alpha",
    "student_steps",
    "teacher_cfg",
    "fake_warmup_steps",
    "student_lr",
    "fake_lr",
    "lr_schedule",
    "ema_window",
    "fake_steps_per_student_step",
    "grad_accum",
    "iterations",
    "batch_size",
    "t_distribution",
    "data_dir",
    "dit_path",
)


def tb_config_text(c: TurboConfig) -> str:
    """TensorBoard config summary over ``_TB_KEYS``."""
    return dataclass_tb_text(c, include=_TB_KEYS)
