"""Inference entry-points for shipped methods (test / test-* commands).

All variants share ``INFERENCE_BASE`` from ``_common`` and add method-specific
flags. Experimental inference commands (exp-test-soft, exp-test-directedit*)
live in ``scripts/experimental_tasks/inference.py``.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from ._common import (
    INFERENCE_BASE,
    ROOT,
    _random_ref_image,
    _resolve_run_mode,
    latest_hydra,
    latest_lora,
    latest_output,
    override_arg,
    run,
    run_command,
)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _mod_flags() -> list[str]:
    """Resolve latest distilled pooled_text_proj for ``MOD=1``."""
    return ["--pooled_text_proj", str(latest_output("pooled_text_proj"))]


def _base_test_args(*, lora_default: bool = True) -> list[str]:
    """Build the shared ``inference.py`` argv prefix used by every ``test*`` command.

    Honors these env levers so they compose uniformly across ``test`` and
    ``test-smc-cfg``:

    - ``NOLORA=1`` skips ``--lora_weight`` (bare DiT). When unset, ``lora_default``
      decides whether the caller wants a LoRA by default.
    - ``SPECTRUM=1`` appends Spectrum flags. ``SEA=1`` (with SPECTRUM=1) swaps the
      growing-window skip rule for the SeaCache SEA-distance trigger; tune via
      ``SPECTRUM_DELTA=`` (default ``auto``) and ``SPECTRUM_REFRESH_RATIO=``.
    - ``SPD=1`` appends SPD (Spectral Progressive Diffusion) flags. Mutually
      exclusive with ``SPECTRUM=1`` (both replace the denoise loop).
    - ``MOD=1`` appends ``--pooled_text_proj <latest>``.
    - ``DAVE=1`` appends the DAVE DC-attenuation flags (``--dave auto``); tune via
      ``DAVE_STRENGTH=``, ``DAVE_SIGMA='lo,hi'`` and ``DAVE_TAU=`` (early-step cutoff).
    - ``FSG=1`` appends Foresight Guidance pre-step latent calibration (CFG-only);
      tune via ``FSG_BAND='lo,hi'``, ``FSG_K=``, ``FSG_D_SIGMA=``, ``FSG_GAMMA=``.
      Composes with ``SPECTRUM=1`` (incl. ``SEA=1``) — calibrated steps are forced
      to actual forwards. No-op under ``SPD=1`` (it replaces the loop).
    - ``XATTN_BOOST=<λ>`` appends ``--xattn_boost λ`` (front-loaded cross-attn
      gain, cond forward only, σ ≥ band; band via ``XATTN_BOOST_BAND=``,
      default 0.85). Composes with every other lever here.
    """
    args = list(INFERENCE_BASE)
    nolora_env = os.environ.get("NOLORA")
    if nolora_env is None:
        include_lora = lora_default
    else:
        include_lora = not _env_truthy("NOLORA")
    if include_lora:
        args += ["--lora_weight", str(latest_lora())]
    if _env_truthy("SPECTRUM") and _env_truthy("SPD"):
        raise SystemExit(
            "SPECTRUM=1 and SPD=1 are mutually exclusive (both replace the denoise loop)."
        )
    if _env_truthy("SPECTRUM"):
        args += _spectrum_flags()
    if _env_truthy("SPD"):
        args += _spd_flags()
    if _env_truthy("MOD"):
        args += _mod_flags()
    if _env_truthy("DAVE"):
        args += _dave_flags()
    if _env_truthy("FSG"):
        args += _fsg_flags()
    if boost := os.environ.get("XATTN_BOOST", "").strip():
        args += ["--xattn_boost", boost]
        if band := os.environ.get("XATTN_BOOST_BAND", "").strip():
            args += ["--xattn_boost_band", band]
    return args


def _fsg_flags() -> list[str]:
    """FSG pre-step latent calibration (Foresight Guidance). ``FSG_BAND='lo,hi'``,
    ``FSG_K``, ``FSG_D_SIGMA``, ``FSG_GAMMA`` tune the live knobs; all optional.
    Composes with SPECTRUM (incl. SEA); no-op under SPD (it replaces the loop)."""
    flags = ["--fsg"]
    if band := os.environ.get("FSG_BAND", "").strip():
        lo, hi = (x.strip() for x in band.split(","))
        flags += ["--fsg_band", lo, hi]
    if k := os.environ.get("FSG_K", "").strip():
        flags += ["--fsg_k", k]
    if ds := os.environ.get("FSG_D_SIGMA", "").strip():
        flags += ["--fsg_d_sigma", ds]
    if g := os.environ.get("FSG_GAMMA", "").strip():
        flags += ["--fsg_gamma", g]
    return flags


def _dave_flags() -> list[str]:
    """DAVE DC-attenuation (training-free diversity). ``DAVE_STRENGTH``,
    ``DAVE_SIGMA='lo,hi'`` and ``DAVE_TAU`` tune the live knobs; all optional.
    ``DAVE_TAU`` (the paper's early-step cutoff, e.g. 0.15) overrides ``DAVE_SIGMA``."""
    flags = ["--dave", "auto"]
    if s := os.environ.get("DAVE_STRENGTH", "").strip():
        flags += ["--dave_strength", s]
    if win := os.environ.get("DAVE_SIGMA", "").strip():
        lo, hi = (x.strip() for x in win.split(","))
        flags += ["--dave_sigma_lo", lo, "--dave_sigma_hi", hi]
    if tau := os.environ.get("DAVE_TAU", "").strip():
        flags += ["--dave_tau", tau]
    if blk := os.environ.get("DAVE_BLOCKS", "").strip():
        lo, hi = (x.strip() for x in blk.split(","))
        flags += ["--dave_block_lo", lo, "--dave_block_hi", hi]
    return flags


def _spectrum_flags(stop_caching_step: int = 27) -> list[str]:
    flags = [
        "--spectrum",
        "--spectrum_window_size",
        "2.0",
        "--spectrum_flex_window",
        "0.25",
        "--spectrum_warmup",
        "7",
        "--spectrum_w",
        "0.3",
        "--spectrum_m",
        "3",
        "--spectrum_lam",
        "0.1",
        "--spectrum_stop_caching_step",
        str(stop_caching_step),
        "--spectrum_calibration",
        "0.0",
    ]
    # SEA=1 opts into the SeaCache SEA-distance trigger (off by default).
    # SPECTRUM_DELTA (default 'auto') / SPECTRUM_REFRESH_RATIO tune it.
    if _env_truthy("SEA"):
        flags += [
            "--spectrum_schedule",
            "sea",
            "--spectrum_delta",
            os.environ.get("SPECTRUM_DELTA", "auto").strip() or "auto",
        ]
        if rr := os.environ.get("SPECTRUM_REFRESH_RATIO", "").strip():
            flags += ["--spectrum_refresh_ratio", rr]
    return flags


def _spd_flags() -> list[str]:
    """SPD single-late knee: one handoff 0.5 → 1.0 at σ0.7. Override on the CLI
    with --spd_stages / --spd_transition_sigmas (passed via ``extra``)."""
    return [
        "--spd",
        "--spd_stages",
        "0.5",
        "1.0",
        "--spd_transition_sigmas",
        "0.5",
    ]


def cmd_test(extra):
    """Inference with the latest LoRA. See ``_base_test_args`` for env levers."""
    run([*_base_test_args(), *extra])


def cmd_gen(extra):
    """Batch generation routed through the daemon (attach-by-default, Phase 1c).

    Same argv as ``make test`` (shares ``_base_test_args`` — NOLORA / SPECTRUM /
    MOD / DAVE / FSG env levers all compose), but submitted as a GPU command job
    so it **queues behind** a live training run instead of OOM-colliding with it,
    survives the terminal closing, and lands a generation manifest in the job
    record (Phase 1a result-lift). ``--queue`` detaches (overnight seed/ckpt
    sweeps), ``--inline`` bypasses the daemon (identical to ``make test``).

    Point at a specific adapter / prompt file / seed grid via ARGS, e.g.
    ``make gen ARGS="--lora_weight output/ckpt/foo.safetensors --from_file prompts.txt"``.
    """
    mode, extra = _resolve_run_mode(extra)
    # _base_test_args() leads with the python exe (INFERENCE_BASE[0]); run_command
    # prepends the interpreter itself (venv python for the daemon), so drop it.
    argv = [*_base_test_args()[1:], *extra]
    run_command("gen", argv, mode=mode)


def cmd_test_hydra(extra):
    # Uses the moe sibling (router-live); static-merge is auto-skipped in
    # library/inference_pipeline.py:_is_hydra_moe detection.
    run([*INFERENCE_BASE, "--lora_weight", str(latest_hydra()), *extra])


def cmd_test_merge(extra):
    """Inference with a baked (merged) DiT from MODEL_DIR (default 'output_temp').

    MODEL_DIR accepts either a directory (picks the latest
    ``*_merged.safetensors`` inside) or a direct ``.safetensors`` path. The
    merged file is a standalone DiT (LoRA folded in), so no ``--lora_weight``
    is passed. The trailing ``--dit`` overrides the base one in
    ``INFERENCE_BASE`` (argparse keeps the last value).
    """
    target = Path(os.environ.get("MODEL_DIR", "output_temp"))
    if not target.is_absolute():
        target = ROOT / target
    if target.is_file():
        chosen = target
    elif target.is_dir():
        candidates = sorted(
            target.glob("*_merged.safetensors"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(f"No '*_merged.safetensors' files found in {target}", file=sys.stderr)
            sys.exit(1)
        chosen = candidates[0]
    else:
        print(f"MODEL_DIR path not found: {target}", file=sys.stderr)
        sys.exit(1)
    run([*INFERENCE_BASE, "--dit", str(chosen), *extra])


def cmd_test_smc_cfg(extra):
    """Inference with latest LoRA + SMC-CFG (arXiv:2603.03281).

    Production defaults (λ=5, α=0.2). Override via --smc_cfg_lambda /
    --smc_cfg_alpha in extra. Honors SPECTRUM / MOD / NOLORA env levers
    (see ``_base_test_args``).
    """
    run([*_base_test_args(), "--smc_cfg", *extra])


def cmd_test_easycontrol(extra):
    """Inference with latest EasyControl weight.

    Reference image is taken from REF_IMAGE env or the first positional arg.
    Falls back to a random image from ``easycontrol-dataset/`` (the EasyControl
    source layout) when neither is supplied.
    PROMPT, NEG, EC_SCALE env vars override defaults. Saves to
    output/tests/easycontrol/ and copies the ref image alongside the generated
    output as ``<name>_ref.png``.

    ``EASYADAPTER=colorize`` targets the colorization checkpoint
    (``anima_colorize``), saves to output/tests/colorize/, defaults the ref to a
    random image under ``post_image_dataset/resized/`` (feed a real B&W manga page
    via REF_IMAGE), and defaults to an EMPTY prompt (caption-free colorization).

    Examples:
      python tasks.py test-easycontrol ref.png --prompt "a girl in a coffee shop"
      REF_IMAGE=ref.png EC_SCALE=0.8 python tasks.py test-easycontrol
      python tasks.py test-easycontrol         # random ref from easycontrol-dataset/
      REF_IMAGE=manga.png EASYADAPTER=colorize python tasks.py test-easycontrol
    """
    adapter = (os.environ.get("EASYADAPTER") or "").strip()
    # Per-adapter inference table: weight prefix, output subdir, ref fallback dir,
    # and whether to default to an empty prompt. The default (ref==target) row is
    # used when EASYADAPTER is unset or unknown.
    _DEFAULT = {
        "weight": "anima_easycontrol",
        "out": "easycontrol",
        "ref_dir": ROOT / "easycontrol-dataset",
        "empty_prompt": False,
    }
    _ADAPTERS = {
        # colorize: feed a real B&W manga page; empty prompt = caption-free colorize.
        "colorize": {
            "weight": "anima_colorize",
            "out": "colorize",
            "ref_dir": ROOT / "post_image_dataset" / "resized",
            "empty_prompt": True,
        },
        # inpaint: ref is the already-masked (gray-holed) PNG; a caption steers the fill.
        "inpaint": {
            "weight": "anima_inpaint",
            "out": "inpaint",
            "ref_dir": ROOT / "post_image_dataset" / "resized",
            "empty_prompt": False,
        },
        # phash_edit: ref is the source image and the prompt is an EDIT
        # INSTRUCTION in the delta grammar ("glasses, -hat"), not a description
        # -- shared tags cancel out of a delta, so identity rides the cond
        # stream. Never force an empty prompt: empty is the identity/no-op arm's
        # caption and returns a copy of the ref.
        "phash_edit": {
            "weight": "anima_easycontrol_phash_edit",
            "out": "phash_edit",
            "ref_dir": ROOT / "post_image_dataset" / "resized",
            "empty_prompt": False,
        },
        # region: ref is a white canvas + paint blob marking where the character
        # goes; the prompt owns identity/scene. Staged conds under
        # post_image_dataset/easycontrol/region/cond_images/ make ready refs.
        "region": {
            "weight": "anima_easycontrol_region",
            "out": "region",
            "ref_dir": ROOT / "post_image_dataset" / "easycontrol" / "region" / "cond_images",
            "empty_prompt": False,
        },
        # subject: ref is a DIFFERENT image of the character to retrieve; the
        # prompt owns layout/pose, so never force an empty prompt.
        "subject": {
            "weight": "anima_easycontrol_subject",
            "out": "subject",
            "ref_dir": ROOT / "post_image_dataset" / "resized",
            "empty_prompt": False,
        },
    }
    spec = _ADAPTERS.get(adapter, _DEFAULT)
    weight_name = spec["weight"]
    out_sub = spec["out"]
    ref_fallback_dir = spec["ref_dir"]

    ref_image = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image and extra and not extra[0].startswith("-"):
        ref_image = extra[0]
        extra = extra[1:]
    if not ref_image:
        ref_image = _random_ref_image(ref_fallback_dir) or ""
    if not ref_image:
        print(
            "Usage: python tasks.py test-easycontrol <ref_image> [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py test-easycontrol [extra...]\n"
            f"   (no ref given and {ref_fallback_dir.name}/ is empty)",
            file=sys.stderr,
        )
        sys.exit(1)

    save_dir = ROOT / "output" / "tests" / out_sub
    save_dir.mkdir(parents=True, exist_ok=True)

    args = [
        *INFERENCE_BASE,
        "--save_path",
        str(save_dir),
        "--easycontrol_weight",
        str(latest_output(weight_name)),
        "--easycontrol_image",
        ref_image,
        "--easycontrol_image_match_size",
    ]
    if scale := os.environ.get("EC_SCALE"):
        args += ["--easycontrol_scale", scale]
    if prompt := os.environ.get("PROMPT"):
        args += ["--prompt", prompt]
    elif spec["empty_prompt"] and not any(a == "--prompt" for a in extra):
        # caption-free default (empty prompt → uncond text path), e.g. colorize
        args += ["--prompt", ""]
    if neg := os.environ.get("NEG"):
        args += ["--negative_prompt", neg]
    args += list(extra)
    run(args)

    pngs = sorted(
        (p for p in save_dir.glob("*.png") if not p.name.endswith("_ref.png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pngs:
        ref_dst = pngs[0].with_name(pngs[0].stem + "_ref.png")
        shutil.copy(ref_image, ref_dst)
        print(f"  > Ref pasted: {ref_dst}")


def cmd_test_turbo(extra):
    """Inference with the latest Turbo student LoRA at 4 steps, cfg=1.0.

    CFG is baked into the student during distillation, so production inference
    runs cfg=1.0 (no double-CFG). Step count defaults to 4 — matching the
    DP-DMD student's ``student_steps=4`` rollout — but extra args can override.
    A ready-made student ships at https://huggingface.co/sorryhyun/anima-turbo-4step
    — point ``--lora_weight`` at a download to try it without distilling.
    """
    weight = latest_output("anima_turbo")
    base = list(INFERENCE_BASE)
    base = override_arg(base, "--sampler", "euler")
    # Per-step-expert checkpoints bind head k to denoise step k, so infer_steps
    # MUST equal the trained head count K (= student_steps); pin it from metadata.
    infer_steps = "4"
    try:
        from safetensors import safe_open

        with safe_open(str(weight), framework="pt") as f:
            md = f.metadata() or {}
        if str(md.get("ss_turbo_per_step_expert", "")).strip() in ("1", "true", "True"):
            K = int(md.get("ss_turbo_step_expert_K", "4") or "4")
            infer_steps = str(K)
            print(
                f"[test-turbo] per-step-expert checkpoint: pinning "
                f"--infer_steps {K} (= trained head count). Override at your own "
                "risk — heads beyond K repeat the last (quality) head."
            )
    except Exception:
        pass
    base = override_arg(base, "--infer_steps", infer_steps)
    base = override_arg(base, "--guidance_scale", "1.0")
    run(
        [
            *base,
            "--lora_weight",
            str(weight),
            *extra,
        ]
    )
