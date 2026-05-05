#!/usr/bin/env python
"""Direct SNR-t bias measurement for Anima (flow-matching DiT).

Reproduces Fig. 1c of Yu et al. *Elucidating the SNR-t Bias of Diffusion
Probabilistic Models* (arXiv:2604.16044) on Anima and dumps the
per-(sample, step) baseline LL/LH/HL/HH gap arrays consumed by the
dcw-learnable-calibrator analysis phase (transfer hypothesis, PCA,
S_pop sensitivity profile, overshoot guard).

Reads cached samples from ``post_image_dataset/lora`` (latents +
post-LLMAdapter text embeds), so no VAE / T5 loading is needed.

Measurement
-----------
For each timestep ``i`` in the inference schedule:

    v_fwd(i) = || v_θ((1 − σ_i)·x_0 + σ_i·ε, σ_i) ||
    v_rev(i) = || v_θ(x̂_i, σ_i) ||
    gap(i)   = v_rev(i) − v_fwd(i)

Per-Haar-subband variants come from a single-level orthonormal 2D Haar
DWT on the velocity tensor's (H, W) plane. v2's controller acts on the
LL band only; LH/HL/HH are recorded so the LL-only assumption can be
verified (paper §5.3, dcw-learnable-calibrator-v2 §"What this is not").

Modes
-----
- **Diagnostic (default)**: baseline only. Pair with
  ``--dump_per_sample_gaps`` to emit ``gaps_per_sample.npz`` for the
  transfer-hypothesis / PCA / S_pop analysis scripts.
- **--dcw_sweep**: also runs reverse trajectories with LL-only DCW
  correction (one_minus_sigma schedule) at a grid of λ values. Used by
  A4 to estimate per-step λ-sensitivity ``S_pop(σ_i)``.

Outputs (bench/dcw/results/<YYYYMMDD-HHMM>[-<label>]/)
------------------------------------------------------
    result.json            standard envelope (args, git, env, metrics, artifacts)
    per_step.csv           wide: step, σ_i, v_fwd / v_rev / gap per config
    per_step_bands.csv     same as per_step.csv but split by Haar subband
    gap_curves.png         (1×3) Fig 1c reproduction, gap overlay across
                           configs, baseline gap broken out by subband
    gaps_per_sample.npz    optional, --dump_per_sample_gaps; per-(traj, step)
                           baseline LL/LH/HL/HH gap arrays

Usage
-----
    # A1: production-env baseline (CFG=4, 28 steps, mod-on by default)
    uv run python scripts/dcw/measure_bias.py \\
        --dit models/diffusion_models/anima-preview3-base.safetensors \\
        --infer_steps 28 --n_images 48 --n_seeds 2 \\
        --guidance_scale 4.0 \\
        --dump_per_sample_gaps --label v2-prod-env

    # A4: λ-sweep for sensitivity profile S_pop(σ_i)
    uv run python scripts/dcw/measure_bias.py \\
        --dit ... --infer_steps 28 --guidance_scale 4.0 \\
        --dcw_sweep --dcw_scalers 0 -0.015 -0.020 -0.025 \\
        --label v2-S_pop

Caveats
-------
- CFG: ``--guidance_scale`` defaults to 4.0 (production env). Setting
  ``--guidance_scale > 1`` live-encodes the unconditional embed via the
  same transient text-encoder block that mod-guidance uses (default
  ``--negative_prompt ""`` mirrors ``inference.py``) and runs the
  cond+uncond pair as a single **batched** DiT forward per step,
  combining as ``v_uncond + s · (v_cond - v_uncond)``. Adds ~30-50% wall
  time vs CFG=1 (the batched path; two-separate-forwards would be ~2×).
  Cached ``_anima_te.safetensors`` sidecars are still cond-only — uncond
  is encoded once at startup and reused across every prompt.

Speed notes
-----------
The hot loop fuses four speedups vs the v1 implementation:

1. **Batched CFG.** Cond and uncond run as a single forward at batch=2·B
   (see ``_cfg_velocity``).
2. **Batched seed sweep.** Forward branch always batches all
   ``--n_seeds`` trajectories per image into one DiT call at batch=N_seeds
   (or 2·N_seeds under CFG > 1). Reverse branch does the same when
   ``--dcw_sweep`` is off (the ``make dcw`` path), running λ=0 across
   all seeds in parallel. See ``measure_forward_norms`` /
   ``run_reverse_batched``.
3. **Batched λ sweep.** When ``--dcw_sweep`` is set, all configured λ
   trajectories share the same step's DiT forward at batch=N_λ (or
   2·N_λ under CFG > 1). Mutually exclusive with seed-batching on the
   reverse branch — λ-sweep mode keeps the per-seed outer loop.
4. **GPU-resident norm accumulation.** Per-step ``‖v‖`` and Haar-band
   norms accumulate on-device; one ``.cpu()`` sync at trajectory end
   instead of 5 syncs per step.

Combined, the prod-env A1 / A4 runs land in ~30-45 min on a 5060 Ti at
1024² (down from several hours at v1 cadence).
- Mod guidance: ON by default with the production-baseline
  ``output/ckpt/pooled_text_proj-0429.safetensors`` checkpoint
  (delta = proj(pos) − proj(neg), schedule applied on blocks
  [mod_start_layer, mod_end_layer)). Setup loads T5 transiently to encode
  the pos/neg prompts, then frees it before the bench loop. Pass
  ``--pooled_text_proj ''`` (empty) for the base-DiT calibration target.
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima import weights as anima_utils  # noqa: E402
from library.inference import sampling as inference_utils  # noqa: E402
from library.inference.adapters import clear_hydra_sigma  # noqa: E402
from scripts.dcw.adapters import attach_loras  # noqa: E402
from scripts.dcw.cache import load_cached, pick_cached_samples  # noqa: E402
from scripts.dcw.haar import BANDS  # noqa: E402
from scripts.dcw.output import (  # noqa: E402
    _accumulate_row,
    make_plot,
    print_summary,
    write_per_band_csv,
    write_per_step_csv,
)
from scripts.dcw.trajectory import (  # noqa: E402
    encode_uncond_embed,
    measure_forward_norms,
    run_reverse_batched,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dcw-bench")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dit",
        type=str,
        default="models/diffusion_models/anima-preview3-base.safetensors",
        help="DiT .safetensors path (default: anima-preview3-base).",
    )
    p.add_argument(
        "--lora_weight",
        type=str,
        nargs="+",
        default=None,
        help="Optional LoRA / HydraLoRA adapter(s) to stack on the base DiT. "
        "Auto-detects HydraLoRA moe (lora_ups.* keys) and attaches router-live "
        "via dynamic forward hooks; plain LoRA goes through the same dynamic path "
        "(math-equivalent to static merge for this measurement).",
    )
    p.add_argument(
        "--lora_multiplier",
        type=float,
        nargs="+",
        default=[1.0],
        help="Multiplier per --lora_weight entry (broadcast if a single value).",
    )
    p.add_argument(
        "--dataset_dir",
        type=str,
        default="post_image_dataset/lora",
        help="Directory with cached *_anima.npz + *_anima_te.safetensors pairs.",
    )
    p.add_argument(
        "--text_variant",
        type=int,
        default=0,
        help="Cached caption variant (crossattn_emb_v<N>); 0 = canonical.",
    )
    p.add_argument(
        "--attn_mode",
        type=str,
        default="flash",
        help="torch | sdpa | xformers | sage | flash",
    )
    p.add_argument("--n_images", type=int, default=8, help="Cached samples to use")
    p.add_argument("--n_seeds", type=int, default=1, help="Seeds per sample")
    p.add_argument(
        "--image_h",
        type=int,
        default=None,
        help="Restrict to cached samples with this image-space height (the "
        "<H> in <stem>_<H>x<W>_anima.npz). Required (with --image_w) for "
        "--compile to converge to a single graph and for direct cross-run "
        "comparability of velocity norms.",
    )
    p.add_argument(
        "--image_w",
        type=int,
        default=None,
        help="Restrict to cached samples with this image-space width.",
    )
    p.add_argument(
        "--shuffle_seed",
        type=int,
        default=None,
        help="Deterministically shuffle the candidate pool before truncating "
        "to --n_images. Default None preserves alphabetical-first selection "
        "(legacy behavior). Used by `make dcw` to broaden prompt diversity "
        "beyond the alphabetical-first 35 stems per bucket.",
    )
    p.add_argument(
        "--infer_steps",
        type=int,
        default=28,
        help="Inference schedule length (v2 prod env = 28).",
    )
    p.add_argument(
        "--flow_shift", type=float, default=1.0, help="σ shift (matches inference.py)."
    )
    p.add_argument("--seed_base", type=int, default=1234)
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile the DiT before the bench loop. Each unique latent "
        "(H, W) pays a one-time warm-up; steady-state is much faster. Best "
        "amortized when n_images is moderate (every sample's ~150+ forwards "
        "run at the same shape, and dynamo auto-flips to dynamic shapes after "
        "the second distinct (H, W)). Pass --no-compile to disable.",
    )
    p.add_argument(
        "--dcw_sweep",
        action="store_true",
        help="Also run LL-only DCW-corrected trajectories at --dcw_scalers "
        "(one_minus_sigma schedule). Used by v2 §A4 to estimate S_pop(σ_i).",
    )
    p.add_argument(
        "--dcw_scalers",
        type=float,
        nargs="+",
        default=[0.010],
        help="λ values to sweep when --dcw_sweep is set (negative on Anima; "
        "v2 §A4 uses {0, -0.015, -0.020, -0.025}).",
    )
    p.add_argument(
        "--dump_per_sample_gaps",
        action="store_true",
        help="Dump per-(traj, step) baseline LL/LH/HL/HH gap arrays "
        "(shape (n_images*n_seeds, n_steps)) to gaps_per_sample.npz. "
        "Consumed by the dcw-learnable-calibrator analysis scripts "
        "(transfer hypothesis, PCA, S_pop).",
    )
    p.add_argument(
        "--save_images",
        action="store_true",
        help="Decode the final reverse-trajectory latent for each "
        "(sample, seed, config) row and save as PNG under "
        "<run_dir>/images/. Loads the VAE transiently after the bench "
        "loop completes, decodes one row at a time, frees the VAE. "
        "Lets you visually compare baseline vs. each --dcw_scalers config "
        "at matched (sample, seed) — i.e., actually see whether a "
        "gap-narrowing λ improves perceptual quality, not just integrated "
        "|gap|. ~25 MB extra peak RAM at 832×1248 × 48 rows.",
    )
    p.add_argument(
        "--vae",
        type=str,
        default="models/vae/qwen_image_vae.safetensors",
        help="VAE path used by --save_images to decode final latents.",
    )
    p.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale. 1.0 = single conditional "
        "forward (matches v1 calibration). >1 live-encodes the "
        "unconditional embed at startup and runs an extra DiT forward "
        "per step, combining as v_uncond + s · (v_cond − v_uncond) "
        "(matches inference.py). v2 §A1 production env = 4.0.",
    )
    p.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Unconditional prompt for CFG > 1 (default '' matches "
        "inference.py default).",
    )

    # Modulation guidance (off by default — base-DiT calibration target).
    # When --pooled_text_proj is set, mirrors inference.py's mod-guidance
    # pipeline so v2 §A1 can run a production-mod-on cross-check.
    g_mod = p.add_argument_group("modulation guidance (optional)")
    g_mod.add_argument(
        "--pooled_text_proj",
        type=str,
        default="",
        help="Path to trained pooled_text_proj weights (.safetensors). "
        "Default enables modulation guidance with the production-baseline "
        "0429 checkpoint and the pos/neg prompts below. Pass an empty "
        "string (--pooled_text_proj '') to disable for the base-DiT "
        "calibration measurement.",
    )
    g_mod.add_argument(
        "--text_encoder",
        type=str,
        default="models/text_encoders/qwen_3_06b_base.safetensors",
        help="Qwen3 text encoder path; only loaded when mod guidance is on, "
        "freed after the steering delta is computed.",
    )
    g_mod.add_argument("--mod_w", type=float, default=3.0)
    g_mod.add_argument(
        "--mod_pos_prompt", type=str, default="absurdres, masterpiece, score_9"
    )
    g_mod.add_argument(
        "--mod_neg_prompt",
        type=str,
        default="worst quality, low quality, score_1, score_2, score_3",
    )
    g_mod.add_argument("--mod_start_layer", type=int, default=8)
    g_mod.add_argument("--mod_end_layer", type=int, default=27)
    g_mod.add_argument("--mod_taper", type=int, default=0)
    g_mod.add_argument("--mod_taper_scale", type=float, default=0.25)
    g_mod.add_argument("--mod_final_w", type=float, default=0.0)

    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label appended to the run dir (<out_root>/<ts>-<label>/).",
    )
    p.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="Override the run-dir root. Default bench/dcw/results/. "
        "`make dcw` redirects to output/dcw/ since calibration "
        "trajectories are runtime artifacts, not published bench results.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if (args.image_h is None) != (args.image_w is None):
        raise SystemExit(
            "--image_h and --image_w must be set together (or both omitted)."
        )
    out_dir = make_run_dir("dcw", label=args.label, root=args.out_root)
    log.info(f"output → {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    log.info("loading DiT…")
    anima = anima_utils.load_anima_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=False,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    # Mod-guidance: pooled_text_proj params are meta tensors when not in the
    # base checkpoint, so load weights BEFORE .to() (matches
    # library/inference/models.py:95-99). Bench-default is off → buffers stay
    # at init zeros after reset, giving an identity per-block addition.
    if args.pooled_text_proj:
        anima_utils.load_pooled_text_proj(anima, args.pooled_text_proj, "cpu")
    anima.to(device, dtype=dtype)
    anima.eval().requires_grad_(False)

    # Transient text-encoder block. Triggered when either mod-guidance is on
    # (encodes pos/neg pooled deltas) or CFG > 1 (encodes uncond crossattn).
    # Loads the Qwen3 text encoder once, runs both encodes, frees it.
    embed_uncond: torch.Tensor | None = None
    needs_text_encoder = bool(args.pooled_text_proj) or args.guidance_scale != 1.0
    if needs_text_encoder:
        from library.anima import strategy as strategy_anima, text_strategies
        from library.inference.models import load_text_encoder

        # Mirror inference.py:909-918 — mod_guidance.tokenize_strategy.tokenize()
        # reads the module-level singletons, so they have to be primed before
        # setup_mod_guidance encodes the pos/neg prompts.
        text_strategies.TokenizeStrategy.set_strategy(
            strategy_anima.AnimaTokenizeStrategy(
                qwen3_path=args.text_encoder,
                t5_tokenizer_path=None,
                qwen3_max_length=512,
                t5_max_length=512,
            )
        )
        text_strategies.TextEncodingStrategy.set_strategy(
            strategy_anima.AnimaTextEncodingStrategy()
        )

        text_encoder = load_text_encoder(args, dtype=torch.bfloat16, device=device)
        text_encoder.eval()

        if args.guidance_scale != 1.0:
            log.info(
                f"CFG={args.guidance_scale}; encoding uncond "
                f"(negative_prompt='{args.negative_prompt}')"
            )
            embed_uncond = encode_uncond_embed(
                anima, text_encoder, args.negative_prompt, device
            )

        if args.pooled_text_proj:
            from library.inference.mod_guidance import setup_mod_guidance

            setup_mod_guidance(
                args,
                anima,
                device,
                shared_models={"text_encoder": text_encoder},
            )
        else:
            anima.reset_mod_guidance()

        # Free the text encoder — neither CFG nor mod-guidance needs it during
        # the bench loop (uncond is one frozen tensor, mod delta is baked).
        del text_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        anima.reset_mod_guidance()

    if args.lora_weight:
        attach_loras(anima, args.lora_weight, list(args.lora_multiplier), device, dtype)

    # Compile last: after LoRA + mod-guidance attach so the wrapped graph sees
    # them. set_hydra_sigma already routes through ``_orig_mod`` so writes to
    # router state work on the OptimizedModule wrapper.
    if args.compile:
        log.info("torch.compile(DiT)…")
        anima = torch.compile(anima)

    samples = pick_cached_samples(
        Path(args.dataset_dir),
        args.n_images,
        image_h=args.image_h,
        image_w=args.image_w,
        shuffle_seed=args.shuffle_seed,
    )
    if not samples:
        shape_msg = (
            f" matching {args.image_h}x{args.image_w}"
            if args.image_h is not None
            else ""
        )
        raise SystemExit(
            f"no cached samples{shape_msg} under {args.dataset_dir}. "
            "Expected *_anima.npz + *_anima_te.safetensors pairs (make preprocess)."
        )
    shape_info = (
        f" @ {args.image_h}x{args.image_w}"
        if args.image_h is not None
        else " (mixed shapes)"
    )
    log.info(
        f"using {len(samples)} cached samples (variant v{args.text_variant}){shape_info}"
    )

    _, sigmas_t = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    sigmas = sigmas_t.cpu()
    n_steps = args.infer_steps
    log.info(
        f"infer_steps={n_steps}, flow_shift={args.flow_shift}, "
        f"σ₀={float(sigmas[0]):.3f}, σₙ={float(sigmas[-1]):.3f}"
    )

    # DCW configs: baseline + optional λ sweep (LL-only, one_minus_sigma).
    configs: list[tuple[str, float]] = [("baseline", 0.0)]
    if args.dcw_sweep:
        for lam in args.dcw_scalers:
            if lam == 0.0:
                continue
            configs.append((f"λ={lam}_LL_oneminussigma", lam))
    n_fwd = len(samples) * args.n_seeds
    cfg_mult = 2 if args.guidance_scale > 1.0 else 1
    fwd_calls = len(samples)
    fwd_batch = args.n_seeds * cfg_mult
    if args.dcw_sweep:
        rev_calls = n_fwd
        rev_batch = len(configs) * cfg_mult
        rev_desc = f"{len(configs)} λ trajectories"
    else:
        rev_calls = len(samples)
        rev_batch = args.n_seeds * cfg_mult
        rev_desc = f"{args.n_seeds} seed trajectories at λ=0"
    log.info(
        f"{len(configs)} config(s) × {len(samples)} samples × {args.n_seeds} seeds: "
        f"{fwd_calls} fwd calls (batch={fwd_batch}) + {rev_calls} rev calls "
        f"(batch={rev_batch}, advances {rev_desc} per call)"
    )

    # Preload cached data onto device.
    log.info("loading cached latents + text embeds…")
    encoded = []
    for stem, npz, te in samples:
        x_0, embed = load_cached(npz, te, args.text_variant, device)
        encoded.append((stem, x_0, embed))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    accum: dict = {
        name: dict(
            v_fwd=np.zeros(n_steps),
            v_rev=np.zeros(n_steps),
            gap=np.zeros(n_steps),
            gap_sq=np.zeros(n_steps),
            v_fwd_bands={b: np.zeros(n_steps) for b in BANDS},
            v_rev_bands={b: np.zeros(n_steps) for b in BANDS},
            gap_bands={b: np.zeros(n_steps) for b in BANDS},
            n=0,
        )
        for name, _ in configs
    }

    per_sample_bands: dict[str, np.ndarray] | None = None
    per_sample_v_rev_bands: dict[str, np.ndarray] | None = None
    per_sample_stems: list[str] | None = None
    if args.dump_per_sample_gaps:
        n_traj = len(encoded) * args.n_seeds
        per_sample_bands = {b: np.zeros((n_traj, n_steps)) for b in BANDS}
        per_sample_v_rev_bands = {b: np.zeros((n_traj, n_steps)) for b in BANDS}
        per_sample_stems = [""] * n_traj

    # Per-(stem, seed_int, config_name) final latent collected for VAE decode
    # at end-of-run, when --save_images is set. Each entry is a single-row
    # tensor shape (1, *x_0.shape[1:]) on CPU/float32 — typically ~520 KB at
    # 832×1248 (16-channel latent, fp32).
    finals_to_decode: list[tuple[str, int, str, torch.Tensor]] | None = (
        [] if args.save_images else None
    )

    t0 = time.time()

    def _seeds_for(img_idx: int) -> list[int]:
        return [args.seed_base + 1000 * img_idx + j for j in range(args.n_seeds)]

    # Phase 1: forward-branch norms — always batched across seeds per image
    # (bit-equivalent to the per-seed serial loop; per-row CPU generators
    # produce the same RNG sequence). Cached per (img, seed) for phase 2.
    fwd_cache: dict[tuple[int, int], tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    pbar = tqdm(total=len(encoded), desc=f"fwd (×{args.n_seeds} seeds batched)")
    for img_idx, (stem, x_0, embed) in enumerate(encoded):
        seeds = _seeds_for(img_idx)
        fwd_results = measure_forward_norms(
            anima,
            x_0,
            embed,
            sigmas,
            noise_seeds=seeds,
            device=device,
            embed_uncond=embed_uncond,
            cfg_scale=args.guidance_scale,
        )
        for seed_idx, res in enumerate(fwd_results):
            fwd_cache[(img_idx, seed_idx)] = res
        pbar.update(1)
        pbar.set_postfix_str(stem)
    pbar.close()

    # Phase 2: reverse trajectories.
    # * --dcw_sweep: keep the per-seed outer loop; each call batches over λ
    #   (sweep semantics — all rows share one initial-noise seed).
    # * default (make dcw): batch all seeds per image; one row per seed at
    #   λ=0. Mutually exclusive — outer-product (seed × λ) is out of scope.
    config_lams = [lam for _, lam in configs]
    if args.dcw_sweep:
        pbar = tqdm(total=n_fwd, desc=f"rev (×{len(configs)} λ batched)")
        for img_idx, (stem, x_0, embed) in enumerate(encoded):
            for seed_idx in range(args.n_seeds):
                seed = args.seed_base + 1000 * img_idx + seed_idx
                rev_out = run_reverse_batched(
                    anima,
                    x_0,
                    embed,
                    sigmas,
                    noise_seeds=[seed] * len(configs),
                    dcw_lams=config_lams,
                    device=device,
                    embed_uncond=embed_uncond,
                    cfg_scale=args.guidance_scale,
                    return_final=args.save_images,
                )
                if args.save_images:
                    rev_results, final_latents = rev_out
                else:
                    rev_results = rev_out
                    final_latents = None
                v_fwd, fwd_bands = fwd_cache[(img_idx, seed_idx)]
                for j, (name, _lam) in enumerate(configs):
                    rev_norms, rev_bands = rev_results[j]
                    _accumulate_row(
                        accum,
                        name,
                        v_fwd,
                        fwd_bands,
                        rev_norms,
                        rev_bands,
                        per_sample_bands,
                        per_sample_v_rev_bands,
                        per_sample_stems,
                        img_idx,
                        seed_idx,
                        args.n_seeds,
                        stem,
                    )
                    if finals_to_decode is not None and final_latents is not None:
                        finals_to_decode.append(
                            (stem, seed, name, final_latents[j : j + 1].clone())
                        )
                pbar.update(1)
                pbar.set_postfix_str(f"{stem} seed={seed}")
        pbar.close()
    else:
        pbar = tqdm(total=len(encoded), desc=f"rev (×{args.n_seeds} seeds batched)")
        for img_idx, (stem, x_0, embed) in enumerate(encoded):
            seeds = _seeds_for(img_idx)
            rev_out = run_reverse_batched(
                anima,
                x_0,
                embed,
                sigmas,
                noise_seeds=seeds,
                dcw_lams=[0.0] * args.n_seeds,
                device=device,
                embed_uncond=embed_uncond,
                cfg_scale=args.guidance_scale,
                return_final=args.save_images,
            )
            if args.save_images:
                rev_results, final_latents = rev_out
            else:
                rev_results = rev_out
                final_latents = None
            for seed_idx, (rev_norms, rev_bands) in enumerate(rev_results):
                v_fwd, fwd_bands = fwd_cache[(img_idx, seed_idx)]
                _accumulate_row(
                    accum,
                    "baseline",
                    v_fwd,
                    fwd_bands,
                    rev_norms,
                    rev_bands,
                    per_sample_bands,
                    per_sample_v_rev_bands,
                    per_sample_stems,
                    img_idx,
                    seed_idx,
                    args.n_seeds,
                    stem,
                )
                if finals_to_decode is not None and final_latents is not None:
                    finals_to_decode.append(
                        (stem, seeds[seed_idx], "baseline", final_latents[seed_idx : seed_idx + 1].clone())
                    )
            pbar.update(1)
            pbar.set_postfix_str(stem)
        pbar.close()
    clear_hydra_sigma(anima)
    log.info(f"done in {time.time() - t0:.0f}s")

    # Optional: VAE decode of stashed final latents → PNGs under
    # <out_dir>/images/. Loads VAE transiently; frees DiT first to
    # maximise free VRAM. Each row decoded individually (memory-safe at
    # any --n_images / config count). Filenames include the config name
    # so a 3-config sweep produces an interleaved baseline / λ_a / λ_b
    # set per (stem, seed) pair, suitable for direct visual A/B/C.
    images_dir: Path | None = None
    if finals_to_decode is not None:
        from PIL import Image

        from library.models import qwen_vae as qwen_image_autoencoder_kl

        images_dir = out_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            f"decoding {len(finals_to_decode)} final latents → {images_dir}"
        )

        # Free the DiT (and the encoded latent / text cache) before
        # loading the VAE — keeps peak VRAM bounded by max(DiT, VAE).
        del anima
        encoded.clear()
        fwd_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        vae = qwen_image_autoencoder_kl.load_vae(
            args.vae,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=None,
            disable_cache=False,
        )
        vae.to(torch.bfloat16)
        vae.eval()
        vae.to(device)

        def _safe(name: str) -> str:
            return (
                name.replace("=", "_eq").replace("λ", "lam").replace(" ", "_")
            )

        with torch.no_grad():
            decode_pbar = tqdm(total=len(finals_to_decode), desc="decode")
            for stem, seed_int, cfg_name, latent in finals_to_decode:
                pixels = vae.decode_to_pixels(
                    latent.to(device, dtype=vae.dtype)
                )
                if pixels.ndim == 5:
                    pixels = pixels.squeeze(2)  # [1, 3, H, W]
                img_t = (
                    (pixels[0].clamp(-1.0, 1.0) * 0.5 + 0.5)
                    .to("cpu", dtype=torch.float32)
                    .mul(255)
                    .round()
                    .clamp(0, 255)
                    .byte()
                    .permute(1, 2, 0)
                    .numpy()
                )
                fname = f"{stem}__seed{seed_int}__{_safe(cfg_name)}.png"
                Image.fromarray(img_t).save(images_dir / fname)
                decode_pbar.update(1)
                decode_pbar.set_postfix_str(fname)
            decode_pbar.close()

        vae.to("cpu")
        del vae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info(f"images → {images_dir}")

    # Reduce: mean over (img × seed); std on the gap from running Σ gap².
    for name in accum:
        n = accum[name]["n"]
        mean_g = accum[name]["gap"] / n
        mean_g_sq = accum[name]["gap_sq"] / n
        accum[name]["gap_std"] = np.sqrt(np.maximum(mean_g_sq - mean_g**2, 0.0))
        for k in ("v_fwd", "v_rev", "gap"):
            accum[name][k] = accum[name][k] / n
        for k in ("v_fwd_bands", "v_rev_bands", "gap_bands"):
            for b in BANDS:
                accum[name][k][b] = accum[name][k][b] / n

    # Metrics envelope.
    ranked = sorted(
        (
            (
                name,
                float(np.abs(accum[name]["gap"]).sum()),
                float(accum[name]["gap"].sum()),
            )
            for name in accum
        ),
        key=lambda t: t[1],
    )
    metrics = {
        "infer_steps": n_steps,
        "n_samples": len(samples),
        "n_seeds": args.n_seeds,
        "text_variant": args.text_variant,
        "configs_ranked_by_integrated_abs_gap": [
            {"config": name, "integrated_abs_gap": a, "integrated_signed_gap": s}
            for name, a, s in ranked
        ],
        "per_band_integrated_signed_gap": {
            name: {b: float(accum[name]["gap_bands"][b].sum()) for b in BANDS}
            for name in accum
        },
    }

    # Write artifacts.
    csv_path = write_per_step_csv(out_dir, accum, sigmas, n_steps)
    log.info(f"CSV → {csv_path}")
    band_csv_path = write_per_band_csv(out_dir, accum, sigmas, n_steps)
    log.info(f"per-band CSV → {band_csv_path}")

    per_sample_path: Path | None = None
    if per_sample_bands is not None:
        per_sample_path = out_dir / "gaps_per_sample.npz"
        np.savez(
            per_sample_path,
            sigmas=sigmas.numpy()[:n_steps],
            stems=np.array(per_sample_stems, dtype=object),
            **{f"gap_{b}": per_sample_bands[b] for b in BANDS},
            **{f"v_rev_{b}": per_sample_v_rev_bands[b] for b in BANDS},
        )
        log.info(f"per-sample gaps → {per_sample_path}")

    plot_written = make_plot(out_dir, accum, n_steps)

    artifacts = (
        ["per_step.csv", "per_step_bands.csv"]
        + (["gap_curves.png"] if plot_written else [])
        + (["gaps_per_sample.npz"] if per_sample_path is not None else [])
        + (["images/"] if images_dir is not None else [])
    )
    result_path = write_result(
        out_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=artifacts,
        device=device,
    )
    log.info(f"result → {result_path}")

    print_summary(accum, ranked, args.dcw_sweep)


if __name__ == "__main__":
    main()
