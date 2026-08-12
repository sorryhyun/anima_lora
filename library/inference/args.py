"""Anima inference argument parser — the single source of truth for every
generation knob ``generate()`` reads off an ``argparse.Namespace``.

Extracted from the top-level ``inference.py`` CLI script so programmatic callers
(``GenerationRequest.to_args()``, ``bench/`` probes, embedders) can build a
fully-defaulted namespace without importing the entry-point module. ``inference``
now delegates its own ``parse_args`` here, so the CLI and every library caller
share one parser definition.

    from library.inference.args import build_default_args
    args = build_default_args(["--prompt", "a fox", "--text_encoder", te, "--save_path", out])

``build_parser()`` exposes the raw ``ArgumentParser`` for callers that want to
add flags or inspect defaults; ``build_default_args(argv)`` parses + runs the
post-parse validation/normalization the CLI relied on.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the Anima inference ``ArgumentParser`` (no parsing).

    Every default here is authoritative — ``generate()`` reads ~40 fields via
    ``getattr``, so this parser is what populates them. Exposed separately from
    :func:`build_default_args` so a caller can append its own flags before
    parsing.
    """
    parser = argparse.ArgumentParser(description="HunyuanImage inference script")

    parser.add_argument("--dit", type=str, default=None, help="DiT directory or path")
    parser.add_argument("--vae", type=str, default=None, help="VAE directory or path")
    parser.add_argument(
        "--vae_chunk_size",
        type=int,
        default=None,
        help="Spatial chunk size for VAE encoding/decoding to reduce memory usage. Must be even number. If not specified, chunking is disabled (official behavior)."
        + "",
    )
    parser.add_argument(
        "--vae_disable_cache",
        action="store_true",
        help="Disable internal VAE caching mechanism to reduce memory usage. Encoding / decoding will also be faster, but this differs from official behavior."
        + "",
    )
    # Fold the causal Conv3d stack into 2D convs for decode (image-only, T=1):
    # ~2x faster decode at ~0.65-0.7x peak memory, numerically equivalent within
    # bf16 noise. Mirrors the preprocess caching path (cache_latents.py); default
    # on. Flag names match that script (--qwen_image_vae_2d / --no_vae_2d).
    parser.add_argument(
        "--vae_2d",
        "--qwen_image_vae_2d",
        dest="vae_2d",
        action="store_true",
        default=True,
        help="Fold the VAE Conv3d stack into 2D convs for decode (default on).",
    )
    parser.add_argument(
        "--no_vae_2d",
        dest="vae_2d",
        action="store_false",
        help="Use the stock 3D causal-Conv3d VAE for decode instead of the 2D fold.",
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        required=True,
        help="Text Encoder 1 (Qwen2.5-VL) directory or path",
    )

    parser.add_argument(
        "--lora_weight",
        type=str,
        nargs="*",
        required=False,
        default=None,
        help="LoRA weight path",
    )
    parser.add_argument(
        "--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier"
    )
    parser.add_argument(
        "--soft_tokens_weight",
        type=str,
        default=None,
        help="Soft tokens weight path (networks.methods.soft_tokens .safetensors). "
        "SoftREPA-style per-layer × per-t bank; spliced into the cross-attn input "
        "of the first n_layers DiT blocks via monkey-patched Block.forward.",
    )
    parser.add_argument(
        "--easycontrol_weight",
        type=str,
        default=None,
        help="EasyControl weight path (networks.methods.easycontrol .safetensors). "
        "Requires --easycontrol_image. Extends DiT self-attention with VAE-encoded reference "
        "K/V plus a per-block additive logit bias.",
    )
    parser.add_argument(
        "--easycontrol_image",
        type=str,
        default=None,
        help="Reference image path for EasyControl conditioning. Encoded by the Anima VAE "
        "and patch-embedded into condition tokens.",
    )
    parser.add_argument(
        "--easycontrol_scale",
        type=float,
        default=None,
        help="Override EasyControl scale (default: ss_cond_scale from the checkpoint, typically 1.0).",
    )
    parser.add_argument(
        "--easycontrol_b_offset",
        type=float,
        default=None,
        help="Additive offset on every block's b_cond gate — the continuous "
        "cond-attention dial (each +1 gives ~e× more cond softmax mass). Applied "
        "live after load_weights, NOT baked into the KV cache. Needed when a "
        "checkpoint's b_cond_init leaves the cond stream disengaged at its "
        "trained point (the subject descriptor engages around +7). Mirrors "
        "scripts/edit.py's flag of the same name.",
    )
    parser.add_argument(
        "--easycontrol_image_match_size",
        action="store_true",
        help="Auto-pick --image_size by free-fitting the reference image's aspect ratio "
        "into the 1024 tier's token band (overrides --image_size). Only effective with --easycontrol_image.",
    )
    parser.add_argument(
        "--include_patterns",
        type=str,
        nargs="*",
        default=None,
        help="LoRA module include patterns",
    )
    parser.add_argument(
        "--exclude_patterns",
        type=str,
        nargs="*",
        default=None,
        help="LoRA module exclude patterns",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for classifier free guidance. Default is 3.5.",
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="prompt for generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="negative prompt for generation, default is empty string",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[1024, 1024],
        help="image size, height and width",
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=50,
        help="number of inference steps, default is 50",
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="path to save generated video"
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")

    parser.add_argument(
        "--flow_shift",
        type=float,
        default=3.0,
        help="Shift factor for flow matching schedulers. Default is 3.0 (matches the official Anima scheduler config).",
    )
    parser.add_argument(
        "--sigma_tail_power",
        type=float,
        default=1.0,
        help="σ-schedule reshape exponent (default 1.0 = canonical schedule). >1 packs "
        "more steps into the low-σ resolve tail (σ<0.45); see bench/sigma_reshape.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="euler",
        choices=["euler", "er_sde", "lcm"],
        help="Sampler to use: 'euler' (deterministic ODE), 'er_sde' (Extended Reverse-Time SDE), or 'lcm' (x0 re-noise — for distilled few-step models). Default is euler.",
    )

    parser.add_argument(
        "--text_encoder_cpu",
        action="store_true",
        help="Inference on CPU for Text Encoders",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device to use for inference. If None, use CUDA if available, otherwise use CPU",
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=[
            "flash",
            # "flash4",  # not supported yet (flash-attention-sm120 disabled)
            "torch",
            "sageattn",
            "flex",
            "sdpa",
        ],  #  "sdpa" for backward compatibility
        help="attention mode",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="images",
        choices=["images", "latent", "latent_images"],
        help="output type",
    )
    parser.add_argument(
        "--no_metadata", action="store_true", help="do not save metadata"
    )
    parser.add_argument(
        "--latent_path",
        type=str,
        nargs="*",
        default=None,
        help="path to latent for decode. no inference",
    )
    parser.add_argument(
        "--pooled_text_proj",
        type=str,
        default=None,
        help="Path to trained pooled_text_proj weights (.safetensors) for modulation guidance",
    )
    parser.add_argument(
        "--res_cond_s",
        type=float,
        default=0.0,
        help="E25c: resolution-conditioning scalar for rescond checkpoints "
        "(s = log2(edge/1024); 0.0 = the trained native point, the default). "
        "Requires a --lora_weight carrying sigma_lowres_res_cond_proj — "
        "setting it without one is an error, not a silent no-op.",
    )
    parser.add_argument(
        "--mod_w",
        type=float,
        default=3.0,
        help="Modulation guidance strength (default 3.0). "
        "Controls w in: emb = t_emb + proj(main) + w * (proj(pos) - proj(neg))",
    )
    parser.add_argument(
        "--mod_pos_prompt",
        type=str,
        default="absurdres, score_9, score_8",
        help="Positive quality prompt for modulation guidance direction",
    )
    parser.add_argument(
        "--mod_neg_prompt",
        type=str,
        default="worst quality, low quality, score_1, score_2, score_3",
        help="Negative quality prompt for modulation guidance direction",
    )
    parser.add_argument(
        "--mod_start_layer",
        type=int,
        default=8,
        help="First block (inclusive) that receives the steering delta. "
        "Default 8 protects tonal-DC blocks 0–7 (matches ComfyUI 'step_i8_skip27'). "
        "Use 14 for the safer preset that avoids drift on anatomy-heavy prompts. "
        "Set 0 to recover pre-0413 uniform behavior.",
    )
    parser.add_argument(
        "--mod_end_layer",
        type=int,
        default=27,
        help="Last block + 1 (exclusive) that receives the steering delta. "
        "Default 27 skips the final compensation block on Anima's 28-block DiT. "
        "Use -1 to apply through the final block.",
    )
    parser.add_argument(
        "--mod_taper",
        type=int,
        default=0,
        help="Number of late slots inside [start, end) to scale by --mod_taper_scale. "
        "0 disables taper (default).",
    )
    parser.add_argument(
        "--mod_taper_scale",
        type=float,
        default=0.25,
        help="Multiplier applied to tapered slots (default 0.25).",
    )
    parser.add_argument(
        "--mod_final_w",
        type=float,
        default=0.0,
        help="w applied at final_layer. 0.0 = don't disturb the output head (default).",
    )

    parser.add_argument(
        "--pgraft",
        action="store_true",
        help="Enable P-GRAFT: load LoRA as dynamic hooks instead of static merge, allowing mid-denoising cutoff",
    )
    parser.add_argument(
        "--lora_cutoff_step",
        type=int,
        default=None,
        help="Step at which to disable LoRA during inference (for P-GRAFT). "
        "LoRA is active for steps 0..cutoff_step-1, disabled for cutoff_step..end.",
    )

    parser.add_argument(
        "--tiled_diffusion",
        action="store_true",
        help="Enable MultiDiffusion-style tiled generation for VRAM reduction at high resolutions",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=128,
        help="Tile size in latent space (default 128 = 1024px). Must be even.",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=16,
        help="Overlap between tiles in latent space (default 16 = 128px). Must be even and < tile_size.",
    )

    parser.add_argument(
        "--spectrum",
        action="store_true",
        help="Enable Spectrum inference acceleration (Chebyshev polynomial feature forecasting). "
        "Skips transformer blocks on predicted steps, running only final_layer + unpatchify.",
    )
    parser.add_argument(
        "--spectrum_window_size",
        type=float,
        default=2.0,
        help="Spectrum initial window size N (default 2.0)",
    )
    parser.add_argument(
        "--spectrum_flex_window",
        type=float,
        default=0.25,
        help="Spectrum flex parameter alpha -- N grows by this after each actual forward (default 0.25)",
    )
    parser.add_argument(
        "--spectrum_warmup",
        type=int,
        default=6,
        help="Spectrum warmup steps (always run full forward) (default 6)",
    )
    parser.add_argument(
        "--spectrum_w",
        type=float,
        default=0.3,
        help="Spectrum Chebyshev/Taylor blend weight (1.0=pure Chebyshev, default 0.3)",
    )
    parser.add_argument(
        "--spectrum_m",
        type=int,
        default=3,
        help="Spectrum number of Chebyshev basis functions (default 3)",
    )
    parser.add_argument(
        "--spectrum_lam",
        type=float,
        default=0.1,
        help="Spectrum ridge regression regularization (default 0.1)",
    )
    parser.add_argument(
        "--spectrum_stop_caching_step",
        type=int,
        default=-1,
        help="Force actual forwards from this step onward (-1 = auto: total_steps - 3)",
    )
    parser.add_argument(
        "--spectrum_calibration",
        type=float,
        default=0.0,
        help="Spectrum residual calibration strength (0.0=disabled, default 0.0). "
        "Adds residual bias correction from last actual forward to cached predictions.",
    )
    parser.add_argument(
        "--spectrum_schedule",
        choices=["window", "sea"],
        default="window",
        help="Spectrum when-to-skip rule: 'window' (default, content-blind growing "
        "window) or 'sea' (SeaCache SEA-filtered input-distance trigger). 'sea' "
        "changes only the decision; the forecast+head reuse path is untouched.",
    )
    parser.add_argument(
        "--spectrum_delta",
        type=str,
        default="auto",
        help="SEA threshold for --spectrum_schedule sea. 'auto' (default) "
        "self-calibrates to --spectrum_refresh_ratio on the first generate; a "
        "float pins delta explicitly (for sweeps).",
    )
    parser.add_argument(
        "--spectrum_refresh_ratio",
        type=float,
        default=-1.0,
        help="Target post-warmup refresh fraction for SEA auto-delta calibration. "
        "<=0 (default) matches the growing-window's own refresh fraction at this "
        "step count (like-for-like at matched compute); a positive float pins it.",
    )

    # SPD (arXiv:2605.18736): forces Euler, mutually exclusive with --spectrum.
    # See networks/spd.py + bench/spd/.
    parser.add_argument(
        "--spd",
        action="store_true",
        help="Enable Spectral Progressive Diffusion: run early steps at low "
        "resolution, spectral-expand to full res at the σ handoff. Training-free.",
    )
    parser.add_argument(
        "--spd_stages",
        type=float,
        nargs="+",
        default=None,
        help="Ascending resolution scales, e.g. '0.5 1.0' or '0.5 0.75 1.0'. A "
        "trailing 1.0 is appended if missing. Default: 0.5 1.0 (single handoff).",
    )
    parser.add_argument(
        "--spd_transition_sigmas",
        type=float,
        nargs="+",
        default=None,
        help="σ thresholds (in [0,1]) at which to expand to each next stage; "
        "len = len(stages)-1. Default: 0.7 per handoff (single-late knee).",
    )

    # Deferred-foveated merge (archived line — _archive/bench/foveated/):
    # training-free, forces Euler, mutually exclusive with --spectrum / --spd.
    # See networks/foveated.py + docs/inference/foveated.md.
    parser.add_argument(
        "--fovea_sigma_c",
        type=float,
        default=0.0,
        help="Enable deferred-foveated merge below this σ: full-grid sampling "
        "above, reduced token sequence (fovea 1:1, periphery 2×2-token groups "
        "merged) below. 0 (default) disables. 0.75 is the benched knee — "
        "foveating inside the authority window makes a different image.",
    )
    parser.add_argument(
        "--fovea_frac",
        type=float,
        default=0.35,
        help="Target fovea fraction of the image (default 0.35). Knob floor "
        "0.25; ≤0.15 falsified on multi-subject prompts. Speed ceiling ~×1.6.",
    )
    parser.add_argument(
        "--fovea_mask_source",
        choices=["combo", "cfgdelta", "x0var", "rect"],
        default="combo",
        help="Fovea mask source, derived at the σ_c crossing from signals the "
        "pre-crossing steps already compute. 'combo' (default) = cfgdelta + "
        "x0var, never loses to the static rect. cfgdelta needs CFG "
        "(guidance_scale != 1) — CFG=1 auto-falls-back to x0var. 'rect' = "
        "static center rect.",
    )

    # Front-loaded cross-attn boost (frontload_text_boost arm (b), Phase-0
    # G1+G2 PASS): scale the cross-attn residual of every block by λ on the
    # CONDITIONAL forward only, gated to σ ≥ band — the plan-writing window
    # where cross-attn text drive exists at all (peaks σ=1, ~0.02 floor below
    # σ≈0.85 — docs/findings/crossattn_self_attn_dominance.md). Improves
    # weak-tag adherence / relation bindings; amplifies ALL caption tags
    # (framing priors ride along). Composes with --spectrum (real forwards
    # boosted, forecast steps extrapolate from boosted features) and with
    # --smc_cfg / --cfgpp / --mod_guidance (they touch the combine or the
    # modulation pathway, not the cond forward). Under CFG 1.0 (turbo) the
    # single forward is boosted. See docs/inference/xattn_boost.md.
    parser.add_argument(
        "--xattn_boost",
        type=float,
        default=1.0,
        help="Cross-attn residual gain λ for σ ≥ --xattn_boost_band, cond "
        "forward only. 1.0 = off (exact identity). Phase-0 winner: 2.0 "
        "(1.5 milder; 2.0 costs ~-8%% saturation on complex prompts).",
    )
    parser.add_argument(
        "--xattn_boost_band",
        type=float,
        default=0.85,
        help="σ cutoff for --xattn_boost (boost at σ ≥ band). Default 0.85 = "
        "the drive-floor σ; ~10 of 28 shifted-schedule steps.",
    )
    parser.add_argument(
        "--xattn_boost_renorm",
        choices=["off", "tok", "img"],
        default="img",
        help="Norm matching for --xattn_boost (inert while boost is off). "
        "'img' (default — Phase-1'' winner) rescales each block's post-"
        "cross-attn state so the per-image MEAN token norm stays on the "
        "gain-1 shell: keeps the boost's content wins, kills the saturation "
        "burn, preserves relative token-norm peaks (neon/highlights). 'tok' "
        "matches every token's norm (flattens peaks → grey tone; bench "
        "reference only). 'off' = raw residual gain (pre-renorm behavior).",
    )
    parser.add_argument(
        "--xattn_boost_renorm_frac",
        type=float,
        default=0.5,
        help="Partial renorm exponent ρ (scale**ρ): 1.0 = full norm match, "
        "0.0 = raw boost. Default 0.5 (Phase-1'' sweet spot at λ=2).",
    )

    # SMC-CFG: Sliding-Mode Control CFG (α-adaptive variant; arXiv:2603.03281).
    # Drop-in CFG modification: replaces w·e with w·(e + Δe) where
    # Δe = -k_t·sign(s), s = (e - e_prev) + λ·e_prev, k_t = α·mean(|e_t|).
    # No extra DiT forwards; one prev-step velocity-residual buffer. Composes
    # with --spectrum / --mod_guidance (operates strictly on the
    # velocity-space CFG combine). See docs/inference/smc_cfg.md.
    parser.add_argument(
        "--smc_cfg",
        action="store_true",
        help="Enable Sliding-Mode Control CFG (defaults λ=5, α=0.2). "
        "Modifies the cond/uncond combine; no extra forwards.",
    )
    parser.add_argument(
        "--smc_cfg_lambda",
        type=float,
        default=5.0,
        help="SMC-CFG sliding-manifold slope λ. Paper sweeps {3,4,5,6}; 5 was best.",
    )
    parser.add_argument(
        "--smc_cfg_alpha",
        type=float,
        default=0.2,
        help="SMC-CFG adaptive gain α ∈ (0, 1]. k_t := α·|e_t|.mean() per "
        "step — self-scales across model / CFG / σ / sample. Paper's fixed "
        "k=0.1 was off by ~14× on Anima (bench/smc_cfg/analysis_and_proposal.md), "
        "so the α path is the only mode now. α=0.2 is the production default.",
    )

    # Trajectory-resolved latent statistics (_archive/proposals/traj_latent_stats.md
    # Phase 0). Passive per-step recorder — pure observation, recorder on/off
    # latents are bit-identical (pinned by tests/test_traj_stats.py). One .npz
    # sidecar per generation. Composes with --spectrum (recorded post-combine
    # in the spectrum loop too) and the tiled path (post-blend).
    parser.add_argument(
        "--traj_stats",
        action="store_true",
        help="Record per-step/per-token/per-channel x̂₀ trajectory statistics "
        "to an .npz sidecar (passive; does not change the generation).",
    )
    parser.add_argument(
        "--traj_stats_dir",
        type=str,
        default="output/traj_stats",
        help="Output directory for --traj_stats sidecars.",
    )
    parser.add_argument(
        "--traj_stats_k",
        type=int,
        default=4,
        help="Bits per channel for the --traj_stats quantization codes (1-8).",
    )

    # FSG: Foresight Guidance (NeurIPS 2025, arXiv 23177). Pre-step latent
    # calibration: at scheduled mid-σ steps, run K forward(cond)-backward(uncond)
    # fixed-point iterations to pull x_t onto the golden path, then denoise from
    # x̂_t. Training-free, deterministic; ~3·K extra forwards per scheduled step.
    # Composes with --mod_guidance / --dave / --cns / --smc_cfg (FSG's
    # calibration uses plain γ-combine; the outer step keeps the configured CFG
    # variant). Ignored under --spectrum / --spd (they replace the loop). See
    # docs/inference/fsg.md and docs/proposal/foresight_guidance.md.
    parser.add_argument(
        "--fsg",
        action="store_true",
        help="Enable Foresight Guidance pre-step latent calibration (CFG only). "
        "Defaults band=[0.59,0.75], K=3, Δσ=0.1, γ=guidance_scale.",
    )
    parser.add_argument(
        "--fsg_band",
        type=float,
        nargs=2,
        default=[0.59, 0.75],
        metavar=("SIGMA_LO", "SIGMA_HI"),
        help="σ-band [lo, hi] where FSG calibrates. Default [0.59, 0.75] -- the "
        "production band from the er_sde/28-step Plan-B calibration "
        "(_archive/bench/fsg). "
        "The contracting band moves DOWN with step count: it was [0.75, 0.85] at "
        "20-step Euler, but at 28 steps σ≈0.84 stops contracting and the sweet "
        "spot is σ≈0.75 (the operator contracts in mid-σ; σ≈0.94 always diverges, "
        "ρ>1 -- the paper's noisy-stage prescription is wrong here). Re-probe with "
        "_archive/bench/fsg if you change infer_steps.",
    )
    parser.add_argument(
        "--fsg_k",
        type=int,
        default=3,
        help="FSG fixed-point iterations per scheduled step. error ~ρ^K, ρ≈0.93 "
        "⇒ K=3-4 captures ~all the gain. 0 disables (bit-exact baseline).",
    )
    parser.add_argument(
        "--fsg_d_sigma",
        type=float,
        default=0.1,
        help="FSG forward-backward interval Δσ (the calibration stride, not the "
        "sampler step). Too large is what makes the noisy end diverge.",
    )
    parser.add_argument(
        "--fsg_gamma",
        type=float,
        default=None,
        help="FSG calibration guidance γ. Default None -- reuse --guidance_scale "
        "(the paper's plain γ-combine).",
    )
    # CFG++ substrate (paper App A.2 / Algorithm 1 lines 9-12). FSG is *defined*
    # on a CFG++ base. Implemented as a σ-scheduled guidance REWEIGHT (CFG++
    # differs from CFG solely in strength scheduling) so it composes with er_sde
    # and Euler alike -- see sampling.cfgpp_guidance_weight. Composes with
    # --spectrum (threaded as a side-channel; the reweight is applied in the
    # spectrum combine). Refused under --smc_cfg (alternative combine) and --spd
    # (mid-loop σ re-spacing; not wired there).
    parser.add_argument(
        "--cfgpp",
        action="store_true",
        help="Use the CFG++ substrate (App A.2): replace the constant-w cond/uncond "
        "combine with the σ-scheduled CFG++ weight. The substrate FSG is defined "
        "on; composes with er_sde and --spectrum. Needs its own λ sweep on Anima.",
    )
    parser.add_argument(
        "--cfgpp_lambda",
        type=float,
        default=1.5,
        help="CFG++ calibration strength λ. NB this is a FLOW-space coefficient "
        "(guidance enters as λ·(1−σ')·σ·Δv), NOT the paper's DDIM ξ̃-space λ=0.6 — "
        "the schedules differ. Default 1.5 = λ* from the er_sde/28-step sweep "
        "(_archive/bench/fsg Plan A): best tracks CFG=4 saturation/contrast/composition "
        "(λ<1.5 under-guides → composition wanders; λ≥2 over-saturates). Only used "
        "with --cfgpp.",
    )

    # CNS: Colored Noise Sampling (arXiv:2605.30332). Recolors the er_sde
    # injected noise by sqrt(1-γ) from a precomputed completion matrix so the
    # fixed stochastic budget lands in unresolved frequency bands. Training-free,
    # er_sde-only (no-op on euler). γ is calibrated per (cfg×aspect) by
    # scripts/calibration/cns_calibrate.py. See docs/methods (_archive/bench/cns/plan.md).
    parser.add_argument(
        "--cns",
        type=str,
        default=None,
        help="Enable CNS noise recoloring on the er_sde path. Pass a path to a "
        "cns_gamma.npz completion matrix, or 'auto' for the shipped default "
        "(networks/calibration/cns_gamma.npz). No-op on --sampler euler/lcm.",
    )
    parser.add_argument(
        "--cns_strength",
        type=float,
        default=1.0,
        help="Blend white↔recolored noise (then RMS-renormalize). 1.0 = full "
        "CNS, 0.0 = pass-through. Safety knob for over-injection off-manifold.",
    )

    # DAVE: DC Attenuation for diVersity Enhancement (training-free, ICML'26).
    # Per-block representation edit ĥ = α·μ + (h−μ) that attenuates the cross-seed-
    # shared DC component to recover same-prompt diversity. Hook-based; standard
    # denoise loop only (no Spectrum/SPD compose). Mask from
    # bench/dave/derive_alpha_mask.py. See library/inference/corrections/dave.py.
    parser.add_argument(
        "--dave",
        type=str,
        default=None,
        help="Enable DAVE DC-attenuation. Pass a path to a dave_alpha.npz mask, "
        "or 'auto' for the shipped default (networks/calibration/dave_alpha.npz).",
    )
    parser.add_argument(
        "--dave_strength",
        type=float,
        default=0.3,
        help="DC attenuation strength: per-block (1−α) = strength·w(ℓ). 0 = off, "
        "1 = fully remove the pooled blocks' DC. Default 0.3 (α≈0.7), a conservative "
        "dose that keeps text/hands clean. The shipped flat mask pools blocks 8–18 "
        "only (final-stage 19–27 excluded, so the patch-grid dot artifact can't "
        "appear). With the default --dave_tau 0.10 window the safe ceiling is high — "
        "strength 0.8 still holds legible text + clean hands while diversifying "
        "harder (incl. scene/season changes), so 0.3–0.8 are all usable; pick by how "
        "much recomposition you want. (At the looser τ0.15, stay ≤0.3.)",
    )
    parser.add_argument(
        "--dave_block_lo",
        type=int,
        default=0,
        help="Lowest block index DAVE may touch (inclusive). Blocks below are "
        "forced to no-op. Use with --dave_block_hi to sweep mid-only vs mid+late.",
    )
    parser.add_argument(
        "--dave_block_hi",
        type=int,
        default=-1,
        help="Highest block index DAVE may touch (inclusive; -1 = last block). "
        "Lower it to spare the final content blocks (e.g. --dave_block_hi 18 to "
        "attenuate only the mid blocks where the DC pins layout, not content).",
    )
    parser.add_argument(
        "--dave_sigma_lo",
        type=float,
        default=0.0,
        help="Low σ bound of the DAVE active window (σ∈[0,1]). Default 0 (all σ).",
    )
    parser.add_argument(
        "--dave_sigma_hi",
        type=float,
        default=1.0,
        help="High σ bound of the DAVE active window. Default 1 (all σ). Late-only "
        "(σ<0.45, where DC energy grows) = --dave_sigma_hi 0.45; early-only = "
        "--dave_sigma_lo 0.45.",
    )
    parser.add_argument(
        "--dave_tau",
        type=float,
        default=0.10,
        help="DAVE paper's temporal cutoff τ: attenuate ONLY the first τ fraction "
        "of denoising steps (the early lock-in window). Default 0.10 — tighter than "
        "the paper's 0.15 because on Anima the text/hand damage scales with how many "
        "steps the dose touches, not the dose magnitude: at τ0.10 even strength 0.8 "
        "holds legible text + clean hands, whereas at τ0.15 strength 0.5 already "
        "garbles them. So τ0.10 strictly dominates τ0.15 at equal dose. Converted to "
        "a σ_lo against the live --infer_steps/--flow_shift schedule, so it tracks "
        "the actual step grid (not a hand-guessed σ). Overrides --dave_sigma_lo/hi "
        "when >0. τ>~0.2 tips into posterization. 0 = off (use the σ window).",
    )

    parser.add_argument(
        "--from_file", type=str, default=None, help="Read prompts from a file"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: read prompts from console",
    )
    parser.add_argument(
        "--infer_batch_size",
        type=int,
        default=1,
        help="Batch size for denoising. Prompts sharing the same text are batched together. Higher values use more VRAM.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the DiT model with torch.compile for faster inference. First run incurs compilation overhead.",
    )
    parser.add_argument(
        "--compile_blocks",
        action="store_true",
        help="Compile each DiT block's _forward individually (battle-tested in "
        "training). Sidesteps the checkpointing-wrapper trap that bites "
        "whole-model --compile. Inductor cache persists across runs.",
    )
    parser.add_argument(
        "--compile_inductor_mode",
        type=str,
        default=None,
        help="Inductor preset for --compile_blocks (e.g. 'default', "
        "'reduce-overhead'). None = inductor default.",
    )
    return parser


def build_default_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse ``argv`` (``None`` reads ``sys.argv``) into a fully-defaulted namespace.

    Builds the parser via :func:`build_parser`, parses, then applies the
    post-parse validation/normalization (mutually-exclusive mode checks,
    ``sdpa`` -> ``torch`` alias, tiled-diffusion geometry guards) the CLI relied on.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.from_file and args.interactive:
        raise ValueError(
            "Cannot use both --from_file and --interactive at the same time"
        )

    if args.latent_path is None or len(args.latent_path) == 0:
        if args.prompt is None and not args.from_file and not args.interactive:
            raise ValueError(
                "Either --prompt, --from_file or --interactive must be specified"
            )

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    if args.tiled_diffusion:
        if args.tile_size % 2 != 0:
            raise ValueError(
                f"--tile_size must be even (patch_spatial=2 requires it), got {args.tile_size}"
            )
        if args.tile_overlap % 2 != 0:
            raise ValueError(
                f"--tile_overlap must be even (patch_spatial=2 requires it), got {args.tile_overlap}"
            )
        if args.tile_overlap >= args.tile_size:
            raise ValueError(
                f"--tile_overlap ({args.tile_overlap}) must be less than --tile_size ({args.tile_size})"
            )

    return args
