# Inference stacks

Training-free runtime methods — sampler acceleration, sampler-boundary corrections, and representation edits that ride on top of any checkpoint. None of these need training (mod-guidance is the exception — its `pooled_text_proj` head is distilled, but it *applies* at inference). Most compose at the sampler boundary (DAVE is the exception — a block-forward hook); read the relevant doc before touching one.

The DiT operates on 5D latents `(B, C, T=1, H, W)`; sampler-boundary plug-ins here receive 5D — match `ndim` against any 4D reference latent they blend against (see root `CLAUDE.md` §"The DiT operates on 5D latents").

## Acceleration

| Doc | What it is | Flag | Load-bearing gotcha |
|-----|-----------|------|---------------------|
| [spectrum.md](spectrum.md) | Chebyshev feature forecasting — cached steps skip all blocks; `final_layer` pre-hook captures outputs. | `--spectrum` | Structure walkthrough in `../structure/spectrum.md`. |
| [spd.md](spd.md) | Spectral Progressive Diffusion — early steps at low res, spectral noise-expansion handoff to full res. Runner in `networks/spd.py`. | `--spd` | v0 = Euler-only, no SMC/Spectrum compose; single-late `0.5→1.0 @ σ0.7` default. `_archive/spd/bench/plan.md` Phase 3, `_archive/proposals/spd_finetune_lora.md` (Case B). |
| [foveated.md](foveated.md) | Deferred-foveated merge — full grid above σ_c, then fovea tokens 1:1 + 2×2 periphery token groups merged (endogenous `combo` mask). Identity-preserving, ×1.37 e2e. Runner in `networks/foveated.py`. **Line ARCHIVED 2026-07-03** (periphery blur constitutive — P4t; runner stays, off by default). | `--fovea_sigma_c 0.75` | Euler-only; σ_c=0.75 and the final bicubic readout are load-bearing; frac floor 0.25. Foveated-Spectrum compose CLOSED by P3, tail un-merge by P4t (`_archive/bench/foveated/`) — don't re-propose. |

> Channel scaling moved to [`../optimizations/channel_scaling.md`](../optimizations/channel_scaling.md) (2026-06-10) — it's a training-time optimizer-geometry feature, invisible at inference after the save-time bake.

## Sampler-boundary corrections

| Doc | What it is | Flag | Load-bearing gotcha |
|-----|-----------|------|---------------------|
| [smc_cfg.md](smc_cfg.md) | α-adaptive sliding-mode CFG correction in velocity space (λ=5, α=0.2). | `--smc_cfg` | Paper's fixed k was ~14× off; ships `sign()` only (tanh ε removed). |
| [cns.md](cns.md) | SDE noise recolorer — per-step injected noise is `sqrt(1−γ)`-shaped toward unresolved freq bands, RMS-renormalized (zero-sum). | `--sampler er_sde --cns auto` | **er_sde-only** (no-op on euler/lcm); faithful to paper Alg. 1. |
| [mod-guidance.md](mod-guidance.md) | Text-conditioned AdaLN via a learned `pooled_text_proj` MLP, distilled once (`project/finished/mod_guidance/`). | `make test MOD=1` | Global-tone lever, not a content lever (σ-FiLM probe was a geometric ceiling). |
| [fsg.md](fsg.md) | Foresight Guidance — **pre-step latent calibration**: at scheduled mid-σ steps run K forward(cond)–backward(uncond) fixed-point iterations to pull `x_t` onto the golden path, then denoise from `x̂_t`. Production stack rides the CFG++ substrate (`--cfgpp`, λ=1.5); composes with `--spectrum`. | `make test FSG=1` | **Line CLOSED 2026-07-12** (bench archived; feature stays). **Mid-σ band only** (default `[0.59,0.75]` @28-step er_sde) — σ≈0.94 diverges. `3·K` extra forwards/step ≈1.8× NFE; matched-NFE A/B never run → not "free quality". Still ignored under `--spd`. |

## Representation edits

Edit intermediate block features inside the forward (a hook), not the sampler boundary.

| Doc | What it is | Flag | Load-bearing gotcha |
|-----|-----------|------|---------------------|
| [dave.md](dave.md) | Same-prompt **diversity** recovery — per-block post-`forward` hook attenuates the cross-seed-shared DC (spatial mean) during the early steps, freeing the seed-specific AC. Flat 8–18 pool. | `--dave auto` / `make test DAVE=1` | Text/hand damage tracks **window width** not dose — defaults `τ0.10·s0.3`; baked `≤18` cap forecloses the patch-grid dots. Block-hook (survives `compile_blocks`), no sampler-boundary compose yet (v0). |
| [xattn_boost.md](xattn_boost.md) | Front-loaded cross-attn boost — per-block `_xattn_gain` buffer scales the cross-attn residual by λ on the **cond forward only**, σ ≥ band (default 0.85, the plan-writing window). Fixes weak-tag relations/bindings. | `--xattn_boost 2` / `make test XATTN_BOOST=2` | Amplifies **all** caption tags (framing priors ride along); can't conjure unknown concepts. The σ-gated CFG arm of the same proposal is CLOSED (style collapse) — don't re-propose. |

## Serving

Infrastructure, not a sampler method — how to *run* inference, not what it does at the boundary.

| Doc | What it is | Entry | Load-bearing gotcha |
|-----|-----------|-------|---------------------|
| [server.md](server.md) | Resident inference server — load DiT/VAE/TE once, serve many generations over a localhost HTTP port + pidfile (the inference twin of `anima_daemon/`). | `python scripts/inference_server.py serve` | Separate process from the training daemon (opposite GPU lifetime); coexists via cooperative `/unload` + idle-TTL so it yields the card to training. |

## Other

| Doc | What it is | Gotcha |
|-----|-----------|--------|
| [invert.md](invert.md) | Embedding inversion — optimize a text embedding to match a target image through the frozen DiT (full and K-slot reference). **Line fully archived 2026-07-04** (probe + bench + proposals → `_archive/`). | Postfix slot-collapse: `anima_postfix.safetensors` is effectively K=1; per-image tail inversion never produced meaningful reconstruction lift. |

User-facing flag reference: [`../guidelines/inference.md`](../guidelines/inference.md).
