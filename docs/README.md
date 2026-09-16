# Documentation

Index of the `docs/` tree. Read the linked doc before working on the thing it describes.

## Methods

Shipped training methods, losses and model assets.

| Doc | Description |
|-----|-------------|
| [methods/svd-down-lora.md](methods/svd-down-lora.md) | SVD-Down LoRA — down-projection seeded from the pretrained weight's own singular vectors instead of random init; default down-init for plain LoRA |
| [methods/hydra-lora.md](methods/hydra-lora.md) | HydraLoRA — MoE multi-head routing (shared-A experts), one cell of the three-axis routing surface in `configs/methods/lora.toml` |
| [methods/timestep_mask.md](methods/timestep_mask.md) | T-LoRA — timestep-dependent rank masking (full rank at noise, rank 1 at clean) |
| [methods/turbo.md](methods/turbo.md) | Turbo (DP-DMD) — diversity-preserved few-step distillation of the CFG=4 teacher into an N-step LoRA student (`make turbo`; published 4-step student on HF) |
| [methods/adaln.md](methods/adaln.md) | AdaLN LoRA — adapting the per-block AdaLN modulation MLPs, and shipping them ComfyUI-compatible |
| [methods/repa.md](methods/repa.md) | REPA — auxiliary loss aligning a mid-block DiT feature to cached PE-Spatial patch tokens |
| [methods/anima-2.9b.md](methods/anima-2.9b.md) | Anima-2.9B — the 40-block community depth expansion; loading, depth-specific LoRAs, regeneration recipes |
| [methods/cjk_vocab_pack.md](methods/cjk_vocab_pack.md) | CJK vocab pack — extra T5-side embedding rows for JA / KO / ZH prompts (text-encoder asset, not a LoRA) |

## Inference

Training-free runtime stacks (acceleration, sampler-boundary corrections, representation edits, the resident server) — indexed in [inference/README.md](inference/README.md).

## Experimental

Wired and runnable, but not part of the default stack — may break or change.

| Doc | Description |
|-----|-------------|
| [experimental/chimera-hydra.md](experimental/chimera-hydra.md) | ChimeraHydra — dual-pool additive MoE (content + freq routers) over disjoint SVD subspaces |
| [experimental/easycontrol.md](experimental/easycontrol.md) | EasyControl — extended self-attn image conditioning; frozen DiT, per-block cond LoRA + scalar gate |
| [experimental/soft_tokens.md](experimental/soft_tokens.md) | Soft Tokens — SoftREPA per-layer × per-t soft text tokens (~1M params); frozen DiT, optional B=1 contrastive |
| [experimental/directedit_editing_v3.md](experimental/directedit_editing_v3.md) | DirectEdit (v3) — flow-inversion image editing; what's actually wired and runnable |
| [experimental/vr_loss.md](experimental/vr_loss.md) | Variance-reduced FM loss — AsymFlow §5.2 control-variate correction at the loss level |
| [experimental/easycontrol_region.md](experimental/easycontrol_region.md) | EasyControl · Region — paint-to-character inpainting task on the EasyControl stack |
| [experimental/byg.md](experimental/byg.md) | BYG — unpaired instruction editing (no paired data, no reward model) |
| [experimental/soup.md](experimental/soup.md) | Soup — uncond-init ΔW LoRA soup (`make soup`) |
| [experimental/cjk_ext_vocab_coverage.md](experimental/cjk_ext_vocab_coverage.md) | CJK ext-vocab row coverage — measured reachable/unreachable rows and the symbol block |

Curation docs — Anima Tagger, position captions, multiview audit, grouping,
masking — live with their code in the sibling
[`anime_tools`](https://github.com/sorryhyun/anime_tools) repo, under
[`docs/`](https://github.com/sorryhyun/anime_tools/tree/main/docs)
(`../anime_tools/docs/`).

## Structure

Architecture walkthroughs — how a component is built.

| Doc | Description |
|-----|-------------|
| [structure/anima.md](structure/anima.md) | The Anima model end to end — text conditioning, VAE, DiT block stack, training-step flow |
| [structure/anima-optimizations.md](structure/anima-optimizations.md) | Non-obvious perf/compile decisions and the *why* behind each |
| [structure/lora.md](structure/lora.md) | Plain LoRA inside Anima — the scaffolding every variant stacks on |
| [structure/ortholora.md](structure/ortholora.md) | PSOFT-integrated OrthoLoRA — exactly-orthogonal bases from SVD + skew-symmetric seeds |
| [structure/hydralora.md](structure/hydralora.md) | HydraLoRA — layer-local MoE over LoRA up-heads |
| [structure/chimera-hydra.md](structure/chimera-hydra.md) | ChimeraHydra — dual-pool additive MoE on the OrthoHydra basis |
| [structure/timestep-mask.md](structure/timestep-mask.md) | T-LoRA — the one-line timestep→rank masking change |
| [structure/modulation.md](structure/modulation.md) | Pooled-text modulation — max-pooled caption summary into the AdaLN stack |
| [structure/spectrum.md](structure/spectrum.md) | Spectrum — Chebyshev feature forecasting at inference (run-or-predict per step) |
| [structure/turbo.md](structure/turbo.md) | Turbo (DP-DMD) — structural walkthrough of the diversity-preserved distillation |

## Findings

Empirical results on the Anima model, keyed by outcome (MEASUREMENT / NO-GO / FALSIFIED / CLOSED / DEMOTED / LANDED / TRAP) — indexed in [findings/README.md](findings/README.md).

## Optimizations

Compiler, kernel, hardware setup, and training-time optimizer geometry.

| Doc | Description |
|-----|-------------|
| [optimizations/for_compile.md](optimizations/for_compile.md) | Changes from sd-scripts for torch.compile / dynamo |
| [optimizations/channel_scaling.md](optimizations/channel_scaling.md) | Channel Scaling — SmoothQuant-style per-channel LoRA gradient rebalance (on by default, α=0.5; inert on frozen-basis ortho variants) |
| [optimizations/sigma_lowres.md](optimizations/sigma_lowres.md) | σ-demoted training (`--sigma_lowres`) — route each step's latent grid by noise level; stacked 768 router + placement spans (opt-in, ~−14% wall) |
| [optimizations/fa4.md](optimizations/fa4.md) | Flash Attention 4 — why it was evaluated and removed |
| [optimizations/adamw_fused.md](optimizations/adamw_fused.md) | AdamW8bit → fused AdamW — why bitsandbytes was dropped |
| [optimizations/hydra_analysis.md](optimizations/hydra_analysis.md) | HydraLoRA — nsys-driven optimization pass (2026-05-03) |

## Guidelines

User-facing guides and references.

| Doc | Description |
|-----|-------------|
| [guidelines/training.md](guidelines/training.md) | Training reference — LoRA variants, caption shuffle, masked loss, dataset config |
| [guidelines/inference.md](guidelines/inference.md) | Inference reference — flags, prompt files, LoRA format conversion |
| [guidelines/difference_between_comfy.md](guidelines/difference_between_comfy.md) | anima_lora vs ComfyUI implementation differences |
| [guidelines/guidebook.md](guidelines/guidebook.md) | Comprehensive guide (English) |
| [guidelines/가이드북.md](guidelines/가이드북.md) | 종합 가이드 (Korean) |
| [guidelines/ガイドブック.md](guidelines/ガイドブック.md) | 総合ガイド (Japanese) |
| [guidelines/指南书.md](guidelines/指南书.md) | 综合指南 (Chinese) |

## Proposals

Active design docs for unbuilt work — see [proposal/](proposal/) for the live
list. Closed proposals move to the gitignored `_archive/proposals/` with a
closure note explaining the verdict.

## Architecture notes

Repo-wide planning docs.

| Doc | Description |
|-----|-------------|
| [multi_model_support.md](multi_model_support.md) | Terrain map for adding a second image model (e.g. Z-Image-Base) alongside Anima — exploratory |
