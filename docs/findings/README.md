# Findings

Empirical results and analyses on the Anima model — what we measured, tested, and learned. Each doc states a premise, runs a probe or A/B against the real model, and records what the data showed. Some lines led to shipped features, some to "don't build it," some are pure measurement of how the base model behaves; all of them are evidence you can build the next decision on. Read the relevant doc before reviving — or re-deriving — an idea it covers.

The leftmost column is the outcome — what the evidence settled:

- MEASUREMENT — a probe characterizing how the base model behaves; no code path changed, but the structure it found constrains later work.
- NO-GO / FALSIFIED — an external method (usually a paper port) whose load-bearing premise does not hold on Anima. Evidence against building it.
- CLOSED — a line we did build or seriously reopen, now settled and retired.
- DEMOTED — a claim that was real but overstated; the doc rewrites what it actually buys.
- LANDED / POSITIVE — the premise held under test and the thing shipped; the note stays as the standing evidence.
- TRAP — a reusable methodology or engineering result surfaced along the way, worth knowing before you trust a similar probe.

## Model behavior — where Anima's signal lives

| Verdict | Doc | What it found |
|---------|-----|---------------|
| MEASUREMENT | [sigma_signal_where_anima_resolves.md](sigma_signal_where_anima_resolves.md) | The σ ≈ 0.75 → 0.45 → 0 resolution staircase: a recognizable picture by 0.75, essentially-final by 0.45, detail-only below. |
| MEASUREMENT | [crossattn_self_attn_dominance.md](crossattn_self_attn_dominance.md) | Text writes the low-σ plan early (cross-attn front-loaded), but self-attn + MLP are the dominant residual pathway at every σ. |
| MEASUREMENT + CLOSED | [traj_stats_front_loaded_commitment.md](traj_stats_front_loaded_commitment.md) | Anime-domain token commitment is front-loaded (~½ by σ=0.5) in a fixed ~3-dim channel subspace, and generation ≡ inversion below σ≈0.92 — but the aggregate structure failed to convert into a per-token/per-image intervention basis twice (tier routing, compute reuse); recorder + intactness gauge stay shipped. |
| MEASUREMENT | [cbs_monitor_vs_fei_routing.md](cbs_monitor_vs_fei_routing.md) | CBS's complexity-monitor boundary and ChimeraHydra's FEI routing peak at opposite ends of the σ trajectory (anti-aligned). |

## External methods evaluated — not adopted

| Verdict | Doc | Why |
|---------|-----|-----|
| FALSIFIED | [selfflow.md](selfflow.md) | Self-Flow rep-loss exploits an info gap a still-learning backbone leaves open; Anima's frozen DiT already closed it → ~zero pressure on a rank-r adapter. |
| NO-GO | [spectral_guidance_no_subspace.md](spectral_guidance_no_subspace.md) | The posterior-mean operator does not collapse onto a few guidable directions on Anima — no low-rank guidable subspace (Phase-0 gate). |
| NO-GO | [ctcal_premise_inverted_on_anima.md](ctcal_premise_inverted_on_anima.md) | CTCal assumes cross-attn is sharp at low noise / degrades at high noise; on Anima that premise is inverted. Don't build. |
| NO-GO | [pe_registers_no_patch_outliers.md](pe_registers_no_patch_outliers.md) | Test-time registers (2506.08010) needs high-norm outlier patch tokens; PE-Core/PE-Spatial have none (0 patches ≥5× median, 256 imgs × all layers — CLS absorbs the sink role). Cached PE features are clean; tagger-ceiling result stands stronger. |
| NOT WORTH | [asymflow_parameterization.md](asymflow_parameterization.md) | The rank-asymmetric velocity parameterization fixes a *pixel-space* dimensionality bottleneck Anima (compressed latent, D_patch/residual = 1/32) doesn't have. |
| CLOSED | [deft_subtractive_coupling_slows_convergence.md](deft_subtractive_coupling_slows_convergence.md) | DEFT reaches the same optima as LoRA but 2–8× slower everywhere — `ΔW` quadratic in `P`, gradient dominated by a `−W₀GᵀP` coupling. Don't re-propose subtractive adapters. |
| SHELVED | [l2p_pixel_transfer.md](l2p_pixel_transfer.md) | Swapping the VAE for RGB-patch tokenization breaks the frozen DiT's native latent-token manifold at Anima's 2048-dim / 28-block scale on a single-GPU budget. |
| PARTIAL | [freetext_text_rendering.md](freetext_text_rendering.md) | GO on training-free writing-region *localization*; NO-GO on native OOD glyph rendering. One reusable capability, one clean negative. |
| MIXED | [seacache_sea_decision_metric.md](seacache_sea_decision_metric.md) | SeaCache's SEA filter is a better cache-decision metric (and shipped as one) but is not a Spectrum replacement. |

## Mod-guidance — what the pooled-text head actually does

| Verdict | Doc | What it says |
|---------|-----|--------------|
| DEMOTED + RESOLVED | [mod_guidance_quality_tag_axis.md](mod_guidance_quality_tag_axis.md) | The pooled-text head is a global-tone/finishing lever, not a content or quality lever. The "quality axis" is really content-magnitude (channel attribution); its text-derivative is orthogonal to the teacher's — an architectural cos ceiling (AdaLN writes DC, the teacher's text response is ~99% AC), not a fit gap. Schedule (σ+layer) axes both falsified; shipped `8–26` full-dose validated. |

## Turbo (DP-DMD) levers

| Verdict | Doc | Why |
|---------|-----|-----|
| CLOSED (dead twice) | [turbo_fei_band_deficit_falsified.md](turbo_fei_band_deficit_falsified.md) | FEI band-deficit reweighting of the CFG-uplift `δ_cfg`: falsified in the CA-era loop (wrong distribution), then legitimately reopened on-trajectory and killed again (σ-matched null). Do not re-propose FEI / band-split levers on the DP-DMD loop. |
| CLOSED | [turbo_tau_critic_interference_lr_artifact.md](turbo_tau_critic_interference_lr_artifact.md) | The fake/critic's cross-band (τ) gradient interference — the load-bearing premise of the τ-split critic — is an LR artifact: 17–19× headroom at the annealed tail LR (4e-6), `G1_FAIL` with G2 inverted at the real peak LR (3e-5). Third band-split lever to die on the DP-DMD loop (see row above). Trap: a probe fine-tune at tail LR measures the linearized neighborhood of the init and inflates SNR — pre-register the peak-LR arm as primary. |
| CLOSED | [turbo_gan_dm_grad_orthogonal.md](turbo_gan_dm_grad_orthogonal.md) | The GAN generator gradient is elementwise DM-orthogonal (agree-energy == permutation null, every τ-bin) at ~6× the applied DM magnitude — big push, zero structured conflict. Kills the whole GAN↔DM gradient-surgery class (OPD² sign gate, PCGrad). Trap: ~0.5 agree-rate is the orthogonal *baseline*; judge energy vs a permutation null, on applied gradients. |

## Premises that held — the thing shipped

| Verdict | Doc | Result |
|---------|-----|--------|
| LANDED | [agsm_reward_premise_holds.md](agsm_reward_premise_holds.md) | Relative FM-ranking survives where absolute FM-MSE doesn't; bounded AGSM trained on the soft-tokens path and helped prompt-following + quality. |
| A/B RESULT | [channel_stats_content_independence.md](channel_stats_content_independence.md) | `per_channel_scaling` calibration is content-agnostic (weights/architecture-driven) — generalizes across datasets, no per-dataset recompute. |

## Reusable traps

| Verdict | Doc | The trap |
|---------|-----|----------|
| TRAP + LANDED | [paired_dw_chaos_floor_deterministic.md](paired_dw_chaos_floor_deterministic.md) | Paired (CRN) training arms decorrelate to ΔW cos 0.413 with zero treatment — flash-attn backward's atomic-add order is the one un-seedable RNG, chaos-amplified over 1200 steps. `--deterministic` (bit-identical checkpoints, twin-validated, ~33% slower) removes the floor — but buys attribution, not magnitude: real treatments saturate the same chaotic subspace (~0.4), so endpoint ΔW cosine is a detector with depth localization, never a ruler for treatment size. |
| TRAP | [spectral_fraction_metric_inverts.md](spectral_fraction_metric_inverts.md) | A fraction-of-Δ spectral metric inverts, and per-block ablation can't see a cumulative artifact — before trusting a spectral probe's verdict. |
| TRAP | [custom_autograd_removal_partitioner_oom.md](custom_autograd_removal_partitioner_oom.md) | Removing a numerically-inert custom autograd Function shifted the compile partitioner and OOMed: **numerically-inert ≠ memory-inert**. |
| CLOSED + TRAP | [torch214_no_win_on_sm120.md](torch214_no_win_on_sm120.md) | torch 2.12→2.14 benched on a real compiled LoRA run: the headline NVGEMM backend is not on by default and has **no sm_120 kernel image** (`cutlass.operators` is sm100-only → `cudaErrorNoKernelImageForDevice` + autotune-pool hang); default-mode drift −0.7 % s/it, `combo_kernels=True` a 1.5–2.4 % loss in both versions, max-autotune 1.7× worse mean. Don't bump for perf. Trap: check `max_autotune_gemm_backends` + the backend's arch gate before benching an "auto" Inductor backend. |
| TRAP | Xid 8 GPU hang (no write-up file; procedure in project memory `project_xid8_gpu_hang_recurring`) | Long runs die to a recurring, loop-agnostic `Xid 8` GPU hang — infra, not code, and not preventable. `journalctl -k` silently scopes to the current boot, which under-reported it 2→1. Defence is `--resume`, not debugging the loop. |
