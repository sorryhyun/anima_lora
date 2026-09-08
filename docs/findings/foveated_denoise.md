# Foveated denoise — findings from the archived line (2026-07-03)

One-day bench arc (P0 → P4t), archived same day by owner decision. Full digest:
`_archive/bench/foveated/report.md`; retired ship proposal:
`_archive/proposals/foveated_denoise.md`. The runner (`networks/foveated.py`,
`--fovea_sigma_c`, Euler-only, off by default) remains in-tree and functional —
×1.37 e2e with a baseline-identical fovea — but the periphery soft blur that pays
for it was judged unacceptable, and P4t proved it unrecoverable. Do not re-propose
the composed foveated-Spectrum line, partial recompute (3b), fovea-region SEA
triggers, or tail un-merge; the specifics are in the report.

Findings that outlive the line:

- **cfgdelta is a free, prompt-aware subject localizer.** Accumulated per-cell
  |v_cond − v_uncond| over the first (high-σ) steps marks where the prompt is
  steering — zero extra models, zero extra forwards, complete by σ≈0.75 (text
  drive is front-loaded). It never lost to a hand-placed oracle rect and beat it
  −29 % subject RMSE on 3-subject prompts (95 % face cover). Reusable wherever a
  "where is the prompt acting" map is needed (region-aware guidance, mask seeding,
  attention probes). Pipeline: normalize → threshold → morphological open → close
  → dilate-1 → re-solve threshold to a target fraction (compactness is
  load-bearing — scattered cells lose their attention neighborhood, +47 % error).
- Calibrate on hard prompts before believing any composed-stack result. The
  aggressive-Spectrum operating point collapses on multi-subject prompts *with no
  foveation involved*; four phases of default-prompt eyeballing read partial
  rescue of a broken baseline as a win. RMSE certifies change, not improvement.
- Blur is a fixed point of the flow at low σ. Once the periphery's LF/mid
  support is pooled away, late full-grid steps cannot re-synthesise detail: a
  blurred-clean state reads as intended bokeh (no-op), σ-scale fresh HF noise gets
  resolved back to smooth (the x̂₀-LF dictates the completion), and the frozen
  σ_c-scale within-group HF renders as confetti speckle (mixed-σ off-manifold).
  Detail comes from HF that was being denoised all along, not from late injection.
- Never rewrite the latent's noise with structured/group-constant ε —
  variance-correct is not distribution-correct; the DiT reads the LF-power excess
  as content. Fresh white noise is on-manifold; group-constant ε is not.
- Merged-rope exactness: the renormalized elementwise mean of member (cos, sin)
  rows is the exact mean-position rope for any symmetric m×m token group — the
  trick carries to any future token-merge scheme (`FoveatedTokenMerge`, in-tree).
- Deferred σ-gating works: foveating only below σ≈0.75 preserves image
  identity exactly (composition locks above it) — the inversion of the paper's
  all-steps merging, and the reason no post-training was needed. The gate concept
  is reusable for any spatially-degraded compute scheme.
