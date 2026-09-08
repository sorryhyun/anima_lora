# Deferred-foveated merge

> **Line ARCHIVED 2026-07-03** — the periphery soft blur is constitutive
> (P4t: no tail treatment recovers it) and was judged unacceptable, so
> production wiring stops here. The runner remains functional as documented
> below, off by default. Digest: `_archive/bench/foveated/report.md`; retired
> ship plan: `_archive/proposals/foveated_denoise.md`; reusable findings:
> `docs/findings/foveated_denoise.md`.

Training-free inference acceleration, image-identity preserving: same image
where it matters, periphery rendered as an intentional soft blur. ×1.37 e2e at
1024²/28-step/CFG 4 (fwd ×2.15 on merged steps), fovea visually
baseline-identical. Runner in `networks/foveated.py`.

```bash
python inference.py ... --fovea_sigma_c 0.75            # defaults: frac 0.35, combo mask
python inference.py ... --fovea_sigma_c 0.75 --fovea_frac 0.25   # faster, knee point
```

## Mechanism

1. Above σ_c (default 0.75): baseline full-grid sampling — composition and
   identity are decided identically to no-foveation. This "deferred" gating is
   load-bearing: text drive is front-loaded and low frequency bands lock by
   σ≈0.75, so foveating inside the authority window makes a *different image*.
2. During the pre-crossing steps, two free signals accumulate: cfgdelta
   (per-cell |v_cond − v_uncond|, prompt-aware — marks where the prompt is
   steering) and x0var (x̂₀ Laplacian energy — marks detail-critical
   cells). Zero extra models, zero extra forwards.
3. At the crossing, the `combo` mask is built: normalized signal sum →
   threshold → morphological open → close → dilate-1 → threshold re-solved so
   the final fraction hits `--fovea_frac`. Compact and subject-following, per
   (prompt, seed).
4. Below σ_c: the DiT block stack runs on a reduced sequence — fovea tokens
   1:1, each 2×2-token periphery group averaged to one token (4096 → ~2100 at
   frac 0.35). Merged rope = renormalized elementwise mean of the members'
   (cos, sin) rows — the exact mean-position rope for symmetric groups. The
   reduced output is broadcast back to the full grid before `final_layer`. The
   latent stays full-res end-to-end — merging is what the compute sees.
5. Final readout: periphery read from the merged representation (avg-pool →
   bicubic up), fovea untouched. Part of the quality contract — skipping it
   leaves never-denoised HF detail in the periphery at decode.

## Knobs

| flag | default | notes |
|---|---|---|
| `--fovea_sigma_c` | 0.0 (off) | 0.75 is the benched knee. Lowering trades less speedup for a bigger untouched window; raising it past 0.75 changes the image. |
| `--fovea_frac` | 0.35 | Floor 0.25 (the fraction-ladder knee); ≤0.15 falsified on multi-subject prompts (masks drop faces). Speed ceiling ~×1.6 from 2×2 pooling — the next speed lever is a larger merge cell, not a smaller fovea. |
| `--fovea_mask_source` | `combo` | `combo` never loses to a static rect (−29 % subject RMSE on 3-subject hard prompts, tie on centered ones). `cfgdelta` needs CFG — at `guidance_scale 1` it auto-falls back to `x0var`. `rect` = static center rect. |

## Composition / scope (v1)

- Euler only — a stochastic sampler request (`er_sde`/`lcm`) falls back to
  Euler with a warning (noise injection into group-shared periphery states is
  unvalidated).
- Mutually exclusive with `--spectrum` / `--spd` (all replace the denoise
  loop). The foveated-Spectrum compose was closed by bench P3: no headroom at
  sane schedules, baseline collapse at aggressive ones — don't re-propose.
- SMC-CFG / FSG / CFG++ are warn-and-ignored — sampler-boundary
  plug-ins unvalidated against group-shared periphery velocities.
- Composes with LoRA / Hydra / soft-tokens / P-GRAFT — per-step adapter
  setters mirror the standard loop; the per-Linear LoRA delta is
  token-count-agnostic.
- Compile: unvalidated. The merged stack introduces one extra token count
  per mask (outside the tier's `dynamic_seq` band) — a compiled model logs a
  warning and will recompile once per generation. Prefer eager for v1.
- Latent grids not divisible by the 4-px merge cell disable foveation for that
  generation with a warning (free-fit tiers are all divisible; exotic manual
  sizes may not be).

## Invariants (pinned by `tests/test_foveated_merge.py`)

- All-fovea mask ≡ identity — the merge plumbing is a pure permutation there,
  bit-exact against the plain forward.
- Merged periphery velocity is group-shared: every token patch in a merged
  cell carries the identical velocity.
- Merged rope is the exact mean-position rope (synthetic-angle test).
- `score_to_cells` lands on the target fraction post-morphology and never
  emits isolated fovea cells (scattered cells lose their attention
  neighborhood — +47 % subject error in P2).

Bench regression surface (archived): `_archive/bench/foveated/` probes
(`probe_token_merge.py` = speed + startup invariants, `probe_mask_sources.py` /
`probe_fraction_stretch.py` = the p2b/p3s numbers) — they imported the promoted
`networks/foveated.py` code; the live invariant pins are
`tests/test_foveated_merge.py`.
