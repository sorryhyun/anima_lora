# DAVE — DC Attenuation for diVersity Enhancement

Training-free representation-level edit that recovers same-prompt sample diversity. Diffusion samples from one prompt look overly similar because the DC component of intermediate Transformer-block features — the per-channel spatial mean `μ^ℓ` — converges across seeds early and pins the global layout. DAVE attenuates that DC during the early denoising steps, letting the seed-specific AC residual `h−μ` breathe:

```
ĥ^ℓ = α·μ^ℓ + (h^ℓ − μ^ℓ)  =  h^ℓ − (1−α)·μ^ℓ          (α ≤ 1)
```

Unlike everything else in this folder, DAVE is not a sampler-boundary correction — it's a post-`forward` hook on each block (forward_hook-not-override invariant) + a per-step `_dave_cur_sigma` model buffer, the Spectrum/mod-guidance pattern. Branchless (`α=1 → atten=0 → exact no-op`), so it survives `compile_blocks()` (the hook fires eager around each block's compiled `_forward`; the DC mean over dims `(1,2,3)` is correct for both the eager 5D `(B,T,H,W,D)` and the native-flattened fake-5D `(B,1,seq,1,D)` layouts).

Paper: [Breaking the Lock-in: Diversifying Text-to-Image Generation via Representation Modulation](https://arxiv.org/abs/2606.06813) (Kwon, Lee, Choi — KAIST/INEEJI, ICML'26, arXiv [2606.06813](https://arxiv.org/abs/2606.06813)). Local PDF at repo root `2606.06813v1.pdf`.

Read first: `bench/dave/README.md` — the full Phase 0→2e log (premise probe, mask derivation, the dose/window sweeps). The "what generalizes" methodology lesson from the refuted dot-predictor lives in `../findings/spectral_fraction_metric_inverts.md`.

## The mechanism

Per target block `ℓ ∈ pool`, during the early window only, the hook rewrites the output:

```
out ← out − (strength · w(ℓ)) · μ        μ = out.mean(dim=(1,2,3), keepdim=True)
```

- `w(ℓ) ∈ {0,1}` is the flat block-pool mask (`networks/calibration/dave_alpha.npz`), so every pooled block attenuates uniformly and out-of-pool blocks stay exact no-ops.
- `strength = (1−α)` is the fraction of DC removed at a pooled block — the single live dose knob (`--dave_strength`). NB: `strength` is not the paper's α. `α = 1 − strength`; `s0.3 ⇔ α=0.7`.
- The early window is a temporal cutoff τ (`--dave_tau`): attenuate only the first τ fraction of denoising steps. It reconstructs the live `--infer_steps`/`--flow_shift` σ-grid and converts τ → a σ boundary, so "first 10% of steps" tracks the actual schedule instead of a hand-guessed σ.

The edit applies to both the cond and uncond CFG forwards (a uniform representation edit; the hook doesn't distinguish the two passes).

## Why it works on Anima — premise verified

DAVE's decomposition (DC = seed-shared conditioning, AC = seed-specific structure) is real on Anima. The Phase-0 probe (`bench/dave/probe_dc_convergence.py`, 8 seeds × 24 steps, CFG 4.0) measured cross-seed cosine similarity of the disjoint DC vs AC residual per (step, block):

| early steps | DC sim | AC sim | gap |
|---|---|---|---|
| all blocks | 0.989 | 0.482 | +0.51 |
| DC-heavy blocks | 0.9993 | 0.611 | +0.39 |

The DC is near-perfectly cross-seed-shared (it carries the conditioning); the AC that holds seed structure sits at 0.48–0.61. There is an energetically-significant shared channel whose attenuation unlocks the diverse AC.

## Block pool — the paper's flat statistical rule, plus an Anima fix

The shipped mask is the paper's A.1.1 selection (a *flat* pool of blocks whose cross-seed DC cosine clears a threshold), not a power-weighted gate:

```
member(ℓ) = DC_sim(ℓ) ≥ 0.99   AND   gap(ℓ) > 0.03   AND   ℓ ≤ 18      →  blocks 8–18
```

- `DC_sim ≥ 0.99` — the paper's statistical lock criterion (App. A.1.1, BH-FDR-tested).
- `gap > 0.03` — Anima-specific correction the paper lacks. Block 0 has `DC_sim ≈ 0.998` (would pass the paper's rule) but its AC is also seed-locked (gap ≈ 0.017), so attenuating it unlocks nothing; the gap term drops it. (On SD3 the paper keeps block 0 — its AC isn't locked there.)
- `ℓ ≤ 18` — the paper's "exclude final-stage blocks." On Anima those blocks don't just have limited impact: attenuating 19–27 imprints a patch-grid dot artifact straight onto `final_layer → unpatchify` (the subtracted DC makes the next LayerNorm boost the AC, which aliases onto the per-patch grid). Baking the cap into the mask structurally forecloses the dots — the worst DAVE degradation can't reappear via `--dave auto`.

## Defaults and operating points

Shipped defaults: flat 8–18 pool, `--dave_tau 0.10`, `--dave_strength 0.3`. The load-bearing lesson from the sweeps:

> **Text/hand damage tracks the *window width*, not the dose magnitude.** Confined to the first ~10% of steps, even `strength 0.8` holds legible text + clean hands; at the looser `τ0.15`, `strength 0.5` already garbles them. So τ0.10 strictly dominates τ0.15 at equal dose — tighten τ first, then spend the recovered headroom on dose.

| operating point | diversity | text / hands |
|---|---|---|
| τ0.10 · s0.30 (default) | strong (framing / pose / hair vary) | legible text, clean hands |
| τ0.10 · s0.80 | strongest (incl. scene/season swaps) | text legible-but-stylized, hands acceptable |
| τ0.15 · s0.50 | strong | garbles text + hands — avoid |

`0.3–0.8` are all usable at `τ0.10`; pick by how much recomposition you want. The dose default is kept conservative at `0.3`.

## Quick start

```bash
make test DAVE=1                                   # shipped defaults: flat 8–18, τ0.10, s0.3
make test DAVE=1 DAVE_STRENGTH=0.8                 # max diversity (text stylizes)
make test DAVE=1 DAVE_TAU=0.15 DAVE_STRENGTH=0.3   # looser window, stay ≤0.3
python inference.py --dave auto --dave_strength 0.3   ...
```

`SPECTRUM=1` / `MOD=1` / `NOLORA=1` compose into the `make test*` targets as usual. `--dave auto` resolves the shipped mask at `networks/calibration/dave_alpha.npz`; pass an explicit path to use a custom one.

## CLI

| Flag | Default | Notes |
|------|---------|-------|
| `--dave` | unset | Path to a `dave_alpha.npz` mask, or `auto` for the shipped flat 8–18 pool. Enables the hooks. |
| `--dave_strength` | `0.3` | DC fraction removed at a pooled block, `(1−α) = strength·w(ℓ)`. `0` = off, `1` = full DC removal. `0.3–0.8` usable at τ0.10. |
| `--dave_tau` | `0.10` | Temporal cutoff: attenuate only the first τ fraction of steps. Converted to a σ_lo against the live step grid. Overrides `--dave_sigma_lo/hi` when `>0`. `τ>~0.2` posterizes. `0` = off (use the σ window). |
| `--dave_sigma_lo` / `--dave_sigma_hi` | `0.0` / `1.0` | Raw σ window (used only when `--dave_tau 0`). |
| `--dave_block_lo` / `--dave_block_hi` | `0` / `-1` | Additional runtime block-range cap (`-1` = last). The shipped mask already zeroes 19–27, so the defaults are permissive; lower `_hi` only to experiment. |

`make` env levers: `DAVE=1`, `DAVE_STRENGTH=`, `DAVE_TAU=`, `DAVE_SIGMA='lo,hi'`, `DAVE_BLOCKS='lo,hi'`.

## The mask artifact

`networks/calibration/dave_alpha.npz` (loaded by `--dave auto`), derived offline by `bench/dave/derive_alpha_mask.py` from the Phase-0 probe — no GPU run:

| key | meaning |
|---|---|
| `weight` | `(num_blocks,)` flat `{0,1}` pool membership — the only key the runtime reads |
| `pool_blocks` | the member indices (shipped: 8–18) |
| `dc_sim_early` / `gap_early` / `power_ratio_early` | per-block early-window probe stats |
| `dc_thresh` / `gap_eps` / `block_cap` | the membership rule (`0.99` / `0.03` / `18`) |
| `source_probe` | provenance |

Re-derive with `uv run python bench/dave/derive_alpha_mask.py` (reads the latest `bench/dave/results/*/per_block.npz`). The mask must have one entry per block — re-derive from a probe on this model if the block count changes.

## Composition

DAVE is a block-forward hook, not a sampler-boundary plug-in, so it's structurally orthogonal to the corrections in this folder — but the compose is v0 / untested: the shipped scoping is the standard denoise loop only.

| Composes with | Status |
|---|---|
| LoRA / OrthoLoRA / T-LoRA / Hydra | Works — the hook is adapter-independent (it edits block outputs). Diversity premise verified with the adapter on. |
| `compile_blocks()` | Works — branchless no-op at α=1; hook fires eager around the compiled `_forward`; DC mean correct for both 5D and native-flatten layouts. |
| `--spectrum` / `--spd` | Not wired — v0 is the standard loop only. Spectrum caches/forecasts block outputs, which the hook would re-edit on cached steps; needs explicit handling before composing. |
| Mod-guidance (`--pooled_text_proj`) | Mechanism-probed 2026-06-11, orthogonal (interaction ≈ 0 on the lock stats — `bench/dave/README.md` Phase 4). MOD re-aims the seed-shared DC (common-mode, lock-invariant); DAVE de-correlates seeds across basins. Compose when a MOD layout failure survives seed re-rolls (it will — MOD failures are seed-independent by construction). |
| CNS / SMC-CFG | Different seam (sampler boundary vs block forward), so no structural conflict, but not A/B-tested together. |

The hooks are removed after each generation (the model is shared across seeds — a stacked hook set would compound the attenuation); `generate()` arms them from `--dave` and tears them down in a `finally`.

## Faithfulness to the paper, and divergences

`library/inference/corrections/dave.py` follows Algorithm 1 (per-block early-window DC subtraction). Deliberate, evidenced divergences:

| Divergence | Reason |
|---|---|
| Flat pool + `gap>0.03` term (paper: flat `DC_sim≥0.99` only) | Anima block 0's AC is *also* seed-locked, so the paper's rule would wrongly include it; the gap term drops it. |
| `τ=0.10` default (paper: `τ=0.15`) | Window-width governs the text/hand damage on Anima; τ0.10 dominates τ0.15 at equal dose (sweep above). |
| `strength=0.3` default (paper: `α∈[0.2,0.5]` ⇒ strength 0.5–0.8) | The paper's eval is text-free ImageNet/COCO; Anima renders text + anime hands, so the conservative dose is the safe default. The full 0.3–0.8 band is exposed. |
| σ-converted τ (paper: step-index cutoff) | Tracks `--infer_steps`/`--flow_shift` instead of assuming a fixed step grid. |

## Limitations / open questions

- Diversity tool only. DAVE recovers *same-prompt* sample variety; it is not a quality or alignment lever. The remaining cost at higher dose is the diversity↔text/hand-coherence tradeoff.
- Eyeballed, not metric-bench'd. The dose/window verdicts are from 2–3-seed visual sweeps (`bench/dave/eyeball.py`); there's no wired PE-Core/Vendi diversity-at-fixed-alignment number. The paper's own block-attribute analysis uses a VLM judge, not a spectral metric.
- The "analytic dot-predictor" probe was refuted. A per-block spectral patch-imprint metric (`bench/dave/probe_patch_imprint.py`) inverts (rewards no-op blocks) and can't see cumulative dots — see `../findings/spectral_fraction_metric_inverts.md`. The `≤18` cap stands on eyeball + the paper, not that probe.
- No Spectrum/SPD compose (v0). Single-prompt mask calibration (the paper uses many prompts; Table 4 says the pool is dataset-agnostic, so single-prompt is likely fine but unhardened).

## Related code

| File | Role |
|---|---|
| `library/inference/corrections/dave.py` | `setup_dave` / `DAVEHooks` — load mask, set buffers, register per-block hooks, τ→σ conversion. |
| `library/anima/models.py` | `_dave_*` buffers + `reset_dave`; `_dave_cur_sigma` restamped from the timestep each forward. |
| `library/inference/generation.py` | arms DAVE from `--dave`, removes hooks in a `finally`. |
| `library/inference/args.py` | `--dave*` CLI surface. |
| `scripts/tasks/inference.py` | `DAVE=1` make lever + `_dave_flags`. |
| `bench/dave/derive_alpha_mask.py` | offline flat-pool mask derivation → the shipped npz. |
| `bench/dave/probe_dc_convergence.py` | Phase-0 premise probe (read-only). |
| `bench/dave/eyeball.py` | dose/window visual sweeps. |
| `networks/calibration/dave_alpha.npz` | shipped flat 8–18 mask (`--dave auto`). |

## Spin-off — DAVE as a training diversity signal

The diagnostic (cross-seed AC sim = same-prompt mode-collapse detector) ports to training: it's wired as a `validate_every_n_steps` pass in the turbo / DP-DMD loop to catch the canonical DMD collapse the in-loop `div` loss misses. See `bench/dave/README.md` § "DAVE as a training diversity signal" and `scripts/distill_turbo/diversity.py`.
