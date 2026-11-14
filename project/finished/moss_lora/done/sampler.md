# ER-SDE and CNS on MOSS-Image (2026-10-17)

**Verdict:** ER-SDE ships as the default (`GenerateRequest.sampler = er_sde`); CNS stays a
bench knob (`sde=1,cns=1`). `denoise(sampler="er_sde")` runs Anima's `ERSDESampler` on
MOSS's snapped shift-5 grid, fed `x̂₀ = x_t − σ·v`; `library/moss/cns.py` recolors its
noise. The shipped γ files (Base and SFT × cfg) and how `MossCNS` picks one:
`docs/inference/cns.md`.

All runs: Base, 1024², CFG 3, uncond skip 22, 50 steps, compiled, seed 42; results
`_archive/bench/moss_spectrum/results/20261017-00{48,105}-cmp_compile/`, images
`/media/sorryhyun/data/moss_image/bench/cmp_compile/`.

## ER-SDE vs Euler

Same cost (52.5 vs 52.2 s), a different sample at the same seed (latent rel-L2 1.22), so a
lean, not a ranking. Over 3 prompts ER-SDE gave sharper line work and denser detail, and
drifted from some prompt details (sunset light mostly gone, blonde → orange, an unasked
yellow tie); 待ってくれ！ stays correct. Spectrum (s50 fw 0.5) keeps its layout: rel-L2 to
its own p50 0.42 / 0.61 / 0.58 vs Euler 0.35 / 0.36 / 0.66.

## CNS

γ from 10 Euler trajectories (5 bench prompts × seeds 0, 1), FFT on the un-patchified 32 ch
/8 grid (`scripts/calibration/moss_cns_calibrate.py`). Precondition holds: σ at γ = 0.5 falls
monotonically 0.94 → 0.41 from lowest to highest band (spread +0.53, aggregate σ50 0.61;
Anima +0.41…+0.52, ≈0.45–0.49).

A/B vs ER-SDE white noise, same draw (5 prompts, p50 and s50 fw 0.5): composition kept,
details move (rel-L2 0.43). Net a near-tie at one seed: p50 + CNS broke the ja_manga eyes;
CNS removed a stray tie fragment (ja_manga s50), restored p50's red neckerchief
(anime_sunset s50) and a green smear (zh_fridge p50); the rest a wash. The calibration
prompts overlap the eval prompts (different seeds).

## With the caststation LoRA

`bench/moss_ab/lora_ab.py --arms euler:0.0,euler:1.0,er_sde:1.0,cns:1.0`,
`caststation_r16` ×1.0, 30 steps, eager, 3 prompts × seeds 1–2; grids
`output/moss/lora/ab_caststation_r16_30st/`. The style survives ER-SDE and renders are
more finished (clean phone hands where Euler has a gloved claw), at some cost to the set's
identity and prompt detail (generic eyes, "sitting on a bench" lost, frontal centred
compositions). CNS vs white again a wash (one signature scribble removed, an "01" tattoo
garbled to "0C").

## Open

- CNS over more seeds and prompts held out from calibration, before it gets a request
  field.
- ER-SDE on SFT.
