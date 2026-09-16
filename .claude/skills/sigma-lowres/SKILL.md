---
name: sigma-lowres
description: σ-demoted training (--sigma_lowres) — routing rules and their validation, span schedules, the shipped combolate recipe vs combo, and the sibling-latent precache requirement. Load before enabling, tuning, or debugging sigma_lowres flags or configs/preprocess.toml's sigma_demote.
---

# σ-demoted training (`--sigma_lowres`)

Opt-in: routes each train step's latent grid by noise level — high-σ steps train on a
lower-res sibling latent of the same image. Full contract + measured arm table:
`docs/optimizations/sigma_lowres.md`.

## Routing rules

Certified per-step routes: 1024→896 on σ>0.5 (half-line), 1024→768 on the σ∈(0.65,0.95)
**window** — non-nesting, hence the stacked router.

`--sigma_lowres_route2` takes **priority** over the primary rule and therefore
**requires** an explicit σ window (`_threshold2` + `_threshold2_max`): an unbounded rule 2
shadows rule 1 everywhere and silently disables yarnsig. That, and malformed
spans/routes, are setup-time errors — the surface validates before the model loads.

`--sigma_lowres_yarnsig` rides along by default and applies to the **primary rule only**
(`off` to drop it).

## Spans

`--sigma_lowres_span early|late|spread[:FRAC]` gates by training progress (bias placed
during subspace selection is amplified). `_span2` is the same for rule 2.

- **combolate** — the shipped recipe, what `configs/base.toml` sets: the stacked router
  with `late:0.75` spans on **both** rules. ~−14% wall, ΔW ~0.75 (endpoint weights close
  to native).
- **combo** — both spans cleared (`sigma_lowres_span = ""` / `_span2 = ""`): the
  unscheduled arm E16 measured best at render level (−18%, inside the seed lottery on
  both corpora) but far from native in weight space (ΔW ~0.37).

## Requirements & interactions

- Needs sibling latents precached: `sigma_demote = "1024:896,1024:768"` in
  `configs/preprocess.toml` chains one emit per route onto every VAE pass; `make
  preprocess-demote` emits them. Siblings are **keys inside** the native VAE npz
  (`demoted_{H}x{W}`, one per route, outside the latents namespace) — not separate files.
- Validation stays native; output is an ordinary LoRA.
- In the soup pipeline, `--sigma_lowres*` is a whole-pipeline knob replayed onto the
  uncond run and folded into its name — see the `soup` skill.
