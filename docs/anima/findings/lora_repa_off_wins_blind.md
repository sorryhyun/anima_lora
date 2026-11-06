# Anima LoRA: relational REPA loses blind to the same recipe without it

> **STATUS (2026-11-06).** A/B RESULT + DEFAULT CHANGED. At alpha 512 / lr 1.25e-5 on
> 30-image sets, the recipe without REPA beat the same recipe with relational REPA
> (weight 0.05, layer 8, PE-Spatial) on two artists, **32–16** (pair p 0.015; sign-flip
> over six (artist, seed) clusters p 0.078, 5 of 6 to no-REPA). The fm loss is the same
> in both arms. `use_repa` is off in the LoRA-family configs since 2026-11-06.
> At the shipped alpha 128 / lr 5e-5 the same A/B **tied, repa 23–22** (s40, below), so
> REPA costs the render at alpha 512 / lr 1.25e-5 and nothing measurable at alpha 128.
> It stays off: it gains nothing at either step and needs the PE caches.
> **Owed:** a long run
> ([`../proposal/lora_batch_clip_ema.md`](../proposal/lora_batch_clip_ema.md) §3).

## Why it was run

s34–s36 ([`lora_alpha_lr_drift.md`](lora_alpha_lr_drift.md)) found every arm tied at
matched ||dW||: rank 16 vs 32, alpha 512 vs 128, on 30 and 354 images. Every arm in
them trained with relational REPA on. The question was whether REPA, which pulls the
layer-8 features toward the same PE-Spatial Gram structure in every run, pulls the arms
to the same solution and hides the difference. The first step was REPA itself, on vs off,
in one recipe.

Relational REPA shipped on in `configs/methods/lora.toml` on sample-grid reads and CMMD
([`../methods/repa.md`](../methods/repa.md) § Known results). It had been A/B'd against
the absolute form (relational preferred ~6:4), not against no REPA.

## Setup

- Recipe: the shipped `configs/methods/lora.toml` with `use_timestep_mask=false`,
  `--network_alpha 512 --learning_rate 1.25e-5` (the s35 a512 arm).
- **repa** (control): the s35 `alpha512_{cast,sweet}_s{42,43,44}` LoRAs.
- **norepa** (treatment): the same flags plus `--network_args use_repa=false`,
  `norepa512_{cast,sweet}_s{42,43,44}`. Training logs carry no `repa/*` keys.
- caststation (30 images, 120 updates) and sweetonedollar (31, 124), training seeds
  42–44. norepa rescaled to repa's median ||dW|| of the same (artist, seed) before
  rendering (`match_dw`). Factors: caststation ×1.04 / 0.99 / 0.97, sweetonedollar
  ×1.04 / 0.95 / 0.94. Dropping REPA barely changes how far the adapter moves.
- Blind pairs at the `make test` recipe, four prompts × render seeds 19 / 20, graded in
  `anima-blind-pairs` (`s37_NOREPA_cast`, `s38_NOREPA_sweet`).
- Runner: `bench/anima_abclip/run_ab.py --exp norepa` (s37), `--exp norepa_sweet` (s38).
  Reports: [`report_norepa.md`](../../../bench/anima_abclip/report_norepa.md),
  [`report_norepa_sweet.md`](../../../bench/anima_abclip/report_norepa_sweet.md).

## Result

| artist | seed | repa | norepa |
|---|---|---:|---:|
| cast | s42 | 3 | 5 |
| cast | s43 | 0 | 8 |
| cast | s44 | 5 | 3 |
| sweet | s42 | 2 | 6 |
| sweet | s43 | 3 | 5 |
| sweet | s44 | 3 | 5 |
| **total** | | **16** | **32** |

- s37 (caststation): norepa 16–8, pair p 0.076, cluster p 0.375. It leans on s43 (8–0);
  s44 goes the other way.
- s38 (sweetonedollar): norepa 16–8 again, all three seeds to norepa, pair p 0.076,
  cluster p 0.125 (the floor for three clusters).
- Pooled: 32–16, no ties, pair p 0.015, cluster p 0.078.

By prompt (cast + sweet):

| prompt | repa | norepa |
|---|---:|---:|
| p0 classroom, school uniform | 5 | 7 |
| p1 hetero hug | 5 | 7 |
| p2 maid, tray, cafe | 1 | 11 |
| p3 bikini, beach | 5 | 7 |

p2 carries most of it; p0 split by artist (cast 1–5, sweet 4–2).

## s40: the same A/B at alpha 128 / lr 5e-5

repa = the s31 / s32 `nomask_{cast,sweet}_s{42,43,44}` LoRAs; norepa = the s39 shift1
LoRAs `norepa128_*` (the same flags plus `use_repa=false`), rescaled to repa's median
||dW|| (×0.95–1.06). Render seeds 23 / 24, sets `s40_REPA128_{cast,sweet}`
(`run_ab.py --exp repa128`, [`report_repa128.md`](../../../bench/anima_abclip/report_repa128.md)).

| artist | seed | repa | norepa |
|---|---|---:|---:|
| cast | s42 | 5 | 2 |
| cast | s43 | 4 | 4 |
| cast | s44 | 3 | 4 |
| sweet | s42 | 4 | 4 |
| sweet | s43 | 4 | 3 |
| sweet | s44 | 3 | 5 |
| **total** | | **23** | **22** |

3 pairs left blank. Pair p 0.62, cluster p 0.69 (norepa side). By prompt repa–norepa: p0 4–7,
p1 6–4, p2 5–7, p3 8–4; the maid prompt that carried s37 / s38 (1–11) is close to even
here. Cross-seed MAD repa → norepa: cast 48.5 → 47.1, sweetonedollar 41.8 → 44.5.

So REPA's cost depends on the step: at alpha 512 / lr 1.25e-5 it lost 16–32, at alpha 128
/ lr 5e-5 it ties. s35 (alpha 512 against alpha 128, tied 22–25) ran both arms with REPA
on, so its a512 arm carried a handicap the a128 arm did not; a512 against a128 without
REPA is open (both arms exist: `norepa512_*`, `norepa128_*`).

## Loss

`loss/current` holds the fm loss plus `0.05 · repa/align_loss`
(`library/training/losses.py::_repa_loss`). Logged, norepa reads ~0.0045 lower on every
seed. With the REPA term taken out, the fm loss of the two arms matches:

| caststation seed | repa, logged | repa, fm only | norepa |
|---|---:|---:|---:|
| s42 | 0.0768 | 0.0731 | 0.0723 |
| s43 | 0.0826 | 0.0784 | 0.0781 |
| s44 | 0.0761 | 0.0722 | 0.0719 |

The per-bin curves of the two arms overlap within a seed; the shape follows the seed's
data and σ order. Grad norms are similar (caststation, last 30 steps: norepa 6.4–9.1,
repa 7.6–8.5). REPA changes the render without changing the fm fit, so loss curves cannot
screen it.

## What changed

`use_repa = false` in `configs/methods/lora.toml`, `configs/methods/lora_const_ema.toml`
and `configs/gui-methods/{lora,tlora,hydralora}.toml`, and in the local (gitignored)
`configs/gui-methods/custom/{lora,soup}.toml`, and in `configs/soup/soup.toml`. The REPA knobs stay in place for a run
that turns it back on. Unchanged:

- EasyControl task configs (`configs/easycontrol/*.toml`): relational REPA is the
  documented default for cond ≠ target tasks, a different regime, not tested here.
- `configs/base.toml` already had `use_repa = false`.

## Limits

- One recipe (alpha 512 / lr 1.25e-5). The shipped alpha 128 / lr 5e-5 tied it in s35,
  but REPA on vs off was not run there.
- 30-image sets, ~120 updates. Not tested on a long run (sincos, ~1416 updates).
- One grader.
- T-LoRA, Hydra and soup ingredients inherit the change untested.
