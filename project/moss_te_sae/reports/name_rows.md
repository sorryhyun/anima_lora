# The name's rows: where the identity lives, and writing onto it

Scripts: `src/entity_dit.py` (20261027-2108), `src/entity_series.py` (20261028-1309),
`src/entity_add.py` (20261027-2120), `src/entity_patch.py` (20261027-2128). Results under
`/media/sorryhyun/new/qwen35_sae/{entity_dit,entity_series}/<stamp>/` and
`output/tests/malfoid_patch/`. Base DiT, INT8 TE, 832×1216 throughout.

Notation: `E` = the one-step effect being measured (`x̂₀(a) − x̂₀(b)`), `r` = the share of
`E` an arm keeps, `rows` / `emb` = the TE output / the `text_embedder` output.

## Claims

1. **A character's identity sits in the name's own TE rows.** Splicing them into an
   invented-name caption keeps 0.96–0.98 of the name's effect.
2. **The DiT, not the TE, expands the name into appearance**, at the name's own
   positions (hair colour decodable from block ~6, peak block 30), and **the image reads
   it in blocks ~16–31**.
3. **Series is laid out linearly over name vectors** already at the TE output (LOO 0.39
   → 0.65 at block 14; null p95 0.11). Hair colour is not (≈ chance at the TE output).
4. **Writing onto the name rows works partly without training.** Tag directions recover
   1/4–1/3 of the teacher effect for DiT-unknown characters; a series centroid carries
   the Vocaloid outfit grammar (series prob 0.84) better than writing `vocaloid`.
5. **A distilled name-row patch has not passed the gate** (24 % of the teacher's recall
   gain against 80 %); the run's learning rate confounds the two inits.

For the proposal: the target only has to change the name rows (approach B, or a
`text_embedder` / text-position adapter). Nothing inside the TE has to change. The
native path is name rows → DiT blocks 0–30 enrich them in place → the image reads them
in 16–31.

## 1–2. Where the DiT expands a name

**Probe.** Hair colour at the name tokens, 40 characters the DiT should know, 10 colours
(majority 0.125). Directions from `{colour} hair` spans next to an invented name; name
vector = mean over 3 contexts minus the invented names' mean; one `x_t` at σ 0.9 with a
zero latent, so the image tokens carry no character. Tag-direction accuracy, span mean /
last token:

| space | TE out | emb out | block 0 | 4 | 8 | 14 | 20 | 30 | 36 | 41 |
|---|---|---|---|---|---|---|---|---|---|---|
| acc | 0.07 / 0.15 | 0.20 / 0.25 | 0.25 / 0.17 | 0.28 / 0.33 | 0.60 / 0.47 | 0.65 / 0.40 | 0.55 / 0.40 | **0.75** / 0.30 | 0.57 / 0.25 | 0.50 / 0.28 |

The TE output does not put colour on the name in a tag-like direction and the two text
blocks add little; colour appears inside the DiT blocks at the name's positions. Miku's
`aqua` ranks 1st from block 20 on (4th at the TE output). `toujou koneko` and `malfoid`
get `white` on top late, as do several misses: a bias of the white direction, not
knowledge.

**Knockout.** σ 0.9 (at σ 0.6 the name's effect is 1–2 % of x̂₀ and the read is noisy).
`E` = name caption − invented-name caption; one noise seed, two x0 images:

| | `rows` (name rows spliced into the invented caption) | `ctx` (caption without the name rows) | ko from block 0 | 16 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|---|
| miku | 0.97–0.98 | 0.23–0.25 | 0.27–0.43 | 0.23–0.34 | 0.22–0.45 | 0.04–0.59 | 0.95–0.99 |
| neru | 0.87–0.98 | 0.04–0.21 | 0.08–0.27 | 0.07–0.31 | 0.17–0.32 | 0.33–0.46 | 0.91–0.98 |
| teto | 0.96 | 0.01–0.08 | −0.03–0.08 | 0.15–0.26 | 0.25–0.34 | 0.38–0.62 | 0.85–0.98 |

`ko N` zeroes the name rows' effect from block N on. Knocking out from 32 costs almost
nothing, from 28 or earlier most of it; Teto's ko 0–12 ≈ 0 says nothing useful is read
before block 12. `ctx` ≈ `ko 0` everywhere (both are what is left without the name rows).
Single cells are not monotone (Miku ko 28 < ko 24 on one x0); read loosely.

Not checked: a full generation with ko ≥ 32 still draws Miku and ko ≤ 24 does not
(tagger recall, 3 seeds × 2 arms, ~6 min).

## 3. Series geometry in the name vectors

56 characters (Vocaloid family 10 incl. Teto, Touhou 13, Hololive 7, Bocchi 4, Madoka 4,
Re:Zero 3, DxD 3, Frieren 2, 10 singletons as distractors), captions without a `series`
field. Leave-one-out nearest-series-centroid accuracy over the 46 series members (null
from 1000 label permutations: mean 0.04, p95 0.11):

| TE out | emb out | block 4 | 8 | 14 | 20 | 24 | 30 | 36 | 41 |
|---|---|---|---|---|---|---|---|---|---|
| 0.39 | 0.50 | 0.57 | 0.63 | **0.65** | 0.63 | 0.57 | 0.59 | 0.22 | 0.20 |

- Series peaks in blocks 8–20 and fades after the read window (36, 41).
- Lexical overlap inflates part of it: `ram / rem / emilia (re:zero)`, `frieren` /
  `fern (sousou no frieren)`, `meiko / kaito (vocaloid)` share series tokens and have the
  highest alignment (0.55–0.89).
- Hololive and Bocchi members render on model from the name (seed 11, by eye) with low
  alignment to their series centroid (0.1–0.3): known one by one, not through a shared
  direction. Vocaloid and Touhou members align within-series (Spearman of alignment
  against the tagger's series probability +0.55–0.69 and +0.36–0.57).
- The tagger's series tag is not a render-quality read: all ten Vocaloids, Reimu and
  Marisa are drawn on model with series prob ≤ 0.16. Misses by eye: Koneko (generic
  blonde), Toyosatomimi no Miko (generic brown hair), Yorigami Shion (blue hair, wrong
  outfit), Homura and Akeno partly. Too few misses to test "low alignment ⇒ miss"; that
  needs a per-character identity read (reference tags, as in phase 0) on a set with more
  misses.

## 4. Writing onto the name rows without training

**Tag directions** (`entity_add.py`). `N` / `T` = the phase-0 name / name + tags
captions, `E = x̂₀(T) − x̂₀(N)`. The injection adds, to `N`'s name rows at the TE or
`text_embedder` output, the mean of `T`'s added-tag rows minus `N`'s base-tag mean:
`mean` puts the same vector on every name row, `split` deals the tags round-robin; scaled
to ρ × the name-row norm (ρ 0.25–2); `rand` is a random direction at ρ 1. One noise seed,
x0 = the name or the tags generation (seed 11). σ 0.9:

| | best injection | r / cos | rand r / cos |
|---|---|---|---|
| malfoid, x0 tags | emb mean ρ2 (≈ emb split ρ2) | 0.35–0.37 / 0.60 | 0.06 / 0.21 |
| malfoid, x0 name | emb split ρ2 | 0.12 / 0.34 | 0.02 / 0.10 |
| koneko, x0 name | emb split ρ2 | 0.27 / 0.38 | 0.03 / 0.06 |
| koneko, x0 tags | te split ρ2 | 0.25 / 0.33 | 0.10 / 0.17 |
| neru, x0 name | te mean ρ0.5 (ρ ≥ 1 negative) | 0.16 / 0.23 | −0.19 / −0.09 |

- For the DiT-unknown characters tag directions recover a quarter to a third of the
  teacher effect against ~0.05 for a random direction; the gain is still rising at ρ 2.
  `split` ≥ `mean`, `emb` ≈ `te`. For Neru (DiT-known) a large injection pushes the
  existing identity away.
- **Read σ 0.9 only.** At σ 0.75 the random direction already reaches r 0.17–0.33: any
  change to the name rows moves x̂₀ along a generic part of `E`.
- **Distil with the full DiT.** Text rows run without the image tokens differ from the
  joint forward by 13–36 % (all rows, blocks 0–31) and the name rows by 20–56 %, most in
  blocks 24–31, the read window. A text-only proxy is off.

**Series centroid** (`entity_series.py`). `okuda reni` + the series' member mean minus
the invented names' mean, on every name row at the `text_embedder` output, ρ × the
name-row norm. Tagger series prob / mean prob of 4 shared tags, seeds 11 / 22:

| series (shared tags) | invented name | series written in the caption | inject ρ 1 | inject ρ 2 | random ρ 2 |
|---|---|---|---|---|---|
| Vocaloid (headset, detached sleeves, necktie, sleeveless shirt) | 0.00 / 0.10 | 0.26 / 0.19 | **0.84 / 0.74** | 0.50 / 0.56 | 0.00 / 0.00 |
| Touhou (bow, frills, hair bow, ribbon) | 0.00 / 0.01 | 0.04 / 0.25 | 0.13 / 0.45 | 0.09 / 0.43 | 0.00 / 0.00 |
| Hololive | 0.00 / 0.00 | 0.04 / 0.12 | 0.00 / 0.11 | 0.01 / 0.25 | 0.00 / 0.01 |
| Bocchi | 0.00 / 0.08 | 0.02 / 0.21 | 0.00 / 0.01 | 0.00 / 0.00 | 0.00 / 0.11 |

One-step `r` / cos against `E = x̂₀(series written) − x̂₀(invented)` at σ 0.9, ρ 1:
Vocaloid 0.92 / 0.67, Touhou 1.03 / 0.76, Hololive 0.27 / 0.44, Bocchi 0.06 / 0.19
(random 0.08–0.19 / 0.13–0.27).

- The Vocaloid centroid carries the outfit grammar and pulls toward the prototype
  (`hatsune miku` 0.39 / 0.92 at ρ 1: s11 a blonde Vocaloid in Miku's outfit, s22 Miku).
- Touhou transfers more than the tagger says: both seeds are Scarlet-sister-like girls
  (mob cap, frilled dress, red ribbon) against a plain schoolgirl for the invented name.
- Hololive adds animal ears and a tail at ρ 2, nothing else; Bocchi does nothing.
- As a warm start for B: usable for series with a shared visual grammar, and it brings
  the prototype along. A centroid without the best-known member, or with her direction
  projected out, is the variant to try. It says nothing for `malfoid` or Koneko.

## 5. The distilled name-row patch on `malfoid`

A patch on the 3 name rows of the TE output (`m|alf|oid`), distilled from the phase-0
tags caption by the velocity loss through the full DiT: 4 teacher generations (seeds
101–104), σ from the training distribution, 200 steps, Adam at lr 0.02 × the name-row
element rms (3.1 s / step). Generation reads the patched rows from the arm's `_te_cache`
with the phase-0 flags, seeds 11 / 22 / 33.

| arm | val gap σ 0.9 / 0.6 / 0.3 (step 0 → 200) | ‖Δ‖ / ‖row‖ | recall | mean prob |
|---|---|---|---|---|
| name only | — | — | 0.29 | 0.26 |
| `split` init (ρ 2) | 0.96 / 0.99 / 0.99 → 0.46 / 0.38 / 0.87 | 2.0 (stuck at init) | 0.42 | 0.37 |
| `zero` init | 1.0 → 0.66 / 0.99 / 1.04 | 0.32 | 0.42 | 0.29 |
| tags (teacher) | — | — | 0.83 / 0.72 (phase 0 / seeds 101–104) | 0.62 / 0.61 |

- Both arms recover 24 % of the teacher's recall gain; the gate is 80 %. Recall over 8
  tags at 1/8 each does not separate the arms (`sweater vest` and `hoop earrings` are
  missed by the teacher too); the mean probability does: `split` 31 % of the gain, `zero`
  8 %. By eye `split` is ahead: all three Slytherin, two blonde with green eyes; `zero`
  gives a Gryffindor and a dark-haired Ravenclaw-like girl.
- `split` s33 has dark green hair: the tie / eyes / robe green binds onto the hair, a
  superposition binding error.
- The lr confounds the arms: `zero` grew to 0.32 of a row in 200 steps and `split` only
  turned its direction at fixed norm. Next is a larger lr and more steps, not a verdict on
  the init.
