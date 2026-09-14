# plan_wake — wake the DiT's own JA glyph units (forward plan, 2026-09-13)

> **Status (2026-09-13 night).** W0–W2 are done and lifted into
> [`reports/wake_w0_w2_2026_09_13.md`](reports/wake_w0_w2_2026_09_13.md):
> the hypothesis, Probe 0/1, the address geometry, the 256² / 24-kana /
> balanced / σ-band arms, the native-rendering probe, the kanji probe and
> the case for W2d. This file is only what comes next. Nothing is shipped.
> **2026-09-14:** W2d run 1's table was found rank-1; the data lever missed,
> the random-init lever lifted trained singles to 13/24 — see *Run 1b amended*.
> The decorrelation lever (Run 1c, same day) held the table at PR 35 and lost
> the pixels (4/24): rank is not what the DiT reads — see *Run 1c*. The
> free-residual hybrid (Run 1d) then rendered every trained single (24/24,
> 20/24 both readers) with a 0.9-norm per-glyph residual and left held-out
> flat (2/64): the DiT reads near-orthogonal addresses, not shape — see
> *Run 1d*. Trained inventory solved; generalisation is not in this g.
> **Run 2 (2026-09-14 pm):** IDS held-out composites 0/16 — every one renders
> as a clean hiragana (あ お も ち); trained composites render. Addresses do
> not compose; W4 is an exposure budget. See *Run 2 result*.
> **Run 3 (2026-09-14 pm):** word addresses. All 92 kana trained (34/36);
> a single-piece word row renders as a multi-glyph *unit* (します してる きた
> もう, 9/32 at this exposure); a sequence of trained addresses renders
> exactly one of them (`line` 0/32). Real words did not change the string
> verdict — see *Run 3 result*. **Order probe (same day):** the base
> renders nonsense 4–5-piece EN words in order (22/24) — sequence reading
> is pretrained; the rows verdict was drawn from single-unit training. A
> strings arm precedes W3 — see *Order probe*.

## Where the line stands (four sentences)

A frozen 2B DiT with a frozen adapter draws a requested glyph from a
rows-only ext-row delta (kana 24/36 singles, kanji atoms 10/12 on seed 0),
identity is decided at σ ≈ 0.8, and the delta is bit-exact on prompts with
no ext id. Free rows do not scale: every character is its own inversion,
rows carry their blank-canvas layout into scene prompts, and multi-row
captions render one glyph (kana: the first; kanji: the parts fused). The
DiT composes kanji at the component level in pixels, so the glyph manifold
exists — inside the DiT, not in row space, where trained addresses are
near-orthogonal. The way out is to *construct* the manifold: an encoder
from glyph shape to row delta, trained through the frozen DiT, whose
output is a static table = a vocab pack.

## The target artefact

**A vocab pack.** The pack is a forward hook on `llm_adapter.embed` that
swaps `table[ext]` in at ext positions; every arm's delta is `table + Δ`.
The W2d encoder runs once over the inventory (per Qwen piece, rendered as
its own string) and the summed table ships as an ordinary pack: same
safetensors + mapping json, same `vocab_pack` key, read by training, TE
caching, `inference.py`, `GenerationRequest` and the register node. DiT,
adapter, T5, Qwen, routing untouched; EN bit-exact by construction. A new
pack changes the digest → `make preprocess-te ARGS=--overwrite` for CJK
captions; existing LoRAs warn on mismatch (existing machinery).

## W2d — amortized glyph encoder

`g(glyph render) → Δ_row` (1024-d, row-norm units), a small CNN over a
grayscale render of the piece text in a random font each step, last layer
zero-init (step 0 = pack rows) plus one learned shared bias for the layout
mode. Same frozen-DiT FM loss, same σ band 0.7–0.9, same `ExtDelta` hook
(the encoder's table replaces the free `raw`). Every step updates every
character — the exposure grind is what this removes. Rows for pieces in
the eval captions are computed at the end and saved in the `ExtDelta`
format, so `eval` / `native` / `classify` run unchanged.

### Run 1 — kana, held-out split (`--arm encoder --held_out 32`)

`data_w2` (92 kana, 512² renders; the train stage caches `latents_512.pt`).
32 kana drawn by seed never appear in a training item (singles, combos or
corpus lines containing them are dropped); their rows get no gradient
except through the shared weights. Eval adds every held-out kana as group
`single_held`. Cost of record: batch 4, compile, no ckpt; 6000 steps ≈ 45
min + eval ≈ 11 min.

Gates:

- **Trained singles ≥ the rows band rate** (24/36 ≈ 67 %) at ≤ the same
  per-character exposure. Below it the encoder is a worse parametrisation
  and the lr / capacity needs one retune before any verdict.
- **Held-out singles > 0** (both readers, 64 renders). Anything nonzero says
  the map generalises; ≥ 25 % is strong. Zero with trained singles at the
  gate means the encoder is a lookup — see kill criteria.

#### Run 1 attempts so far (2026-09-13 night) — the common mode is the bug

Three launches, none reached eval; all killed on the train log. The
parametrisation, not the idea, is what failed each time:

| attempt | encoder | what the log showed |
|---|---|---|
| `-8fd936` | zero-init head + shared bias, lr 3e-4 | rel 18× row norm by step 450 — every one of the 512 hidden weights behind an output coordinate steps by lr in the same direction, so the output moves 512 × lr per step |
| `-e28bba` | head × 1/64, shared bias at lr 1e-3 | rel grows linearly (36× at step 4300), max/mean 1.00, held == train: the shared bias gets the *summed* gradient of every ext token in the batch, a direction consistent enough that Adam marches at full lr with no restoring force (free rows never had this — each row only saw its own few items) |
| `-02254c` | no bias, per-row norm cap 1.5 by output normalisation | `rel_spread` 0.000 through step 600: at the cap the output is `d / ‖d‖`, the internal `d` keeps growing along the common direction, and the per-glyph part is divided by it |

Attempt 4 (`-137e19`, 2026-09-13 21:36) implements the action below in
`wake_probe.py` (`GlyphEncoder.identity` + `common`, `--lr_common
--common_cap --out_scale`, kill rules `--kill_spread{,_step} --kill_max_row`
enforced in the train loop). **The parametrisation held**: `c` pinned at
the 0.75 cap from step 275 and stayed there, max row ≤ 1.4×, spread 0 →
0.25 by step 650–700 (log kept as `train_log_attempt4_137e19.json`). The
kill fired at step 725 on a bad draw — the training-draw spread swings
0.03–0.26 between logs with the font/shift, so the rule now reads
`rel_spread_ref` (font 0, no shift). Attempt 5 (`-f3fab1`, 21:43), same
launch with that instrument, killed at step 625 too: the reference spread
itself swings 0.01–0.14 between logs and never leaves 0.04–0.12 (log
`train_log_attempt5_f3fab1.json`) — the identity is a per-step Adam kick,
not an accumulating signal, at ~1/10 of the per-row identity the free-rows
arms carried. That is the pre-registered branch below: attempt 6
(`-11ebce`, 21:49), same launch with `--out_scale 0.0625` (1/16), killed
at step 725 by the max-row rule (2.24×): the swing scaled ×4 with the head
(spread 0.005 ↔ 0.39, collapsing to ~0 every ~100 steps at 125 / 225 / 325),
so it is the per-step Adam kick, not the head's reach — the identity never
accumulates (log `train_log_attempt6_11ebce.json`). Attempt 7 (`-507c18`,
21:58) keeps 1/16 and takes the plan's one lr retune: `--lr_enc 3e-5`
(10× down), spread rule off for this run (`--kill_spread_step 0`, growth is
slower by design), max-row rule kept; the log gains `feat_spread` (pooled
conv-feature spread across rows, pre-LayerNorm) to tell a dead-feature
collapse from a weight kick. A mis-launch on the unpatched probe
(`-fc2dc5`) was killed before its first log.

Attempt 7 reached step 2125 (max-row rule, 2.006×) and is the informative
one (log `train_log_attempt7_507c18.json`): the lr fix holds — spread on the
training draw climbs monotonically 0.006 → 1.05, `feat_spread` ≈ 0.45 (no
feature collapse), held rows move with the trained ones (1.2–1.4×) — but
`rel_spread_ref` (every row in font 0) is flat at ≈ 0.1 from step 400 on.
The output tracks **font** ~8× more than glyph: a global mean pool keeps
channel statistics (ink mass, stroke weight — font properties) and drops
the arrangement (glyph identity). Attempt 8 (`-83e576`, 22:16) takes the
input lever, both halves flagged: `--enc_pool spatial` (flatten the 6×6×256
grid → 256, LayerNorm, head as before; 3.4M params) and `--font_mode mean`
(the input is the mean render over all 15 fonts — a font-free descriptor,
so there is no font axis to latch onto and `rel_spread_ref` measures the
same thing as the training draw). lr 3e-5, head 1/16, `c` cap 0.75, spread
rule off, max-row rule 2.5×.

Attempt 8 (log `train_log_attempt8_83e576.json`) — the input lever moved
the right quantity: `rel_spread_ref` == training spread (font-free input)
and climbs **linearly** 0.20 (250) → 0.74 (1000) → 1.76 (2125) with no
deceleration, held rows alongside (2.03×), `feat_spread` ≈ 2 (spatial
features separate the glyphs). Killed by the max-row rule at 2.55× with
nothing saved, so whether that growth is glyph signal or Adam drift (the
free rows saturated at ~1×; a linear climb is what a consistent per-row
direction under Adam looks like) is undecided on the norms alone. The
probe now saves `trained.pt` on a kill. Attempt 9 (`-2c3960`, 22:34,
arm dir `encoder_w2_held32_s2k`) is the same recipe for **2000 steps
straight into eval**, kill rules off — a render at ~1.7× identity decides
signal vs drift and gives the first held-out number. If singles render, the
6000-step run gets a cosine lr decay (the parameter-side bound the identity
lacks); if they do not, the growth was drift and the next lever is the
loss side.

**Attempt 9 rendered (`encoder_w2_held32_s2k/`, 2000 steps, identity
1.6× mean / 2.2× max at the end).** Singles 4/24 trained, **1/64
held-out**, combo 0/36, corpus 0/20, EN 24/24 bit-exact. The sheets decide
signal vs drift: every render, trained or held-out, is a crisp single kana
on a blank canvas — the identity is signal. What is missing is
*resolution*: the map at 2000 steps lands on a handful of attractor kana
(た, ヒ/七, ん, サ, む, え, シ, あ) and the misses are shape-neighbours —
ち→た, み→あ, ノ→人, ヤ→チ, レ→シ, ヨ→も, ネ→え, ニ→ん — i.e. the encoder
already generalises *by shape* to held-out glyphs; it cannot yet tell
neighbours apart. Exposure was 1/3 of the rows arms', so the trained gate
is not judged here. Attempt 10 (`-5535f6`, 23:02, arm dir
`encoder_w2_held32_s6k_cos`) is the full 6000 steps with `--lr_decay
cosine` (all groups, to 0 — the late-growth bound), everything else as
attempt 9. Gates apply to this one.

**Attempt 10 verdict (`encoder_w2_held32_s6k_cos/`, 6000 steps, 44 min +
10 min eval).** Identity plateaus at 2.32× (step ~5000, under the decay),
max row 3.0×, held rows 2.35×. Singles **6/24 trained** (の ひ ち む リ チ,
each on one seed; the other seed lands on a neighbour — い, る, あ, ソ,
テ), **3/64 held-out** (テ once, く both seeds), combo 0/36, corpus 0/20,
EN 24/24 bit-exact. The attractor collapse of attempt 9 is mostly gone
(ヲ ヌ Z フ ね を ス now appear) and the held-out misses are the same
shape-neighbour map at finer grain: ま→も, れ→わ, ケ→チ, ア→テ, ネ→テ,
コ→フ, ク→ン, け→ね, ふ→ん, し→ん, ワ/レ→シ.

- **Trained gate: fail** (25 % vs the 67 % band rate at equal exposure).
  The pre-registered kill rule ("< 50 % after one retune") fires on its
  letter. Its premise does not hold, though: the encoder is not failing to
  *memorise* — every trained row renders a crisp kana at the right
  granularity — it is failing to *separate neighbours*, and the seed
  decides which neighbour. The map is real and low-resolution.
- **Held-out gate: pass, weakly** (> 0, far from the 25 % "strong" mark).
  The generalisation is by shape: a held-out glyph gets its nearest
  trained shape, which is exactly what an amortised map should do at low
  resolution and what free rows can never do.

What was learned about the parametrisation (attempts 4–10, one lever
each): centre the identity and own the layout mode as one capped vector
(4); lr 3e-5 not 3e-4 — Adam's per-step kick on shared weights never
accumulates at 3e-4 (5–7); a global mean pool encodes *font*, a spatial
flatten on a font-free mean render encodes *glyph* (8); the identity is
signal, not drift (9); cosine decay bounds it (10). The remaining lever
is neighbour separation — the loss side, not the optimizer or the input.

**Decision owed (not taken here).** Two roads:

1. *Continue W2d on the loss side.* The shelved same-noise classifier CE /
   swap hinge (W2c) was shelved for free rows because it cannot fix
   combos; the encoder's failure is precisely singles-neighbour
   separation and combos are W3's job, so the reason it was shelved does
   not apply. One run: encoder as attempt 10 + a contrastive term over
   the same-noise batch (the row must lower the FM loss on *its* render
   more than on its neighbours'). Cost ≈ 1.5× a train step.
2. *Take the fallback* the kill rule names: glyph-image conditioning
   (AnyText / GlyphControl shape), gated on a glyph image so ext-free
   prompts stay untouched. Leaves the row space; the vocab-pack artefact
   goes away.

**Decision (2026-09-14): road 1.** The result is a low-resolution map, not
a missing one, and one lever aimed at exactly that remains.

#### Run 1b — neighbour separation (`--arm encoder --contrast hinge`)

Step 0, before any training — **baseline the instrument** (≈ 10 min GPU):
`--stage classify` on `encoder_w2_held32_s6k_cos`. The same-noise N-way
diffusion classifier over the single kana says, in loss space, whether the
attempt-10 rows already prefer their own render (and the reader is the
bottleneck) or confuse the same neighbours the reader shows. If the
classifier is already ≥ 90 % on trained singles, the contrastive term has
nothing to add and the lever moves to the eval side (σ / cfg / steps).

The run, one lever (warm start is not a lever — it is attempt 10's table):

- **Warm start** from `encoder_w2_held32_s6k_cos/trained.pt` (`encoder`
  state + `c`), `--init_encoder`; 3000 steps, lr 3e-5 cosine restart,
  everything else as attempt 10. Same held-out 32 (seed-drawn, unchanged).
- **Swap hinge on the same noise.** For every training item (latent `x`,
  noise `ε`, σ) a second forward with the caption's ext row swapped to a
  *negative* row `n`: `L = L_FM(r) + λ · max(0, m + L_FM(r) − L_FM(n))`,
  per-item mean MSE, λ = 1, m = 0 to start (the FM loss is ≈ 0.04 and a
  rendered glyph moves it by ≈ 0.005; a margin is the second run's knob,
  not the first's). Gradient flows into the encoder through both rows: the
  right row explains its render better than its neighbour's row does.
- **Hard negatives from the encoder's own table**: `n` = the trained row
  with the highest cosine to `r` in the current `delta.raw` (189×189, free
  per step), with probability 0.5; otherwise a uniform trained row. **Never
  a held-out row** — a negative gets a direct gradient and would break the
  held-out test.
- **Cost**: two DiT forwards per step, batch 4 each (batch 8 OOMs at 512²
  without ckpt) → ≈ 2× attempt 10's step, ≈ 45 min for 3000 + 10 min eval.
- **Instruments**: `swap_acc` (fraction of items with `L_FM(r) < L_FM(n)`,
  logged every 25 steps — the direct measure of neighbour separation;
  attempt 10's table gives its step-0 value), `rel_spread_ref`, `rel_max`,
  `c`. Kill if `rel_max` > 4× (the hinge can push rows apart without bound)
  or `swap_acc` has not moved by step 1000.

Gates, in order: (1) `swap_acc` ≥ 0.9 on trained items by the end —
otherwise the term is not doing its job and no render verdict is read;
(2) trained singles ≥ 12/24 (double attempt 10; the 67 % band rate is the
run-2 gate now that exposure is not the variable); (3) held-out singles
> 3/64 with the shape-neighbour structure intact. A pass on (1) with a miss
on (2) says the loss-space separation does not reach the pixels at σ
0.7–0.9 — then widen the band to 0.5–0.9 (identity at σ ≈ 0.8, detail
below it) as the one follow-up before the fallback. A miss on (1) is the
kill: the frozen-DiT loss cannot rank neighbours, and road 2 follows.

#### Run 1b amended (2026-09-14) — the table is rank-1; two levers spent, one hit

Before the hinge, the attempt-10 table was read on CPU (SVD of the centred
189×1024 `delta.raw`; both numbers now logged every 25 steps as `table_pr`
and `nn_cos`):

| quantity | encoder, attempt 10 | free rows (band / few / k24) |
|---|---|---|
| participation ratio, centred table | **1.0** (top sv 32.7, next 3.6) | 58 / 28 / 25 |
| cos of every confused pair (ち/た, ま/も, ケ/チ, レ/シ …) | ≥ 0.95 | ~0.04 pairwise |
| nearest-neighbour cos over the 121 kana | ≥ 0.84 | — |
| PR of the zero-init last head layer | 2.8; top direction cos 0.998 with the table axis, 0.85 with `c` | — |

The encoder's identity was a scalar coordinate on (nearly) the layout axis
— why neighbours share an attractor, the seed decides, and a held-out glyph
gets its nearest trained shape. Mechanism: the per-item FM gradient on the
row is a shared direction with an item-dependent magnitude; centring
removes the mean magnitude, not the direction, and Adam on shared weights
marches a sign-consistent direction (∝ lr·steps) while the per-glyph tail
random-walks (∝ lr·√steps). Free rows escape because each row's shared
component saturates on its own, then the tail is all that is left.

Two levers, one run each, same recipe as attempt 10 (6000 steps, cosine,
spatial pool, mean font, held-out 32 by the same seed). Both rulers below:
`report.md` (sfx exact, min over boxes) and the largest-box both-reader
count in brackets.

| run | data | `table_pr` end | `nn_cos` | trained singles | held-out | EN |
|---|---|---|---|---|---|---|
| attempt 10 (`encoder_w2_held32_s6k_cos`) | `w2` | 1.03 | 0.996 | 6/24 [6] | 3/64 [2] | 24/24 |
| **data lever** (`encoder_w2l_held32_s6k_cos`, job `-9c2967`) | `w2l` = `--layout jitter` | 1.03 | 0.996 | 2/24 | 2/64 | 24/24 |
| **rank lever** (`encoder_w2_held32_s6k_cos_rinit`, job `-06e5e1`) | `w2` | 23 → 1.24 | 0.64 → 0.94 | **13/24 [10]** | 5/64 [3] | 24/24 |
| **decorrelation** (`encoder_w2_held32_s6k_cos_rinit_decor`, job `-e34bdb`, `--decor 0.02` on rinit) | `w2` | 23 → **35** | 0.64 → 0.57 | 4/24 [3] | 4/64 [3] | 24/24 |
| **free residual** (`encoder_w2_held32_s6k_cos_rinit_fres`, job `-8928c4`, warm start rinit + `--free_residual 1e-3`) | `w2` | 1.3 → 1.2 (g 1.1) | 0.93 | **24/24 [20]** | 2/64 [2] | 24/24 |

- *Data lever — miss.* `--layout jitter` (random position, size 60–200,
  ink colour, 25 % outline, dark backgrounds, bubble of random size and
  place; `data_w2l`, same item counts as `w2`; the `v1` path rebuilds
  bit-identically) left the table rank-1 for the whole run and `c` at the
  cap. The shared direction is not the constant layout: every item wants
  the same "draw a glyph" mode whatever the canvas. Keep the jitter data
  for scene survival later; it is not a rank lever.
- *Rank lever — hit on the pixels, not on the rank.* `--head_init random
  --init_spread 1.0` (default full-rank init on the last head layer,
  rescaled so the step-0 identity spread is one row norm) starts at PR 23
  on the real renders and the march still takes it to 1.24 by step 2400 —
  but the residual full-rank part (`nn_cos` 0.94 instead of 0.996) doubles
  the trained singles: の ひ は on both seeds, む せ リ チ on one; misses are
  still neighbours (ち→ろ, ぬ→ね, ン→シ, キ→チ) plus two reader misses (み,
  せ drawn right). Held-out is the same shape map (う レ exact; と→を, ま→も,
  れ→わ, テ→チ, こ→く; た お drawn right, misread). `feat_spread` stays 0.15
  (attempt 10: 3.3) — with a full-rank head the conv features no longer
  need to spread. Gate (2) of Run 1b (≥ 12/24) passes on `report.md`,
  misses by two on the both-reader ruler.

**What this settles.** Neighbour separation tracks the table's residual
rank, and the loss never asked for that rank — the FM gradient's consistent
direction eats it under Adam. The hinge as pre-registered would fight the
same march at 2× the step cost, and its hard-negative rule (highest cosine
in `delta.raw`) is void on a table where every cosine is 0.9+.

**Next lever (one run): a decorrelation penalty on the table.** Keep the
random-init recipe and add `λ · mean_{i≠j} cos²(r_i, r_j)` over the centred
trained rows (189×189 per step, no extra DiT forward; λ ≈ 0.02 against an
FM loss of ≈ 0.04, so the term is ≈ ½ the FM loss at PR 1 and vanishes as
rows spread). It opposes the march whatever direction it takes, and it
targets exactly the free-rows geometry (pairwise cos 0.04) that rendered
24/36. Instruments as now; gate = `table_pr` ≥ 10 at the end **and** trained
singles ≥ 12/24 both readers; held-out > 3/64 with the shape map intact.
If PR rises and singles do not, the rank is not what the DiT reads and the
hinge (on a table that now has structure) is the run after; if both rise,
the encoder is at the free-rows band rate and Run 2 (kanji, IDS split)
opens.

#### Run 1c — decorrelation penalty (2026-09-14): the PR gate passes, the pixels go

`--decor 0.02` = `λ · mean_{i≠j} cos²` over the centred *trained* rows of the
encoder table each step (held-out rows excluded from the centring and the
pairs; no extra DiT forward; `decor` / `loss_total` logged with `table_pr`),
on the rinit recipe unchanged (job `20260914-074117-e34bdb`, 54 min).

| quantity, end of run | attempt 10 | rinit | **decor** |
|---|---|---|---|
| `table_pr` | 1.0 | 1.24 | **34.8** (never below 29) |
| `nn_cos` | 0.996 | 0.94 | 0.57 |
| `rel_spread_ref` (identity, row norms) | 2.32 | 2.27 | **0.41** |
| `rel_max` | 3.0 | 3.8 | 1.08 |
| `feat_spread` | 3.3 | 0.15 | 0.03 |
| trained singles (report / both-reader) | 6/24 [6] | 13/24 [9] | **4/24 [3]** |
| held-out | 3/64 [2] | 5/64 [4] | 4/64 [3] |

- *The penalty did its job on the rank at ≈ 1 % of the FM loss* (`decor`
  ≈ 0.018, λ·decor ≈ 4e-4): the march that took rinit from PR 23 to 1.8 by
  step 950 never started — PR 38 at the same step, 35 at the end. It is a
  barrier whose gradient grows on collapse, so it need not be large.
- *And the renders collapsed with it.* Trained singles 4/24: の リ hold,
  everything else is a clean, well-formed **wrong** kana (ひ→ね, ち→る,
  は→に, ぬ→む, せ→え, ン→ヲ, キ→あ) and seed 1 drifts to a handful of
  attractors (ま, ア, ス, デ) across trained and held-out alike. These are no
  longer shape-neighbours — the row points at "some kana" and the DiT rounds
  to the nearest well-formed one. Not reader misses (sheet checked).
- *What was lost is the identity magnitude, not its direction.* Both tables
  that render (attempt 10, rinit) grew their reference spread to ≈ 2.3 row
  norms and their largest row past 3×; decor froze the spread at 0.41 (below
  the step-0 1.0) and the max row at 1.08. The spread grows *through* the
  march — the shared direction with item-dependent magnitude is what the DiT
  reads as "which glyph, how hard" — so a term that opposes the march
  opposes the growth, whatever it does to the cosines. Decorrelated
  directions in the 1024-d row space are DiT-null: rank in row space and
  separation in pixel space are different quantities, and the loss only
  ever asked for the second.

**Verdict (pre-registered branch taken).** PR rose, singles fell: the rank
is not what the DiT reads. The lever that counts is separation *in loss
space*, i.e. the hinge — the run after, as written. Two amendments from
this run: (1) warm-start the hinge from **rinit's** table (the one that
renders, spread 2.3), not decor's; (2) the hard-negative rule "highest
cosine in `delta.raw`" is void on rinit too (nn_cos 0.94, every pair ≥
0.85) — draw negatives from the **reader-confusion pairs** the two runs
agree on (ち/ろ, ぬ/ね, ン/シ, キ/チ, ま/も, テ/チ, こ/く, と/を) with
probability 0.5, uniform trained otherwise. Do not combine `--decor` with
the hinge; it is a spent lever (kept behind its flag as an instrument).

#### Run 1d — free-residual hybrid (2026-09-14): trained solved, held-out flat

`row_i = g(glyph_i) + f_i` on the trained rows (`--free_residual μ`,
μ = 1e-3, `--lr_free 1e-3`; held-out rows get `g` only, no residual and no
gradient), `L = L_FM + μ · mean_i ‖f_i‖²`, `g` warm-started from rinit
(`--init_encoder …/rinit/trained.pt`), everything else the rinit recipe
(job `20260914-084654-8928c4`, 55 min). The semi-amortised fix for the
amortisation gap Run 1c exposed: `f` carries the per-glyph magnitude the
shared head cannot without collapsing; the L2 pushes whatever `g` can
explain into `g`.

| quantity, end of run | rinit | **fres** |
|---|---|---|
| trained singles (report / both-reader) | 13/24 [9] | **24/24 [20]** — the 4 are vl16 misses on ぬ/キ, drawn right |
| held-out | 5/64 [4] | 2/64 [2] |
| combo / corpus CER (sfx) | 0.98 / 0.95 | 0.71 / 0.83 (still one glyph per string, now a *correct* one; ムツ drew both) |
| `free_norm` mean / max (row norms) | — | 0.40 / 1.16 (the 12 eval kana ≈ 0.9) |
| `free_ratio` (f / g spread) | — | 0.17 |
| `g_pr` | 1.24 | 1.1 (the march continued from the warm start) |
| EN | 24/24 | 24/24 |

**What `f` is** (CPU, `trained.pt["free"]`, 151 rows): PR 26 (35 centred),
pairwise |cos| 0.06, nearest-neighbour cos 0.27; cos to `g`'s principal
axis 0.07, cos(f_i, g_i) 0.10. Every reader-confused pair has cos(g) ≥ 0.97
and cos(f) 0.24–0.42 (ち/ろ 0.24, ぬ/ね 0.31, ン/シ 0.42, キ/チ 0.35), at
|f| ≈ 0.9. That is the **free-rows geometry** (W1: pairwise cos 0.04, PR
25/37) reappearing as the residual: the component the DiT reads as *which*
kana is a near-orthogonal per-glyph address of ≈ 0.9 row norm, orthogonal
to everything `g` produces. `g` is a rank-1 "draw a kana" prior plus a
tail the DiT barely reads; it never held the identity, which is why decor
(more `g` rank) and rinit (more `g` tail) moved the pixels so little and a
0.9-norm residual moved them to 24/24. The weak shape structure in `f`
(confusable pairs 0.3–0.4 vs 0.06 mean) is real but far from an address.

**Gates.** (2) trained ≥ 12/24 both-reader: **pass** (20/24). Ratio < 1:
pass (0.17). Held-out > 3/64: **miss** (2/64 — within noise of 3–5/64 over
every encoder arm; held-out glyphs now round to the sharpened *trained*
basins: ま→た/あ, ヨ→あ/ま, ケ→テ/チ). Read flat, not worse.

**What this settles.**

- *The trained-inventory recipe exists.* For any inventory that appears in
  training, `g + f` renders every single (24/24) at 6000 steps with a
  ≤ 1.2-norm residual — no per-character inversion grind, one table, EN
  bit-exact. A kana pack is buildable today from this run's `trained.pt`.
- *Generalisation is not in this `g`.* The identity the frozen DiT reads is
  an address, not a shape coordinate; a shape→row CNN trained through the
  FM loss learns the prior, and every rank / geometry lever on it (data
  jitter, rinit, decor, now the residual's L2 pull) leaves `g_pr` ≈ 1 and
  held-out at 2–5/64. The hinge would sharpen trained neighbours that `f`
  already resolves — **drop it** (Run 1b/1b-amended closed).
- *Composition is the open question, not kana held-out.* Whether a
  composite's address relates to its atoms' addresses is a question about
  `f`-space, and `probes/wake_geometry.py --pairs` reads it directly. Run 2
  (kanji atoms + composites on blank canvases, IDS held-out) is where that
  is answered; the trained side of Run 2 is no longer at risk.

**Next (one run): Run 2 on blank canvases with the hybrid** — kana + the 24
structured kanji + top-200 corpus kanji, IDS-structured held-out, recipe =
fres (warm start `g` from fres, `--free_residual 1e-3`). Gates: trained
kanji singles ≥ 67 % both-reader (the hybrid's own bar now), held-out
composites > 0, and the geometry read: cos(f_明, f_日 + f_月) against the
0.06 floor. Composites at 0 with cos at the floor is the W4 kill (addresses
do not compose; every kanji needs exposure, and W4 is an exposure budget,
not a research line). Scene composites (the layout-prior fix) are the run
after, on whichever table survives.

### Run 2 — kanji + scene composites (opened by Run 1d on the trained side; held-out kana stayed flat)

Inventory kana + the 24 structured kanji + the top-200 corpus kanji with
an IDS-structured held-out split (composites whose atoms are trained).
Data changes from blank canvases to glyphs pasted into real dataset images,
caption = the image's caption + the clause, so the row can only explain
the glyph — the layout-prior fix both product goals need. One lever per
run: kanji first on blank canvases if run 1 is marginal, composites first
if it is clean.

Gate: held-out kanji singles > 0 and native-rendering scene survival
(scene kept on ≥ 75 % of the trained-clause renders, glyph present ≥ 50 %).

#### Run 2 result (2026-09-14): addresses do not compose — W4 is an exposure budget

Blank canvases, one lever. `data_k2` = kana 92 + the 24 structured kanji
(116 chars, 1875 items: 1396 font, 479 corpus); the top-200 corpus kanji
were **left out** so per-char exposure matches the kana runs (with 316
chars it drops ≈ 5× and a trained-gate miss would be unreadable). Held-out
by hand (`--held_out_chars 明休男岩加相困森`): eight composites whose atoms
日月木人口女子田力山石目 all stay trained, with 林 品 晶 好 trained as
composites so the DiT sees composition; 1708/1875 items kept. Recipe = fres
(`g` warm-started from fres, `f` from zero, μ 1e-3), job
`20260914-094804-992cac`, 44 min + 8 min eval.

| group | report | both-reader | note |
|---|---|---|---|
| trained singles (12 atoms + 林品晶好 + あい, ×2 seeds) | 21/36 | 19/36 | atoms 15/24 [14]; misses are *repeats* (木→木木木, 口→ロロ, 田→日日, あ→ああい), 力→カ (same shape), 晶→昌 |
| held-out composites (×2 seeds) | **0/16** | 0/16 | 明→あ, 休→あ, 男→も, 岩→お, 加→あ, 相→ち, 困→あ, 森→あ — clean hiragana, not one kanji stroke |
| combo / corpus | 0/36 / 0/20 | | one glyph per string, as before |
| EN | 24/24 | | |

Instruments as fres: `free_norm` 0.46 / max 1.42, `free_ratio` 0.22,
`g_pr` 1.1, g spread 2.1. f-space composition on the trained composites
(`wake_geometry.py --table free`): 林 ~ 木 cos 0.48 (pct 99), 好 ~ 女+子
0.30 (pct 94), 品 / 晶 at chance vs their atoms but 0.57 to *each other*.

**Reading.**

- *The generalisation question is answered, and not by exposure.* A
  held-out composite's row is `g` alone, and `g` alone draws the prior — a
  kana. Not a wrong kanji, not its atoms fused (the free-rows kanji probe
  fused parts when the rows existed): the shape→row map carries no
  "kanji made of 日 and 月". The f-space structure that does exist is
  *visual similarity* (林 looks like 木, 品 looks like 晶), the same
  confusable-pair correlation fres showed — an address neighbourhood, not
  composition.
- *The trained side regressed and is a confound on the bar, not on the
  verdict.* 19/36 both-reader (53 %) misses the 67 % gate; あ い went 0/4
  where fres had every kana. `f` restarted from zero on a 304-row table
  with 1708 items, and the repeated-atom composites (林 品 晶) taught a
  "repeat the glyph" mode that leaks into atoms (木→木木木, 田→日日). The
  held-out verdict does not depend on it: trained composites 林 好 render
  (4/8), held-out ones render as kana.
- *W4 by the pre-registered branch: an exposure budget, not a research
  line.* Every kanji needs its own `f_i` with exposure; at the kana runs'
  per-char exposure (≈ 800 gradient touches per char over 13 epochs) the
  jōyō 2,136 is ≈ 0.6 M item-views ≈ 150 k steps ≈ **one GPU-day** at 2.3
  it/s, in one table (hybrid, `g + f`), EN bit-exact. That is a budget
  decision, and the repeat-mode leak says the data for it should not
  over-weight repeated-atom composites.

**What the wake line has, closing W2d.** (1) A trained-inventory pack
recipe (`g + f`, Run 1d) that renders every trained single at 6000 steps;
(2) the finding that the frozen DiT reads near-orthogonal per-glyph
addresses that neither rank levers nor shape encoders nor IDS structure
predict; (3) the exposure budget above for kanji at scale. W3 (strings,
DiT-side composition) is unchanged and is the next product step; the
scene-composite data (layout-prior fix) rides on whichever table W3
trains against.

### Run 3 — word addresses (2026-09-14 pm): a row can be a word; a sequence is still one row

**Why it was owed.** Every combo / corpus eval so far scored strings whose
pieces were *not all trained rows* (corpus coverage 19/28), so "one glyph
per string" had never been measured on a sequence of trained addresses,
and every combo was a random kana salad. Two hypotheses, one run: real
words as addresses, and trained-address sequences.

**What a "word address" is.** The pack's ext rows are Qwen pieces and Qwen
already merges many JA words into one piece (ありがとう / 行く / 明日 / いい /
って → one row each; 大丈夫 → 大+丈夫, ドキドキ → ド+キ+ド+キ). A word address
is therefore an *existing* row put into the trained inventory with the
whole-word render as `g`'s input — no pack change, no cache digest change.
The corpus bubbles (`render/ja/resized/boxes.jsonl`, 3 607 lines) hold
1 417 distinct multi-char single-piece words (10 503 occurrences; the top
60 cover ≈ 44 %); only 17 lines are a single piece, typical short lines are
2–5 pieces.

**Data `wd`** (`--words 120 --held_out_words 8 --n_single 40`): kana 92 +
the 120 most frequent single-piece words (mostly grammar: って いい から
ちゃ して じゃ ない …; content words 気持ち 好き ちょっと), 8 held out by seed
(えて かい こと これ だけ なら にも ません); 40 renders per row (8 860 font
items) + the 132 / 641 corpus lines whose every piece is a trained row
(coverage 100 % by construction). Bubble renders now shrink to fit the
ellipse (可愛い overflowed at 400/n px; singles unchanged). Eval adds
`word` (16 trained words), `word_held` (8), `line` (16 held-out corpus lines
of 2–3 pieces, every piece trained: いやいい = いや+いい, なにそれ = なに+それ,
ほらそれ, さわって, フフフ …). Recipe = fres, 8 000 steps, `g` and the 134
overlapping `f` rows warm-started from Run 1d (`--init_free`), word `f`
from zero. Job `20260914-111212-d1542c`, 58 min + eval 15 min; arm
`encoder_wd_w120_s8k_fres_warm`.

| group | n | exact (sfx) | note |
|---|---|---|---|
| single (18 kana × 2) | 36 | **34/36** | all 92 kana trained — the kana pack exists; misses と→こし, れ→ね |
| word (trained) | 32 | **9/32** | します 2/2, してる 2/2, きた, こう, いや, もう, ッド; misses: first glyph only (こう→こ, ただ→た, した→し, には→に), repeats (でも→てもも, もう→もうう), one-glyph collapse (こんな/そんな→ん) |
| word_held | 16 | 0/16 | `g` alone on a word render draws text salad, not even a kana |
| **line** (2–3 trained pieces) | 32 | **0/32** | every render is *one* piece: そオニ→ニ, ほらそれ→ら, さわって→わ, なにそれ→な, いやいい→いい, ちゃく→く, めちゃ→め, フフフ→フ |
| combo / corpus | 36 / 20 | 1/36 / 0/20 | as before |
| EN | 24 | 24/24 | bit-exact |

Instruments: `rel` 2.39 → 1.45, `rel_spread_ref` 2.2 → 0.78, `free_norm`
0.23 → 0.70, `free_ratio` 0.90 (Run 1d: 0.17), `table_pr` 1.1 → 2.8, `g_pr`
1.09. With 246 rows the identity moved almost entirely into `f`; `g`
shrank to a prior.

**Reading.**

- *A row is a unit, and the unit can be a word.* します and してる (three
  glyphs) render from one ext row on both seeds; きた こう いや もう ッド on
  one. So the frozen DiT's "one glyph per address" is really "one *unit*
  per address" — an address can carry a short word's layout. That is new,
  and it is a per-row exposure question (words had 40 renders + a few
  corpus crops and `f` from zero; kana had a warm start). 9/32 misses the
  67 % gate; the miss modes (first glyph only, repeats) are the
  under-exposed row rounding to the nearest kana basin or the repeat mode
  Run 2 showed.
- *A sequence of trained addresses renders one of them.* `line` 0/32 with
  coverage 39/39 is the clean version of the string verdict: the DiT does
  not enumerate ext tokens in a clause, it draws one. Which one is not
  positional (last: そオニ→ニ; second: ほらそれ→ら, さわって→わ, いやいい→いい;
  first: なにそれ→な, めちゃ→め) and not obviously frequency. Random combos
  were not the reason; real words in real order fail the same way. **The
  sequence question is closed on the rows path: strings are W3.**
- *Held-out words are worse than held-out kana.* `g` on a word render
  produces salad, not a kana — the shared encoder has no word prior. Word
  addresses are lookup, one row per word, exposure per row.

**What this adds to the pack.** (1) The full kana inventory renders
(34/36) with words in the same table — the 92-kana pack is this run's
`trained.pt`. (2) A common-word pack is buildable the same way, at a cost
that is per-word exposure; a words-only continuation warm-started from
this run (`f` now non-zero for the 112 words) is the one cheap lever to
see whether words reach the kana bar. (3) Nothing here moves strings; a
2-piece bubble needs the DiT to read two addresses, which is W3's
ext-gated cross-attention LoRA.

**Decision owed.** Whether to spend ~1 h on the words-only continuation
(product value: single-word bubbles from a common-word pack) before W3,
or go to W3 on the kana+word table as is.

### Order probe (2026-09-14 pm): the frozen adapter + DiT read piece sequences in order

**Why it was owed.** Every "sequence" verdict above was drawn from ext
rows, while the EN control was read as a single-word sanity check. The
T5 tokenizer says otherwise: HELLO = ▁H·ELL·O, SORRY = ▁S·OR·RY, HELP =
▁·HE·LP — the EN 24/24 was already a multi-piece result. Real words could
still be recognised as wholes by the adapter's self-attn, so the
discriminating probe is *nonsense* multi-piece words, base model, no
delta: `probes/order_probe.py` (job `20260914-131417-18883e`, 42 renders
in 2.5 min; `output/wake_probe/order_probe/`).

| group | n | exact (sfx / vl) | misses |
|---|---|---|---|
| real (HELLO STOP SORRY HELP PLOVEN OSTREB) | 12 | 11 / 11 | OSTREB s0 (a stray "las" before it) |
| nonsense, 4–5 pieces (GLORPAX MIZUKANE SUMIMASEN TOBRINEK VASQUOLM PILDROME KANTOBRE ZEMURIAL DRAVOKIN NURPELTA OKTABRIS FELMUNDO) | 24 | 21 / 22 | KANTOBRE→KANTOBBFF, OKTABRIS→OKTAABIRS (pieces present, tail scrambled) |
| two words (ZORP KAV, BLIM TOK, WAY NO) | 6 | 6 / 6 | — (WAY NO renders in the *given* order, not as NO WAY) |

**Reading.** A sequence of four or five T5 addresses the DiT has never seen
as a word renders as that sequence, in order, at 512² with the same
template, cfg and steps as the wake evals. So "the DiT does not enumerate
tokens in a clause" and "token → position binding was never learned"
(W2 § what W2 settled, W3) are wrong as general statements: enumeration
and order are a pretrained capability of the frozen adapter (self-attn +
cross-attn, both RoPE'd) and the frozen DiT. What fails is *ext-row*
sequences, and the ext rows differ from EN pieces in exactly two ways —
they are off-manifold random directions the adapter was never trained to
contextualise, and every arm trained them on single-unit canvases
(Run 3: ≈ 1.5 % multi-piece items), so the delta carries "one centred
unit" and nothing ever asked it to be contextualisable. The string
question therefore reopens on the rows path, cheaper than W3:

- **Strings arm** (next, ~1 h): warm-start the Run 3 table (identity is in
  `f`), train on 2–4-piece random-order strings only — no singles — and
  score `line` plus order-flipped pairs (なに vs にな). Non-zero `line`
  means strings are reachable as a static table; still 0 means the frozen
  adapter cannot contextualise off-manifold rows, and *that* is the case
  for W3 (or for slot rows: Δ(piece, i) = Δ_piece + P_i, tokenizer-side,
  still a pack).
- `order_probe.py` is the standing EN control for order; every future arm
  should carry a nonsense multi-piece group so order reading is measured
  for free.

### σ diagnostic for strings (2026-09-14 pm): order is read at σ 0.5–0.8, nothing above 0.9

Two `--stage classify_str` runs (same-noise diffusion classifier, the true
caption S vs named alternatives, both conds; win = S has the lower error,
gap in spread units; jobs `-19743b` JA, `-4a8071` EN; reports under
`encoder_wd_w120_s8k_fres_warm/classify_str{,_en}/`).

**JA — 48 2-kana + 16 3-kana strings of Run 3's rows** (pairs chosen so
every permutation tokenizes to its own kana rows, never a word piece):

| contrast | σ 0.35 | 0.50 | 0.65 | **0.80** | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|
| identity (S < substituted) | 0.34 | 0.59 (+0.5) | 0.56 (+0.9) | **0.77 (+1.3)** | 0.48 (+0.5) | 0.34 | 0.33 |
| count (S < one piece) | 0.50 | 0.41 | 0.48 (+0.8) | **0.72 (+1.1)** | 0.58 (+0.8) | 0.42 | 0.19 (−1.3) |
| order (S < reversed) | 0.44 | 0.58 | 0.58 | 0.64 (+0.1) | 0.55 | 0.61 | 0.47 |

Floor (delta off) is chance or worse everywhere. Identity and count in
the trained rows live at σ 0.65–0.9 (the singles band). **Order is chance
at every σ** — the rows carry no order signal, so they cannot say where
the DiT would read it.

**EN — 24 nonsense two-word strings ("LIZOFUV RIZO"; a word swap keeps the
piece multiset), base model** (no ext id, so trained ≡ floor):

| contrast | σ 0.35 | 0.50 | **0.65** | 0.80 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|
| order (S < swapped) | 0.69 (+0.4) | 0.94 (+1.9) | **0.98 (+2.5)** | 0.90 (+1.1) | 0.54 | 0.44 | 0.31 |
| count (S < one word) | 0.73 (+1.3) | 0.96 (+1.5) | **0.98 (+2.1)** | 0.90 (+2.0) | 0.69 (+1.1) | 0.25 (−0.5) | 0.06 (−1.2) |
| identity (S < substituted) | 0.73 (+1.4) | 0.98 (+1.8) | 0.98 (+1.8) | 0.83 (+0.8) | 0.38 | 0.23 | 0.27 |

**Reading.** In the base, order / count / identity of a piece sequence are
all decided at **σ 0.5–0.8** (peak 0.65) and there is no signal at σ ≥ 0.9.
The "one-piece caption wins at σ 0.95–0.99" pattern shows up for EN too, so
it is a near-pure-noise artefact (fewer tokens = lower error), not a
collapse decision at high σ. Consequences for the strings arm: the band
moves **down**, not up — `--t_min 0.5 --t_max 0.9` covers the order window
and the kana identity peak (0.8); widening above 0.9 buys nothing. The
sampler is a hard affine remap (sigmoid density inside, zero outside), so
the band *is* the weighting.

### Strings arm (2026-09-14 pm, running): `encoder_ws_w120_s8k_strings_warm`

Data `ws` (`--strings_only --n_strings 6000 --words 120 --held_out_words 8`):
no singles at all — 6 000 random-order 2–4-piece strings of trained rows
(kana, trained words at 25 % per slot; each string checked to tokenize to
exactly its intended pieces) + the 132 fully-covered corpus lines. Eval
adds `flip` (12 clean kana pairs × both orders) and `str3` (8 clean 3-kana
strings) to the Run 3 groups. Recipe = Run 3 with `g` frozen (`--lr_enc 0`),
`f` and `c` warm-started from Run 3, band `--t_min 0.5 --t_max 0.9`, 8 000
steps. Job `20260914-134255-fe1767`.

Read: `line` / `flip` / `str3` > 0 → strings are reachable as a static
table (then: does `flip` distinguish orders, or draw the same string for
both?). All 0 with singles intact → the frozen adapter cannot
contextualise these rows → W3 or slot rows. Singles collapsing → the rows
lost identity under string-only exposure (then mix singles back at a
small fraction before any verdict).

#### Strings arm result (2026-09-14 evening): sequences render, order is read, singles inherit the multi-unit prior

67 min train (2.31 it/s) + 15 min eval; `output/wake_probe/encoder_ws_w120_s8k_strings_warm/`.
Delta stayed on-manifold (rel 1.42, max 2.2, Run 3 band); the warm-started
residual dipped 0.78 → 0.52 in the first 25 steps and re-settled at 0.71–0.78.

| group | n | CER sfx / vl | exact | Run 3 |
|---|---|---|---|---|
| single | 36 | 0.861 / 0.917 | **5/36** | 34/36 |
| word | 32 | 0.820 / 0.831 | 3/32 | 9/32 |
| line (held-out corpus, all rows trained) | 32 | 0.747 / 0.740 | **2/32** (ちゃく, フフフ) | 0/32 |
| flip (12 clean kana pairs × both orders) | 48 | 0.719 / 0.719 | **4/48** (たン, タひ, こユ, ユこ) | — |
| str3 (clean 3-kana, unseen) | 16 | 0.625 / 0.583 | **3/16** (んソヒ, にワル, ひくル) | — |
| combo / corpus | 36 / 20 | 0.667 / 0.667 | 1/36 / 2/20 | 1/36 / 0/20 |
| en | 24 | 0.000 | 24/24 | 24/24 |

- **Sequences no longer collapse to one unit.** Nearly every string prompt
  draws 2–4 glyphs; `flip` has both pieces present in 25/48. Every string
  group moved off zero, including *unseen* strings of trained rows (`str3`
  3/16, `flip` 4/48) — composition generalises across strings, no
  per-string exposure.
- **Order is read from the static table.** In `flip`, the first rendered
  glyph equals the caption's *first* piece 28/48 and its *last* piece 5/48
  (chance would be equal); the same two rows swapped change the picture
  (くル→くくル vs ルく→ルくル, うネ→ううう vs ネう→ネうう, こユ / ユこ both
  exact). The frozen adapter does contextualise these rows once the loss
  asks for it. The "cannot contextualise off-manifold rows" branch is
  dead; W3 is not needed for this.
- **The dominant failure is repetition, not identity**: ううう, ねねね,
  くくル, ををら, はめめ — the count is right, the second slot copies a
  neighbour. This is Run 2's repeat mode.
- **Singles inherited the multi-unit prior** (34 → 5/36): a one-kana
  caption now draws the glyph *plus* extras (ひ→ひひ, む→むむろ, ケ→ケケケ,
  テ→ズテテ, チ→チテチ). The identity is still there; the rows now carry
  "several units" the way they used to carry "one centred unit". Unit
  count is a data-distribution prior baked into the rows, symmetric with
  the singles-era artefact. Held-out words 0/16 as before.

**Reading.** Rows can carry order and count, and the DiT reads them, but
the rows also absorb whatever count the training distribution had. The
fix is not two hard steps but one distribution: singles and 2–4-piece
strings mixed so the number of units is only predictable from the number
of ext tokens in the caption. Next arm: same recipe from Run 3's table,
data = 30 % singles + 70 % strings (`--strings_only` gains a
`--single_frac`), band 0.5–0.9. Gate: singles back ≥ 30/36 with `flip` /
`str3` / `line` at or above this arm. Then the repeat mode is the
remaining lever (harder negatives: strings with a repeated piece, so a
repeat is only right when the caption says so).

### Shelved W2 levers (do not reopen without a new reason)

- Same-noise classifier CE / swap hinge on free rows: valid for singles,
  cannot fix combos (frozen DiT does not bind order); superseded by W2d.
- W2b warm start: folded into the encoder's shared bias.
- 92-kana free-rows band run: superseded — Run 3 trained all 92 kana in the
  hybrid table (34/36).
- Synthesis (Δ_山 + Δ_石 in 岩's row): 5 min, low decision value; run only
  as a data point for the encoder's input design.

### EN safety (carry through every arm)

The rows delta is bit-exact on prompts without an ext id. Any arm that
re-introduces the adapter LoRA or a DiT LoRA carries (1) an ext gate (LoRA
scale 0 for a sequence with no ext id), (2) a position mask, (3) an EN
replay term for the mixed-prompt case. The gate's "EN control unchanged" is
measured on the mixed clause, not EN-only prompts.

## W3 — strings and the product condition

Composition is DiT-side work: an ext-gated cross-attention LoRA trained on
multi-glyph canvases (W2e-shaped data) with the W2d pack loaded frozen,
then `plan_render`'s task — EasyControl bubble fill, cond LoRA as JA-SHIP,
no target-stream LoRA. Product captions keep the trained clause grammar
verbatim (`Japanese text reads as "…"`); a Japanese-language clause is not
an address. Once a clause is an address the DiT can read, OCR-quoted
captions become supervision for every training run. This is the one step
that leaves the native form (the gate is runtime behaviour a stock loader
lacks) — decide then. **Gate = plan_render G2 (R1 ≤ 0.5, R2 ≥ 0.7) on
JA-BODY's judge set.**

## W4 — kanji at scale

If W2d generalises, kanji is its held-out test at scale (IDS-structured
split over jōyō); the pack covers every piece the encoder can render.
Otherwise there is no rows path to 2,136 characters.

## Kill criteria

- Run 1 held-out = 0 with trained singles at the gate → the encoder is a
  lookup; free rows and encoder both need per-character exposure. Fallback
  is glyph-image conditioning (AnyText / GlyphControl shape), gated on the
  presence of a glyph image so ext-free prompts stay untouched.
- Run 1 trained singles < 50 % after one retune → the encoder cannot even
  memorise through the frozen DiT; same fallback.
- W2d passes and W3 sits at plan_render's floor → the bubble-fill task
  itself is the blocker; move the woken text side to T2I manga-panel
  generation and re-judge there before killing.

## Instruments and gotchas

Full list in the report; the ones that bite while running the arms above:

- Every GPU stage via `make daemon-run ARGS="--label … --queue
  project/cjk_renderable_anima/probes/wake_probe.py --stage train|eval|native|classify …"`
  then `make daemon-wait JOB=<id>`; the data stage is CPU-only and safe inline.
- **Never below 512²**; block compile before grad-ckpt; batch 8 OOMs at 512²
  without ckpt even compiled; activation budget stays 0.99.
- Rows lr 1e-3 in row-norm units (3e-3 walks off-manifold); watch the
  encoder's delta norm the same way (`rel` in `train_log.json`).
- Read the **largest detector box**, both readers; `report.md` is the
  min-over-boxes score. `--no_floor` for every eval; floor reference =
  W0/W1 (singles 0/16, JA CER 1.000, EN 24/24).
- Ext rows are Qwen-piece keyed (piece text via `mapping["qwen"]` /
  `mapping["char"]`); clear `conds_cache` on delta-scale switches; the VAE
  takes [-1, 1].
- `probes/wake_geometry.py --arm_dir … --pairs 明=日+月,…` for composition
  structure in any arm's rows.
