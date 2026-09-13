# plan_wake — wake the DiT's own JA glyph units (forward plan, 2026-09-13)

> **Status (2026-09-13 night).** W0–W2 are done and lifted into
> [`reports/wake_w0_w2_2026_09_13.md`](reports/wake_w0_w2_2026_09_13.md):
> the hypothesis, Probe 0/1, the address geometry, the 256² / 24-kana /
> balanced / σ-band arms, the native-rendering probe, the kanji probe and
> the case for W2d. This file is only what comes next. Nothing is shipped.

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

**Next action — mean-centre the identity, own the layout mode separately.**

- `Δ_r = (d_r − mean_rows d) + c`. Centring across the full row table each
  step projects the common-mode gradient out of the shared weights, so the
  per-glyph part behaves like free rows (their own, inconsistent gradients,
  which saturated at ~1.0× on every rows arm). `c` is one free vector at the
  rows lr carrying the "big glyph on a blank canvas" mode every rows arm
  converged to (cos-to-mean 0.25–0.53, common part ≈ 0.5–0.75× row norm).
- Bound `c` on the **parameter**, not the output: after each optimizer step
  `c ← c · min(1, cap/‖c‖)` with cap 0.75. Projected descent has no creep;
  output normalisation does.
- Keep `head × 1/64` and the LayerNorm on the pooled features; no cap on the
  centred part, but log `rel_spread` (identity), `rel_common` (‖c‖) and the
  max row norm, and kill if spread is still < 0.05 by step 600 or the max row
  passes 2×.
- Relaunch with the same data / steps / σ band (`--arm encoder --data_tag w2
  --arm_tag held32 --held_out 32`, 6000 steps); the 512² latent cache is
  already built. Gates unchanged.
- If spread grows but trained singles miss the gate: the pooled feature is
  ~90 % background (ink covers ~7 % of the 96² render) — next lever is the
  input, not the optimizer: tight-crop the glyph to the render, or mean-pool
  ink-weighted. If spread stays flat with centring in place, the head's
  gradient through the frozen DiT is too weak at 1/64 — raise `out_scale`
  one notch (1/16) before anything else.

### Run 2 — kanji + scene composites (only if run 1's held-out is nonzero)

Inventory kana + the 24 structured kanji + the top-200 corpus kanji with
an IDS-structured held-out split (composites whose atoms are trained).
Data changes from blank canvases to glyphs pasted into real dataset images,
caption = the image's caption + the clause, so the row can only explain
the glyph — the layout-prior fix both product goals need. One lever per
run: kanji first on blank canvases if run 1 is marginal, composites first
if it is clean.

Gate: held-out kanji singles > 0 and native-rendering scene survival
(scene kept on ≥ 75 % of the trained-clause renders, glyph present ≥ 50 %).

### Shelved W2 levers (do not reopen without a new reason)

- Same-noise classifier CE / swap hinge on free rows: valid for singles,
  cannot fix combos (frozen DiT does not bind order); superseded by W2d.
- W2b warm start: folded into the encoder's shared bias.
- 92-kana free-rows band run: the question it answers no longer matters.
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
  project/cjk_aware_anima_dit/probes/wake_probe.py --stage train|eval|native|classify …"`
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
