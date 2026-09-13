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
