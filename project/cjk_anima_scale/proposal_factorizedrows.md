# proposal — factorized rows: identity per row, modes by context (2026-09-26)

Status: **proposed, nothing run.** It follows from `proposal.md` § 3.1
("Two doublings, one direction"; "A is gated away by context") and the
user's question: train identity only, and get the rest (line, vertical,
manga, eventually SFX) from shared **modes** switched on by context.
`proposal.md`'s α sweep (3.3 a) ran the same day: the line mode's
operating dose is α 1, and no dose bounds either doubling
(`reports/stage_b_2026_09_26.md` § 7). An outside
opinion (`opinion_factorizedrows.md`, 2026-09-26) is folded in: its
context-interaction probe became F0 (§ 2), and its caveats are in § 0 and
F1. Its adapter-activation replay / distillation program (its § 2–4) stays
parked behind F0 (§ 6).

## 0. What the reads already say

- **A row is identity plus a shared direction.** The singles donor's Δ is
  24 % `u_S` (split-half cos 0.82) and 76 % per row. The pieces' is `u_P`
  (split-half 0.97). `u_P` and `uB` sit at cos 0.30, so **each training
  context makes its own shared direction** (`reports/stage_b_2026_09_26.md`,
  `reports/transplant_line_2026_09_26.md`).
- **The shared direction transfers** to rows that never trained with it
  (Stage A pieces, Stage B singles: held-out words ≤ 1 edit 11 → 66 / 160).
- **In the renders, a mode baked into a row is not switched off by
  context.** A line-trained row doubles alone (A), and the seed's singles,
  trained alone, render a spelled string as one big first glyph
  (`reports/spell_2026_09_26.md`). **Both are the same failure in
  opposite directions: a layout bound into the row.** *Why* is open. The
  adapter is not structurally context-free (`LLMAdapterTransformerBlock`
  has self-attention, cross-attention to the Qwen states and an MLP). The
  "context-free at the adapter" read (`rows_manifold_2026_09_16.md`, Δ
  output cos 0.91–0.94) was across quote frames, not across neighbouring
  glyphs. F0 (§ 2) reads the neighbour case.
- **Identity is modular only if it was trained in scene composites.**
  Flat-trained residuals on another table's shared direction render 0 / 64,
  and composite-trained ones 23 / 64 (`../cjk_renderable_anima/reports/transplant_2026_09_16.md`).
  The seed rows (`step1_0921`, grid 50 % : scene 50 %, today's `b0709`
  group) are that kind.
- **Closed, and binding here** (wake roll-up):
  - *Transplant + pinned trigger is not a step saver.* `--pin_dir` curves
    overlap training from scratch. This proposal must not claim fewer
    steps per row. Its claims are about **what renders**: no layout bound
    into the row, and composition for a vocab that never saw line data.
  - *No row-space geometry penalties* (decorrelation / orthogonality /
    whitening). A factorization is a parameterization, not a penalty, but
    it sits in the same family. If § 3 turns into a fight over geometry,
    stop.

## 1. The factorization

```
row_eff(i, ctx) = r_i + Σ_m g_m(ctx) · s_m(kind_i) · v_m
```

- `r_i`: the per-row identity. Trained per vocab, as today.
- `v_m`: one shared vector per mode, trained jointly. A handful of
  parameters.
- `g_m(ctx) ∈ {0, 1}`: the mode's gate, read from context at the hook,
  never from the row:
  - **line**: the row sits in a run of ≥ 2 pack rows in the span. That is
    what gates A (`proposal.md` § 3.4 (0)).
  - **horizontal**: the caption's existing marker
    (`horizontal Japanese text reads as` / `, written horizontally.`).
  - Later modes (SFX, non-manga surfaces) need a caption marker of their
    own.
- `s_m(kind)`: a per-kind scale (single / piece), since `u_S` ≠ `u_P`. It
  can start at one `v_m` per (mode, kind) pair.

**Where it goes.** `src/common/hooks.py::ExtDelta` adds `raw` onto the
pack rows in `llm_adapter.embed`'s forward hook; its pre-hook already sees
the raw ids, so the gate is local. At inference the same rule lives in
`library/anima/vocab_pack.py`'s hook, and the ComfyUI node's `_vendor/`
follows through `make vendor-sync`. Saving is `r_i` in `trained.pt` as
today, plus `modes` (vectors + gate rules). A pack without `modes` reads
exactly as now.

**What training has to supply.** The gradient splits between `r_i` and
`v_m` only if the data shows the same row both with the gate on and with
it off. Items with the row alone (gate off) push the line behaviour out of
`r_i`. Line items (gate on) push the shared part into `v_m`, which gets 36
rows' worth of gradient against each `r_i`'s one. So a factorized run is
**identity items + line items for the same vocabs**. The Stage B donor
data already is that, with its count tier.

## 2. Stage F0 — does the adapter already modulate a row by its neighbours?

No DiT and no renders: the text encoder + `llm_adapter` forward only, with
the pack hook and `ExtDelta` as in training. For a glyph `g`, captions
`Japanese text reads as "…"` with `g` **alone** and `g` **in a spelled
string** (positions balanced; en + swap frames), and two row sets on the
same caption: the seed, and the seed + `u_S` (then the Stage B donor rows
on the donor glyphs).

```
D[l, ctx] = H_rows[l, ctx] − H_seed[l, ctx]      (at g's positions, block boundary l = 0…6)
interaction[l] = D[l, line] − D[l, lone]
```

Read ‖interaction‖ / ‖D‖ and cos(D_line, D_lone) per l, at `g`'s
positions and, separately, at the surrounding positions. Also read
whether ‖D_lone‖ shrinks against ‖D_line‖. The held-out 10 and the 36
donors both have their `u_S` rows built, so no training.

Decision (this selects where the gate lives, not whether doubling goes):
- **interaction ≈ 0 at the output** (cos ≥ 0.95): the adapter passes the
  row's change through blind to the neighbours, so a gate has to act
  outside it, which is § 1 as written.
- **Large interaction, D_lone smaller**: the adapter already switches the
  line mode down alone, and the singles still double. So the bound
  layout is DiT-side or a data gap. The gate location is not the lever,
  so F1 drops in priority, and the opinion's replay / distill program
  (localize where the context enters) is the next read.
- Large but not shrinking alone: context changes the mode without
  bounding it. Read which branch (self-attn / cross-attn / MLP) carries
  it before choosing.

Cost: CPU is enough for a few hundred captions. Qwen-side caching means
the captions are encoded once.

## 3. Stage F1 — factorized donor on Stage B's data

Stage B's donor data unchanged (36 donors, 568 words, count tier 0.3),
same seed, same 90 steps / row. Rows `r_i` + one `v_line` (single kind),
gated by run length. It needs a `--modes line` path in `rows.Rows` / the
hook. That is the only code.

Reads (all floors cached):
- donor singles alone, official · repeat / 144: floor 91 · 29, plain donor
  43 · 50.
- こんにちは spelled: ≤ 2 edits (plain donor 13 / 16) and in-word `dup`
  (plain donor 13 / 16).
- `v_line` on the 10 held-out seed rows: the held-out words ≤ 1 edit
  (`u_S` post hoc gave 66 / 160) and `dup`.

Decision:
- Singles alone back near the floor with こんにちは ≤ 2 edits held → the
  split is learnable by gradient. `v_line` replaces post-hoc extraction,
  and § 4 has an identity recipe to measure against.
- Singles still double with the gate off → the line behaviour stays in
  `r_i` even with `v_line` available. Fall back to post-hoc `u_S` + the
  gate (`proposal.md` § 3.4 (0)), and § 4 reads identities under that
  instead. **This does not close the parameterization.** Gate on / off
  exposure constrains the split but does not force it, and the data
  leaves a hole: the count tier covers 24–40 px only, while b0305's lines
  are ≈ 18 px with no alone counterpart, since a single under 24 px has no
  window in the band law. Closing it takes an F1b with alone items at the
  b0305 px, and that needs a band-law row (a read) first.
- In-word `dup` is not expected to move here (it happens with the gate
  on). The α sweep and `proposal.md` § 3.3 b / d own it.

## 4. Stage I — which scene makes a good identity

With modes carried by `v_m`, the question for `r_i` is: **which training
context gives the identity with the least layout baked in?** Grid 50 % :
scene 50 % (the seed's `b0709`) is the incumbent. It buys identity and
native together, but it also binds "one big glyph filling the bubble"
into the row (the seed's spelled string renders as one big first glyph).

A good identity `r_i`:
1. **alone**: renders its glyph once in a native scene (official, repeat
   at floor),
2. **under a mode**: composes once `v_line` (or `u_S`) is added in a run
   (≤ 1 edit on spelled words, in-word `dup` at floor),
3. **modular**: renders on a foreign shared direction (the 09-16 test).

Candidates. Every one is `r_i` data only (gate off, one glyph per item);
recipes and knobs exist unless marked:
- **I0** `b0709` as is: `scene_single` 0.5 (fill 0.7, ≈ 50 px) +
  `grid_single` 0.5 (1×1–3×3). The incumbent.
- **I1** scene only, px spread: `scene_single` with bubble fill 0.2–1.0,
  so one glyph at 20–60 px and never always filling (merges the count
  tier's `scene_single_small` into identity).
- **I2** grid only at small cells (3×3 / 2×3, 25–85 px). Wake roll-up:
  grid alone is never a seed table, so this is the negative control for
  "native needs scene".
- **I3** I0 + I1's small-fill tier at 1 : 1 : 1.
- **I4** pool diversity: I0 on the non-manga / `sl1w` pools as well.
  Needs a pool check, not code.

Glyph set and start: **cold rows on a micro set** (≈ 12 vocabs the seed
lacks: katakana or common kanji, pack raw rows). A warm start measures the
candidate's Δ on top of `b0709`'s identity, which is the confound. Cold
costs more steps, so keep the arms micro (≈ 25 min each,
`feedback_micro_arms_and_per_glyph_reads`), same draws per row, same
bands (singles 0.7–0.9, band law). The read is per glyph with sheets.

Read each arm on 1–3 above: for 2, `v_line` from F1 (or `u_S`) on
spelled strings of the micro set. Those strings need new floor keys, so
this is the one floor render here. Pick the recipe that wins 2 without
losing 1.

What would close Stage I: every candidate ties on 2. Then the identity
source does not matter beyond scene vs flat (09-16), `b0709` stays, and
the gain is all in the modes.

## 5. Modes beyond line (discovery, after F1)

Discovery works like `proposal.md` § 3.3 d: **twin donors** (same vocabs,
same seed, one context axis changed) → the Δ difference, own-row part
removed → split-half stability → transplant onto held-out rows → the
ruler. Under § 1 a mode that passes becomes a gated `v_m`.

| mode | gate | first read | note |
|---|---|---|---|
| count / alone | run length = 1 | `proposal.md` 3.3 d twin | the first instance of the discovery recipe |
| horizontal | the existing caption marker | a horizontal ruler on the seed: is there a failure? | no horizontal read on record; skip the mode if there is no failure |
| manga vs other surfaces | a new caption marker | twin: manga-bubble pools vs sign / cloth / UI pools | current data is all manga bubble, so "manga" is today's baked-in default |
| SFX | a new caption marker | none possible yet | no SFX data recipe (the renderers are font-based); the reader and corpus exist (Manga109-s + COO). Whether clean-font identity survives SFX distortion is itself open |

Order by the rulers' failure list (doubling, sentence assembly, 3+ glyph
pieces), not by the taxonomy. A mode is worth training only if it
explains one of those failures.

## 6. Order and cost

1. ~~`proposal.md` 3.3 a~~ (ran 2026-09-26): the line mode's dose is α 1.
   In-word `dup` does not separate from composition by dose; it sits
   at +18–23 over the floor at α 0.5–1 and climbs at 2.
2. **F0** (§ 2): the adapter-only interaction probe (CPU, a small
   script, no renders). It decides whether F1 is the next GPU spend.
3. **F1** (§ 3): the `--modes line` code in `rows.Rows` + the hook
   (half a day), then one train + read (≈ 25 + 25 min GPU).
4. **Stage I** (§ 4): I0 / I1 / I2 first (3 micro arms + one floor render
   of the micro set's strings); I3 / I4 only if I1 moves 1 or 2.
5. `proposal.md` 3.3 d, the count twin, which is also § 5's first mode
   discovery.
6. The sent test (`300f_sp` rows + the line mode on the sent ruler: does
   sent contained recover 4 → 11?). It tells whether the sentence mix can
   shrink per vocab once modes carry line composition.
7. Parked: the opinion's adapter replay / distillation (its § 2–4). It
   replays `u_S`, and its teacher carries doubling (the opinion says so).
   Adapter distances do not predict renders here (piece ↔ glyph R² ≈ 0.03,
   `spell_2026_09_26.md`), so it opens only if F0 finds a context-dependent
   component worth localizing.

## 7. What closes it

- F1 **and** F1b (alone items at the b0305 px) show the line behaviour
  stays in `r_i` with `v_line` available, and the post-hoc route cannot
  hold singles alone either → the factorization is not learnable by
  gradient in this parameterization. Modes stay
  post-hoc (`u_S` + gate), and Stage I runs against the post-hoc mode.
- Stage I ties → `b0709` stays the identity recipe (§ 4).
- The gate itself costs identity (strip rows alone lose 14 official to the
  floor, `proposal.md` § 3.1) and a factorized `r_i` does no better → the
  seed row stays the "alone" value and trained rows are line-only.
