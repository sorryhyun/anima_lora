# proposal — the line mode as a transferable direction, and what doubling is (2026-09-26)

Status: **Stage A ran 2026-09-26 — the piece direction transfers out of
sample** (`reports/transplant_piece_2026_09_26.md`, § 1 below); Stage B and
the doubling track are not scheduled. Origin: `reports/transplant_line_2026_09_26.md`
and the user's question — train many singles (なにしてる-like, not just
ありがとう) in diverse lines with more budget, extract their shared Δ
direction, and use it to adapt single / piece rows that were trained for
identity only. Doubling rides that direction, so this proposal carries a
second track: what the doubling factor is and how to bound it.

## 0. What the reads already say

- Singles trained in lines (spell_b) compose a held-out spelled string
  (≤ 2 edits 13 / 32, floor 1) and double when rendered alone (repeats
  53 / 160, floor 19) — `reports/spell_2026_09_26.md` § 4.
- Their Δ splits into a shared component `uB` (20–41 % of each row's Δ
  energy) and a per-row rest. Composition splits between them (per-row
  only 5, shared only 7, both 13); doubling is the shared part's (per-row
  only 22 = floor, shared only 37, both 53) — `transplant_line` § 3–4.
- A direction from another kind does not transfer: 300f_sp's piece
  direction (cos 0.30 to `uB`, ≈ 3 % of the singles' Δ energy) on the seed
  singles renders the seed at 1× and 3× — `transplant_line` § 2.
- 09-16 (`../cjk_renderable_anima/reports/transplant_2026_09_16.md`):
  scene-trained residuals are modular across shared directions (23 / 64 on
  a foreign one), flat-trained ones are not (0 / 64); pinning the shared
  direction bought no training steps. The seed rows are scene-trained, so
  they are the modular kind.

What no read had tested before Stage A: **a shared direction estimated on
one set of rows, applied at a fixed coefficient to rows that never trained
with it.** Everything in `transplant_line` § 4 is in-sample. Stage A (§ 1)
is the first such read, on pieces.

## 1. Stage A — piece leave-out transplant (no training, no floor renders)

All 12 pieces the `native_piece` floor cache holds are among
run0926_300f_sp's 300 trained pieces, and 8 of them are what the piece
ruler reads (`あと きて こう こと こんにちは しい ちょっと った`).

- `u_P` = the normalized mean of the **other 292** pieces' Δ (own-row part
  removed, as in `transplant_line` § 1); step = their mean projection on
  `u_P` (one coefficient for every row, no per-row fit).
- Arms: the seed rows of the 8 held-out pieces + α · step · `u_P`,
  α ∈ {0.5, 1}; a random unit direction ⟂ `u_P` at α 1 (the norm control).
- Read: the piece ruler (native, en + swap, the 8 alone) — floor 3 / 256
  official from the cache, 300f_sp's own rows 16 / 256 official, 76
  contained (`spell_2026_09_26.md` § 8), on disk. ≈ 12 min / arm.
- Leak: expected from grid strings, but 300f_sp is `scene_piece`-only and
  every one of its 20 000 items carries one vocab, so no donor item shows a
  held-out piece (leak 0).

Decision: ≥ half of 300f_sp's gain over the floor, above the random arm →
a generic direction exists for pieces and Stage B is worth its budget. At
the floor → the line mode is per row (or per training set) and the
transplant route closes for both kinds.

**Result (2026-09-26, `experiments/transplant_piece/`, job
`20260926-144514-d7e712`) — passes on contained, just under on official.**
`u_P` is stable (split-half cos 0.97) and the 8's own trained Δ sits at
cos 0.83 to it. Step = 94.1, 65–85 % of a held-out seed row's norm.

| arm | official | contained | repeat |
|---|---|---|---|
| floor | 3 | 27 | 2 |
| u0.5 | 7 | 43 | 4 |
| u1 | 9 | 52 | 3 |
| random ⟂, α 1 | 5 | 11 | 0 |
| 300f_sp (trained) | 16 | 76 | 5 |

(/ 256.) Paired: u1 vs random contained 45 / 4 (p 8e-10), u1 vs floor
30 / 5 (p 2e-5), official 8 / 2 (p 0.11). u0.5 → u1 is flat. The gain is
concentrated in こう / こと / きて. あと / った keep their training gain in
the per-row rest, and the 3+ glyph pieces are 0 in every arm, 300f_sp
included. The piece direction carries no doubling. Caveat: u1 renders some
kana in 223 / 256 vs the floor's 179, so part of the contained gain may be
"more Japanese text" rather than the piece. Official is the reading that
does not depend on it, and it is +6 (n.s.).

## 2. Stage B — a diverse singles donor, held-out kana

- **Donor**: ≈ 30–40 kana singles, trained on real words spelled from them
  (the `scene_spelled` recipe of `experiments/spell_b/`, word list from
  the manga109s phrase pool filtered to the donor glyphs, both
  orientations, all four scene pools), at the piece bands **plus a count
  tier** (§ 3.4 — without one the donor learns doubling as part of the
  line mode). Budget at the trainer's 90 steps / row first, then 2×.
- **Held out**: ≈ 10 kana no donor word contains, and ≈ 5 spelled words
  made only of them.
- **Extract** `u_S` from the donor rows (split-half cos of the two halves'
  means as the stability check); **transplant** onto the held-out rows'
  seed values at one coefficient, α sweep {0.5, 1, 2}; random control.
- **Read**: the held-out spelled words (≤ 1 / ≤ 2 edits) and the held-out
  singles alone (official, repeats). **Needs new floor keys** — the ≈ 5
  spelled words + 10 singles, rendered once into `native_spell/`; that is
  the one place this proposal asks for a floor render
  (`feedback_experiments_use_existing_floor`).
- The donor run doubles as a B2-style read on its own rows (does count hold
  with the count tier in).

Decision: transplant reaches the shared-only in-sample level (≈ 7 / 32
≤ 2 edits) with repeats at floor → the adaptation route is real; then test
transplant + a short post-train against the post-train alone (the 09-16
pinning read says it may buy no steps; this is the one comparison that
decides whether the transplant is worth shipping).

## 3. Doubling — what it is, and how to bound it

### 3.1 Hints already on disk

- **Per-row dose.** Per glyph (repeats of 32, en + swap), against each
  row's projection on `uB`:

  | glyph | `uB` coef | floor | spell_b | shared only | strip |
  |---|---|---|---|---|---|
  | あ | 55.5 | 3 | 17 | 8 | 9 |
  | り | 48.3 | 4 | 9 | 11 | 2 |
  | が | **29.6** | 2 | **5** | **1** | 2 |
  | と | 53.9 | 4 | 10 | 9 | 4 |
  | う | 54.4 | 6 | 12 | 8 | 5 |

  が — the smallest coefficient (and the fewest training items, 75) — is
  the one glyph that barely doubles in any arm. n = 5; a hint, not a fit.
- **Doubling predates training**: the seed's own misses double
  (`floor_score.md`), and scene_piece-only did not remove it
  (`reports/piece_only_2026_09_26.md`) — the base model has a
  "fill the text region" prior that a row can switch on.
- **The recipes fill the box**: `scene_spelled` / `scene_piece` draw
  `fill` 0.7–1.0 of the bubble; the single canvas draws one glyph at
  ≈ 50 px. A row trained only in lines has only ever seen its glyph as
  part of a box-filling line at 19–38 px.
- **The reads carry boxes** (`reads[].box`, per reader region), so a
  rendered glyph's px and the box it fills are measurable from the
  renders already on disk.

### 3.2 Hypotheses

- **H1 — fill prior.** A line-trained row learns "text at line px that
  fills the bubble"; alone, one glyph at that px cannot fill it, so it
  repeats. Predicts: repeats rise with bubble width / fall with glyph px,
  and the doubled render's glyphs sit at the trained line px.
- **H2 — a context-free line mode.** The row carries "I am part of a
  line" regardless of its neighbours in the caption; the adapter cannot
  tell alone from spelled. Predicts: repeats scale with the row's `uB`
  coefficient (the § 3.1 table), independent of bubble size.
- **H3 — a missing count signal.** No training item ever shows a line-mode
  row alone at line px, so nothing penalises the repeat. Predicts: a count
  tier at line px (§ 3.4 a) removes doubling without moving the shared
  direction much.

H1 and H2 are read training-free; H3 is the data fix.

### 3.3 Analyses (training-free unless marked)

- **(a) Dose-response along `uB`** — seed singles + α · c̄ · `uB` (c̄ = the
  five rows' mean coefficient), α ∈ {0.5, 1, 1.5}, on spell_b's keys
  (floor cached). Two curves on one axis: held-out ≤ 2 edits and repeats.
  If composition saturates before repeats rise, there is an operating α;
  if they rise together, the direction is one thing (H2).
- **(b) Size and fill from existing renders** — for every render of a
  single alone (floor, spell_b, strip, shared, u1): the text box (reader
  region), the glyph count read, px = √(box area / glyphs), and the scene's
  bubble size where the sheet shows one. Repeats vs px vs box width across
  arms separates H1 (box-driven) from H2 (row-driven). CPU only.
- **(c) Where the row lands in the image** — cross-attention from image
  tokens to the row's token position, per block, averaged over σ steps:
  one attention blob (one glyph) vs several (a line of it). Compare seed
  vs spell_b vs strip for `"あ"` alone and for the spelled string. Needs a
  cross-attn capture hook on the DiT (not in the line's `src/` today);
  the only analysis here that costs code.
- **(d) A count direction** *(needs the Stage B donor or B2)* — the Δ
  difference between rows trained in lines with the count tier and
  without it, own-row part removed. If it is roughly ⟂ `uB` it is a
  separate, transplantable "one glyph" component; if it is −`uB` the
  count tier just shrinks the line mode.

### 3.4 Levers to bound it (after 3.3 says which)

- **(a) Data: a count tier at line px** — a `scene_single_small` recipe:
  one glyph at the line bands' px (19–38) in a bubble with low fill
  (0.2–0.4), captioned alone. It breaks the "line px ⇒ fill the box" link
  that the current tiers never break (the single canvas is ≈ 50 px, one
  glyph, full frame). Cheaper than B2's 0.7–0.9 single canvas and aimed
  at H1 / H3. First candidate.
- **(b) Row arithmetic** — if (a)/(d) find a count direction ⟂ `uB`,
  transplant line mode + count together; if the dose-response (3.3 a)
  shows an operating α, cap each row's `uB` coefficient at it (a per-row
  clip, not a penalty).
- **(c) A training-time cap** — clip each trained row's projection on a
  frozen `uB` to c_max after every step. Caution: row-space geometry
  penalties (decorrelation / orthogonality / whitening) closed on the W2d
  line (wake roll-up); a hard cap on one measured direction is narrower,
  but it is the same family — only after (a) fails.
- **(d) A caption marker** — the line mode bound to a caption token
  (spelled items captioned `, spelled.` or similar) instead of the rows,
  so a row alone never carries it. Touches the caption convention and the
  TE path; last resort.

## 4. Order and cost

1. ~~Stage A~~ (done: transfers, § 1) and 3.3 b (CPU, open).
   A cheap bridge before Stage B, no training and no floor render: the
   `u_P` step (94) onto the single rows of `こ ん に ち は`, read en-only
   (the floor cache holds those keys). It tests a piece direction on
   singles at piece scale (`transplant_line` § 2 only went to 43).
2. 3.3 a (≈ 35 min GPU, cached floor).
3. If Stage A transfers: Stage B donor with the § 3.4 a count tier (data
   CPU + ≈ 1 h train + eval; the one new floor render).
4. 3.3 c only if 3.3 a/b leave H1 vs H2 open.

## 5. What closes it

- Stage B's transplant at the floor (Stage A was not) → the line
  mode is not a transferable direction; composition is bought per row
  (exposure), and the scale line's budget question stays per-row draws.
- Doubling rising with composition at every α (3.3 a) and surviving the
  count tier (3.4 a) → count is not separable from the line mode in row
  space; the fix moves to the caption / TE side (3.4 d) or is accepted
  as a render-time filter.
