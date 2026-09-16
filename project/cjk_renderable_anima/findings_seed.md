# findings_seed — what the 53k full-inventory table taught (2026-09-16)

> The first full-scale S-line run, `rows_synth_full_fm10k_full_s53k_qoff`
> (433 named rows: 92 kana + 50 voiced + 200 corpus kanji + 91 words;
> 10 000 items, frame-mix composites 0.9 + flat bubble 0.1, no `c_flat`,
> Q off; 53 000 steps ≈ 490 draws per row, 342 min), failed its flat gates
> and was **not** taken as the seed — but it is the largest composite-
> trained table on record, and one day of reading it settled more than
> the run itself. One screen per topic; the dated record is
> [`reports/rows_manifold_2026_09_16.md`](reports/rows_manifold_2026_09_16.md)
> and [`reports/transplant_2026_09_16.md`](reports/transplant_2026_09_16.md),
> the instruments `src/bench/rows_manifold.py`, `src/probe/transplant_table.py`
> and the train flags `--pin_dir / --pin_coef / --pin_orth`. Verdicts are
> also folded into [`findings.md`](findings.md); this file is the
> one-place summary for the seed question.

## The run itself (`report.md`, `native/report.md`)

| eval | result | read |
|---|---|---|
| singles / ext / kanji (flat template) | 13/36 · 18/36 · 18/36 | hiragana mostly holds, **katakana 1/12**, small kana 0, kanji misses to near-shape neighbours |
| word / word_held / line / phrase_held / combo / corpus | 0/32 · 0/16 · 0/32 · 0/32 · 0/36 · 0/20 | the 91 word rows are **untrained** (row norm 0.12 vs 0.58 mean — the sampler gave them ≈ 8 items each); strings are still one-unit |
| EN control | 24/24 | bit-exact by construction |
| native `en` (あ か す 日, 8 held-out scene prompts × 2 seeds) | 36/64, en cos 0.882, IoU 0.13 | scene kept; か 10, あ 11, す 10, 日 5 |
| native `swap` (EN caption, word swapped) | 18/64, en cos 0.932 | か 0/16 — rows still frame-bound at this exposure |

Exposure curve across the day's arms (composite 0.9, singles rate):
1 330 draws/row → 100 %, 670 → 75 %, **490 → 36 %**. Planning number
≈ 1 000 draws per row; words / phrase pieces need a pinned share.
Instrument limit found the same day: the text cache is ≈ 1.3 MB per
caption in RAM — **≈ 10 000 items per data build** on the 46 GB box.

## Geometry of the table (row space)

- **Not isotropic, but the structure is one direction.** Participation
  ratio 42 vs 304 for a gaussian of the same size; PC 1 carries 13 % of
  the centred energy (gaussian 0.6 %). That PC is the table mean m̂ (cos
  0.955): 18 % of all energy, cos 0.78–0.81 to the S0 / S0b row means and
  only 0.27–0.29 to their explicit `c_flat` — the rows build one shared
  trigger on their own, and it is not the flat-canvas vector.
- **Residuals are near-orthogonal** (pairwise cos 0.03 after projecting
  m̂ out; families separable only by how much m̂ / norm they carry:
  leave-one-out family accuracy 0.71 vs 0.46 chance, within-family cos
  ≤ 0.11).
- **Shape structure is weak and local.** ば↔ぱ sit close (pct 98 vs random
  pairs, partner rank 3/50), か↔が borderline (pct 92, rank 5/50); あ↔ア
  same-sound and kanji↔component pairs are random; glyph pixel-cos vs
  Δ-cos Mantel ρ 0.14–0.18 within a family. No coordinate system to
  interpolate, seed neighbours, or compose in — W2d's "held-out flat"
  holds at 433 rows.
- **What predicts a hit** (50 evaluated single rows, spearman): along-m̂
  +0.32, exposure +0.24, norm +0.21, nearest-neighbour cos −0.09. The row
  that renders is the row that grew, along the shared direction.
- **Katakana is not row-space interference.** hira↔kata pairs are at
  random distance, hit and miss rows have the same crowding; katakana rows
  simply carry less at equal exposure (norm 0.31 vs 0.34, along-m̂ 0.088
  vs 0.111). Whatever it is, it is render / reader side or a DiT-side
  effect — not the table.
- Δ vs the pretrained table: cos to own pack row −0.08, energy in the
  stock T5 table's top-256 PCs 0.30 (gaussian 0.25, real rows 0.53) —
  the rows stay off the pretrained manifold, as every earlier arm did.

## The table at the adapter output (the "shared representation" question)

Under every caption frame the training used — `Japanese text reads as`,
`There is a speech bubble that reads`, `She is saying`, `She is holding
a sign that reads`, bare quotes — the image of the delta `d = out(on) −
out(off)` has cos −0.02 … −0.03 to that frame's pretrained EN quote
direction Q (random |cos| 0.025), 0.05 of its energy in the EN-quoted-code
subspace (the *untrained* pack row has 0.10; gaussian 0.015), and moves
the code **away** from the EN word cluster (cos to the mean EN quoted code
0.42 → 0.31). The frozen adapter applies the frame shift to a trained row
exactly as to an untrained one (shift-vs-Q cos 0.35–0.45 both; EN words
0.73), and `d` is context-free across the quote frames (cos 0.91–0.94).
The shared component of `d` is m̂ carried through, and it is ⟂ Q.

So: **frame-mix training does not pull the rows onto the pretrained
quoted-text representation.** Frame independence in the micro 2×2 came
from exposure under the frames, not from a shared code. `She is saying
"…"` shares nothing with the delta beyond what the adapter gives any row.

## Transplant (no training) — is m̂ + residual modular?

`Δ_r = a_r · m̂_fam(53k) + f_r^{donor}` with the donor's own shared
direction projected out, `native` on あ か す 日:

| donor | training | draws/row | `en` hits /64 | en cos |
|---|---|---|---|---|
| 53k table as is | composite | 529 | 36 | 0.882 |
| m̂ alone (no residual) | — | — | 0 | 0.920 |
| P0b residual (raw / matched norm) | flat-only | 234 | 0 / 0 | 0.935 / 0.923 |
| Run 3 residual (raw / matched) | flat-only, warm | 130 | 0 / 0 | 0.931 / 0.927 |
| micro6 residual (raw / matched) | composite | 1 143 | **23 / 18** | 0.889 / 0.887 |
| P0b as is (own trigger) | flat-only | 234 | 32 | 0.851 |

- **Flat-trained identity is conditional on its own trigger** — 0 on
  another trigger at any scale, indistinguishable from the trigger alone
  (scene kept, floor garble in the bubble); with its own trigger the same
  table renders 32/64 (wiping scenes).
- **Composite-trained identity is modular** — micro6's residual renders on
  a trigger it never saw (its own m̂ and the 53k's have cos 0.65). Per
  glyph it is uneven (す 12/16, か 3, 日 2) at equal draws.
- Data type and draws per row are **confounded** in this set (the one
  composite donor is also the highest-exposure one); two 9-minute conds
  would separate them (micro12 residual → 53k m̂; 53k residual → micro6
  m̂). Not run.
- By-product: m̂ alone has the best scene / placement numbers of any cond
  (en cos 0.920, IoU 0.31 vs the table's 0.882 / 0.13). **The residuals
  cost scene, not the trigger.**

## Pinned-trigger training — does inheriting m̂ buy steps?

`--pin_dir` (53k m̂ frozen per family, kana 0.132 / other 0.333; residual
only trained, re-projected ⟂ m̂ every step) on the micro6 frame-mix data:

| arm | steps | `en` hits / en cos | `swap` hits | flat singles |
|---|---|---|---|---|
| scratch 2 k | 2 000 | 54 / 0.868 | 46 | 12/12 |
| pin 2 k | 2 000 | 54 / 0.868 | 32 | 12/12 |
| scratch 500 | 500 | 39 / 0.901 | 9 | 8/12 |
| pin 500 | 500 | 41 / 0.896 | 20 | 10/12 |

Loss and residual-norm curves overlap at every logged step. **No step
saving**: the budget is the per-row identity, and a free row grows its
share of the shared direction while learning it. The pin helps `swap`
early and hurts it at convergence (the frozen direction + projection deny
the row the component the EN frame wants). `--pin_dir` stays as an
instrument (reading a residual against a fixed trigger), not a recipe.

## Why image → row encoders stay closed (the categorical-address reading)

Asked again after the geometry read (user, 09-16 afternoon): could a
glyph-render embedding — flat, in-scene, their average or concat — be
mapped to the ext row? No, and the reason is the same fact seen from
four sides. "Smooth in shape" would mean: interpolating あ → お morphs
the glyph, halving the row thins or fades it, nearby rows draw similar
shapes. What is measured is the opposite on every axis — IDS sums render
an unrelated kana, not a blend (W2d Run 2, 0/16); 0.5× the row makes the
glyph vanish and the scene return (a threshold, not a fade — `native_x0.5`
2/64); held-out kana land on the nearest trained glyph or garble, never
an intermediate shape (W2d Runs 1–2, 2–5/64 under every lever);
row-similarity vs shape-similarity ρ ≈ 0.15 (this table). The DiT reads
an ext row as a **categorical address** — a lookup into units it already
knows, with a pull toward the nearest known one — the way `cat` and
`dog` embeddings have no cat-dog between them. It has no reason to be
otherwise: the text conditioning it was pretrained on (T5 pieces) is
categorical too, and nothing ever put JA glyph shape on an axis.
Contextualisation does not add such an axis: the adapter is ≈ linear on
these rows (corr 0.73) and the frame shift is the same additive vector
for trained and untrained rows (vs Q 0.35–0.45 both), so context moves
the code along the shared direction, not along shape.

Consequences: any encoder — a fresh CNN (W2d), a pretrained vision
feature, flat + scene inputs averaged or concatenated — must output
addresses that are random directions per glyph, which no input
representation makes learnable from ~400 examples; W2d already trained
the map end-to-end through the DiT loss (not as regression) and g
collapsed to rank 1. Averaging *learned* rows across tables is no better
(same-glyph residuals cos ≈ 0.1; the flat one is trigger-conditional, so
the average is the composite row at half amplitude plus noise). The one
observation that would reopen this is a direction in row space along
which the rendered shape changes continuously; none has been seen. The
same fact is why a multi-glyph word is one row (します): the address
names a unit the DiT knows, whatever its length.

## What follows for the seed

- **Composite exposure per row is the cost, and it is per row.** No
  manifold to amortise, no trigger to inherit, no flat shortcut. ≈ 1 000
  draws per row on composites, words / phrase pieces at a pinned share.
- The remaining efficiency levers are **per item**: more trained rows per
  composite (sentences / multi-token strings once the wrapping fix lands —
  every piece in a phrase gets gradient from the same draw), a larger
  glyph in the box (`--scene_fill`, sl1's wider bubbles), and not spending
  draws on rows the readers cannot score.
- The 53k table is a usable **warm start** (`--init_rows`): identity for
  hiragana / kanji is in it, the shared direction is in it, and the
  transplant shows composite-trained residuals survive a change of
  context. What it lacks — words, katakana, small kana, sentences — is
  exposure, which the sentence composites supply.
- Do not re-propose: flat-heavy mixes for new glyphs; seeding rows or
  `c` from Q / EN codes / m̂; residual transplants; row-space regularisers
  for the katakana miss; the 92-kana interference arm as a *blocker* (row
  space already rules the table out — run it only if katakana still fails
  after sentence exposure).
