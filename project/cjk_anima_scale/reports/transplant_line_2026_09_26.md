# transplant_line — the shared "line" Δ: transplant, strip, shared-only (2026-09-26)

The question (user): the rows' Δ share a component when trained in lines
(`spell_2026_09_26.md` § 5). Could identity be learned on the single canvas
(the seed), the shared line component transplanted onto every row, and only
hard sentences post-trained? Three training-free reads on the seed rows and
spell_b's rows, all on keys the floor cache already holds (no floor
renders). **Verdict: spell_b's gain splits across its Δ's shared
component and the per-row rest — held-out composition roughly half and half
(≤ 2 edits: strip 5, shared-only 7, both 13, floor 1 / 32), doubling
mostly in the shared part and super-additive (repeats: strip 22 = floor 19,
shared-only 37, both 53 / 160). The shared part alone costs singles
official 90 → 60 for 7 / 32 held-out — no line mode without count loss.
A direction estimated elsewhere (300f_sp's pieces) is not it: added to the
seed singles at 1× and 3× it renders the seed.** One seed
per arm; 16 renders per cell; hit-level render noise 0–1 per cell.

Script: `experiments/transplant_line/run_exp.py` (`--mode transplant |
strip | shared`). Envelopes under `experiments/transplant_line/results/`.

## 1. The direction, in row space (CPU)

Δ = run row − seed row (effective, `raw × row_scale`), each Δ with its
component along its own seed row removed (the radial part is 2–11 % of
spell_b's Δ energy — the shared part is not norm shrink). Per run: the
mean pairwise cos of its Δ, and the cos of its mean Δ direction to
spell_b's (`uB`):

| run (data) | rows | pairwise | cos to `uB` |
|---|---|---|---|
| run0926_spell_b (5 singles, in lines) | 5 | +0.177 | — |
| stage0709 micro b30 (single canvas, 0.7–0.9) | 17 | +0.026 | +0.02 |
| stage0507 micro b30 | 25 | +0.034 | +0.22 |
| stage0305 micro b30 (small text, sentence scenes) | 25 | +0.042 | +0.34 |
| run0926_300f_sp (300 pieces, scene_piece) | 300 | +0.176 | +0.30 |

Random unit direction: |cos| to `uB` 95 % ≈ 0.06. The shared direction
grows with the low bands / line data and is absent from single-canvas
training.

## 2. Transplant — 300f_sp's direction onto the seed singles

`u` = the normalized mean of 300f_sp's 300 Δ (own-row part removed;
cos to `uB` +0.30); step = the mean projection of spell_b's five Δ on
`u` = **14.4** (their Δ norms are 62–94: `u` holds ≈ 3 % of spell_b's Δ
energy; the pieces moved 94.7 along it). Arms add α · 14.4 · `u` to the
seed rows of あ り が と う (every other row the seed's; `u` never saw
these rows). Read: spelled `あ り が と う` + the five alone, en + swap.

| arm | row moved | held-out ≤ 2 edits / 32 | singles official / 160 | singles repeat / 160 |
|---|---|---|---|---|
| floor (seed) | — | 1 | 90 | 19 |
| `tl_a2_u1` | 4–6 % | 1 | 93 | 19 |
| `tl_a2_u3` | 12–19 % | killed after 4 of 8 prompts | — | — |

u1 is the seed on every cell (±2). u3's 16 spelled renders read (prompts
0–3) were the seed's images nearly one-for-one — one big あ on a flat canvas
or circle bubble, a few of the seed's small tails changed — so it was
stopped and α 6.6 dropped. A tangential step barely moves the row norm
(√(1 + 0.15²) ≈ +1 %), the one lever known to move hits and scene
(`../../cjk_renderable_anima/reports/row_blocks_alpha_2026_09_18.md`).

A first attempt read こんにちは instead; it was stopped because it rendered a
new floor, and the floor keys it had already folded in (en only) stay in
the cache.

## 3. Strip — spell_b minus its shared component (`tl_s1_bstrip`)

spell_b's five rows, each minus its Δ's projection on `uB` (the rest of
its Δ kept). Removed: あ 33 %, り 32 %, が 20 %, と 36 %, う 41 % of the Δ
energy. Read on spell_b's keys (en + swap, of 32; singles summed over the
five, of 160):

| read | floor | spell_b | **strip** |
|---|---|---|---|
| held-out `あ り が と う` ≤ 2 edits | 1 | 13 | **5** |
| held-out ≤ 1 edit | 0 | 7 | **0** |
| `あ り` official · contained | 0 · 2 | 14 · 29 | 13 · 18 |
| `と う` official · contained | 0 · 2 | 5 · 18 | 11 · 15 |
| `あ が り` official · contained | 2 · 2 | 16 · 19 | 9 · 12 |
| singles official | 90 | 43 | **76** |
| singles repeat | 19 | 53 | **22** |

Paired (same prompt × seed × clause, McNemar): repeats spell_b vs strip
36 / 5, **p 8e-7**; strip vs floor 16 / 13, p 0.71. Held-out ≤ 2 spell_b vs
strip 11 / 3, **p 0.057**; strip vs floor 4 / 0, p 0.125.

Sheets (`tl_s1_bstrip/native_spell/`): the held-out spelled string renders
the training word あり / おり (two glyphs, often on a single-glyph canvas)
instead of spell_b's kana line; あ alone is mostly one あ, a few ああああ
left.

## 4. Shared only — the seed plus spell_b's shared component (`tl_s1_bshared`)

The complement: each seed row + its spell_b Δ's projection on `uB` (so
strip + shared − seed = spell_b exactly, checked). Job
`20260926-121048-b72116`. Each row keeps its **own** coefficient on
`uB` (fit on these five rows), so this is an in-sample read, not a
transplant to an untrained row.

| read | floor | spell_b | strip | **shared** |
|---|---|---|---|---|
| held-out `あ り が と う` ≤ 2 edits | 1 | 13 | 5 | **7** |
| held-out ≤ 1 edit | 0 | 7 | 0 | **2** |
| `あ り` official · contained | 0 · 2 | 14 · 29 | 13 · 18 | **15 · 29** |
| `と う` official · contained | 0 · 2 | 5 · 18 | 11 · 15 | **6 · 13** |
| `あ が り` official · contained | 2 · 2 | 16 · 19 | 9 · 12 | **3 · 4** |
| singles official | 90 | 43 | 76 | **60** |
| singles repeat | 19 | 53 | 22 | **37** |

Paired (McNemar), shared vs floor / vs spell_b / vs strip: held-out ≤ 2
7/1 p 0.07 · 3/9 p 0.15 · 7/5 p 0.77; singles repeat 25/7 **p 0.002** ·
14/30 p 0.023 · 25/10 p 0.017; singles official 7/37 **p 5e-6** · 32/15
p 0.019 · 14/30 p 0.023; `あ り` official 15/0 **p 6e-5** · 7/6 p 1;
`あ が り` official 2/1 p 1 · 1/14 **p 0.001**.

## 5. Read

- **Composition is split, doubling is shared.** Held-out ≤ 2 edits is
  5 (per-row part) + 7 (shared part) ≈ 13 (both) — neither half alone
  clears the floor at one seed (p 0.125 / 0.07). Doubling is absent from
  the per-row part (22 = floor) and present in the shared part alone (37,
  p 0.002 vs floor), and the two together double more (53) than either:
  the shared part is the line mode, and the per-row part amplifies it.
- **The per-row part is the trained strings, mostly.** Strip keeps
  `あ り` / `と う` / `あ が り` and renders the held-out string as one of
  them (あり / おり). Shared-only buys `あ り` at spell_b's level (15 vs 14,
  contained 29 = 29) but not `あ が り` (3 vs 16, p 0.001): the shared part
  alone carries the most frequent 2-glyph transition, not the words.
- **A transplant of the line mode buys half the composition and carries
  the doubling with it**, at −30 singles official. Nothing here buys a
  spelled string without count loss; the count signal has to come from
  data (B2), not from row arithmetic.
- **The direction has to be the rows' own kind.** 300f_sp's piece
  direction (cos +0.30 to `uB`, ≈ 3 % of spell_b's Δ energy) moves nothing
  at 1× or 3× — the § 1 agreement across runs is real but too weak to
  transplant.

## 6. Open

- **Out of sample**: § 4 is in-sample (`uB` and each coefficient fit on
  the same five rows). The real transplant — `uB` at a fixed coefficient
  onto singles no run trained — needs a spelled string whose floor the
  cache lacks; not run (no new floor renders without asking).
- A count signal: B2 (`spell_2026_09_26.md` § 7) — the 0.7–0.9
  single-canvas tiers mixed back in — is the data-side test of whether
  count can be held while the line mode is trained.

Repro:
```
export ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack
X=project/cjk_anima_scale/experiments/transplant_line/run_exp.py
.venv/bin/python $X --label a2 --alphas 1 3 --no_random --legs build
make daemon-run ARGS="--queue --stall-timeout 0 $X --label a2 --alphas 1 3 --no_random --legs eval"
.venv/bin/python $X --mode strip --label s1 --legs build
make daemon-run ARGS="--queue --stall-timeout 0 $X --mode strip --label s1 --legs eval"
.venv/bin/python $X --mode shared --label s1 --legs build
make daemon-run ARGS="--queue --stall-timeout 0 $X --mode shared --label s1 --legs eval"
```
