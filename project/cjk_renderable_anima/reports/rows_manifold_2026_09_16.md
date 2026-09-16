# rows manifold probe (2026-09-16): the 53k table has one shared direction and near-orthogonal residuals; nothing in it is the pretrained quote representation

> `probes/rows_manifold_probe.py` read the full-inventory table
> (`rows_synth_full_fm10k_full_s53k_qoff`: 433 named rows, 53 k steps,
> frame-mix composites 0.9, no `c_flat`, Q off) in row space and at the
> adapter output, CPU + one 2-minute GPU pass, no retrain. **Row space:**
> the table is far from isotropic (participation ratio 42 vs 304 for a
> gaussian of the same size) but the structure is one shared direction m̂
> (18 % of the energy, 0.955-aligned with PC 1, the same direction the S0 /
> S0b rows shared: cos 0.78–0.81 to their row means, only 0.27–0.29 to
> their explicit `c_flat`) plus residuals that are near-orthogonal (pairwise
> cos 0.03). Shape structure in the residual is real but weak: ば↔ぱ sit
> close (pct 98, partner rank 3/50), か↔が borderline (pct 92, rank 5/50),
> hira↔kata same-sound and kanji↔component no closer than random, glyph
> pixel-cos explains ρ ≈ 0.14–0.18 of Δ-cos within a family. **Adapter
> output:** the image of the delta under every frame — `reads as`,
> `speech bubble that reads`, `She is saying`, `holding a sign`, bare
> quotes — has cos −0.02 … −0.03 to that frame's EN quote direction Q
> (random |cos| 0.025), carries 0.05 of its energy in the EN-quoted-code
> subspace (the untrained pack row carries 0.10, gaussian 0.015), and moves
> the code *away* from the EN word cluster (cos to the mean EN quoted code
> 0.42 → 0.31). The frozen adapter applies the frame shift to a trained row
> exactly as to the untrained row (shift-vs-Q cos 0.35–0.45 on and off; EN
> words 0.73), and the image is context-free across the quote frames
> (cos 0.91–0.94). So: **no manifold to move along, and no shared
> representation with `She is saying "…"` — the frame-mix arm's frame
> independence did not come from converging on Q; the rows built their own
> shared trigger, and it is ⟂ Q.** Katakana's miss is not row-space
> crowding: hira↔kata pairs are random-distance and nearest-neighbour cos is
> equal for hit and miss rows; katakana rows simply carry less (norm 0.31 vs
> 0.34, along-m̂ 0.088 vs 0.111) at the same exposure.

Instrument: `probes/rows_manifold_probe.py --stage rows` (CPU) and
`--stage adapter --device cuda` (Qwen TE + llm_adapter, 433 rows × 6 frames,
delta on / off from one TE pass). Outputs
`output/wake_probe/rows_synth_full_fm10k_full_s53k_qoff/manifold/{rows,adapter}.json`.
Ruler notes: row space is in row-norm units; "residual" = Δ with the shared
direction m̂ projected out (plain mean subtraction turns every weak row into
−m and makes the untrained word rows look like one tight cluster — the first
pass read that as within-family cos 0.78 for words, an artefact); the EN
quote direction Q per frame = mean over 18 EN words of (framed code − plain
code) at the word's T5 positions, as in `quote_probe.py`.

## Row space (433 rows: kanji 200, word 91, hira 46, kata 46, hira_v 25, kata_v 25)

| statistic | table | gaussian control |
|---|---|---|
| participation ratio (centred) | **42** | 304 |
| energy in PC 1 / top 4 / top 16 / top 64 | 0.133 / 0.213 / 0.342 / 0.609 | 0.006 / 0.024 / 0.092 / 0.319 |
| shared mean m: norm 0.28 of row-norm mean 0.58, energy share | **0.178** | — |
| cos(m̂, PC 1) | 0.955 | — |
| cos(m̂, S0 / S0b / cap075 row mean) | 0.81 / 0.78 / 0.53 | — |
| cos(m̂, S0 / S0b / cap075 `c_flat`) | 0.29 / 0.27 / 0.25 | — |
| pairwise cos raw / residual | 0.155 / **0.031** | — |
| cos(Δ, own pack row) | −0.08 | — |
| energy in the stock T5 table's top-256 PCs (Δ / pack ext rows / stock rows) | 0.30 / 0.33 / 0.53 | 0.25 |

Per family (residual = m̂ projected out; hit = singles `exact`, both seeds):

| family | n | norm | along m̂ | resid within cos | vs others | PC 1 score | hit rate (n) |
|---|---|---|---|---|---|---|---|
| kanji | 200 | 0.839 | +0.448 | +0.005 | −0.016 | +0.17 | 0.50 (36) |
| word | 91 | **0.118** | +0.021 | +0.051 | +0.005 | −0.26 | — (word 0/32) |
| hira | 46 | 0.336 | +0.111 | +0.072 | +0.011 | −0.19 | 0.50 (24) |
| kata | 46 | 0.309 | +0.088 | +0.106 | +0.010 | −0.21 | 0.08 (12) |
| hira_v | 25 | 0.852 | +0.459 | +0.062 | −0.008 | +0.17 | 0.59 (22) |
| kata_v | 25 | 0.780 | +0.398 | +0.071 | −0.004 | +0.10 | 0.83 (6) |

Leave-one-out nearest-family-centroid accuracy on the residual 0.71
(majority chance 0.46): families are separable, mostly by how much of m̂
and how much norm they carry (PC 1 = kanji + voiced vs kana + words), not by
a tight within-family cluster (within cos ≤ 0.11). The word rows are
untrained (norm 0.12 — the 0/32 word eval is the same fact).

Pair tests on the residual (pct = percentile of the mean pair cos among
random pairs of the same pools; partner rank = where the true partner sits
among the pool by cos):

| pair | n | cos | random | pct | partner rank (median / pool) | top-1 |
|---|---|---|---|---|---|---|
| ば↔ぱ (dakuten ↔ handakuten, same glyph, one mark) | 10 | +0.203 | +0.059 | **97.8** | 3 / 50 | 0 |
| か↔が (voiced hira) | 25 | +0.082 | +0.002 | 91.9 | 5 / 50 | 1 |
| カ↔ガ (voiced kata) | 25 | +0.073 | +0.010 | 87.3 | 11 / 50 | 2 |
| あ↔ア (same sound, other shape) | 46 | +0.091 | +0.073 | 62.0 | 21 / 46 | 1 |
| 明↔日 (kanji ↔ IDS component) | 23 | +0.030 | +0.004 | 67.2 | 67 / 200 | 0 |

Mantel test, Δ-cos vs glyph pixel-cos (342 single-char rows, Noto Serif CJK
Black 48 px): spearman +0.27, permutation p 0.003; within family hira
+0.14, kata +0.18, kanji +0.18. So a shape signal exists — glyphs that
differ by a small mark land nearer than random, and pixel similarity
explains a small share of the geometry — but it is nowhere near a
coordinate system (median partner rank 5 of 50 for か↔が, no top-1 hits),
consistent with the W2d verdict that held-out shapes get nothing and with
the び→ぴ / ぼ→ぽ reader confusions on the P0b sheets.

Hit correlates (50 evaluated single-char rows, spearman): along-m̂ / cos to
mean **+0.32**, exposure +0.24, norm +0.21, nearest-neighbour cos (crowding)
−0.09. Per family, hit rows vs miss rows: hira norm 0.43 vs 0.31, kata
0.37 vs 0.29, kanji 0.91 vs 0.84 — crowding equal (≈ 0.2 both). The row
that renders is the row that grew, and grew along the shared direction.

## Adapter output (433 rows × 6 frames; d = code with delta − code without)

| frame | cos(d, Q_frame) | cos(mean d, Q_avg) | d energy in EN-quoted top-16 (untrained row / gauss) | cos to mean EN quoted code: on / off | frame shift vs Q: trained / untrained (EN word) | d pairwise cos |
|---|---|---|---|---|---|---|
| `Japanese text reads as "…"` | −0.020 | −0.037 | 0.049 (0.103 / 0.015) | 0.31 / 0.42 | 0.35 / 0.37 (0.73) | 0.156 |
| `There is a speech bubble that reads "…"` | −0.025 | −0.087 | 0.049 (0.104 / 0.015) | 0.31 / 0.43 | 0.38 / 0.39 (0.73) | 0.164 |
| `She is saying "…"` | **−0.030** | −0.066 | 0.052 (0.114 / 0.015) | 0.34 / 0.45 | 0.40 / 0.43 (0.74) | 0.158 |
| `She is holding a sign that reads "…"` | −0.020 | −0.110 | 0.047 (0.096 / 0.015) | 0.33 / 0.44 | 0.45 / 0.45 (0.74) | 0.163 |
| bare `"…"` | −0.006 | +0.046 | 0.051 (0.102 / 0.015) | 0.30 / 0.41 | 0.25 / 0.28 (0.69) | 0.156 |
| plain (no quotes) | — | +0.181 | 0.047 (0.062 / 0.015) | 0.29 / 0.38 | — | 0.143 |

Random-direction |cos| scale 0.025. Per family, cos(d, Q_avg) is −0.04 …
+0.02 under every frame (words +0.02, nothing else positive). The frame
shift a trained row receives is 0.46–0.56 of its code norm (untrained row
0.67–0.93) — the delta enlarges the code, the frozen adapter adds the same
frame vector on top; the shift's alignment to Q is unchanged by training
(on ≈ off), so the adapter contextualises an ext row the way it always did,
about half as cleanly as an EN word, and the trained part adds no Q
component.

Frame-to-frame cos of d: 0.91–0.94 among the five quote frames, 0.73–0.80
against plain. After projecting out d(`reads as`), the frame-specific
residual is 0.32–0.38 of |d| and its cos to that frame's Q is −0.07 … +0.06:
the image is context-free at the adapter, and the small frame-specific part
is not Q either. The shared component of d (cos of a row's image to the
mean image 0.34–0.50 per family, PR 74–87) is the row-space m̂ carried
through; hit correlates with it (spearman +0.31), i.e. **the rows built
their own render trigger, shared across the table, and it is ⟂ Q** — the
same verdict `synth_s0_s0b_2026_09_15.md` reached for the explicit `c_flat`,
now for a table that had no `c_flat` and trained under four frames.

## What this settles

- No manifold in the trained rows to exploit for new glyphs (interpolation,
  nearest-row init, composition) — the W2d "held-out flat" verdict holds
  at 433 rows; the residual geometry is near-orthogonal with a weak
  small-mark neighbourhood (ば↔ぱ), which is also where the readers confuse.
- Frame-mix training does **not** pull the rows onto the pretrained quoted-
  text representation; `She is saying "…"` shares nothing with the delta
  beyond what the frozen adapter applies to any row. Frame independence in
  the 2×2 (swap 23 → 46) came from exposure under the frames, not from a
  shared code. Don't re-propose "align rows to Q" or "seed from the quoted
  EN code" on the strength of the frame mix.
- Katakana's 1/12 is not interference in row space (no crowding, no
  hira↔kata proximity); the rows are under-grown at equal exposure —
  points at the render / reader side (fonts, JA readers' katakana bias) or
  a loss-magnitude effect of simpler strokes. The planned 92-kana 23 k arm
  still separates interference-in-the-DiT from render-side; row space
  already rules out interference-in-the-table.
- Word rows are untrained at norm 0.12 — the word eval measures nothing
  about words yet; any word arm needs its own exposure share pinned.
