# f0_interaction — Stage F0: does the adapter modulate a row by its neighbours? (2026-09-26)

`../proposal_factorizedrows.md` § 2. Text encoder + `llm_adapter` forward
only (fp32, the pack hooked, an `ExtDelta` carrying effective rows), no DiT,
no renders. **Verdict: the adapter passes a row's change through almost
blind to its neighbours. At the crossattn output, a glyph's change in a word
sits at cos 0.95–0.97 to the same change alone, with gain 0.97: the change
is not smaller alone, it is ≈ 3 % larger. The line-trained donor rows behave
like the norm-matched random direction (0.967 vs 0.962).** So the adapter
cannot tell "alone" from "in a line" for a row's own change. A gate has to
act outside it, at the embed hook, as § 1 is written. F1 is the next GPU
spend. One job, `20260926-192406-09fee4`, ≈ 30 s.

Script: `experiments/f0_interaction/run_exp.py`; envelope
`experiments/f0_interaction/results/20260926-1924-f0/result.json`.

## Setup

- Captions: the native ruler's 8 scene prompts × en + swap, with the glyph
  alone (`"ひ"`) or inside a spelled word (`"ひ ま わ り"`). The script
  asserts that each caption's only pack rows are the word's glyphs.
- At `g`'s position, per block boundary (`L0` = block 0's input, `B1`–`B6`,
  `out` = the embedding the DiT reads):
  - `D_lone` = seed + arm row on `g`, minus seed, with `g` alone
  - `D_self` = the same row change, with `g` in the word (neighbours at the
    seed): the adapter's own context modulation
  - `D_all` = every glyph of the word on its arm row: what a render sees
- Arms (Stage B, on disk): **u1** (held-out seed rows + step · `u_S`, the
  5 held-out words), **rand1** (same step along a random ⟂ unit), **donor**
  (the Stage B donor's trained rows, こんにちは + 12 donor words of 3–5
  glyphs, seed 0 sample).
- Metrics, averaged over pairs (the same glyph, prompt and clause alone):
  - cos(D_x, D_lone)
  - interaction ‖D_x − D_lone‖ / ‖D_lone‖
  - gain = D_x · D_lone / ‖D_lone‖²
  - the seed's own context cos (H_seed at `g`, line vs alone)

## 1. At the output

| arm | pairs | cos self | cos all | inter self | inter all | gain self | gain all | seed ctx |
|---|---|---|---|---|---|---|---|---|
| u1 | 256 | **0.954** | 0.902 | 0.30 | 0.46 | 0.968 | 0.973 | 0.965 |
| rand1 | 256 | 0.962 | 0.944 | 0.27 | 0.33 | 0.960 | 0.954 | 0.965 |
| donor | 832 | 0.967 | 0.924 | 0.24 | 0.38 | 0.971 | 0.967 | 0.975 |

- **The row's own change (self) passes the § 2 rule, cos ≥ 0.95, in all
  three arms.** u1 is the weakest at 0.954. The interaction is the size a
  random perturbation gets (0.27–0.30): generic nonlinearity, not a
  line-specific response.
- **Gain < 1 everywhere**: the change at `g` is a few percent *smaller* in
  the line than alone. "The adapter already turns the mode down alone"
  (§ 2's second branch) is ruled out. If anything, the direction runs the
  other way.
- **Training in lines made no context-dependent row.** The donor rows,
  trained on 568 words plus a count tier, interact less than the random
  arm's step (0.24 vs 0.27).
- **The one `u_S`-specific effect is collective.** With every neighbour
  carrying `u_S` (all), the interaction at `g` is 0.46, against 0.33 for
  random. Most of it is the neighbours' change leaking into `g`'s position:
  neighbour spill 0.19 of ‖D‖ at `g` for u1 vs 0.14 for random. It adds a
  component off `D_lone` (cos 0.90), not more of it (gain 0.97). Under the
  factorization every row in a run carries `v_line`, so this is the
  "gate on" state and needs no gate.
- By position (u1, self): first cos 0.972, inner 0.948, last 0.944. By
  glyph, わ is the outlier (0.91 · gain 0.86); every other glyph is
  0.94–0.98.

## 2. Where the interaction grows

Interaction (self) climbs smoothly with depth in every arm (u1: B1 0.08 →
B3 0.16 → B6 0.31). By branch, divided by ‖D_lone‖ at the block output
(u1):

| block | self-attn | cross-attn | MLP |
|---|---|---|---|
| B1 | 0.080 | 0.054 | 0.049 |
| B3 | 0.081 | 0.085 | 0.085 |
| B5 | 0.131 | 0.067 | 0.142 |
| B6 | 0.142 | 0.021 | 0.166 |

Neighbour dependence builds through self-attention and the MLP in the last
two blocks. Cross-attention to the Qwen states is the smallest branch at
the end. rand1 and donor have the same shape at about 0.8×. Nothing points
at a single branch to localize.

## 3. What it decides

- **§ 2 branch 1 ("interaction ≈ 0, the gate acts outside the
  adapter").** The adapter hands the DiT nearly the same row change whether
  the glyph stands alone or sits in a word. A line mode baked into `r_i`
  therefore reaches the DiT in both contexts, which is doubling A. A gate at
  the embed hook (§ 1, `proposal.md` § 3.4 (0)) is the place to switch it.
- **F1 is the next GPU spend** (§ 3: `--modes line` in `rows.Rows` + the
  hook, one train + read). This read also sets F1's bar. Gradient cannot
  use the adapter to route the line behaviour by context, so the split
  between `r_i` and `v_line` has to come from the data's gate on / off
  exposure. The count tier's coverage hole (no alone items at the b0305 px)
  is the known risk (§ 3, F1b).
- The opinion's adapter replay / distillation (§ 6 item 7) stays parked. F0
  found no context-dependent component worth localizing.
- Caveat, as for every adapter-space read here: adapter distances have not
  predicted renders (piece ↔ glyph R² ≈ 0.03). This read picks where the
  gate lives. It predicts nothing about what renders.
