# proposal — identity as hard as a row takes, no template (2026-10-16)

**Aim** (user 10-16): train the cold rows' identity as hard as the row will
take, while the row learns no template or canvas: glyphs may be memorized,
layouts may not. Lens: Kim et al., "How Diffusion Models Memorize" (arXiv
2509.25705). Their σ = 1 signature is a constant of CFG on this path, so it
cannot be the meter (`reports/commit_2026_10_16.md`).

Revised the same day after a fable review. The changes: M1's word claims cut
back to singles; M2 gets σ 0.95 / 1.0, seed variance and controls; L1 moved
to last; L3 recast on σ × canvas from `probes/canvas.py`; a negative-prompt
probe added as P0a.

## 1. Where the meters stand

| meter | what it reads | status |
|---|---|---|
| **render identity grid** (`probes/identity.py`) | word exact / cer / dup, singles | **the verdict axis.** The floor's lead (words 19 against 7 – 12) is in dup and slot count, which no one-step meter sees |
| **M1** swapword, one step on held-out items (`probes/swapword.py`) | `I` = `V_in` (one meter, ρ 0.98), `G_wrong` | **singles only.** On the 10 run ends `I` against word exact is +0.42 (p 0.23); `T_in_lo` against single exact +0.75 (p 0.01). x2 s12150 → end raises `I` while word exact falls 12 → 9, so it cannot name a step budget (`reports/swapword_2026_10_16.md`) |
| M1's word-free side (`T_in_lo`, `rho`) | meant as the template | **blind.** The canvas is in x_σ. Per glyph it does not follow template exposure (ρ +0.14, n.s.). Whether its in-box low-pass energy is text-ness or a "large centred glyph" habit is undecided |
| commit step-3 seed diversity (`probes/commit.py`) | the plain-type template on the generation path | separates `retrain_kana` (0.40 / 0.58 of EN) from x2 (0.47 / 0.65); only 3 arms are read |
| **negswap** (`probes/negswap.py`, P0a) | the render with the word's swap as the negative prompt | **confounded** (`reports/negswap_2026_10_16.md`): the matched negative cancels the prompt's guidance, and every page washes out, the katakana control too. By eye, retrain_kana-trained rows letter thin dark type under it and the cold rows stay coloured: a lean that the template rides on S (bound to identity), not M |
| **M2** (`probes/freeswap.py`, P0b) | the template where the canvas is free | **gate fails** on the positive control (`reports/freeswap_2026_10_16.md`): the floor reads paler than x2 and sball_lr3 on 8 / 9 words each (−Δsat of M at σ 1), but `up` does not read above hp. Carried into L3 as a secondary read (`reports/l3_full_2026_10_16.md`) |

## 2. M2 — the template where the base chooses the canvas

- **x₀** = the EN render of the identity grid's prompt × seed (p00 / p01 ×
  seeds 0 – 5, on disk in `OUT/commit/EN/`). **x_T** = that seed's noise.
- **Captions**: the 9 hiragana `read` words plus K = 3 swaps each (M1's
  rule), `Text reads as "…"`.
- **σ** ∈ {1.0, 0.95, 0.9, 0.85}. At σ ≤ 0.9 the EN trajectory has already
  pinned the layout (`motivation2.md` § 1). So only σ 1.0 / 0.95 have a free
  canvas. At 0.9 / 0.85 what is still free is style.
- **Reads**:
  - `M` / `S` as in M1, outside and inside the EN text box. M's luminance /
    saturation shift in the box is the pale band.
  - The seed dependence commit found: the variance over the 6 seeds of
    x̂₀(arm) in the box, against x̂₀(ref), at σ 1.0 / 0.95. That is one
    forward per seed.
- **Controls**: the 4 katakana words are a negative control (rows shared,
  must not separate). `hira_cold_up` is the positive one (upper edge + 0.1;
  the sheets show it swaps scene parts).
- **Gate**: the floor's template read above both x2 and sball_lr3 on at
  least 8 / 9 words (sign test p ≈ 0.04 per pair). Katakana must tie, and
  `up` must read above hp. Otherwise M2 is not used, and the template is read
  on renders only (commit's cross-word variance, with `commit.py` keeping
  x̂₀ at a few steps).

## 3. The search protocol

- Every arm keeps saves at 50 / 100 / 150 a row (`--save_every`).
- **Verdict**: the identity grid (word exact, dup, singles), paired against
  the cached floor. The template read (M2 or the render reads) is the second
  axis. M1 is a cheap in-training gauge for singles only.
- An arm wins when it gains on the verdict without a higher template, or
  holds the verdict with a lower template. `dup` stays on the verdict side.
- Micro pairs first (≈ 25 min each), read by glyph and word.
- The base recipe is `hira_cold_sball_lr3`: the best cold singles (17 exact,
  dup 3).

## 4. Candidate levers

**L2 — the wrong word as a negative (a loss).** A hinge on the margin,
max(0, m − (L_wrong − L_true)) in the box, with one swapped caption an item
(K = 1, one more forward).
- It pushes a row's effect apart from its neighbours': the credit problem
  in `idea.md` (`probe_twin`: neighbours turn 0.6 – 0.9 as much). It differs
  from the closed `probe_cf` (the input) and from per-occurrence credit (the
  position).
- Gated to σ ≤ 0.75 so that it cannot move layout, with a small m, read on
  renders only.
- Risk: it may push the rows off the generic ink that the floor's singles
  use.

**L4 — a template anchor on the generation path.** Only if M2 puts the
template in M (P0a leans the other way: S). At σ ≥ 0.9, penalize the word-invariant part of the student's
shift (mean over a word and its swap) against the EN teacher, outside the box
and on in-box luminance / saturation. This is L_pres's extension, so it is a
warm-pass term (user 10-14) unless the user moves it.

**L5 — M1 as a training-time gauge for singles.** Log `I` and `G_wrong` on 32
held-out items at every save. It is not a step budget (§ 1).

**L1 — where the word is decided (σ allocation), last.** It rests on `I` at σ
0.9 against single dup (+0.55, all 17 points; 0.00 against word exact on the
ends). hp's bands already stop at 0.75 – 0.8. `probe_split`
(`_archive/reports/probe_split_2026_10_04.md` § 2) had the word fit living
above 0.75: `up_mix_s0.75` (kana_up's rows above 0.75) held 28 of kana_up's
29 words ≤ 1 edit, and `mix_up_s0.75` 15. The same report has a render
template read ready: flat white (16² patches with std < 6 and mean > 225),
where retrain_kana's singles score 0.236 against kana_mix's 0.154. Run only if L3 / L2 leave it open.

## 5. Phases

| phase | what | GPU | gate |
|---|---|---|---|
| P0a | negswap (done 10-16) | 20 min | confounded; leans template-in-S (`reports/negswap_2026_10_16.md`) |
| P0b | M2 with its controls (done 10-16) | 15 min | failed on `up` > hp; `pale` at σ 1 kept as a secondary read (`reports/freeswap_2026_10_16.md`) |
| P1 | L3 (the high-σ large glyph, in a scene / on a white page) | – | moved to the reports: `reports/l3_micro_2026_10_16.md` (null at 50 / row), `reports/l3_full_2026_10_16.md` (L3a at 150 / row) |
| P2 | L2 micro pair | 25 min | the same |
| P3 | the winner at 150 / row, then warm on `sent_kanji_pres`'s recipe | as § 8 | the dialogue ruler |

Not in scope: the word dup and slot count as a target of their own. They are
sequence-level (`motivation2.md` § 1, `_archive/idea.md` ④), but they sit on
the verdict axis, so every arm is read on them.
