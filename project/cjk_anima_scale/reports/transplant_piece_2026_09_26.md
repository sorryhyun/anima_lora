# transplant_piece — proposal Stage A: the piece direction, out of sample (2026-09-26)

`../proposal.md` § 1. Every earlier read of the shared Δ was in-sample
(`transplant_line_2026_09_26.md` § 4). Here the direction is estimated on
292 of run0926_300f_sp's pieces and added at one coefficient to the seed
rows of the 8 pieces the piece ruler reads, which never contributed to it.
**Verdict: the direction transfers. `u_P` at α 1 buys half of 300f_sp's
contained gain over the floor (27 → 52 / 256, 300f_sp 76), and the
norm-matched random control goes the other way (11, below the floor;
u1 vs random 45 / 4, p 8e-10). Official moves +6 against 300f_sp's +13,
and that alone is not significant (p 0.11). No doubling in any arm. The
Stage A decision rule passes on contained and sits just under it on
official, so Stage B (a singles donor) is worth its budget.** One job,
`20260926-144514-d7e712`, 16 renders per cell.

Script: `experiments/transplant_piece/run_exp.py`. The envelope
`results/20260926-1445-a1/result.json` is the eval leg. The build leg's
envelope was written to the same minute dir and was overwritten
(`project_bench_run_dir_collision`). The per-row move is still recorded in
each arm's `trained.pt` (`transplant.moved`) and in the table below.

## 1. The direction (CPU)

- `u_P` = the normalized mean of the 292 donor pieces' Δ vs the seed, each
  with its component along its own seed row removed. Split-half cos (even
  vs odd donors) is **0.97**. `u_P` holds 18.5 % of the donors' Δ energy.
- Step = the donors' mean projection on `u_P`: **94.1** (sd 21.8).
- The held-out 8 as 300f_sp trained them: their Δ projects 93–136 on
  `u_P` (Δ norm 200–283), `u_P` holds 24 % of their Δ energy, and their
  mean Δ direction is at cos **0.83** to `u_P`. So the 8 already sat on the
  donors' direction; the question was whether that part alone renders.
- Leak: **none**. 300f_sp trained on `scene_piece` only, and every one of
  its 20 000 items carries one unit, so no donor item shows a held-out
  piece. (The proposal expected grid-string leak; that applies to
  run0925_300f, not this run.)
- The move is large: step is 65–85 % of a held-out seed row's norm (seed
  norms 112–149). Row-norm ratio after the move: u0.5 ×1.10–1.15, u1
  ×1.28–1.41, rand1 (a unit direction ⟂ `u_P`, same step) ×1.16–1.32.

## 2. The read

Piece ruler: the 8 pieces alone, native scenes, en + swap, 8 prompts ×
2 seeds (/ 256). The floor comes from the seed dir's `native_piece/` cache
and 300f_sp's reads from its run dir; no floor renders.

| arm | official | loose | contained | repeat |
|---|---|---|---|---|
| floor (seed) | 3 | 6 | 27 | 2 |
| `tp_a1_u0.5` | 7 | 13 | 43 | 4 |
| **`tp_a1_u1`** | **9** | **16** | **52** | 3 |
| `tp_a1_rand1` | 5 | 5 | 11 | 0 |
| run0926_300f_sp (trained) | 16 | 19 | 76 | 5 |

Paired per render (same piece × clause × prompt × seed), McNemar
(gained / lost):

| comparison | official | contained |
|---|---|---|
| u0.5 vs floor | 6 / 2, p 0.29 | 22 / 6, **p 0.004** |
| u1 vs floor | 8 / 2, p 0.11 | 30 / 5, **p 2e-5** |
| u1 vs u0.5 | 4 / 2, p 0.69 | 23 / 14, p 0.19 |
| u1 vs 300f_sp | 7 / 14, p 0.19 | 21 / 45, **p 0.004** |
| rand1 vs floor | 4 / 2, p 0.69 | 6 / 22, **p 0.004** |
| u1 vs rand1 | 8 / 4, p 0.39 | 45 / 4, **p 8e-10** |

Contained per piece (en + swap, / 32):

| piece | floor | u0.5 | u1 | rand1 | 300f_sp |
|---|---|---|---|---|---|
| あと | 1 | 5 | 4 | 3 | 14 |
| きて | 4 | 5 | 8 | 2 | 13 |
| こう | 3 | 7 | 10 | 0 | 17 |
| こと | 6 | 9 | 11 | 2 | 14 |
| こんにちは | 0 | 0 | 0 | 0 | 1 |
| しい | 13 | 16 | 18 | 3 | 9 |
| ちょっと | 0 | 0 | 0 | 0 | 0 |
| った | 0 | 1 | 1 | 1 | 8 |

## 3. Read

- **A generic piece direction exists.** Norm alone does not produce the
  gain. The random ⟂ step of the same size scrambles the rows (contained
  27 → 11, reads turn into English-like gibberish or other kanji), while
  `u_P` moves them up. The direction was fit without the 8 and applied at
  one coefficient, so this is the first out-of-sample transplant on record.
- **It buys about half of what training bought, and the half is
  piece-dependent.** こう / こと / きて get most of 300f_sp's gain from
  `u_P`. あと and った get little (their training gain sits in their
  per-row rest). しい is above 300f_sp (18 vs 9) and was already the
  floor's best piece. The 3+ glyph pieces (こんにちは, ちょっと) are 0
  everywhere, 300f_sp included. The transplant does not reach what
  training did not.
- **α saturates early.** u0.5 → u1 is not significant on either count.
- **The piece direction carries no doubling** (repeat 2 / 4 / 3 vs floor 2),
  unlike the singles' `uB` (`transplant_line` § 4, repeats 19 → 37).
- **Caveat: part of the gain may be "more kana text".** u1 renders a kana
  read in 223 / 256 renders vs the floor's 179 (rand1 178). Some of the
  contained gain can come from the row producing Japanese text at all
  rather than its own glyphs. The official count, which needs both
  readers on the exact piece, is the part that is not subject to this,
  and it is only +6 (n.s.).

## 4. Next

- Stage B (`proposal.md` § 2) is warranted: a singles donor with held-out
  kana, read on spelled strings (single rows, spaced caption) — where the
  singles' line mode lives. It needs the new floor keys the proposal names.
- Cheap follow-ups on this data, no training: the `u_P` step onto the
  single rows of `こ ん に ち は` read en-only (the floor cache holds those
  keys). This tests the piece direction on singles at piece scale (94 vs
  transplant_line § 2's 14.4 / 43).

Repro:
```
export ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack
X=project/cjk_anima_scale/experiments/transplant_piece/run_exp.py
.venv/bin/python $X --label a1 --legs build
make daemon-run ARGS="--queue --stall-timeout 0 $X --label a1 --legs eval"
```
