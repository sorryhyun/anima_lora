# influence_target — the FM-loss target sees neither axis (2026-09-25)

Step 1 after `influence_smoke_2026_09_25.md`: before building a
row-conditioned bank, check that the target it would be pointed at can see
what the rulers see. Value-only, no gradients — the in-box FM loss on
matched correct / doubled renders of the 8 piecenat pieces, at the seed, the
step-5 000 (drift 1.39) and the final (drift 1.66) run0925_300f tables.
Envelope `experiments/influence_target/results/20260925-2037-t1/`; renders,
latents and every read under `output/cjk_anima_scale/influence_target_t1/`;
job `20260925-203743-0a4d69`, 7 min.

181 matched pairs (≈ 23 per piece, the dev item's own scene re-rendered
with the correct string and the last-glyph-doubled form under one rng —
same bubble, font, fill, orientation, tilt, colour; the doubled string a
step smaller to fit), 3 σ bins × 2 noises, noise and σ shared across the
pair and the tables. CIs are 95 % bootstrap over items.

## T1 — identity: does the loss gain rank the pieces like piecenat?

In-box loss gain on the correct render, seed − table (+ = the table fits
the correct render better), final table:

| piece | glyphs | gain @final [CI] | piecenat lenient | official |
|---|---|---|---|---|
| ちょっと | 4 | **+0.0196** [+0.014, +0.025] | 0 → 2 | 0 → 0 |
| ありがとう | 5 | **+0.0164** [+0.011, +0.022] | 0 → 3 | 0 → 0 |
| った | 2 | +0.0149 [+0.010, +0.021] | 0 → 12 | 0 → 3 |
| メン | 2 | +0.0107 [+0.006, +0.016] | 8 → 11 | 1 → 0 |
| こんにちは | 5 | +0.0093 [+0.004, +0.015] | 0 → 0 | 0 → 0 |
| です | 2 | +0.0093 [+0.005, +0.014] | 3 → 14 | 0 → 0 |
| しい | 2 | +0.0029 [−0.002, +0.008] | 13 → 12 (ceiling) | 3 → 2 |
| すごい | 3 | **−0.0007** [−0.004, +0.002] | 4 → 10 | 1 → **6** |

Spearman vs the lenient gain **+0.02**, vs the official gain **−0.06**;
2-glyph mean +0.0094 vs 3+-glyph mean **+0.0112**. At the 5 000 table the
gains are a fifth of the size with mixed signs (しい, すごい negative; ρ vs
lenient +0.36 on noise-sized numbers).

The two largest loss gains are the two long pieces that never rendered;
the one piece with a real official gain gets nothing. The delta reduced the
in-box loss on every piece, rendered or not — the loss is reading the
**pseudo-text line the rows learned** (the wipe of `next.md` § 4a), not the
glyph identity piecenat measured. Within the 2-glyph group the order is
compatible with piecenat (った > メン ≈ です > しい at ceiling), which is
why the smoke's E2 looked like tracking; across glyph counts it is not.

## T2 — doubling: does the correct-vs-doubled margin move?

M = L_in(doubled) − L_in(correct) under the correct caption; ΔM = M_table −
M_seed (+ = the table prefers the correct render more than the seed did):

| piece | doubled | M @seed | ΔM @k5 [CI] | ΔM @final [CI] | piecenat doubled |
|---|---|---|---|---|---|
| です | ですす | +0.000 | −0.0011 [−0.006, +0.005] | −0.0007 [−0.007, +0.006] | |
| すごい | すごいい | +0.014 | **−0.0040** [−0.007, −0.001] | −0.0005 [−0.004, +0.003] | yes |
| しい | しいい | +0.018 | −0.0047 [−0.010, +0.001] | −0.0004 [−0.006, +0.005] | |
| こんにちは | こんにちはは | +0.006 | +0.0007 | −0.0002 [−0.004, +0.004] | |
| ちょっと | ちょっとと | +0.004 | +0.0013 | +0.0035 [−0.001, +0.008] | |
| メン | メンン | +0.009 | +0.0001 | **+0.0045** [+0.000, +0.009] | yes |
| ありがとう | ありがとうう | +0.004 | +0.0024 [+0.001, +0.004] | **+0.0056** [+0.002, +0.010] | |
| った | ったた | −0.001 | +0.0037 [+0.000, +0.007] | +0.0062 [−0.001, +0.015] | |

Group means at the final table +0.002 / +0.002 (2-glyph / 3+), ρ(ΔM,
official gain) **0.00**; the CI clears zero on 2 of 8 pieces, both positive,
one of them a piece piecenat saw doubled (メン → メンン). The seed already
prefers the correct render one-step on 7 of 8 pieces, すごい by the second
largest margin — and すごい doubles in the samples. The one-step margin at
three σ bins is not the sampled doubling, and the delta moved it by a
third of T1's size, in no pattern.

## Verdict

**The FM-loss family — the in-box loss and the correct-vs-doubled margin —
sees neither piece identity nor doubling at the piece level across this
delta.** A price built on it would rank ちょっと above った and すごい at
zero: the wrong shares. This is the row-level echo of the line's oldest
guard (never gate on data-FM-MSE — `docs/guidelines/training.md`,
`fm_val_loss_uninformative`).

Consequences:

- **Validation influence with a loss target is closed** for this line, both
  axes. Bank v2 (row-conditioned) is not built; the estimator fix the smoke
  asked for would sharpen a number that points the wrong way.
- The smoke's E2 reading ("the loss tracked the real piece-native gains")
  is withdrawn — it tracked the pseudo-text line, which also improved the
  loss on pieces that never rendered. The smoke report and the piecenat
  report carry the correction.
- What survives of `idea.md`: only a **non-loss target** — anchoring on
  certified table deltas (Δ of a ruler-verified table, e.g.
  `micro_warm_0923`'s piece rows, against Δ_300f as the negative anchor).
  Untested, and the only candidate left; it is a table-space direction
  read, not a validation loss, so it inherits none of this report's
  machinery beyond the bank loop.
- The reads stay the rulers: piecenat with a seed floor for pieces,
  `native_sent` / `target` for the acceptance axis. `next.md` step (2), the
  scene_piece-only arm, is judged on those, not on any surrogate.

Repro: `run_exp.py --label t1` (defaults; `--dry_run` plans), submitted via
`make daemon-run`, `ANIMA_VOCAB_PACK` set.
