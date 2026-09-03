# Ext-table rank × arm C — C8 (r256) reads best (2026-09-03)

**Status: user eyeball verdict, one training run per arm.** Over the
`cjk_unmask_eval2` grids (8 rows × 3 seeds, same mirror / PP-OCR v1 records /
latents / seeds for every arm, only the ext pack differs) the user's read is
**C8 (rank 256) > C7 (rank 128) = C2 (synthjako2)**, i.e. **rank 256 is the
best pack so far**. This revises the same-day findings §10 tally (C2 0 · C7 ~3),
which counted non-diegetic text events cell by cell; on a whole-grid quality
read C7 lands level with C2 rather than below it, and C8 above both. The
§10 rule still applies: one run per arm ranks nothing until a ≥3-seed
same-recipe rerun exists — this report records the current best guess, not
a settle.

## The rank chain (distill side)

All packs are the cold joint JA+KO+ZH `synthjakozh1` recipe (`param=global`,
span loss, `--trust provenance`, attn blocks 0/13/27, 12k steps, bs 32, lr
1e-3, ext v2 init); only `--rank` (and `--freeze_diag` for fdiag) differs.
Distill jobs: r64 `20260902-2322…`, fdiag `20260903-082912-2bbe09`, r128
`20260903-093812-9b6190`, r256 `20260903-123919-f42668`
(`bench/cjk_distill/results/20260903-{0829,0950,1239}-2c-synthjakozh1-*`).

| pack | rank | final span | recovery | cos(s,t) | disc_far | gain | keys PR / >0.5 (OCR-visited) | codes PR / >0.5 |
|---|---|---|---|---|---|---|---|---|
| ext v2 init | — | 0.490 (first) | — | — | — | — | 234 / 0.1% | 186 / 0.0% |
| synthjakozh1 | 64 | 0.106 | 0.019 | 0.139 | 0.060 | 0.286 | 54.0 / 2.6% | 99 / 0.6% |
| synthjakozh1_fdiag | 64, diag frozen | 0.106 | 0.019 | 0.140 | 0.060 | 0.216 | 53.4 / 2.5% | 93 / 0.6% |
| synthjakozh1_r128 | 128 | 0.088 | 0.018 | 0.139 | 0.058 | 0.296 | 90.6 / 0.4% | 112 / 0.3% |
| **synthjakozh1_r256** | **256** | **0.073** | 0.018 | 0.139 | 0.056 | 0.299 | **140.8 / 0.1%** | **124.4 / 0.2%** |
| synthjako3 (warm, v1) | 64 | 0.111 | 0.008 | — | — | — | 79 / 0.4% | 107 / 0.4% |

Geometry probe: `probes/spread_probe.py` on the same 864 OCR-visited ext rows
(r256 probe job `20260903-123919-212039`). Wall is flat across rank (~2350–2390
s for 12k steps), so rank is free at this scale.

Reading:

- **Rank is the dispersion lever** (§10's correction of §9 holds): the
  trained table is `(init + init·down·up)·diag·gain`, and the shared low-rank
  term dominates the row. PR of the visited keys tracks rank almost linearly
  (54 → 91 → 141) while collisions fall to the init floor (2.6% → 0.4% →
  0.1%). r256 ends **above** the warm JA+KO chain (jako3 79) on both axes.
- **Fit tightens with rank** (span 0.106 → 0.088 → 0.073) while the
  **holdout is unchanged to three decimals** (recovery 0.018, cos(s,t)
  0.139, disc_far 0.056–0.060). So the extra capacity is spent on
  per-row directions the teacher rewards on the visited set, not on
  generalising to unvisited rows — exactly what arm C should care about
  (visited rows are the OCR rows the captions carry) and tier-1 prompting
  should not.
- r256's random-2000 rows still read PR ~19 / 1.8% collisions — the
  unvisited tail is clustered by the v2 init, not a training effect (§10);
  compare packs on the visited set only.

## Arm C readout (train side)

C5 / C6 / C7 / C8 = the C2 recipe (`cjk_unmask_c*.toml`, plain LoRA dim32,
lr 2e-5, 8 ep, `masked_loss=false`, `sincos` shard, `mirror_sincos_ppocr`,
`--ocr_format tags`, v1 records) re-cached through r64 / r64-fdiag / r128 /
r256 respectively. C8: job `20260903-132944-dfd00e`, TE cache
`post_image_dataset/cjk_unmask/te/sincos_jakozh1_r256`, grid
`output/tests/cjk_unmask_eval2/armC8_s{42,7,1234}`.

Cell notes on C8, against the cells where C6/C7 failed (§10):

| cell | C6 / C7 | C8 |
|---|---|---|
| s42 r1 classroom | pseudo-JA band across the top (both) | clean |
| s42 r6 cafe | poster text (both) | poster text ("CIBBOMG"-style) — the one shared failure cell |
| s42 r7 portrait | cat/ribbon ears (both) | clean |
| s7 r8 comic | C7: pseudo-JA spam over a chibi (worst comic cell) | bubbles filled with pseudo-JA, but panel-shaped and diegetic for a `comic` prompt |
| s7 r7 portrait | C7: broken | clean |
| s1234 r2 bedroom | — | lineart (the C5 kind) |
| s1234 r5 windowsill | C6 loses the figure | figure survives |
| s1234 r6 maid | ears in A/C3/C5 | ears |
| s1234 r8 comic | — | small `Aou` scribble top-left |

Net: C8's non-diegetic text events are the s42 r6 poster plus the s1234 r8
scribble (~1–2), against C6 ~2–3 / C7 ~3 / C3–C4 3, and it does not spend
that on figure loss or broken portraits. The lineart-on-colour drop (r2 at
s1234) and the s1234 r6 ears are the recipe-level artifacts every arm C shows
and are not rank-attributable.

**Verdict as recorded**: C8 > C7 = C2; **rank 256 is the working default for
the next pack** (the cost is nil — same wall, same holdout). Rank 512 is the
obvious next point on the chain (PR headroom: init 234, r256 141) but only
worth a distill if the arm-C rerun below confirms the direction.

## What this does not settle

- **One run per arm.** §10 measured the arm-C run-to-run noise at the size of
  the whole C2 (0) … C4 (3) spread, so the ordering above is a single draw.
  The cheapest settle is still the seed-control: C2 and C8 each at 3 training
  seeds of the same recipe. Until then "rank 256 best" is a preference, not a
  finding, and should not be copied into `findings.md` as one.
- **Same-day pipeline effect.** C6/C7/C8 all failing s42 r6 while C2/C5 do not
  is either lottery or something in the r2-era caption/cache path; the
  `ocr_format order` + v2-records arm (still to run, §10) is the first arm that
  changes that path and will confound rank if run on r256 alone — run it on
  r64 too, or read rank and format as separate arms.
- Geometry-vs-render: the r128 arm (C7) moved geometry 54 → 91 without a
  render improvement on the event tally; r256 moved it to 141 with one. If the
  rerun holds, the useful statement is "collisions at the init floor
  (≤0.1%)", not "PR up", since r128 already had collisions at 0.4%.
