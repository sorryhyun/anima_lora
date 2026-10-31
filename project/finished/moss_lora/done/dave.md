# DAVE on MOSS-Image

Shipped 2026-10-18 for Base at s 0.3 · τ 0.10, pool blocks 19–30: `dave=auto` +
`~/ComfyUI-Anima-DAVE` 0.2.0. Port of Anima's DAVE (`docs/anima/inference/dave.md`) to
`library/moss/dave.py`; user doc `docs/moss/inference/dave.md`. Diversity tool only: same-prompt
seed variety, at the cost of text / hand coherence when the window is too wide.

## Differences from Anima

- **Image tokens only**: `μ` is the per-channel mean over the image tokens of
  `[image | text]`, and only those are edited.
- **Seam = `MOSSImageTransformer.layer_hook`**, on the rows that ran the block (under
  skip-22 the hook at 22 sees the cond row alone).
- **τ is a step count**: steps `0 .. round(τ·steps)−1`, at least one.
- **Spectrum**: DAVE lands only on full steps; at 50 steps τ 0.10 (5 steps) sits inside
  the 6-step warmup, and `generate.py` notes when τ reaches past it.
- **Massive channels are left out** of the subtracted mean:
  `library/moss/calibration/massive_channels.npz` (18 channels, the union of each block's
  top 4, 1356 and 42 among them), written by
  `scripts/calibration/moss_massive_channels.py --probe <run>/channels.npz --k 4`.
- **Pool from a DAVE-on sweep**, not `derive_alpha_mask.py` (its `DC_sim` cut does not
  separate MOSS blocks): `library/moss/calibration/dave_alpha.npz`.

## Findings

**Premise probe** (`_archive/bench/moss_dave/results/20261018-1048-vanilla`, anime_sunset + ja_manga ×
8 seeds × 30 steps, CFG 3, skip 22, 1024²). Gate passes (early DC 0.998, AC 0.474, gap
+0.52) but saturated: DC sim ≥ 0.99 on blocks 0–40 and ≈ 1.000 on 0–20 from step 0; the DC
holds 0.98–0.99 of image-token energy on blocks 0–18. AC sim falls over the run (block 20:
0.96 → 0.27).

**Channel probe** (`_archive/bench/moss_dave/results/20261018-1106-channels`, + danbooru_tags). The
DC is massive activations: ch **1356** holds 0.87–0.93 of the early DC energy on blocks
0–21 (|μ| ~1900× median at block 0, ~210× at 30); 1356 and **42** are top 4 on 41–42 of 42
blocks. Cross-prompt DC cosine 1.00 with all channels in (no conditioning). Top 4 out: power
ratio 0.18 (block 0) → 0.29–0.37 (8–30) → 0.47–0.80 (36–41), DC−AC gap +0.33–0.81, so the
premise holds on the rest. Prompt specificity: 0.01–0.03 on blocks 0–3, 0.08–0.09 on 8–12,
0.05–0.07 on 13–36, 0.09–0.11 on 38–40.

**Pool sweep** (`_archive/bench/moss_dave/results/20261018-{1915-dave-8-18,1958-dave-19-30,1928-dave-8-30,1942-dave-8-40}`,
s 0.3, τ 0.10 = 3 steps). Cross-seed AC sim (lower = more diverse):

| | vanilla | 8-18 | 19-30 | 8-30 | 8-40 |
|---|---|---|---|---|---|
| AC sim step 3 | 0.531 | 0.520 | 0.442 | 0.432 | 0.448 |
| AC sim step 29 | 0.406 | 0.379 | 0.209 | 0.205 | 0.192 |
| DC sim steps 20–29 | 0.980 | 0.981 | 0.965 | 0.966 | 0.953 |

Anima's pool (8–18) barely moves MOSS; 19–30 alone does what 8–30 does. The DC-sim drop
comes with 19–30.

**Eyeball, 19-30 · s0.3 · τ0.10** (`output/moss_dave/eyeball_1018/`, ER-SDE 50 steps, ×4
seeds vs vanilla). Text legible on all four; bubble placement, framing and style vary
where vanilla pins them; character traits hold; one seed in four turns the sunset sky blue.
Hand close-ups recompose at a visible cost (a hand half through the mug, a softer render).

## Open

- τ 0.06 (3 steps at 50) for hands; higher dose; SFT, including whether it has the same
  massive channels.
