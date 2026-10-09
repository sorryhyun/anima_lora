# phash_edit bench — raw verdicts

Checkpoint under test: `output/ckpt/anima_easycontrol_phash_edit.safetensors`
(2026-08-21 15:01, job `20260821-081803-dcc842` — the re-mine with identity 2%
+ colorize 20% arms; dim 32 / alpha 128 / b_cond_init −4 / 4 epochs).

## Edit probe (`run_edit_probe.py`, runs 2122 + 2135, 2026-08-21)

Train pairs, stratified by tag_delta; arms noec / noop_b0 / ec_b0 / ec_b−1 /
ec_b−2. Copy metric = MSE vs cond (copy-lock signature: ec ≈ noop).

- **The adapter is NOT globally copy-locked at b0.** Medium deltas (6–8)
  landed with identity preserved (wink+tongue `11478932`, heart-hands→arms-up
  `6951514`) — mse_vs_cond 0.014–0.017 vs noop 0.002–0.013, and the edits are
  visible in the renders.
- **Small deltas (1–3) are pure copy at b0** (`-cum`: ec 0.0020 ≈ noop
  0.0022; `faceless` Δ3: ec 0.0075 vs noec 0.166) — the same boundary
  twin_edit hit. One large pair (Δ13) also copied; the other (Δ14) moved.
- **b_cond offset is a binary switch, not a dial.** At −1 (let alone −2) all
  pairs jump to noec-level mse_vs_cond (0.14–0.38): cond identity gone
  entirely, even on pairs whose edit landed at b0. The cliff sits inside
  (0, −1); there is no usable midpoint. Offset-based copy-lock rescue is
  dead for this checkpoint.
- Tooling: delta-grammar prompts starting with a removal (`-cum`) must be
  passed as `--prompt=-cum` — bare `--prompt -cum` is eaten by argparse.
  Fixed in the probe; `make test-easycontrol PROMPT=-tag` has the same trap
  (env var path is safe, positional/extra argv is not).

## Masked compose (`run_masked_compose.py`, run 2142, 2026-08-21)

The archived directedit_ec "EasyEdit" recipe (DirectEdit inversion, Δz anchor
masked in the edit region, EC cond gray-holed over the same region) with the
phash_edit adapter swapped in. Two cases: `faceless` (Δ3, face box — the
copy-locked stratum) and `arms up, -heart hands` (Δ6, geometry stress).

| arm | adapter | prompts | faceless (out/in) | arms-up (out/in) | verdict |
|---|---|---|---|---|---|
| inpaint_full | inpaint | full ψ_src, ψ_src±δ | 0.0006 / 0.0219 | 0.0015 / 0.0708 | redrew a *face* — semantic edit did NOT land; arms-up degenerated (known geometry limit) |
| phash_full | phash_edit | full captions | 0.1046 / 0.1637 | 0.0050 / 0.0717 | **catastrophic** — full captions are off-distribution for the delta-trained adapter (washed-out/blurred whole frame) |
| phash_delta | phash_edit | src="", ψ_tar=delta | 0.0019 / 0.0130 | 0.0046 / 0.0173 | **faceless LANDED** (only arm that did) with near-inpaint preservation; arms-up did not land (global pink tint, hands unchanged) |
| phash_nomask | phash_edit | src="", delta, no masks | 0.0014 / 0.0019 | 0.0116 / 0.0212 | copy inside the hole too — confirms the probe's copy-lock control |

**Headline: the hybrid (phash_edit adapter + EasyEdit mask recipe + native
delta grammar) beat both parents on the small in-place edit.** Pure trained
EC copy-locks it; pure EasyEdit/inpaint regenerates plausible content but
ignores the semantic tag (`faceless`); the hybrid lands it at b_offset 0.

Caveats: n=1 winning case so far; geometry/pose edits still fail every arm
(archive's position-locked-prior limit stands); the adapter must be prompted
in its delta grammar — full captions break it; the pink/white tint on phash
arms in case 2 is plausibly colorize-arm (20%) bleed-through, untested.

## Diff localization (`run_diff_localize.py`, run 2225, 2026-08-21)

Phase 0 for the position-clause proposal
(`_archive/proposals/phash_edit_position_clauses.md`): CPU 64×64 diff over all
1,856 unique edit pairs — where does each pair actually differ?

- **30.0% single-region** (top blob ≥ 75% of diff mass) — one clause
  suffices; conservative floor, since the 69.6% "multi" class is mostly one
  real edit region plus decode-noise specks (median top_share 0.44; 481
  multi pairs have top_share ≥ 0.5 → mass-floor merge should reach ~50–60%
  clause-addressable).
- **Header histogram non-degenerate**: bottom 173 / center 150 / the seven
  others 14–46. Not everything lands "center".
- **Label noise negligible**: 7 near_zero + 2 global out of 1,856 (6
  silent-delta suspects). The mined data is clean — this *weakens* the
  "dirty labels taught the variant-axis prior" reading; the live mechanism
  is instruction under-determination, not bad pairs.
