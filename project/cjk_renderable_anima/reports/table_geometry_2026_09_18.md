# Table geometry under ΔFM — the free reads (2026-09-18)

> `src/probe/table_geometry.py`, no GPU: nine rows-arm tables (plain-FM
> `src53k` / punct-only / Δ0 `pair0_*`; ΔFM Δ1 / Δ0 `pairEN_*` / `rb62f` / `w8`),
> rows in absolute units (`raw × row_scale`). Discharges plan_synth3 K0(a) and
> the free rows of plan_synth4 R4.0. **The shared direction is per-loss, not
> per-run**: ΔFM tables agree at cos 0.66–0.87 across datasets, inventories
> and sizes (Δ1's 356 rows ↔ Δ0's 12: 0.67–0.72), plain-FM tables at
> 0.65–0.69, and the two losses at 0.27–0.51. Δ1 ↔ punct-only is **0.04** —
> K0(b)'s merge is the worst case the line has. ΔFM rows are orthogonal to
> the pack row they perturb (cos 0.06 vs plain 0.41): plain FM's shared
> direction was mostly the pretrained row itself. No shape structure in any
> table (neighbour = control), so W2d's premise holds under ΔFM too.

## Per table

| table | rows | row_scale | ‖row‖ | shared m̂ energy | PR | pair cos mean / p95 | resid cos mean / \|mean\| / p95 | が↔ガ, が↔か (n) | same-row ctrl (n) | neighbour resid / ctrl resid | cos to pack row (full / resid) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| src53k | 434 | 196.4 | 113.7 | 0.273 | 12.4 | 0.154 / 0.351 | 0.004 / 0.050 / 0.118 | 0.256 (50) | 0.293 (100) | 0.085 / 0.084 | 0.405 / 0.333 |
| punct | 17 | 191.9 | 120.9 | 0.280 | 8.4 | 0.190 / 0.334 | -0.059 / 0.077 / 0.045 | — | — | — | 0.430 / 0.328 |
| **d1** (ΔFM) | 356 | 232.9 | 78.7 | **0.109** | **57.1** | 0.100 / 0.199 | -0.002 / 0.042 / 0.089 | 0.177 (50) | 0.167 (100) | 0.066 / 0.047 | **0.062 / 0.029** |
| pair0_s3000 | 13 | 247.0 | 149.2 | 0.283 | 7.9 | 0.204 / 0.322 | -0.083 / 0.088 / 0.008 | 0.252 (6) | 0.240 (20) | -0.063 / -0.074 | 0.111 / -0.016 |
| pair0_s750 | 13 | 247.0 | 162.3 | 0.255 | 8.6 | 0.176 / 0.296 | -0.083 / 0.086 / 0.001 | 0.225 (6) | 0.209 (20) | -0.057 / -0.086 | 0.143 / 0.024 |
| pairEN_s1500 | 13 | 247.0 | 143.6 | 0.324 | 6.7 | 0.217 / 0.358 | -0.081 / 0.085 / 0.001 | 0.287 (6) | 0.239 (20) | -0.050 / -0.085 | 0.122 / 0.026 |
| pairEN_s750 | 13 | 247.0 | 137.5 | 0.257 | 8.6 | 0.171 / 0.296 | -0.083 / 0.086 / -0.003 | 0.212 (6) | 0.206 (20) | -0.079 / -0.077 | 0.110 / 0.029 |
| rb62f | 12 | 247.0 | 130.9 | 0.219 | 9.2 | 0.144 / 0.236 | -0.089 / 0.092 / 0.001 | 0.146 (6) | 0.144 (20) | -0.087 / -0.080 | 0.107 / 0.038 |
| w8 | 12 | 247.0 | 112.6 | 0.197 | 9.5 | 0.115 / 0.195 | -0.088 / 0.091 / -0.000 | 0.128 (6) | 0.105 (20) | -0.073 / -0.090 | 0.080 / 0.035 |

`shared m̂ energy` = Σ(r·m̂)² / Σ‖r‖² for m̂ the mean row; `PR` = participation
ratio of the row matrix's spectrum (an effective rank); `resid` = rows with
the m̂ component removed; `neighbour` = the が↔ガ (script) and が↔か
(dakuten) pairs, `ctrl` = pairs inside one gojūon row (が↔ぎ), both
same-family; `cos to pack row` = each row against the pretrained ext row it
perturbs.

## Across tables

Upper triangle: cos of the shared directions m̂·m̂. Lower: mean per-row cos
on the ext ids both tables trained (n).

| | src53k | punct | d1 | pair0_s3000 | pair0_s750 | pairEN_s1500 | pairEN_s750 | rb62f | w8 |
|---|---|---|---|---|---|---|---|---|---|
| src53k | — | 0.590 | 0.273 | 0.686 | 0.652 | 0.348 | 0.315 | 0.260 | 0.152 |
| punct | 0.282 (2) | — | **0.043** | 0.450 | 0.447 | 0.140 | 0.163 | 0.140 | 0.064 |
| d1 | **0.140 (343)** | 0.080 (15) | — | 0.211 | 0.253 | **0.722** | **0.670** | **0.672** | **0.659** |
| pair0_s3000 | 0.296 (13) | 0.052 (1) | 0.163 (13) | — | 0.817 | 0.449 | 0.442 | 0.339 | 0.245 |
| pair0_s750 | 0.298 (13) | -0.016 (1) | 0.159 (13) | 0.506 (13) | — | 0.510 | 0.515 | 0.415 | 0.305 |
| pairEN_s1500 | 0.215 (13) | -0.050 (1) | 0.390 (13) | 0.311 (13) | 0.301 (13) | — | 0.871 | 0.761 | 0.731 |
| pairEN_s750 | 0.213 (13) | -0.024 (1) | 0.342 (13) | 0.288 (13) | 0.294 (13) | 0.542 (13) | — | 0.768 | 0.725 |
| rb62f | 0.185 (12) | · | 0.338 (12) | 0.202 (12) | 0.231 (12) | 0.401 (12) | 0.386 (12) | — | 0.811 |
| w8 | 0.148 (12) | · | 0.346 (12) | 0.185 (12) | 0.196 (12) | 0.390 (12) | 0.358 (12) | 0.584 (12) | — |

## Reads

1. **The shared direction is a property of the loss.** Every ΔFM ↔ ΔFM pair
   sits at 0.66–0.87 — Δ1 (356 rows, the `d1` pools, 53 k) against the Δ0
   smokes (12 rows, `pair_d0`, 750–1 500 steps) at 0.67–0.72, the same as
   Δ0 ↔ Δ0. Every plain ↔ plain pair sits at 0.59–0.69. Cross-loss pairs are
   0.27–0.51, and the one same-rows-same-data cross-loss pair (`pair0_s3000`
   ↔ `pairEN_s1500`, 0.449) is no closer than Δ1 ↔ `pair0` (0.21–0.25). K0's
   reading "per-run, the loss rotates it more than the data does" was half
   right: the loss decides it, the run barely does. What that means for
   K1's chunking: **blocks trained under one loss merge at ≈ 0.7, not 0.35** —
   the 0.35 was a cross-loss number.
2. **Δ1 ↔ punct-only = 0.043**: the merge K0(b) evaluates (`rows_synth_d1_merge_punct`,
   plain-FM punct rows overriding Δ1's own) has *no* shared component in
   common — a harder test than the shipped `merge_punct` (0.59) and than any
   same-loss chunking would face. If the donor block still renders there, a
   same-loss merge is safe a fortiori; if it fails, the read is "cross-loss
   merges are dead", not "merges are dead".
3. **Plain FM's shared direction is largely the pretrained row.** Plain rows
   have cos 0.41–0.43 to the pack row they perturb (0.33 even after removing
   m̂); ΔFM rows 0.06–0.14 (Δ1 0.06 / 0.03). Plain FM grows the row partly
   *along itself* — a gain on the pretrained piece — and that component is
   what the scene residual paid for; ΔFM cancels it. It is also why Δ1's
   table is small and isotropic (‖row‖ 79, m̂ share 11 %, PR 57 vs 12): the
   along-row part was a third of plain's energy. Consequence for the α
   lever (plan_synth4): scaling a plain table scales the pack-row gain with
   it; scaling Δ1 does not — the two are different levers and R4.3 should
   not expect the plain α curve.
4. **Same glyph, two losses, unrelated rows.** On the 343 ids Δ1 and `src53k`
   share, per-row cos is 0.14 — barely above the 0.10 any two Δ1 rows have.
   ΔFM Δ0 ↔ Δ1 on 13 rows: 0.34–0.39. A warm start across losses therefore
   starts near cold (the row units fix in `trainables.py` makes it the right
   *size*; the direction is still the source loss's), which prices S2a's
   `pair` arms warm-started from Δ1 above its `plain` arms.
5. **No shape structure, under either loss.** Neighbour pairs (が↔ガ, が↔か)
   equal the same-row control in every table (residual cos 0.085 vs 0.084 on
   `src53k`, 0.066 vs 0.047 on Δ1); the 12-row tables have residual cos
   −1/(n−1) exactly, i.e. the centred configuration and nothing else. W2d's
   "rows are near-orthogonal random directions" stands on ΔFM tables.

## What it changes

- plan_synth3 K0(a): done — the cos table above. K0(b) parked: Δ1 already
  trains the punct rows, so the merge as planned is a cross-loss transplant
  (override, cos 0.04) rather than the same-loss chunking K1 needs; the
  override table exists, its eval was cancelled (user, 2026-09-18).
- plan_synth4 R4.0 free rows: done. The α lever is loss-dependent in *what*
  it scales (read 3), so R4.3's plain-table α sweep is a different
  experiment from the ΔFM one, not a control for it.
- plan_synth2's "chunking" price: 0.35 → ≈ 0.7 for same-loss blocks.

## Addendum (2026-09-18 evening): Δ1 as a warm-start / embedding source — closed

Moved from `plan_synth4.md` R4.0 on 2026-09-19. `src/probe/script_blocks.py`;
Δ1's cos matrix was `output/wake_probe/rows_synth_d1_d1_s53k/manifold/pairwise_cos*`
(arm dir deleted 2026-09-19). Δ1 and the KR / ZH blocks trained on the
preview pack (`plan_synth3.md`, caveat at the top), so Δ1 here is the ΔFM
residual on the plain sentence table.

- Inside Δ1 the top-1 neighbour of a row is its shape confusable (れ↔わ,
  ン↔シ, 達↔遠, び↔ぴ; dakuten rows' top-5 are 66 % dakuten across scripts)
  but the tilt is small: top-1 cos 0.27 of which 0.17 survives removing m̂;
  μ alone explains 10.8 % of a row, the best neighbour 7.5 %, both 13.7 %.
  A shape neighbour is no better a start than `--init_anchor μ`.
- KR (가거고구/나너노누/다더도두) and ZH (气乐变见长时话 + 门书车们这) 12-row
  blocks on the Δ0 recipe: cos(m̂_block, m̂_Δ1) 0.43 / 0.47 (JA Δ0 0.67), so
  Δ1's μ gives a new-script row cos 0.22–0.24 at init (JA 0.31); KR↔ZH 0.66.
  Hangul jamo differences are not parallel (+0.04–0.06, random p95 0.12);
  a simplified row lands nowhere near its shinjitai (气↔気 rank 67/355,
  时↔時 308, 见↔見 339; controls reach the same max). 744-step rows, glyphs
  not rendering yet — geometry only.
