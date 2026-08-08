# E21 — the cancellation drawn over the network: cell-level g-ledger

| | |
|---|---|
| **Status** | **PLANNED 2026-08-08** — this file is the pre-registration, committed **before** `e21_cells.py` exists (theory-first, mirroring 19.1/20). |
| **Question** | E19 established the deep g-level anti-alignment as one global constant (route-uniform 19.0, depth/type-uniform 19.3, operating-point-invariant 19.6) at **pooled** granularity, and the r-level field figure (`e19/fig_score_field.py`) showed the r-level spatial analog is ubiquitous-but-mild, with pooling amplitude-weighted. Open, at the estimand where ρ̄ ≈ −0.91 actually lives (LoRA-gradient space): is the cancellation carried **pointwise by (depth × module-type) cells**, or by **cross-cell global modes**? And cell-wise, *where* does 19.4's phase-borne component live — uniformly (the "carried globally through the chain rule" reading) or concentrated in specific type bands? Payoff either way: the Yang–Song-style mechanism figure at the **correct estimand** — arrow pairs (B_cell, C_cell) over the network's depth×type lattice — with an honest caption dictated by the verdict. |
| **Depends on** | [E14](../e14/) (pooled ρ̄; digests only — its arm sums were deleted); [E19](../e19/) 19.3 + 19.4 **surviving `arm_sums/` stores** (verified on disk 2026-08-08, see Sources); `paper_bench/vector_ledger.py` / `ledger_depth.py` / `ledger_pi.py` (leg + debias + gate conventions); `e19/fig_score_field.py` (r-level counterpart); G11 (PI off-manifold at mid σ — standing restriction) |
| **Instruments** | 21.1 `e21_cells.py` (CPU; cell ledger + lattice figure); 21.2 free re-reads of 21.1. **No GPU item** — see Cost ladder. |
| **In the paper** | Candidate mechanism figure for paper 2 (the reduction's two legs drawn over the network); the locality verdict feeds §4.5/4.6 language — 19.4's "phase share carried globally" becomes testable at cell granularity. |

**Numbering note**: the "derive the amplitudes" idea informally sketched
as "E21" in e20's 20.1 results paragraph is **not** this experiment — it
remains unproposed (blocked on the estimand bridge, 20.4).

## Sources (verified on disk before this pre-registration)

| store | σ centers | routes | arms | size |
|---|---|---|---|---|
| `bench/results/20260807-0745-e193-depth-ledger/arm_sums/` | 0.3, 0.4333, 0.5667, 0.7 | 896, 768 | native a/b, reenc, demote, rp (+`__2` cross-set twins) | (19.3 run) |
| `bench/results/20260807-1400-e194-pi-causal/arm_sums/` | 0.7, 0.8333, 0.9625, 1.0 | 896, 768 | same **+ `<e>pi`** (+`__2`) | 19 GB |

Both are `ArmSumAccumulator` dumps: one fp32 flat LoRA-gradient sum
(77.7M dims) per (arm, bin), two independent draw sets, `groups.json`
mapping (depth-block × module-type) → index spans. σ = 0.7 appears in
**both** stores → free cross-store consistency bin. E14's own arm sums
no longer exist; these two stores are the only g-level vector material
on disk.

## Frozen conventions (pre-registered — no tuning on outputs)

- **Cells** = `groups.json` spans exactly (depth-block × module-type),
  the 19.3 partition, unchanged.
- **Legs** = `ledger_pi.py` verbatim: B = ḡ_rp − ḡ_reenc (reenc ref
  primary), C = ḡ_dem − ḡ_rp, C_π = ḡ_dem,π − ḡ_rp. Second moments
  from **cross-set products only**; ref-noise subtraction where the
  ledger applies it; same-set values as bias checks; per-cell ⊥ and
  reliability mirroring `ledger_depth.py`'s per-slice convention
  (rel_cell from the `__2` twins; **gate rel ≥ 0.5**, failing cells
  drawn hollow and excluded from verdict statistics, count reported).
- **Verdict bins**: σ ∈ {0.3, 0.4333, 0.5667, 0.7, 0.8333}. The
  0.9625 / 1.0 bins are endpoint modes — plotted detached, non-verdict.
- **C_π only on the 19.4 grid** (0.7, 0.8333 verdict; endpoint
  detached). G11 stands: PI is off-manifold with content at mid σ —
  no mid-σ C_π read, and **no GPU top-up to obtain one**. This is a
  correctness restriction, not a budget choice.
- Per-cell 2D planes are **exact** — each cell drawn in
  span(B_cell, C_cell) (bc_plane convention per cell). No PCA-loss
  caveat applies (unlike the r-level quiver figure).

## 21.1 — cell ledger + lattice figure (CPU)

`e21_cells.py`, single pass over both stores:

1. **Cell ledger** (the record; digest `e21_cells.json` in this dir):
   per (store, σ, route, cell): debiased ρ_cell = cos(B⊥, C⊥),
   |B|/|C|_cell, energy shares; at 19.4 bins additionally the π-swing
   Δρ_cell = ρ(B, C_π) − ρ(B, C), h(C_π)/h(C)_cell, and the rotation
   cos(C⊥, C_π⊥)_cell.
2. **Lattice figure** (`e21_lattice_<route>.png`): depth × type grid,
   one arrow pair per gated cell in its exact (B, C) plane, shared
   scale per panel; one panel per verdict σ; a π-overlay panel at
   0.7 / 0.8333 (C and C_π from the same base). Cells below the rel
   gate hollow.
3. **Distributions**: gated ρ_cell histogram/violin per σ against the
   pooled ρ̄ ≈ −0.91 line; Δρ_cell distribution by type band.

## 21.2 — pre-registered readings (free re-reads of 21.1)

| outcome | verdict |
|---|---|
| gated-cell **median ρ_cell ≤ −0.7 at every verdict σ** (IQR ≤ 0.2) | **LOCAL** — the cancellation is pointwise in parameter space; the global constant is locally enforced; the lattice figure may be captioned "the cancellation drawn". Sharpens 19.3, consistent with it. |
| pooled ρ̄ deep but **cell median ≥ −0.5** | **MODAL** — the cancellation is carried by cross-cell global modes; new mechanistic constraint on the scale-covariance account (the flat-diagonal claim must carry a cross-cell structure rider). The figure caption must NOT call the arrows "the cancellation" — it illustrates cells, not the mechanism. |
| in between / σ-dependent | **MIXED** — record per-σ; no downstream claim without a follow-up pre-registration. |

- **π-swing localization** (19.4 bins only): null = uniform positive
  Δρ_cell across type bands (19.4's global chain-rule reading).
  Concentration is claimed only if a pre-named band set — adaln,
  cross-attn — carries ≥ 2× the uniform share of the total swing;
  otherwise the delocalized language stands unchanged.
- **Cross-store 0.7 bin**: cell-ρ agreement (19.3 store vs 19.4 store)
  within the twin-based noise estimate licenses pooling reads across
  stores; disagreement is an **instrument discrepancy flag** — stop
  and diagnose before interpreting any other row.
- **19.3 anti-relitigation guard**: if E21's cells, re-marginalized to
  19.3's own slices, contradict 19.3's uniformity verdict, that is an
  instrument bug until proven otherwise — E21 does not reopen 19.3.

## Kill switches / honesty

- Read-only CPU analysis of committed stores; nothing is refit and no
  constant is tuned. The figure is illustration; verdicts live in
  `e21_cells.json`.
- Per-image cell spread is **not obtainable** (stores are cross-image
  sums) and is not claimed; wording in outputs must say "pooled
  per-cell", never "per-image".
- Endpoint bins detached; C_π restricted per G11 (above).
- Anti-scope: this is NOT an image-space g-field (g-legs live in
  LoRA-gradient space; the image-space rendering exists at r-level
  only — `e19/fig_score_field.py`), and NOT the 20.x amplitude-
  derivation line.

## Cost ladder

| item | cost |
|---|---|
| 21.1 | CPU, ~minutes (memmap reads over ~19+ GB, cell reductions) |
| 21.2 | free (re-read of 21.1) |
| GPU | **none** — mid-σ C_π is G11-forbidden; per-image sums are not stored and not proposed. Any future GPU arm is a new amendment to this file, pre-registered before submission. |
