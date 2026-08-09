# E26 — cross-adapter cancellation geometry (the E7 pair)

| | |
|---|---|
| **Status** | **E26.0 SMOKE PRE-REGISTERED 2026-08-09** — this doc frozen before any run; thresholds below set from the committed sincos reference rows only. Full-grid E26 is **gated** on the smoke and gets its own freeze amendment before running. |
| **Question** | Does the B–C cancellation geometry (near-cancellation, deep ρ, negative I, reliable pooled residual direction) exist on LoRA adapters other than the line's operating point (`anima_soup_sincos`)? Sets the stated breadth of the paper's geometry claims (`paper_v2/revision_plan.md` §3/§5) — **no outcome removes them**; a cross-adapter failure is itself a reportable finding (adapter-specificity), not a shelving. |
| **Licensed by** | `paper_v2/revision_plan.md` §5; the preserved E7 adapters (`output/paper/e7/` — verbatim shipped recipe, dim 32/alpha 128 ⇒ parameter space identical to sincos, designed style axis, artists disjoint from sincos); the committed instruments (`run_sigma_probe.py`, `vector_ledger.py`, E24/E25.0 read conventions). |
| **Explicitly NOT licensed** | Any scope-sentence change from the smoke alone (full grid only); any lever work (E25 territory, unchanged); any per-sample read (E22 → E23a gate, unchanged); π arms (G11). E7's probe *runs* are not reusable (no `arm_sums/`, no repromote arm — verified 2026-08-09) and are not read here. |
| **Adapters** | `output/paper/e7/anima_soup_e7_flat.safetensors` · `anima_soup_e7_dirty.safetensors` (s1001–s1003 siblings = pre-declared optional extension tier, slice-vs-checkpoint axis; not run without an amendment). |
| **Runs** | smoke: `bench/results/<ts>-e260-smoke-flat/` · `<ts>-e260-smoke-dirty/` (labels frozen); read output committed here as `e260_smoke.json`. |

## Design: two tiers

- **E26.0 (this pre-registration)**: one condition — **σ = 0.7, route
  1024→768** (the line's best-behaved bin: E22.4 per-sample holds
  there, E25.0 reliability passes everywhere at it; 768 = the
  larger-residual route) — on both E7 adapters. Go/no-go for the grid.
- **E26 full grid (gated)**: e193's grid per adapter (4 in-window bins
  0.3/0.4333/0.5667/0.7 + endpoint, routes {896, 768}) ≈ **5.7 GPU-h
  per adapter** (e193 measured; NB `revision_plan.md`'s earlier
  ~1.3 GPU-h figure was wrong — corrected 2026-08-09). Its verdict
  thresholds (REPLICATES / PARTIAL / FAILS → scope-sentence wording)
  are frozen in an amendment here *before* it runs; any grid trim is
  decided there, not after results.

## E26.0 frozen protocol

Verbatim e193 (`bench/results/20260807-0745-e193-depth-ledger/result.json`
is the arg authority), **deltas only**:

| knob | e193 | E26.0 |
|---|---|---|
| `--adapter` | `output/ckpt/anima_soup_sincos.safetensors` | the E7 checkpoint |
| `--sigma_window` | `0.23333…,0.76667…,4` | `0.6333333333333333,0.7666666666666667,1` (same bin width ⇒ center 0.7, matching e193's bin 4 after the ledger's 4-decimal rounding) |
| `--demote_edges` | `896,768` | `768` |
| `--endpoint_bin` | on | **off** |
| `--label` | `e193-depth-ledger` | `e260-smoke-flat` / `e260-smoke-dirty` |

Everything else matched: `--num_images 40 --probe_list
project/sigma_lowres/paper_bench/experiments/e13/e1b_probe_list.json
--draws_per_bin 12 --repromote --keep_arm_sums --self_floor
--deterministic --seed 42`, tier 1024, fp32 sums, no grad-ckpt, no
compile. **Probe data held fixed across all adapters** (the e1b
40-stem list): the adapter is the only moving part; for the E7
adapters these stems are mostly never-seen-artist data, which E7
showed is cell-invariant within adapter at map level. Cost: e193 =
5.7 GPU-h / 10 conditions ⇒ **~0.57 GPU-h + startup per adapter**;
store ~1.8 GB each. One probe process per adapter (kernel-path rule);
daemon-queued.

### Reference rows (committed, quoted before running)

sincos @ σ = 0.7 / 768, `data_ref = reenc`:

| source | D | draws | S | F | I | ρ | rel_B | rel_C | h(B) | h(C) | h(B+C) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e193 `ledger_depth.json` global | 40 | 12 | 0.070 | 0.119 | −0.163 | −0.890 | 0.879 | 0.852 | 0.0570 | 0.1767 | 0.0440 |
| e221 `ledger.json` | 16 | 24 | 0.060 | 0.116 | −0.148 | −0.891 | 0.838 | 0.801 | 0.0548 | 0.1623 | 0.0498 |

The signature is protocol-robust across D/draws; note h(B+C)/h(C) ≈
0.25–0.31 (the cancellation removes ~70–75 % of the graph term's
angular cost) and h(B+C) < h(B) < h(C).

### E26.0-1 — cancellation replication (the go/no-go read)

Per adapter, from `vector_ledger.py` on its store (bc_ledger row at
σ = 0.7 / 768):

- **Gates** (readability, not verdict): `rel_cos_B ≥ 0.5` and
  `rel_cos_C ≥ 0.5` (the E24 rel gate). Gate failure ⇒
  **INCONCLUSIVE**, with one pre-declared remedy: a single
  `--draws_per_bin 24` top-up rerun (e221's demonstrated regime),
  recorded as an amendment — no other silent rerun.
- **PASS** iff all three: **I < 0**, **ρ ≤ −0.5**, and
  **h(B+C) < min(h(B), h(C))**. (Deliberately generous vs sincos's
  ρ ≈ −0.89: the smoke asks "is the cancellation present," not "is it
  sincos-deep." Depth comparison is the full grid's job.)
- **FAIL** iff gates pass and any of the three is violated.

**Decision rule (frozen):** both adapters PASS → the full-grid freeze
amendment is licensed. Exactly one PASS → full grid licensed on the
passing adapter only; the failing row is reported either way. Both
FAIL → E26 closes; the geometry claims keep the one-operating-point
scope and the failure is reported in the paper's limitation paragraph.

### E26.0-2 — direction reads (descriptive, NO verdict weight)

From the smoke stores + the e193 store's σ = 0.7 / 768 condition,
E24/E25.0 conventions (legs ⊥ against each condition's own ĝ,
cross-set debias):

- `rel_cos_R` per adapter (R = B + C, the E25.0 estimator) — recorded
  against the E25.0 ≥ 0.5 criterion but gating nothing at smoke level;
  it shapes the full-grid pre-registration (whether axis-field reads
  are worth the grid).
- Cross-adapter cosines in the shared parameter space:
  cos(R̂_flat, R̂_dirty), cos(R̂_e7x, R̂_sincos); same for B̂ and Ĉ.
  Stated prior (E19.6's LoRA-moves-B refutation): **B̂ should be
  highly shared**; C and R̂ are the informative objects. Also stated
  up front: these connect to E7's open checkpoint-dependent floor
  *level* (flat ≈ 0.73 vs dirty ≈ 0.50 in-window cos_floor) —
  amplitude-vs-direction localization of that difference is a
  full-grid deliverable, only previewed here.
- G, h, S/F/I rows for both adapters tabulated next to the reference
  rows.

### Validation gates (before any new quantity is read)

1. The read script reproduces the committed e193 global 768/σ = 0.7
   row (`ledger_depth.json`) and the e221 768/σ = 0.7 row
   (`ledger.json`) from their stores to ≤ 1e−6 in ledger units.
2. `vector_ledger.py` runs on the new stores unmodified (any
   instrument change voids the smoke and requires re-freezing).

### Kill switches / honesty

- Thresholds above were set from the two committed reference rows
  only, before any E7-adapter gradient existed anywhere on disk.
- No scope-sentence edit, no revision_plan claim-ladder edit, and no
  full-grid run happens off the smoke without the gated amendment.
- The smoke's σ = 0.7 / 768 choice is favorable by construction (best-
  behaved bin); a smoke PASS therefore licenses *measuring* the grid,
  never *claiming* it.
- Outputs: `e260_smoke.json` (+ any figures) committed in this dir;
  stores stay under the gitignored `bench/results/`.
