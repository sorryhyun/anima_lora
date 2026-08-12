# E25b — explicit resolution conditioning (CLOSED 2026-08-12)

| | |
|---|---|
| **Status** | **CLOSED 2026-08-12.** 25b-1 **IMPROVES** (gradient level), 25b-2 **FAIL** (render quality), 25b-3 **NULL** (rescue). No ship; `--sigma_lowres_res_cond` stays experimental opt-in; paper-2 material (gradient-level positive + render-level negative together). |
| **Motivation** | The demoted-step residual is angle-borne (E24.2), locally enforced (E21), amplitude-concentrated in the adaln band (E21), and a conditioning input demonstrably organizes the axis field (E28-F1). E25b tested the lever form: make the resolution the router already knows an **explicit conditioning input** on the adaln trunk (SDXL size-conditioning precedent) — does the demoted-step cancellation geometry improve, and does the trained model then render inside the native yardstick? |
| **Verdict in one line** | The conditioning absorbs the compute-graph substitution per step, but training converges to a **different model**, not a closer recovery of native. |

Full frozen registration (verdict branch tables, the E20.4-adjacency
argument, kill switches, launch records) lives in this file's git
history: registration 2026-08-11, Stage 0 as-built `96d2eb10`, Stage 2
result `df531c1f`.

## The lever (as built)

- **Input**: s = log2(step_edge / native_edge) of the grid the step
  actually trains on (0 native, −0.193 @896, −0.415 @768) — the
  router's known decision, no estimation. Sinusoidal embedding (dim
  256) → zero-init `Linear(256, 2048, bias=False)`, added at the
  pooled-text-delta seam before the adaln trunk. ~0.5 MB trainable.
- **Nothing gap-related enters the objective**: loss, target, σ-draw,
  and router are bit-unchanged; the lever is a forward-pass input
  trained by ordinary backprop (the E20.4 known-input rule held —
  any gap shrinkage is emergent, not optimized-for).
- Checkpoint key `sigma_lowres_res_cond_proj` + `ss_sigma_lowres_res_cond`
  stamp; `make merge` refuses it (not a ΔW); inference attaches
  `(proj, s=0)` via `_attach_res_cond` (`library/inference/models.py`);
  probe `--res_cond` applies per-arm s. Invariants pinned in
  `tests/test_sigma_lowres.py::TestResCond` (zero-init identity,
  stamp-without-key hard fail, s=0 delta bit-exact to trainer's).

## Stage 1 — gradient geometry, tenth scale (25b-1 IMPROVES)

Deterministic seed-twins ctrl (combo) vs rescond (combo + flag),
480 steps hews, + one 768 5-bin probe each, one boot. All four gates
passed (twin-start bit-identity, independent ledger reproduction, 5/5
bins readable in both stores, same-family). Read: `e25b_read.{py,json}`.

| σ | r_h = h(B+C) rescond/ctrl | Δρ |
|---|---|---|
| 0.3 | **0.305** | +0.005 |
| 0.4333 | **0.333** | +0.005 |
| 0.5667 | **0.572** | +0.011 |
| 0.7 | **0.645** | −0.006 |
| 0.8333 | 0.943 | −0.004 |

Median r_h **0.572** (bar ≤ 0.9), median Δρ **+0.005** (bar ≤ +0.02) ⇒
**IMPROVES**. Shallow-σ concentrated (helps most where the demote gap
is largest); the **graph leg h(C) shrinks at every bin** (0.32–0.66)
while h(B) is mixed — the conditioning absorbs precisely the
substitution it announces. Cost ≈ 7.7 GPU-h.

## Stage 2 — full-scale ship read (25b-2 FAIL, 25b-3 NULL)

E16.1 protocol: 2 corpora × 3 seeds × 5 arms (native / combo /
rescond / sigma768 / rescond768) + determinism duplicate = 31 runs;
renders + in-batch yardstick in one boot. All gates passed (12/12
flag-pair first-loss identity; determinism dup 0/1092 keys; 6/6
demote-mass pair-identity; yardstick within ±0.02 of recorded).
Read: `e25b_ship_read.{py,json}`; launch `e25b_stage2_launch.py` +
`e25b_stage2_jobs.json`.

Mean within-seed render cos vs native twin (in-batch yardstick
hews **0.9578** / channel **0.9541**):

| arm | hews | channel | vs bar |
|---|---|---|---|
| combo | 0.9575 | 0.9650 | boundary hews, inside channel |
| **rescond** | **0.9450** | **0.9515** | **below BOTH** (−0.0128 / −0.0026) |
| sigma768 | 0.9532 | 0.9674 | below hews, inside channel (E4 shape) |
| rescond768 | 0.9575 | 0.9484 | boundary hews, below channel |

- **25b-2 quality gate: FAIL** (near-miss = miss as frozen).
  Throughput measured, not gated: paired wall rescond/combo **1.0024**
  — the projection is free; cost was never the problem.
- **25b-3 rescue: NULL** — paired Δcos(rescond768 − sigma768) hews
  median +0.0057 (2/3), channel **−0.0206** (0/3); band not rescued.
- **ΔW secondary (corroborate-only): opposite direction 12/12** —
  rescond 0.407/0.465 vs combo 0.744/0.770 toward the native twin.
- CMMD incoherent at this N (the E4 lesson reproduces; read json).
- **Recorded observation**: realized σ>0.5 mass 37.5 % vs the
  E16.1-era 50.8 % at the same seeds — the σ-draw distribution shifted
  between 08-02 and 08-12. In-batch reads unaffected (CRN holds);
  cross-batch scalar comparisons are dead, again.

Cost ≈ 5–6 GPU-h.

## Reading

One consistent story across the three reads: explicit conditioning
**does** absorb the demoted-step residual per step (25b-1, gradient
level), but — since nothing ever pushes the endpoint toward the native
run — training converges to a **different model**: ΔW far from the
native twin, renders outside the native seed lottery on hews. The
lever changes the solution; it does not recover the native one.
Legitimate paper-2 finding; **NO** as a quality-neutral throughput
recipe. E25c (inference-time knob on the trained axis) reads the
rescond checkpoints independently and is unaffected by this FAIL.

## Post-close descriptive (2026-08-12) — the trained lever is ~90 % common-mode

The registered Stage-2 descriptive ‖W·φ(0)‖ (the trained native
offset) was never emitted by `e25b_ship_read.py`; first measured
post-close (`e25b_native_offset.{py,json}`, CPU, all 12 carrier
checkpoints). Decomposition of the trained delta:

| component | norm |
|---|---|
| ‖W·φ(0)‖ — the offset **every** step (native included) receives | 0.130–0.192 |
| ‖W·φ(s) − W·φ(0)‖ — the part that actually distinguishes routes | 0.005–0.024 |
| cos(W·φ(0), W·φ(s)) | 0.994–1.000 |

The lever trained as ~90 % **resolution-independent global bias** on
the adaln trunk and only ~6–13 % resolution label. SGD used the
projection as a free bias parameter outside ΔW; the common-mode
component shifted the native operating point, the LoRA co-adapted
around it, and the endpoint diverged (the ΔW ANTI-direction and the
below-yardstick renders above). This sharpens the mechanistic reading:
the divergence channel is the **parameterization's common-mode
freedom**, not the conditioning idea itself — measured motivation for
**E25e** (re-centered delta W·(φ(s) − φ(0)), which removes the channel
structurally).
