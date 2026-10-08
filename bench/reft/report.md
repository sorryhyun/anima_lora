# ReFT Phase-1' report — ReFT stays retired; register synergy null

**Run:** 2026-07-05 · **Verdict: `REFT IS REDUNDANT` (stays retired), register+ReFT synergy NULL (adoption-blocked upstream).**
Design gates pre-registered in `plan.md`; this report is the execution record.
Runs: `results/20260705-2205-reft_phase1p/` (CFG-1 convention) +
`results/20260705-2242-reft_phase1p_cfg4/` (real-settings re-render).

## What was run (and how it deviates from plan.md)

The original matched-budget Phase-1 was never executed. Phase-1' is the
pragmatic cut, with two deliberate deviations:

1. **Baseline reuse, not matched params.** Reference =
   `output/ckpt/bench_sincos_half_plain.safetensors` (rank-16 plain LoRA,
   `sincos/* @ sample_ratio 0.5`, 4 epochs, lr 5e-5 — recipe pinned from its
   `.snapshot.toml` because the shipped `lora.toml` has since drifted to
   2e-5/8ep). ReFT at sane dims (0.5–2.1M params) cannot match a rank-16
   LoRA's budget, so the question becomes: **does ReFT at its natural size
   land anywhere near the LoRA reference, and do registers+ReFT show synergy
   over either alone?** No matched-budget claim is made.
2. **A register-synergy axis** (not in the original plan). The DiT port of
   ReFT lost LoReFT's position selectivity (the affine map is
   content-conditioned but position-agnostic — the same regime as any weight
   edit); register tokens are the only fixed-identity positions in the
   sequence, so ReFT-at-register-positions is the one theoretically coherent
   position-selective revision: it parameterizes register *content* per-block
   (DSR Theory-2). NB it deliberately does NOT touch the adoption question
   (attention routing to registers), which belongs to the headroom Phase-2
   unfrozen-QKV sweep (`_archive/proposals/headroom_register_tokens.md`).

### Arms (all trained by `run_bench.py` via train.py, baseline recipe)

| arm | trainables | isolates |
|---|---|---|
| **L** *(pre-existing)* | rank-16 plain LoRA | the workhorse reference |
| **reft64** | ReFT d=64, last_8, all tokens (~2.1M) | shipped-era ReFT semantics (`impl/reft.toml`) |
| **reg36** | K=36 learnable registers @ block 8 (~74k) | registers-only control |
| **reg36_reft64** | reg36 + ReFT d=64 at register positions, blocks 20–26 | the synergy arm (reft64's footprint minus the final block — its register edit is stripped before unpatchify and provably gets zero grad, verified on the smoke run) |

Wiring: bench-local `--network_module bench.reft.reft_network` (zero
live-tree restore; registers ride the shared
`networks/register_injection.py`). Eval: `eval_cmmd.py` — paired
holdout-CMMD à la `bench/memorization/generalize.py` (24 sincos holdout
prompts, identical per-item seeds/sizes for every model, eager, PE-Core MMD²
vs holdout+member pools, real-vs-real noise floor) **plus mandatory per-arm
montages** (the headroom arm-A lesson: pixel metrics are blind to framing
collapse). `compare_sheet.py` builds paired-seed side-by-side sheets
(`compare_0.png` in each run dir).

Throughput datum: 1.89 it/s compiled on the 16GB card → ~6 min per 4-epoch
arm; the whole sweep incl. eval is under an hour.

## Results

### CFG-1 pass (trainer's CMMD convention: 20 steps / CFG 1.0)

Noise floor **0.155**:

| model | cmmd_holdout ↓ | cmmd_member | Δ(m−h) |
|---|---|---|---|
| base | 0.608 | 0.737 | +0.13 |
| L (plain r16) | 0.944 | 1.083 | +0.14 |
| reft64 | 0.889 | 1.044 | +0.15 |
| reg36 | **0.577** | 0.681 | +0.10 |
| reg36_reft64 | 0.609 | 0.751 | +0.14 |

Montage gate: PASSED for every arm (no framing collapse). Visual read
(paired seeds): the plain LoRA visibly restyles; **reft64 is near-identical
to base** with a mild global tone/softness shift — the predicted
global-tone-lever profile confirmed by eye; both register arms are base-like.

Anomaly: base beats both trained adapters — resolved by the CFG-4 pass below.

### CFG-4 re-render (real settings: 28 steps / CFG 4.0, same prompts/seeds)

| model | cmmd_holdout ↓ | (CFG-1 was) |
|---|---|---|
| L (plain r16) | **0.182** | 0.944 |
| reg36_reft64 | 0.223 | 0.609 |
| reft64 | 0.235 | 0.889 |
| base | 0.279 | 0.608 |
| reg36 | 0.364 | 0.577 |

All pairwise deltas sit under the 0.155 floor — treat the ordering as
directional, corroborated by the paired-seed comparison sheets.

## Verdicts

1. **ReFT stays retired — no niche.** At real settings reft64 recovers
   roughly half of the base→artist CMMD gap the LoRA closes, at 1/10 the
   params, but stays visually near-base (mild global tone-work; the LoRA
   does real style-work — restyled shading/linework, re-planned
   compositions). It neither wins nor ties L, so the plan's
   `EARNS A NICHE` gate stays shut. The pre-bench structural prediction
   ("uniform residual edits collapse to a global-tone lever on Anima") is
   confirmed by direct observation.
2. **Register+ReFT synergy: NULL.** reg36_reft64 ≈ reft64 (Δ 0.012, deep
   noise) — routing the same ReFT capacity through 36 register tokens adds
   nothing over editing patch tokens directly — and reg36 alone lands
   *worse than base* under guidance. Root cause matches the headroom RQ3
   negative: registers on the frozen base attract almost no attention mass
   (eval reg_ratio 2.5–3.6× vs the ~14–24× a real sink runs; caveat: the
   telemetry snapshot is taken at the final low-σ step, where the sink is
   weakest), so ReFT-parameterized register content has no carrier.
   **The synergy question defers entirely to the headroom Phase-2
   unfrozen-QKV sweep** — do not re-bench ReFT-on-registers before that
   passes its adoption gate. If it ever does, the arm is pre-built
   (`reft_network.py`, `reft_positions=registers`).
3. **Repo lesson (bigger than the bench): CFG-1.0/20-step CMMD misranks
   across adapter families.** The convention ranked bare base above a LoRA
   that is unambiguously best at real settings. It remains valid for its
   in-training job (ranking checkpoints of the SAME adapter); any
   cross-family comparison must render at 28 steps / CFG 4.
4. **Structural finding:** in `reft_positions=registers` mode the final
   block's edit gets exactly zero gradient (output stripped before
   unpatchify) — verified on the smoke run (block 27 weights bit-zero after
   1336 steps while blocks 20–26 trained); the bench module auto-drops it.

## Reproduction

```bash
uv run python bench/reft/run_bench.py                 # 3 arms + CFG-1 eval
uv run python bench/reft/eval_cmmd.py --adapters ... --steps 28 --cfg 4.0
uv run python bench/reft/compare_sheet.py results/<run>/ --height 800
```

