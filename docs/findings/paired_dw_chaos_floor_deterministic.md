# Paired ΔW cosines ride a ~0.41 chaos floor — flash-attn backward is the one un-seedable RNG, and `--deterministic` removes it

> **STATUS (2026-07-27).** TRAP (measurement methodology) + the fix LANDED the
> same day: `--deterministic` (train.py) produces **bit-identical checkpoints**
> across runs, twin-validated over full compiled 1200-step runs. Memory:
> [[project_deterministic_flag_chaos_floor]]. Data:
> `project/sigma_lowres/record/report.md` §"Twin controls".

## The trap

The sigma_lowres Phase-1b in-vivo A/B compared paired training arms
(`--seed 42 --paired_step_rng`, CRN: identical init, data order, σ sequence,
noise) by ΔW cosine (`project/sigma_lowres/bench/compare_ckpt_dw.py`, rank-space
`(UaᵀUb)⊙(DaDbᵀ)`). CRN lockstep was *witnessed* — arms bit-exact at step 2,
tracking to 3–4 decimals through step 10 — so it was tempting to read the
endpoint cosines as treatment magnitude: base↔σ>0.5 = 0.320 "big effect",
base↔σ>0.75 = 0.395 "smaller effect", sigma↔yarnsig = 0.402 "real rope
footprint, well-placed".

**The missing control: the same command run twice.**
`anima_lora_tenth4s_base` vs an identical re-run (`_base_twin`, same seed, same
code, same box) reads:

| pair | cos(ΔW) | treatment difference |
|---|---|---|
| base ↔ base_twin | **0.413** | **NONE** |

With zero treatment, the twin pair lands *above* every treated pair — and its
per-block depth profile (late blocks 0.6–0.83, early/mid 3–11 at 0.14–0.24) is
the same shape the treated pairs show. Consequences for the original table:

- base↔σ>0.75 (0.395) and sigma↔yarnsig (0.402) are **at the floor** — those
  "effects" were unresolved noise. The yarnsig rope footprint measurement was
  vacuous.
- base↔σ>0.5 (0.320), sigma↔896only (0.245), base↔896only (0.184) are
  **below** the floor — real displacement, ordering intact, but magnitudes
  compressed toward the floor rather than toward 1.

The mechanism: CRN seeds every *drawn* random variable, but flash-attention's
backward accumulates dK/dV with **atomic adds** — the floating-point reduction
order varies run-to-run and is not seedable from anywhere. That injects
~1e-4/step wobble which training chaos amplifies multiplicatively; by step 1200
two "identical" runs have decorrelated to cos ≈ 0.41. Everything else in the
LoRA path (GEMMs, norms, elementwise optimizer) is already deterministic;
flash backward is the single leak.

**The rule: never read absolute paired ΔW cosines without either a twin
control (measures the floor) or determinism (removes it).** Orderings across
pairs sharing the floor remain valid; absolute magnitudes and any pair within
~0.05 of the floor do not.

## The fix — `--deterministic`, validated to bit-exactness

`train.py --deterministic` =

- `flash_attn_func(..., deterministic=True)` — the actual fix; threaded
  through `networks/attention_dispatch.py::set_deterministic` as a module
  global read at trace time (set before the first forward, so compiled graphs
  bake it in);
- `torch.use_deterministic_algorithms(True, warn_only=True)` +
  `cudnn.deterministic` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` — belt and
  braces for the rest.

Validation (`tenth4s_det_a` / `_det_b`, identical commands, full tenth×4ep
compiled runs): **0/1092 tensors differ, max abs diff exactly 0.0.** Not
"cosine ≈ 1" — bit-identical files modulo metadata.

Cost: ~33% throughput on tenth (1.23–1.30 vs 1.95 it/s) — the deterministic
flash backward trades atomics for extra recompute/semaphores. Same
GPU/driver/library assumed (determinism is per-environment, not portable).

Scope caveat: train.py path only. Bespoke loops (turbo / spd / mod-distill /
RSD) do **not** inherit it — the standing mirroring rule
([[project_daemon_wiring_pattern]]) applies before running a paired A/B there.

## What this buys — and what it doesn't

Under `--deterministic`, paired arms have **no floor**: any cosine below 1.0
is pure treatment. The deterministic three-arm re-run (report §"Deterministic
three-arm table") resolved the previously-vacuous yarnsig read: rope-vs-plain
= 0.396 with zero noise — a real footprint.

But the same run exposed the second half of the trap: **determinism buys
attribution, not magnitude.** Chaos is intrinsic to the training dynamics —
deterministic kernels make identical commands bit-exact, yet a real treatment
difference is amplified by the same chaotic divergence as noise was, landing
within 0.02 of the nondeterministic numbers everywhere (0.305 vs 0.320, 0.396
vs 0.402). Noise and treatment do not add in cosine; both saturate the same
low-signal subspace (~0.4 whether the perturbation is 1e-4 hardware wobble, a
rope schedule tweak, or noise+treatment together).

So the corrected statement of what endpoint ΔW cosine measures: it is a
**detector with depth localization** — "did the trajectories separate, and in
which blocks" — not a ruler for treatment size. Rankings of treatment
magnitude need short-horizon instruments (single-step gradient probes, where
chaos has no time to act) or functional endpoints (CMMD, renders).

Beyond sigma_lowres: any future paired comparison in this repo (CDM arms,
turbo warm-start geometry, soup ingredient dispersion, tag-dropout variants)
has been silently subject to the same floor whenever it read checkpoint
deltas. Twin-or-deterministic is now the entry bar for that measurement class.
