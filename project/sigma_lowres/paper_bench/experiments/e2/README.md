# E2 — target-strength sweep at the endpoint

| | |
|---|---|
| **Status** | **DONE 2026-07-29** |
| **Verdict** | α-flat at the well-conditioned anchors — **no resolvable target-content share**; the endpoint gap is the graph floor, reproduced along a second axis. |
| **Runs** | `runs/20260729-0946-e2-target-alpha/` (N=12, D=16, endpoint-only, routes {768, 512} + reenc, `--self_floor`, wall 1.0 h) |
| **Instrument** | `bench/run_sigma_probe.py --target_alpha` (landed `7e6ed556`) |
| **Depends on** | [E1](../e1/) (c) — after which this became cheap confirmation, predicted slope ≈ 0 |
| **In the paper** | §4.3 "endpoint ≡ x-zero ≡ α-flat"; the §floor `[pending]` marker was replaced with the measured sweep |

`--target_alpha 0,0.25,0.5,0.75,1`: at σ=1, input = ε unchanged, target
= ε − α·x (per-arm x). Decomposes the endpoint gap into graph share
(α=0, ≡ x-zero-in-target-only) and target-content share (slope in α).
Post-E1(c) this was a cheap confirmation run (predicted slope ≈ 0).

**Instrument.** One full pass per α over every arm, draw seeds shared
across α (slope carries no draw noise), all α in ONE process
(kernel-path chaos rule). α=1 keys unsuffixed, others `@a<α>`; envelope
gains per-α aggregates + `alpha_slope_<arm>`.

## Results

- Paired debiased (arm − reenc, |Δ|>1.5 trimmed), endpoint bin:
  768: α0 +0.070±0.015, α0.25 +0.068±0.013, α1 +0.049±0.010;
  512: α0 +0.269±0.041, α0.25 +0.263±0.039, α1 +0.337±0.067;
  reenc envelope slope +0.003. **α-flat at the well-conditioned
  anchors → no resolvable target-content share; predicted slope ≈ 0
  CONFIRMED. The endpoint gap is the graph floor — E1(c)'s x-zero ≡
  endpoint equality reproduced along a second axis.** (Values also
  consistent with E1(a)'s N=40 draw-limit floors.)
- **Mid-α (0.5, 0.75) is unreadable by construction** — a bonus
  mechanistic result, not hidden noise: the native residual α·x − x̂
  passes near cancellation there, `gnorm_native` dips 60→33,
  `cos_floor` falls to 0.87, and the paired SEMs blow up ~7×. Small-G
  amplification (the paper's §input-branch mechanism) surfacing along
  the α axis. Consequence: the envelope's naive `alpha_slope_*` over
  all five α (+0.034 / +0.180) is inflated by these points — the
  verdict object is the anchor contrast above.

## Ops gotcha

`make daemon-run` consumed `--label` from ARGS as the *job* label
(dispatcher behavior at the time), so the script never saw it and the run
dir was created label-less — renamed by hand afterward. **Fixed
2026-08-01** (flags now scoped to the prefix before the script path); the
E14 launch hit the same trap and prompted the fix.
