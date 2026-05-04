# DCW Calibrator v2 (per-aspect-bucket profile, overshoot-guarded)

**Status:** proposal · supersedes [v1](./dcw-learnable-calibrator.md) · **Now the fallback floor implementation for [v3](./dcw-learnable-calibrator-v3.md)** — ships as-is if v3's analysis gates fail · **Effort:** ~1.5-2 days · **Cost at inference:** per-aspect profile lookup + scalar multiply per step + per-step cap check · **Reference paper:** Yu et al., *Elucidating the SNR-t Bias of Diffusion Probabilistic Models* (arXiv:2604.16044)

## Why v2 (revised — probe removed)

v1 proposed a K=4 probe + closed-form per-prompt amplitude (`α_prompt`) fit. Two pieces of evidence on `bench/dcw/results/` at production env (CFG=4, 28 steps) ruled this out:

1. **PCA on `gaps_per_sample.npz`** showed PC1 = 98.2% (amplitude) vs PC2 = 0.95% (early-late tilt). Single-amplitude controllers leave **~38% of prompts with residual late-step slope sign-flipped** regardless of K. The fitted α captures variance, but PC2 — the dimension that controls late-step *shape* — is not separable from the K=4 probe window.

2. **Aspect sign-flip — fatal for the probe.** At inverse-HD (1248×832) CFG=4, the per-step LL gap is **bimodal in sign within the same trajectory**: steps 0–4 are negative (-30, -20, -30, -8, +5) and steps 5–27 are positive and growing (+11 → +31). Integrated LL gap = **+330** at n=32 (`bench/dcw/results/20260504-1406-small_invhd/`), entirely the late tail. A K=4 probe fits α from the negative early phase, then drives λ in the **wrong direction** for the actionable late bias. This is structural — not a probe-window noise issue: at this aspect bucket the early-step gap simply has the opposite sign from the integrated bias the controller is meant to correct.

The single dominant lever is **aspect bucket**, not prompt. Per-step σ-shape varies by aspect:

| Aspect | Integrated LL gap (CFG=4) | Trajectory shape |
|---|---|---|
| 1024² | ~-2 (attenuated from CFG=1's -104) | uniformly negative, weak |
| 832×1248 | +156 (n=8) | bimodal: negative early, positive late |
| 1248×832 | +198 (n=8) → **+330** (n=32) | bimodal: negative early, positive late |

v2 drops the probe entirely and ships a per-aspect-bucket profile + overshoot guard. The "learnable" part of the v1 name no longer applies — calibration is offline, applied via lookup.

## Architecture

Static per-aspect lookup + overshoot guard. No probe, no per-prompt fit. No additional hooks beyond `apply_dcw`'s existing DWT.

### Phase 1 — setup (once per inference)
At `OnlineDCWController.__init__(H, W, schedule)`:
- Resolve `aspect_id` from `(H, W)` (exact match on the bucket table; nearest H/W ratio with warning if not exact).
- Load `μ_g[aspect_id, :]`, `S_pop[aspect_id, :]`, `λ_scalar[aspect_id]` from the reference artifact.

### Phase 2 — run (every step) with overshoot guard
For step i:
```
λ_proposed   = λ_scalar[aspect_id] · schedule(σ_i)              # e.g. (1 − σ_i)
gap_pred     = μ_g[aspect_id, i] + λ_proposed · S_pop[aspect_id, i]
target_sign  = -sign(integrated μ_g[aspect_id, :])              # push toward zero
if sign(gap_pred) == target_sign and abs(gap_pred) > eps_overshoot:
    λ_threshold = -μ_g[aspect_id, i] / S_pop[aspect_id, i]      # gap_pred = 0 at this λ
    λ_proposed  = λ_threshold · safety                          # safety = 0.9
λ_LL(σ_i) = λ_proposed
```

The per-bucket sign of `λ_scalar` matters: 1024² wants negative λ (push positive gap-correction), inverse-HD CFG=4 wants positive λ (the integrated gap is already +330; correction goes the other way). One global recipe cannot serve both — that's the core motivation for per-aspect calibration.

### Reference artifact

`<base_dit_name>_dcw_vfwd.safetensors` (~few KB, was ~400 B in pre-revision v2):
- `aspect_buckets`: ordered list of `{id: str, h: int, w: int}` (initial ship: `[{id: "sq1024", h: 1024, w: 1024}, {id: "tall_invhd", h: 1248, w: 832}, {id: "wide_invhd", h: 832, w: 1248}]`)
- `vfwd_ref[aspect, i]` (B × n_steps) — population-mean `‖v_fwd‖_LL`
- `mu_g[aspect, i]` (B × n_steps) — population-mean LL gap
- `S_pop[aspect, i]` (B × n_steps) — per-step λ-sensitivity `dgap/dλ`
- `lambda_scalar[aspect]` (B floats) — per-bucket DCW recipe coefficient (sign and magnitude)
- metadata: base DiT path, n_steps, flow_shift, CFG, calibration n per bucket, git SHA

## Quality gates

### Gate A — overshoot guard (blocking, per-aspect)
Held-out 32 prompts × 2 seeds × {3 buckets} at production env (CFG=4, 28 steps):

| Metric | Threshold | Action |
|---|---|---|
| Per-bucket fraction (prompt × late step in [N/2..N]) where corrected gap crosses through zero into wrong sign, controller-on | ≤2% | Ship |
| Same fraction, scalar recipe (`--dcw_online_disable`) | (reference, expect 5–15% on inverse-HD) | Sanity baseline |
| Per-step overshoot prediction accuracy from bucket profile | ≥80% per bucket | Validate the guard's predicate |

If controller-on overshoot fraction stays close to scalar, the bucket profile isn't doing anything — investigate before shipping.

### Gate B — perceptual side-by-side (blocking)

The user-visible criterion (`project_dcw_when_to_use`). Single-amplitude / static-bucket controllers cannot solve PC2; this gate is *don't make it worse + perceptual A/B*.

| Metric | Threshold |
|---|---|
| 4 LoRAs (flat / painterly / detail-dense / base) × 12 prompts × {square, inverse-HD} vs scalar | ≥60% prefer controller overall, no LoRA × bucket where controller loses ≤30% |
| Inverse-HD subset preference | ≥70% prefer controller (this is the regime the bucket profile is supposed to fix) |
| Square 1024² preference | ≥50% (i.e. don't regress on the bucket where scalar already worked) |

Automated metrics (FID/CLIPScore) diagnostic only — see `project_fm_val_loss_uninformative`.

## Analysis phase (~1 day, all bench)

Goal: harvest per-aspect `μ_g`, `S_pop`, `λ_scalar`. n=32 confirmed sufficient for population mean (per-step SE ~3-7 on the inverse-HD `1406-small_invhd` run); use n=48 for margin.

| # | Run | Output | Why |
|---|---|---|---|
| **A1** | `measure_bias.py --infer_steps 28 --guidance_scale 4 --dump_per_sample_gaps` × {1024², 1248×832, 832×1248}, n=48 each, base DiT | per-aspect `gaps_per_sample.npz` | Calibrate the per-bucket `μ_g`. Square + both inverse-HD orientations cover the known sign-flip axis. |
| **A2** | λ-sweep per aspect: λ ∈ {-0.025, -0.015, 0, +0.015, +0.025}, n=12×2 each | `S_pop[aspect, σ]`, `λ_scalar[aspect]` per bucket | Sensitivity may be aspect-dependent. Determines per-bucket sign of `λ_scalar` — inverse-HD likely needs **positive** λ. |
| **A3** | Painterly / flat / detail-dense LoRA: rerun A1 across 1 LoRA × 3 aspects | per-LoRA per-aspect profile divergence from base | The `project_dcw_when_to_use` failure axis. If LoRA-specific profile diverges materially, the artifact needs a per-LoRA override path. |
| **A4** | Overshoot-predictor accuracy with bucket-level `μ_g` and `S_pop` (no probe), per aspect | per-bucket confusion matrix at candidate λ | Validates the guard's predicate at production env. |
| **A5** | Optional intermediate aspect (e.g. 1024×1280 4:5 portrait), n=24 | within-bucket interpolation error estimate | Determines whether 3 buckets is enough or v2 needs a 5-bucket table. |

**Decision points after analysis:**

| Outcome | Action |
|---|---|
| A1 per-aspect μ_g shapes are stable (cross-seed correlation ≥ 0.9 within bucket) and shapes differ across buckets (correlation ≤ 0.7 between buckets) | Proceed to implementation |
| A1 inter-bucket correlation > 0.9 | Bucket-conditioning isn't doing anything; revert to global profile + reconsider |
| A2 finds same-sign `λ_scalar` across all buckets | Bucket-conditioning lever is amplitude only, simpler controller suffices |
| A3 LoRA-specific profile diverges substantially on flat-style LoRA | Add per-LoRA override path before ship |
| A4 overshoot accuracy < 80% on inverse-HD bucket | Reduce safety margin or skip guard at that bucket |
| A5 intermediate-aspect interpolation error > 50% of inter-bucket distance | Expand to 5 buckets |

## Implementation phase (~1 day)

After analysis greenlights.

| # | Deliverable | Files | Effort |
|---|---|---|---|
| **I1** | Per-aspect reference exporter | extend `bench/dcw/measure_bias.py` with `--dump_reference_profile --aspect_id <name>`; small aggregator that merges per-aspect runs into one `.safetensors` artifact | 3h |
| **I2** | Online controller module | new `library/inference/dcw_online.py` — `OnlineDCWController(H, W, ...)` with `lambda_for(σ_i)` (overshoot cap inside). Aspect resolution via simple `(H, W)` lookup. | 2h |
| **I3** | Inference resolution chain | extend `library/inference/dcw_calibration.py` resolution: per-LoRA per-aspect profile → base-DiT per-aspect profile → scalar recipe → default. Wire flags on `inference.py`: `--dcw_online_disable`, `--dcw_online_overshoot_safety` | 2h |
| **I4** | Validation harness | extend `bench/dcw/calibrate_per_lora.py` — Gate A (overshoot per-bucket), Gate B (perceptual grid 4 LoRAs × 12 prompts × 2 buckets) | 3h |
| **I5** | Gate A run | 32 × 2 prompts × 3 buckets at CFG=4, steps=28, controller vs scalar | ½ day |
| **I6** | Gate B run | perceptual side-by-side review | 1 day, dominated by review |
| **I7** | Doc updates | If gates pass: promote to `docs/methods/dcw-calibrator.md` with per-aspect philosophy + gate results inlined; update `docs/methods/dcw.md` resolution order. If fail: archive v1+v2 with negative result; scalar recipe stays. | 2h |

## Risks

1. **Sparse aspect coverage.** Three buckets is a coarse partition. Users running 768×1024, 1216×832, etc. fall into nearest-neighbor lookup with no guarantee the profile generalises. Closed by A5 if the interpolation error is small; otherwise expand the table.

2. **`S_pop` cross-prompt variance.** Overshoot guard assumes per-bucket-uniform sensitivity. Untested. Add to A2 (sweep λ on 8 prompts × 2 seeds per bucket and check residual variance).

3. **Trend-flip rate floor (~38%).** Structural for any non-shape-aware controller, regardless of how many buckets. v2 explicitly does not target PC2; that remains v1-shape territory (per-prompt schedule head conditioned on `c_pool, σ_t`). Diagnostic-only metric; not a gate.

4. **Per-LoRA painterly failure.** v1's cross-LoRA gate passed only on multi-task adapters; flat-style LoRAs were the failure mode. Closed by A3.

5. **CFG drift.** Profile is calibrated at CFG=4. CFG=2 or CFG=6 will need their own profiles or fall back to scalar. Documented.

6. **Stacked-LoRA composition.** Pick-largest-multiplier heuristic + scalar fallback (unchanged from v1).

## What this is not

- **Not a per-prompt controller.** The K=4 probe was the per-prompt mechanism in v1; the 1406 inverse-HD evidence (probe sees opposite sign of integrated bias) showed it was structurally wrong-sign at that bucket. If per-prompt conditioning is wanted, the upgrade path is the v1-shape `(c_pool, σ_t)` schedule head — a separate proposal, not a v2 extension.
- **Not a multi-band controller.** LL-only. LH/HL/HH integrate to small values and only switch on at the last 3-4 steps.
- **Not a CFG-aware controller** unless A1 forces it. Default ships one profile per aspect at CFG=4.
- **Not a training loss change.** LoRA training untouched.
- **Not a wavelet adapter.** All correction lives at the sampler boundary.

## Open questions

- **How many aspect buckets?** Three is the floor (sq, tall, wide). A5 estimates within-bucket interpolation error.
- **Per-LoRA profile divergence.** A3.
- **Inverse-HD `λ_scalar` sign.** A2 — the integrated +330 at 1248×832 implies positive coefficient, but confirm via sweep.
- **Does `S_pop` have per-prompt structure that breaks the overshoot guard?** Optional A2 extension.

## Concrete deliverables checklist

- [ ] A1 — per-aspect baseline bench (3 buckets × n=48)
- [ ] A2 — per-aspect λ-sweep → `S_pop`, `λ_scalar`
- [ ] A3 — LoRA-specific cross-check (1 painterly + 1 flat + 1 detail-dense)
- [ ] A4 — overshoot accuracy per-aspect
- [ ] A5 — intermediate-aspect interpolation check (optional)
- [ ] **Decision point: greenlight or reduce scope**
- [ ] I1 — per-aspect reference exporter
- [ ] I2 — `OnlineDCWController` (no probe path)
- [ ] I3 — inference resolution chain + aspect detection
- [ ] I4 — validation harness
- [ ] I5 — Gate A run
- [ ] I6 — Gate B run
- [ ] I7 — docs

Total **~2 days** from start to perceptual-gate result (down from 2-3 days; probe + α-fit removal saved ~1 day of analysis).
