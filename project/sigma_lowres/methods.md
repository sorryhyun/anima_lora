# sigma_lowres — implementation

What exists in code for the σ-conditional low-res gradient line: the
observability instrumentation (Measurements A/B below) **and, since
2026-07-26, the Phase 1b trainer wiring** (next section) — built opt-in
(`--sigma_lowres`). The E4 grid measured the wiring end-to-end 2026-07-30
(−14.6% wall at fixed steps); the CMMD non-inferiority read is still owed
(`paper_bench/experiments/e4/README.md`).

## Phase 1b trainer wiring (shipped 2026-07-26, opt-in)

The 1024→896 @ σ>0.5 route, wired end-to-end on the train.py path:

- **Sibling cache**: `make preprocess-demote` appends a `demoted_{H}x{W}` key
  *inside* each 1024-tier image's existing native npz (pixel-space LANCZOS
  downscale of the resized PNG to the 896-tier free-fit bucket → VAE
  re-encode — the probe's measured-safe arm). No sibling files, so bucket
  discovery, reconcile, and every `{stem}_*_anima.npz` glob consumer are
  untouched; the key deliberately avoids the `latents_*` prefix (several
  readers grab the first `latents_*` key). Idempotent per-key; emit lives in
  `library/preprocess/latents.py::cache_demoted_latents`. `sigma_demote =  true` in `configs/preprocess.toml` (or env `SIGMA_DEMOTE`; a `"N:D"` string overrides the route) auto-chains the emit after every `preprocess-vae` /
  `preprocess` pass so the sibling keys never go stale (added 2026-07-27).
- **Shared grid derivation**: `library/datasets/buckets.py::demote_bucket_for`
  (+ `SIGMA_DEMOTE_ROUTE = (1024, 896)`) is the single pure function both the
  emit and the trainer fetch call — off-route shapes (native-896 originals)
  return None and always train native.
- **Dataset**: a `SidecarSpec` channel (`base.py::_try_load_demoted_latent`,
  enabled via `enable_sigma_demote`) carries `batch["demoted_latents"]`;
  missing keys degrade that batch to native with a warn-once, never a crash.
  Train datasets only — validation stays native for arm-comparable val loss.
- **Trainer** (`train.py::_maybe_sigma_demote`): σ-first draw via
  `library/runtime/noise.py::draw_flat_sigmas` (split out of
  `get_noisy_model_input_and_timesteps`, bit-exact — pinned by
  `tests/test_sigma_lowres.py`), swap to the demoted latent when **every**
  sample's σ > `--sigma_lowres_threshold` (default 0.5). Exact at
  `train_batch_size=1`; conservative (fewer demotes, never an unsafe one)
  above. The σ marginal is untouched — drawn unconditionally from the same
  density, merely before the noise. Everything downstream (noise, target,
  padding mask, masked loss interpolation, REPA pooling) derives from the
  swapped latent. Compile: demoted token counts are unioned into
  `_derive_token_budget` so the demoted band sits inside the dynamic-seq
  range. Method adapters are refused unless they declare `sigma_demote_safe`
  (REPA does — grid-agnostic adaptive pooling; EasyControl/BYG/soft-tokens
  are refused pending Q5 probes). Density `timestep_sampling` modes
  (scheduler-grid fallback) have no flat draw → warn once, train native.
- **Evidence in the run log**: `sigma_lowres ENABLED` at setup, a first-demote
  INFO with the grid swap + σ, and a demoted-fraction INFO every 500 eligible
  steps.
- **yarnsig rope** (`--sigma_lowres_yarnsig [A,B,C,G]`, added 2026-07-27, bare
  flag = the probe-validated `1,4,0.35,2`): on demoted steps only, RoPE is
  built by `VideoRopePosition3DEmb.generate_embeddings_yarn` — the σ-gated
  YaRN banded alignment that PASSED both pre-registered legs in the SigMa
  probe (`record/yarnsig_report.md` §"SigMa σ-gated YaRN boundaries"). Spatial bands
  with < α·μ(σ) rotations across the demoted extent get the full PI stretch
  to native coordinates, bands above β·μ(σ) keep native integer spacing,
  linear ramp between; μ(σ) = sigmoid(γ·[logit σ − logit σ_c]) with μ from
  the batch-min σ (the sample nearest the gate is where the low-σ liability
  was measured). No second σ-threshold — the gate lives inside the rope
  schedule; native steps and validation are untouched (`train.py` sets
  `anima._sigma_lowres_yarn` for exactly the primary forward's span, cleared
  in a finally). Rope is built outside the compiled block graph
  (`prepare_embedded_sequence`), so no dynamo interaction; not cached (μ is
  continuous per step). Identities pinned in `tests/test_sigma_lowres.py::
  TestYarnsigRope` (μ→0 ⇒ bit-exact native; all-bands-stretch ⇒ uniform PI).
- **ΔW comparison**: `bench/compare_ckpt_dw.py` — paired checkpoint
  displacement cosines in rank space (global + per-block depth profile),
  reproduces the report's tenth4s pair table; paired runs only.
- **`--deterministic`** (train.py, added 2026-07-27): bit-exact
  reproducibility — deterministic flash-attn backward (the one un-seedable
  noise source; `attention_dispatch.set_deterministic`, set before the
  first forward so compile traces it) + `use_deterministic_algorithms
  (warn_only)` + cuDNN determinism + CUBLAS workspace config. Twin runs
  verified bit-identical over a full compiled 1200-step tenth run; ~33%
  slower. With it, paired A/B ΔW cosines have no chaos floor (which twin
  measurement put at 0.413 for nondeterministic tenth runs). Not inherited
  by bespoke loops (turbo/spd/mod) — mirror explicitly.

Smoke-verified 2026-07-26 (satetsu, 8 steps): native grid (152,108) →
(130,92) at σ=0.516, checkpoint saved. Invariant tests:
`tests/test_sigma_lowres.py` (12 — demote grid purity, key-namespace
isolation, σ-draw bit-exactness, emit round-trip/idempotence/loader).

The line's benches are adopted into this home (`bench/`), including
`bench/tier_routing/` (the closed Phase-3a gradient-equivalence probe —
`redundancy.py` still supplies probe-set selection to both measurements below)
and `bench/traj_stats/` (the closed traj_stats line's observability harness,
re-adopted from `_archive/` 2026-07-24: passive recorder bit-exactness bench,
trajectory atlas, gauge + calibration, and `run_reuse_oracle.py` — the
oracle-replay sidecar instrument that is the mandated first step for any new
σ-conditioned intervention, e.g. Phase-1b trainer wiring). The
`blocks_to_swap` × `activation_memory_budget` → peak-VRAM/step-time surface
sweep it briefly hosted moved back to top-level `bench/autotune/` (general
infra, not sigma_lowres-specific).

## Measurement A — latent RAPSD + closed-form crossover

`project/sigma_lowres/bench/rapsd.py` (CPU-scale, no DiT).

- Computes the radially-averaged power spectral density of the probe set's
  cached VAE latents in the DiT's spatial grid, normalized frequency
  r ∈ (0, 0.5] cycles/latent-pixel.
- Under flow-matching noising `(1−σ)x₀ + σε` with unit-variance white noise
  (PSD ≡ 1), the per-frequency signal/noise crossover has the closed form
  **σ_eq(f) = √P(f) / (1 + √P(f))**.
- Outputs: mean P(f) curve, σ_eq(f), predicted σ\*(e) for each demote edge
  (demoted Nyquist = 0.5·e/1024), above-Nyquist SNR A(σ, e), Fig-1-style plot.
- Reusable for any "what does the spectrum predict" question on Anima/Qwen-VAE
  latents.

## Measurement B — per-σ-bin gradient probe

`project/sigma_lowres/bench/run_sigma_probe.py` (~2–2.6 h GPU for 40 images × 6 arms).
The CLI is the whole surface; the internals were split out 2026-07-31 into
`bench/sigma_probe/` — `cli.py` (flags + the cross-flag validation that
derives the run's shape), `kernel.py` (σ grids, the PI/YaRN RoPE patches, the
per-bin gradient estimator), `stats.py` (the pool/arm-sum accumulators and
every cosine/gap/κ reduction). Flags, seeds and outputs are unchanged.

The tier_routing Phase-3a instrument extended with per-σ-bin gradient
accumulators — the estimator class that was *reliable* in 3a (per-bin means
across images, SEM ~0.02), not the per-image ranking that failed there.

- **Arms per image**: native, redraw-floor null (same res, fresh noise draws —
  the "how much do gradients differ anyway" floor), re-encode control
  (decode→re-encode at native res — isolates VAE round-trip cost), and demote
  arms (pixel-space downscale → VAE re-encode → noise; SwD's validated
  "strategy B" ordering — never latent-space downsampling).
- **Binning**: B uniform σ bins × D stratified draws per bin per arm
  (shipped runs: 8 × 8). Uniform bins make the training marginal density
  irrelevant; density-weighting the bins by the trainer's sigmoid σ-density
  reproduces 3a's pooled numbers (consistency check, passed).
- **Per image × bin outputs**: cos_floor, cos_reenc, cos_e, gap_e =
  floor − cos_e, grad norms. Verdicts read off bin-mean curves with a
  mandatory split-half reliability check.
- **CLI**: `--tier <native_edge> --demote_edges <e1,e2,...>` selects the
  operating point (Phase 0 ran 1024→{896,768,512}; Phase 1a ran
  896→{768,512}).
- **Mechanism flags** (record/groundings.md tests G1/G2, added 2026-07-24):
  `--endpoint_bin` appends an exact σ=1.0 bin (input pure ε → any gap = the
  Floor by construction; `--bins 0 --endpoint_bin` = endpoint-only run).
  `--x_zero` zeroes the image in BOTH input and target on every grid
  (captions + demoted shapes kept; implies no reenc arm) — isolates pure
  graph-shape sensitivity. `--per_group` additionally emits per-group gaps
  (15 module types incl. `lora_up` row-splits of the fused qkv/kv
  projections, × 28 blocks; `cosg_floor`/`gapg_*` in rows and headline) —
  bookkeeping over the same flat gradients, zero extra GPU cost. Per-group
  gap_reenc is the per-group validity witness.
- **Daemon-hardened**: `start_heartbeat()` (45 s stderr ticks) keeps the
  120 s daemon stall-watchdog from killing long silent accumulation loops
  (`project_sigma_lowres_phase0` gotcha).

## Operating point / caveats baked into the instrument

- Adapter under probe: `anima_soup_sincos` (trained at native tiers). A
  mixed-res-trained adapter might equalize its own gradients — untested.
- Per-bin cosines use 8 draws → absolute cosines are not comparable across
  bins (floor drops where ‖g‖ is small); **gap subtraction** is the valid
  read, with gap_reenc ≈ 0 as the instrument-validity witness.

## What deliberately does NOT exist

- Any latent-space downscale path (SwD found it inferior; untested here).
- Demote routes other than 1024→896, and demotion under fixed-grid adapters
  (EasyControl / BYG / soft-tokens) — both gated on their own probes (Q1/Q5).
- Sibling cache *files*: the emit went with in-npz keys instead of the
  autoscale-era stem-suffixed sibling files (which needed tier-collapse and
  reconcile special-casing; that pattern stays retired).
