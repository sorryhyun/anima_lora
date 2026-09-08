# Foresight Guidance (FSG)

Training-free, checkpoint-agnostic inference stack that reframes CFG as a
fixed-point calibration toward a golden path — the latent state where the
conditional and unconditional velocities agree. At scheduled timesteps it runs
`K` forward(conditional)–backward(unconditional) iterations over a long interval
`Δσ` to pull `x_t → x̂_t` onto that path, then denoises from `x̂_t`.

- Paper: "Towards a Golden Classifier-Free Guidance Path via Foresight Fixed
  Point Iterations" (NeurIPS 2025, arXiv 23177).
- Proposal / groundings: [`../proposal/foresight_guidance.md`](../proposal/foresight_guidance.md)
  (line CLOSED 2026-07-12 — feature stays shipped; see its §8 and Status below).
- Benches (archived 2026-07-12): `_archive/bench/fsg/probe_golden_path.py` (Phase-0 premise),
  `_archive/bench/fsg/render_compare.py` (render A/B + λ/K calibration).
- Plugin: `library/inference/corrections/fsg.py` (`FSGCalibrator`).

## The operator (flow-matching translation)

The paper is ε-prediction + DDIM; Anima is velocity-prediction flow-matching, so
the forward-backward operator maps onto the reversible Euler ODE — no DDIM
machinery. At a scheduled σ with latent `x`, interval `Δσ`, calibration guidance `γ`:

```
v^γ  = v^u + γ·(v^c − v^u)            # CFG-guided velocity
x'   = x  − Δσ · v^γ(x,   σ)          # denoise σ → σ−Δσ (guided)
x''  = x' + Δσ · v^u(x',  σ−Δσ)       # re-noise back (unconditional)
F(x) = x'' ;  iterate x ← F(x), K times ;  then denoise from x̂ = x^(K)
```

A constant velocity field is an exact fixed point (`F(x)=x`), which is the
bit-exact invariant the test pins (`tests/test_fsg_invariant.py`).

## Where it hooks — the calibration seam

FSG occupies a new seam — pre-step latent calibration — that no other plugin
uses. In `generation.py::generate_body`, immediately after computing `t_expand`
and before the per-step hydra/FEI setters, if the step's σ is in-band the loop
calls `fsg.calibrate(...)` on `latents`; the setters then recompute on the
calibrated latent and the real forward proceeds. Threaded via
`SamplerSideChannels` for the future spectrum/spd runners (they ignore it today).

## Anima-specific band — mid-σ only

The paper concentrates iterations in the *noisiest* stages. On Anima that is the
dead zone: at σ≈0.94 there is barely any conditional structure, so cond≈uncond
and iterating amplifies noise (the operator *diverges*, ρ>1). The operator
contracts and the cond/uncond gap shrinks in σ∈[0.45, 0.85] — the full
working band at 20-step Euler. The band moves down with step count and token
tier (proposal §1a): the shipped default [0.59, 0.75] is the 28-step
er_sde / 1024-tier calibration (Plan B) — at 28 steps σ≈0.84 stops contracting.
A narrow band carries ~all the visible win (latent drift near-identical to the
wide band) while firing on far fewer steps. The Phase-0 probe
(`_archive/bench/fsg/probe_golden_path.py`) is the calibration instrument:
calibrate where ρ<0.95, spend more K where the gap-drop is largest; re-probe if
`infer_steps` or the token tier changes.

## Usage

```bash
make test FSG=1                                  # defaults: band [0.59,0.75] (28-step er_sde calibration), K=3, Δσ=0.1, γ=guidance
make test FSG=1 FSG_BAND="0.45,0.85"             # full 20-step working band (~2× the extra NFE)
make test FSG=1 FSG_GAMMA=4 FSG_D_SIGMA=0.1
```

The validated production stack is fsg on the CFG++ substrate (`--cfgpp`,
λ=1.5 default) — see proposal §0; the node's single `fsg` boolean enables exactly
that stack.

Composes into every `test-*` target like `SPECTRUM=1`/`MOD=1`/`DAVE=1`. Direct CLI:

```bash
uv run python inference.py --fsg --fsg_band 0.45 0.85 --fsg_k 3 --fsg_d_sigma 0.1 [--fsg_gamma 4] ...
```

| Flag | Env lever | Default | Meaning |
|------|-----------|---------|---------|
| `--fsg` | `FSG=1` | off | Enable (CFG-only). |
| `--fsg_band LO HI` | `FSG_BAND="lo,hi"` | `0.59 0.75` | σ-band where calibration fires (28-step er_sde/1024 calibration; moves with steps + tier). |
| `--fsg_k` | `FSG_K` | `3` | Fixed-point iterations/step (error ~ρ^K, ρ≈0.93 ⇒ K=3–4). |
| `--fsg_d_sigma` | `FSG_D_SIGMA` | `0.1` | Forward-backward interval Δσ (calibration stride). |
| `--fsg_gamma` | `FSG_GAMMA` | `None` | Calibration guidance γ; `None` reuses `--guidance_scale`. |

## Cost

Extra forwards = `3·K·M`, M = scheduled steps in-band (`v^c`+`v^u` at σ, `v^u`
at σ−Δσ per iteration). At the production point (28 steps, band [0.59,0.75] ⇒
M=5, K=3) that's 56 + 45 ≈ 101 forwards, ~1.8× plain CFG. The matched-NFE
A/B (does FSG-at-N beat a plain longer baseline at the same N?) was never run
— the line closed 2026-07-12 without it (proposal §8), so do not advertise FSG
as free quality; it's the reopening gate if the line is ever revived.

## Composition

CFG-only. Composes with mod-guidance (FSG's forwards inherit AdaLN steering),
DAVE (forwards see the attenuated features), CNS (orthogonal — FSG is
deterministic). With
SMC-CFG the rule is: FSG calibration uses plain γ-combine; the outer step
keeps the configured CFG variant.

Spectrum (`--spectrum`, incl. `SEA=1`): composes. The spectrum runner reads
`ctx.fsg` and treats every FSG-scheduled σ-band step as a third class of
forced-actual step, alongside warmup and the `stop_caching_step` tail —
calibrate the latent first, force an actual forward (you can't calibrate on a
cached step, and re-observing keeps the Chebyshev basis honest across the
calibration kink), and exclude it from the window/SEA decision domain. So the
schedulers never compete: under `SEA=1` the calibrated step resets the SEA
accumulator like any refresh and is dropped from the auto-δ trace, so δ stays a
like-for-like match at matched *adaptive* compute — FSG's `3·K`-per-step cost
lands identically on the window baseline it's calibrated against. The band/K/Δσ/γ
enter the SEA δ cache key so δ recalibrates when they change. Spectrum's own
limitation carries over (it doesn't set the full hydra/FEI routing, so this
targets base/standard checkpoints).

Still ignored under `--spd` (it grows resolution along the trajectory) — a
warning fires; FSG×SPD is a v2 item.

## Status

**Line CLOSED 2026-07-12; the feature stays shipped as-is.** The Tier-2 rigor
pass mostly ran (production er_sde render A/B: fsg/cfg++ beats cfg++ by eyeball;
saturation confound quantified — Δsat +1.7%p, not a tone bump; CFG++-substrate
anti-confound read passed; node port verified live). The one exception is the
matched-NFE A/B, which was never run: in practice FSG's ~1.8× NFE loses to
the 1×-NFE `--xattn_boost` as the everyday quality lever, so the cost proof
wasn't worth running. Consequently FSG must not be advertised as free
quality; the matched-NFE A/B is the gate for reopening the line (tooling in
`_archive/bench/fsg/render_compare.py`). CFG++ (`--cfgpp`) is independent of
this closure and stays shipped/maintained. Full closing note: proposal §8.
