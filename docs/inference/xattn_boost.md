# Front-loaded cross-attn boost (`--xattn_boost`)

Training-free weak-tag adherence lever: scale every block's cross-attn
residual by λ on the conditional forward only, gated to σ ≥ band
(default 0.85) — the plan-writing window where cross-attn text drive exists at
all (peaks at σ = 1, ~0.02 floor below σ ≈ 0.85 —
`docs/findings/crossattn_self_attn_dominance.md`). Amplify the text voice while
the plan is being written and the plan changes; self-attn + MLP then render the
corrected plan in the normal style. Since Phase-1'' the boost ships norm-
matched by default (`--xattn_boost_renorm img`, ρ 0.5): the boosted hidden
state is rescaled back toward the norm it would have had unboosted, so the
intervention is a *rotation toward the cross-attn direction* on (near) the norm
shell downstream blocks were trained on, not an unconstrained residual add.

Origin: `_archive/proposals/frontload_text_boost.md` arm (b), Phase-0 G1+G2 PASS;
renorm = Phase-1'' arm (g) (`bench/frontload_text_boost/report.md`). The
σ-gated CFG arm (a) of the same proposal is CLOSED — style collapse at
every strength that does anything; do not re-propose plain high-σ CFG-up.

## Usage

```bash
make test XATTN_BOOST=2 PROMPT="..."          # band 0.85, renorm img ρ0.5
make test XATTN_BOOST=1.5 XATTN_BOOST_BAND=0.95   # tighter window (~4 steps)
python inference.py ... --xattn_boost 2.0            # shipped config
python inference.py ... --xattn_boost 2.0 --xattn_boost_renorm off  # raw gain
python inference.py ... --xattn_boost_renorm img --xattn_boost_renorm_frac 1.0
```

`--xattn_boost 1.0` = off (exact identity — pinned by
`tests/test_xattn_gain.py`; the renorm flags are inert while the boost is
off). Phase-0 strengths: 2.0 strictly dominated 1.5 on adherence (fixed 5/7
baseline-failed watch tags vs cfg_hi's 2/7-with-style-collapse). λ ∈ {2.5, 3}
and bands other than 0.85 are unmapped.

## Norm matching (`--xattn_boost_renorm`, Phase-1'')

The raw gain pushes each token's hidden state off the norm distribution the
next block was trained on (mixture-OOD): burn direction +18–25% saturation on
complex prompts. Three modes:

- `img` (default, ρ 0.5) — one shared scale per image matching the
  per-image mean token norm to its gain-1 value. Energy budget bounded,
  but the token-norm distribution keeps its shape — its peaks carry local
  contrast (neon, highlights, speculars), which survive intact. Eyeball
  winner of Phase-1'' (`results/20260712-0013-phase1pp-renormI`).
- `tok` — every token matched to its own gain-1 norm. The tokens whose
  energy legitimately spikes under the boost are exactly the highlight peaks,
  and they get clamped hardest → flat grey tone. Kept as the bench reference;
  don't ship.
- `off` — the raw pre-renorm residual gain.

`--xattn_boost_renorm_frac` ρ applies `scale**ρ`: 1.0 = full match, 0.0 = raw
boost. ρ 0.5 at λ 2 was the tone sweet spot; the ρ grid beyond {0.5, 1.0} is
unmapped. NB the bench's mean-HSV-sat burn detector is misleading in this
direction: boosted arms replace flat saturated washes with detailed scenes
(legitimate whites/greys), dropping the mean while the peaks brighten — judge
tone by eyeball, not Δsat.

## Mechanism

- Per-block non-persistent buffer `Block._xattn_gain`
  (`library/anima/models.py`), read inside the compiled `_forward` — retuned
  per step via `fill_()` with no recompile (same pattern as the
  mod-guidance buffers). Multiplies the cross-attn residual *after* the
  `gate_cross_attn` AdaLN gate, before the residual add. Renorm state lives
  next to it as plain Python attrs (`_xattn_renorm` / `_pertoken` / `_frac`
  — static dynamo guards, one graph variant per combo).
- `library/inference/adapters.py::set_xattn_boost_state` is the single
  runner-facing entry point (gain + renorm together; `gain=1.0` restores
  exact identity incl. renorm-off). Every denoise loop sets it before the
  cond forward and resets before the uncond forward (so the boost lands
  entirely in the guidance delta and none of it is spent amplifying the
  negative prompt), plus a `finally` reset so no state leaks across
  generations.
- Threaded to loop runners via `SamplerSideChannels.xattn_boost` /
  `.xattn_boost_band` / `.xattn_boost_renorm` / `.xattn_boost_renorm_frac`
  (`library/inference/sampler_context.py`).
- One important geometry fact found in Phase-1': the T5-row embedding
  scaling arm can never touch attention *allocation* — `k_norm` (RMSNorm
  per token×head) annihilates positive row scales on the K path, so
  embedding-level boosts are pure V/loudness. Allocation needs a QK logit
  bias (`Attention._ctx_k_bias`, bench arm (d) — probe-only, not shipped).

## What it does / doesn't do

- **Moves relations & bindings between things the model knows** (multi-subject
  role↔attribute bindings, object relations, scene adherence). It does NOT
  conjure unknown concepts — a multiplicative lever on ~zero signal is ~zero
  (theremin/upside-down-umbrella class prompts stay failed). Same law as
  "LoRA cross-attn learns labeled tags only", observed at inference.
- Amplifies ALL caption tags, wanted or not — framing priors (`cropped`,
  an artist's habitual close-up crop) ride the boost, and with style/artist
  tokens in the caption the content win attenuates (the gain splits across
  tokens). The token-selective arm (c) of the proposal targets exactly this.
- Under CFG 1.0 (turbo student) there is no uncond pass; the single
  forward is boosted.

## Compose matrix

| Stack | Status |
|---|---|
| `SPECTRUM=1` | Wired. Real cond forwards boosted; forecast steps extrapolate from boosted cond features (consistent — and warmup covers the earliest in-band steps with actual forwards). |
| `--spd` | Wired (gate reads the re-spaced per-step σ). |
| `--fovea_sigma_c` | Wired; band sits above any sane σ_c so boosted steps are full-grid. |
| `--smc_cfg` / `--cfgpp` | Compose mechanically (they change the cond/uncond combine; the boost changes the cond forward). Interaction grid: Phase 1. |
| `MOD=1` / `--dave` / `--cns` / FSG | Orthogonal pathways (AdaLN modulation / DC attenuation / noise recoloring / mid-σ latent calibration — FSG's band [0.59, 0.75] doesn't overlap σ ≥ 0.85). FSG's internal forwards run at identity. |
| Tiled diffusion | Wired (per-tile cond passes boosted). |

## Files

- `library/anima/models.py` — `_xattn_gain` buffer + renorm attrs + `_forward` read
- `library/inference/adapters.py` — `set_xattn_boost_state` (+ `set_xattn_gain` / `set_xattn_renorm` primitives)
- `library/inference/args.py` — `--xattn_boost`, `--xattn_boost_band`, `--xattn_boost_renorm[_frac]`
- `library/inference/generation.py` — inline + tiled loop wiring
- `networks/spectrum.py`, `networks/spd.py`, `networks/foveated.py` — runners
- `tests/test_xattn_gain.py` — identity/scaling/reset invariants
- `bench/frontload_text_boost/` — Phase-0/1 grids + report
