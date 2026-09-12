# grad_basis init — seed `lora_down` from the task gradient, and ship a universal basis

Status: **CLOSED — E1 KILLED 2026-09-12.** E0 passed (a universal basis reaches
0.633 held-out capture, 3.1× `weight_svd`) but E1's paired runs + blind A/B on
general prompts found **no render-level separation** for any arm against
`weight_svd` — not `basis_file`, not `grad_svd`, not even `kaiming`, and not the
`min_snr` side-arm (all four sets inside the 15–9 seed-twin floor; record in
`bench/grad_init/README.md` §E1). `weight_svd` stays the default; `grad_svd` /
`basis_file` remain as opt-in modes; no catalog artifact is shipped. The rest of
this document is the pre-registration as written.

## TL;DR

`down_init="weight_svd"` seeds `A` with W₀'s top-r right singular vectors —
"where W is big", not "where the task pushes". Measured on this DiT, that basis
passes **0.21** of an artist's first-step full-FT gradient energy. The top-r
row space of the artist's own gradient passes **0.74** (held-out image half),
and — the useful surprise — **another artist's gradient basis passes 0.58–0.66**.
Random passes 0.016.

So two things follow, in order of cost:

1. **`down_init="grad_svd"`** — a per-run sketch pass (~1 s/img, no training)
   before `apply_to` seeds `A` from the run's own gradient. LoRA-GA / LoRA-One
   lineage: with `B = 0` the first optimizer step becomes the rank-r truncated
   full-FT step. Output is unchanged plain LoRA.
2. **A shipped universal basis** (`networks/calibration/grad_basis_r64.safetensors`,
   one per DiT depth) computed once over many artists — a `weight_svd` drop-in
   replacement worth ~0.6 capture with **no per-run backward**, usable from the
   GUI / ComfyUI trainer node / cold starts where a sketch pass is unwelcome.

What this is **not**: a merge-interference fix. Artists share most of the
gradient row space (see anatomy below); this changes the first-step subspace,
not the per-artist address.

## Measurement (bench/grad_init/README.md)

Sketch `S = Ωᵀ·G` per LoRA-target Linear (314 on the 28-block base), frozen
DiT, trainer-default logit-normal σ drawn stratified, 4 σ-redraw passes,
artists split 2×2 (image half × pass parity) so every between-artist number has
a within-artist reliability twin. Capture = `‖S·V‖² / ‖S‖²`.

| a1's first-step gradient energy inside `span(V)` | aak vs abmayo | channel_(caststation) vs sweetonedollar |
|---|---|---|
| a1's own held-out image half | 0.74 | 0.76 |
| the other artist's basis | 0.66 | 0.58 |
| `weight_svd` top-32 | 0.21 | 0.22 |
| random rank-32 | 0.016 | 0.016 |

Anatomy (both pairs): blocks 0–8 and 20–27 are shared (block 0: 0.91→0.87
across the distant pair); **blocks 12–16 are the artist-specific depth**
(0.76→0.36); by kind `cross_attn.kv_proj` / `output_proj` open widest, self-attn
/ MLP ≤0.17. The noise floor is image count, not σ draws (σ-redraw split-half
> image split-half), and beyond its leading directions a rank-32 subspace is
unstable at 30–54 images.

**Caveat from the noise-scale probe (same day, `bench/grad_init/README.md`
§gradient noise scale).** Capture counts noise energy as well as signal. The
consistent gradient ‖g‖² is ~90 % in blocks 18–27; blocks 0–10 have
`B_simple` in the hundreds to thousands, so there the "gradient basis" is the
input-activation covariance basis (EVA-like), not a task direction — that is
why block 0 reads 0.91 "shared". The ranking gradient-basis ≫ weight_svd holds
everywhere (activation covariance is still where updates flow), but E1 should
also read a `layer_start=12`-style arm: if early-block LoRA capacity only
absorbs noise, the init question there is moot and the capacity is better
removed than seeded.

Constraints inherited from the sigma_lowres record (`docs/proposal/lora_distill.md`
§Constraints) that bind here: **placement amplifies** — an init picks the
endpoint mode, so the expected effect is mode selection, not "ΔW stays in the
seed"; ΔW-cosine is not a gate; data-FM loss is not a quality gate; paired reads
need `--deterministic`.

## Mechanism

```
sketch pass (per run or once for the universal basis):
  for each (image, σ, ε) in the pool:   frozen DiT, FM loss, backward
    per target Linear:  S[layer] += (Ωᵀ δ) xᵀ           # q × in, fp32
  V_r[layer] = top-r right singular vectors of S[layer]

init:   A ← V_rᵀ / sqrt(3)     (same Kaiming row-norm match weight_svd uses)
        B ← 0                   (ΔW = 0; plain LoRA, nothing else changes)
```

- Rank handling: store the basis at `r_store = 64`; a run with `network_dim ≤ 64`
  takes the leading `network_dim` columns, larger dims fill the remainder with
  Kaiming. Spectrum is stored alongside for a later per-layer rank allocation
  (EVA-style; **not** in scope now).
- Channel scaling (`_register_channel_scale`) stays as is — it reads
  `lora_down.weight` after init, so it composes with either init.
- adaln Linears (input = one t-embedding vector per sample) get the same
  treatment; their "row space" is the σ-curve of the t-embedding, which is a
  reasonable seed and needs no special case.
- Depth-baked like `channel_stats.safetensors`: a basis is per checkpoint
  arch (28-block base vs 40-block Anima-2.9B). Stamp `ss_num_blocks` and
  refuse on mismatch, the `docs/methods/anima-2.9b.md` envelope.
- Storage: 314 layers × in × 64 fp16 ≈ 110 MB at r=64 for the base. Ship
  through the model catalog (`library/downloads.py`, `model-catalog` skill),
  not in git.

Per-run mode (`grad_svd`) reuses the run's `CachedDataset` and the same
sketch; 1 pass over ≤64 images is ~1 min on the 16 GB box. `basis_file` mode
is a file read.

## Experiments (each has a kill; stop at the first)

**E0 — universal basis generalizes (no training). DONE 2026-09-12 — PASS.**
`bench/grad_init/build_universal_basis.py`, pool = 20 artists × 32 images × 2
passes, held out aak / channel_(caststation) / sweetonedollar / ootomo_takuji.
Held-out capture **0.633** vs ceiling 0.709 and `weight_svd` 0.206 — 3.1×, gate
was ≥ 0.45 and ≥ 2×. Three results that change the plan below:

- **Breadth saturates at N≈1** (0.603 at one pool artist → 0.633 at twenty; the
  2nd σ-pass is worth +0.004). The open question "83 at 1 pass or 16 at 2" is
  moot — 8–16 artists at 1 pass is past the knee, and per-artist equal
  weighting beats raw summing by +0.002. Build the shipped artifact small.
- **Blocks 18–27 are done**: universal 0.804 vs ceiling 0.823 (98 %). Where ~90 %
  of the consistent gradient energy lives, a shipped basis is as good as a
  per-run sketch, so `grad_svd`'s per-run backward buys ~nothing there.
- **Blocks 12–17 are the only gap**: universal 0.408 vs ceiling 0.602 — the same
  artist-specific depth the pairwise probes found, and now the sole remaining
  argument for the per-run mode (and the region E2 already targets).

**E1 — does the seed move the render? (4 paired runs, ~1.5 GPU-h). CODE LANDED
2026-09-12; runs queued.** One artist (`aak`, held out of E0's pool),
`--deterministic --paired_step_rng --seed 42`, `--path_pattern "aak/*"`, the
shipped lora.toml recipe (r=32, α=128, 8 epochs → 256 steps): `kaiming` /
`weight_svd` / `grad_svd` (per-run) / `basis_file` (E0's `grad_basis_universal_r32`).
Read: (a) train loss over the first 200 steps, descriptive only; (b) paired
PE-cos render grid vs the artist's reference set + full-res eyeball
(`bench/grad_init/e1_read.py`). **Kill:** no render-level separation between
`weight_svd` and either gradient seed → keep `weight_svd`, close the line with
the probe numbers as the record (the first-step advantage did not survive
training). **Pass:** ship `basis_file` as the new default `down_init` with the
catalog artifact, keep `grad_svd` as the opt-in when the run's own data is
worth a sketch pass.

**Result (2026-09-12): KILL.** Blind A/B on 12 general `@aak` rows, direct
pairings vs `weight_svd`: basis_file 10–10, grad_svd 9–11, kaiming 9–12,
min_snr 8–13 (pairs; rows split 3–4 / 2–3 / 2–3 / 3–5), every set inside the
seed-twin floor. The member-caption PE-cos read showed the arms landing
different images (cos 0.89–0.95), so the seed selects a mode, but no mode is
preferred. Full table: `bench/grad_init/README.md` §E1.

What landed (all four arms are runnable from a config today):

- `networks/grad_basis.py` — the sketch extracted from the probe, the basis
  artifact (fp16, `in × r`, `ss_num_blocks`-stamped, depth-mismatch refused),
  and the shared `V_rᵀ/√3` copy so a gradient seed is never also a step-size
  change. `down_init="grad_svd"` / `"basis_file"` in `lora.py` +
  `lora_anima/{config,factory,network}.py`; `grad_basis_file` is a network arg.
- The per-run mode sketches inside `train.py` (before `_create_and_apply_network`,
  where the DiT is loaded but no adapter exists yet), writes
  `<output_name>.grad_basis.safetensors`, and hands that path to the factory —
  so both modes share one load path. Measured 1.3 s/image on the 16 GB box.
  **Refused under `blocks_to_swap > 0`**: the swapper desyncs on forwards
  outside the training loop's cadence.
- Invariant test `tests/test_grad_basis_init.py` (19 cases).

**E1 side-arms — the same "early noise picks the mode" hypothesis, cheaper
levers (fold into E1's paired grid, no new line).** Each is a zero- or
near-zero-code knob that attacks the noise the profile measured; a gradient
seed is the strongest form of the same idea (first step = whole-pool
gradient), so they belong in one ranking:

| arm | what it does | cost |
|---|---|---|
| `lr_warmup_steps=0.15` (from 0.05) | shrinks early step magnitude until momentum has averaged | 0 |
| staged accumulation 4 → 2 → 1 over the first 5 % / 10 % / rest | shrinks early direction variance (SNR 0.12 → 0.24; still B≈70-dominated, ~15 % extra compute) | ~30 lines in `loop.py` (runtime `gradient_accumulation_steps` + scheduler step count) |
| `weighting_scheme = "min_snr"` | down-weights σ>0.5 samples that carry 4–10× the noise for equal signal | ~40 lines (**not** a free knob — see below) |
| `layer_start=12` | drops the blocks whose gradient is pure noise at this data scale | 0 (existing knob) |

Read as E1 (paired `--deterministic`, render-level). Kill for the group: no
arm separates from baseline at render level → the early-noise hypothesis is
closed and only the init question remains.

**Correction (2026-09-12): the `weighting_scheme` arm was not a 0-cost knob.**
The shipped choices are `uniform`/`none` (all-ones), `sigma_sqrt` (σ⁻², a ~400×
swing that *up*weights low σ rather than down-weighting high σ — it would raise
gradient variance, testing the opposite of the hypothesis) and `cosmap` (peaks
at σ=0.5). None is Min-SNR-shaped, so `min_snr` was added:
`min(SNR, γ)/(SNR + 1)` with `SNR = ((1-σ)/σ)²` — the v-prediction form, since
rectified flow regresses a velocity — **mean-1 normalized over the run's own σ
density** (Monte-Carlo, fixed seed) so the arm is a reshape and not a learning-rate
change. γ is the pre-existing, previously unconsumed `--min_snr_gamma`
(None → 5.0), which puts the peak at σ≈0.31; at γ=5 the normalized weight runs
2.14 at σ=0.31 → 1.29 at σ=0.5 → 0.54 at σ=0.66 → 0.03 at σ=0.9.
`library/anima/training.py`; tests in `tests/test_grad_basis_init.py`.

**Arm selection (2026-09-12, user):** run the `weighting_scheme` arm only, as an
A/B against the `weight_svd` init arm (same seed, so it is CRN-paired with it).
`lr_warmup_steps`, staged accumulation and `layer_start=12` are not being run.

**E2 (only if E1 passes) — the one separation angle left.** O-LoRA-style
projection of artist-2's basis off artist-1's, restricted to blocks 12–16
cross-attn K/V/out (the only region where artists diverge). Read as a merge
bench (`merge_loras` concat of the two, tag-routed render grid). Low priority;
the probe says the payoff is bounded by ~18 points of gradient energy.

## Cost / placement

E0 + E1 ≈ 2 GPU-h. Code landed 2026-09-12 (see §E1): the two `down_init` modes,
`networks/grad_basis.py`, the `train.py` sketch pass, the `min_snr` weighting
scheme, `tests/test_grad_basis_init.py`, and `bench/grad_init/e1_read.py` for the
render read. **Still owed if E1 passes**: a catalog row for the shipped basis
(it currently lives in `bench/grad_init/results/20260912-1314-e0_univ20/`, 40 MB,
untracked) and a `make grad-basis` builder target. Tier 1.5: the probe is the
bench, `tests/test_grad_basis_init.py` the invariant test.

## Open questions

- ~~Pool for the universal basis: all 83 artists at 1 pass, or 16 at 2?~~
  **Answered by E0: neither — the curve is flat past N≈8.** Ship 8–16 artists
  at 1 pass with per-artist equal weighting.
- Whether `r_store = 64` leaves the leading directions stable enough to be
  worth truncating to 16 for `low_vram` runs, or whether small dims should
  re-sketch.
- Interaction with the soup line's uncond init: sketch on empty captions for
  the shared init, then per-artist? Untested; not blocking.
