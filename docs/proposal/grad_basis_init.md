# grad_basis init — seed `lora_down` from the task gradient, and ship a universal basis

Status: PROPOSAL. Motivating measurement done (no training): `bench/grad_init/`
(2026-09-12, two artist triples). Nothing trained yet.

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

**E0 — universal basis generalizes (no training, ~30 GPU-min).** Build the
basis from N≈16 artists × 2 passes; measure capture on 4 held-out artists with
`probe_subspace.py`. **Kill:** held-out capture < 0.45 or < 2× weight_svd →
ship only the per-run mode.

**E1 — does the seed move the render? (4 paired runs, ~1.5 GPU-h).** One
artist, `--deterministic`, same seed/data order: `kaiming` / `weight_svd` /
`grad_svd` (per-run) / `basis_file`. Read: (a) train loss over the first 200
steps, descriptive only; (b) paired PE-cos render grid vs the artist's
reference set + full-res eyeball. **Kill:** no render-level separation between
`weight_svd` and either gradient seed → keep `weight_svd`, close the line with
the probe numbers as the record (the first-step advantage did not survive
training). **Pass:** ship `basis_file` as the new default `down_init` with the
catalog artifact, keep `grad_svd` as the opt-in when the run's own data is
worth a sketch pass.

**E1 side-arms — the same "early noise picks the mode" hypothesis, cheaper
levers (fold into E1's paired grid, no new line).** Each is a zero- or
near-zero-code knob that attacks the noise the profile measured; a gradient
seed is the strongest form of the same idea (first step = whole-pool
gradient), so they belong in one ranking:

| arm | what it does | cost |
|---|---|---|
| `lr_warmup_steps=0.15` (from 0.05) | shrinks early step magnitude until momentum has averaged | 0 |
| staged accumulation 4 → 2 → 1 over the first 5 % / 10 % / rest | shrinks early direction variance (SNR 0.12 → 0.24; still B≈70-dominated, ~15 % extra compute) | ~30 lines in `loop.py` (runtime `gradient_accumulation_steps` + scheduler step count) |
| `weighting_scheme` Min-SNR-style | down-weights σ>0.5 samples that carry 4–10× the noise for equal signal | 0 (existing knob) |
| `layer_start=12` | drops the blocks whose gradient is pure noise at this data scale | 0 (existing knob) |

Read as E1 (paired `--deterministic`, render-level). Kill for the group: no
arm separates from baseline at render level → the early-noise hypothesis is
closed and only the init question remains.

**E2 (only if E1 passes) — the one separation angle left.** O-LoRA-style
projection of artist-2's basis off artist-1's, restricted to blocks 12–16
cross-attn K/V/out (the only region where artists diverge). Read as a merge
bench (`merge_loras` concat of the two, tag-routed render grid). Low priority;
the probe says the payoff is bounded by ~18 points of gradient energy.

## Cost / placement

E0 + E1 ≈ 2 GPU-h. Code: one `down_init` mode in `networks/lora_modules/lora.py`
(+ the `config.py` validator and `configs/methods/lora.toml` comment), the
sketch extracted from `bench/grad_init/probe_subspace.py` into a small
`networks/grad_basis.py`, a catalog row, a `make grad-basis` builder. Tier 1.5:
the probe is the bench, E1 the invariant test's basis.

## Open questions

- Pool for the universal basis: all 83 artists at 1 pass, or 16 at 2? The
  probe says images beat σ-redraws, so breadth wins — but confirm on E0.
- Whether `r_store = 64` leaves the leading directions stable enough to be
  worth truncating to 16 for `low_vram` runs, or whether small dims should
  re-sketch.
- Interaction with the soup line's uncond init: sketch on empty captions for
  the shared init, then per-artist? Untested; not blocking.
