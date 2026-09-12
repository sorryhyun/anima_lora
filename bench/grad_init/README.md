# grad_init — gradient-SVD init probe (no training)

Question: if `lora_down` were seeded from the task gradient's top-r row space
(LoRA-GA / LoRA-One style) instead of `down_init="weight_svd"`, would two
artists land in different input subspaces, and how much of the first-step
full-FT gradient does each candidate basis pass through?

`probe_subspace.py` accumulates a one-pass randomized sketch `Ωᵀ·G` per
LoRA-target Linear over an artist's cached dataset (frozen DiT, trainer-default
logit-normal σ drawn stratified, `--passes K` revisits with fresh σ/noise),
takes the top-r right singular vectors, and compares bases by (a) mean squared
principal cosine `‖V1ᵀV2‖²/r` (null `r/in`) and (b) **energy capture**
`‖S·V‖²/‖S‖²` — the share of the sketched gradient energy inside `span(V)`.
Each artist is split 2×2 (image half × pass parity) so within-artist
reliability is measured alongside every between-artist number.

```bash
make daemon-run ARGS="bench/grad_init/probe_subspace.py --artists aak,abmayo,akipeko --gradient_checkpointing --passes 4 --save_bases"
```

## 2026-09-12 — aak / abmayo / akipeko, r=32, q=64, 4 passes (`results/20260912-1047-full3_p4_cap/`)

Block token layers (adaln excluded — their input is one vector per image, so
per-sample G is rank-1 and the "subspace" is just the t-embedding span).

| a1's gradient energy captured by | value |
|---|---|
| a1's own held-out image half (reliability ceiling) | 0.74 |
| a2's full basis | 0.66 |
| a2's half basis (same n as the ceiling) | 0.64 |
| `weight_svd` top-32 of W | 0.21 |
| random rank-32 | 0.016 |

Subspace overlap (mean sq. cosine, null 0.016): between artists 0.32; within
artist, image split 0.35 / 0.39, σ-redraw split 0.38 / 0.42; vs weight_svd 0.12.

Where the artist-specific part sits (own-half capture − a2 capture):
`cross_attn.kv_proj` 0.79→0.59 and `cross_attn.output_proj` 0.80→0.63 carry it;
self-attn / MLP are ≤0.06 apart. By depth: blocks 0–8 and 20–27 are shared
(gap ≤0.05), blocks 12–16 are where artists diverge (0.73→0.40, 0.64→0.44).

**Read.**
1. A gradient-seeded basis passes ~3.5× more first-step gradient than
   `weight_svd` (0.74 vs 0.21). weight_svd is a poor proxy for "where the task
   pushes".
2. It does **not** give artists different subspaces: another artist's basis
   captures 0.66 vs 0.74 for your own held-out half. The gradient row space is
   dominated by a shared, artist-agnostic core; the artist-specific residual is
   ~8 points of energy, concentrated in cross-attn K/V/out and mid blocks.
3. Corollary: a *universal* gradient basis computed once over many artists
   would be a near-free init worth ~0.65 capture with no per-run backward pass.
4. Noise floor is images, not σ draws (σ-redraw split > image split), and the
   top-32 subspace beyond its leading few directions is unstable at 32–54
   images per artist — a rank-32 GA init here is "shared core + noise", not a
   precise per-artist address.

## 2026-09-12 — channel_(caststation) / sweetonedollar / aak (a deliberately distant pair), same protocol (`results/20260912-1057-chan_sweet_p4/`)

| channel's gradient energy captured by | value |
|---|---|
| own held-out image half | 0.76 |
| sweetonedollar's basis | 0.58 |
| aak's basis (overlap 0.30 vs 0.26 for sweet) | — |
| `weight_svd` | 0.22 |

Overlap: channel vs sweet 0.26, channel vs aak 0.30, within-artist 0.33–0.34.
The artist-specific gap (own-half − other) grows from 8 to 18 points for the
distant pair, so style distance is visible — but 58% of the first-step
gradient is still shared, and the cross-artist basis still beats weight_svd
2.7×. Same anatomy as the close pair: cross-attn kv/out are the widest kinds
(0.83→0.55 / 0.83→0.58), self-attn/MLP now open too (0.10–0.17), blocks 12–16
are the artist-specific depth (0.76→0.36, 0.65→0.35) while block 0 is fully
shared (0.91→0.87).

## 2026-09-12 — gradient noise scale (`probe_noise_scale.py`, `results/20260912-1107-noise_p4/`)

McCandlish `B_simple = tr Σ / ‖g‖²` from per-sample sketched gradients (LoRA = 0,
batch 1, 4 σ-redraw passes), aak / channel_(caststation), block token layers.

| | aak | channel |
|---|---|---|
| pooled, σ random per sample (what a train step sees) | 67 | 78 |
| σ ∈ [0, 0.34) | 8 | 14 |
| σ ∈ [0.34, 0.5) | 15 | 13 |
| σ ∈ [0.5, 0.66) | 111 | 50 |
| σ ∈ [0.66, 1] | 81 | 100 |

Per-sample SNR at batch 1 is ~0.12 overall, ~0.3 at low σ, ~0.1 at high σ:
high-σ samples carry 4–10× the noise energy of low-σ ones for similar signal.

Depth: ~90 % of the *consistent* gradient energy ‖g‖² sits in blocks 18–27
(block 22 alone 11–14 %); blocks 0–10 each hold < 1 % signal but 1–6 % noise,
so their B is 200–18 000 — at this data scale an early-block LoRA parameter
receives pure noise per step (Adam then random-walks it at LR scale).

Caveat this puts on the capture numbers above: `‖S·V‖²/‖S‖²` counts noise
energy too. Where B is huge (early blocks) the "gradient basis" is really the
input-activation covariance basis — which is why block 0 looked 0.91 "shared"
across artists. The capture ranking (gradient basis ≫ weight_svd) stands, but
the subspace is task-informed only where B is small (blocks ≥ 16).

## 2026-09-12 — E0: does a universal basis generalize? (`build_universal_basis.py`, `results/20260912-1314-e0_univ20/`)

Pool = 20 artists × 32 images × 2 passes (`--max_samples 32` so `sincos`'s 351
images don't own the basis); held out = aak / channel_(caststation) /
sweetonedollar / ootomo_takuji, never in the pool. Ω is seed-fixed across
artists, so per-artist sketches are additive and the pooled basis is the top-r
right singular vectors of `Σ_a S_a`. r=32, q=64, block token layers.

| held-out artist's gradient energy captured by | value |
|---|---|
| its own held-out image half (ceiling) | 0.709 |
| **the universal basis (N=20)** | **0.633** |
| `weight_svd` top-32 | 0.206 |
| random rank-32 | 0.016 |

**E0 gate: PASS** — 0.633 ≥ 0.45 and 3.1× weight_svd. The universal basis
reaches **89 % of the reliability ceiling** on artists it has never seen; per
artist 0.602–0.661 against ceilings 0.671–0.733, i.e. no held-out artist is an
outlier.

**Breadth saturates at N≈1** — the open question ("all 83 at 1 pass, or 16 at
2?") is answered, and it is neither:

| pool | capture |
|---|---|
| N=1 | 0.603 |
| N=2 | 0.614 |
| N=4 | 0.624 |
| N=8 | 0.628 |
| N=16 | 0.631 |
| N=20 | 0.633 |
| N=20, 1 pass only | 0.629 |
| N=20, per-artist equal weight | 0.635 |

One artist already buys 95 % of what 20 buy; doubling the pool adds ~0.4 points
and the second σ-pass adds 0.4. So the shipped artifact does **not** need a
large corpus — 8–16 artists at 1 pass is past the knee, and per-artist
normalization (equal weight, not "whoever has the biggest gradient") is a free
+0.2. The corollary is the honest one: what the basis captures is an
artist-agnostic property of the model + anime-illustration data, not a curated
average of many styles.

**Depth is the whole story** (bands from §gradient noise scale):

| blocks | universal | own-half ceiling | weight_svd |
|---|---|---|---|
| 0–11 | 0.603 | 0.667 | 0.186 |
| 12–17 | 0.408 | 0.602 | 0.099 |
| 18–27 | **0.804** | 0.823 | 0.293 |

In blocks 18–27 — where ~90 % of the *consistent* gradient energy lives — the
universal basis is at **98 % of the ceiling**: there is essentially nothing
per-artist left to seed, and a shipped basis is as good as a per-run sketch.
Blocks 12–17 are where it gives up 19 points to the ceiling, matching the
artist-specific depth the pairwise probes found (0.76→0.36). Blocks 0–11 sit
between, and per the noise-scale caveat their "capture" is activation-covariance
alignment rather than task direction.

By kind, `cross_attn.kv_proj` / `output_proj` keep the widest gap to the
ceiling (0.59/0.63 vs 0.79/0.79); self-attn and MLP are within 0.04–0.05. adaln
rows read 1.000 for every basis (their input is one vector per image, so any
basis spans it) and stay out of the headline aggregate.

**Read.** Ship the universal basis: E0's condition for `basis_file` is met, and
the depth split says the per-run `grad_svd` mode's only real advantage over it
is blocks 12–17. E1 now decides whether *any* gradient seed moves the render;
the E0 numbers argue its `grad_svd` vs `basis_file` arms should separate little
if at all, so a `basis_file`-vs-`weight_svd` separation is the load-bearing read.

## 2026-09-12 — E1: does the seed move the render? KILL (`results/20260912-1410-e1/`, blind sets s22–s25)

Five paired arms on `aak` (r=32, α=128, 8 ep = 256 steps, `--deterministic
--paired_step_rng --seed 42`): `kaiming` / `weight_svd` / `grad_svd` (per-run,
2 passes) / `basis_file` (E0's `grad_basis_universal_r32`) / `min_snr` (γ=5,
weight_svd init). Two reads:

**Member-caption read (`e1_read.py`, 12 member prompts, 20 steps, cfg 1).**
Every arm lands a *different* image — arm-vs-arm PE cos 0.89–0.95 on a 0.73
unrelated floor, i.e. the seed does pick a mode — but CMMD is below the
real-vs-real floor (0.61) for all six columns, so it cannot rank them. Only
descriptive fact: `min_snr` stays closest to base (cos 0.941, CMMD 0.33 vs
0.53–0.56 for the four init arms).

**Blind A/B on general prompts (`e1_blind.py`, 12 `@aak` rows, 28 steps, cfg 4,
2 seeds per set, fresh seeds per set so the shared control never recurs).**
Direct pairings against the shipped `weight_svd` control, graded blind:

| set | pair | pairs W–X (tie) | rows W>X / X>W / tie |
|---|---|---|---|
| s22 | weight_svd vs basis_file | 10–10 (4) | 4 / 3 / 5 |
| s23 | weight_svd vs grad_svd | 11–9 (4) | 3 / 2 / 7 |
| s24 | weight_svd vs kaiming | 12–9 (3) | 3 / 2 / 7 |
| s25 | weight_svd vs min_snr | 13–8 (3) | 5 / 3 / 4 |

All four sit inside the 15–9 seed-twin noise floor with no row-level lean
(pooled side bias A 34 / B 46, mild). **Verdict: the init does not reach the
render.** A 3× first-step gradient capture (0.63 vs 0.21) is fully absorbed by
256 steps of training; even `kaiming` vs `weight_svd` is flat, so the whole
`down_init` axis is render-neutral at this recipe. `min_snr` is flat too — the
"moved least from base" signature of the member read did not translate into a
preference either way.

Both kill conditions in `docs/proposal/grad_basis_init.md` §E1 fire: keep
`weight_svd` as the default, keep `grad_svd` / `basis_file` as documented opt-in
modes with the probe numbers as their record, do not ship a catalog basis.
Reports: `project/cjk_aware_anima/reports/blind_s2{2,3,4,5}_E1_*.md`.
