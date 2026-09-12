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
