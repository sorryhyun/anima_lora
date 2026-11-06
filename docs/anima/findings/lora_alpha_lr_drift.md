# Anima LoRA: alpha and lr trade one for one, and the bilinear drift does not reach the render

> **STATUS (2026-11-06).** A/B RESULT + MEASUREMENT. On the shipped recipe
> (`configs/methods/lora.toml`, no T-LoRA mask) and 30-image sets:
> rank 16 against rank 32 at matched ||dW|| leaned to r16 without significance (s34,
> 28–20, cluster p 0.25); alpha 512 / lr 1.25e-5 against alpha 128 / lr 5e-5 tied
> (s35, 25–22, cluster p 0.62). The same alpha × lr gives the same ||dW|| (within 4 %).
> alpha 512 cuts the bilinear drift share of ΔW from 16 % to 6 % (median over
> modules) and the render does not change. The shipped rank, alpha and lr stay.
> **Open:** s36 repeats s35 on sincos (354 images, ~1416 updates against ~120),
> where the drift share should be several times larger.

## The question

arXiv 2606.12883 (Zhang et al., "The Hidden Power of Scaling Factor in LoRA
Optimization", 2026-06) splits one LoRA step under Adam or Muon into a task-aligned
signal `s·(BΔA + ΔB·A)`, Θ(αη), and a bilinear drift `s·ΔB·ΔA`, Θ(αη²). At a fixed
α·η the update is the same to first order and the drift falls as 1/η. The paper finds
the optimal alpha at `α ≈ C·√r` with a large C (256 for LLMs) at the full fine-tune
lr, against the usual α = r or 2r. Its evidence is LLM loss and benchmarks, plus one
qualitative Flux DreamBooth figure.

Read for Anima, it predicts:

1. at matched step length, high alpha / low lr draws cleaner LoRAs (less drift);
2. the alpha for another rank scales as `√r`, not as `r` (the rsLoRA rule, which
   `adaln_alpha = network_alpha · sqrt(adaln_rank / network_rank)` already follows).

Reference lrs for the full fine-tune (`η_FFT`) of the two lines:

| | source | full fine-tune | their LoRA |
|---|---|---|---|
| Anima (Cosmos-Predict2-2B base) | `nvidia-cosmos/cosmos-predict2`, `documentations/post-training_text2image.md` and `configs/base/experiment/cosmos_nemo_assets_lora.py` | fused AdamW, lr 2^-14.5 ≈ 4.3e-5 × `f_max` 0.4 = **1.7e-5 peak**, no warmup, linear to 0 over 1000 iters, wd 0.2, EMA, batch 8 | r16 / α16 (scale 1), peak 2.6e-5 (2B); r32 / α32, peak 1.1e-5 (14B) |
| MOSS-Image | arXiv 2610.23374, App. A.2 | pretraining only: Muon–AdamW, lr 5e-5 (Muon RMS-matched), β2 0.95, wd 0.01, clip 1, EMA 0.9999, ~8.8 M tokens per update | — |

Anima's shipped LoRA runs lr 5e-5 at scale 4 (rank 32 / alpha 128), about 3×
`η_FFT` with a small alpha; MOSS's runs 5e-5 at scale 4 under Muon, at the
pretraining lr. The paper's own rule at rank 16 and `η_FFT` (scale 64 × 1.7e-5)
would take a first-order step ~5× the shipped one, past the displacement ceiling
the recipe findings measured (GA2 at 1e-4 tangles hands;
[`lora_gradient_noise_recipe.md`](lora_gradient_noise_recipe.md)). The test therefore
holds α·η at the shipped value and moves only the split.

## Setup

Shipped `configs/methods/lora.toml` (rank 32 / alpha 128, lr 5e-5 constant, GA2, median
clip k 2 per micro-batch with σ bands, adapter EMA 0.1, `weight_svd` down init,
channel scaling 0.5, REPA relational, 8 epochs) with `use_timestep_mask=false`.
caststation (30 images, 120 updates) and sweetonedollar (31, 124), training seeds
42–44. Control arm: the s31 / s32 `nomask_{art}_s{seed}` LoRAs. Treatment rescaled to the
control's median ||dW|| (`match_dw`); blind pairs at the `make test` recipe, four
prompts × two render seeds, graded in `anima-blind-pairs`. Runner:
`bench/anima_abclip/run_ab.py --exp {rank,alpha,alpha_sincos}`.

`dw_norm` multiplies by each module's alpha / rank since s35 (before, every arm ran
alpha / rank 4, so older ratios and match factors are unchanged, absolute values 4×).

## s34: rank 16 against rank 32

r16 at alpha 64 (alpha / rank 4 in both arms). Render seeds 11 / 12.

| | r32 | r16 |
|---|---:|---:|
| blind wins, caststation | 10 | 14 |
| blind wins, sweetonedollar | 10 | 14 |
| cross-seed MAD, caststation | 46.2 | 45.2 |
| cross-seed MAD, sweetonedollar | 41.6 | 50.9 |

**r16 28–20, no ties, cluster p 0.25.** Two clusters carry it (cast s44 and sweet s42,
7–1 each); the other four are 4–4 or 5–3 to r32. Lower rank does not make training
seeds agree.

At a fixed alpha / rank, ||dW|| scales as √r under Adam: r16 moved **0.72×** as far as
r32 (1/√2 = 0.707), so it was rescaled up by 1.25–1.43 before rendering. A trained-in
match would use alpha `128 · √(16/32) ≈ 90.5`. The adaln modules also differed in
scale between the arms (alpha / rank 4 at r16, 5.66 at r32, from the √r rule), which
leaves r16 at 0.968× of r32 under the scaled `dw_norm`. Rank 16 halves the file.

## s35: alpha 512 / lr 1.25e-5 against alpha 128 / lr 5e-5

Same α·η, so the same first-order step and a quarter of the per-step drift. Render
seeds 13 / 14. Report: [`report_alpha.md`](../../../bench/anima_abclip/report_alpha.md).

- **||dW||: a512 lands at 0.96–0.99× of a128** on all six (artist, seed) pairs
  (`match_dw` factors 1.01–1.04), as the first-order argument predicts. Loss and
  training were unremarkable.
- **Blind: a128 25–22, 1 tie, cluster p 0.62.** The artists split: caststation 16–8 to
  a128 (s43 alone 8–0), sweetonedollar 14–9 to a512.

| artist | seed | a128 | a512 |
|---|---|---:|---:|
| cast | s42 | 5 | 3 |
| cast | s43 | 8 | 0 |
| cast | s44 | 3 | 5 |
| sweet | s42 | 2 | 6 |
| sweet | s43 | 3 | 5 |
| sweet | s44 | 4 | 3 |

- Cross-seed MAD: caststation 48.4 → 50.4, sweetonedollar 40.9 → 37.9 (a128 → a512).

## How large the drift is

`bench/anima_abclip/drift_share.py` rebuilds `A0 = V_r^T / √3` from the base DiT with the
trainer's own `_top_right_singular_vectors` (q/k/v and cross-attn k/v on their fused
runtime weight), aligns signs per row, and splits the trained adapter as
`ΔW = s·B·A0 + s·B·D` with `D = A − A0`. The second term is the accumulated drift.
Median over modules (sign-aligned row cos to `A0` ≥ 0.997 at the median):

| | `‖B·D‖ / ‖B·A0‖` | mlp.layer2 | adaln self-attn | `‖D‖ / ‖A0‖` |
|---|---:|---:|---:|---:|
| a128, 6 runs | **0.15–0.17** | 0.26–0.30 | 0.32–0.40 | 0.08–0.09 |
| a512, 6 runs | **0.06** | 0.11–0.14 | 0.12–0.15 | 0.03 |

The drift is not negligible at the shipped lr: a sixth of ΔW, a third in mlp.layer2 and
the adaln modules. alpha 512 cuts it 2.7× (A moves a third as far), and the blind A/B
cannot tell the arms apart. At this length the rotation of the input basis neither helps
nor hurts. The `down_init` blind A/Bs found the same for the starting basis (E1,
`bench/grad_init/README.md`: kaiming, `weight_svd` and gradient bases all within the
seed-twin floor).

The share grows with lr × updates (`B·A0` ∝ η·n, `B·D` ∝ η²·n²), so a longer run should
show more of it. sincos (354 images) runs ~1416 updates at the shipped 8 epochs, about
12× caststation; s36 tests it.

## Cross-seed agreement and the blind winner

In each artist of s32, s34 and s35, the arm whose training seeds agreed more (lower
cross-seed MAD) won the blind set in 5 of 6 cases (s32 caststation shift2, s32
sweetonedollar shift1, s34 caststation r16, s35 caststation a128, s35 sweetonedollar
a512; the exception is s34 sweetonedollar, where r16 won at the higher MAD). Most of the
MAD gaps are ±3, about the render-seed spread of one arm (the same r32 LoRAs read 46.2 /
41.6 at render seeds 11 / 12 and 49.1 / 40.6 at 7 / 8), so this may be chance. If it
holds, MAD is a cheap pre-screen before a blind set.

## Transfer, if a longer run shows an effect

The argument needs an Adam-family optimizer (Muon included, per the paper), a B·A adapter
starting at B = 0, and every trainable parameter inside the alpha scale:

- Hydra: the router lr is `unet_lr × router_lr_scale`; lowering lr 4× needs
  `router_lr_scale` 4× to keep the router where it is. `repa_head` (absolute REPA mode
  only) is in the same position; the shipped relational mode has none.
- Turbo: the student starts from an extracted delta (A and B both nonzero), so alpha
  changes the initial ΔW unless the loader rescales it, and the co-trained critic is the
  setting where channel scaling, also an Adam-geometry change, was net-negative.
- MOSS LoRA (Muon, lr 5e-5): already at the pretraining lr, so only the high-alpha /
  low-lr split is open; i2i runs (swap_text 3000 steps) are the long regime. LoKr and
  LoHa train on AdamW with another parameterization and fall outside the argument.

## Files

- `bench/anima_abclip/run_ab.py` — `--exp rank` (s34), `alpha` (s35), `alpha_sincos` (s36).
- `bench/anima_abclip/drift_share.py` — the drift split above.
- `bench/anima_abclip/report_rank.md`, `report_alpha.md` — blind tallies.
- σ profile without the mask (s34 runs, `sigmalog2_{cast,sweet}_s42`):
  [`lora_gradient_noise_recipe.md`](lora_gradient_noise_recipe.md).
