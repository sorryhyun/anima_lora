# ×2 idea — tuning the head in weight space, reading a LoRA finish, and a Jacobian lens

A proposal, opened 2026-11-24. Runs and their numbers: `reports/x2_ovgf_mix_2026_11_24.md`; what
is queued or waiting: `x2_plan.md`.

## The idea

The user (2026-11-24): "픽셀디코더 파라미터 자체는 많지 않잖아. 이거 가중치를 직접 살짝씩 바꾸면서 결과를
튜닝하는 방법은 어려울까?", then "x2같은 경우는 뭔가 gan을 써도 잘 학습이 안되는거같은데 가중치 수정이랑
결과 관측을 반복하면서 오푸스 서브에이전트 스폰식으로 병렬로 돌리면서 머지 하면서 개선하면 어떨까".

The head has 23 – 25 M values with no meaning one by one, and a read of a sheet returns a few
bits, so a search by edit and observation needs a space of tens of knobs. The question this file
follows is which small space holds what the GAN finish does.

## Measured so far

All on the pair `x2ovll_1124_0139` (parent) and `x2ovgf_1124_0920` (its GAN finish), 32 cached
renders, one seed, spread unknown. amp is the high-pass amplitude as a share of the chain's;
stamp32 the texture repeated per token.

| space | what was read | amp, colour | stamp32, colour | grain on flat tokens, levels |
|---|---|---|---|---|
| | the parent | 0.603 | 0.414 | 0.23 |
| | the finish | 0.771 | 0.657 | 0.39 |
| one α for the whole change | α 0.5 / 1.5 | 0.667 / 1.005 | 0.517 / 1.115 | 0.29 / 0.72 |
| one stage's change | up2 alone / out alone | 0.787 / 0.625 | 1.068 / 1.289 | 0.84 / 1.14 |
| singular directions per matrix | top 8 / the rest | 0.829 / 0.632 | 1.188 / 1.695 | 0.91 / 1.94 |

- α is a strength dial: amp, stamp32 and grain rise together (`x2mg_1124_1025`).
- A stage or a singular cut taken alone loses the balance: more stamp and grain than the whole
  finish, and a tone shift of 2 – 3 levels (`x2mg_1124_1025`, `x2rk_1124_1100`).
- The amp rise sits in few directions: the top 8 per matrix, 13 – 37 % of the change's energy,
  give all of it.
- On the ×1 pair (base and sharp, the Hub files; read inline, no script kept): the change is 6 –
  14 % of each matrix, at a cosine of −0.05 … +0.15 to the weight; its stable rank is 18 – 83
  against 230 – 456 for a random matrix of the same shape; 0.4 – 3 % of it lies in the weight's
  own top-64 × top-64 singular subspace; the norm weights and layer scales shrank by 0.4 – 1.3 %,
  the size AdamW's weight decay gives over the run.

## The LoRA finish

`x2lora_1124_1113` (config in `configs/`): the head frozen, a rank-8 LoRA on its 15 linear layers
(0.32 M values), `x2ovgf`'s recipe at LR 2e-4 for 1,000 rollouts. The user: "rank를 8정도로 낮추고
lr를 올린 다음 1k 스텝 정도로 짧게 해보면 어때?" Code: `scripts/pixel_head/train/_lora.py`; the run
writes a plain head file with the LoRA folded in and the factors as `head_x2_lora.safetensors`.

Why train one instead of cutting the trained change: a cut keeps part of a solution whose parts
lean on one another; a rank-limited run has to find its own balance inside the rank. A W-based
SVD variant (training the weight's singular values, or starting from its top directions) is not
used: the finish's change lies almost wholly outside the weight's own top subspace.

## How the LoRA is read

1. The outcome, by the config's gate: amp, stamp32, grain on flat tokens, tone and the held-out
   dB / DINO beside `x2ovgf` and the top-8 cut, and the user's read of the pair sheets.
2. Where the rank went: |scale · B A| / |W| per layer and stage, and the 8 singular values of each
   B A (how many of the 8 a layer uses).
3. The same directions or others: per matrix, the share of the LoRA update inside the top-8 and
   top-64 singular subspaces of `x2ovgf` − `x2ovll`, left and right, against chance (k / dim).
   High → the full finish's sharpening is what a low rank finds, and the diffuse rest is
   balance; low → a different solution, and the full change's spectrum says little about what is
   needed.
4. The dial: the LoRA folded in at multiplier 0.5 / 1 / 1.5, read like the α strips. With biases
   and norms frozen, whether amp and stamp still rise together.
5. Stage ablation: the LoRA removed from one stage at a time (trunk, up1, up2, out). Cleaner
   than the full finish's stage read, since no bias or norm moved with it.
6. The small search, if 1 – 5 leave a usable file: per layer the update is 8 rank-1 terms; a
   gain per term (120 knobs, or 8 shared across layers) scored on amp up and stamp32 / grain
   down over the cached renders, each candidate a fold on the CPU and a read of a few minutes.
   This is the edit-and-observe loop of the idea at a size a search can cover; the sheets of the
   best few go to the user.
7. A seed twin (20 min) before any direction is called the task's: the principal angles between
   two runs' subspaces say whether the directions repeat.

Steps 2 – 5 need one read script over the factors file (not written) and the existing
`head_merge.py`, `sheet_texture.py` and `tile_dict.py`; 2 and 3 are CPU only.

What each outcome leads to: a low-rank finish with its own balance → a full-length run at that
rank, and the LoRA as a possible form of the sharp file (a multiplier in place of a second
file; the form is the user's call). Sharp but off balance like the cut → rank 64, or the
finish stays a full finetune and step 6 is dropped. Neither stamp nor the fixed tile on halftone
is expected to go by rank alone; the unpaired discriminator (`reports/x2_unpaired_2026_11_24.md`) is the step
aimed at them.

## A Jacobian lens on the head (2026-11-25)

The user (2026-11-25): "픽셀 디코더 x2 있잖아 jlens 쓰면 직접 수정 등 가능성이 있으려나?", then
"그냥 아이디어만 거기 써놓자". Nothing below has been run or queued.

The construction is JLD's (arXiv:2610.05967), as `project/cjk_anima_reseed/_archive/idea2.md`
(local) took it for the rows: an early representation read through the sensitivity of a late
output, M = E[JᵀJ], fitted without a target from random output probes (g = Jᵀv, E_v[g gᵀ] = JᵀJ).
Here the early representation is a hidden layer of the second stage and the output is the ×2
image.

What is different from the rows' probe (`_archive/reports/probe_jl_2026_10_07.md` there: 3.85 s
an item through the DiT in fp32, A / B overlap 0.21 – 0.50 against a bar of 0.8):

- The head runs alone on cached features, in fp32, with no DiT in the graph.
- The pattern has a linear definition at the output. The token-periodic and the 8-px-periodic
  parts of the high-pass are what `tile_dict.py` and the fold already measure. A pixel probe
  split into that part and the rest of the high-pass gives M_pat and M_rest, disjoint as the
  rows' in-box / out-box masks were.
- `out2` is linear from `norm2`'s output to the 16 × 16 patches, so at the last layer its
  weight is the Jacobian: each hidden direction's patch template under the window can be read
  on the CPU with no probe.

The edit it would give:

- The generalised eigenproblem M_pat u = λ M_rest u ranks hidden directions by how much they
  move the pattern against how much they move the rest of the high-pass.
- The top directions damped by γ and folded into `out2` (or into the output side of the
  `up2_blocks`): a head file written in seconds, as `head_merge.py` writes an α, with γ as the
  setting. No training.
- The same directions with a gain each are a space of tens of knobs, the size the first
  section of this file asks for.

How it differs from what was read above: α, the stage cut and the singular cut pick directions
by the size of the weight change, and amp, stamp32 and grain moved together under each. The
lens picks them by their effect on the output, so it is the first read of whether sharpness and
the pattern separate in a linear subspace at all.

What each outcome would lead to:

- λ on the held-out fit well over tr(M_pat) / tr(M_rest), the top directions repeating between
  two fits → γ strips of the fold on castle s0, the sky patch and neon_rain, read on amp, the
  8-px fold and the tile share, and the pair sheets to the user.
- λ flat → sharpness and the pattern ride the same directions; no fold of this kind exists,
  and that would be the reason every GAN setting moved along one line (`x2_plan.md`, Next 2).

Limits:

- A subspace is content-independent. The 8-px grid on flat colour (the GAN finish's) is the
  likeliest to sit in one; the token tile on fine colour texture is open. The token tile on
  halftone (the B head's) replaces the dots with a fixed tile, so removing its directions is
  expected to leave blur where the dots were.
- The lens is a linearisation, and the paper warns that optimising against it uses its null
  space. At `out2` the fold is exact; a fold at an earlier layer passes through a norm and a
  block and is checked on renders.

First step, if taken: the read of `out2`'s templates (CPU); a lens at `norm2`'s output fitted
twice on disjoint halves of the cached renders; the held-out λ spectrum and the overlap of the
two fits; then the γ strips. Estimated 10 – 20 min on the GPU, not measured.

Run on 2026-11-26 in another form (`x2lens_1126_2225`, `reports/x2_lens_2026_11_26.md`). Two
points of the section above did not hold. `norm2`'s output reaches the image through `out2` and
the overlap-add, both linear, so nothing is fitted by probes at this layer. And a single
position's patch is 16 px wide, so its 32-px-periodic part is the patch itself: M_pat is
proportional to M for the token tile in every direction, and a flat λ from the probes would have
said nothing about it. The read ranked directions by how the hidden values repeat per phase of
the token instead. Those directions exist and are the head's own position code; damping them
raises the patterns. The edit that worked is of the other kind, completing `out2`'s cancellation
of the 8 × 8 tile on the flat tokens' hidden directions.
