# MOSS self-distillation: the arms of 2026-11-09 and the add90 ema97 verdict

Closed 2026-11-09 on channel (caststation), 30 images. The recipe it settled on is
`configs/moss/selfdistill.toml`; user doc [`experimental/self_distill.md`](../experimental/self_distill.md).
What a longer run does to it, and the bounded forms that follow, are in
[`proposal/self_distill_lora.md`](../proposal/self_distill_lora.md).

The method: the student (the LoRA on the cached caption embeds) regresses on the same DiT's
velocity at the same noisy latent, with the training image through the TE's vision tower ahead
of the caption. Off-policy arm of D-OPSD (arXiv 2605.05204, § 3.3).

## Implementation

- **Teacher cache** (`library/moss/distill.py`, `scripts/moss/distill_cache.py`,
  `DistillCacheRequest`): `{stem}.moss_vis.safetensors` beside each TE file, one `embeds_<kind>`
  per caption the student can draw (`full`, `plain`, `v{k}`, `empty` = the image alone). The page
  is prefilled once and every caption continues from it (`vision_te.encode_variants`). Image at
  the latent's size (~1,000 vision rows at 1 MP). Same TE weights as the student cache (int8
  decoder, vision tower bf16). `--tail` sets the teacher's prompt tail (default `off`),
  independent of the student cache's. Stamped with captions, source digest, encoder and tail;
  re-encodes only what changed. 30 images take about a minute after loading.
- **Train loop** (`library/moss/train.py`, `training_loss(teacher=…)`):
  - `--distill off|base|ema`. `base`: multiplier 0 during the teacher forward. `ema`: a shadow of
    the adapter params at `--distill_ema_decay` (default 0.9999), swapped in through one scratch
    copy (`TeacherWeights`), independent of the saved `ema_window` EMA.
  - The teacher forward runs first, no grad, on the same `x_t` / timestep, cond branch only.
  - `--distill_mix λ`: target `λ·(ε − x₀) + (1 − λ)·u_t`; on the shared `x_t` this equals
    `λ·FM + (1 − λ)·distill` up to a constant.
  - `--distill_add w`: on teacher steps keep the flow MSE and add the teacher MSE (one student
    forward, two targets), scaled each step by `w·sqrt(FM / teacher)` so at `w = 1` both terms'
    gradients on the prediction have the same norm (unscaled, the teacher MSE is ~1/40 of FM and
    its gradient ~15 %). Equivalent to `distill_mix` with a per-step λ; Muon normalises the
    update, so what matters is the mix, not the sum's scale. Logged as `t`.
  - `--distill_extra_sigma s` / `--distill_extra_weight w`: a second term every step, the teacher
    target at a σ drawn uniformly from [s, 1] on the same sample, balanced as above (one more
    no-grad and one grad forward, 2.5 s/step). Its σ and noise come from their own RNG streams, so
    the main term draws what a run without it draws. Logged as `x`, `x_fm`, `x_sigma`, `x_scale`.
  - `--distill_sigma_min s`: teacher only on steps with σ ≥ s, the flow target below (no teacher
    forward there). `--distill_sigma_ramp r`: the teacher's share rises linearly from 0 at
    `s − r` to 1 at `s`.
  - The flow MSE is logged as `fm` on teacher steps. The run JSON counts `distill_steps`.
  - The compile range spans the teacher's longer token axis (caststation: 4192–5613).
  - Refused with `delta_rows`, `trigger_registers` and `ref_adapters all`: their row masks are
    built for the student's sequence, and the teacher's has the image rows ahead of the caption.
- **Gate probe** `bench/moss_self_distill/xhat_probe.py`: one-step flow loss and x̂₀ by σ under
  `cap` / `img_cap` / `img_cap_sure` / `img`, Base, no LoRA, plus full renders per conditioning.
- Tests: `tests/test_moss_distill.py` (teacher weight swap, target replacement, `distill_add`
  balance, the extra term's σ draw and balance).

Cost (caststation, int8 Base, fp32 stream, 30 images ~4,100 image tokens): 1.45 s/step with a
teacher forward every step against 1.10 s for SFT; peak 13.20 GB against 13.19 GB. A σ-gated run
pays the teacher forward only on its teacher steps (hybrid95: 1.08 s/step).

## Probe on channel (caststation), 2026-11-09

`bench/moss_self_distill/results/20261109-1638-caststation/`, 30 images, int8 Base, fp32 stream:

| σ | 0.1 | 0.25 | 0.4 | 0.6 | 0.8 | 0.95 |
|---|---|---|---|---|---|---|
| FM loss, `cap` | 0.4735 | 0.2661 | 0.1957 | 0.1747 | 0.2323 | 0.4576 |
| FM loss, `img_cap` (tail off) | 0.4726 | 0.2653 | 0.1952 | 0.1739 | 0.2306 | 0.4402 |
| images with `img_cap` lower | 100 % | 100 % | 100 % | 90 % | 97 % | 100 % |
| ‖v_img_cap − v_cap‖² | 0.0041 | 0.0026 | 0.0018 | 0.0018 | 0.0029 | 0.0187 |

- `img_cap` is lower at every σ on nearly every image; the gap is ~10× larger at σ 0.95 than below
  0.8. x_t pins most of the image below 0.8, so the one-step gap there is small by construction.
- x̂₀ spread across ε is the same for both (0.0382 vs 0.0377 at σ 0.8).
- Image alone (`img`) is worse than `cap` below σ 0.95; the `sure` tail on the teacher adds nothing.
- Renders (CFG 3, 50 steps): `img_cap` brings back the composition, outfits, props and side
  doodles, and draws far cleaner figures than `cap`. Line weight and the pale flat palette stay
  closer to Base's.

## Runs on channel (caststation)

`channel_(caststation)_layout_sure` cache, `configs/moss/lora.toml` with lr 1e-4 unless noted,
`ema_window 0.139`, 240 steps, seed 0. The σ sequence is the same in every run, so a σ-gated arm
differs from SFT on its teacher steps only. Renders: `scripts/moss/generate.py` (int8, CFG 3, 50
steps), the three `lora_ab.py` caststation prompts, seeds 1–6, sheets in
`output/tests/caststation_distill_ab/`.

| LoRA (`output/ckpt_moss/caststation_*`) | target | teacher steps | FM loss, last 60 steps |
|---|---|---|---|
| `fwdpv` (2026-11-03) | FM | 0 | 0.2827 |
| `distill_base` | teacher (Base) | 240 | 0.2910 |
| `distill_base_lr2e4` | teacher (Base), lr 2e-4 | 240 | 0.2950 |
| `hybrid95` | teacher at σ ≥ 0.95, FM below | 19 | 0.2811 |
| `hybrid90` | teacher at σ ≥ 0.9, FM below | 50 | 0.2836 |
| `ramp85_95` | teacher share 0 at σ 0.85 → 1 at 0.95 | 85 (Σ share 49.5) | 0.2831 |

| `h90x90_w1` | hybrid90 + extra teacher term σ ∈ [0.9, 1] every step, w 1 | 50 + 240 | 0.2859 |
| `h90x90_w05` | the same, w 0.5 | 50 + 240 | 0.2850 |
| `sftx90_w1` | FM every step + extra teacher term σ ∈ [0.9, 1], w 1 | 240 | 0.2861 |
| `add90_ema97` | FM every step + teacher term (`distill_add 1`) at σ ≥ 0.9, EMA 0.97 teacher | 50 | 0.2853 |

- Pure distillation's loss sits at 0.005–0.008 throughout. Muon normalises each matrix's update,
  so the loss scale does not set the step size; the LoRA's ΔW came out at ~90 % of SFT's.
- hybrid90 beats SFT by eye (sheet `sheet_hybrid90.png`): cleaner figures and hands, same palette.
- More Base teacher (the extra-term arms, 240 teacher terms) buys still cleaner figures but pulls
  the style to Base: saturated cyan and yellow, glossy hair, outfits drifting from the prompt
  (s3 bow and legwear, s6 summer uniform). w 0.5 sits between. Where the main term's 50 high-σ
  steps go (teacher or FM) makes no visible difference (`sheet_sftx90.png`).
- add90 ema97 keeps the palette flattest of all arms (the teacher carries the LoRA's style), with
  two odd cells at 240 steps (s4 grey-shadowed face, s6 winter uniform; `sheet_add90.png`).
- On the Space Blue Archive prompt (`bench/moss_ab/space/004`, held out, 12 characters) hybrid90
  keeps Base's layout almost exactly with caststation's palette; SFT rearranges the figures;
  add90 sits between (`space004/sheet.png`).

### 480 steps (16 epochs), 2026-11-09

Same recipe, `--epochs 16`. Renders in `output/tests/caststation_distill_ab/e16/`.

| LoRA (`caststation_e16_*`) | teacher steps | FM loss, last 120 steps | s/step |
|---|---|---|---|
| `sft` | 0 | 0.2472 | 1.07 |
| `hybrid90` | 101 | 0.2488 | 1.13 |
| `add90ema97` | 101 | 0.2509 | 1.15 |

- caststation prompts (`sheet.png`): all three carry the style further; SFT's style is the
  strongest, add90's a little lighter. SFT smears the phone hand (s5) and adds a stray flower
  crown (s6); hybrid90 breaks the bench and pose (s3) and puts a branch across the face (s6).
  add90 has no broken cell, and the two general prompts (s3, s6) come out clean.
- Space 004 (`sheet_space004.png`): SFT moves further from Base's composition than at 240 steps;
  hybrid90 and add90 both keep Base's layout.

### Verdict, 2026-11-09

**add90 ema97 is the working recipe.** Against SFT it gives up a little style strength and is
much steadier on general prompts (s3, s6) and on held-out multi-character prompts, where the
high-σ teacher term holds Base's composition. The EMA teacher is what avoids the Base teacher's
style pull: a 0.97 copy of the LoRA (~33 updates behind) with the image in context teaches
composition while the flow term and the teacher both carry the style. One seed per prompt on
one artist; transfer to other sets is open.
