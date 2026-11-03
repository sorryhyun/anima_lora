# Adapter EMA 0.97: lr 1e-4 without the off-prompt drift

Measured 2026-10-25 on MOSS-Image Base and SFT. Follows
[muon_no_channel_scaling.md](muon_no_channel_scaling.md) (Muon, kaiming, α 0) and answers
its "Muon at lr 1e-4" item. Runs: `output/ckpt_moss/caststation_ema097{,_lr1e4}` (the EMA)
and `…/caststation_ema097{,_lr1e4}/…_raw.safetensors` (the live weights of the same run);
grids: `output/tests/ab_caststation_ema097{,_raw,_lr1e4,_lr1e4_raw}_50st`.

New default: **`ema_decay` 0.97** (t2i). lr stays 5e-5. Since replaced by `ema_window`, the
span as a fraction of the run's updates, default 0.1 for t2i and i2i (0.958 here; 0.14 would
be this 0.97). The i2i point measured so far is 0.97 at 600 updates (span 0.055; sanitize
`tone_w0002_ema`, where the EMA edited cleaner than the raw weights of the same run).

## Setting

`channel_(caststation)_layout` (30 images), 8 epochs = 240 updates, batch 1, seed 0, rank
32 / alpha 128, Muon, kaiming, clip 1, warmup 5 %, constant lr after warmup. The EMA
(`--ema_decay`, `library/moss/training/ema.py::ema_step`) starts at the end of warmup, lerps an
fp32 CPU shadow once per update, and is what the main file holds; the run's raw weights go
to `<name>/<name>_raw`, so each run gives both arms on the same trajectory. 0.97 averages
~33 updates, about one epoch here.

Sampled by `lora_ab.py` (3 prompts × seeds 1, 2 × Base and SFT, 50 steps, CFG 3).

## Measured

ΔW = up·down per adapted linear (252), EMA against raw of the same run:

| lr | ‖ΔW_EMA‖ / ‖ΔW_raw‖ (median, range) | cosine (median, min) | final-epoch loss | min |
|---|---|---|---|---|
| 5e-5 | 0.890 (0.879–0.899) | 0.981 (0.977) | 0.2847 | 7.9 |
| 1e-4 | 0.897 (0.878–0.907) | 0.981 (0.975) | 0.2833 | 8.0 |

The EMA is ~11 % shorter (its ~32-update lag over a steadily growing ΔW) and 2 % off in
direction. Neither the loss nor the step time tells the arms apart; the shadow costs no
VRAM and the step time stayed at 1.94–2.01 s.

## Images

- **lr 5e-5: the direction matters more than the length.** Raw ×0.89 against raw ×1.0
  differs by 6.6–19.8 mean pixel distance (0–255) over the 12 cells; EMA ×1.0 against
  raw ×0.89 by 10.2–44.3. Scaling raw down changes strength only; the EMA changes
  expression and outfit details (wink, vest, phone) at the same style.
- **lr 1e-4: raw drifts off the prompt, the EMA does not.** Neru p1 s1, prompt
  `simple background`: raw paints a solid green (Base) / yellow (SFT) background and adds
  an off-shoulder jacket and a sticker phone; the EMA keeps the plain background and the
  outfit of the 5e-5 EMA (EMA–raw distance 62.1 / 58.7, the two largest cells). Base p2
  (sailor uniform) raw also moves outfit and palette away from both EMAs.
- **lr 1e-4 EMA sits next to the 5e-5 EMA.** Composition and outfits mostly match the
  lr 5e-5 EMA (distance 16–44) while carrying the higher lr's style; user read: the drift
  the earlier 1e-4 runs had is gone.

## Reading

With constant lr and batch 1, the last steps' weights carry the σ / sample noise of the
last few updates; MOSS's σ sampler puts the median σ at ~0.79, where the target is mostly
noise. The 0.97 EMA averages that over about an epoch. At 5e-5 that noise is small enough
that the EMA only resamples details; at 1e-4 the per-step excursions are large enough to
pull single cells off the prompt, and the average removes them.

## Not tested

- A second training seed or dataset: one 30-image set, one seed, 12 cells by eye; most of
  the 1e-4 separation is two cells (Neru p1 s1 on Base and SFT).
- A raw checkpoint ~30 updates before the end, to split "average" from "earlier point on
  the trajectory".
- Other decays (0.99 at 240 updates keeps ~9 % of the warmup-end seed) and longer runs.
- i2i: pinned off in `configs/moss/i2i/*/train.toml` until an i2i arm reads it.
- lr 1e-4 + EMA as the default lr.
