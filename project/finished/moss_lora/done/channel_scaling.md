# Channel scaling on MOSS-Image

Shipped 2026-10-18, removed 2026-10-25: under Muon with a kaiming `down` init the trained LoRA is
the same with or without it. The last commit with the code is `f21bc206`. This file is the log.

**Shipped recipe:** `channel_scaling_alpha` 0.5 + `dc_isolate` on, lr 5e-5, rank 32 /
alpha 128 (`weight_svd`). α is passed only for `algo = lora`. Revert path
`--no-dc_isolate --lr 1e-4`. Calibration: `library/moss/calibration/channel_stats.safetensors`
(α) and `channel_mean.safetensors` (signed token mean `x̄`, `dc_isolate`), both from
`scripts/calibration/moss_channel_stats.py` on Base. Code `library/lines/adapters.py`
(shared with qwen21; scale on the down weight, inert at α 0).

## Findings

**Massive channels reach the LoRA inputs** (`_archive/bench/moss_channel_stats/results/20261018-1120-base`,
Base, `_calib_mixed` 16 × σ {0.1 … 0.9}). The massive channels (1356, 42, …) hold ~99 % of
the residual's DC (`dave.md`); after the norm they arrive at ≤ 26× median (from ~1900×). The one that
matters is ch 42 into `ffn_up_gate` (top channel on blocks 11–41, 11→26×). Leverage α 0 →
0.5: q 4.0–18.7 → 2.0–4.4, o 2.6–6.2 → 1.6–2.7, ffn_up_gate 9.3–22.8 → 3.1–4.9, ffn_down
4.5–34.8 → 2.2–6.2. Weights alone: `norm1.weight[1356]` ≈ 0.000–0.03 on blocks 5–20 (attention
mostly drops it), the shared adaLN gives ch 1356 / 42 shift −6.8 / −4.3 at the FFN input.

**A/B at lr 2e-5** (`caststation_layout_at_r32a128_lr2e5_svd` vs `…_cs05`, cache
`channel_(caststation)_layout`, 20 epochs, seed 0; grids
`output/tests/ab_caststation_layout_at_r32a128_lr2e5_svd{,_cs05}_50st`, Base + SFT × 3
prompts × seeds 1, 2). Loss equal to the third decimal; ch-42 column drift / mean column on
ffn_up_gate blocks 15 / 25 / 35: 0.99, 0.61, 0.92 → 0.25, 0.15, 0.17. Style strength the
same; cs05 closer to the no-LoRA image in 9 / 12 cells (mean pixel distance 49.4 → 43.7).
User read: it touches less of what the style does not need.

**A/B at lr 1e-4** (`caststation_layout_at_r32a128_lr1e4_svd` vs `…_cs05`). Losses within
±0.005; distance to no-LoRA flat (50.7 vs 51.2), which measures how much changed, not
whether it is right. **User verdict: cs05 much better.** α 0 adds off-prompt content
(glasses + garbled sign, a street instead of the park, a broken hand); cs05 stays on-prompt
at the same style strength. Defaults flipped to α 0.5, lr 1e-4.

**Up-gradient decomposition** (`_archive/bench/moss_channel_stats/probe_up_gradient.py`,
`results/20261018-1355-upgrad-fftref`). Channel scaling does not touch `∂L/∂up` (`h = down·x`
is unchanged), and that is where the bias learner lives: medians, LoRA vs full fine-tune:

| | q | o | ffn_up_gate | ffn_down |
|---|---|---|---|---|
| rank-1 fraction | 0.86 vs 0.46 | 0.49 vs 0.39 | 0.82 vs 0.58 | 0.61 vs 0.34 |
| DC share | 0.89 vs 0.39 | 0.38 vs 0.29 | 0.79 vs 0.53 | 0.51 vs 0.24 |
| effective channels | 484 vs 1268 | 123 vs 1084 | 51 vs 320 | 133 vs 1393 |

About half the DC is in the loss itself (FFT DC share 0.4–0.5 on q / ffn_up_gate); the
rank-32 projection concentrates the rest, diluting the non-DC share 2–5× against FFT. So
the lr 1e-4 benefit of α 0.5 is on the `down` side alone.

**DC isolation** (`--dc_isolate`; probe `results/20261018-1407-upgrad-dc`). Row 0 of `down`
:= `x̄/‖x̄‖`, rows 1–31 orthonormalised ⊥ it and re-projected after every step; composes with
α 0.5. DC share of `∂L/∂up[:, 1:]`: q 0.89 → 0.05, o 0.38 → 0.10, ffn_up_gate 0.79 → 0.12,
ffn_down 0.51 → 0.09; `up[:, 0]` takes 90–97 % of the gradient energy. Costs 1.99 vs 1.84
s/step.

- `…_lr1e4_svd_cs05_dc` (α 0.5 + dc, lr 1e-4): loss within ±0.002 of α 0.5, distance 50.0
  vs 50.0, but **text artifacts in 3 / 12 cells** (the artist tag written in the sky, garbled
  hair-accessory text; Claude's read). Hypothesis: the centered gradient learns faster and
  over-fits at lr 1e-4.
- `caststation_layout_at_r32a128_lr5e5` (same, lr 5e-5): loss 0.2207 at epoch 20, distance
  43.7, **no text artifact in 12 / 12**, style close to the lr 1e-4 arms (Claude's read).
  Defaults flipped on this grid (user decision): `dc_isolate` on, lr 5e-5.

## Open

1. **α 0.5 alone at lr 5e-5**: the control the `dc_isolate` flip went ahead without; it
   separates "dc at half the lr" from "half the lr". Then a second seed, and α 0 +
   `dc_isolate` to separate the down side from the up side.
2. **α sweep**: 0.25 / 0.5 / 1.0.
3. **Calibration**: sample saturation (re-probe on `sincos_layout`, compare `s`) and SFT
   (`--model sft`); both 2-min jobs, low priority.
