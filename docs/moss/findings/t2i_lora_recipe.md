# How the t2i LoRA defaults were set: lr, clip, Muon, init, EMA

Measured 2026-10-24 and 25 on MOSS-Image Base (renders on Base and SFT), judged by eye. Each
section changed the default the next one starts from. The result is today's
`configs/moss/lora.toml`: **Muon, lr 5e-5, clip 1, `down_init` kaiming, adapter EMA
(`ema_window` 0.1)**. Channel scaling and `dc_isolate`, both in play below, were removed on
2026-10-25 (`project/finished/moss_lora/done/channel_scaling.md`).

## Common setting

One dataset, `channel_(caststation)_layout` (30 images, layout-JSON captions,
`plain_caption_rate` 0.2), seed 0, LoRA rank 32 / alpha 128 on q/k/v/o/ffn. Runs:
`output/ckpt_moss/caststation_*`. Each LoRA was sampled by `bench/moss_ab/lora_ab.py --arms
er_sde:1.0` (3 prompts × seeds 1, 2 × Base and SFT, 50 steps, CFG 3) and laid side by side with
`bench/moss_ab/ab_cols.py`. Weight stats: `_archive/bench/moss_channel_stats/lora_weight_stats.py`,
mean over the 252 adapted linears, ΔW = scale · up · down, DC = the calibrated token mean x̄.

Every arm is one training seed on one 30-image set. Arms share the seed, so their σ draws match
and losses pair by epoch; **the loss never separated any two arms here**, and the verdicts are
read on samples.

## 1. lr × grad clip × grad accum (AdamW)

AdamW, `weight_svd` down init, channel scaling α 0.5, `dc_isolate`, 20 epochs = 600 samples.
`max_grad_norm 1e9` is "no clip"; with `grad_accum` the optimizer takes 600 / `grad_accum`
updates. Columns: `output/tests/ab_caststation_clip_sweep/cols/`.

| column | run | lr | clip | grad_accum |
|---|---|---|---|---|
| clip1_lr5e5 | `caststation_lay08p02_at` | 5e-5 | 1 | 1 |
| clip7_lr5e5 | `…_clip7` | 5e-5 | 7 | 1 |
| noclip_lr2e5 | `…_lr2e5_noclip` | 2e-5 | off | 1 |
| ga2_noclip_lr1e4 | `…_ga2_lr1e4_noclip` | 1e-4 | off | 2 |
| ga2_noclip_lr5e5 | `…_ga2_lr5e5_noclip` | 5e-5 | off | 2 |

- **5e-5 / clip 1 is the best of the five.** Neither loosening the clip nor trading lr against it
  beat it.
- **1e-4 without a clip distorts**, even at `grad_accum` 2.
- **5e-5 at `grad_accum` 2 without a clip is a toss-up** with clip 7, and departs most from the
  other columns in content.
- Per-epoch loss of the three no-clip runs agrees to ~0.005 at every epoch and shares the same
  bumps: with one seed the σ draws dominate the curve.

## 2. Muon at the AdamW lr, and `dc_isolate`

Basis: arXiv 2605.10468 (fine-tuning with an optimizer other than the pretraining one disrupts
pretrained knowledge; on a Muon-pretrained model LoRA-Muon beat LoRA-Adam). MOSS was pretrained
with Muon–AdamW (`project/finished/moss_lora/report_notes.md`). `torch.optim.Muon` with
`adjust_lr_fn="match_rms_adamw"`, weight decay 0.01; otherwise section 1's winner. Columns:
`output/tests/ab_caststation_dc_isolate/cols/`.

| | ‖ΔW‖_F | stable rank (of 32) | top direction's energy | energy on DC input | ‖ΔW x̄‖ |
|---|---|---|---|---|---|
| AdamW + `dc_isolate` (`…_at`) | 1.156 | 6.5 | 17.7 % | 1.04 % | 1.98 |
| Muon + `dc_isolate` (`…_at_muon`) | 1.161 | 18.5 | 5.5 % | 1.27 % | 2.57 |
| Muon (`…_at_muon_nodc`) | 1.146 | 18.7 | 5.4 % | 0.45 % | 1.24 |

- **Muon at the AdamW lr moves the adapter by the same amount** (‖ΔW‖_F within 0.5 % on every
  target) over ~3× the rank directions, as in the paper's rank-8 LLM result.
- **Under Muon, `dc_isolate` becomes a bias channel**: the DC row stays on x̄ (cos 0.98; AdamW
  lets it drift to 0.79) and the token-mean output shift doubles. Muon's orthogonalised step
  already caps the near-rank-1 bias part of the `up` gradient, which is what `dc_isolate` did
  for AdamW.
- **Images**: the three carry the set's style equally (flat pale colour, thin clean line) with no
  artifacts; the user judged Muon better than AdamW on composition.

## 3. The `down` init and channel scaling under Muon

Columns: `output/tests/ab_caststation_muon_alpha/cols{,_cs0,_nodc}/`. Runs
`caststation_lay08p02_at_muon_nodc{,_cs075,_cs075_kaiming,_cs0_kaiming,_cs075_svdnodc,_cs075_svdnodc_loud1}`.

**Why channel scaling's α moved under Muon.** Channel scaling was tuned for Adam's per-element
step, where a column's output leverage is `|x[c]|`. Muon orthogonalises the `(32, in)` `down`
update by a left multiply, which keeps the gradient's input-channel proportions, so the loud
channel's lead is closer to `|x[c]|²`. Loud channel's leverage over the mean channel in the update
of `down`, α 0.5 → 0.75: ffn_up_gate 6.9× → 1.6×, ffn_down 4.2× → 1.3×, q 3.2× → 1.3×.

**Only one channel is loud.** Calibrated `mean|x|` over the target's median, blocks 11–41: at
ffn_up_gate ch 42 sits at 11–26× (up to 12 % of Σ|x|²), the next channels at 7–10× and ~1 % each.
q / k / v are 4–9× with the top channel changing per block, so the FFN's pre-norm lifts ch 42.

**`weight_svd` leans on it.** The base weight's top right singular vectors carry ch 42 (34 % of
`h` energy at ffn_up_gate) and the input DC (‖D₀ x̄‖ 4–6× a kaiming init's at the FFN input and q).
That is the bias part of the `up` gradient `dc_isolate` was built to contain. The Muon update of
`down` stays 14–20 % of the init, so whatever the init holds, the trained LoRA holds.

The inits tested besides kaiming and `weight_svd`: `weight_svd_nodc` (top-r right singular vectors
of `W(I − ûûᵀ)`, `û = x̄/‖x̄‖`) and `svd_drop_loud 1` (also projects ch 42 out, so its column starts
at zero). Both are still `TrainRequest` options.

| | α | ‖ΔW‖_F | stable rank | energy on DC input | ‖ΔW x̄‖ | ch 42 share, FFN input |
|---|---|---|---|---|---|---|
| `weight_svd` | 0.5 | 1.146 | 18.7 | 0.45 % | 1.24 | 27.8 % |
| `weight_svd` | 0.75 | 1.151 | 18.5 | 0.45 % | 1.24 | 27.8 % |
| kaiming | 0.75 | 1.151 | 18.5 | 0.03 % | 0.43 | 6.8 % |
| **kaiming** | **0** | 1.147 | 18.5 | 0.05 % | 0.49 | 7.3 % |
| `weight_svd_nodc` | 0.75 | 1.145 | 18.5 | 0.01 % | 0.32 | 22.9 % |
| `weight_svd_nodc`, `svd_drop_loud 1` | 0.75 | 1.141 | 18.3 | 0.02 % | 0.33 | ≈ 0 % |

- **ch 42's share follows the init, not α.** 7.3 % without scaling is the uniform-init level
  (7.0 %); the 27.8 % under `weight_svd` is the init's column, which scaling never reached. Under
  Muon the update does not grow the loud channel on its own.
- **Kaiming with α 0 is the α 0.75 run to three decimals**: loss within 0.0005 over 20 epochs,
  same |g| range, same 20 min and 11.77 GB. With a kaiming init α had nothing left to compensate,
  which is why channel scaling was removed.
- **Projecting x̄ out of the init is enough for DC**: the gradient does not refill it over 600
  steps, and both nodc arms end below kaiming.
- **Images.** All arms carry the style equally, add no artifacts, and write the trigger as text
  on Miku's name plate. α 0.75 kept the no-LoRA pose more often than α 0.5; kaiming α 0 kept the
  no-LoRA composition most often (Base p1 s2's white shirt and striped tie, SFT p2 s2's red
  neckerchief), with clean hands. `svd_drop_loud 1` kept composition as well but distorted hands
  (Base p0 s1's V sign into a point, SFT p1 s1's fingers on the phone), so kaiming won.

## 4. Adapter EMA at lr 1e-4

Muon, kaiming, 8 epochs = 240 updates, warmup 5 %, constant lr. The EMA starts at the end of
warmup and lerps an fp32 shadow once per update; the main file holds it and the run's raw weights
go to `<name>/<name>_raw`, so each run gives both arms on one trajectory. Runs
`caststation_ema097{,_lr1e4}`; grids `output/tests/ab_caststation_ema097{,_raw,_lr1e4,_lr1e4_raw}_50st`.
Decay 0.97 averages ~33 updates, about one epoch here.

| lr | ‖ΔW_EMA‖ / ‖ΔW_raw‖ (median) | cosine (median, min) | final-epoch loss |
|---|---|---|---|
| 5e-5 | 0.890 | 0.981, 0.977 | 0.2847 |
| 1e-4 | 0.897 | 0.981, 0.975 | 0.2833 |

- **At 5e-5 the direction matters more than the length.** Raw ×0.89 against raw ×1.0 differs by
  6.6–19.8 mean pixel distance; EMA against raw ×0.89 by 10.2–44.3. The EMA changes expression
  and outfit details at the same style.
- **At 1e-4 raw drifts off the prompt and the EMA does not.** Neru p1 s1 (`simple background`):
  raw paints a solid background and adds a jacket and a sticker phone on Base and SFT; the EMA
  keeps the plain background. The 1e-4 EMA sits next to the 5e-5 EMA in composition while
  carrying the higher lr's style. Most of this separation is those two cells.
- No VRAM or step-time cost.

Reading: with constant lr and batch 1, the last weights carry the noise of the last few updates
(median training σ ~0.79, where the target is mostly noise); at 1e-4 those excursions pull single
cells off the prompt, and the average removes them.

`ema_decay` 0.97 became the default, since replaced by `ema_window`, the span as a fraction of the
run's updates: 0.1 for t2i and i2i (0.958 at 240 updates; 0.14 would be this 0.97). The i2i point:
0.97 at 600 updates on sanitize `tone_w0002_ema`, where the EMA edited cleaner than the raw weights.
Section 1's lr was kept at 5e-5; the prompt_tail caststation run in
[assistant_tail.md](assistant_tail.md) trained at 1e-4 with the EMA.

## Not tested

- A second training seed or dataset, for every section.
- lr 1e-4 with clip 1 at `grad_accum` 2 (`caststation_ga4_lr1e4_s{0,1001}`, `grad_accum` 4, were
  trained but not compared).
- `lokr` / `loha` under Muon: the trainer keeps them on AdamW.
- `svd_drop_loud 1` with ch 42's update undamped, to tell whether the hands broke from the zero
  init or from shutting ch 42 twice; a ch 42 column started at the mean column norm instead.
- A raw checkpoint ~30 updates before the end, to split "average" from "earlier point on the
  trajectory"; other EMA spans; lr 1e-4 + EMA as the default lr.
- Muon / kaiming on i2i against the older recipe the first i2i LoRAs shipped with.
