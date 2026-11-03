# Under Muon: a `weight_svd` init without the input DC and the loud channel

Measured 2026-10-25 on MOSS-Image Base. Follows [muon_channel_init.md](muon_channel_init.md),
whose kaiming arm is the shipped default; tests
`_archive/proposals/lora_dcfree_svd_init.md`. Runs:
`output/ckpt_moss/caststation_lay08p02_at_muon_nodc_cs075{,_kaiming,_svdnodc,_svdnodc_loud1}`;
side-by-side columns: `output/tests/ab_caststation_muon_alpha/cols_nodc/` (none / svd α 0.5 /
svd α 0.75 / kaiming / svd-nodc / svd-nodc-loud1). Weight stats:
`_archive/bench/moss_channel_stats/lora_weight_stats.py`; init probe:
`output/tests/loud_init_probe.py`.

The default stays kaiming. `svd_drop_loud 1` gives the lowest output DC shift and no
loud-channel path, and kept the no-LoRA composition best, but hands came out more distorted.

## The inits

- `--down_init weight_svd_nodc`: `down` = top-r right singular vectors of `W(I − ûûᵀ)`,
  `û = x̄ / ‖x̄‖` from `channel_mean.safetensors`, / √3. `down · x̄ = 0` at init; nothing
  holds it there afterwards.
- `--svd_drop_loud k`: also projects out the k input channels with the largest calibrated
  `mean|x|` (`channel_stats.safetensors`), so their `down` columns start at zero.

## Setting

As [muon_channel_init.md](muon_channel_init.md): `channel_(caststation)_layout`, 20 epochs =
600 samples, seed 0, rank 32 / alpha 128 on q/k/v/o/ffn, lr 5e-5, clip 1, Muon,
`dc_isolate` off, channel scaling α 0.75. The arms change only the init. Sampled by
`lora_ab.py --arms er_sde:1.0` (3 prompts × seeds 1, 2 × Base and SFT, 50 steps, CFG 3).

| column | run | `down_init` | min |
|---|---|---|---|
| svd α 0.75 | `…_cs075` | `weight_svd` | 20.6 |
| kaiming | `…_cs075_kaiming` | kaiming | 20.3 |
| svd-nodc | `…_cs075_svdnodc` | `weight_svd_nodc` | 20.3 |
| svd-nodc-loud1 | `…_cs075_svdnodc_loud1` | `weight_svd_nodc`, `svd_drop_loud 1` | 20.9 |

## Only one channel is loud

Calibrated `mean|x|` per input channel against the target's median, blocks 11–41:

- **ffn_up_gate**: ch 42 at 11–26×, up to 12 % of Σ|x|², top-3 in 33 of 42 blocks. Next
  are ch 2807 / 1317 / 330 / 1356 at 7–10×, about 1 % each.
- **q / k / v**: 4–9×, and the top channel changes from block to block. ch 42 is not loud
  here, so the FFN's pre-norm is what lifts it.
- **o**: 3–6×. **ffn_down**: up to 12×, except block 41 (38×, six channels over 20×).

Init only, every third block 11–41. lev² = the channel's share of `(‖down[:, c]‖ · mean|x[c]|)²`;
"uniform" is a kaiming-like init with equal column norms:

| ffn_up_gate | capture of the `weight_svd` basis | ch 42 lev² | top-5 lev² |
|---|---|---|---|
| `weight_svd` | 100 % | 33.9 % | 41.6 % |
| nodc | 99.4 % | 23.8 % | 32.5 % |
| nodc + ch 42 | 98.6 % | 0 % | 13.8 % |
| nodc + every channel over 5× | 98.1 % | 0 % | 0.2 % |
| uniform | — | 7.0 % | 11.0 % |

With ch 42 dropped the next four channels sit near the uniform level, so `svd_drop_loud 1`
is enough. On q, o and ffn_down, nodc is already near uniform and the extra projection
costs 0.1–0.2 pp of capture.

## Measured

Weight space, mean over the 252 adapted linears:

| | ‖ΔW‖_F | stable rank | energy on DC input | ‖ΔW x̄‖ | ch 42 share, FFN input |
|---|---|---|---|---|---|
| svd α 0.75 | 1.151 | 18.5 | 0.45 % | 1.24 | 27.8 % |
| kaiming | 1.151 | 18.5 | 0.03 % | 0.43 | 6.8 % |
| svd-nodc | 1.145 | 18.5 | 0.01 % | **0.32** | 22.9 % |
| svd-nodc-loud1 | 1.141 | 18.3 | 0.02 % | **0.33** | **≈ 0 %** |

- **DC**: projecting x̄ out of the init is enough. The gradient does not refill the DC
  direction over 600 steps, and both arms end below kaiming.
- **ch 42**: the share follows the init. The Muon `down` update stays 14–20 % of the init
  ([muon_channel_init.md](muon_channel_init.md)), so nodc keeps most of `weight_svd`'s
  share, and a column that starts at zero stays near zero (α 0.75 also holds ch 42's update
  leverage at 1.6× the mean channel's).
- **Loss**: paired by epoch, svd-nodc − kaiming averages +0.00002 and svd-nodc-loud1 −
  kaiming −0.00013, all within ±0.0016. Loss does not separate the four inits.
- **|g|** (median per 30 steps): `weight_svd` 0.25–0.58, kaiming 0.10–0.29, svd-nodc
  0.13–0.32. The three move together epoch by epoch (same seed, same σ draws). Under Muon
  |g| does not set the step size.

## Images

Judged by eye by the user:

- **svd-nodc-loud1** changes the no-LoRA composition least while carrying the set's style
  (flat pale colour, thin clean line). On SFT p2 s2 it keeps the red neckerchief, collar
  ornament and framing (kaiming turns the neckerchief blue). On Base p1 s2 it keeps the
  white shirt and striped tie (kaiming: black sweater; svd-nodc: yellow bow and vest).
- **Hands are worse** with svd-nodc-loud1 (e.g. Base p0 s1 turns the V sign into a point,
  SFT p1 s1's fingers on the phone).
- All arms still write the trigger as text on Miku's name plate.

## Not tested

- A second seed or dataset. The hand distortion may be seed noise.
- `svd_drop_loud 1` at α 0.5, which frees ch 42's update leverage (6.9× the mean channel's)
  while keeping the zero init. If the hands recover there, the cause is ch 42 being shut
  twice (zero init, then a damped update).
- A ch 42 column started at the mean column norm instead of zero (≈ 7 % share, as kaiming).
