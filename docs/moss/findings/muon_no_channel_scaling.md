# Under Muon: channel scaling does nothing once `down` starts kaiming

Measured 2026-10-25 on MOSS-Image Base. Follows [muon_channel_init.md](muon_channel_init.md)
(Muon, α 0.75, kaiming) and closes its "Muon without channel scaling" item. Runs:
`output/ckpt_moss/caststation_lay08p02_at_muon_nodc_cs075_kaiming` (α 0.75) and
`…_cs0_kaiming` (α 0); side-by-side columns:
`output/tests/ab_caststation_muon_alpha/cols_cs0/` (none / svd α 0.75 / kaiming α 0.75 /
kaiming α 0). Weight stats: `_archive/bench/moss_channel_stats/lora_weight_stats.py`.

New default: **Muon, lr 5e-5, `down_init` kaiming, `channel_scaling_alpha` 0, `dc_isolate`
off.** The LoRA trainer no longer reads a calibration file by default.

## Setting

As [muon_channel_init.md](muon_channel_init.md): `channel_(caststation)_layout`, 20 epochs =
600 samples, seed 0, rank 32 / alpha 128 on q/k/v/o/ffn, lr 5e-5, clip 1, Muon,
`dc_isolate` off, kaiming `down`. The two arms differ only in `channel_scaling_alpha`.
Sampled by `lora_ab.py --arms er_sde:1.0` (3 prompts × seeds 1, 2 × Base and SFT, 50 steps,
CFG 3).

| column | run | α | min |
|---|---|---|---|
| kaiming α 0.75 | `…_cs075_kaiming` | 0.75 | 20.3 |
| kaiming α 0 | `…_cs0_kaiming` | 0 | 20.1 |

## Measured

Weight space, mean over the 252 adapted linears; `weight_svd` α 0.75 from the earlier doc
for scale:

| | ‖ΔW‖_F | stable rank (of 32) | top direction's energy | energy on DC input | ‖ΔW x̄‖ | ch 42 share, FFN input |
|---|---|---|---|---|---|---|
| svd α 0.75 | 1.151 | 18.5 | 5.5 % | 0.45 % | 1.24 | 27.8 % |
| kaiming α 0.75 | 1.151 | 18.5 | 5.5 % | 0.03 % | 0.43 | 6.8 % |
| kaiming α 0 | 1.147 | 18.5 | 5.5 % | 0.05 % | 0.49 | 7.3 % |

- **The two kaiming arms are the same run to three decimals.** Paired by epoch (same seed,
  same σ draws) the loss differs by at most 0.0005 over the 20 epochs; |g| per 30 steps
  moves in the same 0.03–1.1 range; 20 min and 11.77 GB peak on both.
- **ch 42's share follows the init, not α.** 7.3 % without scaling is the uniform-init
  level (7.0 %, [dcfree_svd_init.md](dcfree_svd_init.md)); α 0.75 takes it to 6.8 %. The
  27.8 % under `weight_svd` came from the init's ch 42 column, which scaling never
  touches (α only reaches the update of `down`, 14–20 % of the init). Under Muon the
  update does not grow the loud channel's share on its own.
- **Output DC shift** 0.49 vs 0.43: both a third of `weight_svd`'s, both within the
  seed-to-seed range this metric is likely to have (one seed measured).

## Images

`cols_cs0/`, one seed each, judged against the no-LoRA column. Style carry (flat pale colour,
thin clean line, small highlights) is equal across the three LoRA columns; none adds
artifacts; all still write the trigger as text on Miku's name plate.

- **α 0 keeps the no-LoRA composition most often.** Base p1 s2 keeps the white shirt and
  striped tie (α 0.75: black sweater and bow); SFT p2 s2 keeps the red neckerchief
  (α 0.75: blue); SFT p1 s1 and Base p1 s1 keep the vest and phone pose (α 0.75 adds a
  black hoodie). This is the composition behaviour [dcfree_svd_init.md](dcfree_svd_init.md)
  credited to `svd_drop_loud 1`, here without that arm's hand distortion: every p0 peace
  sign is a clean V on both arms.
- Base p2 s1 is the one cell where α 0 moves further from the base (wink, hair flower).

## Reading

Channel scaling was built for AdamW's per-element step ([muon_channel_init.md](muon_channel_init.md)
has Muon's). Under Muon the measured update on ch 42 stays at the init's share whether or not it is
rescaled: with a kaiming init there is no loud column to begin with,
and the orthogonalised update does not create one. The only thing α did under Muon was
compensate for `weight_svd`'s ch 42 column, which kaiming removes at the source. Dropping
it also drops the trainer's dependence on `channel_stats.safetensors`, a calibration taken
on the t2i forward and never re-taken with reference tokens present (i2i).

`channel_scaling_alpha 0.5` stays the right setting for `--optimizer adamw`
(`docs/moss/channel_scaling.md`).

## Not tested

- A second seed or dataset: one 30-image set, one training seed, as every arm in this
  series.
- The same pair on an i2i cache. The i2i LoRAs shipped so far trained with the older
  default (`weight_svd`, α 0.5, `dc_isolate` on); the 1-hour colorize arm
  (`project/finished/moss_lora/README.md`, v3 basis) is the place to read Muon / kaiming / α 0 against
  luma r, chroma error and per-glyph lettering.
- Muon at lr 1e-4 without scaling (the lr where plain AdamW LoRAs went off-prompt).
