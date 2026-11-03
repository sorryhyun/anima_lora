# Channel scaling on MOSS-Image (removed)

**Removed 2026-10-25**, together with `dc_isolate`. Under the Muon default with a kaiming `down`
init the trained LoRA is the same with or without channel scaling, and Muon does `dc_isolate`'s
job on its own. The last commit with the code is `f21bc206` (`absorb_channel_scale`,
`isolate_dc`, `reproject_dc` in `library/lines/adapters.py`, tests in `tests/test_moss_lora.py`).
The calibration files stay for `down_init weight_svd_nodc` and `svd_drop_loud`. Log, per-target
tables and open items: `project/finished/moss_lora/done/channel_scaling.md`.

## The loud channel

MOSS's residual stream carries a handful of massive-activation channels (1356, 42, …) that hold
almost all of the image tokens' DC offset, the same on every prompt and seed
(`docs/moss/inference/dave.md`). RMSNorm and the adaLN modulation damp them but do not remove
them: at the FFN input, channel 42 is the loudest of 3072 channels on blocks 11–41, 11 to 26×
the median, with a peak-to-mean of only 2.7, so it is loud on every token and behaves like a
bias line.

![ch 42 and ch 1356 at the FFN input per block; block 35's loudest channels](assets/channel_scaling_loud_channel.png)

Adam moves every weight by roughly the same amount per step, so through channel 42 one step
moves the adapter output 25× further than through a typical channel, on every token at once.
The cheapest way to lower the loss is then a global shift of the FFN input on every prompt.

## Channel scaling

Each input channel got a scale `s[c] = mean|x[c]|^α` (normalised to mean 1, from a calibration
file), stored as `down · s` and undone in the forward as `down · s⁻¹`. Output and the `up`
gradient stay as they were, and a uniform Adam step then moves the output by `|x[c]|^(1−α)`
instead of `|x[c]|`. α 0.5 shrinks the loudest channel's step 25× → 5× and α 0.75 25× → 2.2×
under Adam. The saved file held `down · s⁻¹`, a plain LoRA, with `channel_scaling_alpha` stamped.
Peak memory, s/step and the loss curve (to three decimals) were unchanged.

Twin runs on caststation (r32 / α128, 20 epochs, `weight_svd`, seed 0), one seed per cell. At
α 0.5 the loudest channel's lead over the average channel drops from 9–23× to 3–5× on every
block, and the `down` update on channel 42 falls from 0.5–1.0× an average column to 0.13–0.25×
(≈ 1/s, s ≈ 3.7–4.8).

- **lr 2e-5**: same style strength by eye; the channel-scaled LoRA stays closer to the no-LoRA
  image in 9 / 12 cells (mean pixel distance 49.4 → 43.7).
- **lr 1e-4**: clearly better by eye. Plain LoRA adds things the prompt never asked for (glasses
  and a garbled-text sign, a street with a car in place of the park, a broken hand pose); the
  channel-scaled twin stays on-prompt at the same style strength. Pixel distance to the no-LoRA
  image does not see this difference.

![lr 1e-4 twins on the four cells named above: no LoRA | α 0 | α 0.5](assets/channel_scaling_lr1e4_ab.webp)

## The `up` gradient and `--dc_isolate`

Channel scaling only reaches the `down` side. The `up` gradient is built from `h = down · x`, and
because every token carries the same DC offset, much of each step's `up` gradient is "shift every
token's output the same way". Measured with the trainer's loss
(`_archive/bench/moss_channel_stats/probe_up_gradient.py`): that bias share is 80–90 % of the `up`
gradient at the FFN and attention inputs, against 40–50 % in the full-fine-tune gradient. Per step
the `up` gradient is ~0.8 rank-1, and channel 42's share of it is 3–6× its share of the full
gradient (the entanglement GraLoRA, arXiv:2505.20355, describes).

`--dc_isolate` made the calibrated DC direction one rank row of `down` and kept the other 31 rows
orthogonal to it (re-projected after every optimizer step), so one `up` column learned the bias.
The bias share on the other 31 columns drops to 5–12 %. At lr 1e-4 it over-fit (the artist tag
rendered as text in 3 / 12 cells); at lr 5e-5 all 12 cells are clean and on-prompt with style
strength close to lr 1e-4. The α 0.5-only control at lr 5e-5 was not run, so the lr and the flag
are not separated.

## Under Muon

Muon steps the input axis like SGD, so α 0.5 left the loud channel's update at 6.9× the mean
channel's output leverage (1.6× at α 0.75): `docs/moss/findings/muon_channel_init.md`. That
leverage was `weight_svd`'s: with a kaiming init, α 0 and α 0.75 give the same loss to three
decimals, the same ‖ΔW‖ and stable rank, and ch 42 at 7.3 % vs 6.8 % of the FFN input share, so
the recipe runs with no scaling and reads no calibration file:
`docs/moss/findings/muon_no_channel_scaling.md`. `dc_isolate` under Muon is a bias channel (token-mean output
shift 2.57 with it, 1.24 without): `docs/moss/findings/muon_lora.md`.

## Default history

All on caststation (30 images, Base, `weight_svd`, 20 epochs), compared by eye. Under AdamW,
raising the scale (alpha / rank) raises the effective lr of `up` by the same factor.

1. **Rank 16, alpha 16, lr 1e-4.** Alpha 90 at lr 1e-4 matched its style but started copying the
   set's signature (watermark and handle); alpha 90 at lr 5e-5 transferred partially.
2. **Rank 32, alpha 128, lr 2e-5, no channel scaling.** lr 1e-4 pushed the style hard but lost
   prompt details on the complex gallery prompts; lr 2e-5 kept them.
3. **Channel scaling α 0.5, lr 1e-4**: strong style, on-prompt.
4. **Plus `dc_isolate`, lr 5e-5**: the AdamW default.
5. **Muon, `dc_isolate` off, lr 5e-5.** Muon at the same lr moves the adapter by the same Frobenius
   norm as AdamW but spreads it over ~3× the rank directions; by eye it beat AdamW on the same seed.
6. **α 0.75, kaiming `down` init.** Kaiming cuts the output DC shift from 1.24 to 0.43 at the same
   ‖ΔW‖ and stable rank.
7. **α 0, kaiming `down` init, Muon**: the current default.

## Calibration

Still live. Two files under `library/moss/calibration/`, both from
`scripts/calibration/moss_channel_stats.py` on Base (`_calib_mixed`, 16 samples ×
σ {0.1, 0.3, 0.5, 0.7, 0.9}, the training forward), keyed
`layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}` (k / v share q's input):

- `channel_stats.safetensors` (3.9 MB): `mean|x|` per channel; read by `svd_drop_loud`.
- `channel_mean.safetensors`: the signed token mean `x̄`; read by `down_init weight_svd_nodc`.

```bash
make daemon-run ARGS="scripts/calibration/moss_channel_stats.py --label <name>"
cp _archive/bench/moss_channel_stats/results/<run>/channel_{stats,mean}.safetensors library/moss/calibration/
```

The probe also writes `stats.npz` (per-block dominance, leverage, peak-to-mean), `dominance.png`
and the massive channels' rank at each input; `--model sft` profiles SFT, and `--channel_stats` /
`--channel_mean` on the trainer point at other files. A 10 % error in `mean|x|` is 5 % in `s`, and
the adaLN modulation is near-constant over σ 0.1–0.95, so the σ grid need not match training.
Regenerate only if the base DiT changes.
