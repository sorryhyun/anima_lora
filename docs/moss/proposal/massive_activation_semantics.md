# Massive activations as a semantic carrier: what arXiv 2605.13974 means for MOSS

Status: reading note, 2026-10-30. Question 1 measured 2026-11-13 (§ Question 1 result); the
rest is unmeasured.

Paper: Turri, Bucciarelli et al., *Few Channels Draw The Whole Picture: Revealing Massive
Activations in Diffusion Transformers*, [arXiv 2605.13974](https://arxiv.org/abs/2605.13974)
(v2, 2026-07-07; NeurIPS 2026 accepted list). Project page and code:
[aimagelab.github.io/MAs-DiT](https://aimagelab.github.io/MAs-DiT). Models: FLUX.1-schnell,
FLUX.1-dev, FLUX.2-klein, Qwen-Image, SANA1.5. MOSS is not among them.

## What the paper shows

Massive activations (MAs) are selected per sample, layer and stream: the top-k channels by
|mean over tokens| of the hidden state.

- **Disruption.** Zeroing the top-k channels at every layer and step collapses the image.
  Zeroing an equally sized bottom-k set barely changes it. On the image stream FID and
  aesthetic score drop more; on the text stream, prompt alignment drops more.
- **Spatial structure.** The top-k image-token maps (k = 12) are min-max normalised per
  channel, then clustered with K-means (K = 2). The result is a foreground mask: on
  FLUX.2-klein, mIoU ≈ 0.5 against BiRefNet at layer 10, versus ~0.2 for bottom-k
  (GenAI-Bench, 1,600 prompts).
- **Transport.** Two generations share the same noise but use different prompts. Inside
  that mask, the target's top-k channels are replaced with the source's; on the text
  stream, channels are replaced without a mask. The source's subject moves into the
  target with its layout kept, and middle layers give the best trade-off. Without the
  mask the edit spreads (FLUX.1-schnell DINO-I 3.0 vs 20.1). An image-conditioned
  variant extracts a subject's MAs from a reference and injects them into new
  generations; it is training-free and on par with dedicated editing backbones on
  DreamBench++.

## What MOSS already has on file

- The massive channels are 1356 and 42 on every block, plus 16 more:
  `library/moss/calibration/massive_channels.npz` (18 channels, per-block top 4, derived
  on Base).
- **Constant offset.** 1356 is ~1900× the median channel at block 0. Together these
  channels hold ~99 % of the image tokens' mean, the same on every seed and prompt
  (`docs/moss/inference/dave.md`). On blocks 0–18 the DC term is 0.98–0.99 of their energy
  (`findings/attention_sink_chat_template.md`).
- **Effect on LoRAs.** At the FFN input, ch 42 is 11–26× the median channel on blocks
  11–41, with a peak-to-mean of 2.7, so a LoRA sees it as a bias line
  (`project/finished/moss_lora/done/channel_scaling.md`). The `weight_svd` basis leans on it
  (`findings/t2i_lora_recipe.md` §3).
- **int8.** After the 256-column Hadamard, ch 42 is ~1.6× its group, so it does not
  hurt the int8 base (`project/finished/int8_train/reports/moss_scoping.md`).

## How the two fit

The paper says MAs carry prompt-specific semantics. MOSS's measurements say the same
channels are a near-constant offset across prompts. Both can hold:

- The mask step min-max normalises each channel over tokens, which removes the offset.
  What clusters into a subject is the per-token variation on top of it, which on MOSS's
  blocks 0–18 is the ~1–2 % of energy left after the DC term.
- Transport copies whole channel values. Where the offset is the same in both
  generations, only that variation actually moves.

The hypothesis: on MOSS, 1356 and 42 are a fixed DC plus a small, spatially structured,
prompt-dependent part, and the second part is what the paper's results depend on.

## Open questions for MOSS

1. **Is there a mask in MOSS's MAs?** Run the paper's K-means on the 18-channel maps per
   block and step, and score it against a BiRefNet mask of the final image. It needs no
   training and only forward hooks, like `bench/moss_sinks/`. If it works, MOSS gets a
   training-free foreground mask: a candidate region for i2i preservation, or for
   weighting the loss.
2. **What a LoRA's ch 42 column reads.** `project/finished/moss_lora/done/channel_scaling.md` treats ch 42 as a bias
   line because of its DC. If its variation is a foreground map, a LoRA weight on that
   column also reads a subject mask, not only a constant. Splitting the column's
   contribution into a DC part (`ΔW x̄`, already measured) and a part that varies per
   token would show which one the LoRA uses. `svd_drop_loud 1` (ch 42 dropped from the
   init) gave worse hands (`findings/t2i_lora_recipe.md`); this would be one reading of
   that.
3. **Transport as an i2i baseline.** MOSS is single-stream, so the paper's "encoder
   stream" maps to the text rows of the same sequence. Image-conditioned transport would
   be a training-free reference point next to the MOSS i2i LoRAs (`docs/moss/methods/i2i.md`)
   and the vision-tower reference (`findings/vision_tower_reference.md`).
4. **DAVE already agrees.** DAVE leaves these 18 channels out of its mean edit
   (`docs/moss/inference/dave.md`), and the paper's disruption result says touching them breaks
   generation. Nothing to change there.

## Question 1 result (2026-11-13)

`bench/moss_ma_mask/probe_masks.py`, run `results/20261113-1014-base`: Base, fp32 stream,
1024², 30 steps, CFG 3, 16 subject-on-background prompts × 2 seeds. Block outputs of the
cond forward on steps 0 3 7 11 15 19 23 29, K-means on the raw channel maps and on their
min-max normalised maps, the paper's polarity rule, mIoU against BiRefNet on the final
image. Channel sets: top12, top2, bottom12, rand12, the 18 of `massive_channels.npz`, and
all channels but top12 (`rest`). A random mask of the same area scores 0.30–0.32.

**The gate failed.** It was fixed on the paper's protocol (last step): top12 ≥ 0.5 and ≥ 0.1
above bottom12 and rand12, block picked on seed 0, read on seed 1. Raw: block 0, 0.437 /
0.352 / 0.242. Normalised: block 7, 0.221 / 0.329 / 0.393. At the last step the top-channel
maps follow edges and texture, and top12 falls from 0.44 (block 0) to ~0.25 (blocks 24–36).

**The paper's pattern is at block 10 on the middle steps** (read after the gate; step and
block picked on seed 0, numbers from seed 1):

| K-means on | step | top12 | top2 | fixed18 | bottom12 | rand12 | rest |
|---|---|---|---|---|---|---|---|
| raw | 7 | 0.488 | 0.488 | 0.423 | 0.313 | 0.309 | 0.317 |
| normalised | 11 | 0.525 | 0.462 | 0.454 | 0.348 | 0.323 | 0.311 |

- Raw top12 beats bottom12 on 16 / 16 seed-1 renders and rand12 on 15 / 16. The block-10
  band holds on steps 3–19 in both readings (`probe.png`).
- **Two channels are enough.** Raw top2 gives the same mask as raw top12: the distance is
  set by the loudest channel. `sheet_step7_b10_raw.png` shows a subject silhouette with
  speckle on most pages, noise for bottom12.
- The other channels hold a cleaner split at the same place, without a usable polarity:
  `rest` scores 0.56 with the better polarity at step 7, block 10 (0.36 under the rule),
  and 0.65 at block 21 over the 29 renders with ≥ 5 % foreground. It often cuts ground from sky rather than subject from
  scene.
- In blocks 22–36 the rule picks the background: normalised top12 at block 24, last step,
  is 0.22 under the rule and 0.51 with the better polarity.
- top12's DC share of energy is 0.999 at block 0 and 0.95–0.97 past block 28, so the mask
  is read from the 0.1–5 % left over, as the hypothesis above says.
- BiRefNet finds ≤ 2 % foreground on three renders (lighthouse, subway) and marks most of
  one manga page; leaving out the three renders under 5 % moves the block-10 numbers by
  0.02–0.03.

Not measured: the uncond branch, a smoothed mask, any location inside the block (FFN
input or activation), a LoRA, and i2i reference rows.

## i2i LoRAs on these channels (2026-11-13, weights only)

`bench/moss_ma_mask/lora_channels.py`, run `results/20261113-1051-lora-channels`. Uncensor
(with and without `mod_rank` 16), sanitize v3, swap_text v3 and v6, colorize v5, against
the t2i ama_mitsuki pair; all rank 32, Muon, kaiming `down`, so a column's share of `ΔW`
starts uniform (18 channels: 0.59 %, ch 1356 + 42: 0.065 %). The modulation is shared by
all 42 blocks with no per-block term.

**Base drives these channels through the modulation.** Of 3,072 channels, 1356 and 42 are
the two largest entries of the attention gate (−9.6 / +4.3, rms 0.48), the FFN shift
(−6.0 / −4.8, rms 0.18) and the FFN gate (+31 / +9, rms 2.7) at t = 0, and the same at
t = 900. The FFN shift has 64–90 % of its energy on the 18 channels. And **Base closes
ch 42 to attention at t = 0**: its input multiplier `1 + scale_a` is −0.010 there (second
smallest of all channels, median 2.28), against −0.17 … −0.30 at t 100–900.

**The i2i modulation adapters reopen it for the reference rows.**

| `1 + scale_a` on ch 42 | t = 0 (reference) | t = 900 (image) |
|---|---|---|
| Base | −0.010 | −0.268 |
| uncensor mod16 | −0.035 | −0.259 |
| swap_text v3 | −0.034 | −0.229 |
| swap_text v6 | −0.076 | −0.231 |
| t2i ama_mitsuki mod16 | −0.009 | −0.267 |

- In `scale_a`, ch 42 is the most-moved channel of 3,072 in all three i2i adapters at t = 0
  and ch 1356 is in the top 7; the sign flips between t = 0 and t = 900 on both. The t2i
  adapter leaves both alone there (ranks 17 and 2,628).
- `scale_f` moves ch 42 most in the t2i adapter too, so that chunk is not i2i-specific.
- It is the place the adapters act, not always the same move. The copy block
  `c_fill_bw10_mod16_s800` takes the entry to −0.048, swap_text from C
  (`swap_dense2516_en400_c_e2`) to −0.026, the int8 uncensor rerun to −0.019,
  `moss_i2i_character_ref_offkv_cinit` to +0.127 (the other sign); `1111_1546` and
  `1112_0047` leave it at −0.014 and −0.006. Ch 42 is rank 1–3 of the `scale_a` delta in
  all of them but the uncensor rerun (rank 7).
- The adapter's large part is elsewhere: the FFN-shift delta (74–75 % of `‖Δmod‖²`,
  `findings/mod_rank_i2i.md`) has 1–3 % of its energy on the 18 channels, and changes the
  shift on ch 1356 and 42 by ≤ 1.1 %.

**The block adapters lean on them mildly on the read side, more on the write side.**

| share of `ΔW` on ch 1356 + 42 (uniform 0.065 %) | t2i | uncensor | sanitize | swap_text | colorize |
|---|---|---|---|---|---|
| `ffn_up_gate` columns, all block ranges | 0.05–0.07 | 0.12–0.17 | 0.12–0.15 | 0.12–0.20 | 0.12–0.17 |
| `k` columns, blocks 36–41 | 0.07 | 0.18–0.19 | 0.20 | 0.16–0.21 | 0.08 |
| `o` rows (write), by block range | 0.47–0.94 | 0.69–2.27 | 1.20–1.57 | 0.72–1.04 | 0.30–0.93 |

- Weighted by the calibrated input, the 18 channels carry 21–25 % of the i2i adapters'
  read at the FFN input of blocks 26–35; the input itself has 15 % there and the t2i
  adapter reads 16 %.
- The i2i `q` / `k` adapters respond to the input mean 2–5× more than a `ΔW` blind to it
  (`‖ΔW x̄‖ / ‖ΔW‖‖x̄‖` 0.04–0.10 against 0.018; t2i 0.03 falling to 0.008), but ≤ 0.23 of
  that response comes through the 18 channels.
- Every adapter writes to ch 1356 and 42 through `o` at 5–35× the uniform share, t2i
  included; the two removal tasks most.

Hypothesis (untested): on a clean reference, ch 42's per-token map is hidden from
attention by Base's `scale_a`, and the modulation adapter's gain on the copied tokens
(`findings/mod_rank_i2i.md`) comes partly from letting it through. Not measured: what
ch 42 holds on reference rows, and a render with that one entry of `Δmod` zeroed.
