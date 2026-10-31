# Massive activations as a semantic carrier: what arXiv 2605.13974 means for MOSS

Status: reading note, 2026-10-30. Nothing in it has been measured on MOSS yet.

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
  (`docs/moss/channel_scaling.md`). The `weight_svd` basis leans on it
  (`findings/muon_channel_init.md`, `findings/dcfree_svd_init.md`).
- **int8.** After the 256-column Hadamard, ch 42 is ~1.6× its group, so it does not
  hurt the int8 base (`project/int8_train/reports/moss_scoping.md`).

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

## Open questions for MOSS (unmeasured)

1. **Is there a mask in MOSS's MAs?** Run the paper's K-means on the 18-channel maps per
   block and step, and score it against a BiRefNet mask of the final image. It needs no
   training and only forward hooks, like `bench/moss_sinks/`. If it works, MOSS gets a
   training-free foreground mask: a candidate region for i2i preservation, or for
   weighting the loss.
2. **What a LoRA's ch 42 column reads.** `channel_scaling.md` treats ch 42 as a bias
   line because of its DC. If its variation is a foreground map, a LoRA weight on that
   column also reads a subject mask, not only a constant. Splitting the column's
   contribution into a DC part (`ΔW x̄`, already measured) and a part that varies per
   token would show which one the LoRA uses. `svd_drop_loud 1` (ch 42 dropped from the
   init) gave worse hands (`findings/dcfree_svd_init.md`); this would be one reading of
   that.
3. **Transport as an i2i baseline.** MOSS is single-stream, so the paper's "encoder
   stream" maps to the text rows of the same sequence. Image-conditioned transport would
   be a training-free reference point next to the MOSS i2i LoRAs (`docs/moss/methods/i2i.md`)
   and the vision-tower reference (`findings/vision_tower_reference.md`).
4. **DAVE already agrees.** DAVE leaves these 18 channels out of its mean edit
   (`docs/moss/inference/dave.md`), and the paper's disruption result says touching them breaks
   generation. Nothing to change there.
