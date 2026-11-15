# MOSS-Image reports

Closed checks: each was measured and settled, and its outcome is applied in the code or the catalog. Live
measurements are in [`../findings/`](../findings/README.md).

| Doc | What it covers | Applied as |
|---|---|---|
| [bf16_residual_stream.md](bf16_residual_stream.md) | An fp32 stream around the same bf16 GEMMs cuts the prediction error 2.4–3.5x for +1.5 % time; renders move 4–12/255 and read better on text and fingers. Training with it, and swap_text. | [`methods/fp32_stream.md`](../methods/fp32_stream.md) |
| [caststation_casbox.md](caststation_casbox.md) | caststation captions with `mask_dilate 16` (held objects reach their holder's clause) and OCR boxes on text lines; balanced hybrid95 over them softens renders (high-frequency ×0.6) because the flow-matching student is already 0.010 below the Base teacher at σ 0.95. | `CaptionsRequest.mask_dilate`, `CacheRequest.text_boxes`; caststation trains flow-only |
| [fa2_tile_sm120.md](fa2_tile_sm120.md) | FA2 runs its A100 / H100 forward tile on sm_120. 128×32 is free of error and worth 0.7 % of a step. | [`optimization/fa2_sm120.md`](../optimization/fa2_sm120.md) |
| [heal_finetune.md](heal_finetune.md) | KBlueLeaf's MOSS-Image-heal (stream max 3k vs 58k) is fp16-safe, but bf16 error, GPTQ int8 error and a caststation LoRA tie Base; its prior is warmer and knows danbooru artist tags. The shipped int8 ConvRot / fp16 files measured. | only our GPTQ file kept (`moss_dit_int8`) |
| [self_distill_arms.md](self_distill_arms.md) | Self-distillation arms on caststation (2026-11-09): the gate probe, Base and EMA teachers, replace / add / extra-term forms at 240 and 480 steps. add90 ema97 keeps Base's composition on general and multi-character prompts for a little style; superseded as the default on 2026-11-14 by balanced hybrid95. | [`experimental/self_distill.md`](../experimental/self_distill.md), `configs/moss/selfdistill.toml` |
| [sm120_bf16_hmma.md](sm120_bf16_hmma.md) | No bf16-input, fp16-accumulate HMMA on sm_120: PTX has no form for it, and the SASS bit nvdisasm reads as `.BF16` faults as an illegal encoding at any offset. FA2's fp16-accumulate path keeps its bf16 → fp16 casts. | [`optimization/fa2_sm120.md`](../optimization/fa2_sm120.md) |
