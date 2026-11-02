# Reference-side adapters for MOSS i2i

Status: closed, 2026-11-02. Both plans were built and tried; neither beat the shared bank at
200 pairs and 600 steps ([ref_kv_adapters.md](../findings/ref_kv_adapters.md),
[ref_bank_adapters.md](../findings/ref_bank_adapters.md)). The default stays `ref_adapters = off`.
The measurements below are an inference probe on the swap_text v3 LoRA and those A/Bs. Applies to
isolated references at a fixed `ref_timestep` (swap_text, sanitize, colorize v3).

**A**, tried: the reference stream ran on Base under `no_grad`, and the image rows read it through
trained key / value adapters, as a text encoder is read through cross-attention. 29% off the
step, copy worse on all four pages, letters tie. **B**, tried: the reference rows keep a full
adapter bank of their own, separate from the image rows', and the stream stays in the graph.
Copy ties, one Korean line improves, untranslated glyphs drop out on two seeds of three on one
page, and the step costs 16–25% more. A reference stream with no adapter at all is dead; the
probe is in [What the probe showed](#what-the-probe-showed).

## What the probe showed

The reference rows carry the copy, and the LoRA's work on them is what makes it hold.

`project/int8_train/src/ref_lora_off_render.py` renders the four swap_text eval pages
with the v3 en400_ko1709 LoRA, same seed, prompt and sampler as the existing renders, but
builds `RefCache` with some adapters at multiplier 0 on the reference rows. Sheets:
`output/moss_i2i_eval/swap_text/results/v3_en400_ko1709_e2_reflora_compare/`
(`<page>.png` the text boxes at 2x, `page_<page>.png` the full page with the most-changed
non-text window). Mean |ΔL| against the source outside the text boxes, 0–255:

| page | v3 | ref blocks off | + mod off | k / v adapters only | q / o / ffn only |
|---|---|---|---|---|---|
| 1483610_0007_en | 5.6 | 22.9 | 23.3 | 10.1 | 7.2 |
| 3124789_0042_en | 8.0 | 31.9 | 32.1 | 12.9 | 9.6 |
| 589883_0004_ko | 2.7 | 15.2 | 15.7 | 7.9 | 4.1 |
| 737486_0019_ko | 2.6 | 15.5 | 15.8 | 6.6 | 3.4 |

- The text survives every arm: the strings render, and only the errors move
  (`분이나요` → `분인나요`, `벌거` → `벌겨`, `좀…더` goes vertical). The reference adapters do
  not hold what to write; the bbox and prompt do.
- With the adapters off on the reference, the copy breaks: balloon outlines thin or double,
  screentone turns to noise, the page reads as redrawn. 4–6x the error.
- The modulation adapter at t = 0 is not the lever: ±1–2 with it off.
- Of the block adapters, the ones on the reference's own stream (`q`, `o`, `ffn`) carry more
  than the `k` / `v` projections the image rows read. Neither half alone reaches v3.

Caveats. This splits a LoRA trained with every adapter on, so the halves are not
independent: "k / v only" is a floor for a LoRA trained that way, not its value. And that
arm applied the k / v adapters inside the reference's own attention too. Plan A, trained on
its own with the stream clean, came in at 7.3 / 10.1 / 4.0 / 3.9 on these pages.

It agrees with the detach report: cutting the reference branch turned the late-block
`o` / `v` / `ffn_down` gradients to cos 0.41–0.45 at epoch 1 against floors of 0.80–0.85
([moss_ref_detach_grad.md](../../../project/int8_train/reports/moss_ref_detach_grad.md)).
Those are the blocks the i2i tasks share for copying.

## Plan A, tried

`ref_adapters = kv`, `ref_rank`: Base stream under `no_grad`, Base modulation at t = 0, export
adapters on `proj.k` / `proj.v` applied to the `k` / `v` the block hands out and not to the ones
it attends with. On 200 pairs and 600 steps against the shared bank:

| | shared bank | A |
|---|---|---|
| s/step | 2.55 | 1.80 |
| outside-text \|ΔL\| | 6.7 / 9.4 / 3.3 / 3.3 | 7.3 / 10.1 / 4.0 / 3.9 |
| rest loss, steps 401–600 | 0.020 | 0.051 |

Lettering left untranslated comes back thin and line art lighter; letters tie. At the shared
bank's training time, 848 steps, |ΔL| reads 7.0 / 9.8 / 3.8 / 3.5 and the balloon stays thin; A keeps
every untranslated glyph on three seeds, and Korean gets slightly worse. The code stays in the tree:
`_Block.forward` splits in Python, and `ref_rows` and `main_rows` compile apart with their own
length marks, since in one graph a reference length `n - ref_start` raises Inductor's
`CantSplit`. Details in [ref_kv_adapters.md](../findings/ref_kv_adapters.md).

## Plan B, tried

`ref_adapters = all`, `ref_rank`: Base + `q k v o ffn` at `ref_rank` on the reference rows, in the
graph, Base modulation at t = 0. Image and text rows keep today's bank. On 200 pairs and 600
steps against the shared bank:

| | shared bank | B |
|---|---|---|
| s/step | 2.55 | 3.18; 2.95 on the control's memory plan, which ran out of VRAM at step 296 |
| outside-text \|ΔL\| | 6.7 / 9.4 / 3.3 / 3.3 | 6.5 / 9.1 / 3.4 / 3.0 |
| rest loss, steps 501–600 | 0.020 | 0.017 |
| seeds of 3 losing untranslated glyphs, `589883_0004_ko` | 1 (two glyphs) | 2 (a whole heading on one) |

The step did not stay as today: both adapters run on every row, zero-masked on the rows they do
not own. The bank adds about 1.2 GB on the card in training and 0.2 GiB in generation. The code
stays in the tree, one file with `ref_layers_N_proj_*` keys. `generate.py` reads the stamp; the
ComfyUI node reads neither plan's. Details in
[ref_bank_adapters.md](../findings/ref_bank_adapters.md).

## Not in scope

- A reference with no stream at all, its `k` / `v` projected per block from the block-0
  embedding. Nothing here supports it; the stream adapters are the larger half.
- Removing the reference queries from the forward (the stream's `x` would change only
  through its FFN).
- Removing the t = 0 modulation. It is not the lever, and a stream with unit gates is a
  regime Base has not seen.
