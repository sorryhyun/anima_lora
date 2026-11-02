# Reference-side adapters for MOSS i2i

Status: proposal, 2026-11-02. Plan A was built and tried; it lost on copy
([ref_kv_adapters.md](../findings/ref_kv_adapters.md)). What is left to try is plan B, not built.
The measurements below are an inference probe on the swap_text v3 LoRA and that A/B. Applies to
isolated references at a fixed `ref_timestep` (swap_text, sanitize, colorize v3).

**A**, tried: the reference stream ran on Base under `no_grad`, and the image rows read it through
trained key / value adapters, as a text encoder is read through cross-attention. 29% off the
step, copy worse on all four pages, letters tie. **B**, proposed: the reference rows keep a full
adapter bank of their own, separate from the image rows', and the stream stays in the graph. It
spends the compute where the probe says the copy lives. A reference stream with no adapter at
all is dead; the probe is in [What the probe showed](#what-the-probe-showed).

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

Lettering left untranslated comes back thin and line art lighter; letters tie. The rest loss is
flat over the last 200 steps, and more steps are untested. The code stays in the tree:
`_Block.forward` splits in Python, and `ref_rows` and `main_rows` compile apart with their own
length marks, since in one graph a reference length `n - ref_start` raises Inductor's
`CantSplit`. Details in [ref_kv_adapters.md](../findings/ref_kv_adapters.md).

## Plan B

Today one rank-32 bank does both jobs with the same weights: on the image rows it draws
the edit, on the reference rows it turns the page into something the image rows can copy
from. B gives the reference rows their own bank: Base + `q k v o ffn` at `ref_rank`, in the graph.
The image rows read that bank's `k` / `v`, and image and text rows keep today's bank.

- **Parameters:** a full second bank on the reference path.
- **Step cost:** as today. A's split is not needed: the rows stay in one forward, under a mask.
- **Generation:** `RefCache` built once with the reference bank on.
- **Cross-step cache of the reference:** no.

B's prior is v3 or better. The probe and A show the reference's own adapters matter; neither
shows that sharing one bank hurts, so B is a bet on capacity, not a measured gap.

### Implementation

- `TrainRequest.ref_adapters` gains `all` beside `off` (today, one shared bank) and `kv` (plan A),
  at `ref_rank`. Requires `ref_isolate` and `ref_timestep >= 0`; stamped in the LoRA, with
  `ref_rank`.
- A second adapter per target in `LoRANetwork`, with the complementary row mask: the
  main bank's delta on image and text rows, the reference bank's on reference rows.
  `set_row_mask` already confines a delta per row (`library/lines/adapters.py`); the mask
  builder `row_mask()` in `library/moss/training/data.py` keeps text rows only and needs
  the three-way form (image + text / reference), ANDed with `delta_rows` when both are set.
  Two `LoRANetwork`s on one linear do not stack: `_patched_forward` captures
  `nn.Linear.forward` as the base, so the second patch replaces the first. One network,
  two banks.
- Saved as two files, `<name>.safetensors` and `<name>.ref.safetensors`, or two key
  prefixes in one; the loader applies the reference bank only where the reference is built.
  Two files are easier for the ComfyUI node.

## Test

swap_text on `en50_ko150`, `train_v3_en50_ko150.toml` with `ref_adapters = all`, 600 steps, about
25 minutes. Control: the shared-bank arm of the A result (renders `refab_default_e3_ema`). Quality
first, then training time if the two read the same: outside-text |ΔL|, the Japanese left
untranslated, per glyph on the Korean lines.

If it holds, the en200_ko800 v3 int8 2-epoch recipe (92 minutes) against
`v3_int8_e2_ema_bf16_fp32s`. Then sanitize full data (2,216 steps, 2.4 hours) against
`moss_i2i_sanitize_idt_w0003_late_lr5e5_e8`: five held-out pages, seeds 42 / 43 / 44, since
sanitize has no bbox and its copy is the whole page.

## Generation and ComfyUI

A file stamped `ref_adapters` renders wrong on a loader that ignores the stamp: the main bank
on the reference rows, the reference bank nowhere. `generate.py` reads plan A's stamp; the
ComfyUI node reads neither.

- **`generate.py`:** the reference bank goes on around `RefCache.lookup`; the main bank stays
  off there.
  `set_multiplier` is a lifted float under `dynamic=True`, so this may cost no recompile.
- **ComfyUI-MOSS-Toolkit:** `lora_compat.load` patches weights through ComfyUI's patcher,
  which has no notion of rows. B needs the reference bank patched during `build` and
  unpatched after, with the main bank the other way round — two patch sets on the same
  weights, in sequence. The node has to read the stamp and refuse an older flow. The native
  `preprocess_refs` path has the same case.
- **An older node on a stamped LoRA** applies the main bank to the reference with no
  warning. The node ships the stamp check before any such LoRA does.

## Tests

Plan A's are in `tests/test_moss_ref_adapters.py`. For B:

- The reference rows' output equals a forward with the reference bank alone applied; the image
  rows' equals one with the main bank alone.
- The stamp round-trips through `ref_settings` and `load_for_inference`; both files load.
- `generate.py`'s `RefCache` path matches the uncached forward.

`CONTRIBUTING.md` needs a bench and an invariant test for a method change:
`project/int8_train/src/ref_adapters_eval.py` is the bench; the first item above is the test.

## Not in scope

- A reference with no stream at all, its `k` / `v` projected per block from the block-0
  embedding. Nothing here supports it; the stream adapters are the larger half.
- Removing the reference queries from the forward (the stream's `x` would change only
  through its FFN).
- Removing the t = 0 modulation. It is not the lever, and a stream with unit gates is a
  regime Base has not seen.
