# Reference-side adapters for MOSS i2i

Status: proposal, 2026-11-02. Nothing below is built. The measurements are an inference
probe on the swap_text v3 LoRA and one eager timing from the detach report. Applies to
isolated references at a fixed `ref_timestep` (swap_text, sanitize, colorize v3).

Two plans share one implementation and one test. **A**, the cross-attention form: the
reference stream runs on Base under `no_grad`, and the image rows read it through trained
key / value adapters, as a text encoder is read through cross-attention. **B**, the split: the
reference rows keep a full adapter bank of their own, separate from the image rows', and
the stream stays in the graph. A saves compute; B spends it where the probe says the copy
lives. The earlier idea, a reference stream with no adapter at all, is dead; the probe is
in [What the probe showed](#what-the-probe-showed).

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
arm applied the k / v adapters inside the reference's own attention too, so it is plan A
with the stream perturbed, not plan A.

It agrees with the detach report: cutting the reference branch turned the late-block
`o` / `v` / `ffn_down` gradients to cos 0.41–0.45 at epoch 1 against floors of 0.80–0.85
([moss_ref_detach_grad.md](../../../project/int8_train/reports/moss_ref_detach_grad.md)).
Those are the blocks the i2i tasks share for copying.

## The two plans

Today one rank-32 bank does both jobs with the same weights: on the image rows it draws
the edit, on the reference rows it turns the page into something the image rows can copy
from. Both plans give the reference rows adapters of their own.

| | A: KV-only reference | B: split adapters |
|---|---|---|
| reference stream | Base weights, Base modulation at t = 0, `no_grad` | Base + its own bank (`q k v o ffn`, `ref_rank`), in the graph |
| what the image rows read | `k` / `v` from the Base stream plus a trained delta (export adapters on `proj.k` / `proj.v`, rank `ref_rank`) | the reference bank's `k` / `v` |
| image / text rows | today's bank | today's bank |
| trainable parameters on the reference path | 2 × 42 export adapters | a full second bank |
| step cost | the reference forward with no backward: 20% off an eager pass measured (`detached` arm of the detach report); the compiled plan is unmeasured | as today |
| generation | `RefCache` built once on Base, export deltas applied to the cached `k` / `v` | `RefCache` built once with the reference bank on |
| cross-step cache of the reference | the Base stream's per-block `x` is page-only, 1.06 GB bf16 per full page, so still no (0.3 TB for sanitize) | no |

A is the probe's "k / v only" arm with the stream clean, so its prior is the 6.6–12.9
column: the export adapters have to learn, from the image side's loss alone, what the stream
adapters now contribute. B's prior is v3 or better; the probe does not show that sharing
one bank hurts, so B is a bet on capacity, not a measured gap.

In A the reference's own attention must see Base `k` / `v`, or the stream depends on
trained parameters and `no_grad` cuts their gradient again. The export delta is applied to
the `k` / `v` the block hands out, not to the ones it attends with. That needs the split
inside `_Block.forward` (the reference rows first with `kv_out`, then the main rows with
`ref_kv`), the same split the detach probe ran eagerly.

### Implementation, shared

- `TrainRequest.ref_adapters = off | kv | all` (off = today, one shared bank) and
  `ref_rank`. `kv` is plan A, `all` is plan B. Requires `ref_isolate` and
  `ref_timestep >= 0`; stamped in the LoRA, with `ref_rank`.
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
- Plan A only: the `no_grad` reference part inside the compiled block, `ref_kv` under the
  training plan's strict mark (never run; generation compiles with `dynamic=True`), and two
  GEMM calls per linear, so the int8 base quantizes activations per call. Block swap: one
  `__call__` per block, the split inside `forward`, or the forward hook queues eviction after
  the first call.

## Test

swap_text on `en200_ko800`, the v3 int8 2-epoch recipe, 92 minutes per arm, control
`v3_int8_e2_ema_bf16_fp32s` (renders exist). Read: per glyph on the five Korean lines, the
outside-text |ΔL| above against the source, `tone_map.py`. Stop an arm on a glyph the
control gets right and it misses, or outside-text error above the control's on any page.

Order: A first, since it answers the compute question and its prior is the weaker one;
B after, or alone if the step cost is acceptable as it is. Only A needs the split block and
its phase-2 list; B needs the second bank and the mask.

Then sanitize full data (2,216 steps, 2.4 hours) on the plan that held, against
`moss_i2i_sanitize_idt_w0003_late_lr5e5_e8`: five held-out pages, seeds 42 / 43 / 44, since
sanitize has no bbox and its copy is the whole page.

## Generation and ComfyUI

Both plans stamp `ref_adapters`, and a loader that ignores the stamp renders wrong: the
main bank on the reference rows, the reference bank nowhere.

- **`generate.py`:** the reference bank goes on around `RefCache.lookup` (plan A: the
  export delta on the cached `k` / `v` after a Base build); the main bank stays off there.
  `set_multiplier` is a lifted float under `dynamic=True`, so this may cost no recompile.
- **ComfyUI-MOSS-Toolkit:** `lora_compat.load` patches weights through ComfyUI's patcher,
  which has no notion of rows. Plan A fits: `reference.py::build` runs on the unpatched
  model and applies the export deltas to its output, with the main LoRA patched afterwards or
  on a separate patcher. Plan B needs the reference bank patched during `build` and
  unpatched after, with the main bank the other way round — two patch sets on the same
  weights, in sequence. Both need the node to read the stamp and refuse an older flow.
  The native `preprocess_refs` path has the same two cases.
- **An older node on a stamped LoRA** applies the main bank to the reference with no
  warning. The node ships the stamp check before any such LoRA does.

## Tests

- With `ref_adapters = kv`, the reference rows' block output equals the same rows on Base
  weights, and the exported `k` / `v` differ from Base by the export delta only.
- With `ref_adapters = all`, the reference rows' output equals a forward with the
  reference bank alone applied; the image rows' equals one with the main bank alone.
- No main-bank gradient changes when the reference branch is detached (plan A).
- The stamp round-trips through `ref_settings` and `load_for_inference`; both files load.
- `generate.py`'s `RefCache` path matches the uncached forward on both plans.

`CONTRIBUTING.md` needs a bench and an invariant test for a method change: the probe
script and the swap_text arms are the bench; the first two items above are the test.

## Not in scope

- A reference with no stream at all, its `k` / `v` projected per block from the block-0
  embedding. Nothing here supports it; the stream adapters are the larger half.
- Removing the reference queries from the forward (the stream's `x` would change only
  through its FFN).
- Removing the t = 0 modulation. It is not the lever, and a stream with unit gates is a
  regime Base has not seen.
