# A reference bank of its own: copy ties, untranslated glyphs drop out, 16–25 % slower

Measured 2026-11-02 on swap_text, MOSS-Image Base. Plan B of
`_archive/proposals/reference_adapters.md` (`ref_adapters = all`) against the
shared bank. Run: `output/ckpt_moss_i2i/moss_i2i_swap_text_v3_en50_ko150_refall_rw2_mod16_ema5e5_int8_e3`;
renders `output/moss_i2i_eval/swap_text/results/refab_all_e3_ema`, seeds 1 and 2 of
`589883_0004_ko` in `refab_{default,all}_e3_ema_s{1,2}`; sheets `refab_all_e3_compare/`.

**Verdict: the default keeps the shared bank at this size.** The extra bank buys no copy, and
one page loses Japanese it should keep on two seeds of three. One Korean line improves.

## Setting

As [ref_kv_adapters.md](ref_kv_adapters.md): 200 pairs, the v3 int8 recipe, 3 epochs = 600 steps,
EMA finals, same sampler. The arm adds `ref_adapters = all`, `ref_rank = 32`: 252 adapters of
the reference's own on `q k v o ffn`, Base modulation at t = 0, main bank on image and text rows.
Built on a forked RNG, and the σ draws match the control's.

The control's memory plan ran out of VRAM at step 296: the bank adds 74M parameters (149M
in all), about 1.2 GB on the card across masters, gradients, Muon momentum and EMA. The run
below is swap 40, budget 0.05 against the control's swap 38, budget 0.1.

## Measured

| | default | B |
|---|---|---|
| s/step | 2.55 | 3.18 (swap 40, budget 0.05); 2.95 at the control's plan, until it died |
| minutes | 25.4 | 31.8 |
| peak allocated | 12.85 GB | 11.53 GB (other plan) |
| generation peak | 13.5 GiB | 13.7 GiB |

Training loss, B / default:

| steps | edit region | rest (copied) |
|---|---|---|
| 1–100 | 0.312 / 0.311 | 0.285 / 0.277 |
| 101–200 | 0.242 / 0.248 | 0.127 / 0.124 |
| 201–300 | 0.182 / 0.186 | 0.040 / 0.042 |
| 301–400 | 0.186 / 0.192 | 0.039 / 0.043 |
| 401–500 | 0.156 / 0.162 | 0.016 / 0.019 |
| 501–600 | 0.143 / 0.148 | 0.017 / 0.020 |

Outside the text boxes, seed 0, against the source:

| page | \|ΔL\| default | \|ΔL\| B | off by > 32, default | off by > 32, B |
|---|---|---|---|---|
| 1483610_0007_en | 6.7 | 6.5 | 3.6 % | 3.3 % |
| 3124789_0042_en | 9.4 | 9.1 | 6.6 % | 6.0 % |
| 589883_0004_ko | 3.3 | 3.4 | 0.9 % | 0.9 % |
| 737486_0019_ko | 3.3 | 3.0 | 1.5 % | 1.1 % |

- **Untranslated text, `589883_0004_ko`.** |ΔL| does not see this. Read per glyph on three seeds:

  | seed | default | B |
  |---|---|---|
  | 0 | `銭` gone, `湯` broken | `試` `銭` `湯` gone, both `洗` of the box and sign broken or gone, `円` of `千五百円` gone, `て` `の` broken |
  | 1 | clean | `追加サービス` a scribble, `き…`, `お頼`, `い` of `本洗い`, `円` of `千五百円` gone |
  | 2 | clean | clean, with strokes nearer the source's weight |

- **Letters, seed 0.** English exact on both. Korean: `분인가요` → `분인나요`, `싼다` → `씬다`,
  `벌거` → `벌겨` on both; `이건` → `이런` / `이전`. `가득해` is right only under B (`가늑해` on
  default). `벌겨 주기?` is black under B, as in the source, and magenta on default.
- **Lettering.** On `3124789_0042_en` B sets the two long balloons smaller and in more lines,
  with looser strokes. The words are right.

## Wiring check

The drop is not a generation bug. On the real LoRA, one forward through `RefCache` against the
joint masked forward differs by 1.4 % rel. RMS; the default LoRA's cached and joint forwards differ
by 1.8 %, and putting the main bank on the reference by mistake gives 74 %. On a tiny model the
compiled blocks follow the bank switch (2e-7 against eager).

## What it does not settle

Why the glyphs go. The sheets show the fault; nothing here separates the reference bank's
output from the main bank erasing more. Seeds were re-rendered on one page only, at 200 pairs
and 600 steps, so whether it shrinks with data is unmeasured. The 92-minute en200_ko800 arm
was not run.

## Implementation

Both adapters sit on one linear (`_RowBanks` in `library/moss/lora.py`). In training the rows
share one forward under `bank_masks()`, and both adapters run on every row, zero-masked on the
rows they do not own. `RefCache` builds under `LoRANetwork.reference()`, so generation runs one
bank per forward. The bank is saved in the same file as `ref_layers_N_proj_*`; the last block's
`q`, `o` and `ffn` reference adapters are never read and stay at their init. ComfyUI does not
read the stamp, so it applies the main bank to the reference and skips this bank.
