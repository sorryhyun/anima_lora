# Reference adapters on swap_text: neither plan beats the shared bank

Measured 2026-11-02 on swap_text, MOSS-Image Base. Plans A and B of
`_archive/proposals/reference_adapters.md` against the default, where one adapter bank serves the
image, text and reference rows. Script `project/int8_train/src/ref_adapters_eval.py`.

**Verdict: the default keeps the shared bank.** A (`ref_adapters = kv`) is 29 % faster and loses
the copy on all four pages, by half the gap still at equal training time. B (`ref_adapters = all`)
ties on copy, costs 16–25 % on the step and 1.2 GB, and drops untranslated glyphs on one page.
Letters tie in both.

## Setting

200 pairs, 50 en + 150 ko drawn from `pairs_en200_ko800.jsonl` (seed 0,
`staging/pairs_en50_ko150.jsonl`). The v3 int8 recipe: rank 32, `mod_rank 16`, `region_weight 2`,
Muon lr 5e-5, `ema_window 0.1`, int8 stream, fp32 stream, swap 38, budget 0.1, 3 epochs = 600
steps. Arms build their extra adapters on a forked RNG, so the σ draws match the control's on all
600 steps and losses pair step by step. EMA finals, the control's seed and sampler, one seed per
page unless noted.

| arm | run (`output/ckpt_moss_i2i/moss_i2i_swap_text_v3_en50_ko150…`) | what the reference rows get |
|---|---|---|
| default | `_rw2_mod16_ema5e5_int8_e3` | the main bank, at `ref_timestep 0` |
| A | `_refkv_rw2_mod16_ema5e5_int8_e3`, `…_s848` | Base under `no_grad`, at Base modulation; image and text rows read it through 84 export adapters on `proj.k` / `proj.v` (`ref_rank 32`) |
| B | `_refall_rw2_mod16_ema5e5_int8_e3` | 252 adapters of their own on `q k v o ffn` (`ref_rank 32`), Base modulation at t = 0 |

Renders `output/moss_i2i_eval/swap_text/results/refab_{default,kv,all}_e3_ema` (and `_s{1,2}` seeds
of `589883_0004_ko`), `refab_kv_s848_ema`; sheets `refab_{e3,s848,all_e3}_compare/` beside them.

## Cost

| | default | A | A, 848 steps | B |
|---|---|---|---|---|
| s/step | 2.55 | 1.80 | 1.80 | 3.18 (swap 40, budget 0.05); 2.95 at the control's plan |
| minutes | 25.4 | 18.0 | 25.4 | 31.8 |
| peak allocated | 12.85 GB | 12.76 GB | | 11.53 GB (other plan) |

B's bank adds 74M parameters (149M in all), about 1.2 GB across masters, gradients, Muon momentum
and EMA; the control's memory plan ran out at step 296, hence swap 40 / budget 0.05. Generation peak
13.7 against 13.5 GiB.

## Loss

Training loss per 100 steps, arm / default:

| steps | A edit | A rest (copied) | B edit | B rest |
|---|---|---|---|---|
| 1–100 | 0.308 / 0.311 | 0.296 / 0.277 | 0.312 / 0.311 | 0.285 / 0.277 |
| 201–300 | 0.221 / 0.186 | 0.112 / 0.042 | 0.182 / 0.186 | 0.040 / 0.042 |
| 401–500 | 0.174 / 0.162 | 0.048 / 0.019 | 0.156 / 0.162 | 0.016 / 0.019 |
| 501–600 | 0.163 / 0.148 | 0.052 / 0.020 | 0.143 / 0.148 | 0.017 / 0.020 |

Under A the copied tokens' loss reaches 0.05 about 200 steps later, then sits at 2.5× the default's
(the last window reads flat only because its σ averages 0.70 against 0.77). B tracks the default.

## Copy, outside the text boxes

Mean |ΔL| against the source, 0–255, seed 0; in brackets the share of pixels off by more than 32:

| page | default | A | A, 848 steps | B |
|---|---|---|---|---|
| 1483610_0007_en | 6.7 (3.6 %) | 7.3 (4.5 %) | 7.0 | 6.5 (3.3 %) |
| 3124789_0042_en | 9.4 (6.6 %) | 10.1 (7.8 %) | 9.8 | 9.1 (6.0 %) |
| 589883_0004_ko | 3.3 (0.9 %) | 4.0 (1.5 %) | 3.8 | 3.4 (0.9 %) |
| 737486_0019_ko | 3.3 (1.5 %) | 3.9 (2.4 %) | 3.5 | 3.0 (1.1 %) |

For scale, `v3_int8_e2` (1,000 pairs, 2,000 steps) reads 6.1 / 8.4 / 2.8 / 2.8.
The 848-step comparison re-measured the pixel share on its own mask: off by more than 32,
default 3.8 / 7.3 / 1.3 / 2.2 %, A 600 steps 4.8 / 8.5 / 1.9 / 3.1 %, A 848 steps 4.3 / 8.0 / 1.7 / 2.7 %.

- **A loses the copy, and it shows**: the Japanese balloon left untranslated on `589883_0004_ko`
  comes back thin, `737486_0019_ko`'s line art lighter. At 848 steps (the default's minutes;
  warmup and EMA span scale with the run) half the gap remains, over three seeds on
  `589883_0004_ko` too (|ΔL| 3.76–3.78 against 3.25–3.34). Closing the rest loss at that slope would
  take about 550–800 more steps. A trained alone beats the earlier "k / v only" probe arm by 2.8–3.9.
- **B ties on |ΔL| but drops glyphs |ΔL| does not see.** Untranslated text on `589883_0004_ko`, read
  per glyph:

  | seed | default | B |
  |---|---|---|
  | 0 | `銭` gone, `湯` broken | `試` `銭` `湯` gone, both `洗` broken or gone, `円` of `千五百円` gone, `て` `の` broken |
  | 1 | clean | `追加サービス` a scribble, `き…`, `お頼`, `い` of `本洗い`, `円` gone |
  | 2 | clean | clean, strokes nearer the source's weight |

  A at 848 steps keeps every glyph on three seeds (`加` muddled on s1), with fuzzier strokes, and on
  one seed keeps balloon outlines the default loses.

## Letters

English exact in every arm. Korean, seed 0: `분인가요` → `분인나요`, `싼다` → `씬다` and `이건` →
`이런` in all three; `가득해` is right only under B. A at 848 steps reads `이먼` on three seeds of
`589883_0004_ko` where the default gets `이건` right once, and breaks `질 속이` onto two lines. B
sets `3124789_0042_en`'s long balloons smaller and in more lines. One seed on seven Korean lines at
200 pairs: nothing separates the arms. (The wrong syllables follow the tokenizer:
[ko_token_split.md](ko_token_split.md).)

## Wiring check for B

The glyph drop is not a generation bug. On the real LoRA one forward through `RefCache` against the
joint masked forward differs by 1.4 % rel. RMS (the default LoRA: 1.8 %; the main bank on the
reference by mistake: 74 %). On a tiny model the compiled blocks follow the bank switch (2e-7
against eager). Nothing here separates B's bank from the main bank erasing more.

## Implementation

- **A**: `_Block.forward` splits in Python when `ref_bank` is set: `ref_rows` under `no_grad`, then
  `main_rows` with the export in its graph. They compile apart, each with its own length mark (in
  one graph a reference length `n - ref_start` raises Inductor's `CantSplit`). Eager, the export's
  `k_norm` residuals ran the card out of memory at step 1. `delta_rows` does not run with `kv`.
- **B**: both banks sit on one linear (`_RowBanks` in `library/moss/lora.py`); in training both run
  on every row, zero-masked on the rows they do not own, under `bank_masks()`. `RefCache` builds
  under `LoRANetwork.reference()`, so generation runs one bank per forward. The bank is saved as
  `ref_layers_N_proj_*`; the last block's `q`, `o` and `ffn` reference adapters are never read.
- The ComfyUI node does not read the stamp: it applies the main bank to the reference and skips A's
  export and B's bank.

## Not tested

Either plan at full data (the 92-minute en200_ko800 arm was not run), whether B's glyph loss
shrinks with data, and more seeds than `589883_0004_ko`'s three.
