# k / v adapters alone on the reference: 29 % off the step, the copy gets worse

Measured 2026-11-02 on swap_text, MOSS-Image Base. Plan A of
`_archive/proposals/reference_adapters.md` (`ref_adapters = kv`) against the
shared bank. Runs: `output/ckpt_moss_i2i/moss_i2i_swap_text_v3_en50_ko150{,_refkv}_rw2_mod16_ema5e5_int8_e3`;
renders `output/moss_i2i_eval/swap_text/results/refab_{default,kv}_e3_ema`; sheets
`refab_e3_compare/` beside them; script `project/int8_train/src/ref_adapters_eval.py`.

**Verdict: the default keeps the shared bank.** A loses the copy on all four pages, and letters
tie.

## Setting

200 pairs, 50 en + 150 ko drawn from `pairs_en200_ko800.jsonl` (seed 0,
`staging/pairs_en50_ko150.jsonl`). The v3 int8 recipe: rank 32, `mod_rank 16`,
`region_weight 2`, Muon lr 5e-5, `ema_window 0.1`, int8 stream, fp32 stream, swap 38, budget
0.1, 3 epochs = 600 steps. The arm adds `ref_adapters = kv`, `ref_rank = 32`: the reference
stream runs on Base under `no_grad`, at Base modulation, and the image and text rows read it
through 84 export adapters on `proj.k` / `proj.v`. Those are built on a forked RNG, and the σ
draws match on all 600 steps, so the losses pair step by step. EMA finals, seed and sampler as
the control renders, one seed per page.

## Measured

| | default | A |
|---|---|---|
| s/step | 2.55 | 1.80 (−29 %) |
| minutes | 25.4 | 18.0 |
| peak allocated | 12.85 GB | 12.76 GB |

Training loss, A / default:

| steps | edit region | rest (copied) |
|---|---|---|
| 1–100 | 0.308 / 0.311 | 0.296 / 0.277 |
| 101–200 | 0.275 / 0.248 | 0.192 / 0.124 |
| 201–300 | 0.221 / 0.186 | 0.112 / 0.042 |
| 301–400 | 0.209 / 0.192 | 0.079 / 0.043 |
| 401–500 | 0.174 / 0.162 | 0.048 / 0.019 |
| 501–600 | 0.163 / 0.148 | 0.052 / 0.020 |

The copied tokens' loss reaches 0.05 about 200 steps later under A, then sits at 2.5× the
default's. The last two windows read flat only because σ moves: the last 100 steps average 0.70
against 0.77. Controlled for σ, the rest loss over 401–600 falls ×0.86 per 100 steps under A
and ×0.82 under default.

Mean |ΔL| against the source outside the text boxes, 0–255:

| page | default | A | probe "k / v only" |
|---|---|---|---|
| 1483610_0007_en | 6.7 | 7.3 | 10.1 |
| 3124789_0042_en | 9.4 | 10.1 | 12.9 |
| 589883_0004_ko | 3.3 | 4.0 | 7.9 |
| 737486_0019_ko | 3.3 | 3.9 | 6.6 |

For scale, `v3_int8_e2` (1,000 pairs, 2,000 steps) reads 6.1 / 8.4 / 2.8 / 2.8.

- **Copy.** Worse on every page, and it shows. The Japanese balloon left untranslated on
  `589883_0004_ko` stays bold under default and comes back thin under A (dark pixels in that
  crop: source 3,346, default 2,764, A 2,562); `737486_0019_ko`'s line art is lighter. Pixels
  off by more than 32 rise from 3.6 / 6.6 / 0.9 / 1.5 % to 4.5 / 7.8 / 1.5 / 2.4 %.
- **Letters.** A tie: `분인가요` → `분인나요`, `이건` → `이런`, `싼다` → `씬다` and `가득해` → `가늑해`
  on both, English exact on both. One seed on seven Korean lines at 200 pairs, so this says
  only that nothing separates the arms.
- **Against the probe.** A beats the "k / v only" arm by 2.8–3.9. That arm's adapters were
  trained alongside the stream's, so trained alone they learn part of what the stream adapters
  did, and not all of it in 600 steps.

## At the same training time

Run: `..._refkv_rw2_mod16_ema5e5_int8_s848`, 848 steps at 1.80 s/step, 25.4 min, the default's
time; same config with `epochs = 5`, `max_steps = 848`. Warmup and the EMA span scale with the
run (42 and 85 updates). Renders `refab_kv_s848_ema`, seeds 1 and 2 of `589883_0004_ko` in
`refab_kv_s848_ema_s{1,2}`; sheets `refab_s848_compare/`.

| | default, 600 steps | A, 600 | A, 848 |
|---|---|---|---|
| outside-text \|ΔL\| | 6.7 / 9.4 / 3.3 / 3.3 | 7.3 / 10.1 / 4.0 / 3.9 | 7.0 / 9.8 / 3.8 / 3.5 |
| pixels off by more than 32, % | 3.8 / 7.3 / 1.3 / 2.2 | 4.8 / 8.5 / 1.9 / 3.1 | 4.3 / 8.0 / 1.7 / 2.7 |

Loss over the last 100 steps, A 848 against default 600, controlled for σ: rest ×1.94 (1.83–2.06),
edit region ×1.04 (0.95–1.13). At equal minutes up to 18, rest is ×1.81 and edit region ×1.02.

- **Copy.** Still behind on all four pages, by about half the gap. Over three seeds on
  `589883_0004_ko`, |ΔL| reads 3.76–3.78 against 3.25–3.34, and the untranslated balloon stays
  thin: dark pixels 1,960–2,017 against 2,197–2,287, source 2,541.
- **Untranslated glyphs.** A keeps every glyph on three seeds, with `加` of `追加` muddled on s2.
  Default loses `銭` and breaks `湯` on s0. A's strokes are fuzzier on `銭湯` and `楽`.
- **Balloon outlines.** One seed: A 848 keeps the `Uwah` outline and the lower-right edge of the
  `SAKURA-CHAN` balloon, both of which default loses. Those sit inside the text boxes, so |ΔL|
  does not read them.
- **Letters.** English exact on both. On `589883_0004_ko` over three seeds, A reads `이먼` on all
  three, `뭔뭔가요` on s1 and `분있어요` / `분인어요`; default reads `분인나요` / `분이며요` and gets
  `이건` right on s2. On `737486_0019_ko`, seed 0, `싼다` → `씬다` and `가득해` → `가늑해` on both,
  and A breaks `질 속이` onto two lines.

The default keeps the shared bank at equal time. A's gain over 248 more steps is real and
smaller than the gap. Closing the rest loss at this slope would take about 550–800 steps more.

The numbers compare arms at this size; no arm's |ΔL| reaches the 2,000-step control. Plan B (a full bank of the reference's own) is in [ref_bank_adapters.md](ref_bank_adapters.md).

## Implementation

`_Block.forward` splits in Python when `ref_bank` is set: `ref_rows` under `no_grad`, then
`main_rows` with the export in its graph. They compile apart, each with its own length mark:
in one graph a reference length `n - ref_start` raises Inductor's `CantSplit`. Run eagerly, the
export's `k_norm` residuals ran the card out of memory at step 1. `delta_rows` does not run with
`kv` (its mask spans reference rows `main_rows` does not hold). The ComfyUI node does not read
the stamp, so it applies the main bank to the reference and skips the export.
