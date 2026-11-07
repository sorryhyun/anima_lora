# Step 3: the Hangul block, 500 steps from C

Date: 2026-11-08. Step 3 of [plan.md](../plan.md). Sheets (`output/tests/lora_blocks/`): `hangul_s500/zoom_<page>_s<seed>.png`
(rows edition, ksplit e2, block alone, stack; columns the caption's text boxes at 2x),
`hangul_s500/page_589883_s1.png` (full page: source, e2, alone, stack), `hangul_s500/grad_in.json`.

Result in one line: **500 steps on synthetic font lettering do not make the strokes crisper. Alone,
the block writes the right syllables in a thin font with uneven line breaks and sizes, and garbles
the Japanese balloons the caption leaves out. Stacked on ksplit e2, it blurs or doubles the
strokes on one page and bolds or tints them on the other.**

## Setup

- **Data** (`src/build_hangul_data.py`, `post_image_dataset/i2i/hangul_block/`): 600 Korean
  edition pages of swap_text v3 (eval works out). On each page, the OCR boxes whose 5 px ring is
  ≥ 93 % white (inside a balloon) are whitened, and another page's Korean OCR line (½–1.5× the
  length) is lettered in one of 17 Korean faces (`kozh/` + `ko_extra/`, cmap-checked), centred,
  at the largest size that fits. Condition: the whitened page (`blank`, 313) or the page
  with its own lettering (`orig`, 287). The caption is swap_text's layout JSON, with the new
  strings at their ink boxes. That gives 1,531 lines and 750 distinct syllables, plus 150 ko
  edition twins from the ksplit cache (`src/link_hangul_twins.py`): 750 samples.
- **Cache**: `configs/hangul_cache.toml` (layout JSON, `token_split syllable`, tail off), 4.1 GB.
- **Run**: `configs/hangul_s500.toml`. Initialised from C (`c_fill_s400`); otherwise the
  sanitize_ft_c recipe (muon, lr 5e-5, rank 32, int8 plan, EMA off) with `region_weight 2`
  and no gate (training has `cut` only). 500 steps, 2.49 s/step, 20.4 min, peak 13.3 GB.
  `output/ckpt_moss_i2i/hangul_s500[/…_sN].safetensors`.
- **Eval** (`src/hangul_eval.sh`): swap_text's two Korean eval pages (`589883_0004_ko`,
  `737486_0019_ko`) on seeds 0–2. The settings are those of the ksplit e2 renders (bf16 DiT,
  fp32 stream, swap 10). Arms:
  - `hangul_s500`: the block alone
  - `stack_e2_hangul_s500`: ksplit e2 + the block as one rank-64 concat
    (`src/concat_lora.py`; each update exact, e2's metadata)
- **Read** (`src/hangul_read.py`): `grad_in` is the mean Sobel magnitude of luma inside the
  caption's boxes (+4 px), divided by the same on the Korean edition page. The edition of
  `737486` is a soft scan, which e2 out-sharpens (1.30), so the number ranks arms against
  each other and does not measure distance from the edition.

## Reads

| arm | 589883 s0 / s1 / s2 | 737486 s0 / s1 / s2 |
|---|---|---|
| ksplit e2 | 0.89 / 0.91 / 0.91 | 1.30 / 1.30 / 1.32 |
| block alone | 0.78 / 0.87 / 0.79 | 0.93 / 0.81 / 0.90 |
| e2 + block | 0.92 / 0.92 / 0.96 | 0.89 / 0.88 / 1.15 |

Per glyph, from the sheets:

- **Syllables.** Every arm writes every captioned syllable right on seeds 0–2. This includes
  `분인가요`, and `벌거`, which is drawn as the caption gives it (an OCR misread of `씻겨`).
- **Block alone.** Each line is legible, but the strokes are thinner and greyer than e2's.
  The lettering looks like the synthetic data:
  - lines broken one word per line (`질` / `속이`)
  - a smaller middle line (`이건` on 589883 s0)
  - letter-spaced `이 건` / `원가요 ?` (s1)
  - `벌거 주기?` in magenta (s0)

  Outside the caption, the two Japanese balloons the caption leaves out (`きー気になる…`,
  `うわっすごい…`) come out as pseudo-glyphs (`page_589883_s1.png`). The block never saw
  uncaptioned text on a page stay put: every OCR line of a synthetic page was in its caption.
- **Stack.**
  - `737486` s0 / s1: `좀··더` and the long balloon are blurred and doubled. This is the one
    arm whose strokes go softer than e2's.
  - `589883`: `뭔가요?` is set heavier and larger than its line. On s1, `벌거 주기?` sits on
    an orange plate.
  - `737486` s2: close to e2.
  - The uncaptioned Japanese stays (e2's copy holds).

## What it means

- Font lettering as data teaches the block a font's layout (breaks, sizes, colours of
  the sample), not crisper strokes. Where e2 already letters in a manga gothic, the block
  alone letters thinner.
- The exact concat does not compose. The two blocks share the reference path (C's copy and
  e2's), and their sum overshoots: the doubled strokes and the orange plate. This is P4's
  interference, read without a fine-tune.
- The block alone is not a swap_text model either: it rewrites text the caption leaves out.

## Not tested

- The s100–s400 saves; a run from kaiming; the block on colorize v5 (`to_colorize4`).
- Line breaks taken from the edition (render the new string into the original box's line
  structure) and one font per page. Both would remove the layout the block learned, but
  not the thin weight.
- Japanese / English text left out of the synthetic captions (the fix for the garbling).
