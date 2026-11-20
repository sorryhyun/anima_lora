# Caption rows pick up the page and the image reads it back in blocks 26–41; the template rows run a fixed trajectory

Measured 2026-11-14 on colorize v5 (`moss_colorize_v5`, isolated reference at `ref_timestep` 0)
and on Base. Probes, runs and per-page arrays: `_archive/bench/moss_text_stream/` (`probe.py`,
`caption_rows.py`, `README.md`).

## Setting

MOSS is single-stream: the text rows pass all 42 blocks with the image rows, read them, and
are dropped at the head. The question is what those rows hold on the way and whether the
image rows use it.

Teacher-forced, one forward per condition, σ 0.95 / 0.6 / 0.3, generation's DiT (int8 block
linears, fp32 stream). A probe forward replaces the text rows fed into a block range and leaves
the image rows alone; `dv` is the mean squared difference of its velocity from the live
forward's.

Two captions:

- **Empty** (160 illustration pairs, 32 in the swaps). An illustration's bare caption is the
  empty prompt: the 3 + 7 chat-template rows, identical on every page at block 0 (maximum
  difference between pages 0.0). Whatever differs between pages further up was read from the
  image or the reference.
- **Layout JSON** (24 manga pages, 63–936 rows, median 361): `comic`, a box per character with
  its hair / eye / clothing colours, a box-only text item per OCR line. To get the same tokens
  with another page's state, page A's caption is run on page B and those rows are fed into A's
  forward.

Controls, on every run: the forward's own rows (`dv` 0 when stored at the stream's precision),
own rows × (1 + 1e-3·ε), and own rows plus an isotropic draw with the swap's row norms.

## The floor

A 1e-3 jitter of the text rows moves the velocity by 0.00018 / 0.00039 / 0.00103 (empty
caption) and 0.00019 / 0.00044 / 0.00109 (layout JSON). The same page's rows under another
noise draw, ten to thirty times further from the live rows, give the same number on the empty
caption (median ratio 1.01–1.08). The response does not grow with the change: an input that
differs at all re-draws the rounding inside the blocks. The bf16 DiT has a floor too
(0.00010 / 0.00021 / 0.00053); the fp32 stream does not remove it. Every swap below is read
against this floor and against the isotropic draw.

## Template rows (empty caption)

**The rows move far, the same way on every page.** Cosine to the block-0 input falls from 0.76
after block 0 to 0.17 after block 41. At most 0.3 % of the rows' energy differs between pages
through block 40 (0.5–0.7 % on block 41).

**The image needs that trajectory.** Feeding the block-0 input at every block takes the flow
MSE from 0.195 / 0.264 / 0.452 to 0.462 / 0.912 / 1.320 (`dv` 0.266 / 0.648 / 0.873). The full
colour caption breaks the same way (`dv` 0.264 / 0.628 / 0.848).

**The page is in the 0.3 %.** Held-out ridge probes (5 folds over pages) from the ten rows:

| σ | palette, rows after block 0 → best block | palette, mean of `x_t` | layout, rows after block 0 → best block | layout, mean of `x_t` |
|---|---|---|---|---|
| 0.95 | 0.48 → 0.56 | 0.25 | 0.39 → 0.44 | 0.14 |
| 0.6 | 0.61 → 0.69 | 0.51 | 0.41 → 0.41 | 0.21 |
| 0.3 | 0.61 → 0.67 | 0.52 | 0.39 → 0.40 | 0.22 |

`palette`: spatial mean of target − reference latent (128-d). `layout`: 4×4 cell RMS of the
reference latent. Both hold to block 37 and fall on block 41. The page-specific part is mostly
the page, not the noise draw: the same page under another draw sits at 0.03–0.25 of the
distance to another page.

**The image does not read it back.** Another page's rows at every block:

| rows fed | σ 0.95 | σ 0.6 | σ 0.3 |
|---|---|---|---|
| own × (1 + 1e-3·ε) | 0.00018 | 0.00039 | 0.00103 |
| same page, other noise draw | 0.00019 | 0.00040 | 0.00104 |
| another page | 0.00026 | 0.00059 | 0.00179 |
| own + isotropic draw of that norm | 0.00029 | 0.00050 | 0.00129 |

Above the isotropic draw on 14 / 9 / 7 of 32 pages, median ratio 0.90 / 0.95 / 0.96. What is
above the floor sits in blocks 26–41 (0.00023 / 0.00059 / 0.00178); blocks 1–13 and 14–25 are
at the floor.

**Where the rows read.** Text queries' attention mass, σ 0.95:

| block | image keys | reference keys | template keys |
|---|---|---|---|
| 0 | 0.37 | 0.38 | 0.25 |
| 13 | 0.25 | 0.12 | 0.63 |
| 21–37 | 0.07–0.11 | 0.01–0.04 | 0.83–0.93 |
| 41 | 0.69 | 0.08 | 0.22 |

Block 41's text output reaches no loss; its numbers are untrained. Image queries put 0.05–0.46
of their mass on the template and, under an illustration's full caption, at most 0.02 on the
caption keys.

## Caption rows (layout JSON)

| rows fed | σ 0.95 | σ 0.6 | σ 0.3 |
|---|---|---|---|
| own | 0 | 0 | 0 |
| own × (1 + 1e-3·ε) | 0.00019 | 0.00044 | 0.00109 |
| same page, other noise draw | 0.00054 | 0.00082 | 0.00139 |
| **the caption run on another page** | **0.00137** | **0.00234** | **0.00388** |
| … caption rows alone | 0.00133 | 0.00173 | 0.00324 |
| … template rows alone | 0.00021 | 0.00095 | 0.00174 |
| … blocks 1–13 / 14–25 alone | 0.00019 / 0.00020 | 0.00044 / 0.00044 | 0.00108 / 0.00107 |
| … blocks 26–41 alone | 0.00133 | 0.00230 | 0.00383 |
| own + isotropic draw of that norm | 0.00054 | 0.00120 | 0.00239 |
| the bare caption (character items dropped) | 0.00136 | 0.00067 | 0.00146 |

- **The image reads the page-derived part of the caption rows.** The swap is above the
  isotropic draw on 23 / 23 / 21 of 24 pages (median ratio 2.2 / 1.7 / 1.2) and above the noise
  swap on 24 / 23 / 24.
- **All of it in blocks 26–41**, the range where Base's item-content attention follows a box
  ([bbox_grounding.md](bbox_grounding.md)) and where the sanitize LoRA edits.
- **The caption tokens carry it.** The template rows alone are at the floor at σ 0.95.
- **It is as large as the caption's content.** The swap moves the velocity as much as dropping
  every character item at σ 0.95 and 1.7–2.4 times as much at σ 0.6 and 0.3 (median over pages),
  and it costs flow MSE at every σ (0.1758 → 0.1769, 0.2707 → 0.2723, 0.5182 → 0.5207); dropping
  the character items costs it at σ 0.95 only.
- **The caption rows turn page-specific late.** Share of their energy that differs between the
  two pages, σ 0.95: 0.6 % after block 13, 1.0 % after 30, 3.0 % after 35, 13.5 % after 40.
  The template rows of the same forwards: 0.3 %, 0.04 %, 0.1 %, 0.6 %.
- Inside the character boxes the swap moves the velocity as it does over the page (0.00145
  against 0.00137 at σ 0.95).

## Base, no LoRA, no reference

`caption_rows.py --base_t2i`: the first 12 of the 24 pages with the same captions, Base t2i on
the colour target (flow MSE 0.686 / 0.379 / 0.531). v5 on the same 12 pages beside it.

| rows fed | Base, σ 0.95 / 0.6 / 0.3 | v5, same pages |
|---|---|---|
| own × (1 + 1e-3·ε) | 0.00013 / 0.00050 / 0.00119 | 0.00017 / 0.00042 / 0.00110 |
| same page, other noise draw | 0.00183 / 0.00134 / 0.00190 | 0.00052 / 0.00083 / 0.00135 |
| **the caption run on another page** | **0.00551 / 0.00473 / 0.00555** | 0.00139 / 0.00304 / 0.00484 |
| … blocks 26–41 alone | 0.00592 / 0.00459 / 0.00548 | 0.00132 / 0.00299 / 0.00478 |
| own + isotropic draw of that norm | 0.00128 / 0.00199 / 0.00342 | 0.00049 / 0.00138 / 0.00265 |
| the bare caption | 0.01229 / 0.00124 / 0.00212 | 0.00158 / 0.00064 / 0.00148 |

- **The read-back is Base's.** The swap is above the isotropic draw on 12 / 12 / 10 of 12 pages
  (median ratio 4.6 / 1.7 / 1.7), in blocks 26–41 again; the caption rows' page share reaches
  9.6–16 % after block 40. The colorize LoRA did not create the path.
- **At σ 0.95 Base leans on it four times as much as v5** (0.0055 against 0.0014) and reads the
  noise draw's state too (14 times the floor). v5 has the reference in every forward, and the
  two runs differ in the reference as well as the LoRA, so the drop is not the LoRA's alone.
- On Base at σ 0.95 blocks 14–25 are also above the floor (0.00074), an eighth of 26–41's.

## Reading

The template rows are registers: a per-block, per-σ value the image rows depend on, with a
page summary riding on it that nothing reads. A caption token is different. Through block 25
its row is the caption's content; from block 26 it holds what the page has at the place the
item names, and the image rows read that. Base does this without a LoRA, most at high σ where
`x_t` holds little; an i2i LoRA with a reference in the sequence uses it less there.

## Not tested

- Which tokens of an item carry the page state (coordinates, colour words, JSON syntax), and
  what the state is: no probe was fitted on the caption rows.
- A matched caption on the donor page. Page A's caption on page B points its boxes at the
  wrong content, so the fed rows also hold B's response to that.
- The cache drew captions at `tag_dropout` 0.5; v5 trained at 0.8.
- Which rounding sets the floor (int8 activations or the bf16 attention).
- The vision-tower rows a self-distill teacher reads
  ([self_distill.md](../experimental/self_distill.md)).
