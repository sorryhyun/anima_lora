# MOSS: grounding prompt tokens in RoPE space (CDPE, after SpaG-DiT)

Status: proposal, 2026-11-16. A training-only form on position clauses (and boxed text lines) is built and ran twice
([First run](#first-run-training-only-on-position-clauses-2026-11-16)); the rest is not built. Source: SpaG-DiT
("Enhancing Spatial Grounding for Diffusion Transformers", Westlake / Alibaba Yuvion, NeurIPS 2026),
camera-ready paper and supplementary material, read 2026-11-19. Code is announced at
`github.com/Alibaba-VELLDEPTH/SPaG-DiT` and not yet released. The design below was first drawn
from the poster (poster 153058). Where the paper differs from that reading, the section below follows the paper.

## What SpaG-DiT does (paper + supplementary)

- **Base model: Qwen-Image.** Its MSRoPE places text tokens on the diagonal, after the image
  grid; image positions are centred on 0.
- **Spatial priors are continuous.** Each grounded token `i` gets `P(i) = (x*_i, y*_i)` in
  image-centred patch-grid coordinates (x right, y down, image spans `[−W/2, W/2] × [−H/2, H/2]`,
  patch centres at half-integers on an even grid). No normalisation to `[0, 1]` and **no 3×3
  quantisation**; the poster's `"جميع": [0.31, 0.14]` is an illustration in Fig. S1 only. Tokens
  without a prior get `(0, 0)`.
- **Which tokens.** "Each text token to be rendered or bound". MultiLingual captions are Gemini
  prose with the OCR strings in quotes (`In the upper right corner …, the Korean text "…"`); the
  tokens inside the quotes take the OCR box. COCO is re-captioned so that each annotated object's
  class name appears once in quotes (`a "person" … a "surfboard"`), and those tokens take the bbox.
  The position words themselves are not moved.
- **Prior sources (S2).** EasyOCR boxes (MultiLingual), COCO boxes (GenEval training), and
  **VLM-GG** when no annotation exists: the attention of each text token over the image patches
  in the N-th decoder layer of Qwen2.5-VL, fed the training image and caption; the prior is
  its attention-weighted centroid (Eq. S1), mapped to the DiT's patch units if the grids differ.
- **CDPE (Eq. 4).** `g(i) = (t_base + i + m + [λ_y ŷ_i], t_base + i + m + [λ_x x̂_i])` with
  `m = −t_base + ⌊L/2⌋` and `[·]` truncation toward zero. Every text token's diagonal is shifted
  by `m`, grounded or not, at training and inference. "Bounded" = the coordinates lie inside the
  grid; "sign-symmetric" = `[−z] = −[z]`. λ values are given in neither the paper nor the
  supplementary. As printed, the reference index is `i + ⌊L/2⌋`, which is not centred on 0
  (possibly a sign slip; it does not matter on MOSS).
- **Two rejected alternatives (Eqs. 1–2).** OPE = the diagonal plus the offset; IPE = image
  coordinates alone, dropping `i`.
- **Training-time only (Alg. S1).** Inference is plain t2i with `(0, 0)` everywhere.
- **Setup.** LoRA rank 256, α 256, LR 5e-5, batch 2, BF16, short side 512 with 10 aspect
  buckets; 50k MultiLingual images, comparisons at 25k / 75k steps; H100.

Results:

| read | vanilla SFT | SpaG-DiT | other |
|---|---|---|---|
| MultiLingual, 5 scripts, Sen.ACC / NED (Table 1) | 0.105 / 0.284 | 0.174 / 0.433 | |
| Korean ablation, Sen.ACC (Table 4) | 0.408 | CDPE 0.655 | IPE 0.646, OPE 0.630 |
| Korean, prior source, Sen.ACC (Table S1) | 0.408 | OCR boxes 0.655 | VLM-GG 0.611 |
| GenEval overall / position, 512², no best-of-N (Table 2) | 84.52 / 77 | 85.95 / 78 | |
| Creature, human accuracy, 145 samples (Table 3) | 69.0 % | 75.9 % | |

- The PE form moves Sen.ACC by 0.01–0.025, against +0.24 for having positions at all.
- Attention-centroid priors keep about 85 % of the OCR-box gain.
- No error bars; one run per arm.

In the paper's terms, MOSS's text tokens are IPE on `(h, w)` with the sequence kept on `f`:
CDPE's two goals, without the diagonal shift.

## Why MOSS fits better than Qwen-Image

MOSS RoPE has four axes, `(r, f, h, w)` (`library/moss/model.py`, `_RoPE.forward`):

| tokens | r | f | h, w |
|---|---|---|---|
| text token i | 0 | i | **0, 0** |
| image | 0 | text length L | its grid, centred on 0 |
| i2i reference (`aligned`) | 1 | L | the image's grid |

- **The order lives on f alone.** Text tokens all sit at the image centre `(h, w) = (0, 0)`,
  which is the symmetric midpoint anchor CDPE builds by hand. A grounded token's `(h, w)`
  can move without touching its sequence position. Qwen-Image's problem, a diagonal that
  carries both order and space, does not exist here.
- **Today the spatial link is read from text.** A token sits equally far, in RoPE terms,
  from every patch at the same distance from the centre. The only route from "이 글자" to
  "이 말풍선" is the model reading the bbox digits in the layout JSON.

## The proposal

For each grounded span in a layout-JSON prompt, set `(h, w)` of its tokens to the span's
bbox centre in latent-cell units (`/16`). Every other token stays at `(0, 0)`, and f is
untouched.

Worked example: swap_text, `318615_0056_ko`, page 1024 × 1456, grid 64 × 91.

```json
{"label":"text","caption":"text","text":"키리노\n힘내라","bbox":[124,-289,224,-160]}
```

The bbox is `abs_centralized yxyx` with centre y = 174, x = −224.5 px, which is
`(h, w) ≈ (+10.9, −14.0)`.

| tokens | now `(f, h, w)` | 3×3 cell (clause form) | box centre (paper's form) |
|---|---|---|---|
| JSON structure, `"label"`, `"bbox"` digits | (i, 0, 0) | same | same |
| `키` `리` `노` `힘` `내` `라` | (i, 0, 0) | (i, 0·δ, −1·δ) | (i, 10.9, −14.0) |

The balloon's own patches are then at `Δh ≈ Δw ≈ 0` from its syllables in every spatial band.

### Unlike the paper, keep it at inference

The paper drops CDPE at inference because a t2i prompt has no positions. **A layout-JSON
prompt does.** The same offsets can be computed at generate time, so there is no train /
inference mismatch. This applies to every prompt that carries image positions:

- **swap_text and OCR rendering.** Each text item has a bbox. The syllables (`token_split
  syllable`) get the box centre.
- **The other direction: layout control in t2i.** Every layout-JSON item with a bbox
  (character, object, text) grounds its `caption` / `text` tokens. Today the bbox reaches
  the model only as digits. This proposal gives it a geometric channel.
- **Thin position clauses** (`On the left, akita neru, …`). These carry coarse positions
  only, so they use 3×3 cell centres (left / centre / right × top / middle / bottom). The
  paper's priors are continuous; the cells are what these captions can name.

## Open choices

- **Scale.** The paper uses true patch coordinates times λ, with λ unpublished. Base only
  ever saw text at `(0, 0)`, and a displacement of 14 cells rotates the low bands by several
  radians.
- **Multi-line spans.** One centre per item, or split the bbox's height across lines
  (`키리노` above `힘내라`).
- **Token extent.** The text value only, or also the item's `caption` / `label` tokens. The
  paper moves only the quoted tokens (the string, the class name) and leaves the position words
  at `(0, 0)`; `clause_hw` moves the whole clause, its `On the left,` header included.
- **Spatial extent.** Each token gets the box centre, or tokens spread along the line's
  width. OCR boxes give one centre per box; VLM-GG gives each token its own centroid.
- **Prior source for captions without positions.** VLM-GG-style attention centroids would give
  tags a position where no clause or box names one (40 of 106 ama captions carry no clause).
  Table S1 puts attention priors at about 85 % of the OCR-box gain.

## Plan

**P0, zero-shot on Base** (int8, fp32 stream, ≤10 min).
- Prompts: t2i layout-JSON with two or three boxed text items at distinct positions.
- Arms:
  1. plain;
  2. CDPE continuous;
  3. 3×3 cell centres;
  4. **jitter**: the same offsets permuted across items;
  5. **equal-norm random** offsets.
- The jitter and random arms are required. Position probes on MOSS sit on a floor of
  about 1e-4…1e-3 (`project_moss_patching_floor`).
- Read:
  - OCR v6 hit rate inside the named box, and how far each item's centroid is from its box;
  - the grounded tokens' attention mass on their box, in a few blocks;
  - renders.
- This tells whether Base already uses `(h, w)` on text keys at all.
- Expected failure: Base, never trained on off-centre text, renders worse with no gain in
  placement. In that case P1 decides, and P0 only sets δ.

**P1, swap_text LoRA A/B.**
- The current swap_text recipe (int8 stream, copy-block init) at a fixed short budget,
  CDPE on (train and generate) against off. Same seed and data.
- Read: per-glyph OCR against the intended text, the box hit rate, and renders.
- The paper's main claim is training efficiency, so also read the step count at which the
  off arm reaches the on arm's OCR.

**P2, t2i layout control** (only if P1 passes). Use the casbox OCR boxes
(`casbox_1115_1226`) and layout-JSON object boxes. Generate with offsets from a user-written
layout and read placement.

## First run: training-only on position clauses (2026-11-16)

The paper's setting instead of P0 / P1: coarse positions, offsets during training only,
plain prompts at inference. `TrainRequest.clause_hw` (0 = off) moves the tokens of each position
clause (`On the top right, …`) to their 3×3 cell's centre, a third of the grid side per cell times
`clause_hw`, in the blocks' RoPE. The text refiner keeps `(0, 0)`: no adapter reaches it. Text
clauses are not moved. Code: `library/moss/training/grounding.py`.

Caststation (30 images, the `casbox_1115_1226` recipe, 480 steps, same seed). Arms: the control,
the control trained again (`casboxr_1116_1149`), `clause_hw` 0.25 (`cashwq_1116_1110`, about 5
cells at 1024 px) and 1.0 (`cashw_1116_1110`, about 21 cells). Renders without offsets, sheets in
`output/tests/caststation_thin/cashw_sheets/`.

| read | control | retrain | 5 cells | 21 cells |
|---|---|---|---|---|
| training loss at σ 0.85–0.95, against the control | | +0.0002 | −0.0002 | +0.0015 (0.45 %) |
| training loss below σ 0.85 | | same | same | same |
| boxed `初音ミクです` written as the plain string (Base: 0 / 6, it draws the official logo) | 2 / 8 | 2 / 8 | 0 / 8 | 8 / 8 |
| general tag prompts, left / right placement in clause prompts | | same | same | same |

- The loss is paired by step (same samples, σ and noise in every run). 5 cells cannot be told from
  a retrain. 21 cells is felt only where layout is decided.
- Invented cm ruler labels on the height-chart prompt (control 8 / 8) and one broken hand looked
  like offset effects until the retrain: it draws the rulers 1 / 8 and the hand whole.
- Open: other strings and characters, a longer set.

**Text lines on their boxes too** (`cashwb_1116_1219`, `clause_hw 1` + `text_box_hw 1`). The
tokens inside each boxed text line's quotes (`"…" [y0,x0,y1,x1]`) take the box centre in latent
cells, beside the clause offsets: 27 / 30 captions, 4,387 rows off `(0, 0)`, 171 of them text. Same
recipe, renders without offsets.

| read | control | retrain | clause only | clause + box |
|---|---|---|---|---|
| training loss at σ 0.85–0.95, against the control | | +0.0002 | +0.0015 | +0.0005 |
| training loss at σ ≥ 0.95 | | +0.0007 | +0.0028 | +0.0006 |
| boxed `初音ミクです` written plain (`prompts_text_more`, 6 seeds) | 1 / 6 | 2 / 6 | 6 / 6 | 1 / 6 |
| invented cm ruler labels, height chart (2 seeds) | 2 / 2 | 0 / 2 | 0 / 2 | 2 / 2 |
| general tag and clause prompts (16) | | same | same | same |

Adding the boxes returns every read to the control: the high-σ loss cost, the plain string and the
ruler habit. Neither arm moves the general renders. Sheets: `output/tests/caststation_thin/cashwb_sheets/`.

## Controls for the ama read (2026-11-17)

Is the ama gain grounding, or a perturbation of half the prompt? On the ama cache 66 of 106 full
captions carry clauses and 31 % of all prompt tokens move (casbox: 27 of 30, 50 %).

- **Offsets at inference** on `amahw_1116_1239`, clause prompts: none, true cells
  (`--clause_hw 1`, `amahw_clause_inf`), mirrored cells (`--clause_hw -1`, `amahw_clause_infmir`).
  The layout does not follow the cells: with mirrored offsets the close-up stays left and the figure
  right in 6 / 6 clause renders. Either sign changes a render by about a re-roll. The user's read:
  no difference worth having. Generation stays without offsets.
- **Mirrored-cell training arm** `amamir_1117_1257`: `clause_hw = -1.0` (each clause at the cell
  opposite its own, same distance from `(0, 0)`), otherwise the `amahw` config; 66 / 106 captions,
  5,301 rows off `(0, 0)`. Rendered without offsets as the other arms (`amamir_anima`,
  `amamir_clause`). Trained a day after `amahw`, with commits to the MOSS training code between.
- **Panty-line prompts** (`prompts_pantyline.txt`: `pantylines` + `tight pants`, a tag in none of
  the 106 training captions): `amahw_pl`, `amamir_pl`. The `amathin` / `amathinr` checkpoints are
  gone, so this set has no arm without `clause_hw`.
- **The user's read (2026-11-17): `amahw` beats `amamir`.** The true-cell arm is clearly better
  than the mirrored one, so the ama gain needs each clause at its own cell: grounding, not
  perturbation. One mirror arm against one true arm, and no retrain of either.
- **Inference offsets stay open.** Whether `--clause_hw 1` at generation helps the true-cell LoRA
  is not settled (the first read found no difference worth having). Generation keeps running
  without offsets.
- From reading the code: the epoch sample renders apply the offsets (`training/sampling.py`) while
  `generate` runs without them; on a `tags` cache a clause caption never trains unshifted; a row of
  four or more columns is spread over ±⅓ of the width (five columns reach ±0.4).

Renders and sheets: `output/tests/ama_clause/` (`sheets_1117_full/`, 1024 px per render). Config:
`_archive/configs/pe_lora/amamir_1117_1257.toml`.

## Implementation notes

- **RoPE.** `_RoPE.forward` takes an optional `text_hw (L, 2)`. With it absent, the output is
  unchanged, so the per-sample forward stays op-for-op with the pipeline.
  `forward_batched` takes one per sample.
- **Span → token map.** Find the grounded spans' character ranges in the templated prompt
  and map them to token ids. The `token_split` wrapper raises on offset-mapping calls, so
  syllable mode needs its own span bookkeeping.
- **Cache.** The TE cache does not change: the offsets are positions, not embeddings. The
  per-token `text_hw` is either stored beside the cache or recomputed from the caption at
  load. The mode and δ are stamped in the LoRA, and generate follows the stamp.
- **ComfyUI.** The MOSS Toolkit's text encode / sampler needs the same `text_hw` path
  (`custom-nodes` skill).
- **Anima.** Does not apply. Text enters through cross-attention with no shared RoPE frame
  between image queries and text keys.
