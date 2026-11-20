# MOSS: grounding prompt tokens in RoPE space (CDPE, after SpaG-DiT)

Status: proposal, 2026-11-16. A training-only form on position clauses (and boxed text lines) is built and shipped
(`clause_hw`, [runs](../reports/clause_hw.md)); P0 – P2 and the extension are not built. Source: SpaG-DiT
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
  at `(0, 0)`. `clause_hw` follows it since 2026-11-20: a clause's tags move and its `On the left`
  header stays (`clause_header` off; the user's decision, [runs](../reports/clause_hw.md)).
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

## Extension: positions for objects outside the clauses (proposed 2026-11-20)

The user's idea. Today a token moves only if it sits in a position clause (a subject) or inside a
boxed text line. A thing the caption names anywhere else stays at `(0, 0)`: a prop in the shared
bag, furniture, a held item. SAM3 can find such a thing from its tag, and its tokens can take the
mask's centroid the same way a clause takes its subject's (`clause_centres`).

Measured, for scale:

- Position-stage records in `workspace/captions/position/analysis` (local), 1,182 images: 528
  `proposed`, 536 `skip:too-few-instances`, 88 `skip:count-mismatch`, 30
  `skip:no-discriminative-tags`. A single-subject image gets no clause, so nothing in its caption
  moves.
- ama: 40 of 106 captions carry no clause. casbox: 3 of 30.
- Paper, Table S1: attention-centroid priors keep about 85 % of the OCR-box gain, so a prior does
  not have to be exact.

What gets a position:

| tokens | position | status |
|---|---|---|
| a subject's clause | 3×3 cell (default), or the SAM3 mask centroid (`clause_centres`) | built; no winner on caststation 2026-11-20, the cell stays |
| a boxed text line | the OCR box centre (`text_box_hw`) | built |
| an object tag in the shared bag (a prop, furniture, a vehicle, an animal) | centroid of the SAM3 mask its tag finds | this proposal |
| an object tag inside a subject's clause (a held phone, a straw) | its own mask centroid in place of the subject's | this proposal |
| a part of a character (eyes, hair, a hand) | stays with its clause | left out: the user reads part-level positions as ambiguous |
| scene and whole-image tags (`white background`, `indoors`, the artist, the rating) | `(0, 0)` | unchanged |

Rules to settle before building:

- **Which tags count as objects.** A list drawn from anime_tools' tag groups, fixed before the run.
  A tag off the list is never prompted.
- **One tag, several masks** (three `chair`s): the tag has one token span and no single centre. It
  stays where it was (`(0, 0)` in the bag, the subject's centre in a clause).
- **No detection, or a score under the stage's threshold:** the tag stays where it was.
- **Where it is computed.** In the position stage (anime_tools), beside the subject pass, written to
  the same analysis record as an `objects` list (tag, box, score) with its own label indices. The
  trainer reads it as it reads the subjects. This needs an anime_tools release and a pin bump.
- **Single-subject captions.** Whether the subject's own tags take the subject's centroid when the
  caption has no clause is a separate choice: the bag does not say which tags belong to the subject.

Order: the centre-against-cell read on caststation (2026-11-20) found no winner overall, the
centroid ahead only on prompts that carry clauses. The finer positions come after the same A/B on a
second set (ama), not before.
Read: the clause A/B's sheets, plus prompts that name a held item or a prop (the caststation
`drinking straw` / `holding phone` cases), rendered without offsets.

## Runs

Run records moved to [`../reports/clause_hw.md`](../reports/clause_hw.md): the first caststation runs
(2026-11-16, clauses alone and with text boxes), the ama controls (2026-11-17: inference offsets,
the mirrored-cell arm, the user's read that set `clause_hw = 1.0`), and the placement and header A/Bs
(2026-11-20: mask centroids against cells, header moving against staying).

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
