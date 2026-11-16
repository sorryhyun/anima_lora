# MOSS: grounding prompt tokens in RoPE space (CDPE, after SpaG-DiT)

Status: proposal, 2026-11-16. Nothing is built or probed. Source: the NeurIPS 2026 poster
of SpaG-DiT ("Enhancing Spatial Grounding for Diffusion Transformers", Westlake / Alibaba
Yuvion, poster 153058). No preprint was found, so δ, the multi-token rule, and the
train / inference handling below are inferred from the poster figures.

## What SpaG-DiT does (as read from the poster)

- **Base model: Qwen-Image.** Its MSRoPE places text tokens on the diagonal `(n, n)`, after
  the image grid.
- **VLM-GG.** A VLM gives each grounded span (a scene-text word, a novel concept) a
  normalised position. The poster's example is `"جميع": [0.31, 0.14]`. That position is
  quantised to a 3×3 grid, offset `O = (O_x, O_y) ∈ {-1, 0, 1}²`.
- **CDPE.** Ungrounded tokens keep their diagonal position. A grounded token moves from
  `(n, n)` by `g(n, O)` into its cell around a symmetric midpoint anchor `(m, m)`, which the
  poster calls a "bounded, sign-symmetric perturbation". The poster rejects two alternatives:
  - a plain offset, which "perturbs the diagonal monotonically";
  - image coordinates for text, which "breaks the text sequence".
- **Training-time only.** Inference is plain t2i with no offsets. The attention figure shows
  grounded tokens attending to their own region. The poster claims faster learning of
  multilingual scene text (Hindi, Arabic, JA, KO, Bengali) and of novel concepts. It reports
  GenEval 84.52 → 85.95 against SFT, with Pos 77 → 78, at 512², no best-of-N.

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

| tokens | now `(f, h, w)` | CDPE 3×3 (poster) | CDPE continuous |
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
  only, so they would use the poster's 3×3 offsets (left / centre / right × top / middle /
  bottom).

## Open choices

- **Scale.** True cell coordinates, or a shrunk δ (the poster's "bounded" offsets). Base
  only ever saw text at `(0, 0)`, and a displacement of 14 cells rotates the low bands by
  several radians.
- **Multi-line spans.** One centre per item, or split the bbox's height across lines
  (`키리노` above `힘내라`).
- **Token extent.** The text value only, or also the item's `caption` / `label` tokens.
- **Spatial extent.** Each token gets the box centre, or tokens spread along the line's
  width. The poster uses one offset per span.

## Plan

**P0, zero-shot on Base** (int8, fp32 stream, ≤10 min).
- Prompts: t2i layout-JSON with two or three boxed text items at distinct positions.
- Arms:
  1. plain;
  2. CDPE continuous;
  3. CDPE 3×3;
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

## Implementation notes

- **RoPE.** `_RoPE.forward` takes an optional `text_hw (L, 2)`. With it absent, the output is
  bit-identical, so the per-sample forward stays op-for-op with the pipeline.
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
