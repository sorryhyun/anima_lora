# clause_hw: run records

Position-clause tokens on RoPE's h / w axes during training (`clause_hw`, after SpaG-DiT's CDPE).
Method: [`../methods/clause_hw.md`](../methods/clause_hw.md); design and the paper:
[`../proposal/cdpe_text_grounding.md`](../proposal/cdpe_text_grounding.md). Three rounds: the first
caststation runs (2026-11-16), the ama controls that set the default (2026-11-17), and the placement and
header A/Bs on caststation (2026-11-20).

## 2026-11-16, caststation: training-only on position clauses

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

Text lines on their boxes too (`cashwb_1116_1219`, `clause_hw 1` + `text_box_hw 1`). The
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

## 2026-11-17, ama: controls for the read

Is the ama gain grounding, or a perturbation of half the prompt? On the ama cache 66 of 106 full
captions carry clauses and 31 % of all prompt tokens move (casbox: 27 of 30, 50 %).

- Offsets at inference on `amahw_1116_1239`, clause prompts: none, true cells
  (`--clause_hw 1`, `amahw_clause_inf`), mirrored cells (`--clause_hw -1`, `amahw_clause_infmir`).
  The layout does not follow the cells: with mirrored offsets the close-up stays left and the figure
  right in 6 / 6 clause renders. Either sign changes a render by about a re-roll. The user's read:
  no difference worth having. Generation stays without offsets.
- Mirrored-cell training arm `amamir_1117_1257`: `clause_hw = -1.0` (each clause at the cell
  opposite its own, same distance from `(0, 0)`), otherwise the `amahw` config; 66 / 106 captions,
  5,301 rows off `(0, 0)`. Rendered without offsets as the other arms (`amamir_anima`,
  `amamir_clause`). Trained a day after `amahw`, with commits to the MOSS training code between.
- Panty-line prompts (`prompts_pantyline.txt`: `pantylines` + `tight pants`, a tag in none of
  the 106 training captions): `amahw_pl`, `amamir_pl`. The `amathin` / `amathinr` checkpoints are
  gone, so this set has no arm without `clause_hw`.
- **The user's read (2026-11-17): `amahw` beats `amamir`.** The true-cell arm is clearly better
  than the mirrored one, so the ama gain needs each clause at its own cell: grounding, not
  perturbation. One mirror arm against one true arm, and no retrain of either.
- Inference offsets stay open. Whether `--clause_hw 1` at generation helps the true-cell LoRA
  is not settled (the first read found no difference worth having). Generation keeps running
  without offsets.
- From reading the code: the epoch sample renders applied the offsets (`training/sampling.py`) while
  `generate` runs without them (the sample renders run without them since 2026-11-20); on a `tags` cache a clause caption never trains unshifted; a row of
  four or more columns is spread over ±⅓ of the width (five columns reach ±0.4).

Renders and sheets: `output/tests/ama_clause/` (`sheets_1117_full/`, 1024 px per render). Config:
`_archive/configs/pe_lora/amamir_1117_1257.toml`.

## 2026-11-20, caststation: where the clause sits, and whether its header moves

Two A/Bs on the casbox recipe (30 images, 480 steps, same cache `moss/casbox_tags_sure` and seed, renders
without offsets). On the header the two reads and the user's decision differ; both are recorded below.

### Facts

Code added that day: `clause_centres` (each clause at the centroid of its subject's SAM3 mask, read
from the position stage's `_position_report/analysis`), `clause_header` (whether `On the left` moves
with its tags), and the epoch sample renders run without offsets, as `generate` does.

Cell centre against the subject, in fractions of the image side:

| set | clauses | cell vs SAM3 mask centroid, mean abs (h / w) | box centre vs mask centroid |
|---|---|---|---|
| caststation (`caststation_dil16`) | 79 in 27 captions | 0.087 / 0.087 (5 – 6 cells of 64) | 0.033 / 0.022 |
| ama (`workspace/captions/position/analysis/ama_mitsuki`, local) | 150 in 64 captions | cell vs box centre: 0.053 / 0.134 | not measured |

The position names are the subjects' order among themselves (`assign_positions`), not thirds of the
image: on ama `left` subjects centre at −0.16 of the width, the cell at −0.33.

Arms (configs in `_archive/configs/pe_lora/`, local):

| run | placement | header | rows off `(0, 0)` |
|---|---|---|---|
| `cascell_1120_1056` | 3×3 cell | moves | 4,216 |
| `cascen_1120_1056` | mask centroid (79 / 79 clauses) | moves | 4,432 |
| `casnh_1120_1127` | 3×3 cell | stays | 3,864 |
| `cennh_1120_1134` | mask centroid | stays | 4,066 |

Training loss against `casbox_1115_1226` (`clause_hw` off), paired by step (same samples, σ and noise;
n = 306 / 129 / 45 steps per bin):

| run | σ < 0.85 | σ 0.85 – 0.95 | σ ≥ 0.95 |
|---|---|---|---|
| `casboxr_1116_1149` (off, retrain) | −0.0002 | +0.0002 | +0.0006 |
| `cashw_1116_1110` (cell, header moves, 11-16 code) | +0.0001 | +0.0015 | +0.0028 |
| `cascell_1120_1056` | −0.0001 | +0.0015 | +0.0015 |
| `casnh_1120_1127` | −0.0001 | +0.0009 | +0.0008 |
| `cascen_1120_1056` | −0.0001 | +0.0005 | +0.0018 |
| `cennh_1120_1134` | −0.0001 | +0.0007 | +0.0021 |

Spread: the two same-config pairs (`casbox` / `casboxr`, `cashw` / `cascell`) are apart by +0.0002 and
+0.0000 at σ 0.85 – 0.95 and by +0.0006 and −0.0012 at σ ≥ 0.95.

Reads: `prompts_anima` / `prompts_clause` / `prompts_text`, 8 prompts each, one seed per prompt
(1 – 8), 24 pairs per set, arm sides shuffled per pair (`bench.turbo.blind_sheets`). Sheets and keys:
`output/tests/caststation_thin/blind_1120/` (local).

Placement, cell against mask centroid (header moving in both):

| read | set | cell | centroid | no pick |
|---|---|---|---|---|
| the user | `cascell` vs `cascen` | 11 | 11 | 2 |
| Claude, blind | `cascell` vs `cascen` | 15 | 5 | 4 |
| an agent, blind | `cascell` vs `cascen` | 15 | 5 | 4 |
| the same agent | `cashw_1116_1110` vs `cascen` | 13 | 6 | 5 |

By prompt set, cell / centroid / no pick:

| read | general tags | clause prompts | text prompts |
|---|---|---|---|
| the user | 5 / 2 / 1 | 1 / 6 / 1 | 5 / 3 / 0 |
| Claude | 7 / 0 / 1 | 4 / 4 / 0 | 4 / 1 / 3 |
| agent | 7 / 0 / 1 | 2 / 4 / 2 | 6 / 1 / 1 |
| agent, `cashw` set | 6 / 1 / 1 | 1 / 4 / 3 | 6 / 1 / 1 |

- The agent read the `cashw` set after the first one, so it had already seen the centroid renders.
- Invented cm ruler labels on the two height-chart rows: centroid arm 2 / 2, `cascell` 0 / 2, `cashw`
  0 / 2, `casnh` 1 / 2. On 11-16 two same-config runs drew them 2 / 2 and 0 / 2.

Header, moving (`cascell`) against staying (`casnh`), cell placement:

| read | header moves | header stays | no pick |
|---|---|---|---|
| an agent, blind | 13 | 6 | 5 |
| Claude, arm sides recognisable from the earlier set | 7 | 5 | 12 |

- By prompt set, moves / stays / no pick: agent 5 / 2 / 1, 6 / 2 / 0, 2 / 2 / 4; Claude 4 / 0 / 4,
  1 / 2 / 5, 2 / 3 / 3. On the 11 rows both picked, they picked the same arm.
- The user's decision (2026-11-20): "내생각에 효과 있는듯 칸중심 논문구현으로 확정하자". The cell placement
  with the header left at `(0, 0)` is the default since then (`clause_header` off by default,
  `configs/moss/lora.toml` unchanged at `clause_hw = 1.0`).
- `cennh_1120_1134` is rendered (`output/tests/caststation_thin/cennh_*`, local) and not read.

### Interpretation

- Placement did not produce a winner. All four reads split the same way by prompt kind: the mask
  centroid on prompts that carry position clauses, the cell on plain tag prompts. The cell arm's lead
  on text prompts leans on the ruler rows, a habit that re-rolls between same-config runs.
- The mask centroid costs less loss at σ 0.85 – 0.95 than the cell (+0.0005 against +0.0015, spread
  0.0002), and leaving the header at `(0, 0)` cuts the cell arm's cost (+0.0009). Neither loss gap
  showed up as a render gain in the Claude and agent reads.
- On the header, the agent's blind read prefers the moving header 13 to 6 and the user settled on the
  staying one. One seed per prompt and one training run per arm: neither read is against a
  same-config twin of today's arms on renders. The default is one flag (`--clause_header`) away from
  the earlier form.
- Open: the same two A/Bs on ama, where the cell and the subject sit further apart (0.13 of the
  width) and the 11-17 read found a gain from grounding.
