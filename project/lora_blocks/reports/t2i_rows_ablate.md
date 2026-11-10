# A t2i style LoRA with parts of its delta switched off at inference

Date: 2026-11-10.

Result in one line: **with the delta kept on the image rows only (`img_rows`), a t2i style LoRA
renders as the full LoRA does, and with the delta on the text rows only (`text_rows`) it renders
as no LoRA: the style, the outfit changes and the pose changes all come through the image rows.**

## Setup

- **LoRA:** `output/ckpt_moss/caststation_fwdpv.safetensors` (flow matching, 30 images, lr 1e-4,
  240 steps, rank 32, int8 base, fp32 stream, tail `sure`), trained with one delta for every row.
- **Script:** [`src/t2i_ablate.py`](../src/t2i_ablate.py). One DiT, the three `lora_ab.py`
  prompts (p0 Miku, p1 Neru, p2 natural-language sailor uniform) × seeds 1 / 2, 1024², ER-SDE
  50 steps, CFG 3, Spectrum, bf16 Base with the fp32 stream, int8 TE. Every arm eager (the row
  arms set a per-forward mask on the adapters), so these renders are not comparable with the
  compiled int8 sheets in `output/tests/caststation_distill_ab/`.
- **Arms:**

  | arm | delta kept |
  |---|---|
  | `none` / `full` | nothing / everything |
  | `img_rows` | image rows only; the text rows run on Base |
  | `text_rows` | text rows only |
  | `b14_41` / `b26_41` | blocks 14–41 / 26–41 only |
  | `iso_none` / `iso_full` | `none` / `full` with the text queries reading text keys only, in every block |

- **Output:** `output/tests/t2i_ablate/caststation_fwdpv/` (`sheet_p<i>.png`,
  `sheet_p<i>_iso.png`, `big_*.jpg` at 1024 px per cell).

## Rows

By eye, all six cells:

- **`img_rows` matches `full`.** The pale flat palette and thin line, and with them what the LoRA
  changes in content: p2 s1's white short-sleeved uniform becomes the black long-sleeved one,
  p1 s2's side view becomes the frontal phone-over-mouth pose, p0 s1's grin becomes an open mouth.
- **`text_rows` matches `none`.** Base's composition, glossy shading and outfit on every cell,
  with small differences (p1 s2's tie and vest, p2 s1's bench colour).

Final-latent distance, as a share of ‖full − none‖ per cell (range over the six cells):

| arm | to `full` | to `none` |
|---|---|---|
| `img_rows` | 0.39–0.69 | 0.95–0.99 |
| `text_rows` | 0.94–1.00 | 0.41–0.72 |
| `b14_41` | 0.42–0.72 | 0.99–1.03 |
| `b26_41` | 0.51–0.95 | 0.93–1.01 |

The run has no noise floor for this number (the sampler amplifies any change), so it ranks the
arms and does not say how close "close" is; the read above is the sheets'.

## Blocks and the text stream's read, in short

- **`b26_41` keeps the style, and the content changes with it** (p2 s1 is the black uniform with
  the last 16 blocks alone). Neither rows nor block ranges separate the style from the change in
  outfit and pose.
- **Cutting the text queries off the image keys leaves Base working and costs attributes**
  (`iso_none`: Neru's yellow eyes turn orange, an unprompted red tie, `smiling` weaker on p2,
  scribbled pseudo-text on p1 s2). The style still attaches under the cut (`iso_full`, p0 and
  p2 s1), and the LoRA moves the render as much as without it: ‖iso_full − iso_none‖ is
  0.77–1.20 of ‖full − none‖, the layout kept on 3 cells of 6.

## Claims

1. **A t2i style LoRA's effect is its delta on the image rows.** The same ΔW on the text rows
   does close to nothing, so a separate adapter bank for the text rows, or `delta_rows` limited
   to the image rows, would change little.
2. **The LoRA does not act through the text stream's read of the image.** The style attaches with
   that read cut, and the cut does not make the LoRA keep Base's layout more often.

## Not tested

- A second LoRA, more seeds, the Space 004 prompt.
- A LoRA trained under any of these restrictions: every arm cuts a jointly trained LoRA at
  inference. Open ones: adapters on blocks 26–41 only (the backward would stop at block 26), and
  training with the text stream isolated ([`request.md`](../request.md)).
