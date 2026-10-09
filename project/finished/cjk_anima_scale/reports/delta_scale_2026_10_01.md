# delta_scale — glyph order in the adapter output, and the seed's Δ shrunk (2026-10-01)

`experiments/delta_scale`: why `garble_replace`'s strings repeat
(`かかな応さぃなしいしいい`). A CPU probe of the adapter output, then the seed
rows with every row's Δ × s on the `sent` ruler. No training.

**Verdict.**
- At Δ 1 the share of glyph order in the adapter output falls with the reads
  across arms (seed 0.18 → warm 0.10 → cold 0.009 on かなしい). Shrinking the
  seed's Δ restores that order information, and the reads fall anyway:
  Δ 0.9 ≤ 1 edit 92 → 48, Δ 0.75 → 5. Order information in the context is
  not what limits the reads. That lever is closed, at inference and as a
  training-time row cap.
- Δ < 1 keeps the floor's banner and lengthens the line inside it. At 0.9
  the extra slots repeat the word's glyphs (dup 100 → 124, p 0.002); at 0.75
  they take unrelated glyphs and the line runs to sentence length. The rows
  pull the base's sentence-length line toward the word's length and do not
  get all the way: the slots left over are the repeats.
- The adapter's attention to Qwen is the same on every arm and on EN. The
  trained rows did not change how the adapter reads Qwen.

## Probe (CPU)

- 4-glyph words かなしい / パソコン / 山田太郎, all 24 slot orders, in
  `<prompt>, japanese text. Japanese text reads as "<word>".` with one prompt
  (`1girl, solo, blonde hair, school uniform, classroom, …`). EN control:
  `cat dog sun red` in the `english text` form.
- Every word token's adapter output (`crossattn_emb`) and its DiT cross-attn
  keys (`k_norm(k_proj(·))`, blocks 0 / 7 / 14 / 21 / 27, averaged) are split
  into identity + slot + residual (balanced two-way, over the 96 tokens).
- Why it matters: DiT cross-attention applies RoPE only to self-attention
  (`library/anima/models.py:377`), so its keys carry no position. A patch can
  tell slot 3 from slot 1 only through what the adapter's self-attention
  wrote into the token. A routed row is the same vector in every slot.
- Numbers came from a scratch run of the same code as the `probe` leg (no
  envelope).

Shares at Δ 1 (adapter output; the DiT keys are within ± 0.03 of these):

| arm | identity | slot | residual |
|---|---|---|---|
| EN control | 0.33 | 0.16 | 0.51 |
| pack only (Δ 0) | 0.53–0.63 | 0.12–0.14 | 0.24–0.33 |
| seed | 0.76–0.87 | 0.08–0.18 | 0.05–0.06 |
| warm | 0.83–0.85 | 0.09–0.11 | 0.06–0.07 |
| warm_short50 | 0.85–0.89 | 0.07–0.08 | 0.04–0.07 |
| cold_grid50 | 0.87–0.92 | 0.02–0.04 | 0.06–0.09 |
| cold | 0.88–0.97 | 0.01–0.03 | 0.02–0.09 |

- Training turns the word tokens into context-free stamps: the residual
  falls from 0.3 (pack) to ≈ 0.05 on every trained arm (EN: 0.5).
- Adapter → Qwen, last block, heads averaged: adjacent glyph queries overlap
  0.94–0.99, entropy 3.1–3.3, mostly on the prompt's first token. The same
  on every arm and on EN.

Slot share vs Δ scale:

| arm | 1.0 | 0.75 | 0.5 | 0.25 |
|---|---|---|---|---|
| seed | 0.08–0.18 | 0.17–0.34 | 0.25–0.36 | 0.22–0.29 |
| warm | 0.09–0.11 | 0.19–0.21 | 0.26 | 0.23–0.26 |
| warm_short50 | 0.07–0.08 | 0.15–0.21 | 0.21–0.27 | 0.23–0.25 |
| cold_grid50 | 0.02–0.04 | 0.06–0.10 | 0.14–0.18 | 0.18–0.19 |
| cold | 0.01–0.03 | 0.03–0.06 | 0.07–0.11 | 0.13–0.14 |

Absolute variance, かなしい (a share can rise only because identity falls):

| Δ | seed identity | seed slot | seed DiT-key slot | cold identity | cold slot | cold DiT-key slot |
|---|---|---|---|---|---|---|
| 0 | 14.1 | 2.8 | 26.7 | 14.1 | 2.8 | 26.7 |
| 0.25 | 7.9 | 5.9 | 49.8 | 14.6 | 3.8 | 28.5 |
| 0.5 | 8.9 | 8.4 | 61.9 | 24.6 | 2.4 | 15.3 |
| 0.75 | 14.7 | 9.8 | 62.5 | 39.7 | 1.1 | 6.9 |
| 1.0 | 20.0 | 4.7 | 31.4 | 45.5 | 0.43 | 2.8 |

- The seed's Δ adds real order information: slot variance peaks at 0.75
  (3.5× the pack's) and halves at 1.0.
- Cold's Δ removes it at every scale: identity grows, slot falls
  monotonically, to 1/10 of the pack's in the DiT keys.

## Renders — the seed rows at Δ s, `sent`, 184 per arm

Grid: 23 strings × 4 prompts × 2 seeds, `en` clause, 512², 28 steps, cfg 4,
routed (`garble_replace`'s `read` leg). Paired per render against the seed's
routed floor cache (= Δ 1).

| Δ | official | contained | ≤ 1 edit | ≤ 2 edit | dup | kana |
|---|---|---|---|---|---|---|
| 1.0 (floor) | 29 | 64 | 92 | 129 | 100 | 169 |
| 0.9 | 10 (+4 / −23, p 3e-4) | 31 | 48 (+6 / −50, p 1e-9) | 80 | 124 (+40 / −16, p 0.002) | 171 |
| 0.75 | 2 (+1 / −28, p 1e-7) | 7 | 5 (+1 / −88, p 3e-25) | 15 | 107 (+50 / −43, p 0.53) | 181 |

| Δ | box | box h | flat white | en_cos | box IoU vs EN ref |
|---|---|---|---|---|---|
| 1.0 | 0.156 | 0.180 | 0.132 | 0.922 | 0.198 |
| 0.9 | 0.137 | 0.157 | 0.101 | 0.932 | 0.258 |
| 0.75 | 0.119 | 0.125 | 0.064 | 0.946 | 0.379 |

On the sheets (`delta_scale_seed_s0{90,75}/sheets_sent/`, columns EN ref |
floor | arm):

- **0.9:** the floor's banner in the same place and size; the line inside
  is longer, with the word's glyphs doubled: `かがんんがえ`, `かががんがええ`,
  `かずんんばぶえええ` for かんがえ (floor `かんがえ` / `かかんがが`).
- **0.75:** still the banner. First and last glyph held, the middle replaced
  by other glyphs, sentence length: パソコン → `パマフプン`, `パイコン`,
  `ボイイワコンン`; かなしい → `た使尽惹妨したいい`, `か僕唇樹力したいい`.
  Repeats stay (`ココンン`, `しいい`).
- Placement moves toward EN ref as Δ falls (IoU 0.379 at 0.75, the highest
  of any arm so far), with the string gone.

## What it changes

- A row cap or a smaller Δ at bake trades identity for nothing; both are out.
- The line's length is the open lever. The base wants a sentence-length line;
  the rows shorten it. `garble_replace` trains in σ 0.6–0.85, while the line's
  place and length are set at σ ≥ 0.9 (`sigma_split_2026_09_30.md` § x̂0 per σ),
  and its canvases carry long garble lines.

Results: `experiments/delta_scale/results/20261001-1139-read-s090/`. Δ 0.75's
read ran in job `20261001-111419-b82a3c`, stopped during Δ 0.5 before its
envelope was written; its numbers are from the job's log. Arms:
`output/cjk_anima_scale/experiments/delta_scale_seed_s{090,075}/`.
