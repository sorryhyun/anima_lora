# polish_seed — the first warm polish on `seed_retrain_0930` (2026-09-30)

`experiments/polish_seed` (`--label s0930mu01`, read `read_mu01`). The polish of
`experiments/polish_b1` moved onto the new seed: all 1 362 trained singles
(`retrain_kana` + kanji b1–b4), warm from the seed, every other row frozen at it.
Beside it, `experiments/target4k`: the seed's own `target` ruler at the user's render
shape. Both read routed.

**Verdict: the polish cost identity.** Sent official 29 → 11 / 184 and ≤ 1 edit
95 → 60. The two `target` rulers hold (512² 5 → 5 / 14, 4 k 8 → 7 / 14). The renders
show more doubling and thinner, smaller text.

- Dropping the monochrome and line-art scenes (`--color`) does not help: sent
  official 10, 4 k target 8 → 4.
- The rows each rotated a few percent on their own, with no shared move.
- Warm polish in this shape stops (§ Verdict and next).

## Setup

| | |
|---|---|
| rows | 1 362 singles, warm from `seed_retrain_0930/trained.pt` (md5 `af99aa93…`) |
| table | `polish_b1`'s: p0709 `scene_single` 0.30 · p0507 `scene_window` 0.20 + `scene_single_small` 0.05 + `scene_line` (26–36 px) 0.15 · p0305 `scene_window` 0.10 + `scene_line` (14–22 px) 0.20 |
| volume | 4 steps / row × 1 362 = 5 448 steps, batch 4, 21 793 items, one epoch (`polish_b1`: 10 / row × 503) |
| trainer | μ 0.1, lr 1e-3 cosine, warmup 0.1; job `20260930-121835-74a647`, 2.41 it/s |
| line pool | 20 151 lines over 1 208 glyphs, per-glyph median 17 (b1: 14 658 over 491, median 47); 154 glyphs have no line, mostly b4's dataset-tail kanji (姉 満 顔 毎 …) |
| end of run | warm_cos 0.996, mean row norm 249.6 → 243.9 |

Two harness notes (both handled in the wrapper):

- `src/data/vocabs.py` reads only the first `list:` source. Four `list:@` specs
  therefore train b1's kanji alone, so the files go in as one spec.
- The encode fold sends `！？` to base T5. They count as allowed non-row
  characters in a line.

## Reads

The reads are paired against the seed's routed floor caches (same key × prompt ×
seed). `official` = sfx ∧ VL exact.

### `sent` — the acceptance strings + the chain's `read`, `en` clause, 512², 4 prompts × 2 seeds

| | floor | polish |
|---|---|---|
| official | 29 | **11** (+3 / −21) |
| ≤ 1 edit | 95 | **60** |

Official by string, floor → polish (of 8 each):

- はい 3 → 2
- おしい 1 → 0
- やったネ 5 → 0
- ちょっと来い 2 → 2
- こんにちは 2 → 0
- かなしい 2 → 0
- かんがえ 1 → 0
- たすけて 1 → 0
- てつだう 1 → 0
- ことば 1 → 1
- カメラ 2 → 3
- パソコン 3 → 2
- アイドル 2 → 0
- 小山田 1 → 0
- 日本人 2 → 1
- The other eight strings are 0 → 0.

What the renders show (contact sheet, seed 0, prompts 0–3):

- **Scene and text placement are the floor's.** Both arms put the text in the same
  top banner on the same scenes. There is no paste and no wipe.
- **More doubling**: かかなしいい, おおおしいい, こんにこちは, アイイドル, 日日本人. The
  floor doubles too (おおしい, ことと), less often.
- **Thinner, smaller, fainter text** on the bed and curtain prompts (やったネ,
  こんにちは, かなしい). This matches the p0305 `scene_line` items: 14–22 px lines,
  small in large bubbles or signs (`data/sheet_p0305_scene_line.png`).
- Slightly more stray small text lines beside the banner (the floor has some).

### `target` — the user's verbatim captions, 2 seeds

| shape | floor | polish |
|---|---|---|
| 512² | 5 / 14 (はい 5, こんにちは 0) | 5 / 14 (+2 / −2) |
| 768×1344 (4 032 tokens) | 8 / 14 (はい 5, こんにちは 3) | 7 / 14 (+1 / −2) |

**The 4 k floor** is `experiments/target4k`: the seed's rows at the shape of the
user's ComfyUI renders (`target_prompts.txt`'s header). Its cache is at
`seed_retrain_0930/routed/target_4k/`.

- At 4 k every render in both arms is the same composition: a small landscape scene
  on a black portrait canvas, with the caption's string as a large subtitle under it.
- The hits are mostly that subtitle. 512² draws the same subtitle band in about
  half its renders, with the scene filling the canvas.
- **The composition does not change with the polish.** Seed decides it in both arms:
  seed 0 is the small framed scene, seed 1 a wide bar scene. On the paired sheet
  (floor | polish per prompt) scene size and placement match. The non-black canvas
  share per render is floor 0.05–0.20 vs polish 0.05–0.17, within ±0.03 per file.
- **What changes is the text:**
  - Seed 1's subtitles are smaller and thinner (p00, p01). p06 s1 reads garble
    ((こん 衾俊…).
  - Seed 0's bubbles carry longer runs of small vertical lines (p03, p06).
  - p04 s0 gains a subtitle the floor did not draw.
  - This is the direction `sent` shows at 512².
- Naturalness was not scored, and 4 k was read on 14 renders.

## Reading

- **The layout held and the glyphs did not.** Rows barely moved in aggregate
  (warm_cos 0.996), yet the reads fell on 20 of 23 strings.
- Two candidates, not separated by this run:
  - **Per-row exposure is low.** At 4 steps / row the broad line pool (1 208
    glyphs) gives each row little identity signal.
  - **The line tiers teach a shared style.** The p0305 `scene_line` items teach
    "small thin line in a big bubble". The text on the sheet moves that way.
- Still unread: `polish_b1` on b1 at 10 / row held the C3 words (≤ 1 edit 47 → 42 / 96)
  on another grid, so the two do not compare directly.

## `--color`: the same polish without monochrome / line-art scenes

**The canvases were not the cause.**

**Setup.** A scene is kept when ≥ 0.08 of its 128² thumbnail's pixels are coloured
(saturation > 0.15, value > 0.15). By eye, < 0.06 is greyscale or line art with a
speck of colour, and 0.06–0.12 mixes pale-tinted sketches with flat colour on white.
The filter keeps 973 of 1 897 scenes: s1 251, s1w 321, sl1w 242, ja_comic 159.
Everything else as above. Jobs: train `20260930-133338-f9afe3`, read `…-34c25c`
(`results/20260930-1428-read_color_mu01/`).

| | floor | polish | color |
|---|---|---|---|
| sent official | 29 | 11 | 10 (+3 / −22) |
| sent ≤ 1 edit | 95 | 60 | 46 |
| target 512² | 5 | 5 | 6 |
| target 768×1344 | 8 | 7 | 4 (+0 / −4) |

## Scene fidelity: EN-reference PE cos (`sent`, 184 paired)

`en_cos` is the PE-Spatial pooled cos to the same prompt × seed rendered with an
EN caption (`src/eval/enref.py`); `en_cos_out` is the same outside both text boxes.

| | floor | polish | color | > floor (polish / color) |
|---|---|---|---|---|
| en_cos | 0.9219 | 0.9263 | 0.9281 | 82 / 104 |
| en_cos_out | 0.9230 | 0.9265 | 0.9278 | 77 / 97 |
| box_iou | 0.198 | 0.249 | 0.238 | 73 / 67 |

The means rise slightly: the renders sit a little closer to the base's EN render,
and the text a little closer to where the EN word lands. Paired, the effect is weak:

- polish beats the floor on en_cos in fewer than half the pairs;
- color beats it in 104 / 184 (sign test p ≈ 0.09).

It comes with the identity loss above.

## Where the rows moved (CPU)

Δ = polished − seed, in effective units (`raw` × `row_scale`):

- 1 344 of the 1 362 warm rows moved. |Δ| / |row| has median 0.048 and p90 0.12.
- The across-row mean Δ carries **0.7 %** of Δ's energy, and 99.5 % of Δ is
  tangential (no shrink). The color arm is the same (0.7 %, 99.5 %).
- So this is not a shared-direction move (unlike `real_kana`'s shrink). Each row
  rotated a few percent on its own, and that alone cost sent 29 → 11.

The train log agrees:

- FM loss stays flat at 0.09–0.13 over 5 448 steps.
- `warm_drift` climbs to 0.156 at step 1 400, then the anchor and the cosine lr
  pull it back to 0.064.
- A row sees ≈ 16 batches at lr up to 1e-3. That is a random walk half-reverted,
  not learning.

## Verdict and next

**Warm polish in this shape stops.** The shape is: warm from converged rows, lr
1e-3, 4–10 steps / row, μ ≤ 0.3.

- Every warm pass on trained rows has lost identity: `polish_b1`, `polish_t4k`,
  `real_kana`, and both arms here.
- Every arm that held identity trained its rows cold from the pack: `p1_mix`,
  `retrain_kana`, C3.
- A Fable 5.1 review, asked for a second opinion, reached the same verdict.
- μ 0.1 was shown to keep singles (the anchor sweep) only at dense per-row touches,
  never at 4 touches / row.
- `plan_polish` § SFX puts an SFX share into a warm polish, which is the shape
  that lost here. It waits on the cold micro arm below.

## The 4 k letterbox is the base's (controls, 2026-09-30)

The same 14 target captions at 768×1344, same seeds / steps / cfg, in three
conditions:

- **seed rows**, as above;
- **raw pack**: `target4k --raw`, Δ scale 0, routed, job `20260930-144451-b922f4`;
- **no pack**: `target4k --base`, Anima with no vocab pack, job `…-145132-9e35b7`. T5
  encodes each quoted Japanese span as one `<unk>` (`… ▁as ▁" <unk> ".`); Qwen3
  still reads the Japanese.

| | official | text sits | text reads |
|---|---|---|---|
| seed rows | 8 / 14 | a large subtitle under the scene | the string (with doubling) |
| raw pack | 0 / 14 | a small subtitle line, sometimes a bubble | garble (イだなー, 支撞, is.) |
| no pack | 0 / 14 | mostly vertical lines inside speech bubbles; small or no subtitle | Japanese-looking lines (まだよ, えんよ, こげきたばいよ, 変港かルんこぇは直感…アス？) |

- **All three draw the same composition.** Seed 0 is the small framed scene on a
  black portrait canvas, and seed 1 the wide bar scene. The non-black share
  (0.05–0.20) matches per file within noise. The letterbox belongs to the base with
  this caption at this shape. It is not a row problem, and no 4 k polish is sized
  for it.
- **The rows move where the text goes.** With no pack the base puts plausible
  Japanese lines in its own bubbles. With pack rows the text leaves the bubble for a
  subtitle band, and the trained rows make that band large and correct. This is the
  direction of idea.md § 1: lone data teaches size and paste. It supports that
  idea's premise: the base can lay out a Japanese line in a bubble, and the row need
  only change what the line says.
- These are 7 prompts × 2 seeds, read by eye for placement; there is no
  bubble-vs-subtitle count yet.

Next:

1. **The wanted styles in a cold micro arm.** 36 donors cold on `p1_mix`'s table
   plus the `scene_line` tiers, 90 steps / row, read against `p1_mix` on C1 / C2.
   - If they hold, lines and SFX go into the next seed batch's recipe.
   - If they fall, those styles cannot be row-trained and SFX is handled outside
     the rows.
2. (optional) idea.md § 2b check 1: Δ off above / below σ 0.5 on the seed.

Results: `experiments/polish_seed/results/20260930-1218-s0930mu01/` (data + train),
`…/20260930-1313-read_mu01/` (reads), `…/20260930-1428-read_color_mu01/`;
`experiments/target4k/results/` (`…-1155-s0930`, `…-raw`, `…-1454-base`). Renders
are under `output/cjk_anima_scale/experiments/polish_seed{,_color}_mu0.1/` and
`seed_retrain_0930/routed/target_4k{,_raw,_base}/`.
