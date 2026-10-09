# sigma_split — the seed's rows gated by σ, and x̂0 per σ (2026-09-30)

`experiments/sigma_split`: `idea.md` § 2b check 1 and § 3 (a) / (b), with no
training. The rows are `seed_retrain_0930`, read routed. The sampler's
commitment-σ switch (`generate_body`'s `context_alt` + `tag_drop_sigma`)
feeds one conditional embedding while σ ≥ switch and another below it; CFG's
negative pass is untouched. Row Δ is added at encode, so "seed" is the caption
encoded at Δ scale 1 and "raw" the same caption at Δ 0 (the pack rows exactly).

**Verdict.**
- The rows act only above σ 0.5. Below it they, and the caption itself, change
  no text.
- Switched at σ 0.8 instead, the seed's rows write their string into the
  base's own garble layout: 16 / 29 official, 61 / 92 ≤ 1 edit, with the
  garble arm's placement.
- Layout and identity are both decided between σ 0.9 and 0.7. The seed's rows
  put their large text down at σ ≥ 0.9, before the scene has formed.
- `idea.md`'s identity-at-0.3–0.5 is out. Its data half (the base's own
  layout, only the string replaced) survives, at the band where text is
  decided: `plan_garble_replace.md`.

## Setup

- Grid: the `sent` ruler's `retrain_read` grid, 23 strings × 4 prompts × 2 seeds
  = 184 renders per arm, `en` clause, 512², 28 steps, cfg 4, Euler with
  flow shift 3.
- Floor: the seed's routed cache (`seed_retrain_0930/routed/native_sent/`),
  paired per render.
- Schedule: with shift 3 and 28 steps, σ 1 → 0.9 is steps 0–7, 0.9 → 0.69 is
  steps 7–16, 0.69 → 0.5 is steps 16–21, and σ < 0.5 is the last 7 steps.
- Plumbing check: one floor key rendered through the split path with seed on
  both sides matches the cached floor render to mean |Δpx| 1.48 / 255.

| arm | above switch | below switch |
|---|---|---|
| floor | seed | seed |
| `hi` | seed | raw |
| `lo` | raw | seed (the product condition, § 3 b) |
| `garble` | `{p}, japanese text. She is saying something.` | JA caption, seed (§ 3 a) |

## Switch at σ 0.5 (job `20260930-154128-cdfb8c`)

Words, 184 renders each; McNemar vs floor.

| arm | official | ≤ 1 edit | contained | dup |
|---|---|---|---|---|
| floor | 29 | 92 | 64 | 100 |
| `hi` | 30 (+2 / −1, p 1.0) | 89 | 65 | 99 |
| `lo` | 0 (−29, p 4e-9) | 2 | 1 | 122 |
| `garble` | 1 (−29, p 6e-8) | 2 | 7 | 153 |

- Per string, `hi` matches the floor within ±2 on every one of the 23 strings.
- `lo` is 0 official on all 23; its ≤ 1 edit hits are カメラ 1 and テレビ 1.
- `garble`'s only official hit is はい 1 / 8.
- **On the sheets:**
  - `hi` is near pixel-identical to the floor.
  - `lo` keeps the raw pack's banner of 1–2 small garble lines. At most a
    fragment of the string survives (`やった ネ+`, `…ちは~`).
  - The `garble` renders are the same image for every string on a given
    (prompt, seed). Swapping the whole clause to `reads as "X"` with the seed
    rows below 0.5 leaves no visible change.

Placement: idea.md § 1's measures, plus the EN-reference ruler (the same
prompt × seed rendered with `English text reads as "hi"`).

| arm | box | box h | flat white | en_cos | box IoU vs EN ref |
|---|---|---|---|---|---|
| floor | 0.156 | 0.180 | 0.132 | 0.922 | 0.198 |
| `hi` | 0.156 | 0.180 | 0.131 | 0.922 | 0.194 |
| `lo` | 0.102 | 0.111 | 0.068 | 0.945 | 0.448 |
| `garble` | 0.086 | 0.253 | 0.100 | 0.932 | 0.177 |

- **The seed's large banner and white canvas come from the rows' action above
  σ 0.5.** With the pack rows up there (`lo`):
  - the text box shrinks by a third and flat white halves;
  - the scene moves toward the base's EN render (en_cos 0.922 → 0.945);
  - the text lands where the EN word lands (IoU 0.20 → 0.45).

  This is idea.md § 1's size / paste prior, now read on the new seed.
- `garble` draws the layout idea.md asks for: small box area, tall boxes
  (vertical lines), several bubbles.

## Switch at σ 0.8 (job `20260930-163358-13d992`, arms `lo`, `garble`)

Same grid, same floor. With a switch at 0.8, the rows or the JA caption act
on the last 16 of 28 steps.

| arm | official | ≤ 1 edit | contained | dup |
|---|---|---|---|---|
| floor | 29 | 92 | 64 | 100 |
| `lo` @ 0.8 | 7 (+7 / −29, p 3e-4) | 28 | 33 | 140 |
| `garble` @ 0.8 | **16** (+11 / −24, p 0.04) | **61** | 47 | 123 |
| (`lo` / `garble` @ 0.5) | 0 / 1 | 2 / 2 | 1 / 7 | 122 / 153 |

| arm | box | box h | flat white | en_cos | box IoU vs EN ref |
|---|---|---|---|---|---|
| `lo` @ 0.8 | 0.104 | 0.119 | 0.066 | 0.945 | 0.414 |
| `garble` @ 0.8 | 0.095 | 0.253 | 0.095 | 0.932 | 0.172 |

- **Placement is the upper caption's.**
  - `lo` @ 0.8 has `lo` @ 0.5's box, flat white and EN-ref IoU: the raw
    pack's small banner.
  - `garble` @ 0.8 has `garble` @ 0.5's tall boxes and several bubbles.
- **The string is the rows'.**
  - With the base's garble layout above 0.8, the seed's rows below it recover
    55 % of the floor's official reads (16 / 29) and 66 % of its ≤ 1 edit
    reads (61 / 92). No row was trained for this.
  - Per string, `garble` @ 0.8 is at or above the floor's ≤ 1 edit on
    たいせつ 6 / 5, こうえん 4 / 1, かんがえ 2 / 1, てつだう 3 / 2 and 日本人 5 / 3.
    It is lowest on the kanji names: 山田太郎 0, 小山田 0, 大丈夫 1.
- **On the sheets** (こんにちは, やったネ):
  - the bubbles sit where the base put them;
  - the string arrives in pieces (`こんには`, `やったえ`, `やっ った`), often
    split across two bubbles;
  - glyphs are larger and fewer per bubble than the garble's.
- **The garble scaffold beats the raw-pack one** (16 vs 7 official). The base
  asked for Japanese without a quote leaves more, and larger, text cells for
  the rows to fill than the pack rows' banner does.
- Read with the x̂0 table below: by σ 0.8 the base has placed its text, and
  the string is still open. Rows acting only below 0.8 can write into the
  base's layout, and rows acting only below 0.5 cannot.

## x̂0 per σ (`--traj`, job `20260930-154901-b368a9`)

- Samples: the 4 grid prompts at seed 0, for こんにちは and やったネ.
  `en` uses their EN pairs `hello` / `we did it`; `garble` has no string and
  runs once per prompt.
- x̂0 = x_t − σ·v (after CFG), decoded at the step nearest each σ. σ 0 is the
  final image. Every decode is read: official for JA, the VL read for EN.
- Sheets: `output/cjk_anima_scale/experiments/sigma_split_s0.5/traj/sheet_traj_<cond>.png`.

| cond | σ 0.9 | σ 0.69 | σ 0.5 → 0 |
|---|---|---|---|
| `ja_seed` | hit 2 / 8, ≤ 1 edit 4: the large banner already reads; the figures are still a blur | hit 4, ≤ 1 edit 7 | unchanged |
| `ja_raw` | a faint text band | small garble lines | garble strokes still change until ≈ 0.33 |
| `lo` | as `ja_raw` | as `ja_raw` | ≤ 1 edit 1 / 8 at the end |
| `garble` | the bubbles already placed, as white blobs | bubbles + vertical line texture | strokes change until ≈ 0.33 |
| `garble_ja` | = `garble` | = `garble` | 0 / 8, the same image as `garble` |
| `en` | a faint text band, 0 / 8 | hit 4 / 8 (`hello`, `we did it.`) | unchanged |

- **Where the text goes is decided at σ ≥ 0.9, and what it says between 0.9
  and 0.7.** The base does this for EN, and places the garble bubbles the
  same way.
- **The seed's rows commit earlier than EN.** Their JA string already reads
  at σ 0.9, while EN at 0.9 is still a band. This is the paste prior as a
  trajectory: the rows lay down large text before the scene forms.
- **Below 0.5 only stroke detail moves, and it follows x_t, not the caption.**
  This is why `lo` and `garble` stay at 0.

## What it changes

- **idea.md § 2's band argument is closed.** At 0.3–0.5 the caption channel
  moves no text at 512², for the seed's banners and the base's small garble
  lines alike. Rows trained only there would have nothing to act on at
  inference. idea.md § 2b's reviewer objection is borne out, and more
  strongly: the caption itself has no leverage there, not just the rows.
- **The premise stands.** The base lays out the wanted layout (bubbles,
  lines) by itself, and the rows add the paste.
- **Keeping layout out of the row has to come from the data, not the band.**
  If the target is the base's own canvas with only the string replaced, the
  FM residual outside the glyphs is ≈ 0 at any σ, so even at 0.7–0.9 the row
  gets no layout gradient. That is `plan_garble_replace.md`.

Results: `experiments/sigma_split/results/20260930-1617-s05/`, `…/20260930-1657-s08/`,
`…/20260930-1624-traj/`. Renders: `output/cjk_anima_scale/experiments/sigma_split_s0.5/`, `…_s0.8/`.
