# S-line micro loop — 6 rows, cap / Q / composite share, EN-reference ruler, swap clause (2026-09-15)

> Lifted out of the former `history.md` on 2026-09-15; text unchanged. The cap is
> not a lever, composite share 0.4 → 0.9 is; Q-in-training is glyph-dependent
> (か frame-independent, 日 → 目); `en cos` / `box IoU` against a shared
> `English text reads as "hi"` reference replace the floor ruler; the rows are
> bound to the JA clause frame.
> Index: [`README.md`](README.md). Backticked paths (`src/…`, `plan_synth.md`, `output/…`) are relative to the line root `project/cjk_renderable_anima/` or the repo root, as they were in `history.md`.

## Cap-only isolation killed; micro arms launched (2026-09-15 16:18)

The cap075 run (`870a1d`, 15 min in of ≈ 2.6 h) and its native job were
terminated (user: too large a run for the question). Replaced by a
**micro inventory** — `--only_chars あかす出人日` (3 kana + 3 corpus kanji,
each one Qwen piece with a pack row: あ 186 / か 207 / す 46 / 出 77 / 人 48
/ 日 8) — on the S0b data recipe at 1/10 scale: data `synth_micro6` =
1 600 items (960 flat singles in the font bubble, 160 per row; 640 scene
composites over the 174 s0 scenes, all singles; `--natural_frac 0
--strings_frac 0 --words 0`, 43 s CPU), eval 36 prompts (single 6, combo
18, en 12). 2 000 steps, batch 4, compile, else the S0b train argv. Three
arms, same data and seed, native `あ,か,す,日` × EN clause × 2 seeds:

| arm | `c_flat` | Q | native conds |
|---|---|---|---|
| `m6_s2k_cap075` (`9cd3f8` / `5b8ed5`) | on, cap 0.75 | off | `full,c,fc` |
| `m6_s2k_nocflat` (`da79f9` / `05a337`) | **off** | off | `full` |
| `m6_s2k_q1` (`ab9012` / `403026`) | off | **fixed on in training** (`--out_vec_train 1.0`, frame `reads_as`) | `full` (= f + Q, the deployed cond) + `fq0` (f with Q off) |

Code: `--only_chars` may carry kanji and `--scenes` builds the piece map
at `--words 0`; the train stage takes `--out_vec --out_vec_train <scale>`
(`OutVec` on for every step, vector saved as `out_vec` in `trained.pt`);
eval and native apply a saved `out_vec` to every trained cond
(`--out_vec_scales 0` on native gives the Q-off diagnostic).

What the micro arms can decide: where the trigger lands (`f` alone hit /
kept), whether Q-in-training escapes the subtitle mode (Q at inference
alone: 13 / 54 / 9), whether the rows train at all without `c_flat`. What
they cannot: 246-row interference (24 kana at 512² fell to 10/36 in W2) —
the winner still needs one full-scale run before it is a recipe.

## Micro arms result (2026-09-15 17:52): cap = drop; Q-in-training lifts kept 27 → 38 and the lift survives Q off

All three at 2.5 it/s, 13 min train each; flat eval singles 12 / 11 / 11
of 12 (sheets: 6/6 clean big glyphs, kanji included), EN 24/24 everywhere.
Train end points are indistinguishable (loss 0.120; row norm 122 / 127 /
124; rel 0.62 / 0.65 / 0.64; cap arm `c_flat` pinned at 0.75, leak 0.30).
Native, `あ か す 日` × 8 prompts × 2 seeds = 64, both readers:

| arm | cond | hit | kept | hit & kept |
|---|---|---|---|---|
| cap 0.75 | `f` alone | 56 | 30 | 24 |
| cap 0.75 | `f + c` | 46 | 29 | 18 |
| cap 0.75 | `c` alone | 0 | 59 | 0 |
| no `c_flat` | `f` alone | 57 | 27 | 23 |
| Q fixed | `f + Q` (deployed) | 52 | **38** | **29** |
| Q fixed | `f`, Q off (`fq0`) | 48 | 39 | 27 |

Reads:

- **The cap is not a lever at 6 rows**: cap 0.75 and no `c_flat` are the
  same arm on every number and on the sheets. With 160 flat exposures per
  row `f` absorbs canvas + trigger + identity whatever shared vector is
  offered; `c` alone draws the font bubble with garbage inside (canvas +
  weak trigger, no identity). Entanglement is in the response, as the S0b
  note said.
- **Q-in-training is a partial pass.** Kept 27–30 → 38 (+8–11 of 64), hit
  & kept 23–24 → 29, hits with Q on 52 (mild subtitle shrink, not the
  halving seen with inference-only Q). The prediction on record (kept ≥
  56) fails; hits ≥ 25 passes. The gain is **baked into `f`**: with Q off
  at inference (`fq0`) kept stays 39 — so no inference hook is needed to
  keep it, which matters for shipping as a plain pack.
- **Wipes are seed-shaped, not prompt-shaped.** Across all arms seed 0
  renders the training canvas (big glyph, flat ground) on 7/8 prompts;
  seed 1 keeps the scene on most (か in the sky / window / a bubble; 日
  often inside a white patch pasted into the scene = the canvas carried
  as an object). Per-char n = 16 is two seeds, so kept differences under
  ≈ 8 are within the seed split.
- The flat prior comes from the 60 % flat share: `f` learns the flat
  canvas first and 40 % composites do not undo it. Next micro lever is
  the data mix (composite share 0.4 → 0.9), Q on vs off, same 6 rows —
  not the cap, not `c_flat`.

Artefacts: `output/wake_probe/rows_synth_micro6_m6_s2k_{cap075,nocflat,q1}/`
(`report.md`, `native/report.md`, `native/sheet_<char>_en.png`).

## Micro arms, composite share 0.9 (2026-09-15 19:02): the data mix is the lever; Q-in-training CLOSED

Same 6 rows, data `synth_micro6_c9` = 160 flat singles (27 per row) +
1 440 scene composites (`--scene_frac 0.9`, else the micro recipe), no
`c_flat`, Q off vs Q fixed on (`--out_vec_train 1.0`). Jobs `54e3e5` /
`fe327c` (Q off), `2e8bf4` / `7a7320` (Q on). Train end points again
alike (loss 0.091, row norm 129 / 132, rel 0.66 / 0.68).

| arm | flat singles | native cond | hit | kept | hit & kept |
|---|---|---|---|---|---|
| 0.4, no `c_flat` (above) | 11/12 | `f` | 57 | 27 | 23 |
| 0.4, Q fixed (above) | 11/12 | `f + Q` | 52 | 38 | 29 |
| **0.9, Q off** | **12/12** | `f` | **60** | **45** | **41** |
| 0.9, Q on | 10/12 | `f + Q` | 48 | 41 | 30 |
| 0.9, Q on | | `f`, Q off (`fq0`) | 28 | 45 | 19 |

- **Composite share is the lever**: 0.4 → 0.9 lifts hit & kept 23 → 41
  with singles still 12/12 and EN 24/24; combos 5/36 (from 0–1). Sheets
  (か): seed 1 keeps the scene 8/8 (glyph on a wall / sky / window / 2koma
  panel, often no bubble); seed 0 still wipes on 4/8 (was 7/8) and draws
  a bubble with か on the rest. The residual wipe is the flat prior from
  the 160 flat items — the next data point is flat 0 (needs the
  `n_flat > 0` assert lifted and an identity check without any flat
  exposure).
- **Q-in-training is closed.** At 0.4 it moved nothing that survives seed
  noise. At 0.9 the routing the arm was built for *did* happen — `f`
  trained under Q lost its own trigger (Q off: hits 60 → 28, 日 3/16) —
  and it buys nothing: per char (Q on vs Q off, hit / kept) あ 16/9 vs
  16/13, か 15/11 vs 14/11, す 14/13 vs 15/11, **日 3/8 vs 15/10** — the
  kana are a tie on both rulers and the whole hit gap is 日, whose row
  under Q learned the wrong glyph (sheets: 目 / 月-like strokes in 13 of
  16 cells, on a kept scene). The か sheets of the two arms are
  visually interchangeable (seed 1 kept 8/8, seed 0 wipes 3–4/8). So Q
  takes the trigger only when the flat items are scarce, gains no scene
  survival for it, and cost one of three kanji its identity (n = 1
  kanji; a mode pull, not proof it hits every kanji). Closed as "no
  gain, one casualty" — not re-proposed at train time; output-space
  regularisers against Q would inherit the same pull.
- The 0.4-share Q arm's kept +10 is therefore read as seed noise / a
  mix artefact, not a Q effect.

Artefacts: `output/wake_probe/rows_synth_micro6_c9_m6c9_s2k_{qoff,qon}/`.

## EN-reference ruler (2026-09-15 19:40, user): score against `English text reads as "hi"`, not against the floor

User's objection to the scene-kept ruler: the product question is not
"did the delta leave the scene alone" but "does the ext row behave like
an EN word token" — render the same prompt and seed with `English text
reads as "hi"` and ask how little changes when the word becomes the
glyph. Implemented as `src/eval/enref.py` + `--stage enref` (16 shared refs
under `output/wake_probe/native_enref/512_28_4/`, arm-independent) +
`--stage native_rescore` (re-scores an existing native from its stored
reads; no re-render / re-OCR). Three columns per trained render:
`en cos` (PE-Spatial, whole image), `en cos out` (patch tokens outside
the glyph box ∪ the word's box), `box IoU` (glyph box vs the word's
box). Sheet labels carry `e<en cos out>`.

What the refs themselves say (`sheet_enref.png`): the base puts "hi" as
a **small line in a subtitle bar or on the scene** (rarely a bubble), and
on p06 (`portrait, simple background`) it draws a **big "hi" on a flat
ground in both seeds** — the base wipes that prompt for EN too, so a JA
glyph on a flat ground there is correct and the old ruler was penalising
it. Readers are JA-tuned (5/16 read exactly `hi`), so boxes are scored,
not reads (15/16 have a detector box).

Rescore of the five micro arms (`floor` row = the ruler's ceiling: same
scene, garbage JA text):

| arm | cond | hit & kept (old) | en cos | box IoU |
|---|---|---|---|---|
| floor (any arm) | – | – | 0.933 | 0.36 |
| 0.4, cap 0.75 | `f` | 24 | 0.817 | 0.06 |
| 0.4, cap 0.75 | `c` alone | 0 | 0.921 | 0.26 |
| 0.4, no `c_flat` | `f` | 23 | 0.797 | 0.09 |
| 0.4, Q fixed | `f + Q` | 29 | 0.829 | 0.10 |
| 0.4, Q fixed | `fq0` | 27 | 0.852 | 0.15 |
| **0.9, Q off** | `f` | **41** | **0.860** | 0.07 |
| 0.9, Q on | `f + Q` | 30 | 0.838 | 0.09 |
| 0.9, Q on | `fq0` | 19 | 0.859 | 0.16 |

- **Ordering unchanged**: composite share is still the lever (0.797 →
  0.860) and Q on is still below Q off on the same data (0.838 vs
  0.860). The two rulers agree on the wipes — old-kept renders sit at
  en cos 0.925 (min 0.80), old-wiped at 0.708 (max 0.91) — except p06,
  where trained renders (0.91–0.93) are *closer* to the EN ref than the
  floor is (0.87). p01 (`bedroom, on bed`) is the hard prompt on every
  arm (0.65–0.68 vs floor 0.88).
- **`en cos out` ≡ `en cos`** (±0.002): the text boxes are too small to
  move a pooled PE feature; the masked column adds nothing. Keep `en
  cos`; treat the old kept as a cheaper proxy of it (they rank arms the
  same).
- **Box IoU is near zero everywhere (0.05–0.16) while even the floor's
  garbage text lands at 0.36.** The trained glyph does not go where the
  word would (the subtitle bar / a text line); it goes where the training
  canvas put it (large, centred). This is the one number that separates
  "scene survived" from "behaves like a text token", and no arm moves
  it — including Q, whose subtitle mode still leaves IoU 0.09–0.16. The
  `fq0` conds (Q off at inference) and `c` alone have the highest IoU,
  i.e. weaker deltas sit closer to the base's placement.

Ruler decision: `en cos` replaces the kept margin as the scene ruler;
`box IoU` is the placement ruler, currently unpassed by everything.
**Floor renders are off by default from here** (user, 19:35): `stage
native` renders only the trained conds and scores them against the shared
EN refs; `--native_floor 1` restores the delta-off cond and the old kept
margin (the scorer is built only when a floor exists). Follow-up in
flight: `--native_clauses swap` (`…, english text. English text reads as
"か".` — the EN ref's caption with only the word swapped, so the pair
differs in nothing but the ext row) on the 0.9 Q off / Q on and 0.4 no
`c_flat` arms, output `native_swap/`.

## Swap-clause natives (2026-09-15 20:05): the rows are frame-bound; あ already behaves like a word token in the EN frame, か does not

`--native_clauses swap` = the EN ref's caption with only the word swapped
(`…, english text. English text reads as "か".`), same seeds; scored
against the "hi" refs. Floor here = an *untrained* ext row in the EN
frame: en cos 0.969, box IoU 0.51 (the base renders it as EN salad in
the subtitle bar — an unknown row is treated as an EN word).

| arm | cond | both-reader hits (JA clause → swap) | en cos (JA → swap) | box IoU (swap) |
|---|---|---|---|---|
| 0.9, Q off | `f` | 60 → **23** | 0.860 → 0.903 | 0.09 |
| 0.9, Q on | `f + Q` | 48 → 23 | 0.838 → 0.885 | 0.13 |
| 0.9, Q on | `fq0` | 28 → 12 | 0.859 → 0.906 | 0.25 |
| 0.4, no `c_flat` | `f` | 57 → 10 | 0.797 → 0.830 | 0.12 |

Per char, 0.9 Q off (hits / en cos): あ **11/16** / 0.944, か **0/16** /
0.890, す 3/16 / 0.926, 日 9/16 / 0.852.

Sheets (`native_swap/sheet_{あ,か}_swap.png`, 0.9 Q off):

- **あ is the target behaviour**: small glyph on the scene, often where
  "hi" sat (subtitle bar on p00 s1, a bubble on p04, a small centred `あ.`
  on p06 where the ref draws `ao`), scene intact, a trailing period as
  the EN frame's punctuation. The row *is* a word token there.
- **か comes out as Latin strokes** (`ɟn` / `jn` / `fn` / `刀`-like): the EN
  frame makes the DiT read the row as a Latin word and the identity is
  distorted to fit; placement and scene are right (p00 s1 in the subtitle
  bar, p01 s1 small on the bed). 0/16 read as か.
- So what the JA frame (`japanese text` tag + `Japanese text reads as`)
  supplies is not only the trigger but part of the **script decision**;
  the rows were only ever trained inside it and hold identity to a
  varying degree without it (あ yes, か no, す/日 partly).
- **Q on changes *which* glyphs survive the frame, not how many** (user
  caught this, 20:15 — the 23 = 23 total hid it). Swap hits per char, Q
  on vs Q off: あ 8 vs 11, **か 7 vs 0**, す 4 vs 3, 日 4 vs 9. On the Q-on
  か sheet the glyph is か in ≈ 10/16 cells (p00, p01, p03 in the sky,
  p05, p07 2koma) where Q off gave Latin strokes in 16/16; and the same
  rows with Q off at inference (`fq0`) fall back to `ɟn` / `fn` / `カ`.
  So Q supplies part of the "this is a glyph" decision the JA frame
  used to supply, making か frame-independent — the routing the arm was
  built for, working *across frames*. The cost is the identity pull
  already seen (日 → 目, あ slightly distorted in the EN frame). Verdict
  corrected from "no gain" to **glyph-dependent: frame independence vs
  identity pull, not rankable on 6 rows**; the highest placement IoU of
  any cond is also Q-on's (`fq0` 0.25).

Readings: (1) the EN frame gives the placement and scene we want for
free when identity survives — so **frame is a data lever**: train the
same rows under a frame mix (JA clause / EN swap clause / bare quotes)
so identity stops leaning on the clause; eval on both clauses. (2) box
IoU is harsh on small boxes (あ visually in the "hi" slot still scores
0.08); read it with the sheets, not alone. (3) `fq0` under swap (IoU
0.25, en cos 0.906, 12 hits) is the weakest delta and the closest to the
base's placement — the placement/identity trade-off is the delta norm,
which every arm drives to the same ≈ 125–130.

## Flat 0 (2026-09-15 20:25 → 20:50): composite-only is worse on every ruler and wipes exactly as much — CLOSED

Same 6 rows and train argv as the 0.9 Q-off arm; data `synth_micro6_c10`
= 1 600 scene composites, 0 flat (`n_flat > 0` assert in `src/data/synth.py`
lifted to `>= 0`; composites default to singles when no flat kind is
in). Jobs `861ded` (train + eval) / `a61d3e` (native, `--native_clauses
en,swap` in one job). Arm `rows_synth_micro6_c10_m6c10_s2k_flat0`. Train
end point as every arm (loss 0.104, row norm 125, rel 0.64).

| | 0.9 Q off (flat 10 %) | flat 0 |
|---|---|---|
| flat singles / combo / EN | 12/12 / 5/36 / 24/24 | 10/12 / 0/36 / 24/24 |
| JA clause both-reader hits / en cos / IoU | 60 / 0.860 / 0.07 | **46** / 0.856 / 0.10 |
| swap clause hits / en cos / IoU | 23 / 0.903 / 0.09 | **5** / 0.887 / 0.14 |
| wipes (en cos < 0.80) seed 0 / seed 1, JA clause | 11/32 / 2/32 | 11/32 / 3/32 |

Per char, JA clause (c9 → flat 0): あ 16 → 15, か 14 → 15, す 15 → 11,
**日 15 → 5**; swap: あ 11 → 2, か 0 → 0, す 3 → 0, 日 9 → 3.

- **The residual wipe is not the flat prior.** Seed-0 wipes are the same
  count with zero flat items, and the same picture (big glyph on a white
  ground) — that ground is the DiT's own mode for a strong ext row, not a
  learned canvas. Composite share 0.4 → 0.9 took the wipes from 7/8 to
  4/8; removing the last flat items takes them nowhere. The delta norm
  (≈ 125 on every arm) is what overrides the scene.
- **Composites have a canvas too: the bubble.** か seed 1 draws a **white
  disc on black with か inside** on 5/8 prompts (`sheet_か_en.png`) — the
  erased bubble region blown up to the whole canvas. With no flat items
  the rows learn "glyph inside a round white bubble" as the unit, so the
  wipe changes shape rather than count. Every composite is a bubble; the
  EN refs put "hi" in a subtitle bar or straight on the scene and rarely
  in a bubble, which is also why box IoU never leaves 0.1: the row goes
  where its training surround was.
- **Flat exposure holds identity and frame independence.** 日 renders as
  Latin **"a"** on 5/8 seed-0 prompts (scene intact, `sheet_日_en.png`),
  出 fails both flat-eval seeds as an H-like Latin form in a bubble, and
  the swap-clause hits collapse 23 → 5 (あ 11 → 2). The big clean glyph
  on a flat ground is what keeps a row on its glyph rather than sliding
  to the nearest Latin letter; small in-bubble glyphs alone do not.

Verdict: **flat 0 closed**; flat 10 % + composite 0.9 stays the micro
recipe of record. The data lever that the bubble finding points at is
*where the composite text sits* (bubble / subtitle bar / directly on the
scene, as the EN refs do) — the composite analogue of the position-jitter
idea (user, 20:20) — scene-stage work, queued after the frame-mix arm.

## Scene frames s1 (2026-09-15 21:24 → 22:30): the prompt frame is a data axis — `sign` gives a non-bubble placement, `bubble_reads` triples the bubble yield

Flat 0 said the composites' own canvas is the bubble, and the swap clause
said the rows are bound to the one `reads as` frame. Both are the scene
prompt's doing — every s0 scene was `…, speech bubble, english text.
English text reads as "hi"`. `--scene_frames` (`src/scenes/stage.py`,
`FRAMES`) now draws the frame per prompt and records `frame` /
`clause_tpl`; the data stage swaps the JA text into the *same* frame
(`She is saying "か".`, `He is holding a sign that reads "か".`; `English
text reads as` → `Japanese text reads as`), so the composite caption is
the frame the base drew the scene under. Pronoun frames go to solo counts
only; `sign` drops other held objects from the action slot. `--scenes
s0,s1` composes runs. Also from tonight: the drawn glyph inherits the
**anchor's ink colour** (`anchor_ink`, median of the box's non-fill
pixels, when it contrasts ≥ 60 with the fill; s0: 29/174 anchors are
coloured — red / yellow / pink / blue) instead of always black, so ink
colour is not one more constant the rows can absorb (user, 00006's purple
"hi"); and s0's 826 rejected renders were pruned from disk.

Run `s1` (job `48bb04`, 1 000 prompts, frames `reads_as,bubble_reads,
saying,sign`, else the s0 recipe): **251 kept (25 %)** vs s0's 17 %.

| frame | prompts | kept | yield | top rejects | region short side (median) | what it looks like |
|---|---|---|---|---|---|---|
| `bubble_reads` `There is a speech bubble that reads "…"` | 338 | 111 | **33 %** | small_box 79, multi_box 55, open 46 | 73 px | clean single bubbles, varied shapes / placements |
| `sign` `He is holding a sign that reads "…"` | 172 | 56 | **33 %** | read_miss 41, open 23, no_box 22 | **99 px** | a held board with big lettering — **the non-bubble placement**; the flood finds the board like a bubble |
| `saying` `She is saying "…"` | 177 | 36 | 20 % | multi_box 42, small_box 42, read_miss 34 | 72 px | a bubble anyway (the `speech bubble` tag wins); one panel-border false pass (886) |
| `reads_as` (s0's) | 313 | 48 | 15 % | read_miss 97, multi_box 90 | – | as s0 |

Reads: (1) the clause form changes what the base draws far more than
expected — `bubble_reads` doubles `reads_as`'s yield on the same tags,
`reads_as` loses a third to read_miss (garbled anchor). (2) `sign` is the
first placement that is not a bubble and still erasable (flat board,
ring-median fill), with the largest glyph budget of any source. (3)
`saying` buys nothing over `bubble_reads` while the bubble tag is in its
generals; a bubble-less variant would drop the tag. Follow-up in flight:
`s1sfx` (job `cd7ee6`, 300 prompts, frame `sfx` = `sound effects` tag +
`English SFX reads as "BAM"` — the trainer's OCR clause grammar — with
its own onomatopoeia anchors and the open-fill path allowed: bubble-less
frames erase the plain rectangle).

Pool for the next micro arm: s0 174 + s1 251 (+ s1sfx) = 425+ scenes over
four frames; composites carry the frame in the caption, so the arm *is*
the frame-mix arm of `plan_synth.md`'s decision tree, with the frame
coming from the image rather than a caption-only lever.

## s1sfx + filter fixes + fonts (2026-09-15 22:00 → 23:00)

**Filter fixes** (from the user's picks 873 / 424 / 00001 / 832 on the s1
sheets): 873 = reader miss on a clean bubble (JA-tuned readers), 424 and
00001 = `small_box` by 11 px and by 2 px, 832 = a false `erase_miss` — the
outline is broken at 12 o'clock, nine seeds leak, three find a *pocket*
between the outline and the letters that encloses the box by bbox but covers
none of it. Two rules now in `bubble.py` / `src/scenes/judge.py`: (1) a fill
whose interior covers < 50 % of the text box is not a bubble; (2) with no
closed bubble the region is grown 1.2× → 1.35× → 1.5× and the first size
whose **erase seam** (the 3-px ring just outside the paint rectangle) is
≥ `--scene_open_uniform 0.9` fill-coloured is kept — the letters inside
vanish into the same colour, an outline crossing the edge shows as a cut,
and growing from the smallest step keeps a broken outline instead of
painting it over. s1 re-judged 251 → **269 kept (27 %)**; 18 open scenes
in, 832 among them with its bubble intact. Renders also inherit the
**anchor's ink colour** (`anchor_ink`) and 30 % of composites are tilted
±7° on their own layer (`tilt_frac` / `tilt_deg`; box = the tilted alpha
bbox).

**Fonts** (`assets/fonts/FONTS.md`, user's list from oekaki-zukan 516 +
two BOOTH picks): 源暎アンチック, 源柔 / 源真ゴシック M+B, コーポレート・ロゴ,
たぬき油性マジック, 破線G, こよみゆる, plus Noto Serif CJK (= 源ノ明朝); **Noto
Sans CJK is out** (user: reads ambiguous next to the manga faces; the
Chinese-form memory was DroidSansFallback, dropped 09-14 — verified on
直骨誤令海天込 that Noto index 0 is the JP face). `pick_font` draws
uniformly among the faces whose cmap covers the string (こよみゆる is JIS
L1 only; 破線G is a dashed decorative face, kept at equal weight on the
user's call). 16 faces.

**s1sfx** (job `cd7ee6`, 300 prompts, `sound effects` tag + `English SFX
reads as "BAM"`, open fill allowed): **102 kept (34 %)** after the
re-judge (94 before); rejects read_miss 92, multi_box 84. **The base does
not draw manga SFX for this frame — it draws the word on a title-card bar
or banner** (dark bar at the bottom, a pink banner across the chest, a
subtitle box), 95 of 102 with no closed bubble, region short side median
**131 px** (bubbles 73, signs 99). The rectangle erase paints in the bar's
own colour so it is invisible, the glyph inherits the bar's lettering
colour (pink / orange / white on navy), and the result is a third
placement family — text on a panel — with the largest glyph budget of any
source. The caption says `Japanese SFX reads as "…"` over a picture that
is a banner, which is what the base itself drew for that clause.

Scene pool now: s0 174 + s1 269 + s1sfx 102 = **545** over five frames.

## Frame-mix 2×2 on 6 rows (2026-09-15 23:38 → 09-16 00:50): frames are the lever, Q is inert, 日 was exposure

Data `synth_micro6_fm` = the `synth_micro12_fm` recipe (scenes s0 + s1, 443
kept scenes over four frames, share 0.9, no `c_flat`) on the six micro rows,
1 600 items (≈ 1 330 samples per row at 2 000 steps × 4). Arms
`rows_synth_micro6_fm_m6fm_s2k_{qon,qoff}` (jobs `e4b19f` / `af35b8`,
`d7a880` / `78c719`); m12fm (`74db20` / `74f7f3`) is the same recipe on 12
rows, i.e. half the samples per row. Native sheets now lead every seed's
cells with the `hi` reference render (`src/eval/native.py`, this commit).

| arm | rows | frames | Q | en hit | en cos | IoU | swap hit | en cos | IoU | singles |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.9 Q off | 6 | 1 | off | 60 | 0.860 | 0.07 | 23 | 0.903 | 0.09 | 12/12 |
| 0.9 Q on | 6 | 1 | on | 48 | 0.838 | 0.09 | 23 | 0.885 | 0.13 | 10/12 |
| m12fm | 12 | 4 | off | 44 | 0.891 | 0.18 | 26 | 0.926 | 0.24 | 18/24 |
| m6fm Q on | 6 | 4 | on | 54 | 0.870 | 0.12 | 44 | 0.902 | 0.14 | 11/12 |
| **m6fm Q off** | 6 | 4 | off | 54 | 0.868 | 0.12 | **46** | 0.905 | 0.17 | 12/12 |

- **Frame-mix doubles the swap hits** (23 → 46) at a cost of 6 `en` hits and
  with the flat eval untouched: the rows are no longer JA-frame-bound.
- **Q is inert on frame-mix data** (Q on vs off within 2 hits and 0.003 cos on
  every clause); the earlier "Q on below Q off" was a single-frame artefact.
  Q off is the recipe.
- **日 was exposure, not frames**: 12/16 on both clauses at ≈ 1 330 samples per
  row (m6fm), 2 / 0 at ≈ 670 (m12fm, with み 0/2, は 1/2 on the flat eval).
  Planning number: **≈ 1 000 samples per row** saturates the micro rows.
- Placement (IoU 0.12 / 0.17) is above the single-frame arms but below m12fm
  and the floor's 0.36 — still the open ruler. s1sfx scenes unused so far.

**Full-inventory launch** (00:55 → relaunched 01:32, jobs `9393f3`
train+eval / `106e0e` native): `synth_full_fm10k` = 92 kana + 68 ext kana
(`--kana_ext`) + 200 corpus kanji (`--kanji 200`, last 室:9) + 100
single-piece words (`--words 100 --held_out_words 8`; 435 ext rows touched),
**10 000 items**, the m6fm Q-off recipe at **53 000 steps** (≈ 460 samples
per row, ≈ 6 h) → `rows_synth_full_fm10k_full_s53k_qoff`. The first launch
(`d0b023`, 30 000 items) was killed at the latents step: the probe's text
cache is ≈ 1.3 MB per caption in RAM (25 288 captions → 33 GB RSS with the
latents and the DiT load still to come, swap full on the 46 GB box).
**Budget rule for the probe: ≈ 10 k items per run** (data `synth_full_fm`,
30 k rendered items, is on disk and unused).
Below the 1 000-per-row planning number by design: it is the **seed**
checkpoint — if singles / ext / kanji hold at a decent rate it becomes the
warm start for further vocab exposure and sentence-rendering arms rather
than being rerun from scratch.

## Full-inventory seed result (2026-09-16 07:45): 53k steps at ≈ 490 samples/row is not enough — the flat gates fail, words are zero

`rows_synth_full_fm10k_full_s53k_qoff` (train 341.6 min at 2.59 it/s, jobs
`9393f3` / `106e0e`): 433 texts over 10 000 items (median 26 items per
text, words 754 items total ≈ 8 each), 53 000 steps × 4 ≈ 490 samples per
row. Loss flat 0.09–0.11 throughout; table mean 0.58 row norms, one shared
row (ext 58974, not a Qwen piece — the clause's common piece) at 2.2.

| gate | seed 53k | S0 24k (old recipe, 09-15) | m6fm 2k (6 rows) |
|---|---|---|---|
| single | **13/36** | 20/36 | 12/12 |
| single_ext | 18/36 | 21/36 | – |
| single_kanji | 18/36 | 26/36 | – |
| word | **0/32** (held 0/16) | 3/32 | – |
| line / phrase_held | 0/32 / 0/32 | 2/32 / 1/32 | – |
| combo | 0/36 | 1/36 | 5/36 |
| en | 24/24 | 24/24 | 24/24 |
| native en hit / en cos / IoU | 36 / 0.882 / 0.13 | – | 54 / 0.868 / 0.12 |
| native swap hit / en cos / IoU | 18 / 0.932 / 0.32 | – | 46 / 0.905 / 0.17 |

- **Katakana is the failure family on the singles sheet**: hiragana singles
  mostly render (の う ち む は ま と), katakana mostly do not (ケ→ん-like,
  テ→ヲ, キ→ボ, リ→り, ン/チ garbage), small kana (ゃ ッ ュ ィ) 0. Kanji misses
  are near-shape kanji (長→最, 違→遼, 相→紀), i.e. identity partly there.
- **Words got ≈ 8 items each** (7.5 % of items over 92 words) and 33 word
  rows sit at norm ≈ 0 — the composite sampler draws by item, so the word
  share is what the flat share left, not a per-row budget. Word rows are
  effectively untrained.
- **Native**: swap 18/64 with か 0/16 and す 1/16 — the frame independence of
  m6fm is gone at this exposure; en cos is the highest of any arm (0.93) and
  IoU 0.32 / か 0.56 because the rows draw little (weak delta ≈ base
  placement, the same fq0 pattern).
- The shared row at norm 2.2 (row 58974; every caption touches it) is the
  micro arms' 7th / 13th row and carried the trigger there too — not new.

Verdict: **not a seed as is.** Per-row exposure ≈ 490 sits where m12fm's
≈ 670 already lost glyphs; the exposure curve (1 330 → 100 %, 670 → 75 %,
490 → 36 % singles) is steeper than linear at scale, and words need their
own share. Warm-starting from this table is possible (P0b showed a warm
start keeps identity) but the flat gates argue for the 1 000-per-row budget
first: 433 rows × 1 000 / 4 ≈ 108k steps (≈ 12 h) with a word share pinned
(≥ 25 % of items) — or the 92-kana intermediate arm at 23k steps to check
that the katakana failure is exposure and not interference.

## `scenes_sl1` — sentence-anchored scene pool (2026-09-16 08:00, user)

User's ask: a ~1k scene pool whose anchors are short EN sentences ("that is
what I said"), to be composited with JA words / sentences of 3–10 pieces
that make sense — the sentence-rendering data the seed line needs. Corpus
supply on the JA side: 1 650 training-corpus lines of 3–10 pieces with every
piece a pack row (1 610 distinct, 722 kana-only).

Smoke `scenes_sl1smoke` (job `79c987`, 64 prompts, 40 anchors of 2–5 words
without commas / apostrophes, frames reads_as + bubble_reads + saying, 4.3
min): **19 kept (30 %)**, read_miss only 11 % — the readers read multi-word
EN back exactly more often than single words (s1 reads_as: 31 %). The base
wraps the sentence inside one bubble; region short side median 71 px, long
side 77–202 px, and the bubble size tracks the anchor length ("thank you so
much" 196×160). The composite draws one line (`render_into_scene`, no
wrap), so the region's long side / 32 px sets the phrase length a scene can
take (3–6 glyphs at the smoke's sizes) — the data stage already draws only
texts that fit each region's capacity.

Full run `scenes_sl1` (job queued 08:12): 3 400 prompts (≈ 1 020 kept at 30
%, ≈ 3.8 h), the 40 smoke anchors + 15 longer ones (5–7 words) to widen the
region spread for 8–10-piece phrases. Open for the data step: a curated /
filtered JA phrase source (`--natural_frac` draws corpus lines, which are
adult-manga lines) and multi-line wrapping in the composite draw for the
long phrases.
