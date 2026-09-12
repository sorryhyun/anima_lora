# plan_vl_respace — the shipped VL reader, retrained on spaced targets (2026-09-10)

The shipped manga reader (B′ = `vl16_tower_lr1e-5`, Hub
`sorryhyun/paddleocr-vl-1.6-manga-lora`) **never emits a space**. It trained
under `TARGET_NORM = 1` — `crop_dataset.normalize_target` joined the whitespace
out of every target ([`whitespace_fixed.md`](whitespace_fixed.md)) — and no run
has been retrained on the fixed rule for shipping.

## Evidence

- `kukiyuusha/13573906` (an English doujin page): B′ reads the left balloon
  `LOOKHOWDEEPITCOULDGOINSIDEME…`; stock VL-1.6 on the same crop reads
  `LOOK HOW DEEP IT COULD GO INSIDE ME…`. Token dump of the right balloon: B′
  emits `['BUT','YOU',"'",'RE','NOT',…]`, stock `['BUT','▁YOU',"'",'RE','▁NOT',…]`
  — the word-boundary `▁` pieces are gone, not the decode.
- The tokenizer reproduces it exactly: `"BUTYOU'RENOTALLOWED"` tokenizes to B′'s
  sequence. B′'s training set held **0** spaced targets out of 77 164 (≈ 2 965
  under `TARGET_NORM = 2`; 69 with real English words, all glued), and 0 of its
  4 790 val predictions carry a space.
- On the trainer's 3 008-page tree (v0.6.2 sidecars): 51 lines carry a ≥ 10-letter
  Latin run with no space, 4 with one. Hangul is negligible (2 lines, both misreads).
- No gate saw it: `eval_manga109.exact_key` is whitespace-blind by design.

Why the strip was there at all: §O1 folded whitespace for manga-ocr's WordPiece
vocabulary. VL's tokenizer has word-boundary pieces, so the reason never applied.

## Why not ship `vl16_pl_kozh`

It is the one `TARGET_NORM = 2` run and it does space (62 / 4 790 val preds), but
its pseudo rows come from the CC-BY-NC-SA AnimeText pool — research-only
([`plan2.md`](plan2.md) § Constraints). The shippable recipe is B′'s data alone:
Manga109-s COO SFX + the 1 : 1 `<text>` speech replay.

## R0 — does `TARGET_NORM = 2` recover spacing? — **yes, on `pl_kozh`** (2026-09-10)

Daemon job `20260910-221911-6a3193` (scratch `vl_space_ab.py`): B′ vs
`vl16_pl_kozh` vs stock on the same crops — every sidecar line with a ≥ 6-letter
Latin run (87, incl. mixed JA+EN credits) plus the kukiyuusha balloon `skip_en`
dropped, and 300 random Japanese lines.

| | B′ (shipped) | `pl_kozh` | stock |
|---|---|---|---|
| Latin lines with a space (of 87) | 7 | **71** | 79 |
| Japanese lines with a space (of 300) | 0 | 5 | 51 |
| guard-rejected (Latin / JA) | 0 / 0 | 0 / 3 | 1 / 2 |

- `pl_kozh` equals stock whitespace-collapsed on 45 / 87 Latin lines, 54 / 87
  ignoring whitespace — the gap is misreads on both sides, not spacing.
  kukiyuusha: `BUT YOU'RE NOT ALLOWED TO TRY IF YOU'RE STILL THIS WEAK, DOCTOR.`
  (B′ glued, `pl_kozh` = stock); `I mean… What an embarrassing outfit to sleep
  in…!!`, `©2017 Manjuu Co., Ltd. & …` likewise.
- Japanese: `pl_kozh` = B′ on 214 / 300 ignoring whitespace (the rest is reader
  drift both ways — one long balloon truncated, some SFX read better). It puts a
  space in only 5 / 300 Japanese lines against stock's 51, so R3's line-break
  spaces are rarer than the 3.8 % of targets suggests — keep R3, but it is a
  small fold.
- Caveat stands: `pl_kozh`'s ~15 k Korean pseudo rows are stock-taught and
  spaced, so some of its English spacing may come from them. R2 answers for the
  shippable recipe; R2b is the fallback if B′'s 69 English targets are not
  enough.

## R1 — a whitespace-sensitive metric first

`exact_key` stays as the Japanese gate (every number in `eval.md` is on it).
Add beside it, in `eval_manga109.py` / `eval_table.py`:

- **`space_exact`** — the same folds (NFKC, hearts, dot runs) but whitespace
  *collapsed*, not deleted; reported on the val / test rows whose target carries
  a space (~3.8 % — COO speech with a line break or U+3000, and the Latin rows).
- **English probe set** — the 52 Latin crops from R0 with stock's reads as the
  draft reference; hand-check before gating (same `status = draft` rule as the
  sincos labels). Score: share of lines whose read, whitespace-collapsed, equals
  the reference.

B′ scores ~0 on both by construction, so R1 costs no GPU and needs no re-run of
old arms.

## R2 — the arm: B′ verbatim under `TARGET_NORM = 2`

```
make daemon-run ARGS="--queue --stall-timeout 0 \
  project/cjk_aware_anima_dit/ocr/finetune_vl16_lora.py --skip_stock_val \
  --train_tower --tower_lr 1e-5 --lr 1e-4 --epochs 1 --bs 8 --grad_accum 2 \
  --seed 0 --run vl16_b2_norm2"
```

No `--extra_manifest`. ~1.5 h (B′'s job took 1.49 h; `history.jsonl`'s `wall`
is validation seconds, not training time). `args.json` is `vars(args)` and
does not record `TARGET_NORM`; the run is clean by date (after `efd846cf`) and
gets no `WHITESPACE_DIRTY` marker. Queued 2026-09-10 as B′'s exact argv with
only `--run` changed.

Gate, both halves:

1. **Japanese does not regress past B′** — sincos SFX ♡-blind ≥ 372 / 617,
   COO SFX ≥ 2124 / 2558, COO speech ≥ 2256 / 2559 (B′ 375 / 2127 / 2259 minus
   the ±3 spread the SSL rows show; no repeat-seed run exists on this line).
2. **Spacing comes back** — R1 `space_exact` on the spaced val rows well above
   B′'s ~0, and the English probe set within reach of stock.

### R2 run 1 — `vl16_b2_norm2` (2026-09-11): spacing half passes, sincos fails

Job `20260910-222619-f177ce`, 1.51 h, train 77 164 (grey 38 582 / 38 582) —
B′'s set. Evals: `eval_sfx.py` / `eval_manga109.py` (jobs `20260911-000123-*`),
English probe `vl_space_b2.py` (`…000203-ffccf2`). `eval.md` rewritten with a
`COO spaced` column (`eval_table.space_key`: `exact_key`'s folds, whitespace
collapsed, on test rows whose target has a space).

| | B′ | `pl_kozh` (NC) | `b2_norm2` | gate |
|---|---|---|---|---|
| sincos SFX ♡-blind | 375 | 391 | **350** / 617 | ≥ 372 — **fail (−25)** |
| sincos symbol-blind (all `So` dropped, = the stage's new default) | 389 | 398 | **362** | −27 |
| COO SFX ♡-blind | 2127 | 2167 | 2138 / 2558 | ≥ 2124 — pass |
| COO speech ♡-blind | 2259 | 2273 | 2255 / 2559 | ≥ 2256 — −1, jitter |
| COO spaced (of 198) | 1 | 14 | **19** | — |
| in-domain val SFX / speech | 86.2 / 88.2 % | 86.9 / 91.4 % | 87.0 / 91.2 % | — |
| val preds with a space (of 4 790) | 0 | 62 | 77 | — |
| English probe: lines with a space (of 87) | 7 | 71 | **74** | stock 79 |

- **Spacing is back without NC data.** kukiyuusha reads `BUT YOU'RE NOT
  ALLOWED TO TRY IF YOU'RE STILL THIS WEAK, DOCTOR.`; Japanese gets a space in
  5 / 300 lines, at balloon line breaks (`な なんですか あなた…`) — R3's fold.
- **sincos lost 51, won 26.** Hearts read as kana or another symbol (`びく♡` →
  `びくん` / `びく♬`), dakuten dropped (`あ゙っ` → `あっ`), runaways 2 → 7;
  none of the lost reads carry a space. Symbol-blind scoring recovers only 3.
- **No second cause found.** Libraries identical to B′ (torch 2.12.0,
  transformers 5.16.1, peft 0.20.0); the trainer calls `area_batches` with the
  same seed then and now; everything else in the diff since B′ is flag-gated or
  eval-only. The one default-path change is the normalizer, which touches
  ~3.8 % of targets.
- **Open: variance or effect.** In-domain improved while the 617-row
  out-of-domain gate dropped, and this line has never measured seed variance
  (the ±3 bound is the SSL rows' spread, not a repeat). Next: the same argv
  with `--seed 1` (1.5 h) — if it clears 372, ship the better seed; if it
  lands near 350 too, the spaced targets cost doujin SFX and R2b / a heart-
  weighted fix is the lever.

## R2b — only if R2 fails the English half

69 English targets may be too few to keep `▁` alive through a fully fine-tuned
tower. Then append a **licence-clean** spaced-English set: synthetic English
balloon crops rendered from OFL fonts over manga-style backgrounds
(`ocr/synth_sfx.py`, still unwritten, was the parked lever for this), at a
2–5 % append — col100's lesson is append small, never swap. Not from the
AnimeText pool, hayai's outputs or `JustANormalTinkerer/animetext-ocr` (NC, and
the last has no spacing).

## R3 — a reader-side rule for Japanese spaces

`TARGET_NORM = 2` teaches a space at every balloon line break and U+3000
(`あ 麻美さん`), so the new reader will put spaces inside Japanese speech. The
sidecar and the caption clause want those gone. Fix in
`anime_tools.ocr.sfx.normalize_read`, not in training: drop a space whose two
neighbours are both CJK (kana, kanji, CJK punctuation); keep it next to Latin,
Hangul or digits. The model keeps the boundary, the record gets the
caption-friendly form, and `dedupe_speech` keeps matching B′-era text.

## R4 — ship

1. Upload the R2 (or R2b) checkpoint to `sorryhyun/paddleocr-vl-1.6-manga-lora`,
   card updated (training rule, spacing, Manga109-s attribution unchanged).
   **Done 2026-09-11 — `vl16_b2_norm2` (seed 0) shipped as v2 on the user's
   call, despite the sincos half of the R2 gate (350 < 372); no seed-1 run.**
   Hub commit `4caffe65`; v1 (B′) stays at revision `3b5fe022`. The card gained
   a Versions section stating the trade, and every table was re-based onto the
   current `exact_key` (/617 sincos, 86 checked; COO numbers from `eval_table`).
   `adapter_config.json` was published with the Hub base id and sorted
   `target_modules`, as v1 was.
2. **The catalog will not re-fetch it.** `Asset.missing()` only checks that the
   three files exist in `dest`, and the `sfx_reader` row pins no revision, so
   every existing install keeps the dirty weights. Move the row's `dest` to a
   new directory (`models/paddleocr_vl_1.6_manga_lora_v2`) or pin a revision
   with a stamp file the probe checks.
3. anime_tools release with R3 + the catalog change, trainer pin bump.
4. Re-run the OCR stage over the tree, regenerate
   `probes/ocr_merge_sheet.py --baseline_dir` against the v0.6.2 sidecars;
   `kukiyuusha/13573906` is the smoke.

Scope note: with `skip_en` on (the stage default), ASCII-only English lines are
dropped anyway; the fix matters for mixed lines, `--keep_en` runs, Korean, and
the Japanese line-break spaces R3 handles.

## R5 — the glyph fold (`TARGET_NORM = 3`)

R2's post-mortem named hearts-read-as-kana and dropped dakuten as the sincos
loss, and the trainer taught the annotators' spelling verbatim while
`exact_key` folded every variant away — four targets for one pause (`・・・` /
`...` / `…` / `……`), `♥` competing with `♡` 41 : 296, `あ゛っ` carrying a
spurious space because NFKC splits the spacing mark off with one. `textnorm.py`
(2026-09-12) makes one rule for all three surfaces — training target, scoring
key, record form — and `crop_dataset.TARGET_NORM` goes to 3.

### R5 run 1 — `vl16_b2_norm3` (2026-09-12): in-domain up, sincos fails harder

Jobs `20260912-014858-b7bbc4` (train, 1.47 h) → `…-e7d8a5` (COO) / `…-cdc7a7`
(sincos). `args.json` differs from `vl16_b2_norm2` in `run` alone — same seed 0,
same 77 164 rows, **a pure A/B on the fold**.

| | B′ | `b2_norm2` | `b2_norm3` | gate |
|---|---|---|---|---|
| sincos SFX ♡-blind | 375 | 350 | **319** / 617 | ≥ 372 — **fail (−56)** |
| sincos strict | 312 | 297 | 274 | — |
| COO SFX ♡-blind | 2127 | 2138 | **2157** / 2558 | ≥ 2124 — pass |
| COO speech ♡-blind | 2260 | 2255 | 2255 / 2559 | ≥ 2256 — −1, jitter |
| COO spaced (of 192) | 1 | 19 | 17 | — |
| in-domain val SFX / speech | 86.2 / 88.2 % | 87.0 / 91.2 % | **87.3** / 91.1 % | — |
| val speech runaways | — | 168 | **9** | — |
| val preds with a space (of 4 790) | 0 | 77 | 71 | — |

Every COO/val column is flat-to-better and the 617-row doujin gate drops again.
Read down the *rescored* table only — `eval_table.py` re-derives all three rows
on the current key; the numbers printed in each run's own report are not
comparable (`eval.md` § Comparability).

### Where the 56 rows went — contact sheets

`output/tests/ocr_norm3_diff/{lost,won}_*.png` (scratch `sheet_norm3.py`): every
sincos SFX row the two runs disagree on, crop + GT / B′ / norm3.

- **B′ right → norm3 wrong 87, the other way 31.** 59 of the 87 are one shape:
  the trailing **♡ is read as a kana** — `びく♡` → `びくん`, `ぱん♡` → `ぱんん` /
  `ぱんッ`, `ガク♡` → `ガクル`, `パン♡パン♡パン♡` → `パンレパンレ`. The filler is
  almost always one of ん ッ ト ル し ィ ☆ ♪.
- Heart emission, on the 497 GT-heart rows: B′ 313 → norm2 312 → **norm3 271**,
  and ♡-blind exact on those rows 61.4 → 58.8 → **52.1 %**. The two earlier runs
  agree to one row on this axis; only norm3 moves it.
- The 31 wins are voicing, not symbols — `ひく` → `びく`, `ガャボ` → `ぢゃぼ`,
  `ペシー` → `プシー`. That is fold 1 (dakuten) doing its job.

### The cause is the tokenizer: `♥` is one token, `♡` is three bytes

`PaddleOCR-VL-1.6`'s tokenizer has **`♥` (U+2665) as a single id (99252)** and
**no `♡` (U+2661) at all** — it falls back to three raw UTF-8 byte pieces.
`fold_glyphs` maps `♥ → ♡`, so `TARGET_NORM = 3` deleted the only cheap heart
the decoder had and asked for a 3-token byte sequence in its place.

Raw sincos SFX predictions, hearts emitted:

| | `♥` | `♡` | total |
|---|---|---|---|
| B′ | 143 | 205 | 348 |
| `b2_norm2` | 144 | 204 | 348 |
| `b2_norm3` | **0** | 299 | **299** |

norm3 does exactly what it was taught — it never writes `♥` again — and the
49 hearts it stops writing are not replaced by `♡`. **73 of the 87 lost rows
carry a heart in the label**; B′ read a heart on 49 of them (21 of those with
the single-token `♥`), norm3 on 11. The single `♥` id was acting as the sink
for the drawn heart; remove it and the probability mass goes to whatever is
still one token in that slot — ん ッ ト ル し, exactly what the sheet shows.

The byte-fallback set is wider than the heart and it is SFX-shaped:
`ぁぃぅづ ぱぴぷぺぽ ぶぼ ぎぐ ざぜぞぢ ゾヂヅ ♡ ♬`. **Every handakuten hiragana
is byte-fallback** — `ぱん`, `ぴく`, `ぷしゅ` each cost 3 tokens for their first
glyph. Any future target rule should be read against this list first.

### The dot fold is not the cause, and is worth keeping

`・・・` is 3 tokens, `…` is 1, so the dot fold made 12 877 targets *cheaper*,
and its one visible effect is good: **val speech runaways 168 → 9** (repeating
`・・・` was the runaway's favourite loop). What it did not do is score: the key
already folds dots, so B′ collects 16 rescued rows for free (`ウズ・・・・` reads
as `ウズ…`) where norm3 needs only 7 — training the fold in bought **9 rows the
key was already giving away**. *A fold the scoring key already applies has no
headroom by construction.* Neutral-to-good, not the regression.

### R5b — flip the heart fold, keep the rest

`TARGET_NORM = 4`: `fold_glyphs` with the heart table inverted — **`♡ ❤ → ♥`**,
the single token, instead of the reverse. Dot, dash, wave and dakuten folds
unchanged.

```
make daemon-run ARGS="--queue --stall-timeout 0 \
  project/cjk_aware_anima_dit/ocr/finetune_vl16_lora.py --skip_stock_val \
  --train_tower --tower_lr 1e-5 --lr 1e-4 --epochs 1 --bs 8 --grad_accum 2 \
  --seed 0 --run vl16_b2_norm4"
```

Only the **training target** flips. `exact_key` and
`anime_tools.ocr.sfx.normalize_read` keep folding to `♡` — the scorer is
♡-blind and the record wants one spelling, so the direction is free downstream
and the two surfaces stay consistent with each other.

Gate: sincos ≥ 350 (norm2's number — R5b is a norm2 delta, not a B′ one), COO
and val no worse than norm2, and `♥` back in the raw predictions. If it lands
near 350 the fold ships with the flip; if it lands near 319 the heart token is
not the whole story and the next move is a repeat seed, which this line still
has never measured (`R2 run 1` § Open).

### Why the heart was never learnable in the first place

Heart share, SFX rows:

| set | rows | with a heart |
|---|---|---|
| train | 38 582 | 85 (**0.22 %**) |
| COO test | 2 558 | 8 (0.31 %) |
| sincos gate | 617 | 497 (**80.55 %**) |

The `びく` family alone: 299 train rows, tails `ッ` 90 / `っ` 86 / none 69 —
**zero with a heart**; 152 sincos rows, `♡` 129 / `っ♡` 8 — **~92 % with one**.
The model has never once been shown `びく♡`, so its heart behaviour is leaked
from the base model, not trained, which is why a target-rule change can swing
it by 42 rows at all. It also means **COO cannot see this axis** (0.31 %): the
COO columns rising while sincos falls is not a contradiction, the two gates
measure disjoint things.

R5b treats the symptom. The cure is heart-bearing *positives*, and the only
licence-clean source is synthetic — `ocr/synth_sfx.py`, still the parked lever
from R2b, now with a second reason to exist. Append small, never swap
(col100's lesson).

### Hard negatives — free to insert, wrong tool for this

The train set has **0 empty targets**; every crop is a box that holds text, so
the reader has never been taught "there is nothing here". Adding them costs no
code: `Collate` builds `prompt + "" + eos` and labels the single EOS, and
`load_split(extra=[…])` already appends a sibling manifest. A decoration
negative set (burst spikes, free-floating hearts, screen-tone) would target the
one mode the sheet shows beside the heart — a mark read as a glyph — at the
price of teaching the reader to drop real trailing glyphs.

But the 87 lost rows are not hallucinations over empty space: 73 of them have a
real heart drawn in the crop. That is a **missing positive and a missing token**,
not a missing negative. Negatives stay parked behind R5b and the synthetic set.

### Label fix — 12 rows (2026-09-12)

The user hand-corrected 12 sincos SFX labels off the lost sheet: 222 517 803
805 832 833 834 861 866 945 946 954. Rescored on the new labels, **B′ 375 →
365, norm2 350 → 346, norm3 319 → 323** — the gap narrows from −56 to −42.

**The pass is not an unbiased sample.** All 12 rows came from the *lost* sheet
(B′ right, norm3 wrong) and none from the *won* sheet, so every correction there
can only cost B′ or pay norm3. That B′ drops 10 rows on 12 edits is a hint the
617-row basis leans its way — several labels were drafted off an older reader's
output — but it is not measured until the `won` 31 are checked at the same
strength and a random sample of the untouched 499 gives a label-noise rate.

Two of the 12 are contested: 805 and 866 both replace a trailing `レ` with `♡`.
At matched magnification 861 and 954 show a top notch and two lobes (a heart);
805 and 866 show no notch and a rightward hook at the foot (a `レ`), and this
artist writes both in the same page. Left as the user set them.

The stored `sfx_*.jsonl` carry `text`, so `eval_table` cannot see any of this —
`eval.md` stays on the old basis until `eval_sfx.py` re-runs for all three arms.

## Anti-re-proposal

- Do not ship `vl16_pl_kozh` or `vl16_pl_20k` — NC pseudo rows.
- Do not patch spacing at inference (a word segmenter, or a stock re-read of
  Latin lines) — the user chose the retrain (2026-09-10).
- Do not judge a spacing arm on `exact_key` alone; it cannot see the regression.
- Do not reintroduce a whitespace strip in training targets for any VL arm.
- Do not pick a target spelling without checking the tokenizer first — R5 lost
  49 heart reads by folding the single-token `♥` into byte-fallback `♡`.
- Do not judge a heart-slot change on the COO columns; at 0.31 % heart rows they
  cannot see the axis. Only the sincos gate can.
