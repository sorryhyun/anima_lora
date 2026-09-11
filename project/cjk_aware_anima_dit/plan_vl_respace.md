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

## Anti-re-proposal

- Do not ship `vl16_pl_kozh` or `vl16_pl_20k` — NC pseudo rows.
- Do not patch spacing at inference (a word segmenter, or a stock re-read of
  Latin lines) — the user chose the retrain (2026-09-10).
- Do not judge a spacing arm on `exact_key` alone; it cannot see the regression.
- Do not reintroduce a whitespace strip in training targets for any VL arm.
