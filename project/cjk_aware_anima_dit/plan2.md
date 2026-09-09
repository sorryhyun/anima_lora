# plan2 — the reader, label side (2026-09-09)

Reopens the **reader** half of this line only; the DiT half stays frozen
(`plan.md`). Premise: [`findings.md`](findings.md) § Tower and
`output/ocr/eval/tower_drift.md` (`ocr/tower_drift.py`). Read
[`eval.md`](eval.md) for every number quoted here — all rows are on the
♡-blind / 617 basis unless a column says `strict`.

Three items:

- **P1** — self-training on the AnimeText crops (Noisy Student with a
  cross-reader agreement filter). The lever the drift table points at.
- **P2** — a ~1k hand-labelled target-domain set, diversity-picked. Small on
  its own; its second job is calibrating P1's filter.
- **P3** — the open question the SSL post-mortem left: does a tower that is
  *good on its SSL objective but drifted out of the projector's space* actually
  hurt the SFT, or was the read-through gate over-cautious? Cheap ladder.

Order: P3-L0 first (an hour, no new code), then P2's pick + P1's teacher sweep in
the queue while labels are typed, P1's filter calibrated on P2's labels, then
P1's SFT ladder.

## Premise, in four numbers

| fact | number | where |
|---|---|---|
| the SFT endpoint is set by the labels, not the init | SFT-from-SSL vs B′ cos 0.981 / 0.976, floor (B′ vs col100) 0.976 | `tower_drift.md` |
| the anchored SSL never left stock | ssl_all cos 0.979 to stock, B′ 0.953; identical on grey and colour crops | `tower_drift.md` |
| the tower is not at the task's ceiling | in-domain val 86–88 %, doujin gate 61 %; ×3 epochs drifts further (0.937) and loses 15 gate lines | `eval.md` |
| where B′'s misses are | 55 / 313 ♡-only, 61 ♡/〜/ー-only; ♡ in 497 / 617 gate rows vs 337 / 77k COO train rows | findings § Tower |

So: more labels, in the target domain, carrying the symbols COO lacks. Nothing
label-free is queued.

## P1 — self-training with cross-reader agreement

**Teacher.** B′ (`output/ocr/vl16_tower_lr1e-5/best` + its `tower.safetensors`;
`Vl16Reader`). Second opinion: the manga-ocr fine-tune (`mocr_lr5e-5`,
`MangaOcrReader`) — a different architecture and vocabulary, so its errors are
uncorrelated with B′'s plausible-word rewrites and runaways (findings § A/B).
hayai v2.1.5 is a third vote if the yield needs it.

**Pool.** `$ANIMA_ANIMETEXT_ROOT/animetext_crops/manifest_all.parquet` —
2,041,609 crops (`image_id, k, w, h, box, path, split, src`), mixed PNG/WebP, go
through `path`. CC-BY-NC-SA → the student is research-only, exactly as the SSL
tower would have been; the shipped reader stays B′ unless the licence question
is settled separately.

**Step 0 — teacher sweep (`ocr/pseudo_label.py sweep`).** Batched, left-padded,
area-sorted, `use_cache=True`, `bs 32`. **Measured 2026-09-09** on random draws
from `manifest_all` (640 + 2000 crops, RTX 5070 Ti): B′ **36.5 crops/s**,
manga-ocr fine-tune **190 crops/s**; crop decode off the other volume is
negligible (2.7k crops/s). So the full pool is ~15.5 h of B′, not 31 h. Given
the measured yield below, the first pass is a **100k draw** (seeded, stratified
over the three AnimeText splits): **~46 min B′ + ~9 min manga-ocr**. Write one
parquet per reader: `image_id, k, pred, pred_norm, n_tokens, runaway`.

The sweep must batch on `crop_dataset.token_batches` (24k budget), not the
eval readers' fixed `bs 32` — `manifest_all`'s large-crop tail reaches 122.6k
tokens in a 32-crop batch and will OOM 16 GB partway through 100k.

**Step 1 — the filter.** This is the whole method; everything else is the B′
recipe.

1. **Runaway / length guard** — `n_tokens < max_new_tokens`, `len(pred_norm) ≤ 96`
   (`crop_dataset.MAX_TARGET_CHARS`), no character run longer than 4, no
   repeated 2–3-gram covering > 50 % of the string.
2. **Agreement** — `exact_key(B′) == exact_key(mocr)` on the **♡-blind** key
   (`eval_table.py`'s fold), then the pseudo-label **takes B′'s string**, so
   hearts and `〜` survive into the label where manga-ocr would have dropped
   them. Speech-only crops will agree far more often than SFX; report the two
   yields separately.
3. **Self-consistency (optional, only if 2. yields < 40k)** — B′ reads each crop
   under two `augment.Augment` draws; keep on ♡-blind agreement. Doubles the
   B′ sweep. **Not needed**: measured yield is 47.4 % (below), so 100k draw →
   ~47k kept, above the 40k the top arm wants.

Print, for the kept set vs COO train: size, mean length, ♡ share, small-kana
share, `ー`/`〜` share, script mix. **If the kept set carries no more ♡ than COO
does, the arm cannot fix the ♡ misses and the plan says so before any SFT.**

**Measured on a 2000-crop draw (2026-09-09), before any of this is written:**

| quantity | value |
|---|---|
| guards drop (♡-blind, B′ side) | 12.9 % — ngram-dominated 9.8, char-run 2.7, too-long 0.3, empty 0.1 |
| ♡-blind agreement among the survivors | 948 / 1742 |
| **kept** | **47.4 %** of the draw → ~47k per 100k, ~95k per 200k |
| kept mean length | 9.3 chars |
| ♡ (incl. ♥) share of kept | **2.2 %** vs **0.44 %** in COO train (337 / 77k) |
| `ー` share of kept | 11.6 % |
| small-kana share of kept | 32.7 % |

So the precondition passes: the kept set carries ~5× COO's ♡ density, ~880 ♡
rows at the 40k arm against COO's 337. It is still far below the gate's own
80 % (497 / 617) — this buys density, not the gate's distribution.
Calibrate the filter on P2's labels when they exist: precision of the kept
rows against the human string, ♡-blind and strict.

**Step 2 — manifest.** `derived/manifest_pseudo_<name>.parquet` in the COO
manifest schema (`split, kind, id, book, page, text, joined, orient, w, h, poly,
path`, plus `source = "pseudo"`): `split = "train"`, `kind = "sfx"` for every
row (AnimeText boxes are not kind-tagged; `load_split` only draws `sfx` and
`speech`, so an unknown kind would be silently dropped), `book = f"at{image_id}"`,
`page = 0`, `id = f"{image_id}_{k}"`, `orient` from the aspect ratio, `poly` from
`box`, `path` **absolute** (pathlib's `derived_root() / abs_path` resolves to the
absolute path, so the other volume needs no symlink). Because pseudo rows count
as `sfx`, pass `--speech_ratio 38634 / (38582 + N_pseudo)` so the real speech
draw stays at its full count instead of inflating with N.

**Step 3 — the SFT ladder.** B′ recipe verbatim (`finetune_vl16_lora.py
--train_tower --tower_lr 1e-5 --lr 1e-4 --bs 8 --grad_accum 2 --epochs 1`) +
`--extra_manifest pseudo_<name>`. Append shares, smallest first (col100's lesson:
a 1.6 % append was +2, a 22 % swap was −34 — never swap):

| arm | pseudo rows | share of train | wall |
|---|---|---|---|
| `vl16_pl_4k` | 4,000 | ~5 % | ~94 min |
| `vl16_pl_20k` | 20,000 | ~21 % | **105 min (measured)** |
| `vl16_pl_40k` | 40,000 | ~34 % | ~127 min |

**The "B′ = 210 min / 77k rows" this table used to carry was a misread.**
`history.jsonl`'s `wall` is set inside `evaluate()` (`finetune_vl16_lora.py`
:430), so 210.5 is **seconds of the validation pass**, not training minutes.
B′'s own daemon job took **89.6 min** end to end (77,164 crops at the 15.0
crops/s its log reports = 85.7 min, plus that 3.5 min val). Both runs sit at
the same throughput — 15.0 vs 15.7 crops/s — so the whole ladder is ~5.4 h,
not the 13.4 h the old figure implied. Take run walls from the daemon job
record, never from `wall`.

Then `eval_sfx.py` + `eval_manga109.py`, and `eval_table.py --write` so the
row lands on the one basis.

**Gate.** ♡-blind sincos ≥ **390** / 617 (B′ 375 + 15, outside the 374–380
band); strict reported beside it (the SSL arms lost hearts while ♡-blind stayed
flat — the same trade here would be a fail on the thing the labels were for).
COO SFX / speech must not drop below B′ (2127 / 2259).

**Kill.** `vl16_pl_40k` ≤ 380 → the pseudo-label lever closes; write the kept
set's ♡ share next to the verdict so the reason is legible. Round 2 (student
becomes teacher, re-sweep the full pool) only after a round-1 lift.

### Result — `vl16_pl_20k` PASSES the gate (2026-09-09)

Ladder collapsed to the single 20k arm: each rung retrains the whole 77k
recipe, so the append is marginal (4k = +5 min on an ~89 min base) and a ~5 %
append was underpowered against a 374–380 band. Run: 6072 steps, 105 min,
final loss 0.083.

| metric | B′ | `vl16_pl_20k` | Δ | gate |
|---|---|---|---|---|
| sincos SFX ♡-blind | 375 / 617 | **402** / 617 (65.2 %) | **+27** | ≥ 390 — **pass** |
| sincos SFX strict | 312 | **334** | +22 | reported — rose too |
| COO SFX | 2127 / 2558 | **2189** (85.6 %) | +62 | must not drop — held |
| COO speech | 2259 / 2559 | **2260** (88.3 %) | +1 | must not drop — held |
| in-domain val SFX | 86.2 % | 86.1 % | −0.1 | — |
| in-domain val speech | 88.2 % | **90.8 %** | +2.6 | — |

Best result the line has produced: col100 reached 377, ×3 epochs 360, and every
SSL arm was flat or worse. Two things make it hard to explain away. **Strict
rose with ♡-blind** (+22 / +27), so unlike the SSL arms this did not trade
hearts for kana — the 442 ♡ rows the append adds against COO's 337 land where
the plan said they would. And **COO rose too** (+62 SFX), an independently
labelled set the pseudo rows never touched, so the lift is not sincos-specific.

The in-domain speech jump (+2.6) is the pool's shape showing through: vertical
crops agreed at 64.7 % vs horizontal 29.4 %, and vertical is the speech-shaped
half, so a good share of the rows entering the manifest tagged `sfx` are
really speech.

**Open, in this order.** (1) `vl16_pl_40k` — 47,939 kept rows are already on
disk, so it is one ~127 min run and no new sweep; the ladder's shape is unmeasured
with one point. (2) Round 2 (student re-teaches the pool) is now unlocked by
the round-1 lift. (3) P2's 1k hand labels would still calibrate the filter's
precision, which is inferred here, never measured.

## P2 — ~1k target-domain labels, diversity-picked

Interpretation of "the most different items": the labelled set we have is COO
(grey Manga109). Pick the AnimeText crops that are **farthest from COO train in
the tower's feature space** — the target-domain crops the labels say least about
— and hand-label those. (If the intent was instead the COO rows farthest from
the COO centre, the same script runs with the pool swapped; say so.)

**Features (`ocr/pick_label_set.py`).** Mean-pooled `last_hidden_state` of the
**B′ tower** (task-aligned; `tower_drift.py::feats` is the encoder path), L2-
normalised, on a 20k COO-train sample and on a 200k AnimeText draw (reuse P1's
draw so every picked crop already has teacher reads). ~15 min on the GPU.

**Selection.** Score each candidate by cosine distance to its nearest COO
neighbour (faiss / torch top-1 over 20k). Then, to stop the pick collapsing onto
one cluster of outliers, farthest-first (k-center greedy) over the candidates
seeded with the 20k COO set, taking 700; plus 300 uniform from the candidates
above the median distance. Drop candidates whose B′ read is empty or a
runaway (junk: chrome, tally marks, Latin logos) — this is a *text* label set.
Write `assets/labels_animetext_1k.tsv` in the sincos schema (`row, stem, box,
det_score, engine, kind_rec, text_rec, kind_hand, text_hand, status, src_row,
note`) with `text_rec` = B′'s read and `status = draft`, so the existing
draft → check flow applies unchanged.

**Labelling.** B′ is ~60 % right on the gate, so roughly 400 of 1,000 drafts
need an edit. The sincos pass used a vision-LLM draft between the reader and the
user (`note: opus …`); do the same here — it halves the typing.

**Use.** Split 800 train / 200 held-out (`split` column). Two jobs:

1. **Calibrate P1's filter** — run the filter on the 1k; precision of kept
   rows vs `text_hand`, ♡-blind and strict, per kind. This is the number that
   decides whether P1's 200k kept set can be trusted, and it needs only the 1k.
2. **Append arm** `vl16_lab800` — B′ recipe + `--extra_manifest labels_at
   --extra_repeat 4` (3,200 effective rows, ~4 % share — col100's scale). The
   200 held-out are a **target-domain val**: in-domain val has not predicted the
   gate once (`eval.md`), so a second number that does is worth more than the
   arm itself.

Gate as P1's. Expectation: small on its own (800 rows against 77k); the value
is 1. and the held-out val.

## P3 — does a drifted tower actually hurt the SFT? (cheap)

What is known: the 30-step pixel-SimMIM tower (cos 0.38, norm ×0.22) fed a
30-step SFT scored **0 %** where the stock-tower smoke scored 48 %
(`sft_init_tower_smoke` vs `vl16_tower_smoke`). That conflates two things — the
tower was bad on its *own* objective too (30 steps), and 30 SFT steps cannot
re-align a projector. So "drifted-but-SSL-good hurts SFT" is asserted, not
measured. The read-through gate (cos ≥ 0.9) was written on that assertion, and
it is what forced the anchored target that turned out inert.

Ladder, cheapest first. Same seed, same `--max_train`, same val for every rung.

**L0 — re-alignment budget (no new code, ~1 h).** B′ recipe from the existing
collapsed tower vs stock, short: `--max_train 2000 --val_limit 512` (4k rows →
250 steps at bs 8×2, ~25 min each):

```
finetune_vl16_lora.py --run p3_l0_stock     --train_tower --max_train 2000 --val_limit 512 --skip_stock_val
finetune_vl16_lora.py --run p3_l0_collapsed --train_tower --max_train 2000 --val_limit 512 --skip_stock_val \
    --init_tower output/ocr/simmim_smoke/ep1/tower.safetensors
```

Read: if the collapsed init lands within a few points of stock-init at 250
steps, projector re-alignment is cheap and the gate was over-cautious. If it is
still near 0, the collapse destroyed information rather than rotating it — but
L1 still has to run, because this tower never optimised its objective.

**L1 — a tower that is good on the pixel objective (~3 h).**
`ssl_tower_simmim.py --target pixel --run simmim_pixel_e1 --epochs 1` on
`draw20k` (the S1 shape: 4.4k steps), `--save_every 1000`. Log the read-through
numbers *for the record only* — do not gate on them. Then L0's pair from
`simmim_pixel_e1/ep1/tower.safetensors`. This is the actual test of the
user's question: a tower that has left the projector's space *and* is good at
its SSL task.

**L2 — is the information still there? (one flag, ~40 min).** Add
`--train_projector` to `finetune_vl16_lora.py`: tower frozen, projector (26 M)
in the fp32-master path + the LM LoRA — the selector already matches
`model.projector.`, so it is a prefix filter on `tower_param_names`. Run from
the L1 tower and from stock at L0's budget. If projector-only from the drifted
tower matches arm B's ballpark (66 % in-domain, frozen tower), the drift was a
rotation the projector can undo; if not, it was a loss.

**Decision.** L1's SFT within ~3 points of stock-init at equal budget →
retire the read-through gate, and an **EMA-teacher SSL arm** (data2vec / DINO:
the target moves with the student, intermediate-layer average, no stock
anchor) is allowed to run un-gated with S2's rule (≥ +15 over B′, kill at
≤ +10). L1 far below stock → both ends of the dial are now measured (anchored =
inert, un-anchored = harmful), label-free tower SSL closes for good, and no EMA
arm runs.

Cost: L0 ~1 h, L1 ~3.5 h, L2 ~40 min. All through `make daemon-run` with
`ANIMA_MANGA109S_ROOT` / `ANIMA_ANIMETEXT_ROOT` set (the AnimeText path holds a
space — `--manifest_name`, never the path, in `ARGS`).

## Scripts to write

- `ocr/pseudo_label.py` — `sweep` (reader × draw → parquet), `filter` (guards +
  ♡-blind agreement + the kept-set stats), `manifest` (COO schema, absolute
  paths, the `--speech_ratio` it wants printed).
- `ocr/pick_label_set.py` — pooled B′-tower features, nearest-COO distance,
  k-center greedy + uniform tail, sincos-schema TSV with B′ drafts.
- `finetune_vl16_lora.py --train_projector` (P3-L2).

## Anti-re-proposal

- No lr / mask / corpus sweep on the anchored SSL target (`tower_drift.md`).
- No colorized swap; appends stay ≤ a few percent until a lift is measured.
- No context / margin / prompt-hint arms (findings § Context).
- Never put a `/71` or pre-`acd41d72` number next to one from `eval.md`.
