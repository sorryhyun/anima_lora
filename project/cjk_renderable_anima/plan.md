# plan — cjk_renderable_anima (forward plan, 2026-09-14 evening; P0 added 2026-09-14 night)

> Supersedes the decision state in the 2026-09-13 wake plan
> ([`reports/wake_plan_2026_09_13.md`](reports/wake_plan_2026_09_13.md)); the dated run record is
> [`reports/`](reports/README.md). Verdicts live in [`findings.md`](findings.md);
> this file is only what comes next: phases, gates, kill criteria, and the
> recipe of record. Nothing is shipped.

## Where the line stands (four sentences)

A frozen DiT, frozen llm_adapter and frozen text encoder render every kana
and short common words from a delta on the vocab pack's ext rows
(Run 3: 34/36 kana, します/してる as one-row units), EN bit-exact by
construction. The frozen adapter + DiT read piece sequences in order as a
pretrained capability (nonsense EN words 22/24), and a static row table
trained on strings carries order and count through that path (`flip`
first-glyph-follows-caption 28/48 vs 5/48, unseen 3-kana 3/16, `line`
0 → 2/32). Whatever unit count the training distribution has lands in the
rows — singles-only rows draw one unit, strings-only rows draw several
(singles 34 → 5/36) — so the count must come from the caption, not the
row. The remaining misses are the repeat mode (ううう, ねねね) and the
per-row exposure grind for new characters; no DiT-side change is needed
for sequences. The 256² kill was drawn under the default σ sampling and
does not extend upward: at 384² the base spells EN 24/24 and the 512²
band rows read 21/36 (2026-09-14, `rows_w24_band/eval_384/`), so canvas
shape is an open lever — for cost, and for rows that stop carrying one
blank 512² layout.

## The target artefact (unchanged)

A vocab pack: a static table over ext rows, loaded by the existing
`llm_adapter.embed` hook, shipped as safetensors + mapping json under the
same `vocab_pack` key (training, TE caching, `inference.py`,
`GenerationRequest`, the register node). Rows are Qwen pieces; a new pack
changes the digest → `make preprocess-te ARGS=--overwrite` for CJK
captions. Anything that leaves this form (DiT/adapter LoRA, runtime gate)
is a fallback, not a phase.

## Recipe of record

The strings arm launch with the mixed data (P1). Every GPU stage through
the daemon (`daemon` skill), `--stall-timeout 0`; the data stage is
CPU-only and safe inline.

```bash
# data: 30 % singles + 70 % random-order 2–4-piece strings of trained rows
.venv/bin/python project/cjk_renderable_anima/probes/wake_probe.py \
    --stage data --arm encoder --data_tag wm --units kana --units words:120/held=8 \
    --strings_only --n_strings 6000 --single_frac 0.3

# train + eval: Run 3 table warm-started, g frozen, band 0.5–0.9
make daemon-run ARGS="--label wake-mixed --stall-timeout 0 --queue \
    project/cjk_renderable_anima/probes/wake_probe.py --stage train eval --arm encoder \
    --data_tag wm --arm_tag w120_s8k_mixed_warm --train_steps 8000 --batch 4 \
    --t_min 0.5 --t_max 0.9 --compile 1 --grad_ckpt 0 --aggressive_recompute 0 \
    --seeds 2 --no_floor --lr_common 1e-3 --common_cap 0.75 --out_scale 0.0625 \
    --lr_enc 0 --lr_decay cosine --kill_spread_step 0 --kill_max_row 0 \
    --enc_pool spatial --font_mode mean --head_init random --init_spread 1.0 \
    --init_encoder output/wake_probe/encoder_wd_w120_s8k_fres_warm/trained.pt \
    --init_free    output/wake_probe/encoder_wd_w120_s8k_fres_warm/trained.pt \
    --free_residual 1e-3 --lr_free 1e-3"
```

Fixed by measurement, do not retune without a new reason: canvases from
the P0a pool — singles `--shapes "384,448,512:2,384x512,512x384"`,
strings `"512,640,768:2,512x768,768x512"` (P1, one measurement owed on
the 768 family's batch); σ band
**0.5–0.9** (order/count/identity are decided at σ 0.5–0.8 in the
base, identity of kana rows at 0.8, nothing above 0.9 — the sampler is a
hard band, so the band is the weighting); `g` frozen (it is a prior,
`free_ratio` 0.9); rows lr 1e-3 in row-norm units; cosine decay; eval
groups `single word word_held line flip str3 combo corpus en`, both
readers, largest box, plus the EN order control (`order_probe.py`) on any
change that touches EN.

## Phases

Order (2026-09-14 night): **P0a → P0b → P1 → P2 → P3 → P4.** P0a decides
the canvas pool every later phase trains on; P0b is the singles table at
scale; P1 onwards is the strings work as planned, on that pool.

### P0a — canvas shapes A/B — **PASSED 2026-09-14 (better at cost)**

Same recipe as Run 3 (`encoder_wd_w120_s8k_fres_warm`: fres, band
0.7–0.9, 8 000 steps, batch 4, lr_enc 3e-5, `g`/`f` warm-started from
`encoder_w2_held32_s6k_cos_rinit_fres`), same inventory (`wd`: 92 kana +
120 words, 8 held out), the data dir rebuilt with a shape pool. The
instrument is in (`--shapes` on data, per-shape latent caches and
one-shape batches on train, `--eval_shape WxH` on eval; 512² renders are
bit-identical to the old code, so `wd` itself is untouched).

```bash
# data (CPU): singles pool 384–512, squares and 3:4 / 4:3
.venv/bin/python project/cjk_renderable_anima/probes/wake_probe.py \
    --stage data --arm encoder --data_tag wds --units kana --units words:120/held=8 \
    --n_single 40 --shapes "384,448,512:2,384x512,512x384"

# A: matched exposure (8 000 steps) — the cost readout
make daemon-run ARGS="--label wake-shapes-a --stall-timeout 0 --queue \
    project/cjk_renderable_anima/probes/wake_probe.py --stage train eval --arm encoder \
    --data_tag wds --arm_tag w120_s8k_fres_warm_shp --train_steps 8000 --batch 4 \
    --t_min 0.7 --t_max 0.9 --compile 1 --grad_ckpt 0 --aggressive_recompute 0 \
    --seeds 2 --no_floor --lr_common 1e-3 --common_cap 0.75 --out_scale 0.0625 \
    --lr_enc 3e-5 --lr_decay cosine --kill_spread_step 0 --kill_max_row 0 \
    --enc_pool spatial --font_mode mean --head_init random --init_spread 1.0 \
    --init_encoder output/wake_probe/encoder_w2_held32_s6k_cos_rinit_fres/trained.pt \
    --init_free    output/wake_probe/encoder_w2_held32_s6k_cos_rinit_fres/trained.pt \
    --free_residual 1e-3 --lr_free 1e-3"
# second eval of A on a non-square canvas
make daemon-run ARGS="--label wake-shapes-a-384x512 --queue \
    project/cjk_renderable_anima/probes/wake_probe.py --stage eval --arm encoder \
    --data_tag wds --arm_tag w120_s8k_fres_warm_shp --eval_shape 384x512 \
    --eval_tag 384x512 --eval_groups single,word,en --seeds 2"
# B: matched wall — A's recipe at the step count that equals Run 3's 58 min
```

Band 0.7–0.9 is measured at 384² too: the same-noise classifier on the
512² band rows at 384² (`rows_w24_band/classify_384/`, job
`20260914-152513-e84609`) peaks at σ 0.80 with top-1 0.73 (512²: 0.69),
chance at 0.65 and 0.95 — identity's σ does not move with canvas size on
this grid, so one hard band serves the whole pool.

Readouts, both seeds, both readers, largest box, 512² eval unless
stated: singles /36, word /32, EN /24, wall min, it/s; A also at 384×512.
Run 3 bar: singles 34/36, word 9/32, EN 24/24, 58 min at 2.3 it/s.

Gate — **mixed shapes are better or cheaper**, either of:

- *cheaper at parity*: A singles ≥ 32/36 and word ≥ 7/32 and EN 24/24,
  with A's wall ≤ 0.75 × Run 3's (the pool's mean token count is 0.72 ×
  512²'s, so ≥ 1.3 × it/s is expected; below that the compile families
  ate the gain); or
- *better at cost*: A singles ≥ 34/36 with the 384×512 eval singles ≥
  28/36 (a non-square canvas the 512²-only rows never saw), whatever the
  wall.

B is run only when A passes cheaper-at-parity, to state the "same
result at 0.6× budget" claim on its own numbers (gate: B singles ≥
32/36).

**Result (arm `encoder_wds_w120_s8k_fres_warm_shp`, jobs
`20260914-154123-df61b7` / `-8978a0`).** Same ruler for both arms
(largest detector box; the plan's 34/36 bar is the sfx reader):

| eval | A mixed | Run 3 |
|---|---|---|
| single 512², sfx / both readers | **36/36 / 29/36** | 33/36 / 26/36 |
| word 512², sfx / both | 10/32 / 7/32 | 9/32 / 9/32 |
| line / combo / corpus | 0/32 / 0/36 / 0/20 | 0/32 / 1/36 / 0/20 |
| EN 512² | 24/24 | 24/24 |
| single **384×512**, sfx / both | **34/36 / 29/36** | — |
| word / EN 384×512 | 11/32 (9 both) / 21/24 (floor 21/24: base case + punctuation, delta bit-exact) | — |
| train wall / it/s | **47.8 min / 2.79** | 58 min / 2.30 |

Instruments tracked Run 3 step for step (`rel` 1.50 vs 1.45,
`free_ratio` 0.91 vs 0.90, `table_pr` 2.80 vs 2.83 at step 7525). Every
512² single miss is a vl misread of a correct render (ひ→U, ち→5, ぬ→奴,
キ→丰); at 384×512 two are real (み→る, ケ→タ one seed each). Verdict:
**better at cost** — singles above Run 3 at 512², 29/36 on a canvas the
512² rows never saw, at 0.82× the wall. Cheaper-at-parity missed on the
wall alone (1.21× it/s, not 1.33×: fixed per-step cost + the 512² family
at 34 % of items), so B was not run. The pool is the recipe of record.

Outcomes: pass → the pool is the recipe of record for singles and the
P1 pool (`512,640,768:2,512x768,768x512`) is used for strings. A singles
in 28–31 → shapes cost a little identity; run once more with
`512:3` weighting, then decide. A singles < 28/36 → the pixel loss at
384–448 dilutes identity (the 256² verdict extended one tier up); stay
at 512², P0b and P1 run at 512² and the shape lever is closed with this
number.

### P0b — singles at scale (next; ~2–3 GPU-hours)

**Launched 2026-09-14 17:13** (a kana-only launch at 17:03 was stopped at
3.6 min to add kanji, user call). Instrument in: `--kana_ext` (the 68 voiced
/ handakuten / small kana, each its own Qwen piece + pack row) and
`--kanji 100` (the 100 most frequent corpus kanji that are single pack rows,
by **frequency, not school grade** — 出気人日私今…何(27th)…実:17; the rows the
corpus uses and the base for kanji-bearing words later; the list carries the
corpus's adult vocabulary 精 射 液), both as singles ×`n_single`; eval groups
`single_ext` (12 hira + 6 kata) and `single_kanji` (18), each on its own rng
so `single` / `word` / `word_held` / `combo` / `en` stay P0a's sets (`line`
changed). Data `wdsek` (`wds` args + `--kana_ext --kanji 100`): 15 880 items
(font 15 580 incl. 700 combos, corpus 300 — corpus crops stay kana-only via
`_corpus_lines`' `KANA_RE`, so kanji bubbles are not used yet). Arm
`encoder_wdsek_w120_s24k_p0b`: P0a's recipe at **24 000 steps fixed** (user
call; ≈ 2.4 h at 2.79 it/s, ≈ 6 epochs, ≈ 240 renders/row — below the
~400/row the budget line assumed), band 0.7–0.9 (singles band),
`--lr_enc 0` (`g` frozen per the recipe of record), `g` + `f` warm from P0a
A, new rows' `f` from zero. Jobs `20260914-171315-ba4465` (train + eval
512²), `20260914-171322-ecb831` (384×512 eval: single, single_ext, single_kanji,
word, en). Kanji gate (not pre-registered before launch, set now):
`single_kanji` ≥ 18/36 — half the kana bar's rate, since kanji start from
`g` alone at 60 % of the renders/row. `native` on 8 kana owed after the gate.

The kana table every later phase warm-starts from: the full kana
inventory as singles, on the P0a pool.

- **Inventory**: 92 basic kana + voiced / handakuten / small kana
  (がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ + katakana
  counterparts + ぁぃぅぇぉっゃゅょ / ァィゥェォッャュョ, ~70 rows) + the 120
  words as units. Instrument owed first: a `--kana_ext` data flag that adds
  those rows to the inventory and an eval group `single_ext` (18 drawn),
  so the new rows are scored separately from the 92.
- **Distribution**: singles / word units only (`--n_single 40`); no
  strings — the count prior is P1's problem, and P1 warm-starts from this
  table (the plan's 2-step, done once).
- **Budget**: 2–3 GPU-hours at the it/s P0a measured; steps = that wall ×
  it/s (at ~3 it/s ≈ 22–32 k steps, batch 4 ≈ 10 epochs over ~11 k items,
  ≈ 400 renders per row — above W1's 290). Cosine decay to 0 over the
  full run.
- **Warm start**: `g` and `f` from P0a's A
  (`encoder_wds_w120_s8k_fres_warm_shp/trained.pt`); new rows' `f` from
  zero.

Gate: basic singles ≥ 32/36, `single_ext` ≥ 24/36 (the 67 % bar the rows
band hit on 24 kana), word ≥ 8/32, EN 24/24, and the P0a non-square
eval ≥ 28/36 when the pool is in. Instruments: `table_pr`, `rel_max`
(does the table hold ~280 rows), plus the `native` stage on 8 kana
before P1 starts — the scene-prompt layout leak is what the shape pool
is supposed to remove.

Fails on `single_ext` alone → exposure, not capacity: extend by one more
hour once. Fails on the basic 92 → the table does not hold ~280 rows at
once; see P3 kill criteria (per-block packs).

### P1 — mixed distribution (after P0b; ~1 h 20 min)

On the P0a pool for strings (`--shapes "512,640,768:2,512x768,768x512"`
when P0a passed — bubbles with 2–4 pieces need the canvas; batch 4 at
768² may need `--grad_ckpt 1` or a 2-item batch for that family; measure
once). Warm start from the P0b table.

`--single_frac` on the strings data: each font item is a single kana/word
with probability 0.3, otherwise a 2–4-piece string; corpus lines as before.
Warm start from Run 3 (not from the strings arm — its rows carry the
multi-unit prior).

Gate (both, same eval set as the strings arm):

- **singles ≥ 30/36** (Run 3 bar 34/36 minus reader noise) — the count
  prior is gone from the rows;
- **`flip` ≥ 4/48, `str3` ≥ 3/16, `line` ≥ 2/32, and the `flip` order
  statistic (first glyph = caption's first piece) ≥ 24/48 with last-piece
  ≤ 8/48** — nothing the strings arm won is lost.

Outcomes: pass → P2. Singles pass, strings drop → the two priors compete
in `f`; try `--single_frac 0.5` once, then P2 on whichever holds both.
Strings pass, singles still < 30 → the count prior is in `c` (shared
bias), not `f`: freeze `c` (`--lr_common 0`) and rerun once. Both fail →
see kill criteria.

### P2 — repeat mode

The second slot copies a neighbour (ううう for うネ). Two levers, one run
each, on the P1 data:

1. **Repeated-piece strings as data** (`--repeat_frac`): strings with a
   deliberate repeat (うう, ねねね) at ~10 %, so a repeated glyph is only
   right when the caption has the repeated row. Cheapest; tests whether
   the rows can separate "same row twice" from "two rows".
2. **Swap negatives in the loss** if (1) is flat: same render, caption
   with one piece replaced by its neighbour, a margin on the FM error
   (the W2c swap-contrastive shape, which was shelved for *singles* because
   it cannot fix combos — here the combo is the target).

Gate: `flip` both-pieces-present ≥ 36/48 and exact ≥ 12/48; `str3` ≥
6/16; `line` ≥ 8/32. This is the product-relevant bar: a 2–3-piece
bubble reads.

### P3 — inventory and bake

- **Contextualisability transfer probe** (10 min data + ~40 min train):
  add ~8 kana absent from the table (dakuten singles: が ぎ ぐ げ ご ざ じ
  ず) with **singles-only** exposure on top of the P2 table (`f` frozen for
  old rows), then eval strings mixing new and old rows. Renders → new
  characters need only singles exposure (the 2-step is done once);
  collapses → every new batch of characters trains on mixed data.
  This decides the kanji budget.
- **Full kana inventory**: voiced / handakuten / small kana ≈ 70 rows at
  40 renders/row, in the mixed distribution.
- **Common words** to the kana bar: words-only continuation warm-started
  from P2 (`f` non-zero), single-word bubbles as the eval.
- **Bake**: `trained.pt` → `ext_embed` rows summed into a pack copy,
  safetensors + json, new digest; `examples/09_cjk_vocab_pack.py` smoke; `make preprocess-te
  ARGS=--overwrite` note in the model card.
- **Scene prompts** (`native` stage) on the baked pack before anything is
  published: a kana clause hung off a real scene prompt, the rows' layout
  prior measured in composition.

### P4 — kanji at scale (W4)

An exposure budget (~1 GPU-day jōyō in one table) whose shape P3's
transfer probe sets. Data: IDS-balanced, repeated-atom composites
down-weighted (Run 2 leak), mixed distribution. Not a research phase
unless P3 shows the rows cannot hold 2 000+ addresses at once (table PR
and `rel_max` are the early instruments).

## Fallbacks (not phases)

- **Slot rows** — tokenizer routes the i-th piece of a quoted string to
  row (piece, i); train `Δ(piece, i) = Δ_piece + P_i`, bake the sum. Still
  a pack. Only if P1/P2 cannot hold order and count in one table.
- **W3 DiT-side ext-gated cross-attn LoRA** — only if slot rows also fail;
  carries the EN-safety list (ext gate, position mask, EN replay on mixed
  prompts) and leaves the native pack form.

## Kill criteria

- P1 fails both gates after the two pre-registered reruns → order/count
  and identity cannot coexist in one static table → slot rows.
- P2 both levers flat (`flip` exact < 8/48) → the repeat mode is a DiT
  prior the rows cannot override → ship the single-unit pack (P3 bake of
  the Run 3 table: kana + words as units) and put strings on the W3 list.
- P3 transfer probe collapses *and* the full-kana mixed run drops old
  rows below 30/36 → the table does not scale in place; kanji (P4) is
  re-budgeted as separate packs per script block, or killed.

## Instruments and gotchas

- `probes/wake_probe.py`: stages `salad data train eval classify
  classify_str native`; `--strings_only --n_strings --word_frac
  --n_flip_eval --n_str3_eval` (data), `--cls_lang en|ja --cls_pairs
  --cls_triples` (classify_str), `--t_min --t_max` (hard band).
- `probes/order_probe.py`: base-model EN order control (nonsense
  multi-piece words); run it after any change on the EN path.
- `probes/wake_geometry.py`: table PR / pairwise cos / composition pairs.
- 256² is dead (base cannot spell EN there, 11/24) but 384² is not (EN
  24/24, 512² rows read 21/36 there); the pool is P0a's call. Block
  compile before grad-ckpt; batch 8 OOMs at 512²;
  clear `conds_cache` on delta-scale switches; ext rows are Qwen-piece
  keyed (a kana "pair" is only an order contrast when every permutation
  tokenizes to its own rows); the readers under-read kana — look at the
  sheet before counting a miss; wait on daemon jobs with a Monitor poll of
  `job.json`, not a background `daemon-wait` (killed on low host memory).
