# plan — cjk_renderable_anima (forward plan, 2026-09-14 evening)

> Supersedes the decision state in [`plan_wake.md`](plan_wake.md), which
> stays as the dated run record. Verdicts live in [`findings.md`](findings.md);
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
for sequences.

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
    --stage data --arm encoder --data_tag wm --words 120 --held_out_words 8 \
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

Fixed by measurement, do not retune without a new reason: 512² canvases;
σ band **0.5–0.9** (order/count/identity are decided at σ 0.5–0.8 in the
base, identity of kana rows at 0.8, nothing above 0.9 — the sampler is a
hard band, so the band is the weighting); `g` frozen (it is a prior,
`free_ratio` 0.9); rows lr 1e-3 in row-norm units; cosine decay; eval
groups `single word word_held line flip str3 combo corpus en`, both
readers, largest box, plus the EN order control (`order_probe.py`) on any
change that touches EN.

## Phases

### P1 — mixed distribution (next; ~1 h 20 min)

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
  safetensors + json, new digest; `examples/09` smoke; `make preprocess-te
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
- Never below 512²; block compile before grad-ckpt; batch 8 OOMs at 512²;
  clear `conds_cache` on delta-scale switches; ext rows are Qwen-piece
  keyed (a kana "pair" is only an order contrast when every permutation
  tokenizes to its own rows); the readers under-read kana — look at the
  sheet before counting a miss; wait on daemon jobs with a Monitor poll of
  `job.json`, not a background `daemon-wait` (killed on low host memory).
