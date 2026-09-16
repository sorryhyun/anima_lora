# cjk_renderable_anima — make Anima render CJK text it already knows how to draw

Promoted from the wake line of [`../cjk_aware_anima_dit/`](../cjk_aware_anima_dit/)
on 2026-09-14. The premise (user's, 2026-09-13): *Anima garbles Japanese
text not because it cannot draw the glyphs but because the T5-side query
stream never gave it an address for them — wake the weights, don't teach a
script.* Confirmed for kana on 2026-09-13, extended to word-level units on
2026-09-14. Everything here runs on a **frozen** DiT, frozen llm_adapter,
frozen text encoder; the only trainable object is a delta on the vocab
pack's ext rows, so EN prompts are bit-exact by construction and the
artefact ships as an ordinary vocab pack.

[`plan.md`](plan.md) is the forward plan (phases P0a–P4, gates, kill
criteria, recipe of record); [`plan_synth.md`](plan_synth.md) is the **S
line's** live plan (status, data mix, recipe, gates, decision tree) and
[`synth.md`](synth.md) its as-built reference — the rows arm retrained
from scratch on self-generated scene composites with a per-source layout
vector (P0b's `native` and table-parts probes showed the rows address the
whole training canvas, not the glyph, and that nothing in the table
separates the two); [`findings.md`](findings.md) holds the settled
verdicts one screen per topic; [`reports/`](reports/README.md) is the dated run record
(indexed: W0–W2, W2d Runs 1–3, order probe, σ diagnostic, strings arm,
canvas-shape gate, the S line).
[`diagram.html`](diagram.html) is the one-figure picture of what trains and
how (frozen Anima path + the hybrid address table; open in a browser).
[`datacheck.md`](datacheck.md): the corpus-crop labels are mostly wrong OCR
reads of hand-lettered SFX — read before trusting `line` / `corpus` evals.

## What is established (2026-09-15)

| claim | evidence | where |
|---|---|---|
| The frozen DiT holds JA glyph units; a rows-only ext-row delta makes it draw a requested kana | W1 8 hiragana 7/8 at seed 0; identity decided at σ ≈ 0.8; EN 24/24 bit-exact | `reports/wake_w0_w2_2026_09_13.md` |
| Trained addresses are near-orthogonal random directions, not shape coordinates | held-out kana 2–5/64 across every encoder lever (data jitter, random init, decorrelation, free residual); IDS composites 0/16 | `reports/wake_w2d_encoder_2026_09_13_14.md` Runs 1–2 |
| The hybrid table `Δ_r = g(x_r) + f_r` renders every trained inventory | Run 1d 24/24 (12 kana); Run 3 **34/36 with all 92 kana** in a 246-row table | `reports/wake_w2d_encoder_2026_09_13_14.md` Run 1d; `reports/wake_words_strings_2026_09_14.md` Run 3 |
| **One ext row can be a multi-glyph word** — the DiT reads a row as a *unit*, not a glyph | Run 3: します 2/2, してる 2/2, きた こう いや もう ッド from one row each (9/32 at 40 renders/row) | `reports/wake_words_strings_2026_09_14.md` Run 3 |
| A sequence of trained addresses renders exactly **one** unit; which one is not positional | Run 3 `line` 0/32 with row coverage 39/39 (そオニ→ニ, ほらそれ→ら, いやいい→いい, なにそれ→な) | `reports/wake_words_strings_2026_09_14.md` Run 3 |
| **A static row table can carry order and count** — strings arm (no singles, 2–4-piece random strings, band 0.5–0.9): unseen `str3` 3/16, `flip` 4/48 with first glyph = caption's first piece 28/48 vs last 5/48, `line` 2/32 (from 0); singles fell 34 → 5/36 because the rows absorbed the multi-unit prior; repeat mode is the main miss | `reports/wake_words_strings_2026_09_14.md` Strings arm result | `output/wake_probe/encoder_ws_w120_s8k_strings_warm/` |
| **The frozen adapter + DiT read T5 piece sequences in order** — nonsense 4–5-piece EN words (GLORPAX, MIZUKANE) render 22/24, WAY NO in the given order; EN control words were multi-piece all along (HELLO = ▁H·ELL·O) | `probes/order_probe.py`, base model, no delta | `reports/wake_words_strings_2026_09_14.md` Order probe |
| **The render trigger and the flat canvas are both common to every flat item, so one shared vector takes both** — S0b's cap 0.75 → 1.5 moved the trigger from `f` into `c_flat` (`f` alone 25 → 1 hit); a P0b warm start keeps identity but never grows a trigger (0/64); the pretrained quoted-EN direction Q is a canvas-free trigger but halves exact hits at inference (S0 hit & kept 15 → 9) | `findings.md` *Settled — trigger vs canvas*; `reports/synth_s0_s0b_2026_09_15.md` | `output/wake_probe/rows_synth_s0b_s24k_S0b/`, `…/native_q/` |
| Kanji at scale is an exposure budget, not a research line | Run 2: trained composites render, held-out composites are kana; jōyō 2 136 ≈ one GPU-day in one table | `reports/wake_w2d_encoder_2026_09_13_14.md` Run 2 |

The `line` row and the order-probe row together narrow the string
verdict: the DiT *does* enumerate addresses in order — for pieces it was
pretrained on. Ext rows were trained on single-unit canvases and sit
off-manifold, so nothing asked them to be contextualisable. Whether the
rows path reaches strings is an open, cheap question (a strings-only arm
warm-started from Run 3); W3 (DiT-side) is the fallback if that arm stays
at 0, not the next step.

## The artefact

A vocab pack delta: `output/wake_probe/encoder_wd_w120_s8k_fres_warm/trained.pt`
(gitignored) — `delta.ext_ids` + `delta.raw` in row-norm units over 246 ext
rows (92 kana + 112 common words + eval pieces), `free` (the per-row
residual), `encoder` (the glyph CNN, only needed to extend the table),
`row_text` (row → piece text). Loading it into the pack is the existing
`ExtDelta` hook (`probes/wake/hooks.py`); baking it into a shipped pack
(safetensors + json, new digest → `make preprocess-te ARGS=--overwrite` for
CJK captions) is not done yet.

Inventory facts that matter when extending it:

- The pack's ext rows are **Qwen pieces**. Qwen merges many JA words into one
  piece (ありがとう / いい / って / 明日 → one row each); 大丈夫 → 大+丈夫 and
  ドキドキ → ド+キ+ド+キ are sequences and inherit the one-unit limit.
- The kana inventory (`KANA` in `probes/wake/common.py`) is the unvoiced 46 + 46
  only: no dakuten / handakuten / small kana singles. Those reach the table
  only inside word pieces (じゃ って いっぱい プロ); び appears nowhere.
- Every row needs its own exposure (≈ 40 renders/row got words to 9/32,
  kana with a warm start to 34/36). Held-out anything is 0: the shared
  encoder `g` is a prior, the identity lives in `f`.

## Formulation

[`formulation.pdf`](formulation.pdf) (source `formulation.tex`, build with
`tectonic formulation.tex`) — the Run 3 training written out: address
`Ẽ_r = E_r + s ρ Δ_r`, hybrid `Δ_r = α(ψ_θ(x_r) − ψ̄) + c + 𝟏[r ∈ ℛ_tr] f_r`,
rectified-flow loss on σ ∈ [0.7, 0.9] through the frozen DiT, `μ‖f‖²` pull,
AdamW groups, and what each eval group reads.

## How to run

All GPU stages go through the daemon (`daemon` skill); the data stage is
CPU-only and safe inline. Corpus bubbles come from
`post_image_dataset/render/ja/{resized,heldout}/boxes.jsonl` (local, not in
the repo).

```bash
# S line, the recipe of record (2026-09-16 evening): the sentence arm — every
# item a tategaki composite on the sl1w pool, kinds pinned as hard quotas
# (--scene_mix), rows arm, warm-started from the 53k + punctuation table.
# One daemon job, all stages; the native read is a second job on the same arm.
# <manga109s> is the local Manga109-s derivation; the path stays out of the
# repo (dialogue_2_10.tsv = dialogue_3_10.tsv + its 2-piece lines,
# <manga109s>/derived/make_dialogue_2_10.py).
make daemon-run ARGS="--label sent --stall-timeout 0 --queue \
    project/cjk_renderable_anima/probes/wake_probe.py --stage data train eval --arm rows \
    --scenes sl1w --scene_drop sl1w:332,957 --scene_min_tokens 900 --data_tag synth_sent_q \
    --units kana --units kana_ext --units kanji:200 --units words:100/held=8 \
    --units 'list:、,。,・,ー,～,〜,！,？,「,」,！！,・・・,・・・・,っ,ッ' \
    --phrase_file <manga109s>/derived/dialogue_2_10.tsv --phrase_min_pieces 2 --phrase_pieces 40 \
    --scene_mix single=0.1,short=0.5,sentence=0.4 --short_pieces 2-5 --short_max_lines 1 \
    --sentence_min_letters 6 --sentence_min_glyph 20 --sentence_fill 0.9 --scene_vertical 1 \
    --n_items 10000 --scene_frac 1.0 --natural_frac 0 --strings_frac 0 \
    --flat_bubble 1.0 --scene_fill 0.7 --scene_min_glyph 28 --scene_max_lines 2 \
    --shapes 512 \
    --init_rows output/wake_probe/rows_synth_full_fm10k_merge_punct/trained.pt \
    --train_steps 24000 --batch 4 --t_min 0.7 --t_max 0.9 \
    --compile 1 --grad_ckpt 0 --aggressive_recompute 0 \
    --lr_rows 1e-3 --lr_decay cosine --free_residual 1e-3 --box_weight 4 \
    --seeds 2 --no_floor --c_flat 0 --arm_tag sent_s24k"
make daemon-run ARGS="--label sent-native --stall-timeout 0 --queue \
    project/cjk_renderable_anima/probes/wake_probe.py --stage native --arm rows \
    --data_tag synth_sent_q --arm_tag sent_s24k --native_chars あ,か,す,日 \
    --native_clauses en,swap --seeds 2 --delta_parts full"

# Archived — the W2d Run 3 encoder recipe (hybrid g + f). The encoder arm has not
# run since 2026-09-14; it is kept because extending the hybrid table needs it,
# and because a rows warm start reads an encoder source's shared `common` vector.
.venv/bin/python project/cjk_renderable_anima/probes/wake_probe.py \
    --stage data --arm encoder --data_tag wd \
    --units kana --units words:120/held=8 --n_single 40

make daemon-run ARGS="--label wake-words --stall-timeout 0 --queue \
    project/cjk_renderable_anima/probes/wake_probe.py --stage train eval --arm encoder \
    --data_tag wd --arm_tag w120_s8k_fres_warm --train_steps 8000 --batch 4 \
    --t_min 0.7 --t_max 0.9 --compile 1 --grad_ckpt 0 --aggressive_recompute 0 \
    --seeds 2 --no_floor --lr_common 1e-3 --common_cap 0.75 --out_scale 0.0625 \
    --lr_enc 3e-5 --lr_decay cosine --kill_spread_step 0 --kill_max_row 0 \
    --enc_pool spatial --font_mode mean --head_init random --init_spread 1.0 \
    --init_encoder output/wake_probe/encoder_w2_held32_s6k_cos_rinit_fres/trained.pt \
    --init_free    output/wake_probe/encoder_w2_held32_s6k_cos_rinit_fres/trained.pt \
    --free_residual 1e-3 --lr_free 1e-3"
```

Outputs land in `output/wake_probe/<arm>_<data_tag>_<arm_tag>/`: `report.md`
(per-group CER / exact), `eval_reads.json` (every render, both readers),
`sheet_<group>.png`, `train_log.json`, `trained.pt`. `probes/wake_geometry.py`
reads any arm's table (`--table free|raw`, `--pairs 明=日+月,…`).

Stages: `salad` (base-model probe), `data`, `train`, `eval`, `classify`
(same-noise diffusion classifier over σ), `native` (scene prompts + kana
clause), `scenes` (the S line's self-generated composite pool), `enref`
(the EN-reference renders the ruler scores against), `native_rescore`
(re-read an existing native run). Arms: `rows` (free delta — **the recipe
of record** since the S line opened, 2026-09-15; every run since
2026-09-14 is a `rows` run), `rows_adapter` (+ llm_adapter LoRA — drifts
EN, kept as the negative control), `encoder` (W2d hybrid; last run
2026-09-14, kept to extend the hybrid table).

**What the table trains on is one flag: `--units`** (`probes/wake/units.py`),
repeated once per source — `kana`, `kana_ext`, `kanji:200`,
`words:100/held=8`, `chars:あかす出人日` (a hand-picked base, the
textual-inversion regime) and `list:、,。,！！` (literal ext-row units, the
punctuation arm). `*W` sets a source's draw weight in the S-line singles
pool and `/held=K` holds K units out; each source also fixes the eval group
its units are scored under (`single` / `single_ext` / `single_kanji` /
`word` + `word_held` / `single_extra`). Typed order never changes the data —
sources resolve in a canonical order, so a recipe rebuilds its data dir byte
for byte however it was typed. Default is `kana` (the 92); naming no kana
source trains punctuation / kanji / words alone.

The CLI's argparse groups name the stage **and arm** that read each flag.
They were relabelled 2026-09-16 with no flag, default or behaviour change:
`--init_rows` / `--pin_*` are rows-arm flags that used to sit in the
encoder group, `--free_residual` is read by both arms, and the S-line
group mixed data flags with train flags (`--box_weight`, `--c_flat`).

Gotchas that bite (full list in `reports/wake_w0_w2_2026_09_13.md`): never
below 512²; block compile before grad-ckpt; batch 8 OOMs at 512²; rows lr
1e-3 in row-norm units (3e-3 walks off-manifold); read the largest detector
box with both readers; clear `conds_cache` on delta-scale switches; ext rows
are Qwen-piece keyed; a caption whose tokenizer merges the target into a
larger piece misses the row (the `eval_coverage.json` line).

## Files

| path | what |
|---|---|
| `plan.md` | forward plan — phases, gates, kill criteria, recipe of record |
| `deploy_plan.md` | Hub v2 layout (`old/ delta/ comfy/ diffusers/`), bake, pre-upload gates, license, migration |
| `findings.md` | settled verdicts, rulers, gotchas, do-not-re-propose |
| `findings_seed.md` | what the 53k full-inventory table taught (2026-09-16): its evals, row-space geometry, adapter-output vs Q, transplant, pinned-trigger arms — the one-place summary for the seed question |
| `freetext.md` | the shelved FreeText line (2026-06) re-read against this one: its "no Korean glyph prior" root cause falls to krzh16; its Stage-1 attention localizer is a possible render-free ruler for wipe / frame binding / katakana (not built) |
| `reports/README.md` | index of the dated run record — W0–W2, the 09-13 plan as written, W2d, words/strings, canvas/scenes, S0/S0b, micro loop (split out of the former `history.md` 2026-09-15) |
| `reports/wake_w0_w2_2026_09_13.md` | W0–W2: hypothesis, Probe 0/1, address geometry, the 256² / 24-kana / balanced / σ-band arms, kanji probe |
| `probes/wake_probe.py` | the instrument's entry point — stages salad / data / train / eval / classify / classify_str / native |
| `probes/stage/` | one module per stage (salad, data, train, classify, eval + native) |
| `probes/wake/` | the stages' shared plumbing — models, hooks, readers, renders, inventories, encoder, argparse |
| `probes/wake_geometry.py` | row-table geometry (PR, pairwise cos, composition pairs) |
| `probes/order_probe.py` | base-model order control — nonsense multi-piece EN words, no delta |
| `formulation.pdf`, `.tex` | the training written as equations |

## Open

- **Singles at scale (next; plan P0b).** P0a passed: Run 3's recipe on a
  mixed 384–512 canvas pool (`wake_probe.py --shapes`, one-shape batches)
  renders singles 36/36 sfx at 512² (Run 3 33/36) and 34/36 at 384×512 at
  0.82× the wall, so the pool is the recipe of record. P0b trains the full
  kana inventory (basic + voiced + small, ~160 rows + 120 words) as
  singles for 2–3 GPU-hours warm-started from P0a's table (instrument
  owed: `--units kana_ext` + a `single_ext` eval group); the strings work below
  then runs on a 512–768 pool.
- **Mixed arm (after P0b).** The strings arm showed rows carry order + count but
  absorb the data's unit-count prior (singles 34 → 5). One distribution —
  30 % singles + 70 % 2–4-piece strings, Run 3 warm start, band 0.5–0.9 —
  should return singles while keeping `flip` / `str3` / `line`. Then the
  repeat mode (ううう, ねねね) is the lever: strings with a repeated piece
  as negatives.
- **W3 — DiT-side, fallback only.** Not needed for order/count (strings arm);
  kept for the case the mixed arm cannot hold singles and strings at once; the design is an ext-gated
  DiT-side change with EN kept to a *limited, measured* touch (ext gate on
  ext-free sequences, position mask, EN replay on mixed prompts — the
  "EN safety" list in `reports/wake_plan_2026_09_13.md`). To be designed in `plan.md`.
- **Word pack completion.** Words reached 9/32 at 40 renders/row with `f`
  from zero; a words-only continuation warm-started from Run 3 is the one
  cheap lever to see whether they reach the kana bar (single-word bubbles).
- **Pack bake.** Fold `trained.pt` into a shipped pack file + digest; the
  full kana inventory (voiced, handakuten, small kana) is ≈ 70 more rows of
  exposure.
- **Kanji.** Exposure budget (~1 GPU-day jōyō in one table); the data must
  not over-weight repeated-atom composites (repeat-mode leak, Run 2).
