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

## State (2026-09-20)

The line trains a **rows** arm on self-generated scene composites (the S
line): one ext-row delta per unit, no encoder, no `c_flat`, no flat items.
Two steps — **step 1** the single-glyph seed table, **step 2** the sentence
pass on top of it.

- **Every launch states its pack.** `ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack`
  is raw (log sha `7b9fce0bb57b`); `configs/base.toml` still defaults to
  `anima_cjk_vocab_pack_preview` (= raw + a 502-row delta, sha
  `5f52aefce82a`), which silently based every run between 2026-09-17 17:36
  and 2026-09-19 — the arms and what it invalidated are in
  [`reports/s2b_and_raw_pack_rerun_2026_09_19.md`](reports/s2b_and_raw_pack_rerun_2026_09_19.md).
  Those preview-pack arm dirs were deleted 2026-09-19; the reports are their
  only record.
- **Current seed table: `step1_0920`** (2026-09-20, arm
  `rows_step1_0920_s53k`) — `step1_0919`'s ΔFM argv and data with one flag
  changed, `--box_share 0.25`, the area-independent in-box loss. Singles
  10 → **20**/36, `single_kanji` 3 → 8, native `en` 8 → **19** of 64, with
  the scene *better* held (en cos 0.931 → 0.934). It closes about a third of
  the native gap to the plain seed `src53k` (13/36 singles, native 36) and
  not all of it; `single_small` is 0/36 on both.
- **Step 2 on record: `step2_0919`** (2026-09-19, arm
  `rows_step2_0919_plain_6k`) — the sentence pass on the *older*
  `step1_0919` seed, plain FM, 6 k steps sized to its pool. Sub-exact pooled
  lift +0.086 with the lift now on held groups too (`short` +0.125,
  `short_held` +0.141); singles held at the seed's level. Owed: the same
  step on `step1_0920`.
- **Ruler:** exact match is floor-saturated on every multi-glyph group;
  `src/probe/sub_exact.py` (glyph recall minus a permutation control) is
  the ruler that orders sentence arms.

[`plan.md`](plan.md) is the single forward plan (step 1 recipe, step 2 and
the vocab → merge → sentence loop, the kanji budget K).
[`synth.md`](synth.md) is the S line as built, including the exact
`step1_0920` / `step2_0919` argv. [`findings.md`](findings.md) holds the
settled verdicts one screen per topic; [`reports/`](reports/README.md) is
the dated run record (indexed: W0–W2, W2d Runs 1–3, order probe, σ
diagnostic, strings arm, canvas-shape gate, the S line). The four
`plan_synth*.md` files are archived under `_archive/cjk_renderable_anima/`
(`plan.md` carries the redirect table).
[`diagram.html`](diagram.html) is the one-figure picture of what trains and
how (frozen Anima path + the address table; open in a browser), **the
formulation written out** (*The address, written out*), the pack → step 1 →
step 2 → bake pipeline with the warm-start guard, and the "why it is hard"
figure: the row → glyph map is a hash, so every shortcut to `f` (encoder,
composition, contrastive, codebook, warm start) is held-out 0 and the cost
is exposure per row.

## What is established

| claim | evidence | where |
|---|---|---|
| The frozen DiT holds JA glyph units; a rows-only ext-row delta makes it draw a requested kana | W1 8 hiragana 7/8 at seed 0; identity decided at σ ≈ 0.8; EN 24/24 bit-exact | `reports/wake_w0_w2_2026_09_13.md` |
| Trained addresses are near-orthogonal random directions, not shape coordinates | held-out kana 2–5/64 across every encoder lever (data jitter, random init, decorrelation, free residual); IDS composites 0/16 | `reports/wake_w2d_encoder_2026_09_13_14.md` Runs 1–2 |
| The hybrid table `Δ_r = g(x_r) + f_r` renders every trained inventory | Run 1d 24/24 (12 kana); Run 3 **34/36 with all 92 kana** in a 246-row table | `reports/wake_w2d_encoder_2026_09_13_14.md` Run 1d; `reports/wake_words_strings_2026_09_14.md` Run 3 |
| **One ext row can be a multi-glyph word** — the DiT reads a row as a *unit*, not a glyph | Run 3: します 2/2, してる 2/2, きた こう いや もう ッド from one row each (9/32 at 40 renders/row) | `reports/wake_words_strings_2026_09_14.md` Run 3 |
| A sequence of trained addresses renders exactly **one** unit; which one is not positional | Run 3 `line` 0/32 with row coverage 39/39 (そオニ→ニ, ほらそれ→ら, いやいい→いい, なにそれ→な) | `reports/wake_words_strings_2026_09_14.md` Run 3 |
| **A static row table can carry order and count** — strings arm (no singles, 2–4-piece random strings, band 0.5–0.9): unseen `str3` 3/16, `flip` 4/48 with first glyph = caption's first piece 28/48 vs last 5/48, `line` 2/32 (from 0); singles fell 34 → 5/36 because the rows absorbed the multi-unit prior; repeat mode is the main miss | `reports/wake_words_strings_2026_09_14.md` Strings arm result | `output/wake_probe/encoder_ws_w120_s8k_strings_warm/` |
| **The frozen adapter + DiT read T5 piece sequences in order** — nonsense 4–5-piece EN words (GLORPAX, MIZUKANE) render 22/24, WAY NO in the given order; EN control words were multi-piece all along (HELLO = ▁H·ELL·O) | `src/probe/order_probe.py`, base model, no delta | `reports/wake_words_strings_2026_09_14.md` Order probe |
| **The render trigger and the flat canvas are both common to every flat item, so one shared vector takes both** — S0b's cap 0.75 → 1.5 moved the trigger from `f` into `c_flat` (`f` alone 25 → 1 hit); a P0b warm start keeps identity but never grows a trigger (0/64); the pretrained quoted-EN direction Q is a canvas-free trigger but halves exact hits at inference (S0 hit & kept 15 → 9) | `findings.md` *Settled — trigger vs canvas*; `reports/synth_s0_s0b_2026_09_15.md` | `output/wake_probe/rows_synth_s0b_s24k_S0b/`, `…/native_q/` |
| Kanji at scale is an exposure budget, not a research line | Run 2: trained composites render, held-out composites are kana; jōyō 2 136 ≈ one GPU-day in one table | `reports/wake_w2d_encoder_2026_09_13_14.md` Run 2 |

The `line` row and the order-probe row together narrow the string
verdict: the DiT *does* enumerate addresses in order — for pieces it was
pretrained on. Ext rows were trained on single-unit canvases and sit
off-manifold, so nothing asked them to be contextualisable; the strings arm
then showed a static table *can* carry order and count, so no DiT-side
change is needed for sequences.

## The artefact

A vocab pack delta: any arm's `output/wake_probe/<arm>/trained.pt`
(gitignored) — `delta.ext_ids` + `delta.raw` in row-norm units, `free` (the
per-row residual), `row_text` (row → piece text), and on encoder-arm tables
`encoder` (the glyph CNN). Loading it into the pack at run time is the
`ExtDelta` hook (`src/common/hooks.py`). Baking it into a
shipped pack pair is `scripts/toolkits/bake_vocab_pack.py` (rows summed at
`ExtDelta` scale 1, json `render` block, `provenance` tier `render`; new
digest → `make preprocess-te ARGS=--overwrite` for CJK captions). The sent
24k table (503 rows) is baked as
`models/vocab_packs/anima_cjk_vocab_pack_sent_s24k/`
and symlinked into ComfyUI's `vocab_packs/` (2026-09-17); the Hub preview
pack is the same recipe's `sent_s24k_a1_s05` arm
([`deploy_plan.md`](deploy_plan.md)):

```bash
.venv/bin/python scripts/toolkits/bake_vocab_pack.py \
    output/wake_probe/rows_synth_sent_q_sent_s24k \
    --out models/vocab_packs/anima_cjk_vocab_pack_sent_s24k/ \
    --comfy_dir /media/sorryhyun/data/comfy_models/vocab_packs
```

Inventory facts that matter when extending it:

- The pack's ext rows are **Qwen pieces**. Qwen merges many JA words into one
  piece (ありがとう / いい / って / 明日 → one row each); 大丈夫 → 大+丈夫 and
  ドキドキ → ド+キ+ド+キ are sequences and inherit the one-unit limit.
- `KANA` in `src/common/text.py` is the unvoiced 46 + 46; the dakuten /
  handakuten / small kana are `KANA_EXT` (68 rows, `--units kana_ext`) and
  small kana are not a singles concept (a lone ゃ renders full-size) — their
  own 18 rows train through `--units small` (two-glyph digraphs, host row +
  small row: あっ きゃ ニャ), and the merged uses are word pieces (じゃ って
  いっぱい プロ).
- Every row needs its own exposure (≈ 40 renders/row got words to 9/32,
  kana with a warm start to 34/36). Held-out anything is 0: the shared
  encoder `g` is a prior, the identity lives in `f`.

## Formulation

[`diagram.html`](diagram.html) *The address, written out* — the address
`ẽ_r = e_r + ρ f_r`, the residual `r_X`, the paired loss `L_Δ` with the
`μ‖f‖²` pull, the σ band, and a symbol table giving every term's value,
kind and role. (The old `formulation.pdf` / `.tex` were the **Run 3**
hybrid-encoder training — `Δ_r = α(ψ_θ(x_r) − ψ̄) + c + 𝟏[r ∈ ℛ_tr] f_r` —
a design the line dropped when the encoder closed at held-out 0; removed
2026-09-20, in git history if ever wanted.)

## How to run

All GPU stages go through the daemon (`daemon` skill); the data stage is
CPU-only and safe inline. Corpus bubbles come from
`post_image_dataset/render/ja/{resized,heldout}/boxes.jsonl` (local, not in
the repo).

**The recipe of record — the `step1_0920` and `step2_0919` argv, with what
every block is doing — is [`synth.md`](synth.md) *The recipe as run*.**
Reproduced there rather than here so it stays beside the design it
implements. The two other stages that take an arm:

```bash
# native — the scene-prompt read, a second job on a finished arm
make daemon-run ARGS="--label step1-native --stall-timeout 0 --queue \
    project/cjk_renderable_anima/src/wake_probe.py --stage native --arm rows \
    --data_tag step1_0920 --arm_tag s53k --native_chars あ,か,す,日 \
    --native_clauses en,swap --seeds 2 --delta_parts full"

# target — the user's own ComfyUI captions (assets/target_prompts.txt:
# hoshino ai by @akipeko saying はい / こんにちは) rendered verbatim at the
# Comfy canvas, floor vs trained, both readers. No rulers; the question is
# only "does it say はい".
make daemon-run ARGS="--label step1-target --stall-timeout 0 --queue \
    project/cjk_renderable_anima/src/wake_probe.py --stage target --arm rows \
    --data_tag step1_0920 --arm_tag s53k --eval_shape 768x1344 --steps 30 --seeds 2"

# Archived — the W2d Run 3 encoder recipe (hybrid g + f). The encoder arm has not
# run since 2026-09-14; it is kept because extending the hybrid table needs it,
# and because a rows warm start reads an encoder source's shared `common` vector.
.venv/bin/python project/cjk_renderable_anima/src/wake_probe.py \
    --stage data --arm encoder --data_tag wd \
    --units kana --units words:120/held=8 --n_single 40

make daemon-run ARGS="--label wake-words --stall-timeout 0 --queue \
    project/cjk_renderable_anima/src/wake_probe.py --stage train eval --arm encoder \
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
`sheet_<group>.png`, `train_log.json` (with `warm_cos` / `warm_drift` on a
warm start), `trained.pt`, and **`eval_summary.png`** — the one-glance sheet:
headline numbers of everything the arm has (train, eval groups, `native/`,
`target/`) plus a hit and a miss per group, framed green / red. It is
rewritten whenever `eval`, `native` or `target` finishes; `--stage summary`
rebuilds it from the json manifests (CPU only). `src/bench/wake_geometry.py`
reads any arm's table (`--table free|raw`, `--pairs 明=日+月,…`).

Stages: `salad` (base-model probe), `data`, `train`, `eval`, `classify`
(same-noise diffusion classifier over σ), `native` (scene prompts + kana
clause), `scenes` (the S line's self-generated composite pool), `enref`
(the EN-reference renders the ruler scores against), `native_rescore`
(re-read an existing native run), `target` (the user's own captions from
`--target_prompts`, verbatim, floor vs trained at `--eval_shape`),
`summary` (rebuild `eval_summary.png`). Arms: `rows` (free delta — **the recipe
of record** since the S line opened, 2026-09-15; every run since
2026-09-14 is a `rows` run), `rows_adapter` (+ llm_adapter LoRA — drifts
EN, kept as the negative control), `encoder` (W2d hybrid; last run
2026-09-14, kept to extend the hybrid table).

**What the table trains on is one flag: `--units`** (`src/data/units.py`),
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
larger piece misses the row (the `eval_coverage.json` line); the real
corpus-crop labels are mostly wrong OCR reads of hand-lettered SFX (two in
three) — do not trust a `line` / `corpus` eval on them.

## Files

| path | what |
|---|---|
| `plan.md` | **the forward plan** — step 1 (seed-table recipe, the loss question, glyph size, α), step 2 (the sentence pass + the vocab → merge → sentence loop), K (the kanji budget), open risks, the fallbacks |
| `synth.md` | **the S line as built** — why composites, the scene prompts and pools, the instrument in build order, the ΔFM loss as built, the `step1_0920` / `step2_0919` argv, the rulers, the budget |
| `diagram.html` | the one-figure picture (open in a browser) — the frozen Anima path + the address table, **the formulation written out** (`ẽ_r`, `r_X`, `L_Δ`, the symbol table), the pack → step 1 → step 2 → bake pipeline, and the "why it is hard" figure |
| `_archive/cjk_renderable_anima/plan_synth*.md` | archived 2026-09-20: `plan_synth` (budgets/pools/rulers), `plan_synth2` (the ΔFM line Δ0–Δ2), `plan_synth3` (S2 and the loop), `plan_synth4` (the step-1 recipe arms R4.3–R4.6 and K). Kept for the arms they record; `plan.md` has the redirect table |
| `deploy_plan.md` | Hub v2 layout (`old/ delta/ comfy/ diffusers/`), bake, pre-upload gates, license, migration |
| `findings.md` | settled verdicts, rulers, gotchas, do-not-re-propose |
| `findings_seed.md` | what the 53k full-inventory table taught (2026-09-16): its evals, row-space geometry, adapter-output vs Q, transplant, pinned-trigger arms — the one-place summary for the seed question |
| `freetext.md` | the shelved FreeText line (2026-06) re-read against this one: its "no Korean glyph prior" root cause falls to krzh16; its Stage-1 attention localizer is a possible render-free ruler for wipe / frame binding / katakana (not built) |
| `reports/README.md` | index of the dated run record — W0–W2, the 09-13 plan as written, W2d, words/strings, canvas/scenes, S0/S0b, micro loop (split out of the former `history.md` 2026-09-15) |
| `reports/wake_w0_w2_2026_09_13.md` | W0–W2: hypothesis, Probe 0/1, address geometry, the 256² / 24-kana / balanced / σ-band arms, kanji probe |
| `src/wake_probe.py` | the instrument's entry point — stages salad / data / train / eval / classify / classify_str / native / enref / native_rescore / target / summary / scenes; registry in `src/stages.py` |
| `src/common/` | plumbing three or more stages share — paths, text metrics, prompts, shapes, models, hooks, readers, bubble, `render/{flat,scene}` |
| `assets/target_prompts.txt` | the `target` stage's default captions — the user's ComfyUI prompts of 2026-09-17 (hoshino ai by @akipeko at the bar, saying はい / こんにちは), one full caption per line, expected text = the quoted span |
| `src/data/` `src/train/` `src/eval/` `src/scenes/` | one package per role: its stage module(s) plus what only that stage reads (units + inventory; trainables + encoder; enref, native, classify, salad; judge) |
| `src/cli/` | argparse, one module per reading stage |
| `src/probe/` | standalone questions over a finished table — model-running: `order_probe.py` (base-model EN order control), `transplant_table.py`, `merge_tables.py`, `quote_dir_save.py`; free readers over what a stage already wrote: `sub_exact.py` (glyph-lift ruler for the floor-saturated multi-glyph groups), `cross_sheet.py` (two arms' `native` renders row-interleaved on one sheet) |
| `src/bench/` | rulers over a finished table, no sampler — `wake_geometry.py` (PR, pairwise cos, composition pairs), `rows_manifold.py`; local name, no repo `result.json` envelope |
| `tests/` | line-local tests (imports, CLI golden dump, units / shapes / CER / kinsoku): `.venv/bin/python -m pytest project/cjk_renderable_anima/tests` — not part of the repo suite |

## Open

Ordered in [`plan.md`](plan.md) *Order*; the short form:

- **Which loss the seed table takes at full inventory** (`plan.md` S1a) —
  ΔFM wins at 12 rows, plain wins at 374–434, and the two full-table arms
  also differed in lr. The control is `--pair_loss 0 --lr_rows 1e-3` on
  `step1_0920`'s own build; nothing else should run first.
- **Round 2 of the sentence loop** — step 2 on the `step1_0920` seed.
- **α as a deployment knob** (`plan.md` S1c) and **glyph size / the
  small-bubble pool** (S1b), then **K1** at `kanji:400`–`600`.
- **Publishing** — the Hub v2 layout and gates G1–G4 / G6 are unrun
  (`deploy_plan.md`).
