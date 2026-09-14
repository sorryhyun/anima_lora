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
criteria, recipe of record); [`findings.md`](findings.md) holds the settled
verdicts one screen per topic; `plan_wake.md` is the dated run record
(W2d Runs 1–3, order probe, σ diagnostic, strings arm, canvas-shape gate).

## What is established (2026-09-14)

| claim | evidence | where |
|---|---|---|
| The frozen DiT holds JA glyph units; a rows-only ext-row delta makes it draw a requested kana | W1 8 hiragana 7/8 at seed 0; identity decided at σ ≈ 0.8; EN 24/24 bit-exact | `reports/wake_w0_w2_2026_09_13.md` |
| Trained addresses are near-orthogonal random directions, not shape coordinates | held-out kana 2–5/64 across every encoder lever (data jitter, random init, decorrelation, free residual); IDS composites 0/16 | `plan_wake.md` Runs 1–2 |
| The hybrid table `Δ_r = g(x_r) + f_r` renders every trained inventory | Run 1d 24/24 (12 kana); Run 3 **34/36 with all 92 kana** in a 246-row table | `plan_wake.md` Run 1d, Run 3 |
| **One ext row can be a multi-glyph word** — the DiT reads a row as a *unit*, not a glyph | Run 3: します 2/2, してる 2/2, きた こう いや もう ッド from one row each (9/32 at 40 renders/row) | `plan_wake.md` Run 3 |
| A sequence of trained addresses renders exactly **one** unit; which one is not positional | Run 3 `line` 0/32 with row coverage 39/39 (そオニ→ニ, ほらそれ→ら, いやいい→いい, なにそれ→な) | `plan_wake.md` Run 3 |
| **A static row table can carry order and count** — strings arm (no singles, 2–4-piece random strings, band 0.5–0.9): unseen `str3` 3/16, `flip` 4/48 with first glyph = caption's first piece 28/48 vs last 5/48, `line` 2/32 (from 0); singles fell 34 → 5/36 because the rows absorbed the multi-unit prior; repeat mode is the main miss | `plan_wake.md` Strings arm result | `output/wake_probe/encoder_ws_w120_s8k_strings_warm/` |
| **The frozen adapter + DiT read T5 piece sequences in order** — nonsense 4–5-piece EN words (GLORPAX, MIZUKANE) render 22/24, WAY NO in the given order; EN control words were multi-piece all along (HELLO = ▁H·ELL·O) | `probes/order_probe.py`, base model, no delta | `plan_wake.md` Order probe |
| Kanji at scale is an exposure budget, not a research line | Run 2: trained composites render, held-out composites are kana; jōyō 2 136 ≈ one GPU-day in one table | `plan_wake.md` Run 2 |

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
`ExtDelta` hook (`probes/wake_probe.py`); baking it into a shipped pack
(safetensors + json, new digest → `make preprocess-te ARGS=--overwrite` for
CJK captions) is not done yet.

Inventory facts that matter when extending it:

- The pack's ext rows are **Qwen pieces**. Qwen merges many JA words into one
  piece (ありがとう / いい / って / 明日 → one row each); 大丈夫 → 大+丈夫 and
  ドキドキ → ド+キ+ド+キ are sequences and inherit the one-unit limit.
- The kana inventory (`KANA` in `wake_probe.py`) is the unvoiced 46 + 46
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
# data: kana 92 + top-120 single-piece corpus words (8 held out), 40 renders/row
.venv/bin/python project/cjk_renderable_anima/probes/wake_probe.py \
    --stage data --arm encoder --data_tag wd --words 120 --held_out_words 8 --n_single 40

# train + eval: the Run 3 recipe (hybrid, warm start g and f from Run 1d)
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
clause). Arms: `rows` (free delta), `rows_adapter` (+ llm_adapter LoRA —
drifts EN, kept as the negative control), `encoder` (W2d hybrid, the
recipe of record).

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
| `plan_wake.md` | dated run record: W2d Runs 1–3, order probe, σ diagnostic, strings arm |
| `reports/wake_w0_w2_2026_09_13.md` | W0–W2: hypothesis, Probe 0/1, address geometry, the 256² / 24-kana / balanced / σ-band arms, kanji probe |
| `probes/wake_probe.py` | the instrument — data / train / eval / classify / native |
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
  owed: `--kana_ext` + a `single_ext` eval group); the strings work below
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
  "EN safety" list in `plan_wake.md`). To be designed in `plan.md`.
- **Word pack completion.** Words reached 9/32 at 40 renders/row with `f`
  from zero; a words-only continuation warm-started from Run 3 is the one
  cheap lever to see whether they reach the kana bar (single-word bubbles).
- **Pack bake.** Fold `trained.pt` into a shipped pack file + digest; the
  full kana inventory (voiced, handakuten, small kana) is ≈ 70 more rows of
  exposure.
- **Kanji.** Exposure budget (~1 GPU-day jōyō in one table); the data must
  not over-weight repeated-atom composites (repeat-mode leak, Run 2).
