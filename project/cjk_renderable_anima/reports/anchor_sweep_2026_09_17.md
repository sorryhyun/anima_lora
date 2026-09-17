# Warm-start erasure and the init anchor — μ sweep (2026-09-17)

**Question.** The sentence arm (`sent_s24k`, warm-started from the 53k + punct
table) rendered no singles at all (0/36) — did it inherit anything?

**Finding.** The warm start was loaded correctly (447/503 rows by ext id) and
erased in the first ~50 steps: AdamW at lr 1e-3 in row-norm units with no
warmup moves every coordinate ≈ lr per step regardless of the gradient, and a
0.58-norm row has a per-coordinate rms of ≈ 0.018. `train_log` shows the
signature — delta norm 98.6 → 47 by step 50, then regrown from scratch to 140.
At 24k the warm rows ended at **cos 0.096** to their source (99 % below 0.5);
the sent2_comic arm warm-started from it: cos 0.146. The canvas changed too
(frame-mix → 100 % tategaki composites on sl1w), so the FM gradient actively
opposes the old rows, not just rotates them.

**Fix.** Two flags (`src/cli/train.py`): `--lr_warmup N` (linear, all groups,
multiplies into the cosine) and `--init_anchor μ` (rows arm: μ · mean_r ‖f_r −
f₀_r‖² over the rows `--init_rows` filled, replacing the ‖f‖² pull on those
rows; new rows keep `--free_residual`). The anchor's gradient is 0 at f = f₀,
so the warmup is what saves the first steps. `train_log` gains `warm_cos` /
`warm_drift`.

**Smoke sweep.** `data_synth_sent_q` reused (no data stage), 3000 steps,
cosine, warmup 500, batch 4, σ 0.7–0.9, eval `single,single_ext,single_kanji,
word,word_held,en` × 2 seeds. Arms `rows_synth_sent_q_smk3k_a{0,0p1,0p3,1}`;
the source table evaluated on the same set as `rows_synth_sent_q_src53k`
(trained.pt symlink).

| μ | warm_cos 500 / 1000 / 3000 | single | single_ext | single_kanji | word | FM loss (last 20) |
|---|---|---|---|---|---|---|
| source 53k+punct | — | 13/36 | 20/36 | 18/36 | 0/32 | — |
| 0 | 0.566 / 0.437 / 0.352 | 0/36 | 1/36 | 0/36 | 0/32 | 0.0971 |
| 0.1 | 0.890 / 0.883 / 0.928 | 10/36 | 19/36 | 15/36 | 0/32 | 0.0980 |
| 0.3 | 0.918 / 0.926 / 0.957 | 10/36 | 18/36 | 15/36 | 0/32 | 0.0987 |
| 1 | 0.948 / 0.955 / 0.976 | 13/36 | 18/36 | 17/36 | 0/32 | 0.0995 |

- Warmup alone (μ = 0) only slows the erasure: cos keeps falling until the lr
  is gone. The anchor is the switch — any μ ≥ 0.1 keeps ≥ 80 % of the source's
  singles; μ = 1 inherits the source verbatim. Cost ≤ 2.5 % FM loss at 3k.
- `word` 0/32 is the source's own ceiling, not a sweep effect.
- The mid-run dip (0.88–0.93) is the equilibrium; cos recovers as the cosine
  winds down. A 24k schedule has ~8× the lr integral, so the dip lasts longer.

**Target stage** (`--stage target`, the user's ComfyUI captions verbatim at
768×1344, floor vs trained, both readers; `assets/target_prompts.txt`):

| table | はい (8) | こんにちは (6) |
|---|---|---|
| floor | 0 | 0 |
| sent_s24k | 0 (は fires: はうは / はは応) | 0 |
| smk3k μ 0.1 | 0 (は fires 5/8, い never follows) | 0 |
| smk3k μ 0.3 | **1** (`(はい)` in the bubble, p00 s0; p03 はじい…) | 0 |

- **こんにちは is one Qwen piece and has no trained row** — trained ≡ floor
  byte for byte. It needs a `--units list:こんにちは` row and its own exposure.
- はい = は + い, both trained; は renders, the sequence mostly does not (the
  one-unit limit). A `list:はい` row is the words-row remedy.
- Prompts without the `japanese text` tag (`Text that reads as "はい"`) fall
  to the EN trigger (latin reads only).
- Without the soup LoRA the plain sampler draws a dark bar with a small inset
  at this canvas; text is tiny. Absolute target numbers are not Comfy numbers.

**Recipe for the 24k run.** README sent recipe + `--lr_warmup 500
--init_anchor 0.3`, plus `--units 'list:…,はい,こんにちは'` and a singles
quota above 0.1 (0.1 = ~2 renders per unit over ~500 units; the 53k rows
needed ~40). Read `warm_cos` as the Pareto axis: singles holding while
short/sentence rise = one representation carries both; sentence gains only as
singles fall = the residual-only anchor (shared direction free) is the next
variant, W3 the fallback.
