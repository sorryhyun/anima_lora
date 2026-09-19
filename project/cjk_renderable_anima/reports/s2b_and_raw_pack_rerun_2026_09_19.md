# S2b round 1, the preview-pack finding, and Δ0 re-run on the raw pack (2026-09-18 night → 09-19 morning)

> One session, three things. **S2b** (plan_synth3's sentence run) was
> re-scoped to "sentence composition on warm rows only" and ran 6 k steps
> from Δ1: singles held at Δ1's level, sentences did not move. Baking its
> table for ComfyUI exposed that **every wake run since 2026-09-17 17:36
> trained on the baked preview pack, not the raw pack** — Δ1 included. Δ0
> was then re-run on the raw pack: floor 0/24, ΔFM ≥ plain at 1 500 steps,
> lr 5e-3 the best arm (13/24), and two preview-pack results do **not**
> survive — row blocks (16/24 → 0/24) and the katakana floor finding.
> Glyph-size jitter (new data flag) reads 0–1/24 and `--box_weight 1` 1/24
> — box weight 4 stays.
>
> **2026-09-19, later:** the preview-pack arm dirs (31, incl. Δ1
> `rows_synth_d1_d1_s53k` and S2b `rows_synth_s2b_s2b_plain_d1_6k`) were
> deleted; list in the caveat at the top of `plan_synth3.md`.

## 1. S2b round 1

**Re-scope (user, 2026-09-18):** a sentence pass warm-starts to grow
composition on rows that are already trained — no cold vocab inside it. The
recipe plan_synth3 carried (`merge_punct` seed, `words:100/held=8`,
`list:はい,こんにちは`, 24 k) was replaced:

- Seed Δ1 (`rows_synth_d1_d1_s53k`), loss plain, `--init_anchor 0.3
  --lr_warmup 500`, lr 1e-3 cosine, σ 0.5–0.9, box weight 4, batch 4.
- Inventory = Δ1's units only (`kana`, `kana_ext*1`, `kanji:200*1`, the punct
  `list:`). In S2a, `words:` + `--phrase_pieces 40` were 143 of 446 rows
  starting cold with no anchor. `はい` is は + い (not one Qwen piece — the
  `list:` assertion rejects it); こんにちは is one piece with no row and was
  left out.
- 15 small-kana rows (`っ ッ ゃ ょ ィ ぁ ェ ゅ ォ ぅ ァ ぇ ぃ ぉ ャ`) are not
  drawable as singles, so Δ1 lacks them; they sit in 2 693 of 8 998
  multi-glyph items (っ 1 630). Warm-started from `sent_s24k_a1_s05` via
  `--init_rows <sent>,<Δ1>` (later table overrides by ext id). Log: 365/366
  from the sentence table, 351/366 overridden by Δ1, anchor on 366 rows.
- Data `data_synth_s2b` (no `--pair_ref`): 9 998 items, single 1 000 / short
  5 000 / sentence 3 998; the covered pool without piece rows is 4 564 lines,
  **475 distinct sentences / 454 distinct shorts** (S2a's inventory: 11 119
  lines / 6 142 sentences). Steps cut 24 k → **6 k** for that pool.
- Job `20260918-222047-984df3` (64.7 min), arm
  `rows_synth_s2b_s2b_plain_d1_6k`; native `20260918-232802-b82006`.

Training: loss flat 0.083–0.102 (as every arm of this line); `warm_drift`
peaks 0.23 at ≈ step 1 100 and decays to 0.147 (S2a peaked 0.13), `warm_cos`
0.978 at the end, row norm 77–79 throughout.

| group | exact (sfx) | sub-exact lift |
|---|---|---|
| single / single_ext / single_kanji | 12 / 17 / 18 of 36 | — |
| short | 0/16 | +0.235 |
| short_held | 0/16 | +0.045 |
| phrase | 0/16 | +0.030 |
| phrase_held | 0/16 | +0.047 |
| en | 24/24 | — |
| pooled | | **+0.089** |

- **Singles equal Δ1's (12 / 17 / 18)** — the first sentence pass that did
  not pay in singles.
- Pooled lift +0.089 is under the gate (+0.161) and under S2a plain (+0.131;
  different eval strings, inventory and pool — not paired). The lift is on
  `short` (strings seen in training) only; held groups sit at +0.03–0.05
  where S2a plain had `short_held` +0.174. Read: a 475-sentence pool at 6 k
  memorises its shorts and does not generalise.
- Native (`en` / `swap`, 16 per cell): あ 7/10, か 2/3, す 3/1, 日 5/6 → 17 /
  20 of 64 vs Δ1's 22 / 19. `はい` `en` 4/16, and **1/8 on S2a's first four
  prompts where S2a plain read 5/8**; はい lands alone only on the simple
  prompts (p4 / p6 / p7), inside pseudo-JA elsewhere. `おしい` 2/16; every
  ≥ 4-glyph string 0/16 on both clauses; `swap` 0/16 on every sentence.

The table was baked as `models/vocab_packs/vocab_pack_test/` (366 rows, on
the preview pack it trained on, sha `08ad36cb40e5…`) and linked into
`../comfy/models/vocab_packs/` for the user's ComfyUI test.

**Defect in this run:** the 15 small-kana rows carry their delta twice — see
§2 (the base pack already held `sent_s24k_a1_s05`).

## 2. The preview-pack finding

`bake_vocab_pack.py` reported its base as `anima_cjk_vocab_pack_preview`.
`configs/base.toml` `vocab_pack` has pointed there since the preview pack was
built (file 2026-09-17 17:04; commit `b1028bd4` 23:20), and that pack **is**
raw + `sent_s24k_a1_s05`'s delta: 502 rows differ, mean |Δ| 105.6, and
preview − raw equals `raw × row_scale` of that table to 3.8e-6.
`wake_probe` reads the configured pack (`checkpoints().vocab_pack`), so by
the pack sha in each job's log (`7b9fce0bb57b` raw / `5f52aefce82a` preview):

| pack | runs |
|---|---|
| raw | `sent2_s24k`, `smk3k_*`, `sent_s24k_a1`, `sent_s24k_a1_s05`, `pairEN_s1500` (09-17 17:30) |
| **preview** | every other `pair_d0` arm from 17:36 (Δ0, Δ0b, lr / σ-min / flat arms, `--box_weight 1`, row blocks, α, warmup), **Δ1 53 k**, S2-smoke, S2a (three arms), the KR / ZH blocks, S2b |

Measured on Δ1: 355 of its 356 rows were already trained in the preview
pack; on them |S| 137 vs |Δ1| 79, cos(S, Δ1) 0.137.

What stands: comparisons inside one pack (S2a's ΔFM vs plain, μ 0.3 vs 0.1,
the row-block family among themselves), and every arm's score as a score of
(its pack + its delta). What does not:

- Δ1 as a ΔFM-from-scratch table — it is a 53 k ΔFM residual on the plain
  sentence table (whose own singles are 11 / 19 / 19 against Δ1's 12 / 17 /
  18). Its `trained.pt` is meaningless on the raw pack.
- "ΔFM ↔ plain 0.14" (`table_geometry`) — the same 0.137, a residual's
  cosine to the table under it.
- Δ0's "ΔFM saves no draws" — `pairEN_s1500` 10/24 ran raw, `pair0_s1500`
  20/24 ran preview, where the 12 dakuten rows were trained `kana_ext` rows.
- The S2-smoke floor finding — "untrained katakana dakuten render at floor"
  was the preview pack's trained rows (§3: raw floor 0/24).

`base.toml` is unchanged; every run below sets
`ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack` (the daemon
captures `ANIMA_*`; each log shows sha `7b9fce0bb57b`).

## 3. Δ0 on the raw pack

Data `data_synth_pair_d0` (12 dakuten rows `がぎぐげござガギグゲゴザ`, 2 000
paired single-glyph composites), argv of `synth_pair_2026_09_17.md`: σ
0.7–0.9, batch 4, cosine, `--free_residual 1e-3 --box_weight 4 --c_flat 0`,
2 seeds. Native: `が,ガ,ご,ゴ` × `en,swap`, 64 per clause; joint = both
readers ∧ en cos ≥ 0.85, tail = en cos < 0.85.

| arm (raw pack) | steps | `single` /24 | row norm | native `en` both / joint / tail | `swap` | en cos `en` | IoU `en` |
|---|---|---|---|---|---|---|---|
| floor | — | **0** | — | — | — | — | — |
| plain lr 1e-3 (`pair0_s1500_raw`) | 1 500 | 7 | 114 | 13 / 10 / 9 | 5 / 2 / 7 | 0.906 | 0.17 |
| ΔFM lr 1e-3 (`pairEN_s1500`, 09-17) | 1 500 | 10 | — | — | — | — | — |
| ΔFM lr 2e-3 (`pairEN_s1500_lr2e-3_raw`) | 1 500 | 10 | 122 | 10 / 8 / 8 | 1 / 1 / 0 | 0.926 | 0.31 |
| **ΔFM lr 5e-3** (`pairEN_s1500_lr5e-3_raw`) | 1 500 | **13** | 144 | **14 / 11 / 10** | **6 / 5 / 1** | 0.912 | 0.25 |
| ΔFM lr 2e-3 `--row_blocks 62` (`pairEN_rb62f_lr2e-3_raw`) | 744 | **0** | 116 | — | — | — | — |
| ΔFM lr 5e-3 `--row_blocks 62` (`pairEN_rb62f_lr5e-3_raw`) | 744 | 4 | 248 | — | — | — | — |
| ΔFM lr 2e-3, size jitter, min 12 px (`synth_pair_d0sz`) | 1 500 | 0 | 70 | — | — | — | — |
| ΔFM lr 2e-3, size jitter, min 16 px (`synth_pair_d0sz16`) | 1 500 | 1 | 73 | — | — | — | — |
| ΔFM lr 2e-3 `--box_weight 1` (`pairEN_s1500_lr2e-3_bw1_raw`) | 1 500 | **1** | 68 | — | — | — | — |

For scale, the same recipes on the preview pack: plain 1 500 → 20/24, ΔFM
2e-3 1 500 → 19/24 (native `en` 18 / 15 / 6, `swap` 12 / 12 / 0), row blocks
→ 16/24.

Reads:

- **The raw floor is 0/24.** The katakana-floor finding is retired.
- **On one pack ΔFM is not behind plain at 1 500 steps** (10 vs 7 singles;
  native `en` 10 vs 13, of which plain's 13 are 12 × が and ΔFM's are spread
  が 4 / ガ 2 / ご 1 / ゴ 3). Both are under-trained; 24 items price
  nothing finer. The scene axis is as before: ΔFM holds it (en cos 0.926 vs
  0.906, IoU 0.31 vs 0.17, `swap` tail 0 vs 7).
- **lr:** 1e-3 = 2e-3 = 10/24, 5e-3 = 13/24 with norm 122 → 144, native up on
  both clauses, en cos 0.926 → 0.912. No off-manifold collapse at 5e-3 (the
  3e-3 finding was plain FM). "2e-3 × 1 500 ≡ 1e-3 × 3 000" was a
  preview-pack result and is not reproduced here.
- **Row blocks do not wake a cold row.** 0/24 at 2e-3 with an ordinary norm
  (116); at 5e-3 the norm reaches 248 — 1.7× the mixed 5e-3 arm's — for
  4/24. The block grows the row fast in a direction that does not draw the
  glyph; mixed batches cancel what a single-row block absorbs. Its 16/24
  was fine-tuning on trained rows. Rider: 744 steps is half the mixed arms'
  draws and no mixed 744-step raw control exists.
- **Box weight 4 stays, on the raw pack too:** w = 1 reads 1/24 against
  w = 4's 10/24 at the same lr and steps, and the row grows half as far
  (norm 68 vs 122) — the same norm the size-jitter arms stop at. Near-misses
  are the undakuten or sibling glyph (ギ → キー, グ → タ / ダ, ゴ → ゴデス).
  Three arms now land at norm ≈ 70 and ≈ 0/24 by three routes (smaller
  glyphs ×2, no box weight): what they share is less in-box gradient per
  step, which is a travel reading, not yet a size one.
- **Glyph-size jitter** (new, below): p50 54 → 30 / 34 px halves the row's
  growth at equal steps (norm 122 → 70 / 73) and the `single` template
  reads 0 / 1 of 24. Raising the minimum 12 → 16 px moved the median 4 px —
  the same experiment twice. Unread: the natives (the template asks for a
  large glyph, so it may be the wrong ruler for small-trained rows) and a
  travel-matched arm (lr 5e-3 or more steps).
- **Correction, same day (`plan_synth4.md` R4.5): it is not travel.**
  `weighted_fm_loss` normalises by the whole-canvas weight sum, so the
  in-box share of the loss follows the box area — 6.83 % on `d0`, 2.87 % /
  3.27 % on the size builds, 1.84 % at `--box_weight 1` — and the three
  norm-≈ 70 arms are one arm. The norm peaks by step ≈ 400 and then falls
  (`d0sz` 101 → 70, `d0` 134 → 122, lr 5e-3 190 → 144): with weight decay 0
  the end norm is where the in-box gradient balances μ‖f‖², so more steps
  do not recover it. The arm that separates size from share is `d0sz` at
  `--box_weight 12` (share ≈ 7.0 %).

Native-stage gotcha found here: `--native_floor 1` needs training canvases
under the data dir's `img/` (the old scene-kept ruler) and asserts after all
renders are done; and once `floor_*.png` exist in `<arm>/native/img` the
ruler switches on by itself on every later run. The raw-pack floor renders of
`pair0_s1500_raw` were moved to `native/floor_img/`.

## 4. Code added

`--scene_size_jitter j` (`src/cli/data.py`, `src/data/synth.py`
`_render_jittered`): each item's fill is scaled by a log-uniform draw in
[j, 1], so the glyph lands between `--scene_min_glyph` and the bubble fit; a
draw under the minimum falls back to the full fit; a `--pair_ref` sibling
shares the fit; 0 = off, the old behaviour. plan_synth4 R4.5's data arm.
Builds: `data_synth_pair_d0sz` (min 12, j 0.2: short side p05 17 / p50 30 /
p90 51 px) and `data_synth_pair_d0sz16` (min 16, j 0.3: 20 / 34 / 53) against
the original 38 / 54 / 80.

## 5. What this leaves open

- The line's step-1 evidence since 09-17 evening has to be re-read per §2;
  plan_synth3 / plan_synth4 / plan_synth2 / findings / deploy_plan carry the
  caveat at the top of plan_synth3 until that re-read.
- The loop (vocab step → merge → sentence step) is recorded in plan_synth3;
  its seed question is now open again — Δ1 is only usable as preview + Δ1.
- Candidates not run: size-arm natives, size 16 px at lr 5e-3, a mixed
  744-step raw control for the row-block verdict, ΔFM 5e-3 at 3 000 steps.
