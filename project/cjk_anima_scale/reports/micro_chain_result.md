# micro_chain_result — the chain at 30 / 30 / 30 on 24 warm rows (2026-09-24)

The first run of the band chain (`stage0709 → stage0507 → stage0305`) as
built: the production configs, the run file `configs/runs/run0923_micro.toml`
(24 rows, all warm in `rows_step1_0921_merged`), 30 steps per row per stage.
The plan it answers (`plan_micro_chain.md`, deleted with this record — git
`aab7e86c` has the last copy) asked one question: **does the chain move warm
rows the right way at the production budget** — each stage buys what it is
for and keeps what the previous stage bought. Two chains ran: the configs as
they were (μ 0 / 0.01 / 0.01) and, after that one failed its regression rule,
μ = 0.1 on every stage. Every job went through the daemon on the raw pack
(`ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack`, sha
`7b9fce0b…`); the ledger has the seven job ids.

## 1. Rows and stages

- 8 kana `あ い お な ア ナ ラ ル`, 8 kanji `人 日 口 女` (simple) +
  `精 聞 動 願` (dense), 8 pieces `それを はじ やはり すご メン アン プロ ファ`
  — 24 units, one Qwen token each; the trainer's `rows warm start: 25/25`
  (the 24 + one symbol row the scene captions touch).
- Per stage, by the law: `stage0709` draws singles only (scene 48–53 px +
  grid 62–210 px; pieces have no window), `stage0507` both (pieces 0.715 /
  singles 0.285 after `scene_short` dropped — no short lines), `stage0305`
  pieces only (`scene_piece` / `grid_string` 0.5 / 0.5 after `scene_sentence`
  dropped). 4 000 items a stage, 750 steps (25 rows × 30) × batch 4.
- Rulers: `single` 16 × 2 seeds, `word` 8 × 2, `en` 12 × 2 (exact, sfx
  reader), native on `あ い 日 願` (16 renders each, `swap` clause),
  `cf_sense --cf_rows piece`, and `regress.json` on the previous stages'
  groups. The **seed baseline** (`--seed_only`, § 4) is the same rulers on
  the untrained-here seed table.

## 2. The reads

Exact hits; native = the `swap` clause's hits of 16.

| read | seed | 0709 μ0 | 0507 μ0.01 | 0709 μ0.1 | 0507 μ0.1 | 0305 μ0.1 |
|---|---|---|---|---|---|---|
| single (/32) | 24 | 27 | **20** | 27 | 26 | 26 |
| word (/16) | 2 | 2 | 5 | 2 | 3 | 5 |
| en (/24) | 24 | 24 | 24 | 24 | 24 | 24 |
| native あ / い / 日 / 願 | 15 / 8 / 5 / 6 | 15 / 9 / 4 / 7 | 9 / 2 / 4 / 5 | 16 / 9 / 4 / 9 | 14 / 6 / 4 / 7 | 14 / 6 / 4 / 7 |
| dense kanji 精聞動願 (/8) | 4 | 6 | 3 | 5 | 5 | 5 |
| warm_cos (drift) | – | 0.976 (0.18) | 0.952 (0.31) | 0.9985 (0.045) | 0.985 (0.13) | 0.993 (0.07) |

Per row, μ = 0.1 chain end vs seed: singles — `い` 1 → 2, `お` 1 → 1,
`動` 1 → 1, `願` 1 → 2, `口` `聞` 0 → 0, the rest 2 → 2; pieces —
`すご` `はじ` `アン` `メン` 0 → 1, `ファ` 1 → 1, `プロ` 1 → 0, `それを` `やはり`
0 → 0.

## 3. Verdicts, rule by rule

| rule (plan § 5) | μ 0 / 0.01 | μ 0.1 |
|---|---|---|
| 0709 `word` == seed, exactly (determinism) | ✓ 2 / 16, same two rows — after the table fix (§ 4); before it 0 / 16 | ✓ |
| 0709 `single` / native ≥ seed, `warm_cos` ≥ 0.9 | ✓ 27, cos 0.976 | ✓ 27, cos 0.9985 |
| 0507 `word` > seed by ≥ 2 | ✓ +3 | **✗ +1** — then +2 more at 0305 |
| 0507 regress `stage0709/single` within 1 | **✗ 27 → 20**; native あ 15 → 9, い 9 → 2 — chain stopped here | ✓ 27 → 26 |
| 0305 regress `single` == 0507 (no draws) | – | ✓ 26 / 32, same rows; native identical |
| 0305 `word` ≥ 0507, cf_sense peak in 0.3–0.5 | – | `word` ✓ 3 → 5; **cf_sense cannot read it** (§ 5) |
| 0709 dense kanji ≥ seed, 願 native up | ✓ 6 / 8, 願 6 → 7 | ✓ 5 / 8, 願 6 → 9 |
| chain end vs seed: singles ≥, pieces > | – | ✓ 24 → 26, 2 → 5 |

**What it says.** At μ ≤ 0.01 the 24–32 px single draw at 0.5–0.7 undoes
0709's identity (the plan's reading: the anchor is too light or the
small-single recipe does not belong in 0507). μ = 0.1 holds the singles
(27 → 26, native あ/い down 2–3 of 16) but ties the pieces to the same
anchor: 0507 buys +1 instead of +3, and the chain ends at +3 / 16 on pieces.
One anchor is buying single preservation and piece movement at once. The
chain-end rule passes, so μ = 0.1 is the setting (`configs/stage*.toml`);
the piece read is flat-ish, which the plan calls a **budget statement** —
the next cell is the same chain at 60 / 60 / 60, not a band change. Two
seeds × 8 pieces is a 16-row ruler; +3 is inside what one row flip per seed
can do.

## 4. What the run changed in the tools

1. **`--seed_only`** on `eval` (`cjk_scale/eval.py::seed_wrapper`): the run's
   `seed_table` as a rows-arm `trained.pt` with a synthetic `args`, under
   `rows_<stage>_<run>_seed/` (the probe's `--arm_tag seed`, so the
   stage's data dir is read unchanged), restricted to the rows the stage's
   `words.json` names (24 of 2 274) so `cf_sense` draws from the rows a
   trained table carries; no `regress`.
2. **The stage table carries the run's inventory.** The first 0709 run built
   its table from the rows the captions touch, so the 8 piece rows (no window
   at 0.7–0.9) were not in `trained.pt` at all: `word` read 0 / 16 because
   the eval fell back to the raw pack's rows, and 0507 would have warmed the
   pieces cold. Now `table = touched ∪ inventory` (`train.py::inventory_ext`,
   the units' ext ids off the eval strings' captions) and the norm pull
   applies to touched rows only (`RowTable(touched=…)`, `weight_decay` 0), so
   an untouched row is exact — the rerun read `word` 2 / 16 on the same two
   rows, and 0305 left the 16 singles at 0507's values row for row. Steps are
   `steps_per_row × table rows` (25 here, 750 a stage).
3. `init_anchor` 0 / 0.01 / 0.01 → **0.1** on all three stages (the config
   comments carry the read).

Preserved tables: `rows_stage0709_run0923_micro_v0_17rows/` (the
17-row table), `_mu0/`, `rows_stage0507_run0923_micro_mu001/`; the
unsuffixed dirs are the μ = 0.1 chain.

## 5. Open

- **The piece `cf_sense` ruler renders at 55–195 px**, so it peaks at
  0.7–0.8 for every table and moves 0 at 0.35 / 0.5. It cannot say whether
  the 12–24 px band trained; the only 0305 read is `word` (+2). A small-px
  `cf_sense` (render at the stage's px) is what would read 0305.
- Native at 0507 loses 2–3 of 16 on `あ` / `い` under μ = 0.1 and does not
  recover at 0305 (no single draws there). Not in the rules; noted.
- Wall: ≈ 21–25 min a stage (data + 750 steps + the three rulers), 15 min the
  seed baseline.
