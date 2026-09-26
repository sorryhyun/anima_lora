# floor_score — the seed table's floor on the product rulers

The floor of record for `rows_step1_0921_merged` (2 274 rows). **Every
"vs seed" read on these rulers uses this table** (same prompts, seeds 0–1);
a new floor measurement replaces the row it changes, with its job id.

## Method

Floor arm = the seed rows' own dir — the **whole** seed table, all 2 274
rows, no inventory filter ever (a filtered floor renders frozen singles as
raw-pack rows, below any honest floor). Job
`20260925-220101-328726`, `ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack`.

Metrics: **official** = `hit_sfx ∧ hit_vl`, **loose** = `hit_sfx ∨ hit_vl`,
**contained** = the vocab string appears inside any box read (sfx or VL).

## The floor

| ruler | floor |
|---|---|
| `sent` はい (16) | **3** (loose 4) |
| `sent` おしい (16) | **5** (loose 5) |
| `sent` やったネ (16) | **0** |
| `sent` ちょっと来い (16) | **0** |
| `sent` こんにちは (16) | **0** |
| `target` はい verbatim (8) | **0** |
| `target` こんにちは verbatim (6) | **0** |
| `word` exact (18 × 2) | **6/36** |
| `word` contained (18 × 2) | **11/36** |
| `en` (24) | **24/24** (ceiling) |

Notes on reading it:

- **The seed's own `word` misses already carry the doubling shape**
  (すごい → ずすぎい, あと → あどと, こう → こえう) — doubling is a seed
  property, not something a run introduces.
- やったネ / ちょっと来い / こんにちは and both `target` cells have a
  **zero floor**: a trained 0 there is "no gain", never damage; any hit is
  pure gain (small n).
- `en` is at ceiling, so it detects damage only.

Piece-alone native floors (8 pieces, en clause, jobs `…-1829d3`):
lenient った 0, です 3, すごい 4, メン 8, しい 13 of 16; official totals
en 5, swap 0. Detail: `reports/piece_2026_09_25.md`.

## Provenance

`sent` / `target` strings are the fixed acceptance rulers
(`product_criteria.md` Axis 1). The `word` / `en` rows were drawn from
run0925_300f's dev vocabs (18 / 24 of its 300) — a run with a different dev
set re-measures those two rows only.

Outputs live **flat in the seed table's arm dir**
(`output/cjk_anima_scale/rows_step1_0921_merged/`: `eval_reads.json`,
`native_sent/`, `target/`, `native_piece/`, sheets), per the line rule that
seed-side reads of record live with the seed, not under a run.

**Since 2026-09-26 that dir is every run's floor** (`paths.floor_dir()`,
`eval.ensure_floor`): a run's eval renders only the keys (string × clause,
or group × string) the cache lacks and folds them into the same read files,
so the files grow past the rows above (`native/` あ / い, `native_spell/`,
more `word` / `piece` / `sent` keys). The table above stays the floor of
record for its rows; the per-run `floor/` copies before that date were
folded in by copy (`eval.import_floor`), the cache's own keys winning.
