# floor_score — the seed table's floor on the product rulers

Two seeds, two floors. **`seed_retrain_0930`** (2026-09-30, § New seed
below) is `paths.SEED_ROWS`: every later run reads against it.
**`rows_step1_0921_merged`** (§ Old seed) is `paths.SEED_ROWS_0921`, the
floor of record the retrain itself was read against (the retrain's
experiments call `paths.pin_old_seed()`). Same prompts, seeds 0–1; a new
floor measurement replaces the row it changes, with its job id.

## New seed — `seed_retrain_0930` (2026-09-30)

`retrain_kanji_b4`'s merged rows (the old seed + `retrain_kana` + kanji
b1–b4, 2 683 rows; md5 `af99aa93…`), copied whole into
`output/cjk_anima_scale/seed_retrain_0930/` (`seed.json`). Read **routed**
(the rows trained routed; the bake ships routing on), so the cache is
`seed_retrain_0930/routed/`. Rendered by `scale.py <run> floor` (the floor
arm alone: the run's rulers + `eval.ACCEPT_READ` on `sent`):
`retrain_kana` job `20260930-100904-25d9af`, `retrain_kanji_b4` job
`20260930-100905-b7c3c6`; totals in `routed/floor_<run>.json`.

Paired = same prompt × seed against the old seed's cached render (new-only /
old-only, McNemar). The old `sent` acceptance renders are unrouted (the old
floor of record; old こんにちは is its piece row), the kana and kanji words'
are routed (`native_route/` and the C3 caches).

| ruler | new | old | paired |
|---|---|---|---|
| `sent` はい (16) | **7** | 3 | +5 / −1 |
| `sent` おしい (16) | **6** | 5 | +3 / −2 |
| `sent` やったネ (16) | **9** | 0 | +9 / −0 |
| `sent` ちょっと来い (16) | **6** | 0 | +6 / −0 |
| `sent` こんにちは (16) | **9** | 0 | +9 / −0 |
| acceptance total (80) | **37** | 8 | +32 / −3 |
| `target` はい verbatim (8) | **5** | 0 | |
| `target` こんにちは verbatim (6) | **0** | 0 | |
| `sent` kana words たいせつ かなしい かんがえ たすけて こうえん てつだう ことば (7 × 16) | **4 4 3 5 1 3 5 = 25** | 0 | +25 / −0 |
| `sent` なにしてる テレビ カメラ パソコン アイドル (16 each) | 3 2 7 5 8 | – | no old cache |
| `sent` C3 words 山田太郎 小山田 日本人 大丈夫 何時間 愛してる (6 × 16) | official **0 2 3 1 1 0 = 7**, ≤ 1 edit **45** / 96 | 0, ≤ 1 edit 3 | +7 / −0; ≤ 1 edit +44 / −2 (p 3e-11) |
| `native` あ / い, en + swap (64) | **33** (loose 47) | 46 (routed) | +6 / −19 (p 0.015) |
| `single` kana alone, `retrain_kana`'s 24 (384) | 112 (loose 207, contained 291) | – | no old cache |
| `single` kanji alone, b4's 24 (384) | 46 (loose 93, contained 145) | – | no old cache |
| `eval` single, kana 18 × 2 / b4 kanji 18 × 2 | 22 / 36, 11 / 36 | – | |
| `eval` en (24) | **24/24** | 24/24 | ceiling |

- **The acceptance strings move 8 → 37 / 80, every string the same way**;
  every paired `sent` cell together is +57 / −3 (p 6e-14).
- The C3 words sit where `retrain_kanji_b1` put them (≤ 1 edit 45 vs 47,
  +3 / −5): the chain did not erode them. Exact reads stay low (7 / 96).
- **The one drop is lone あ** (en 14 → 8, swap 14 → 11; い en 10 → 11,
  swap 8 → 3, loose 10 → 11) on the native control, unread by eye.
- `en` at ceiling: no damage.

Baked 2026-09-30 with routing on:
`models/vocab_packs/anima_cjk_vocab_pack_seed_retrain_0930/`
(`bake_vocab_pack.py --glyph_route`, base the raw pack with `fold`,
baked sha `eee7fec51835…`).

## Old seed — `rows_step1_0921_merged`

The floor of record for `rows_step1_0921_merged` (2 274 rows). Every "vs
seed" read before 2026-09-30 uses this table.

### Method

Floor arm = the seed rows' own dir — the **whole** seed table, all 2 274
rows, no inventory filter ever (a filtered floor renders frozen singles as
raw-pack rows, below any honest floor). Job
`20260925-220101-328726`, `ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack`.

Metrics: **official** = `hit_sfx ∧ hit_vl`, **loose** = `hit_sfx ∨ hit_vl`,
**contained** = the vocab string appears inside any box read (sfx or VL).

### The floor

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

### Provenance

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
