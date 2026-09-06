# CJK ext-vocab: row coverage, unreachable rows, and the symbol block

*Measured 2026-09-03 on the `synthjakozh1` training pool (cold joint JA+KO+ZH,
ext v2, `--param global`). Reproduce with
`project/cjk_aware_anima/probes/visit_stats.py`; the run is
`bench/cjk_distill/results/20260903-1239-2c-synthjakozh1-r256`. Training
refinements that follow from these numbers live in
`_archive/cjk_aware_anima/plans/plan_zh2.md` (the encoder line is frozen —
`project/cjk_aware_anima/plan.md`).*

## What the table is

The ext table appends rows to the LLM Adapter's frozen 32,128-row T5 query
embedding (`library/anima/ext_vocab.py`; design in
`project/cjk_aware_anima/findings.md` §2, §9). Rows come in blocks, in this
order, and **a block never re-indexes the blocks before it**:

| block | rows | what | init |
|---|---:|---|---|
| `qwen` | 30,951 | every Qwen token whose surface is pure CJK (spaces allowed) | `qwen_embed[id] @ W` (anchor map) |
| `char` | 28,017 | every codepoint in the CJK ranges that is *not* a clean single Qwen token and round-trips through the tokenizer | contextual Qwen state → key map (v2) |
| `sym` | 6,118 | every Qwen token whose surface is made only of *symbol-route* chars (below) | anchor map, same scale as `qwen` |
| `sym_char` | 4,472 | symbol-route chars that are byte-fragments | contextual map, same scale as `char` |

58,968 rows through 2026-09-02 (v1/v2 assets, every trained pack so far);
69,558 with the symbol block (`ext_embed` asset rebuilt 2026-09-03; the
pre-symbol asset is kept as `ext_embed_v2`).

## Reachability — rows nothing can look up

"Lookup 없음" in the structural sense is almost empty:

- **2 CJK token rows are unreachable**: fullwidth `１０` and `２０` (Qwen ids
  77150 / 80091). Qwen pre-splits digits one at a time, so the two-digit
  token is never emitted. Harmless (a row nobody addresses).
- **0 corpus characters inside the routed ranges lack a row.** Every CJK char
  in the four staged corpora resolves to a `qwen` or `char` row.
- In the symbol block, 75 Arabic tokens carrying tashkeel (e.g. `مْ`, `يّ`)
  do not re-tokenize to themselves in isolation (Qwen splits the diacritic
  off when the token is alone); they can still be emitted inside a word.
  Kaomoji use bare Arabic letters (`٩` `۶` `و`), which are fine.

Everything else is reachable. The problem is observation, not lookup.

## Visit statistics (training pool, 4 caches, 200,744 pairs)

Visited rows: **9,831 / 58,968 (16.7 %)** — identical to the run's
`rows_visited`. Per block × script × visit band (row counts):

| block | script | total | 0 | 1–4 | 5–49 | 50–499 | 500+ |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen | han | 24,953 | 18,949 | 904 | 2,051 | 1,761 | 1,288 |
| qwen | hangul | 3,564 | 1,538 | 285 | 535 | 503 | 703 |
| qwen | kana | 1,525 | 630 | 85 | 252 | 223 | 335 |
| qwen | han+kana | 544 | 346 | 36 | 67 | 55 | 40 |
| qwen | han Ext-A | 108 | 108 | 0 | 0 | 0 | 0 |
| qwen | fullwidth/punct | 244 | 222 | 0 | 8 | 9 | 5 |
| char | han | 12,491 | 11,816 | 144 | 253 | 200 | 78 |
| char | han Ext-A | 6,484 | 6,481 | 0 | 2 | 1 | 0 |
| char | hangul | 8,933 | 8,929 | 1 | 3 | 0 | 0 |
| char | kana / fullwidth | 108 | 108 | 0 | 0 | 0 | 0 |

Per cache (distinct rows / total visits): `cache_tags` 3,456 / 5.39 M,
`cache_ko` 1,609 / 6.54 M, `cache_desc_ko` 1,946 / 0.24 M, `cache_zh`
6,329 / 5.82 M. Among visited rows the median visit count is 62, p90 3,246,
p99 32,853, max 425,147 (`、`); the top-100 rows carry 34.8 % of all visits.
Top rows: `、` `리` `的` `の` `乳` `이` `스` `머` `人` `기`.

### Why

- **`char` rows are ~97.6 % unvisited by construction.** `char_row_surfaces`
  enumerates the *entire* Unicode CJK ranges: all 6,484 Ext-A ideographs,
  8,929 Hangul syllables/jamo that Qwen has no single token for (i.e. the
  composable syllables real Korean never uses), and 11,816 rare Han. A tag
  corpus cannot visit them. This is the "zero-shot tail" the global map is
  meant to cover; whether it does is unmeasured (see plan_zh2 U0/U4).
- **`qwen` rows are 70 % unvisited because of Qwen's vocabulary, not ours.**
  Qwen's CJK tokens are general-web Chinese (`想起来` `培训班` `第五届`
  `毫无疑` `クリック`); a glossary-driven tag corpus over 16,883 captions
  shares only the tag vocabulary with it. 18,949 / 24,953 Han tokens, 1,538 /
  3,564 Hangul, 630 / 1,525 kana never occur.
- **KO has many visits on few rows** (6.5 M visits, 1,609 rows): Qwen's Hangul
  tokens are mostly syllables, so Korean spells its tags from a small set.
- **Visit count does not gate a row in this run.** `--param global` has
  `n_tunable_rows = 0`; the only trainable objects are the shared rank-256
  map, the per-dim diagonal and the scalar gain, and every row — visited or
  not — goes through them (provenance is `mapped` for all 58,968). The
  trained pack has `global_gain = 0.299` and `diag rms = 0.749`: an
  unvisited row is therefore not "left alone", it is scaled down to roughly
  a third of its init norm along with everything else. The holdout draws
  from the same corpus distribution, so nothing in the eval sees the
  unvisited 83 %.

## `<unk>` audit — the symbol gap

The staged caches still contain `<unk>` on the student side: `cache_tags`
1,950, `cache_ko` 1,347, `cache_desc_ko` 141, `cache_zh` 10,775 tokens.
Re-encoding the source text and locating each `<unk>` shows **none of them
is an ext lookup failure**. All are non-CJK symbols that the stock T5 spiece
path cannot spell:

| fragment | JA | KO | ZH | note |
|---|---:|---:|---:|---|
| `^^^` / `^` | 317 | 241 | 427 | danbooru emoticon tags (`^^^`, `^ ^`) |
| `<` | 110 | 71 | 122 | `:<`, `<<<` |
| `·` | 40 | – | 1,252 | zh name separator (萊莎琳·斯托特) |
| `~` | 16 | – | 872 | zh title tilde (…告白~天才们的恋爱头脑战~) |
| `×` `☆` `\` `♪` `♀` `♂` `⚡` | ~100 | ~20 | ~330 | |

(counts over the first 40 shards of each cache). T5 also folds a whole run
into **one** `<unk>` — `^^^`, `^`, `☆` and `\` were the same token — and the
EN caption on the teacher side gets the same `<unk>`, so these tags carried
no identity on either side of the distillation. Kaomoji (`(˘ω˘)`,
`٩(◕‿◕｡)۶`, `˗ˏˋ ˎˊ˗`) add a tail of IPA / combining / Greek / Arabic / Yi
characters.

### Fix: symbol routing, carried by the pack (2026-09-03)

Same mechanism as CJK, no model change: the chars are routed to the Qwen
tokenizer and get ext rows.

- **Which chars route is decided at build time and written into the pack
  json** (`mapping["route"] = {ranges, chars}`; `Route.from_mapping`).
  `symbol_route_chars` scans `SYMBOL_CANDIDATE_RANGES` (ASCII/Latin-1
  symbols, IPA + combining marks, Greek, Arabic, phonetic extensions,
  U+2000–2BFF, supplemental punctuation, enclosed/compat CJK, Yi radicals,
  variation selectors, emoji) and keeps a char iff T5 emits `<unk>` for it
  and Qwen round-trips it: **6,960 chars**. Letters T5 can spell (é, ß …)
  are excluded by the test, so EN/European prompts are untouched.
- Rows are appended after the CJK blocks (`sym` 6,118 tokens, `sym_char`
  4,472 chars; `mapping["sym_rows"] = [58968, 69558]`). Every pre-existing row
  id, every staged cache and every trained pack stays valid. Norm scale and
  the contextual standardization are computed on the CJK rows only and
  *applied* to the symbol rows (`build_ext.py`), so the CJK rows are what a
  symbol-free build produces.
- `HybridT5Encoder.from_mapping` merges `sym`/`sym_char` into its two
  lookups; `segment_runs(text, route)` and `encoder.routes(text)` use the
  pack's rule. A pack without `route` routes the legacy CJK ranges — bit
  identical to before. Pure-EN prompts stay bit-identical (test).
- Result on a 1/20 sample of the three tag corpora (10,184 pairs): **0
  residual `<unk>`** (was ~1,400 per 10k pairs).

Compatibility matrix:

| node / trainer code | pack | behaviour |
|---|---|---|
| new (`Route`-aware) | new (`route` present) | symbols route to rows |
| new | old (no `route`) | CJK only, identical to before |
| old (3.9.0 vendor tree) | new | no crash (row-count check passes); symbols stay `<unk>`, symbol rows unused |

The ComfyUI Anima Adapter node picks the rule up in 3.9.1 (`vocab_pack.py`
uses `encoder.routes`; the vendored `ext_vocab.py` is re-synced with `make
vendor-sync`).

### What the symbol block does *not* solve

Rows give the symbols identity; nothing yet gives them meaning. The
distillation teacher is the EN caption through stock spiece, where `^^^` is
still `<unk>` — a span loss on it teaches the row to imitate `<unk>`. The
existing caches were staged before the block exists, so a distill on them
visits 0 symbol rows (see the table: `sym` 6,118 / 0 visited). Both are plan
items (`plan_zh2.md` U5: `tags_sym` register with a wiki-description
teacher; re-stage the caches).

## Quote partition: an isotropic mirror for quoted spans (2026-09-05, DiT line D1)

The DiT-side line (`project/cjk_aware_anima_dit/plan.md`, principle 1–2, 8)
treats ext rows as content-free *addresses*. A pack may now carry a second
block that is exactly that:

- **`mapping["iso"]`** = `{recipe: gauss_rows_v1, seed, dim, norm, rows: [start, end)}`.
  `ext_vocab.iso_block(seed, n_rows, dim, norm)` regenerates it byte-equal on
  any machine (NumPy legacy `RandomState` stream, float64 row-major draw,
  per-row normalise, one float32 cast — no BLAS, no torch RNG). The block is a
  **row-for-row mirror** of the trained blocks at offset `start` (= the trained
  row count), so the same `qwen`/`char`/`sym` lookups serve it. A pack may
  ship the rows or only the record (`make_random_pack.py --mode iso-partition
  [--no-iso-rows]`); `load_ext_assets` / the node's `load_vocab_pack` call
  `materialize_iso` either way.
- **`route.quotes`** = `[["「","」"],["『","』"],["\"","\""]]`. `Route.quote_spans`
  is one non-greedy, non-nesting regex over the caption (a stray opener
  matches nothing). The span rule runs *before* `segment_runs`: the spiece
  side is tokenised exactly as without the partition (EN bit-identical by
  construction), and only routed runs are cut at the delimiters
  (`HybridT5Encoder.encode_cjk_run`). Quoted content → `T5_TABLE_SIZE +
  start + row`, no minted-word / C-fallback substitutions; bare CJK → the
  trained row as before; delimiters keep their old path (`「」` → trained
  row, `"` → spiece). Both halves must be present (`encoder.quote_routing`);
  a pack with neither, or only one, encodes bit-identically to before.
- **`ext_vocab.pack_digest(table, mapping)`** = sha256 over the materialised
  float32 table bytes + the id/route keys (`qwen char sym sym_char word
  word_sub route iso`; `training`/`stats` excluded). `train.py --ext_pack
  <prefix>` stamps it as `ss_ext_pack_sha` (+ `ss_ext_pack` name) on the
  LoRA (`run_unmask_r2.py` passes it); `load_dit_model` warns when a stamped
  LoRA is loaded with no pack; the ComfyUI Adapter node (3.10.0) compares it
  with the loaded pack's digest in either node order and warns on mismatch.
- The caption grammar is quote-aware since anime_tools `efb235c`
  (`position_clauses.quoted_spans`): a comma or `. On the` inside an open
  pair is content, `compose_caption` round-trips. `cache_te_ext._quote_safe`
  therefore no longer rewrites commas (only an inner `"`).

Compatibility matrix (extends the one above):

| node / trainer code | pack | behaviour |
|---|---|---|
| new (`iso`-aware) | partitioned (`iso` + `route.quotes`) | quoted spans → mirror, bare CJK → trained rows |
| new | seed-only partitioned (no iso rows in the safetensors) | block regenerated at load; same digest |
| new | old (no `iso`) | identical to before |
| old (≤ 3.9.1 vendor tree) | partitioned with rows | row-count check passes; quoted spans hit *trained* rows (no quote rule) — the digest check does not exist there either |
| old | seed-only partitioned | refused (row-count mismatch) |

First built pack: `output/ckpt/cjk_vocab_pack_synthjakozh1sym_r256_isoq`
(C9's 69,558 rows + mirror at rows 69,558–139,116, seed 0, norm 212.165 =
the native T5 mean row norm; mirror PR 1009, pair cos 0.0002; sha256
`2cf81cbc…`). Tests: `tests/test_ext_vocab_iso.py`.

## Build caveat: the anchor fit is not bit-reproducible across runs

Three builds of identical inputs on the same GPU gave anchor-map held-out
cos 0.7090 / 0.7100 / 0.7147 (v2 asset: 0.7128); in-process on CPU the fit is
bit-deterministic (seeded holdout, two consecutive fits equal). Shared rows
between two builds agree to mean cos 0.991 (min 0.969). A pre-existing
property — the ridge solve at `ridge=1e-2` amplifies BLAS reduction-order
noise — not caused by the symbol block. Consequence: a "strict superset" is
guaranteed by construction (same code path, CJK-only statistics), not by
byte comparison across builds; pinning the fit (store `W`, or fit on CPU
single-thread) is a cheap plan item.
