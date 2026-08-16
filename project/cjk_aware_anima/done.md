# CJK-aware Anima — done

What is finished. Details and measured numbers live with the code that
produced them — this file only says *what exists* and *where*.

*Line home: [`motivation.md`](motivation.md) (why, incl. directions already
ruled out) · [`plan.md`](plan.md) (what remains) ·
[`report_0816_phase2.md`](report_0816_phase2.md) (Phase 2 measured verdicts —
2b unit gates + 2c first pass).*

## Phase 0 — probe (2026-08-15)

- [x] Language/routing arm sweep — `bench/cjk_adapter/run_bench.py`,
      results `results/20260815-1836/result.json` (+ rendered grids).
      Established `t5en` as the teacher and closed romanization and reverse
      routing.
- [x] `process_escape` mojibake fix (c8cf3ce2) — native-CJK prompt strings
      survive the CLI/config path.

## Phase 1 — zero-shot ext vocab

- [x] Ext table build — `bench/cjk_adapter/build_ext.py` →
      `assets/ext_embed.{safetensors,json}`.
- [x] Hybrid encoder (`ext_vocab.HybridT5Encoder`) — CJK spans → ext ids,
      bit-identical on pure EN. Promote to `library/anima/` in Phase 3.
- [x] Acceptance harness — `run_bench.py --ext`. Phase 2c reuses it unchanged,
      swapping the sidecar.
- [x] Zero-shot arm measured (same result dir) — anchor mapping alone is
      insufficient, so Phase 2 training is required.

## Phase 2a — data build

Builders and their measured results: [`datasets/README.md`](datasets/README.md).

- [x] `datasets/wikidata_lexicon.py` — EN↔JA/KO/ZH proper-noun lexicon (CC0).
- [x] `datasets/lovehina.py` — native-register JA eval set (MIT). Held out,
      never trained on.
- [x] `datasets/mt.py` — `MTEngine` (Hy-MT2, Apache-2.0) + `--probe` idiom
      benchmark + `--smoke`; per-batch resume cache, `--gpu-budget`.
- [x] `datasets/tag_glossary.py` — EN→JA tag glossary from the Danbooru wiki,
      lexicon and MT → `assets/tag_glossary_ja.json` + `_review.md`.
      `--reselect` re-derives choices with no GPU.
- [x] `datasets/tag_overrides.json` — hand-pinned wording (committed; beats
      every automatic source on rebuild).
- [x] `datasets/build_pairs.py` → `post_image_dataset/cjk_distill/`
      (`pairs.jsonl`, `coverage.json`, `spotcheck.md`).
- [x] `tests/test_cjk_glossary.py` — 19 invariants (script detection, alt-pool,
      cache resume).
- [x] Gates passed: proper-noun substitution, kana saturation, occurrence
      coverage (`coverage.json`).

Still open in 2a — see [`plan.md`](plan.md#phases--gates): user sign-off on
`tag_glossary_review.md`, and the `natural` prose register (implemented, not
run).

## Phase 2b — loop + gates (2026-08-15/16)

Measured verdicts: [`report_0816_phase2.md`](report_0816_phase2.md).

- [x] `scripts/distill_cjk/` — corpus cache, ext-table ladder, four objectives,
      the training loop (`make exp-cjk-cache` → `exp-distill-cjk`). The
      **one-off gate drivers live with the line** in [`gates/`](gates/), not in
      `scripts/`: `scripts/distill_cjk/` holds only what a later phase reuses.
- [x] `tests/test_cjk_distill.py` — G1 EN bit-exactness.
- [x] `scripts/distill_cjk/build_query_bank.py` — real cross-attn probe queries
      (DiT forwards at 2–3 σ) → `bench/cjk_distill/assets/query_bank.safetensors`.
      `attn_bank.build_bank` refuses random directions without an explicit flag.
- [x] `attn_bank.fit_centers` — the readout's common offset, projected out.
- [x] `load_pairs` splits **by image**, not by pair (no sibling leakage; the
      near metric is populated).
- [x] Gates G0b / G1 / G0 / G2 all passed. Settled: `param=global`, `loss=span`.
- [x] `gates/g34.py` — the closing gates G3 (teacher ceiling per
      register) + G4 (corpus health + trust ablation), `make exp-cjk-gates`.
      `data.CachedPairs.apply_trust` re-derives span weights from `via` on load,
      so a trust arm is not a cache rebuild; `--eval_limit 0` scores the whole
      holdout, which the per-register decomposition needs.
- [x] G3 passed: the `tags` teacher ceiling is **0.823**, so the 2c `≥0.6` gate
      stands (73% of teacher). Found that `recovery_attn` is a **mix
      statistic** — the readout floor is register-dependent by 100× — and
      confirmed D6's eval-only demotion quantitatively (0.015/0.021 of
      addressable gap in readout space).
- [x] G4 passed: JA/EN token ratio 0.96–1.20 (no length pathology),
      `mt_unverified` = 36.6% of span tokens, 2,656/58,968 rows visited. The
      trust ablation is flat — `all` ≡ `provenance` to 3 dp and `verified_only`
      is *worse* on every column, so at 10⁴ pairs noisy supervision beats none.
      It does **not** close the 2a glossary sign-off (aggregate loss cannot see
      a single row bound to the wrong meaning).

**Phase 2b is closed.** Gates G0b / G0 / G1 / G2 / G3 / G4 all green; the
handoff to 2c is [`report_0816_phase2.md`](report_0816_phase2.md#next).

## Phase 2c — first pass (2026-08-16)

Measured verdicts:
[`report_0816_phase2.md`](report_0816_phase2.md#phase-2c--first-pass-2026-08-16).

- [x] Trained packs — `output/ckpt/cjk_vocab_pack_global{,_row}.{safetensors,json}`
      (`param=global` is the keeper; `_row` indistinguishable end-to-end).
      Envelopes `bench/cjk_distill/results/20260816-1450-2c-global/`, `…-1511…`.
- [x] `gates/g5.py` — the flat-gate oracle (span_perfect / span_plus /
      ref_remap / prefix_bound). Verdict: flat `cos_vs_en` demoted from gate to
      control. Envelope `…-1532-g5-flat-ceiling/`.
- [x] Flat probes (`…-1533-2c-probe-flatmax/`, `…-1557-2c-probe-spanflat/`) —
      no flat term ships; buying flat points costs readout-space alignment.
- [x] Rendered eval — `assets/ja_eval_prompts.json` (20 prompts, five
      registers) through `run_bench.py --ext --prompts`; gate + grid per pack
      under `bench/cjk_adapter/results/20260816-1618-2c-gate-global/`,
      `…-1619-2c-grid-global/`, `…-1634-2c-gate-globalrow/`,
      `…-1634-2c-grid-globalrow/`.
- [x] `gates/coverage.py` — CPU-only per-prompt span-visit diagnostic; ties
      each render failure to its 0–40-visit content tokens and sizes the D1
      widening (the next work item — see [`plan.md`](plan.md#corpus--where-the-next-win-is)).

## Phase 2a — D2 (2026-08-16)

- [x] `datasets/commentary.py` — native-JA danbooru artist commentary via the
      gelcrawl route (`curl_cffi` + SpoofDPI). 434,800 raw →
      `assets/commentary_ja.jsonl`: **73,015** unique JA records (5.2 M chars,
      4,775 unique kanji), 3,347 with a human EN translation. Per-line promo
      stripping; zh/ko kept in the raw cache for when ja-only scoping lifts.
- [x] `commentary.py --mt` — the JA→EN teacher side (Hy-MT2-7B, greedy).
      Names pinned JA→EN off the D5 lexicon (fires on ~13% of records; the 1.8B
      renders 十時愛梨 "Jūji Aira" and hallucinates 虹ヶ咲 → "Neko no Hikari",
      which is why the 7B is worth 4× the wall clock). Length-bucketed batching
      with per-bucket `max_new_tokens` **and** batch size — batch 32 in the
      ≤128-char bucket OOMs the 7B at a 13 GiB weight budget. Resumable through
      `mt.py`'s prompt-keyed cache; `--from-cache` harvests a stopped pass on
      CPU. Partial run: **5,721 of 69,668** translated, 3.6% rejected by the
      output gate (`untranslated_cjk` / `empty` / `runaway`).
- [x] `build_pairs.py --commentary` — the D2 `commentary` register (9,068 pairs
      = 5,721 MT + 3,347 human). Span-less by construction. Measured in
      [`report_0816_phase2.md`](report_0816_phase2.md#d2--what-the-commentary-corpus-buys-2026-08-16).
- [x] `datasets/manga_text.py` — danbooru text-detection corpus, piloted and
      **rejected as corpus material** (register duplicates D4; OCR noise arrives
      MT-laundered and undetectable). Kept for geometry: mask validation and
      Phase 4. See [`datasets/README.md`](datasets/README.md).

## Phase 2c — D1-wide (2026-08-16)

Measured verdicts:
[`report_0816_phase2.md`](report_0816_phase2.md#d1-wide--the-gelcrawl-widening-measured-2026-08-16).

- [x] `build_pairs.py` / `tag_glossary.py` take **multiple caption roots**
      (`--captions` curated / `--raw-captions` raw / `--tag-rules`). Raw
      crawler roots are normalized through gelcrawl's `tag_rules.yaml` via
      `library.captioning.tag_rules` — the same rules that produced
      `image_dataset`, so the roots agree on the rating band. Dedup key is the
      artist-relative path, not the bare stem
      ([[project_booru_id_space_collision]]); first root wins.
- [x] Corpus rebuilt at width — **3,008 → 16,128 captions**, 18,990 → **45,230
      pairs**, 500+ visit band **381 → 756**, 4.3× total visits. This is the
      current `post_image_dataset/cjk_distill/`.
- [x] Measured that widening buys **visits, not vocabulary** (rows visited flat
      at ~6,400) and therefore does **not** move any `v=0` token — the render
      grid's zero-visit failures are a glossary **wording** defect (119 tags
      where a native-kanji candidate lost to a katakana loanword), which is a
      human review axis, not an automatic fix.
- [x] Established that a glossary rebuild **requires `--mt`** — the CPU-only
      path was tried and reverted (drops the back-translation verification
      layer, regresses 1,991 wordings).

## Phase 2c — D1-pairs tail fill (2026-08-16)

- [x] `datasets/tag_pairs.py` — `p1atdev/danbooru-ja-tag-pair-20241015` (CC0,
      151,431 rows) as a **fill-only** source for tags the glossary leaves
      unresolved. **5,248 tags filled; unmapped segments 42,530 → 13,714**
      (−68%), ext rows visited 6,424 → 6,538, 500+ band 756 → 778. Fills only —
      a resolved wording is never re-opened on a CPU pass (the rebuild failure
      in [`datasets/README.md`](datasets/README.md#do-not-rebuild-the-glossary-without---mt));
      the source's own Chinese/latin noise is re-guarded here, not trusted.
      New provenance `via: tagpair` at trust 0.6 — declared in all
      `TRUST_POLICIES`, because `apply_trust` defaults an unknown `via` to
      **1.0**.
- [x] `tests/test_cjk_glossary.py` +3 invariants — fill-only contract, the
      script guards, and the explicit-trust-weight rule.
- [x] Measured that the same source is the **strongest open lever on the 2a
      ship blocker** (booru sense vs MT's string translation; 35% agreement on
      our MT-derived wordings) — recorded as D1-pairs item 2 in
      [`plan.md`](plan.md#corpus--where-the-next-win-is), not yet run.

## Phase 2c — D1-pairs item 2, the arbiter re-selection (2026-08-17)

Measured verdicts:
[`report_0816_phase2.md`](report_0816_phase2.md#d1-pairs-item-2--the-arbiter-re-selection-measured-2026-08-17).

- [x] `tag_glossary.py --mt` takes the tag-pair names as arbitration
      candidates (same guards as the fill; per-candidate `src` provenance;
      winner `via: tagpair_verified`, trust 1.0 in all `TRUST_POLICIES`).
      Ranking: at equal F1 + kana-tier, community beats the MT rendering;
      kana-over-kanji untouched. Opt-out `--no-tag-pairs`.
- [x] Ran it fused with the **owed widened `--mt` rebuild** (glossary counts
      were still narrow-corpus): 14,678/14,753 tags worded, coverage 99.86%,
      **4,438 wordings moved / 0 pinned regressions**, corpus untranslated
      segments **13,714 → 878**. Pre-state kept at
      `assets/tag_glossary_ja.pre_item2.json`.
- [x] Found the arbiter's structural blind spot — the **polysemy class**
      (`bow`→蝶結び back-translates to the sense, not the string; F1 cannot
      verify it). Review filter extended to surface `mt_verified`-with-
      community-rival rows + the D1-words katakana/kanji section
      (`tag_glossary_review.md`, 400 sourced rows). The wording ceiling now
      runs through the human sign-off.
- [x] Retrained (`2c-item2`) + re-rendered: tags/tags_alt readout recovery
      0.915/0.934; grid moves on the coverage-bound prompts (t3 rider+horse,
      t2 background back), names unchanged (D5 still un-run). Renders under
      `bench/cjk_adapter/results/20260817-0138/`.
- [x] +2 `tests/test_cjk_glossary.py` invariants: community-beats-MT at F1
      tie; kana-beats-kanji with the rival surfaced to `kanji_review_rows`.

## Reusable for Phase 2b onward

| Piece | Where |
|---|---|
| Teacher-side encoding | `run_bench.py::SplitTokenizeStrategy` |
| Adapter tensors without a DiT load | `build_ext.py::read_tensor` pattern — extend to `net.llm_adapter.*` |
| Bespoke distill-loop template | `scripts/distill_mod/` → `scripts/distill_cjk/` |
| Adapter injection point | `library/anima/models.py::LLMAdapter` (~:2599) |
| Tag vocabulary / entity list | `caption_index.json` (`make caption-index`) |
| OCR + text detection (Phase 4) | MIT backend of `make mask` (`scripts/tasks/masking.py`) |
