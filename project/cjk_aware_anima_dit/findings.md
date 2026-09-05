# CJK-aware Anima, DiT side — findings

Settled verdicts of this line, one entry per phase, evidence pointer beside
each. The encoder-side verdicts it builds on are in
[`../cjk_aware_anima/findings.md`](../cjk_aware_anima/findings.md) (read-only).

## D0 — ISO1 vs C9 direct blind set: flat (2026-09-05)

`reports/blind_s13_ISO1_vs_C9.md` (in the old line's `reports/`): 48 pairs,
16 v2 rows × seeds 6/7/8, both arms fresh to the grader. **ISO1 23 – C9 20,
tie 5; rows 6-6 (tie 4); p 0.76.** The isotropic table and the trained
r256 pack are indistinguishable for unmask training on this grid.

- The transitivity claim ISO1 ≈ HOT > C9 (s12 + s11) does **not** survive
  the direct test; transitivity has now failed twice in this protocol
  (s03/s04, s12/s13). Do not chain blind sets — pair the arms you want to
  compare.
- Pooled s01–s13: a content-free table is never worse than the trained pack
  for the OCR route, and rows must exist (C9 > P). The isotropic block is
  therefore the OCR-route default on **cost** grounds (seed-generated,
  deterministic, no distill), not on quality grounds. The hypothesis doc's
  "structured low-rank spread hurts" mechanism is weakened, not confirmed.
- Mechanism note carried from `bench/frontload_text_boost`: `k_norm`
  strips row scale on the K path, which is why HOT (norm ×5) ≈ ISO1 (s12).

Gate outcome: **proceed to D1** with the isotropic block for 「…」 spans;
bare CJK tags keep the trained rows (plan principle 2).

## OCR reader for D2/D3 — PaddleOCR-VL-1.6 is not an upgrade over PP-OCRv6 (2026-09-05)

`reports/0905_paddleocr_vl16_vs_ppocrv6.md`; probes `probes/ocr_vl16_ab.py`,
`probes/ocr_vl16_prompt_batch.py`; raw outputs `output/tests/vl16_{ab,prompt}/`.
40 sincos pages with PP-OCRv6 sidecars, VL-1.6 read three ways (page
`Spotting:`, page `OCR:`, `OCR:` on PP-OCRv6's own quads); disputed lines
checked against the pixels.

- **Character accuracy is a wash on the same crops** (69/132 identical after
  punctuation normalization; each reader wins about half the disputed
  lines; vs manga-ocr 0.767 PP / 0.774 VL). VL is not a drop-in accuracy
  upgrade — consistent with the public manga fine-tune figure (stock model
  27 % sentence accuracy on Manga109 crops).
- **VL wins symbols and recall.** Hearts survive (30 lines vs 4), `ー` and
  small kana come back as themselves (what `anime_tools.ocr._text`'s
  `normalize_ja` patches by hand), and page `Spotting:` finds 260 lines vs
  PP-OCRv6's 132 — SFX, chat chrome — with per-column boxes and silence on
  the tally-mark / screentone pages.
- **VL loses by rewriting.** Page-level modes swap the printed word for a
  likelier one (`狠狠地`→`狼狽地`, `おい`→`あい`, `喘いで`→`噛いで`) and
  greedy decoding runs away on short SFX crops (2/132 crops, 1/40 pages).
  PP-OCRv6 garbles but never rewrites. **This decides the D3 CER judge**:
  a reader with an LM prior would "repair" a half-rendered line toward the
  prompted word and inflate the held-out-vs-never-seen gap. The render→OCR
  CER instrument uses PP-OCRv6.
- **Prompt hints are not a lever.** The chat template concatenates text
  after the fixed task token; "vertical / right-to-left / Japanese manga"
  hints churn 40–60 % of crop outputs in random directions (sim vs PP
  0.845 → 0.844 / 0.848 / 0.858) and make hinted `Spotting:` 2× slower,
  lower (0.69 → 0.63), and degenerate on 2–3 of 12 pages. Native Spotting
  order is not R→L either (39 vs 69 adjacent pairs) — `reading_order`
  stays geometric.
- **Batching is the throughput lever.** Left-padded, area-sorted crops:
  4.0 → 18.5 crops/s at bs 32 (2.6 GB); pages 0.56 → 1.40 pages/s at bs 8
  (4.5 GB). Outputs churn at the byte level (`♥`↔`❤️`, ellipsis lengths,
  column-break placement) with unchanged line counts and similarity.
  Two shipped gotchas: `generation_config.json` has `use_cache: false`
  (vision tower re-run per token; `use_cache=True` is byte-identical at
  11×), and transformers 5.16's image processor exposes `size`, not the
  card's `min_pixels`.

Decision for the plan: **D2's records stay PP-OCRv6** (detector +
confidence + v2 post-processing, unchanged); VL-1.6 enters only as an
optional hybrid pass — `OCR:` on PP-OCRv6's quads with a repetition guard,
preferred where PP's score is low or the two disagree on symbols — and as a
**detector for the "masked but no OCR line" floor**: its Spotting recall is
the one thing that could shrink sincos' 44-of-133, which caps every unmask
arm. Measure that floor with both detectors in D2 before deciding whether
the hybrid pass is built at all.

## D1 — deterministic table + route partition + LoRA stamp (2026-09-05)

Code landed; the gate render is queued (daemon job `20260905-210248-114990`,
arm `C9ISOQ` = the C9 recipe re-cached through the partitioned pack, 8-row
grid at seeds 42/7/1234 into `output/tests/cjk_unmask_eval2/armC9ISOQ_s*`,
to be read against `armC9_s*`; the prompts are CJK-free, so anything outside
the s02 floor is a bug, not a result).

What exists now (pointers, not repeats — contract in
`docs/experimental/cjk_ext_vocab_coverage.md` §"Quote partition"):

- `library/anima/ext_vocab.py`: `iso_block` / `IsoSpec` / `materialize_iso`
  (seed-regenerated isotropic mirror, NumPy legacy stream, byte-equal
  across machines), `Route.quotes` + `quote_spans` (one regex, non-nesting),
  `HybridT5Encoder.encode_cjk_run` (span rule before `segment_runs`; EN
  bit-identical by construction), `pack_digest`.
- `make_random_pack.py --mode iso | iso-partition [--no-iso-rows] [--norm]`
  (norm default = native T5 mean row norm 212.165, measured off the DiT's
  `llm_adapter.embed`; ISO1 had used the trained mean 203.9).
  Built: `output/ckpt/cjk_vocab_pack_synthjakozh1sym_r256_isoq` (sha
  `2cf81cbc…`; mirror rows 69,558–139,116, PR 1009).
- Stamp: `train.py --ext_pack` → `ss_ext_pack_sha` / `ss_ext_pack`
  (`run_unmask_r2.py` passes it); `load_dit_model` warns on a stamped LoRA
  with no pack; Adapter node 3.10.0 compares digests in either node order
  (`vocab_pack.check_pack_vs_adapter`, `adapter._record_ext_pack_stamp`),
  regenerates seed-only blocks, cuts routed runs at quote boundaries in
  `VocabPackTokenizer`. Vendor tree re-synced; node committed locally, not
  yet pushed / registry-published (publish with the first partitioned pack).
- Grammar: anime_tools `efb235c` (`quoted_spans`; comma / `. On the` inside
  `「」『』""` is content; `compose_caption` round-trips) — pin bumped,
  `uv lock` + `uv sync` done. `cache_te_ext._quote_safe` keeps commas now.
- Tests: `tests/test_ext_vocab_iso.py` (determinism, EN bit-exact, quoted
  content only on the mirror with `「」`/`黒髪` staying on trained rows, three
  spellings → same ids, `"…"` order phrase routes, ASCII inside quotes stays
  spiece, digest invariance under regeneration, grammar, inference warn) +
  the earlier route tests: 58 passed.

Design choices worth knowing: the mirror is a full row-for-row copy (so
one id map serves both blocks; 285 MB fp32 shipped, or 0 bytes seed-only),
quoted content bypasses minted-word rows and the C fallback (those are
trained content), and the rule is inert unless *both* `iso` and
`route.quotes` are present — every existing pack, cache and blind set is
untouched.

