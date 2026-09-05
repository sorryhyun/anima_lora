# PaddleOCR-VL-1.6 vs PP-OCRv6 on manga pages (2026-09-05)

**Question.** Does `PaddlePaddle/PaddleOCR-VL-1.6` (0.9B VLM, Apache-2.0, native in
transformers >= 5.0, no remote code) read our pages better than the PP-OCRv6 ONNX
det+rec pair `anime_tools.ocr` ships?

**Setup.** 40 sincos pages that already carry PP-OCRv6 sidecars
(`post_image_dataset/cjk_unmask/ocr/sincos`), resized tier (~1 MP). Three VL readings
per page — `Spotting:` on the x2-upscaled page (boxes + text), `OCR:` on the page, and
`OCR:` on each PP-OCRv6 detector quad (same boxes → recognizer-only comparison). Greedy,
bf16, sdpa, RTX 5070 Ti. manga-ocr records (`ocr_records_sincos.jsonl`) used as a noisy
third reader; disputed lines checked against the pixels by eye. Probe:
`probes/ocr_vl16_ab.py`; raw outputs `output/tests/vl16_ab/{ab.jsonl,ab.md}`.

## Numbers

| | PP-OCRv6 | VL-1.6 same-crop | VL-1.6 Spotting |
|---|---|---|---|
| lines on 40 pages | 132 | 132 (same boxes) | 260 |
| pages with any text | 40 | – | 36 |
| lines carrying a ♡/♥ | 4 | 30 | – |
| best-match sim vs manga-ocr (84 ref lines), mean | 0.767 | 0.774 | 0.694* |
| ref lines matched ≥ 0.9 | 35 | 38 | 15* |
| runaway repetition outputs | 0 | 2 / 132 | 1 / 40 pages |
| wall per page (this box) | ~0.1 s | 0.84 s (3.3 crops) | 1.7 s (median 1.2, max 10.6) |
| VRAM | ~0.3 GB | 2.1 GB peak | 2.1 GB |

\* Spotting emits one line per *column*, so a manga-ocr balloon line only ever
best-matches a fragment — the metric under-reads it; see the qualitative read.

Same-crop agreement: 69/132 crops identical after punctuation/heart normalization;
27 crops below 0.7 similarity, nearly all SFX where **both** are garbage.

## What VL-1.6 does better

- **Symbol coverage.** Hearts are kept (30 lines vs 4 — PP-OCRv6 drops ♡ almost
  always), `ー` comes back as `ー` (PP: `1`/`|`/`=` → `メ=ュー`, `かほ`; the
  `normalize_ja` fixups in `anime_tools.ocr._text` exist only for this), small kana and
  dakuten are right more often (`ムうムう`→`ムラムラ`, `エチ`→`エッチ`, `でけえ`→`でけぇ`,
  `林示止`→`禁止`, `K`→`OK`, `2名`→`20名`, `どうでー`→`どうぞー`).
- **Structure.** The crop reading breaks a balloon into its columns with `\n`; Spotting
  returns quads (`<|LOC_n|>` × 8, 0–1000 grid on the fed image) per column, so the column
  join we do by geometry comes for free — and it finds the SFX / chat chrome PP-OCRv6's
  DB detector misses (260 vs 132 lines: `ぬぎっ`, `ぱん ぱん`, `ヒクゥ`, `22:34` timestamps).
- **Silence on non-text.** Spotting returned nothing on the four pages where PP-OCRv6's
  only line was tally marks or screentone garbage.

## What it does worse

- **LM-prior hallucinations.** Page-level modes swap a plausible word for the printed
  one: `狠狠地` → `狼狽地` (both Spotting and page-OCR, crop mode got it right),
  `おい` → `あい`, `喘いで` → `噛いで`, `ご主人様` → `ごー主人様`, `カウンター` →
  `カウニター`, `いいなー` → `いいな一`. PP-OCRv6 has no such failure class — it garbles,
  it doesn't rewrite.
- **Runaway repetition** on short SFX crops (`ぉぉぉ…`×100, `゜゜゜…`, `ふくっ`×100 in
  Spotting): 2/132 crops + 1/40 pages with greedy decoding. Needs a repetition guard
  (`repetition_penalty` / `no_repeat_ngram_size`, area-scaled `max_new_tokens`, and a
  post-hoc n-gram collapse) before it can write sidecars unattended.
- **Spotting precision < crop precision.** The x2-upscaled page is capped at 2048
  vision tokens, so each glyph gets fewer pixels than a crop at 1280 tokens; the
  `ー`→`一`/`二` confusion and most misreads above are Spotting-only.
- **Cost.** ~15× PP-OCRv6 wall (still only ~1.5 h for a 3k-image set in Spotting
  mode), and it is a torch model — `anime_tools.ocr` is deliberately torch-free ORT.

## Gotchas learned

- The Hub `generation_config.json` ships **`use_cache: false`**; `generate()` then
  re-runs the vision tower every token (108 ms/token, 7.8 s of `.tolist()` syncs per
  page). `use_cache=True` gives byte-identical text at 85 tok/s (11×). Eager attention
  is 4× slower than sdpa again.
- Image processor `size` is `{shortest_edge, longest_edge}` in pixel counts (112,896 /
  1,003,520); the card's `processor.image_processor.min_pixels` attribute doesn't exist
  in transformers 5.16.
- Public data point: jzhang533's PaddleOCR-VL-For-Manga fine-tune reports the stock
  model at **27 % full-sentence accuracy on Manga109-s crops** (70 % after fine-tune) —
  our pages are cleaner digital doujin lettering, hence the near-parity here.

## Verdict

Recognition quality on the balloon text we care about is **a wash at the character
level** (each reader wins about as many disputed lines) — VL-1.6 is not a drop-in
accuracy upgrade over PP-OCRv6. Its real wins are the ones our `_text.py` post-pass
is compensating for by hand (hearts, `ー`, small kana) plus detection recall and
per-column structure; its real cost is a new failure class (plausible-word
rewrites, runaways) that a caption pipeline cannot see. If we adopt it, the sane shape is
**hybrid**: keep PP-OCRv6's detector + confidence, run VL `OCR:` on the quads (0.25
s/crop, no LOC parsing), guard repetition, and prefer the VL string only where PP's
score is low or the two disagree on symbols. Spotting-as-detector is worth a second
look only if SFX recall matters for the `japanese text` presence tag.

## Follow-up: prompt context and batching (same day)

Probe `probes/ocr_vl16_prompt_batch.py`, outputs `output/tests/vl16_prompt/`.

**Manga hints in the prompt do nothing useful.** The chat template concatenates the text
verbatim after the image (`User: <image>OCR:\nAssistant:\n`) and the model is trained on
the six fixed task tokens, so a hint is a perturbation, not an instruction. On the 132
crops:

| prompt | sim vs PP | sim vs manga-ocr | identical to `OCR:` | runaway | empty |
|---|---|---|---|---|---|
| `OCR:` | 0.845 | 0.592 | 132 | 2 | 0 |
| `OCR: Japanese manga dialogue, vertical text, read top to bottom.` | 0.844 | 0.589 | 58 | 2 | 0 |
| system "This is a Japanese manga page … right to left" + `OCR:` | 0.848 | 0.597 | 50 | 0 | 2 |
| `Japanese manga OCR:` | 0.858 | 0.585 | 76 | 1 | 0 |

40–60 % of outputs change, in both directions: the hints fixed `噛いで`→`喘いで` and
broke both runaway crops, but also produced `あとでいっぱい`→`あとでいっない`,
`フーッ`→`71`, `春山花奈ちゃんへ`→`(哲)`, and hallucinated `なんかくてなかった` on a
SFX crop. Net zero. For `Spotting:` on 12 pages the hints are actively harmful: wall
doubles (2.3 → 4.5 s/page), manga-ocr similarity drops 0.69 → 0.63/0.64, and 2–3 pages
per variant degenerate into hundreds of repeated lines (`おくり`×200, `ざくくく`×60).
The model's native Spotting order is not right-to-left either (39 R→L vs 69 L→R adjacent
pairs), so keep our geometric `reading_order` regardless.

**Batching is the throughput lever.** Left-padded batches (`padding_side="left"`,
crops sorted by area so padding is small):

| | bs=1 | bs=4 | bs=8 | bs=16 | bs=32 |
|---|---|---|---|---|---|
| crops/s | 4.0 | 7.6 | 9.7 | 12.3 | 18.5 |
| identical to bs=1 | 132 | 126 | 125 | 124 | 126 |
| peak VRAM | 1.8 GB | 1.9 | 2.0 | 2.2 | 2.6 |

Pages (`Spotting:`, x2 upscale): 0.56 → 1.15 → 1.40 pages/s at bs 1/4/8, peak 4.5 GB
at bs=8. Batched outputs are not byte-identical (bf16 under padding flips `♥`↔`❤️`,
`・・・` run lengths, and where a `\n` column break lands; one SFX crop flips) but line
counts and manga-ocr similarity are unchanged (0.702 both), and per-page line-set match
is ≥ 0.96 on every dialogue page. For a 3k-page set that is ~35 min Spotting at bs=8 or
~10 min crop-mode at bs=32, against ~5 min for PP-OCRv6.
