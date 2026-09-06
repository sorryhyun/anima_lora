# plan_ocr — a reader that reads SFX, then text-kind segmentation (2026-09-06, rev. 2)

*Side line of [`plan.md`](plan.md). It runs beside D2, never in front of it:
D2 proceeds speech-only on the C10 recipe ([`plan_base1.md`](plan_base1.md),
decision 2 as amended) and picks up whatever this line ships when a gate here
passes. The line has two goals, in order — **G-O1: a reader that reads
hand-lettered onomatopoeia** on doujin pages (the one thing neither PP-OCRv6
nor PaddleOCR-VL-1.6 can do), and **G-O2 (stretch): a text-kind segmenter**
(speech / SFX / other) that replaces the text-rule `kind` and gives `make
mask` a kind-aware mask. G-O1 unblocks the SFX sentence in captions
(`DROP_KINDS`); G-O2 unblocks the floor pages and the mask policy. Both are
also plain product wins for `anime_tools.ocr` (Tagger, masks, captions).*

*Rev. 3 (2026-09-06, evening): O2 arm B′ — the VL-1.6 LoRA with the **vision tower unfrozen** — passes the doujin gate (38 / 71) without O3; VL is the pick per decision 1, O3 is demoted to an optional lift, O4 starts. `findings.md` § O2b.*

*Rev. 2 (2026-09-06): the first eval is **in-domain** — the official COO
book split restricted to Manga109-s, fully automatic, no hand labels on the
critical path; the sincos hand labels move to the O2 gate. O2 trains **two
bases** on the same crops — manga-ocr and a PaddleOCR-VL-1.6 crop LoRA — and
the gate picks. PP-OCRv6 rec retrain stays the kill fallback.*

## What is measured today (2026-09-06)

| fact | number / pointer |
|---|---|
| stock manga-ocr on the 71 sincos `kind: sfx` crops (CPU, 12 % pad) | **~12 / 71** right (`ごめんっ`, `ぱんッ` ×2, `もじもじ`, `スリスリ`, `がくん`, `ぐちゃ`, `ブルン`); the rest are *fluent speech* — `ばるん` → `どうしているんじゃ`, `ばん ばん` → `というわけじゃない`. The glyphs are legible; the decoder's speech prior is the failure |
| PP-OCRv6 / VL-1.6 on the same lines | garble (`でくv`, `ゴmvvv`, `はんぱん`) or miss (six pink SFX on 12440144 at any resolution) — `plan_base1.md` table. VL keeps hearts (30 lines vs 4), `ー` and small kana, finds 2× the lines (260 vs 132 incl. SFX), but rewrites toward likelier words and runs away on short SFX crops (2 / 132) — `reports/0905_paddleocr_vl16_vs_ppocrv6.md`; prompt hints are not a lever (settled) |
| COO (`annotations_COO/` inside the Manga109-s v2026 zip) | **45,422** transcribed onomatopoeia polygons on 8,519 spreads of 87 books; 4–16-point polygons; 1,576 `onomatopoeia_link1` + 148 `link2` truncation links; text len p50 2 / p90 5 / max 28; 181-char set = 70 % katakana, 30 % hiragana, 3 % symbols; **99 hearts** in all; box min side p10/p50/p90 = 25 / 60 / 174 px |
| official COO split ∩ Manga109-s | the COO repo's `COO-data/books_{train,val,test}.txt` (109 books: 89 / 10 / 10) restricted to the 87 Manga109-s books = **74 / 7 / 6 books**. Test books: LoveHina_vol14, MAD_STONE, MukoukizuNoChonbo, ParaisoRoad, SaladDays_vol18, SyabondamaKieta = **2,558 COO lines** after truncation-link joining (98 joined; 2,759 counting the link elements); val: HinagikuKenzan, HisokaReturns, YasasiiAkuma, YoumaKourin, YukiNoFuruMachi, YumeiroCooking, YumeNoKayoiji. The published TRBA+2D 81.2 % is on the full 10-book test, so ours is *near*-comparable (6 of those 10 books), not identical |
| manga-ocr decoder vocab vs COO chars | 6,144 WordPiece entries cover 175 / 181 chars = **99.9 % of occurrences**; missing: `゛ ゔ ♫ ♬ ♩ ゜` (132 occurrences). `♡ ♥ ♪ ★ ☆ 〜 ー` are in vocab. No tokenizer surgery |
| PaddleOCR-VL-1.6 as a trainable base | ERNIE LM 18 layers × 1024 + NaViT tower 27 layers × 1152, ≈ 0.9 B, 1.8 GB bf16 at `models/paddleocr_vl_1.6`; loads via `AutoModelForImageTextToText` (`probes/ocr_vl16_ab.py`), so a LoRA is HF Trainer + `peft` (not yet in the venv). ~0.25 s/crop, 2.1 GB VRAM at inference (≈ 10× manga-ocr's wall). A maintainer's public Manga109 fine-tune of this model exists (stock 27 % sentence acc quoted from it), so the recipe is known to work |
| Manga109-s main annotations | **123,212** `<text>` boxes with transcriptions (speech, axis-aligned) + frame / face / body — the speech-side supervision and the replay set |
| sincos SFX vs COO style | sincos: digital doujin, hiragana-heavy, pink/white outlined lettering, `♡` on most lines, min side p10/p50 = 33 / 65 px. COO: 80s–00s printed B&W, screentone, katakana-heavy, heavy stylisation, many truncated. Same glyph inventory, different surface — the gap this plan has to cross |
| AnimeText test split, already on the external drive | 73,725 danbooru pages, 602,127 line polygons + 16,446 hard negatives, **no transcriptions** (`datasets/manga_text.py` docstring); in-domain unlabelled pool. CC-BY-NC-SA |
| text masks today | MIT text-pixel mask (`make mask`) says *text*, not kind; SAM3 `speech bubble` boxes balloons on 34 / 97 sincos pages only (`plan_base1.md`) |

## Assets (none of it enters the repo)

| asset | path | licence / rule |
|---|---|---|
| Manga109-s v2026 | `~/manga109s/Manga109s_released_2026_05_21/` (`images/<Book>/NNN.jpg`, `annotations/`, `annotations_COO/`, `books.txt`) | no redistribution; results **and pretrained models** may be published, commercial use allowed, with attribution + the three citations (readme cond. 2, 5–7); ≤ 20 % of any volume's pages in any publication |
| COO split lists | `assets/coo_split_manga109s.json` (in-tree — book *names* only, derived from the COO repo's `books_{train,val,test}.txt` ∩ `books.txt`, generated once by `ocr/build_manga109_crops.py --write_split`) | book names carry no licensed content |
| AnimeText test split | `/media/sorryhyun/새 볼륨/dataset/{test-00000-of-00001.parquet, polys_test.json}` | CC-BY-NC-SA-4.0 — anything trained on it is research-only; a shipped reader must have a build without it |
| sincos records | `post_image_dataset/cjk_unmask/ocr_records_sincos_hybrid.jsonl` (71 `sfx`, 259 `speech`, 8 `chrome`), pages `post_image_dataset/resized/sincos/` | in-tree |
| readers | `kha-white/manga-ocr-base` (ViT + BERT-ja WordPiece decoder; wrapped as `MangaOCR` in `../cjk_aware_anima/datasets/manga_text.py`, decode-only, no MeCab); PP-OCRv6 ONNX (`anime_tools.ocr._onnx`: `TextDetector` / `TextRecognizer` / `OcrEngine`); VL-1.6 (`models/paddleocr_vl_1.6`, `probes/ocr_vl16_*.py`) | — |

Code reads the two roots from env — `ANIMA_MANGA109S_ROOT`, `ANIMA_ANIMETEXT_ROOT`
(`ANIMA_`-prefixed because the daemon forwards only that prefix to its jobs,
`anima_daemon.config.CAPTURED_ENV_PREFIXES`) — never a literal path in a
tracked file (principle 9 of `plan.md` applied to licensed data).

## Decisions

1. **Fine-tune, never from scratch; two bases race, manga-ocr is the
   default.** manga-ocr-base already reads vertical Japanese natively, is
   wrapped in-tree, and its vocab covers COO; the failure is a language
   prior, which a fine-tune on 45k short kana bursts attacks directly.
   **PaddleOCR-VL-1.6 with a LoRA on the crop `OCR:` task is the second
   arm** on the identical crops: it keeps hearts / `ー` / small kana without
   rules (decision 6's patching disappears if it wins) and its tower is
   resolution-native, but it shares the autoregressive failure family
   (rewrites, runaways) and costs ≈ 10× per crop. Selection: whichever
   passes the O2 gate; on a tie manga-ocr wins on wall and deployment; VL
   wins outright only if it passes *and* removes the heart rule. PP-OCRv6's
   CTC head stays the *speech* recognizer of record (`plan_base1.md`
   decision 1 holds) and its rec retrain is the kill fallback, not an arm —
   its crop convention lays vertical glyphs on their side (a documented weak
   point of `anime_tools.ocr._onnx`) and its training stack (Paddle) is not
   in the venv. The tuned model is a **third reader**, slotted by `kind` and
   by the floor, not a replacement.
2. **Speech replay is part of the recipe from the first run.** Fine-tuning on
   COO alone would trade the speech prior for an SFX prior; Manga109-s
   `<text>` crops ride along (target ≈ 1 : 1 by batch; note COO strings are
   2–5 chars and speech 10–30, so tokens are far from balanced — a 1 : 2
   arm exists only if the speech control slips) so the reader stays a
   general manga reader. The speech control in O0 is what proves it.
3. **The first eval is in-domain and automatic; the sincos hand labels are
   the second gate.** O0 measures every reader on the **official COO book
   split ∩ Manga109-s** (74 / 7 / 6 books; test = 2,558 COO lines +
   a matched draw of `<text>` speech crops from the same six books) — no
   hand labelling on the critical path, and a number near-comparable to the
   published TRBA+2D 81.2 %. Speech crops use the *same* book split so no
   test book leaks through the replay set. The 71 sincos SFX crops + a
   speech control measure what this line is *for* (the doujin gap) and gate
   O2 second; their ground truth is human (drafted by the assistant off the
   contact sheet, corrected by the user), stored as
   `assets/sfx_labels_sincos.tsv` — the file `plan_base1.md` B1 planned,
   with a `text` column added. One file serves B1 and this plan; it is
   drafted while O1/O2 run.
4. **The domain gap is crossed by data, not by prompts.** Three levers, in
   order: **colorized COO** (the EasyControl colorize LoRA repaints
   Manga109-s spreads as flat-tint doujin pages while leaving the lettering
   and the layout alone — real hand-drawn glyphs, human labels, ship-clean),
   then synthetic doujin-style crops (outlined kana + hearts over doujin
   backgrounds — the lettering style neither COO nor its colorized copy
   has), then self-training on AnimeText's in-domain polygons. The
   self-trained round is a research build (NC).
5. **Segmentation is a second gate, not a second track.** O5 starts only
   after O2 passes; its supervision is what the reader work already
   assembled (COO = sfx, Manga109 `<text>` = speech, AnimeText = text of
   unknown kind + hard negatives). It replaces `ocr_sfx.line_kind` + the MIT
   mask, not SAM3 (already out, `plan_base1.md` decision 4). A page-level VL
   `Spotting:` LoRA (COO polygons as targets — detection + kind + text in
   one model) is O5's *candidate*, not O2's: Spotting is where VL emitted
   hundreds of junk lines and doubled wall, and it is a 1654×1170 spread
   task, not a crop task. It is earned by the crop LoRA passing O2 first.
6. **Hearts and truncation are handled by rule, not by vocabulary.** `♡` is
   in vocab but has 99 training examples; the wire-up takes the heart from
   VL's read (which keeps hearts) when the tuned reader drops one — moot if
   the VL arm is the one that ships. COO `link1/link2` pairs are trained as
   their *joined* string on the union box when the parts share a page, else
   dropped.
7. **Every GPU step is a daemon job**; CPU steps (crops, scoring) run
   directly. Nothing here blocks D2.

## Phases

### O0 — in-domain split + scorer + baselines (½ day, CPU + one short GPU job)

- `assets/coo_split_manga109s.json`: the official COO `books_{train,val,test}`
  intersected with `books.txt` (74 / 7 / 6; the lists above). Written once,
  asserted by the crop builder. No private split — a by-name 77 / 5 / 5 would
  make the held-out number incomparable to the COO leaderboard.
- `ocr/eval_manga109.py`: reads a crop manifest (O1's, or a quick test-only
  cut of it), runs any reader, reports **exact match**, **normalised char
  similarity** (`build_ocr_records.sim` after `normalize_ja`) on the COO
  test crops, and a **speech control** (same metrics on the matched
  `<text>` crops from the six test books). Any reader = `MangaOCR`,
  PP-OCRv6 `TextRecognizer` (crop rotated per its own convention), VL-1.6
  crop `OCR:` (batched, `use_cache=True`, greedy, **no** repetition guard —
  the runaway count is part of the stock row; a guard is a wiring knob).
  Writes `reports/ocr_eval_<reader>.md`.
- Baselines in the report: stock manga-ocr, PP-OCRv6 rec, stock VL-1.6 on
  the COO test crops + the speech control. These are the O2 reference rows.
- Started in parallel, not on the path: the sincos label draft
  (`assets/sfx_labels_sincos.tsv`, `stem, box, kind_hand, text_hand` — the
  71 `sfx` rows get a transcription, the 259 `speech` rows get a `kind_hand`
  check and keep the record text unless obviously wrong; conventions:
  glyphs as printed, small kana, `ッ`, `ー`, hearts as `♡`, repeated blocks
  joined with a space per the `JOIN_SEP` rule, no romaji) and
  `ocr/eval_sfx.py` (same scorer over that TSV with the pilot's
  `deskew_crop`, 12 % pad; stock-reader rows including VL's from
  `ocr_raw_vl16_sincos.jsonl`).

*Gate:* split file written and asserted; stock manga-ocr, PP-OCRv6 and
VL-1.6 numbers on the COO test crops + speech control in the report. (The
sincos ≈ 12 / 71 reproduction is O2's precondition, not O0's gate.)

**O0 — DONE 2026-09-06 (gate PASS; `findings.md` § O0).** Stock rows on the
2,558 COO test lines: manga-ocr **16.4 %** exact / 0.336 sim (speech sim
0.824), VL-1.6 **20.1 %** / 0.449 but 4.1 % runaways (speech sim 0.845),
PP-OCRv6 rec 3.0 % (its speech row is invalid — bubble boxes are
multi-column). Published COO TRBA+2D: 81.2 %. Joined truncation lines are
near-unreadable stock (1 / 98). Sincos label draft + `eval_sfx.py` still
owed before the O2 gate.

### O1 — COO + speech crop builder (½ day, CPU)

- `ocr/build_manga109_crops.py`: `ANIMA_MANGA109S_ROOT` → deskewed crops via
  `deskew_crop` (orientation preserved; pad 12 %), for (a) every COO polygon
  (links joined per decision 6), (b) a matched draw of `<text>` boxes, both
  assigned by the O0 split file (book-level; speech and COO share it). Drop
  min side < 16 px. Emit an image folder + `manifest.parquet` under
  `~/manga109s/derived/` (never in-tree) and a stats block (count per
  split, length histogram, char coverage) into `findings.md`.
- Augmentations decided here, applied at train time: random pad 5–25 %,
  ±8° rotation, JPEG, contrast / invert (white-on-dark lettering), a
  colour-tint pass (sincos SFX are pink on skin tones — COO is grey).

*Gate:* ≥ 40k COO crops kept; test books contain ≥ 2k.

**O1 — DONE 2026-09-06 (gate PASS; `findings.md` § O1).** 43,535 COO lines
(train 38,582 / val 2,395 / test 2,558; 1,724 joined, 55 min-side drops) +
a count-matched 43,589 speech crops, 3.7 GB under `~/manga109s/derived/`,
`manifest.parquet`. Augmentation spec = `ocr/augment.py` (`Augment`).
Two O2 inputs from the stats: speech targets need NFKC + whitespace strip
(newlines / full-width punctuation are 6 % of speech chars and outside
manga-ocr's vocab); 1 : 1 by count is 5.4 : 1 by characters.

### O2 — fine-tune, two bases (2 × 1–2 GPU-h on the daemon)

- `ocr/finetune_manga_ocr.py`: HF `VisionEncoderDecoderModel` +
  `Seq2SeqTrainer`, tokenizer built from `vocab.txt` (the wrapper's decode
  path in reverse — chars → WordPiece ids; no fugashi), lr 2e-5 → 5e-5
  sweep on the val books, 3–5 epochs, batch 64, bf16, COO : speech 1 : 1.
- `ocr/finetune_vl16_lora.py`: `peft` LoRA (r 16–32) on the LM's attention
  + MLP of `models/paddleocr_vl_1.6`, tower frozen, crop `OCR:` prompt via
  the chat template, same manifest / mix / epochs, batch by area with
  left padding (the batching rule from the A/B), lr 1e-4 → 2e-4 sweep on
  the val books. `uv add peft` is the only new dependency.
- Both: checkpoints + `eval_manga109` per epoch, then `eval_sfx` on the
  best val epoch. Each is a `make daemon-run` job, output under
  `output/ocr/<run>/`.
- Third arm only if the speech control slips: 1 : 2 COO : speech (decision 2).

*Gate (G-O1 first pass), per base:* COO test exact **reported** (the
published baseline is 81.2 % on the 10-book test); sincos SFX **exact ≥ 35 /
71** (from ~12) **and** sincos speech control char similarity ≥ stock − 0.01
**and** COO speech control ≥ its O0 stock row − 0.01. Pick per decision 1.
*Kill for a base:* sincos SFX exact < 25 / 71 while COO test ≥ 70 % → that
model learned COO's surface, not kana; O3 becomes mandatory for it before
any wiring. *Kill for the line's bases:* both miss after O3 → PP-OCRv6 rec
retrain (Paddle stack, crop convention fixed first) is the only remaining
route.

**O2 — DONE 2026-09-06 (in-domain PASS, doujin gate MISS on both bases;
`findings.md` § O2).** Corrected crops (§ O1 correction — the pilot's
`deskew_crop` transposed axis-aligned boxes; O0 rows re-run). manga-ocr
lr 5e-5 × 4 ep: COO test **73.5 %** (stock 26.2, published 81.2), speech
0.975 = stock, sincos gate **10 / 71** (♡-blind 12). VL-1.6 LoRA lr 1e-4 ×
2 ep: COO 64.7 %, 194 runaways, sincos **13 / 71** (19). Kill clause fires
for arm A (< 25 while COO ≥ 70) → O3 mandatory; O3 runs on manga-ocr first
(synth before colorized — the residual is hearts + outlined kana), VL only
if that falls short. lr 2e-4 / 8-epoch arms cancelled (curves still rising,
not the bottleneck).

**O2 amended the same day — arm B′ PASSES (`findings.md` § O2b).** Same
recipe as B with `--train_tower --tower_lr 1e-5` (NaViT tower + projector
full FT, 439 M, fp32 master), 1 epoch, ~90 min: COO test **81.7 %** (at the
published 81.2 %), speech 0.986, sincos gate **38 / 71** (♡-blind 41), sincos
speech 0.910. All four gate clauses hold → **decision 1 resolves to VL**
(passes + reads `♡` natively, so decision 6's heart rule is dropped). The
frozen tower, not the decoder prior, was the doujin gap. O3 is no longer
mandatory for any surviving base; it is an optional lift for the residual
(8+-char lines, square blocks, `ぱん♡` confusions). Weights public at
`sorryhyun/paddleocr-vl-1.6-manga-lora` (adapter + `tower.safetensors`).

### O3 — crossing the doujin gap (1 day) — **optional lift since rev. 3**

*Not on the path any more: arm B′ passed the gate without it. Run a lever here only if O4 wants more margin on the residual; the +10 gate below then reads against B′'s 38 / 71.*

Only what O2's residual asks for, in this order, on the surviving base(s):

- **Colorized COO** (`ocr/colorize_manga109.py`, one daemon job): the
  EasyControl colorize LoRA (`EASYADAPTER=colorize`, empty prompt — the
  caption-free path in `scripts/tasks/inference.py`) over the train-split
  spreads that carry a COO polygon, at the **1536 tier** or upscaled back
  to native 1654×1170 before cropping — a 1191×840 output (what the first
  hand trial produced) drops COO's p10 min side from 25 to ≈ 18 px, at the
  16 px floor. Output under `~/manga109s/derived/colorized/`, then the O1
  crop builder re-run on it with the *same* polygons (layout is preserved;
  the 12 % pad absorbs drift). Hand trial 2026-09-06 on
  `AisazuNihaIrarenai/015`: the three SFX (どんどん ×2, がちゃがちゃ)
  and the bubble text came through black and unchanged, art repainted
  flat-tint with pink washes — the doujin *surface* without the doujin
  *lettering*. What it buys: the background / screentone half of the gap
  with zero label noise and no NC taint; what it does not: outlined pink
  kana, hearts (still synth's job).
  *Its own gate, before the GPU hours:* a 20-page pilot at the chosen
  resolution — (a) stroke-mask IoU between binarized source and colorized
  crops (drop a crop below the threshold set here, expected ≥ 0.8), (b)
  stock manga-ocr on source vs colorized crops of the same polygons: reads
  must agree with each other at the rate the source read agrees with the
  label. Pass → colorize a 2k-page subset first (≈ 10k crops), mixed 1 : 1
  with the grey COO crops; the full 8.5k spreads only if the 2k mix moves
  the sincos number. Fail on (b) → glyphs are being redrawn; the lever is
  out and synth moves up.
- **Synthetic doujin SFX** (`ocr/synth_sfx.py`): COO's text distribution
  (plus a hiragana re-weight to sincos' mix and `♡` appended at sincos'
  rate) rendered in handwriting-leaning JA fonts, outlined (white / pink
  stroke), placed on doujin-like backgrounds cut from `image_dataset/`
  pages *outside* the text masks, vertical and diagonal. 50–100k crops;
  mixed 1 : 1 : 1 with COO (grey + colorized pooled) and speech; O2 rerun.
  This is the lever for hearts, outline style and hiragana.
- **Self-training on AnimeText** (research build only): run the O2/synth
  reader + VL-1.6 crop `OCR:` over AnimeText's short-kana polygons (min
  side ≥ 32 px, reader length ≤ 6, both readers agree after `normalize_ja`,
  logprob above the O0-tuned gate), take the agreed strings as
  pseudo-labels, one more epoch. Tag the checkpoint `-nc`.

*Gate:* +10 exact on the sincos set over O2, both speech controls held;
each lever's contribution reported separately (colorized-only, +synth,
+self-train) so the ship build knows which of the NC-free levers carried it.

### O4 — wire it in, then the DiT-side payoff (½ day + one arm)

- `build_ocr_records.py` gains a third reader (`--sfx_reader <ckpt>`):
  every `kind: sfx` line (any engine) is re-read by it; on floor pages
  (masked, no line) the MIT mask's connected components ≥ 32 px become
  crop quads and are read too — a read that passes the O0 gate becomes a
  record with `engine: sfx_reader`. Hearts patched from VL per decision 6
  (skipped — the VL arm ships). **A decode guard (repetition / aspect-tied
  length cap) is part of the reader wiring**, not optional: B′ still runs away
  on ~4 % of crops. Re-measure the floor (23 → ?) and the SFX
  read accuracy on sincos.
- `anime_tools`: the reader lands as `anime_tools.ocr.sfx` (same
  `OcrLine` sidecar, weights fetched via `anime_tools.downloads`), so `make
  preprocess`'s OCR stage and the Tagger node can use it. **Deployment
  shape is decided at O2 pick time**, not after — **decided rev. 3: VL**
  (torch + remote modeling files + peft adapter 24 MB + `tower.safetensors`
  878 MB from `sorryhyun/paddleocr-vl-1.6-manga-lora`, batching rules from
  the A/B, decode guard): manga-ocr = a torch
  `VisionEncoderDecoder` (or its ONNX export) behind the existing session
  layer; VL = torch + the remote modeling files + the batching rules — the
  heavier of the two, which is why a tie goes to manga-ocr. Pinned-rev bump.
- **Arm C11** (`plan_base1.md`'s arm framework): C10 + the SFX sentence
  (`Japanese SFX reads as "…"`) now that the lines are readable; 3 seeds,
  grids, blind set vs C10. `DROP_KINDS` loses `sfx` only if C11 passes.

*Gate:* floor down by the SFX-page count on the contact sheet; C11 spam ≤
C10 on 3 seeds and blind ≥ C10 inside the seed-twin floor. *Kill:* C11
spams → SFX stays out of captions, the reader still ships for masks/Tagger.

**O4 — IN PROGRESS 2026-09-06 evening (`findings.md` § O4).** The reader
ships as `anime_tools.ocr.sfx` (rev 46ebbb5, rows `vl16_base` /
`sfx_reader`, decode guard = repetition + **area**-tied cap — an aspect cap
threw away 60 % of speech reads). `ocr/reread_records.py` (the `--sfx_reader`
wiring, as its own script) re-reads every `kind: sfx` record and reads the
MIT-mask components on every masked page: floor **23 → 8**, sincos gate rows
4 → 37 / 71 in the pipeline, `kind` from the hand labels. The user's
"all lines through VL" arm measured too: best on the manga-ocr reference
(0.810 vs 0.786), kept as the D2 records recommendation; C11 runs
single-variable on the SFX-only re-read. Arm C11 (**one seed, s42 — user's
call**) running; verdict owed. **O5 parked** (user's call, same evening):
`ocr/kind_seg.py` is written and CPU-smoked, never trained.

### O5 — text-kind segmentation (stretch; 2–3 days; after O2 passes)

- **Task:** per-page polygons with `kind ∈ {speech, sfx, other}` (other =
  chrome, signs, titles, credits). Consumers: `kind` in the records
  (replacing `ocr_sfx.line_kind` + hand rules), `make mask` (kind-aware
  masks: e.g. keep SFX pixels, mask chrome), the floor (a detector that
  actually boxes pink SFX).
- **Supervision:** COO polygons = `sfx`; Manga109-s `<text>` = `speech`;
  AnimeText polygons = `text` with kind from the O2 reader + rules
  (pseudo; research build) plus its `hard_negative` boxes as background;
  sincos hand labels (O0) = the in-domain eval. A 200-page hand-labelled
  doujin set is the ship-grade eval if the pseudo route is used.
- **Model, two candidates:** (a) one instance-segmentation head anime_tools
  can run through its ONNX session layer (YOLO-seg class of model; DBNet++
  if polygons matter more than speed — COO's own leaderboard has DBNet++ at
  72.9 Hmean, recall 60.9, so recall is the known weakness), exported to
  ONNX; the deepghs AnimeText YOLO12 weights are the research-build init if
  their licence is acceptable for that build. (b) a VL-1.6 `Spotting:` LoRA
  on the same page-level targets (decision 5) — only if the VL crop arm
  passed O2; it would fold detection, kind and text into one model, at VL's
  page wall.
- **Two builds:** ship (Manga109-s + synthetic + hand labels) and research
  (+ AnimeText). Only the ship build lands in `anime_tools`.

*Gate:* sincos kind accuracy ≥ B1's rule on the hand labels, SFX recall on
the 133 masked pages ≥ 0.8 of the hand count (the floor pages are the
point), speech recall ≥ PP-OCRv6's DB head. *Kill:* SFX recall no better
than MIT's mask → the mask stays the localiser and only the reader ships.

## Not doing here

- **No from-scratch OCR** and no new architecture; no VL prompt sweeps
  (settled), no VL as *sole* reader, **no page-level VL `Spotting:` tune
  before the crop LoRA passes O2** (`plan.md`'s not-doing list is amended to
  this: the crop LoRA is now an O2 arm).
- **No PP-OCRv6 rec retrain** unless both O2 bases miss after O3 (kill
  clause); it needs the Paddle stack and a vertical-crop convention fix
  first.
- **No private book split**: the official COO split ∩ Manga109-s is the only
  split.
- **No SAM3 balloon soft prompt** (plan_base1 decision 4); O5 is the
  balloon-free route to kind.
- **No romaji SFX** in captions; glyphs only.
- **No AnimeText-trained weights in a shipped build**; no Manga109-s images
  or crops in the repo, on HF, or in a node — colorized derivatives
  included (they are still Manga109-s pages) — weights only, attributed.
- **No D2 dependency**: D2 freezes speech-only; C11 is a follow-up arm.

## Order and budget

O0 (½ d, automatic; the sincos label draft starts here in parallel) → O1
(½ d) → O2 (2 bases × 1–2 GPU-h + ½ d; sincos labels must be corrected by
the time the best epochs exist) → O3 only on O2's residual (1 d + the colorize job, ≈ 2k pages of EasyControl inference on the daemon, overlapping O2's second base) → O4 (½ d
+ one 3-seed arm ≈ 2 GPU-h) → O5 (2–3 d, stretch). Four working days to a
shipped SFX reader and the C11 verdict; a week with segmentation. GPU via
`make daemon-run` throughout.

## Deliverables

- `assets/coo_split_manga109s.json`, `ocr/eval_manga109.py`,
  `reports/ocr_eval_*.md` with the stock rows (manga-ocr, PP-OCRv6, VL-1.6)
  on the COO test + speech control.
- `assets/sfx_labels_sincos.tsv` (kind + text, human), `ocr/eval_sfx.py`.
- `ocr/build_manga109_crops.py`, `ocr/augment.py`, `ocr/finetune_manga_ocr.py`,
  `ocr/finetune_vl16_lora.py`, `ocr/colorize_manga109.py` + its 20-page
  pilot table in `findings.md`, `ocr/synth_sfx.py` (O3), the manifest +
  stats in `findings.md`.
- Weights: **`sorryhyun/paddleocr-vl-1.6-manga-lora`** (public 2026-09-06;
  adapter + fine-tuned tower, model card carries the Manga109-s attribution
  and citations). `sorryhyun/manga-ocr-sfx` not published — arm A lost the
  pick. Any `-nc` build stays local.
- `build_ocr_records.py --sfx_reader`; `anime_tools.ocr.sfx` + pinned rev;
  arm C11 (`configs/gui-methods/custom/cjk_unmask_c11.toml`, grids, blind
  set, `reports/09xx_c11_sfx_sentence.md`).
- O5: `ocr/segment/` (data builder, train, ONNX export), the 200-page eval
  set, `anime_tools.ocr.kind` + kind-aware `make mask`.
