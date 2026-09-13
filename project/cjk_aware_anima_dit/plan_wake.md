# plan_wake — the address hypothesis: wake the DiT's own JA glyph units (2026-09-13)

> **Status (2026-09-13).** Opened from the user's reframing of `plan_render`'s
> floor; W0 ran the same day (three daemon jobs, ~2.5 h GPU). Probe 0 and the
> `rows_adapter` arm are positive on the two claims that matter — the units
> exist, and a **frozen DiT draws font-clean kana from text-side changes
> alone**. **W1 passed the same evening**: on 8 hiragana, a rows-only delta
> with the DiT *and* the adapter frozen renders the requested kana 9/16 exact
> on both readers (floor 0/16, EN control 24/24 untouched) — the address alone
> wakes the glyph. Sequences do not compose yet (combos 0/36 exact, CER 1.0 →
> 0.70): that is W2. Nothing here is shipped.

## The hypothesis

`plan_render` read JA-SHIP / JA-BODY's floor as "the DiT cannot render JA from
ext rows, even with a trainable body". The user's counter-reading:

> Anima observed lots of Japanese, Korean, English, even Chinese characters in
> pretraining. It cannot render CJK because it could not *learn those tokens* —
> the stock T5 SentencePiece collapsed every CJK caption to `<unk>` — so instead
> of learning to render, Anima learned to **garble**. English works because the
> address (T5 pieces spelling the word) existed. The aim moves from "teach the
> DiT a script" to **"wake up weights"**: find the address, not the capacity.

Why it fits what the line already knew: `findings.md` § 9 (encoder line) —
the DiT reads near-context-free per-token codes, the row is a key and the six
adapter blocks store the value; the text-binding probe — a 1-image LoRA at 400
steps reproduces the exact JA line crisply (glyph drawing is cheap for the
DiT, the address is what never forms). JA-BODY never tested this: it added a
target-stream LoRA at 1e-4 under a whole-crop inpaint loss where the glyphs are
a few percent of the pixels. JA-SHIP had no trainable weight on the row → pixel
path. **No arm in the line had ever put a pixel loss on the rows with the DiT
frozen.** W0 is that arm.

## W0 — the probes (2026-09-13)

Instrument: [`probes/wake_probe.py`](probes/wake_probe.py), stages `salad` /
`data` / `train` / `eval`; outputs under `output/wake_probe/<stage|arm>/`
(`report.md`, `eval_reads.json`, `sheet_*.png`). Readers: AnimeText detector
boxes → the manga-tuned SFX reader **and** stock PaddleOCR-VL-1.6 on the same
crop; *agreement between the two* is the unit test, because the manga-tuned
reader alone hallucinates kana from salad.

### Probe 0 — is the base salad made of real units?

Base Anima, no LoRA, 12 EN prompts × 3 seeds at 768² (`salad`, job
`20260913-115024-d72433`). 369 detector boxes.

| | boxes | SFX emits CJK | VL16 emits CJK | **both agree (CER ≤ 0.5)** |
|---|---:|---:|---:|---:|
| all prompts | 369 | 85 % | 71 % | **19 %** |
| manga-panel prompts (4koma, 2-person, shouting) | 99 | | | **~50 %** (30/62, 14/19, 2/3) |
| signs / neon / storefront | 168 | | | 1 % (2/168) |
| KO / ZH prompts | 47 | | | 0 |

Agreed strings: 「いい!」「おだきるっ!」「だはいい」「おりいー」「う…」「ろう…」
and long nonsense sentences with real particles (は/を/に/が) and kanji in
Japanese orthographic proportions. **Bubble text is real characters in random
order; sign text is true salad.** KO/ZH: nothing agreed — the KO/ZH texture
prior exists (readers emit Hangul/Han) but no unit survived two readers.

### Probe 1 — does an address wake a glyph? (frozen DiT)

Data: 92 kana (46 hiragana + 46 katakana) as 1–3-char strings rendered in 15
Noto Sans/Serif CJK weights + Droid (1252 renders, bubble or plain canvas) +
479 kana-only corpus bubble crops (`render/ja/resized/boxes.jsonl`, ≤ 6 chars);
caption = the render line's clause shape
`manga, speech bubble, japanese text. Japanese text reads as "…".`. 512²,
batch 4, plain rectified-flow loss (`fm_training_batch`), DiT + Qwen frozen,
grad-ckpt. Eval = 18 singles + 18 held-out combos + 10 held-out-artist corpus
lines + 12 EN words, 2 seeds, T2I, delta scaled 0 (floor) vs 1 (trained),
`conds_cache` cleared per switch. Readout = CER of the largest detector box.

| arm | trainable | steps · lr | job | single CER floor → trained | exact | what the images show |
|---|---|---|---|---|---|---|
| `rows` | delta on the 307 ext rows the captions touch (Qwen-piece keyed) | 2000 · 3e-3 (row-norm units) | `-bb860f` | 1.000 → 1.000 | 0/36 | layout moves at seed 0 (big single glyph in a bubble; ひ→「ま。」, れ→「わ…」, ン→「ズ」, ケ→「チケ」), salad at seed 1; delta grew to 2.4× row norm, loss flat 0.047 → 0.051 |
| `rows_adapter` | rows delta + r16 LoRA on all 60 Linears of `llm_adapter.blocks` | 2000 · 1e-3 / 1e-4 | `-462c9e` | 1.000 → 0.917 | 3/36 (ち×2, ま) | **font-clean single kana every time** — but every row → ら / ん / ち (seed picks); EN control 21/24 |
| **`rows` on 8 chars** (`--only_chars あいうえおかきく`, 40 renders each, 37 rows) | rows delta only (adapter frozen too) | 2500 · 1e-3 | `-2f62a2` | 1.000 → **0.438** | **9/16 both readers** (あ い う え か き く at seed 0; あ か at seed 1; お→「あ…」) | one clean glyph per prompt, the right one; combos 0/36 exact but CER 0.70 (first char often right: ああい→あ, ええ→え, えくく→くくく); corpus lines 0.74; EN 24/24; delta 1.12× row norm, loss 0.036 → 0.034 |
| EN control (no training) | — | — | — | 0.000 both | 24/24 | the pipeline reads; the base spells Latin at 512² under the same clause |

Sheets: `output/wake_probe/rows_few/sheet_single.png` is the one to look at —
floor row of tiny salad bubbles, trained row of あ い う え か き く; and
`rows_adapter/sheet_single.png` for the mode-collapsed perfect ら / ん / ち.

## What W0 settles

1. **The units exist.** Half of the base model's bubble text survives two
   independent readers. Pretraining did factor the bubble texture into
   characters — the "garble" is a marginal over real glyphs, not stroke noise.
   (Signs are a different, texture-only prior; KO/ZH have no agreed units.)
2. **Drawing needs no DiT training.** With the DiT and Qwen frozen, a rows
   delta + a rank-16 LoRA on the adapter produces font-quality hiragana on a
   clean canvas. `plan_render`'s S6 direction ("DiT 본체 LoRA가 맞는 방향")
   is not needed for *capacity*. **Never spend a body LoRA on glyph shape
   again.**
3. **Discrimination is the whole remaining problem.** The 92-row arms
   collapsed to 3 modes: the shared capacity learned the marginal ("one big
   kana in a circle") and a few prototypes. Per-row exposure was ~60 samples
   (8000 samples / 92 chars ≈ 4.6 epochs of 13 renders each), an order of
   magnitude under the textual-inversion regime, and the plain FM loss is
   dominated by layout at high σ — identity is decided at low σ on a few
   percent of the pixels. Three cheap levers, none tried yet: exposure (W1),
   σ-restricted / stroke-masked loss (W2), per-character routing instead of
   Qwen-piece rows (W2).
4. **Rows-only at 3e-3 walks off the manifold** (2.4× row norm, no loss
   payoff). Keep rows ≤ 1e-3 in row-norm units; the adapter LoRA is the
   sample-efficient carrier.

## Phases

### W1 — can rows discriminate at all? **PASSED 2026-09-13**

The textual-inversion regime: 8 hiragana, 40 renders each, 2500 steps, rows
only at 1e-3, adapter frozen. Gate G-W1 was single-kana exact ≥ 4/8 on both
readers, floor 0: **7/8 at seed 0, 2/8 at seed 1, 9/16 pooled, floor 0/16,
both readers identical on every hit.** The address alone is sufficient; a
37-row delta of norm ≈ 1.1× the pack row (≈ 150 KB of parameters) turns
「Japanese text reads as "か"」 into a か. W2 is a scaling + composition
problem, not a capacity or a body question.

What did not pass inside W1, and shapes W2: (a) seed 1 renders the glyph
less often (2/8) — the row wins the layout at seed 0 and loses it to the
manga-prompt prior at seed 1; (b) 2–3-char strings render one glyph, usually
the first (ああい→あ) or a repetition (えくく→くくく) — the eval combos were
held out, but 200 combos were in training and still did not teach order.

### W2 — scale to 92 kana, and make sequences compose

Two problems, keep them separate. **Scale**: W1's per-row exposure was ~270
samples (40 renders × 2500·4/563); the 92-kana arms gave ~60. Matching W1's
exposure at 92 kana is ~11k steps (~2.5 h) — run that first, rows only,
before any fancier loss, because W1 says exposure is the lever. **Composition**:
the DiT composes Latin words from T5 pieces, so it can render a sequence from
a sequence of codes; the question is why 200 training combos did not teach
it. Levers, one arm at a time, ~45 min each on the shipped probe:

- `--t_max 0.6`-style σ restriction (the DiT-scale σ; layout is fixed above
  it) — needs a flag on `fm_training_batch`'s `t_min/t_max`, already exposed.
- stroke mask: `masked_loss` on the glyph's own pixels (font renders carry the
  mask for free; corpus boxes use the bubble box).
- per-character rows: route each kana to its own row (the isoq-style char
  partition) instead of the Qwen-piece rows the shipped pack uses (「いい」 is
  one piece today) — a json edit + `data` rebuild.
- balanced batches: every batch carries 4 different characters in the same
  layout, so layout cancels and identity is the only gradient.

**Gate G-W2: held-out 2–3-char combos exact ≥ 50 % in order, singles ≥ 80 %,
EN control unchanged.**

EN safety (user's question 2026-09-13). The rows delta is bit-exact on any
prompt without an ext id — the hook only touches ext positions, like the
pack. The adapter LoRA is not: it patches every Linear of
`llm_adapter.blocks` for every token, and the W0 `rows_adapter` arm already
moved the EN control 24/24 → 21/24 exact. Any arm that keeps the adapter LoRA
carries, in this order: (1) an **ext gate** — LoRA scale 0 for a sequence with
no ext id (one line in the hook; EN-only prompts become bit-exact), (2) a
position mask — the delta added at ext positions only (mixed EN-tag + JA-clause
prompts still leak through the adapter's self-attn, but far less), (3) an EN
replay term — EN-only caption batches with an MSE to the stock adapter's
output, for the mixed-prompt case the product actually uses. The W2 gate's
"EN control unchanged" is measured on the mixed clause, not EN-only prompts.

### W3 — the product condition (this is the "OCR-enhanced training" payoff)

Back onto `plan_render`'s task with the woken text side: EasyControl bubble
fill (S1 caches, JA edition), cond LoRA as JA-SHIP, **plus** the W2 rows +
adapter LoRA loaded frozen or co-trained — no target-stream LoRA. The
render judge unchanged (R1 / R2 rulers). This is where the line's captions
stop being inert: once a text clause is an address the DiT can read, the
OCR-quoted captions (`Japanese text reads as "…"`) become supervision for
every training run, not just a spam suppressor — the s21 "rows are addresses,
content is inert" verdict was measured on a DiT that could not read them.
**Gate = plan_render G2 (R1 ≤ 0.5, R2 ≥ 0.7) on JA-BODY's judge set.**

### W4 — kanji

Capacity question, not address: 2,136 jōyō against a frozen DiT whose agreed
bubble units already include kanji (Probe 0). Same probe, `--only_chars` on
the top-200 corpus kanji first. Not before W3.

## Kill criteria

- ~~W1 fails~~ — W1 passed; the address discriminates. The glyph-image path
  (AnyText / GlyphControl shape) is now the fallback for *composition* only:
  if W2's sequence arms all render one glyph, order is not learnable from the
  text side at this scale.
- W2 passes and W3 sits at plan_render's floor → the bubble-fill task itself
  (holed cond, whole-crop loss) is the blocker; move the woken text side to
  T2I manga-panel generation and re-judge there before killing.

## Instruments and gotchas

- `probes/wake_probe.py --stage salad|data|train|eval --arm rows|rows_adapter
  [--data_tag few --only_chars …]`; every GPU stage via `make daemon-run`. Data
  stage is CPU-only (~30 s) and safe inline.
- **Read the largest detector box, not the whole image** — the whole-image
  read of a 512² T2I picks up the tiny background bubbles the manga prompt
  also produces; `report.md`'s table is the whole-image/min-over-boxes score,
  the largest-box readout is what this plan quotes (recompute from
  `eval_reads.json`).
- Ext rows are **Qwen-piece** keyed (the pack json's `qwen` map): 「いいよ」 is
  two rows (いい + よ). A "per-character" claim needs the char partition.
- The VAE takes `IMAGE_TRANSFORMS` range ([-1, 1]); its docstring says [0, 1].
- `generate()` caches post-adapter `crossattn_emb` by prompt in
  `shared["conds_cache"]` — clear it when the delta/LoRA scale changes (the
  render judge learned this the hard way).
- Rows lr is in **row-norm units** (mean pack-row norm 195.7); 3e-3 blew the
  delta to 2.4× the row norm by step 2000.
- Reader agreement is the unit test for "real glyph"; a single manga-tuned
  reader says 85 % of salad boxes contain CJK.
- Costs on the 5070 Ti: salad 8 min; 92-kana train 28 min (1.2 it/s at 512²,
  batch 4, grad-ckpt) + eval 15 min + reading 3 min.
