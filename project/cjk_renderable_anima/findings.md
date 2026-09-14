# cjk_renderable_anima — findings

What this line has settled, in the form a later decision needs: the verdict,
the number it turns on, and the mechanism note that exists nowhere else. One
screen per topic, no chronology. The dated run record is
[`plan_wake.md`](plan_wake.md) (W2d Runs 1–3, the order probe, the σ
diagnostic, the strings arm); the forward plan is [`plan.md`](plan.md); the
W0–W2 report is `reports/wake_w0_w2_2026_09_13.md`. The predecessor lines'
verdicts are read-only in
[`../cjk_aware_anima_dit/findings.md`](../cjk_aware_anima_dit/findings.md)
(DiT side, OCR readers, captions) and
[`../cjk_aware_anima/findings.md`](../cjk_aware_anima/findings.md)
(the vocab pack itself).

Line status (2026-09-14): frozen DiT + frozen adapter + a delta on the
pack's ext rows renders every kana and short common words (Run 3), and a
static table trained on strings carries **order and count** (strings arm).
Nothing is shipped; the mixed arm (plan P1) is the next gate.

---

## Rulers — read before comparing any two numbers

- **Renders**: 512², 28 steps, cfg 4, seeds 0/1, the trained clause
  template `Japanese text reads as "…"` on a manga bubble or plain prompt
  (`TPL_BUBBLE` / `TPL_PLAIN`); EN control `English text reads as "…"`.
  **Never below 512²** — the base cannot spell even EN at 256² (11/24).
- **Reading**: AnimeText detector boxes, both readers (SFX reader `sfx`,
  PaddleOCR-VL `vl`); `report.md` scores the best box per image, the
  per-item tables quote the **largest detector box**. `exact` is the
  normalised sfx read equal to the reference (NFKC, casefold, punctuation
  stripped). The readers under-read kana (力/カ, つ/っ, き こ し at large
  size are vl-unprintable) — a "miss" on a clean glyph is often the reader;
  look at the sheet before counting.
- **`--no_floor`**: the delta-off renders are identical across arms (JA
  CER 1.000, singles 0/16, EN 24/24) — the W0/W1 numbers are the floor.
- **Classifier stages** (`classify`, `classify_str`): same-noise diffusion
  classifier, summed FM error per latent, right = argmin; never pool errors
  by raw sum over σ (σ 0.95's spread buries the band).
- **Coverage**: `eval_coverage.json` — a caption whose tokenizer merges the
  target into a larger piece misses the row; every string verdict here is on
  strings with coverage 100 %.

## Settled — the address exists; the glyphs are in the DiT

- **The frozen DiT holds JA glyph units.** A rows-only delta on 8 hiragana
  (37 rows, 2 500 steps) renders the requested kana 7/8 at seed 0; EN
  bit-exact. Identity is decided at **σ ≈ 0.8** (classifier top-1 19/48
  there, chance at σ ≤ 0.65 and 0.95); the σ band 0.7–0.9 lifted 24-kana
  singles 10 → 24/36. Every W1/W2 arm and Run 3 confirm it.
- **Trained addresses are near-orthogonal random directions**, not shape
  coordinates: pairwise cos 0.04, energy in the stock table's top-256 PCs
  0.31 (gaussian 0.25, real T5 rows 0.45), cos to own pack row −0.05.
  Adapter ≈ linear on them (corr(row-cos, out-cos) 0.73). There is no
  manifold to fit without the DiT.
- **Held-out generalisation is zero on every encoder lever.** W2d shared
  glyph encoder `g`: data jitter, random head init, decorrelation (PR 35,
  pixels lost 4/24), hinge — held-out kana 2–5/64 each; IDS composites
  0/16 (each renders as a clean unrelated hiragana). The hybrid
  `Δ_r = g(x_r) + f_r` renders every *trained* row (Run 1d 24/24; Run 3
  34/36 over all 92 kana in a 246-row table) with `g` reduced to a prior
  (`free_ratio` 0.90). **Identity lives in `f`; every row needs its own
  exposure** (≈ 40 renders/row; words with `f` from zero reached 9/32).
- **One ext row can be a multi-glyph word.** します / してる (three glyphs)
  render from one Qwen-piece row on both seeds; きた こう いや もう ッド on
  one. The DiT reads a row as a *unit*, and the unit can carry a short
  word's layout. Held-out words 0/16 — `g` has no word prior.
- **Kanji are an exposure budget, not a research line.** Atoms render
  (10/12 at kana-level exposure); composites decompose into their parts
  (林 → 木, 森 → 木林) and combos fuse at the component level in pixels.
  jōyō 2 136 ≈ one GPU-day in one table; repeated-atom composites leak
  (Run 2 repeat mode) and must not be over-weighted.

## Settled — sequences: the DiT reads order, and a static table can carry it

- **Order reading is a pretrained capability of the frozen adapter + DiT.**
  The wake EN control was multi-piece all along (HELLO = ▁H·ELL·O). Base
  model, no delta: nonsense 4–5-piece words (GLORPAX, MIZUKANE) 22/24
  exact, two-word swaps 6/6 with "WAY NO" in the *given* order
  (`probes/order_probe.py`). There is no T5 encoder in the loop; the
  T5-side table is the adapter's query vocabulary, the adapter (6 blocks,
  self-attn + Qwen cross-attn, **both RoPE'd**) bakes position into each
  vector, and the DiT's cross-attention — which has no text-side PE — reads
  the set. "T5 is position-unaware" is the wrong frame.
- **Where σ decides.** EN two-word classifier (`classify_str --cls_lang
  en`): order / count / identity all peak at **σ 0.65** (order win 0.98,
  gap +2.5 spread units), live over σ 0.5–0.8, and are **absent at
  σ ≥ 0.9**. The "one-piece caption wins at σ 0.95–0.99" pattern appears
  for EN too — a near-pure-noise artefact, not a collapse decision. Kana
  rows trained on singles carry identity/count at 0.65–0.9 and **no order
  signal at any σ**. The `--t_min/--t_max` sampler is a hard affine remap
  (sigmoid density inside, zero outside): the band *is* the weighting, and
  for strings it moves **down** to 0.5–0.9, never up.
- **Rows trained on single-unit canvases enumerate nothing** — every
  multi-row caption renders exactly one unit (Run 3 `line` 0/32 with
  coverage 39/39; kana combos → the first/strongest; kanji combos fuse).
  This is a data artefact, not a DiT limit: the delta carries "one
  centred unit" and nothing asked it to be contextualisable.
- **Rows trained on strings carry order and count through the frozen
  adapter.** Strings arm (Run 3 table warm-started, `g` frozen, 6 000
  random 2–4-piece strings + 132 covered corpus lines, no singles, band
  0.5–0.9, 8 000 steps): `flip` pairs both pieces present 25/48, **first
  rendered glyph = caption's first piece 28/48 vs its last piece 5/48**;
  the same two rows swapped change the picture (こユ / ユこ both exact,
  くル→くくル vs ルく→ルくル). Unseen 3-kana strings 3/16, `flip` 4/48,
  `line` 2/32 (from 0). Composition generalises across strings of trained
  rows — no per-string exposure. The "frozen adapter cannot contextualise
  off-manifold rows" branch and the DiT-side W3 are **not needed** for
  order.
- **Unit count is a data-distribution prior that lands in the rows.**
  Singles-only rows draw one unit; strings-only rows draw several even for
  a one-kana caption (singles 34 → 5/36: ひ→ひひ, ケ→ケケケ, チ→チテチ —
  identity kept, extras added). Symmetric artefacts of the same mechanism.
  The count must be predictable only from the number of ext tokens in the
  caption → one mixed distribution, not a two-step curriculum.
- **The dominant string miss is repetition** (ううう, ねねね, ををら, はめめ):
  count right, second slot copies a neighbour. It is Run 2's repeat mode
  on the string side and the open lever after the mixed arm.

## Settled — what does not move it

- 256² training (dead: no text competence there); balanced batches (FM
  loss is a batch mean, free rows only see their own items); rows lr 3e-3
  (walks off-manifold at 2.4× row norm); `--t_max 0.6` (backwards —
  identity is above it); adapter LoRA (`rows_adapter`: drifts EN, kept as
  the negative control); same-noise CE / swap hinge on free rows (fixes
  nothing combos-related); regressing rows onto existing embeddings or
  Latin letters; encoder-only generalisation (rank-1 table under every
  lever); IDS composition of addresses.

## Gotchas that cost time

- The promoted `wake_probe.py` needs `project/cjk_aware_anima_dit/ocr` on
  `sys.path` for `pseudo_label` (the readers stayed in the frozen line);
  the promotion dropped the line and every eval died at the reader —
  fixed 2026-09-14.
- Ext rows are **Qwen pieces**: にな, して, った are single word rows, so a
  kana "pair" is only an order contrast when *every permutation* tokenizes
  to its own kana rows (`_clean_kana_strings`); `_string_pairs` checks it.
- Clear `conds_cache` on every delta-scale switch; the VAE takes [−1, 1];
  block compile before grad-ckpt; batch 8 OOMs at 512² even compiled;
  activation budget 0.99; `--stall-timeout 0` on every daemon job (the
  eval's reader load is a long quiet phase).
- A background `daemon-wait` client from the agent harness gets killed on
  low host memory — the daemon job survives; wait with a Monitor poll on
  `output/daemon/jobs/<id>/job.json` `state`.
- Warm-started `f` dips (0.78 → 0.52 in 25 steps at lr 1e-3, Adam moves a
  row ≈ 0.03 row-norm units/step) and re-settles by step ~2 000; do not
  read the dip as identity loss.

## Do not re-propose

- A shape encoder, g-geometry or hinge as a route to held-out kana/kanji.
- Composing addresses (Δ_a + Δ_b, IDS sums) — the DiT composes in pixels,
  not in row space.
- DiT-side or adapter-side LoRA *for order* — the static table carries it.
- A hard two-step curriculum (singles then strings) — each step bakes its
  unit count into the rows; mix instead.
- Widening the σ band above 0.9 — nothing is decided there.

## Open

- Mixed arm gate (plan P1): singles ≥ 30/36 back with `flip` / `str3` /
  `line` at or above the strings arm.
- Repeat mode: whether repeated-piece strings as negatives stop the second
  slot copying its neighbour.
- Does contextualisability transfer to rows added later with singles-only
  exposure (the "2-step once vs per batch of characters" question)?
- Pack bake: `trained.pt` → shipped safetensors + json + digest; the full
  kana inventory (voiced, handakuten, small kana ≈ 70 rows) and the
  common-word pack at the kana bar.
- Scene prompts (`native` stage): the rows' layout prior in a real
  composition; not re-measured since the string arms.
