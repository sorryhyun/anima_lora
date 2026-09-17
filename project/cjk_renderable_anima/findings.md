# cjk_renderable_anima — findings

What this line has settled, in the form a later decision needs: the verdict,
the number it turns on, and the mechanism note that exists nowhere else. One
screen per topic, no chronology. The dated run record is
[`reports/`](reports/README.md) (indexed: W2d Runs 1–3, the order probe, the σ
diagnostic, the strings arm, the S line); the forward plan is [`plan.md`](plan.md); the
W0–W2 report is `reports/wake_w0_w2_2026_09_13.md`. The predecessor lines'
verdicts are read-only in
[`../cjk_aware_anima_dit/findings.md`](../cjk_aware_anima_dit/findings.md)
(DiT side, OCR readers, captions) and
[`../cjk_aware_anima/findings.md`](../cjk_aware_anima/findings.md)
(the vocab pack itself).

Line status (2026-09-15): frozen DiT + frozen adapter + a delta on the
pack's ext rows renders every kana and short common words (Run 3), and a
static table trained on strings carries **order and count** (strings arm).
The S line (scene composites + a per-source layout vector `c_flat`) put
the mechanism on real scenes (S0 hit & kept 15/64 from P0b's 2) but no arm
has cleared the native gate; the blocker is now named — the *render
trigger* and the *canvas* are both common to every flat item, so one
shared vector takes both (see *Settled — trigger vs canvas*). Nothing is
shipped; the cap-only isolation run (`s24k_S0b_cap075`) is the live arm.

---

## Rulers — read before comparing any two numbers

- **Renders**: 512², 28 steps, cfg 4, seeds 0/1, the trained clause
  template `Japanese text reads as "…"` on a manga bubble or plain prompt
  (`TPL_BUBBLE` / `TPL_PLAIN`); EN control `English text reads as "…"`.
  **Never below 384²** — the base cannot spell even EN at 256² (11/24) but
  spells at 384² (24/24), where the 512² band rows read 21/36 and identity
  still peaks at σ 0.8 (2026-09-14 gate; `rows_w24_band/eval_384/`,
  `classify_384/`). Canvas shape is a recipe lever, not a ruler: Run 3's
  recipe on a 384–512 mixed pool (P0a, `encoder_wds_w120_s8k_fres_warm_shp`)
  gave singles 36/36 sfx at 512² (Run 3 33/36) and 34/36 at 384×512 at
  0.82× the wall; eval stays at 512² unless a shape is the question.
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
  (`src/probe/order_probe.py`). There is no T5 encoder in the loop; the
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

## Settled — trigger vs canvas: why the S-line tables wipe or go silent

The S line splits every ext row into `f_r` (per glyph) + `𝟙[flat]·c_flat`
(one vector, on for flat-canvas items only). Four measurements, 2026-09-15,
fix what each part holds — read them before touching the cap, the init or
the data mix.

- **`c` holds the canvas *and* the "render text here" trigger; `f` holds
  identity and (depending on the cap) part of the trigger.** Native, EN
  clause, 64 renders, hit / kept / hit & kept: S0 (cap 0.75) `f` alone
  25 / 48 / 15, `f+c` 31 / 29 / 10; S0b (cap 1.5, one flat layout,
  `--scene_fill 0.7`) `f` alone **1** / 50 / 0, `f+c` 25 / 24 / 9; P0b
  (flat-only hybrid) every part without its common vector 0/16. Raising
  the cap 0.75 → 1.5 moved the trigger out of `f` into `c` (S0b's leak
  fell 0.28 → 0.18 and every flat ruler got worse: singles 20 → 15, word
  3 → 1). **A larger cap = more absorption, not less leak.** Mechanism:
  `c` is switched on for flat items only, so it can absorb anything
  common to *all* flat items — canvas and trigger alike; only the
  composites show the trigger without the canvas, so the composite share
  and glyph size decide what `f` is forced to keep, the cap only decides
  what `c` is allowed to take.
- **In row space the two are already separate.** Rows are mostly ⊥ to
  `c` (mean cos 0.18–0.28), `c` is ⊥ to the stock T5 table mean (−0.07)
  and lies outside its top-256 PCs (energy 0.30, gaussian 0.25, real rows
  0.45). What is entangled is the DiT's *response*, not the parameters —
  no row-space regulariser can split it.
- **A warm start from P0b's rows saves identity, not exposure.** With
  `--init_rows` (rows = P0b raw − its common vector, `c_flat` seeded with
  that vector, cap 0.75), 2 k composite steps at lr 3e-4 keep singles at
  25/36 but leak stays flat at 0.27 and native `f` alone is **0/64** hit,
  61 kept; `f+c` 8 / 32 / 3. Flat items are satisfied at step 0, so only
  the composite gradient moves `f`, and at 40 % share with small glyphs it
  builds no trigger in 2 k steps. (lr 3e-3 on the same warm start halves
  the row norm by step 100 and leaves an inert table: singles 1/36.)
- **The pretrained model has a canvas-free text trigger, and it is not a
  drop-in for `c`.** At the adapter output, the shift an EN token gets
  from being quoted (`reads as "…"`, `speech bubble that reads "…"`, `she
  is saying "…"`, bare quotes) is one shared direction Q (per-word cos
  ≈ 0.5, cross-frame 0.73–0.91); `c`'s image is ⟂ to it (−0.07). Adding Q
  at the ext positions of `f` alone (`--out_vec`, native conds `fq<s>`):
  S0 rows 25 / 48 / 15 → 13 / 54 / 9; S0b rows 1 / 50 / 0 → 4 / 54 / 2.
  Q removes the wipes and restores the scene where `f` had blanked it, but
  pushes the DiT into its subtitle "small text line in a scene" mode, so
  the single big glyph the exact ruler needs shrinks or gets embedded
  (target-in-any-read unchanged at 44 → 46; か 14 → 4 hits, ぐ 0 → 3 with
  dakuten). Untested and the only remaining use: Q fixed on *during
  training* so `f` learns identity that renders under it.
- **A table trained under the frames does not converge on Q.** The 53k
  full-inventory rows (no `c_flat`, four caption frames, Q off) image at
  the adapter output with cos −0.02 … −0.03 to every frame's Q (`She is
  saying "…"` included; random 0.025), 0.05 energy in the EN-quoted-code
  subspace (untrained row 0.10), and move the code away from the EN word
  cluster (0.42 → 0.31). The frozen adapter adds the same frame shift to a
  trained and an untrained row (vs Q 0.35–0.45 both; EN words 0.73). The
  rows' shared direction (18 % of the table's energy, = the S0/S0b row
  mean, ≠ their `c_flat`) is the self-built trigger and is ⟂ Q; hit
  tracks it (+0.32). Residual geometry is near-orthogonal (pairwise cos
  0.03) with a weak small-mark neighbourhood (ば↔ぱ pct 98, か↔が 92,
  あ↔ア / kanji-component random). `reports/rows_manifold_2026_09_16.md`.
- **Identity is conditional on the trigger it was trained with, and only
  composite training makes it modular; inheriting the trigger buys no
  steps.** Transplant probe (`src/probe/transplant_table.py`, no training):
  a flat-only donor's residual (P0b 24 k, Run 3 8 k; own shared direction
  projected out) on the 53k m̂ renders 0/64 in scenes at any scale, the
  same as m̂ alone (scene kept, floor garble), while the composite-trained
  micro6 residual on the same m̂ renders 23/64 with the scene kept. Then
  `--pin_dir` arms (53k m̂ frozen per family, residual only trained, ⟂
  projected) on the micro6 frame-mix data: `en` 54 = 54 at 2 k and 41 vs
  39 at 500 against from-scratch, curves overlapping; `swap` 20 vs 9 at
  500 but 32 vs 46 at 2 k. m̂ alone scores the best scene / placement of
  any cond (en cos 0.920, IoU 0.31) — the residuals cost scene, not the
  trigger. `reports/transplant_2026_09_16.md`.
- The artist-handle mode (`@greatdoggo` → logo) is a third, contextual
  direction, ⟂ to Q and to every kana code — not a "draw a fixed mark"
  address to borrow.

- **The wipe is not the delta norm alone, and it is what hides the base's
  JA pseudo-text** (ΔFM Δ0 / Δ0b, 2026-09-17, 12 dakuten rows). Paired-
  difference FM (`--pair_loss 1`: the sibling's residual under the same ε, σ
  subtracted) cancels ≈ 80 % of the residual — the scene part — and at
  equal row norm (144 vs 149) the native keeps its scene: en cos 0.935 vs
  0.855, images under 0.85 6 vs 20 of 64. The glyph then lands as scene
  text **beside lines of JA pseudo-text**, reader hits halve (joint hit ∧
  en cos ≥ 0.85: 15–20 vs plain 28). That pseudo-text is in no
  teacher-forced residual (an EN-frame sibling changes the paired loss by
  nothing); it is the base free-running under `japanese text`, and plain
  FM removes it only by wiping the scene with it. Wipe ↔ co-text is one
  axis on single-glyph data: a σ split of the two losses interpolates
  along it. `reports/synth_pair_2026_09_17.md`.

## Settled — what does not move it

- 256² training under the default σ (dead: EN 11/24 there; 384² is alive
  and the band is the same, so this is one tier, not "below 512²");
  balanced batches (FM
  loss is a batch mean, free rows only see their own items); rows lr 3e-3
  (walks off-manifold at 2.4× row norm); `--t_max 0.6` (backwards —
  identity is above it); adapter LoRA (`rows_adapter`: drifts EN, kept as
  the negative control); same-noise CE / swap hinge on free rows (fixes
  nothing combos-related); regressing rows onto existing embeddings or
  Latin letters; encoder-only generalisation (rank-1 table under every
  lever); IDS composition of addresses; the `c_flat` cap in either
  direction (0.75 → 1.5 handed the trigger to `c`); a P0b warm start as a
  shortcut through the composite stage; Q added at inference over rows
  trained with a `c`; ΔFM as an exposure saver (no point beats plain FM at
  equal draws), and reweighting ΔFM against plain FM (`--pair_sigma_min`)
  or moving the sibling's caption frame (`--pair_ref_frame en`) as a way
  to keep the scene *and* drop the pseudo-text.

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
- Any hook on `llm_adapter.embed` that must see ext ids registers with
  `prepend=True` — the vocab pack's clamp pre-hook rewrites them to
  `<unk>` first; the first `OutVec` native pass was silently inert
  (fq renders byte-identical to `f`), caught only by an md5 compare.
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
- Raising `c_flat_cap` (or removing it) to "let the vector settle" — S0b
  measured the direction: the trigger follows the room.
- Seeding rows or `c` from EN / quoted-EN codes, or a row-space
  "canvas-component" regulariser — the split is already clean in row
  space; the coupling is in the DiT's response.
- Q as an inference-time replacement for a trained `c`.
- Flat-only (or flat-heavy) exposure for new glyphs with the trigger
  supplied (transplant: 0/64 for every flat donor), and pinning /
  seeding the shared direction as a step saver (`--pin_dir`: neutral on
  `en`, worse on `swap` at convergence) — identity is the cost and it is
  per row.

## Open

- **Is the exposure budget the lr integral?** ΔFM at `--lr_rows 2e-3` ×
  1 500 steps ≡ 1e-3 × 3 000 (19 vs 18/24, same native read); every S-line
  exposure number was taken at 1e-3 cosine, so draws and ∫lr were never
  separated for plain FM (3e-3 stays closed). `plan_synth2.md` L0.
- Does the ΔFM co-text need unaddressed text — i.e. does it vanish when
  the caption is a whole sentence of trained rows (`plan_synth2.md` Δ2)?
- Mixed arm gate (plan P1): singles ≥ 30/36 back with `flip` / `str3` /
  `line` at or above the strings arm.
- Repeat mode: whether repeated-piece strings as negatives stop the second
  slot copying its neighbour.
- Does contextualisability transfer to rows added later with singles-only
  exposure (the "2-step once vs per batch of characters" question)?
- Pack bake: `trained.pt` → shipped safetensors + json + digest; the full
  kana inventory (voiced, handakuten, small kana ≈ 70 rows) and the
  common-word pack at the kana bar.
- Cap-only isolation (`rows_synth_s0b_s24k_S0b_cap075`, S0b data, cap
  0.75): attributes S0b's `f` collapse to the cap vs `--scene_fill 0.7`;
  reads `f` alone hit vs S0 25 / S0b 1.
- Q as a *training-time* trigger: rows trained with the quoted-EN
  direction fixed on for every item, `c_flat` at 0.75 for canvas only —
  does `f` then carry identity that renders under Q in a scene? Needs the
  `OutVec` hook in the train stage; not before the isolation run reads.
- Composite share 40 → 60 % and `--scene_min_box` 56 → 72 (plan_synth
  tree) once the cap is attributed.
