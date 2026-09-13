# CJK-aware Anima, DiT side — plan (archived 2026-09-08; reopened lines folded back 2026-09-12)

The DiT-side line is frozen. Every plan file this directory carried —
`plan.md`, `plan_base1.md`, `plan_ocr.md`, `plan_det.md`, `plan_ssl_tower.md`,
and the three that outlived the freeze (`plan2.md`, `plan_vl_respace.md`,
`plan_render.md`, archived 2026-09-12) — moved verbatim to
`_archive/cjk_aware_anima_dit/plans/` (gitignored tree,
preserved in the private mirror; the pre-move versions are in git history).
The dated `reports/` moved with them to
`_archive/cjk_aware_anima_dit/reports/`, so a `reports/…` link from
[`findings.md`](findings.md) resolves against that archive, not against this
directory. Each plan carries a status banner written at archive time, so the
archived copy says what ran and what did not.

What stays live here: [`findings.md`](findings.md) (every settled verdict, and
the only file to read before reopening anything), `ocr/`, `probes/`, `render/`,
`assets/`, and § Still open below — the open items of the two reopened lines,
in one place so nothing is carried only in an archived file.
The shipped surfaces are untouched — the SFX reader
(`sorryhyun/paddleocr-vl-1.6-manga-lora`, wired as `anime_tools.ocr.sfx`), the
AnimeText detector defaults, and the D1 quote-partitioned pack.

## Why frozen

The line was two goals and a side line; the side line is what delivered.

Shipped (the OCR stack). A reader that reads hand-lettered onomatopoeia —
arm B′, a PaddleOCR-VL-1.6 LoRA with the vision tower unfrozen, which took
the sincos SFX gate from stock 6 to ~50 % and COO test to 81.7 % against the
published 81.2 %. Plus `deepghs/AnimeText_yolo` as the detector (floor 38 → 4
on sincos' masked pages), the O4c–O4e caption rules (SFX/speech dedupe, glyph
floor, the `…`/`♡` normalisation that stopped the guard eating the longest line
on the page), and PP-OCRv6 retired to the explicit torch-free pair. D1 shipped
too: the deterministic isotropic table, the pack's quote-route partition, and
`ss_ext_pack_sha` on every LoRA.

Not answered (the DiT goals). D2–D6 never ran. G-A — manga trains
healthily unmasked at *corpus scale* — was measured only on `sincos` (351 of
the 873 text-masked images), where the shipped caption clauses cost nothing and
win nothing: three blind reads, three ties, pooled 35–36 on 71 decisive pairs.
G-B — a LoRA learns CJK semantics for isotropic addresses — is
unmeasured; D5a, the decisive experiment of the hypothesis, was never
executed. The paired-edition corpus never entered: alignment stopped at 2 of
240 works, so arm T and the contrastive address are designs, not results.

Where the remaining headroom is, measured. On the reader: the label side,
not the representation side. Five arms in a row moved in-domain COO and left
the doujin gate flat — colorized-COO at 1.6 %, the col1500 swap, ×3 epochs,
LP-FT, and label-free tower SSL — while B′'s misses are 55/313 ♡-only against
♡ in 497 of 617 gate rows. On the DiT: nothing was refuted, only unrun; a
reopening starts at D2's census, not at a new instrument.

Read [`findings.md`](findings.md) § Label basis first — the sincos gate changed
units twice (PP-box → AnimeText, then 619 → 617 rows), and **no `/ 71` figure
converts to a `/ 617` one.**

## Still open (2026-09-12)

Three lines outlived the freeze and are now archived with the rest. What they
*settled* is in their closing banners and in [`findings.md`](findings.md); what
follows is only what they did not answer, ordered by cost.

### The VL reader (`plan_vl_respace.md`)

The shipped v3 reader spaces and folds (R5b closed below). Open, cheapest
first:

1. ~~**R5b — flip the heart fold.**~~ **Done 2026-09-13 — shipped as Hub v3.**
   `vl16_b2_norm4` (`TARGET_NORM = 4`: `textnorm.training_target`, heart as
   the single-token `♥` in the training target only; `exact_key` and
   `normalize_read` keep `♡`), B′'s argv, seed 0, a pure A/B on the fold
   against norm3. On the corrected sincos labels: ♡-blind **354** / 617 (norm2
   346, norm3 323, B′ 365), strict 305 (norm2 294), raw `♥` back in 322
   predictions (norm3 0); COO SFX 2147 / speech 2254 (norm2 2138 / 2255), val
   86.8 % (jitter), and the dot fold keeps COO runaways at 31 (norm2 191).
   Every gate cleared, so the heart token *was* the whole R5 story. Hub
   `26292839`; `eval.md` carries norm3 + norm4 rows. The seed repeat stays
   unbought.
2. **The label pass.** 12 sincos SFX rows were corrected 2026-09-12, all 12
   drawn from the arm's *lost* sheet, so the pass is one-directional by
   construction (B′ 375 → 365, norm3 319 → 323). Owed: the same check on the
   31 *won* rows, and a random sample of the untouched 499 to put a number on
   the basis's label-noise rate. Then `eval_sfx.py` re-runs for all three arms
   — the stored `sfx_*.jsonl` carry `text`, so `eval_table` cannot see a label
   fix and `eval.md` sits on the pre-fix basis until it does.
3. **R3 — the CJK space rule, never written.** `TARGET_NORM = 2` teaches a
   space at every balloon line break, so v2 puts spaces inside Japanese
   speech. Drop a space whose two neighbours are both CJK, keep it next to
   Latin / Hangul / digits, in `anime_tools.ocr.sfx.normalize_read` (which
   today only collapses runs) — not in training.
4. ~~**R4.2 — the catalog will not re-fetch v2.**~~ **Fixed upstream
   2026-09-13** (anime_tools `54cdbc5`): `Asset.revision` passes through to
   the Hub fetch and a `dest` row stamps `<dest>/REVISION`; `missing()` reports
   every file until the stamp names the pin, so a pre-pin install re-fetches
   instead of reading v1/v2 under a v3 name. `sfx_reader` pins `26292839`
   (v3). Reaches this checkout when v0.7.1 is tagged and the pin moves.
5. **R4.4 — the tree re-run.** OCR stage over the 3 008 pages on v2 + R3,
   `probes/ocr_merge_sheet.py --baseline_dir` against the v0.6.2 sidecars;
   `kukiyuusha/13573906` is the smoke.
6. **The cure, not the symptom: heart positives.** R5b only stops the bleeding.
   Train SFX carries a heart on 0.22 % of rows against the gate's 80.55 %, and
   the model has never once seen `びく♡`. The only licence-clean source is
   synthetic — `ocr/synth_sfx.py`, still unwritten, parked since R2b and now
   with a second reason to exist. Append small, never swap (col100's lesson).

Not open: R2b (the synthetic English set — R2 passed its English half), and a
seed-1 repeat of `vl16_b2_norm2` (superseded; if a repeat seed is bought, buy
it for R5b).

### Render — can the DiT write JA from ext rows (`plan_render.md` → [`plan_wake.md`](plan_wake.md))

Three arms ran. EN clears R1 and not R2; JA-SHIP sits at the floor for a
structural reason (nothing trainable on the ext-row → pixel path); JA-BODY,
built to remove exactly that objection, sits at the floor too.

**Reframed 2026-09-13 — [`plan_wake.md`](plan_wake.md) is the live plan.**
The user's reading: the DiT learned to *garble* because the T5 side never
gave it a CJK address, so the aim is to wake weights, not train a script.
W0 confirmed both halves that matter: half of the base model's bubble salad
survives two independent readers (real units), and a **frozen DiT** draws
font-clean kana from a rows delta + `llm_adapter` LoRA alone. What is open
is row discrimination (the 92-row arms mode-collapse). Consequences for the
items below: EN-BODY no longer gates the interesting question (it still
settles whether the *body recipe* breaks rendering, nothing more), and no
future arm spends a target-stream LoRA on glyph shape.

1. **EN-BODY — the one run that closes the line.** A `render_en_body`
   descriptor (stock pack, so no `train_ext_rows`; same target-stream and
   `llm_adapter` LoRA, `target_lr` 1e-4, budget 0.3) on the S1 EN caches, 4
   epochs ≈ 80 min, judge ≈ 30 min. **G0′: R1 ≤ 0.3 and no regression against
   EN e4's 0.254.** G0′ passing beside G2's floor is the clean negative the
   whole plan exists to produce — a trainable body with the clause in front of
   it still does not read ext rows. G0′ *failing* says the body recipe breaks
   rendering outright and the JA floor means nothing, which is why the JA
   number cannot be written up before it.
2. **S5′ — the write-up**, conditional on 1: a new § (render) in
   [`findings.md`](findings.md), a dated report, and the closed-lines memory.
3. Parked behind a JA arm that leaves the floor, i.e. probably never:
   JA-BODY-RAND (SHIP vs RAND on a body that can learn), the KO arm, EN-CLEAN
   (ring-fill — the product condition, where placement and line-wrap are
   actually learned), and a tagger pass on the caption bag.
4. **Independent of all of it**: S0b's residue. ~11 % of holed boxes carry a
   garbage clause because every accepted brush-lettered box OCRs to noise, and
   nothing image-side separates it cheaply. The lever is reader disagreement —
   a second read (hayai) of the accepted boxes only, rejecting a box whose two
   reads disagree past a CER threshold. A short daemon job, never built. **A
   G1/G2 miss must be re-run behind this pass before it counts as the clean
   negative.**

### KO/ZH pseudo-labels (`plan2.md`)

The mining works and the arm ran; the gate that would score it does not exist.

1. **K3 — a KO/ZH gate set.** Hand-label KO/ZH crops from K1's `kscreen` hits
   into the sincos schema (`text_rec` = stock's read, `status = draft`), split
   gate / calibration. The calibration half also buys the first precision
   number for the kept pseudo rows per voter, which no run on this line has
   measured. Until it exists, `vl16_pl_kozh` has only a Japanese-side cost
   (−11 sincos, −22 COO SFX against `vl16_pl_20k`) and no measured benefit, and
   **no KO/ZH claim can be made either way**.
2. **Then re-read K4 run 1 against it** — the arm is already trained and its
   eval rows are on disk; K3 turns an unanswerable run into a verdict without
   buying another 1.5 h.
3. **The ceiling is licensing, not accuracy.** The AnimeText pool is
   CC-BY-NC-SA, so anything K3 certifies stays research-only. A shippable KO
   reader needs a licence-clean source — the same gap `synth_sfx.py` fills for
   hearts.

## Do not re-propose

Carried out of the archived plans so the list survives the gitignored tree.
Each has its evidence in the plan named beside it.

- **Do not size or screen a KO/ZH pool with B′** — it misreads both as
  Japanese, and a B′-selected arm cannot then score B′ (plan2 K0's `zh` row is
  exactly that confound). Do not re-run K0's `zh` arm off a B′ screen.
- **Do not label KO/ZH from `JustANormalTinkerer/animetext-ocr`** — its reads
  *are* hayai's, so it is the voter, not a teacher, and its Korean carries no
  띄어쓰기 (plan2 K1).
- **Do not judge a KO/ZH arm on the Japanese gates alone**, and do not re-run
  K4 run 1 expecting a different Japanese number. Build K3 first (plan2).
- **Do not ship `vl16_pl_kozh` or `vl16_pl_20k`** — NC pseudo rows
  (plan_vl_respace, plan2 § Constraints).
- **Do not patch spacing at inference** (a word segmenter, a stock re-read of
  Latin lines) — the user chose the retrain 2026-09-10 (plan_vl_respace).
- **Do not judge a spacing arm on `exact_key` alone**; it is whitespace-blind
  by design and cannot see the regression (plan_vl_respace R1).
- **Do not reintroduce a whitespace strip in training targets** for any VL arm
  (plan_vl_respace, `whitespace_fixed.md`).
- **Do not pick a target spelling without checking the tokenizer first** — R5
  lost 49 heart reads by folding the single-token `♥` into byte-fallback `♡`
  (plan_vl_respace R5).
- **Do not judge a heart-slot change on the COO columns** — at 0.31 % heart
  rows they cannot see the axis; only the sincos gate can (plan_vl_respace R5).
- **Do not read an EasyControl-only floor as a pack verdict.** Its target
  stream is the frozen DiT and `cache_llm_adapter_outputs = true` keeps the
  `llm_adapter` out of the graph, so such an arm has **no trainable weight
  between an ext row and a pixel** (plan_render S5).
- **Do not compare a render CER to any `/ 617` or `/ 71` figure** in
  `findings.md` — different units, different task (plan_render).
- No prompt hints or language tags for the reader; no SSL-tower arms; no
  page-context margin sweep — all closed (`findings.md`, `eval.md`).
- Block-compile first on OOM for EasyControl at 768, not grad checkpointing
  (closed line `dynamo_limit_contextvar`); and every GPU step goes through the
  daemon — a bare background process dies at ~1 min.

## The encoder-side predecessor

[`../cjk_aware_anima/`](../cjk_aware_anima/findings.md), frozen 2026-09-05 —
same shape: `findings.md` live, plans and reports archived.
