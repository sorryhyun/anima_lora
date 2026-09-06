# CJK-aware Anima, DiT side — plan (2026-09-05)

*Successor to [`../cjk_aware_anima/`](../cjk_aware_anima/findings.md) (encoder
side, frozen 2026-09-05). Premise: the T5 ext rows are **content-free
addresses**; whatever CJK means to the image is learned by the DiT / LoRA
from image data whose captions carry CJK through those rows. The old line's
verdicts are not repeated here — read `findings.md` §9 (a row is a key, the
six adapter blocks store the value), §14 (a unique OCR line binds to the
image, not to its rows; the `japanese text` tag is the address) and
`reports/0905_isotropic_ext_hypothesis.md` before proposing anything.*

## Goals

Two, ranked. Text *rendering* is not a goal until G-B says rows can carry
content.

- **G-A — manga trains healthily with text masks OFF at corpus scale.** The
  2026-09-01 reframe. Measured so far only on `sincos` (351 images); the
  corpus has **873 text-masked images across 70 artist dirs** (3,008 total)
  and has never been trained unmasked.
- **G-B — a LoRA learns CJK semantics for isotropic addresses.** The
  untested premise of the hypothesis (claim 2): nobody has shown that a row
  with no EN content acquires tag meaning (「猫耳」 → cat ears) from an
  ordinary dataset with CJK captions. If it does, the vocab pack becomes a
  seed + id map and JA prompting rides on trained LoRAs; if it does not, the
  trained tag-tier pack stays the only JA surface and this line closes.

## Premise (measured)

| fact | number / pointer |
|---|---|
| blind: rows must exist | C9 > P 19-5 (s01), rows 7-1 p 0.07 — the one row-level near-signal in 12 sets |
| blind: content is inert or harmful | HOT > C9 19-6 (s11, replicates s05); COLLAPSE ≈ HOT; ISO1 ≈ HOT 18-22 (s12); R ≈ ROTATE ≈ C9 |
| blind noise floor | same recipe, seed 2: 9-15 on 24 pairs; effective n = rows (8–16), not pairs |
| tagger recall as instrument | sign-correct on P-sized gaps, 0.03 margin is not a ranking margin (`0905_blind_g0g1_readout.md`) |
| PE-Spatial pooled cos | saturated (0.98+ to base); not a readout |
| one-image / three-image text binding | text binds to image identity at 400 steps regardless of key; `ja_ext` ≡ `ja_native` (`0904_text_bind_probe.md`) |
| text-masked images corpus-wide | 873 / 3,008 (`post_image_dataset/masks/`, 70 dirs; sincos 133, suujiniku 51, greatodoggo 36, …) |
| sincos OCR glyph size at the ~1024 tier | column width p10 25 / p50 58 / p90 124 px (227 PP-OCR v2 lines) |
| OCR reader | PP-OCRv6 (kept); PaddleOCR-VL-1.6 ties on crops, keeps ♡/ー, rewrites words, 2× the detections (`reports/0905_paddleocr_vl16_vs_ppocrv6.md`) |
| glyphs lost by downscale | at 0.5× (512 tier) 18 % of lines fall under 16 px (≈2 latent px), 34 % under 24 px; at 0.75× (768) 5 % / 18 % — measured 2026-09-05 on sincos, the *large-text* case |
| shipped surfaces | `synthja_v4` (HF `anima-vocab-pack-ja`), `AnimaVocabPackLoader` 3.9.1 with `route`; both untouched by this line |
| paired-edition manga corpus (added 2026-09-05) | 240 works / 10 artists, **5,339 JA pages with 5,280 EN twins** of the same page (relettered, artwork identical), KO edition for 136 works (3,334 pages); 3 of the artists are already in `image_dataset/`; JA page width p10/p50/p90 1003/1361/2400 px (glyphs larger than sincos at the 1024 tier); page alignment run on 2 of 240 works so far. External private checkout — never named or pathed in this repo (principle 9); its own `anima_cjk_dit.md` (under that checkout's docs dir, not this repo's) carries the crawl / alignment / registration detail |
| quote-span delimiter in captions so far | 「…」 only — 0 of 3,008 `image_dataset/` captions contain `"`, no probe or eval prompt has used it; 「」 themselves are CJK-punctuation range chars and route to ext rows like any kanji (no span rule exists yet) |
| position-clause grammar vs quotes | quote-blind: `"Are you okay? I'm fine, really."` parses as two flat tags (split on the inner comma; measured 2026-09-05). JA lines rarely carry ASCII commas; EN twins always do |

## Principles

1. **The table is content-free, deterministic and versioned.** An isotropic
   block is regenerated from `(seed, n_rows, dim, norm)`; the pack json
   carries those plus the routing. A LoRA trained through it is coupled to
   it: `save_weights` stamps the table hash (`ss_ext_pack_sha`, next to
   `ss_num_blocks`) so a mismatch at load is detectable, never silent.
2. **The shipped tag tier is not sacrificed by accident.** Until G-B is
   decided, the pack keeps the trained rows for bare CJK tags and a separate
   isotropic block for the OCR route (「…」 spans), partitioned by the pack's
   `route` field. One pack, two blocks, no row shared.
3. **Three training seeds or it ranks nothing.** Every arm-vs-arm claim needs
   ≥ 3 seeds of the same recipe; blind sets gain power from rows (prompts),
   not from seeds. Fast training exists to buy seeds, not to run more arms.
4. **Glyphs must survive in latent space** for any arm whose readout is
   per-row content. Full pages at the 512 tier are fine for G-A (layout,
   spam, adherence) and for G-B on *tags*; they are not fine for text
   content — 18–34 % of even sincos' large lines become blobs. Text-content
   arms use the 768 tier or native-resolution crops around the OCR boxes
   (≈1024 tokens each, so as cheap as a 512 page).
5. **The `japanese text` presence tag stays in every caption** that has
   text. It is the only confirmed address; arm B (no tag) is what spams.
6. **Instruments before arms.** An arm is not run until a readout exists
   that separates a known-different pair (masked A vs unmasked B at the same
   scale, or EN vs JA prompt adherence) beyond its own seed floor.
7. **Dedicated data dirs.** The 512 / 768 / crop variants live in their own
   dataset dir with their own `cache_dir`; `configs/preprocess.toml`'s
   `target_res` is never flipped (a retier orphans the main 1024 caches under
   `preprocess-reconcile`).
8. **The quote span is delimiter-agnostic on input, `"` on output** (decided
   2026-09-05). The D1 span rule accepts `「…」`, `『…』` and `"…"`, so every
   existing cache, blind-set prompt and probe keeps routing unchanged. New
   caption builders emit ASCII `"…"`: it is a native spiece token with a
   pretrained "literal string" meaning (an isotropic 「 row carries none),
   and it is script-neutral, so the paired corpus gets symmetric captions —
   `japanese text, "大丈夫？"` / `english text, "Are you okay?"` /
   `korean text, "…"` — under one rule. Consequence: the v3 eval prompt set
   (D3) uses `"` too; an arm trained on `"` captions is never graded on 「」
   prompts (one delimiter per comparison, or the delimiter becomes an arm).
9. **The paired corpus stays outside this repo.** It is adult doujinshi
   scraped for research; this tree refers to it only as "the paired-edition
   corpus", carries no path, artist list or sample, and its blind sets and
   grids are never pushed. The corpus checkout owns its own tooling and
   documentation (alignment, registration, the region-diff gate, manifest
   schema); this repo's builders consume an accepted-pairs manifest.

## Phases

### D0 — ISO1 vs C9 direct blind set — DONE 2026-09-05: flat (23-20, rows 6-6)

*Verdict in [`findings.md`](findings.md) D0: isotropic ≥ trained, not >; gate passed on cost grounds, proceed to D1.*

Job `20260905-191022-ff8d51` (`regrid_set.py --set s13_ISO1_vs_C9 --arms
ISO1 C9 --seeds 6 7 8`, v2 prompts, `cjk_unmask_eval3`, 48 pairs, both
arms at seeds the grader has not seen). Removes the transitivity the ISO1
claim rests on (s03/s04 already broke transitivity once).

*Gate:* ISO1 ≥ C9 on the row sign test → the isotropic block is the OCR-route
default and D1 builds it. C9 > ISO1 beyond the floor → the trained pack keeps
the OCR route too, and G-B runs on *both* tables (the s11 HOT result would
then be scale-specific after all and `0905_isotropic_ext_hypothesis.md` is
amended). Either way D2–D3 proceed; only the table changes.

### D1 — deterministic table + route partition + LoRA stamp — DONE 2026-09-05 (gate passed)

*Shipped 2026-09-05 (see [`findings.md`](findings.md) D1 for pointers): all six
bullets below are in the tree; the gate render (`C9ISOQ` arm, daemon job
`20260905-210248-114990`) landed inside the seed-twin floor. Runtime contract:
[`docs/experimental/cjk_ext_vocab_coverage.md`](../../docs/experimental/cjk_ext_vocab_coverage.md)
§"Quote partition"; tests `tests/test_ext_vocab_iso.py`.*


- `make_random_pack.py --mode iso` from a seed: rows i.i.d. Gaussian at the
  native T5 row norm (ISO1's recipe), no source pack read; bytes
  reproducible across machines (CPU single-thread, fixed dtype).
- Pack json: `route` gains a **quote block** — spans inside 「…」 (and the
  `order`-format phrase) resolve to the isotropic block; bare CJK keeps the
  trained rows (`synthjakozh1sym_r256` or `synthja_v4`). Row ids of the two
  blocks are disjoint; the encoder change is the routing rule only.
- `save_weights` stamps `ss_ext_pack_sha`; `load_dit_model` / the Adapter
  node warn on mismatch. Unit tests: EN bit-exact with the partitioned pack;
  determinism (two builds byte-equal); a 「…」 caption touches only the
  isotropic block.
- Node: `AnimaVocabPackLoader` reads the seed and regenerates the block if
  the pack ships without rows (optional — shipping 73 MB of rows is fine).
- **Span rule** (principle 8): one regex over the caption before
  `segment_runs` — `「…」`, `『…』`, `"…"` (non-greedy, per tag) mark the
  isotropic route; the delimiter chars themselves stay on their current
  path (「」 → ext row, `"` → spiece). Unit test: the three spellings of the
  same line produce the same ext ids for the content.
- **Quote-aware caption grammar.** `parse_caption` splits a quoted line on
  its inner comma (premise table). Fix at the source in `anime_tools`
  (`position_clauses`: a comma inside an open `「『"` is not a tag separator;
  `compose_caption` round-trips) and bump the pinned rev — the alternative,
  substituting commas inside lines in every builder, leaks into OCR CER.
  Test: the EN-twin caption above parses to one tag.

*Gate:* tests green; the C9 recipe re-cached through the partitioned pack
renders the 8-row grid inside the s02 floor against ISO1 (the partition
changes nothing for CJK-free eval prompts by construction — this is a
sanity check, not an experiment).

### D2 — the OCR-diverse corpus (2 days; CPU + one daemon OCR job)

Build once, at native resolution, then derive the tier variants.

- **Sources**, two. (a) Every text-masked image corpus-wide (873) plus the
  text-free images of the same artist dirs as the negative class. (b) **The
  paired-edition corpus** (premise table; principle 9): the same page of the
  same work lettered in JA and EN, KO for about half — artwork identical,
  only the glyphs differ. It enters through an accepted-pairs manifest
  produced in its own checkout: phash + Needleman–Wunsch page alignment
  (exists there; run on 2 of 240 works — run it on all), then per-pair
  **registration** (the editions are different scans at different sizes,
  sometimes with borders — grayscale ECC / feature homography, EN and KO
  warped onto the JA page), then a **non-text region-diff gate**: a pair
  whose registered twins differ outside the OCR boxes (decensoring, SFX
  redraws, credit stamps) is rejected, because a "text-only" pair that also
  differs elsewhere teaches the tag the wrong thing. Parent tags for these
  pages come from a Tagger pass (the source tags are not danbooru). What
  the pairs buy, and why they are worth the plumbing: **a contrastive
  address** — on identical art, `japanese text` vs `english text` can only
  mean the script, where on the 873 alone the tag can be satisfied by
  looking like manga; and **recurrence volume** — ~5,300 dialogue pages
  where 873 pages were unlikely to give the census 50 rows at 100 images.
  OCR via
  `anime_tools.ocr` (PP-OCRv6, v2 post-processing, gate 0.70, mask-complement
  regions as in `datasets/ocr_text_captions.py`); records per image with
  boxes. PP-OCRv6 stays the reader: PaddleOCR-VL-1.6 is a wash on the same
  crops and rewrites words (`findings.md`, OCR-reader entry;
  `reports/0905_paddleocr_vl16_vs_ppocrv6.md`). Its one job here is the
  **floor measurement** below — run page `Spotting:` (batched, bs 8,
  `use_cache=True`) beside the PP-OCRv6 pass and count the masked images
  that get a line from either detector. The optional hybrid pass (VL `OCR:`
  on PP-OCRv6's quads, repetition-guarded, preferred on low PP score or
  symbol disagreement) is built only if that count moves.
- **Captions**: production caption + `japanese text` + lines in the `tags`
  format (the trained shape; `order` is one arm, not the default), quoted
  with `"` (principle 8); twins get `english text` / `korean text` + their
  own lines the same way (EN via PP-OCR's Latin model, or the scanlation's
  text where the checkout has it). The
  44-of-133 "masked but no OCR line" floor from sincos will recur —
  measure it with both detectors (PP-OCRv6 DB; VL-1.6 Spotting found 260
  lines where PP found 132 on 40 sincos pages, mostly SFX / chrome); those
  images train as arm B and cap every unmask arm.
- **Census (the number this phase exists for)**: per Qwen token in the OCR
  lines, how many *distinct images* carry it. Findings §3's rule is
  O(100+) visits for identity; report the count of rows at ≥ 20 / ≥ 100 /
  ≥ 300 images. If fewer than ~50 rows reach 100 images, G-B on OCR text
  is not testable on this corpus and D5b is dropped before it is built.
- **Resolution census**: per line, glyph column width at 512 / 768 / native;
  the fraction under 16 px per tier decides D4's tier (principle 4).
- **Variants**: (i) full pages at the 768 tier (default for G-A; 512 only if
  the census says < 10 % glyph loss corpus-wide), (ii) native-resolution
  crops around merged OCR boxes with ≥ 128 px margin, captioned with the
  in-crop lines + parent tags (for D5b). For a paired page the crop box is
  cut from the JA edition and the **same registered box** from each twin,
  so a JA crop and its EN twin differ only in glyphs — a minimal pair for
  the stroke prior, with the twin as a built-in negative. Both variants in
  dedicated dirs with their own `cache_dir`; VAE + TE cached through the D1
  pack. Crops teach strokes (self/MLP), pages teach the plan (where text
  goes, that a bubble exists) — neither variant trains alone; D4/D5b mix
  them as two subsets of one dataset (free-fit handles the mixed tiers).

*Gate:* census tables in `reports/09xx_ocr_corpus.md`; the recurrence count
decides whether D5b exists.

### D3 — instruments (1 day, CPU; renders via `make gen`)

- **Blind pairs** (exists, `../cjk_aware_anima/probes/blind_pairs.py`):
  the v2 16-row prompt set grows to **32 rows** (power comes from rows);
  seeds 2 per row.
- **JA-tag adherence** for G-B: a prompt set where every prompt is a bag of
  dbv4-scorable tags rendered in three forms — EN, JA through the trained
  pack, JA through the isotropic block — and tagger recall per form
  (`unmask_grid_judge.py` machinery). Calibration: EN vs JA-through-ISO1 on
  the *base* model must show a large gap (ISO1 rows carry nothing, so JA
  adherence should be near the no-prompt floor); that gap is what a G-B
  LoRA has to close.
- **Non-diegetic text count** (exists, `unmask_grid_ocr.py`): floor only.
- **Script swap** for the contrastive arm: the same prompt and seed with
  `japanese text` ↔ `english text` (↔ `korean text`) swapped; PP-OCR's
  script classification of whatever text renders, paired per (prompt,
  seed). Calibration on the base model and on arm A: the swap should move
  nothing (the tag is a style proxy there); on arm T it must move the
  script. Cheap, no grader.
- **Render→OCR CER on recurring lines** for D5b: prompt a recurring line
  through the isotropic block, PP-OCR the render, CER vs the line; held-out
  lines (present in ≥ 100 training images but not in the prompt's other
  content) vs never-seen lines. Floor readout until D5b exists. The judge
  is **PP-OCRv6, not a VLM reader**: PaddleOCR-VL rewrites toward the
  likelier word (`狠狠地`→`狼狽地`, `おい`→`あい`), which on a half-rendered
  line means repairing it toward the prompt and inflating exactly the gap
  this instrument measures.

*Gate:* masked A vs unmasked B at corpus scale (D4's first two arms, 1 seed
each) are separated by the count and by a 16-row blind set. If no readout
separates them, G-A cannot be measured and the corpus-scale claim stays
unmade.

### D4 — G-A at corpus scale (½ day per arm, ~1 h GPU each at 768)

Arms, plain LoRA dim 32 / lr 2e-5 / 8 epochs (the C9 recipe), 768 tier,
**3 training seeds each**:

| arm | mask | captions | rows |
|---|---|---|---|
| A | on (production) | production | stock |
| B | off | production (no text tag) | stock — the spam control |
| P | off | + `japanese text` | none looked up |
| U | off | + tag + `"…"` lines | isotropic block (D1) |
| T | off | U + the EN/KO twins as extra images with `english text` / `korean text` + their lines | isotropic for JA/KO lines; EN lines native spiece |

A/B/P/U train on both sources' JA pages (873 + the paired corpus's JA
editions); T adds the twins. T is the contrastive arm: identical artwork
under two tags forces the presence tag to mean the script.

*Gate:* U ≥ A on the 32-row blind set (row sign test) and U's non-diegetic
count ≤ A's, on the pooled 3 seeds; B must lose to U (else the readout is
broken, not the arm). U vs P is the corpus-scale replication of s01 — if
it comes out flat here, "rows must exist" was a sincos artifact and the
shipped recommendation is the presence tag alone. T's own gate is D3's
script swap: T moves the script where A/U do not, and T ≥ U on the blind
set. T > U → the shipped manga recipe trains twins whenever a work has
them; T ≈ U → the contrast is free but not needed, pairs stay a D5b
substrate only.

*Kill:* U < A at corpus scale → manga stays masked in production; G-A closes
at "sincos only", and D5b is not run (text pixels that hurt training are not
a substrate for content learning).

### D5a — G-B on tags (1 day; 512 tier is the right tier here, 3 seeds)

The decisive experiment of the hypothesis, cheap because tags are coarse
content. The ordinary LoRA dataset (`image_dataset`, 3,008 images) with
JA-tag caption variants from `datasets/tag_glossary.py` (caption-variants
machinery, JA on 50 % of variants), TE cached twice: through the trained
pack and through the isotropic block. One LoRA per table, 3 seeds, 512
tier, 8 epochs.

Readout = D3's JA-tag adherence, three prompt groups: tags visited ≥ 100
images in training, visited 5–99, never visited. Plus EN adherence as the
regression control.

*Gate:* ISO+LoRA recall on visited-≥100 tags reaches the EN recall band
(claim 2 holds); the gap on never-visited tags vs the trained pack is the
price of claim 1, reported as a number. If ISO+LoRA does not learn even the
≥ 100 band, claim 2 is false: the line closes with the trained tag-tier pack
as the only JA surface, and D5b is not run.

### D5b — G-B on text (2 days; native crops, 3 seeds) — only if D2's census and D5a pass

Train the C9 recipe on D2's crop variant (every glyph at native size), 3
seeds — JA crops with their EN/KO twins, mixed with D4's winning page arm
so the plan is not forgotten. Readout = D3's CER on recurring lines,
held-out vs never-seen; plus the D4 grid to check the crops did not teach
spam; plus the script swap on crop-shaped prompts (a twin-trained crop
model that cannot swap the script has learned the bubble, not the rows).

*Gate:* held-out recurring lines render at CER below never-seen lines beyond
the seed floor → rows carry text content when a line recurs. That is the
prerequisite the deferred glyph line (`_archive/cjk_aware_anima/plan2_glyph_line.md`)
always needed; it reopens only on this gate. *Kill:* no CER gap → OCR rows
are addresses only, at any scale; text rendering goes to the EasyControl
glyph-strip route or nowhere.

### D6 — ship (after D4, independent of D5)

Whatever D4 selects (U or P) becomes the documented manga recipe: the
`japanese text` tag + optional 「…」 lines, `masked_loss=false`, the
partitioned pack as a release asset, `ss_ext_pack_sha` in every LoRA, one
new `docs/methods/` deep-dive (`cjk_manga_training.md`), i18n of the guidebook
line. The tag tier ships as it does today unless D5a replaces it.

## What `bench/frontload_text_boost` + `docs/findings/crossattn_self_attn_dominance.md` already tell this line (read 2026-09-05)

Measured on the base DiT in July; none of it was written for CJK, all of it
applies.

1. **Cross-attn writes the low-frequency plan at σ ≥ 0.85 and fades to a
   ~0.02 floor below; self-attn + MLP carry 85–94 % of the residual at every
   σ and render the plan.** So a T5 row is consumed only in the plan window.
   Consequences: (a) the `japanese text` *category* tag sustains cross-attn
   mass (2.4× uniform) while the literal glyph string decays below uniform —
   the attention-side twin of s01 / §14 (presence tag is the address, rows
   are not); (b) training steps at low σ contribute ~nothing to what a row
   means — a G-B arm may **sample σ toward the plan window** (the T-LoRA /
   timestep-mask machinery exists) as a speed lever, to be A/B'd, not
   assumed; (c) full pages at 512 train exactly the part of the text problem
   the plan owns (layout, bubbles, that text exists) and none of the part
   self/MLP must learn (strokes) — principle 4 restated from the mechanism.
2. **The glyph deficit is upstream.** With a glyph in the caption the model
   spends *more* collinear drive in the feature-commit band (σ 0.5–0.8,
   +6.8 % paired, 25/28) but never rotates (cos 0.999): the early plan does
   not encode which glyphs, and self/MLP has no prior to elaborate them. For
   D5b that means two things must be learned at once — plan-level binding of
   *which* rows (needs recurrence, hence D2's census) and a stroke prior
   (needs native-resolution glyphs, hence crops). Neither alone renders text.
3. **`k_norm` annihilates row scale on the K path** (RMSNorm per token×head),
   so embedding-level scaling is pure V/loudness and never touches
   allocation. That is the mechanism for s12 (HOT ≈ ISO1): norm ×5 was never
   going to matter. Do not run norm / gain arms on the table.
4. **A LoRA rides self/MLP** (sincos LoRA: all pathways +3–5 %, cross-attn
   share unchanged; the trigger is a switch, style is delivered downstream),
   yet labeled tags *are* learnable through cross-attn and the boost proves
   "small voice ≠ no vote" in the plan window. G-B is plausible; expect its
   signal to be small and plan-window-local, which is what the readout must
   be sensitive to.
5. **`--xattn_boost` is a free diagnostic for G-B.** It multiplies whatever
   text drive exists and conjures nothing (theremin-class prompts stay
   failed at every λ). Render D5a's JA-tag prompts with and without
   `XATTN_BOOST=2`: adherence that grows under the boost is a learned-but-
   weak binding, adherence that stays at the floor is absent. And the
   token-selective arm (c) that never shipped for lack of a span-annotation
   surface has a free span here: **ext ids ≥ 32128 are the span** (the same
   gate the adapter LoRA used). `--token_values` in the bench is the
   reference implementation; a `--xattn_boost_ext_only` flag is one line of
   plumbing if D5a needs the aimed version.
6. **Instrument rules that carry over.** Read paired deltas per (prompt,
   seed), never per-arm means (cross-prompt std 0.078 vs paired 0.0025 —
   the same reason the blind sets are paired). Cross-attn *mass* is
   sink-confounded and not portable across prompts — never use it as a
   readout. The velocity-level paired probe (`ANIMA_CFG_DELTA_LOG`,
   `glyph_delta_probe.py` pattern) *is* portable: for D3 add a **row-drive
   probe** — the JA prompt through the isotropic block vs the same prompt
   with the CJK span dropped, rel Δ‖v_cond − v_uncond‖ at σ ≥ 0.85, paired.
   It is direction-blind between arms (a floor gate like the old residual
   probe, not a selector), costs no rendering judgement, and says whether
   the trained DiT drives *anything* off the rows before a tagger is asked.
   Compiled renders are not bit-stable across processes — both arms of a
   blind set render in one job (`regrid_set.py` already does).

## Not in scope / anti-goals

- **No new distill arms** on the trained pack — rank, coverage, geometry,
  symbol register, name tier are all measured and closed (`findings.md`
  §4, §12–§14; `0905_blind_g0g1_readout.md`).
- **No 1–3-image text-binding reruns** (§14: a unique line binds to the
  image at any budget).
- **No content-alignment for the OCR route** (vec2vec, description
  embeddings, EN substitution) — the route is content-free by design.
- **No text rendering before D5b's gate**, and no 512-tier arm whose
  readout is text content (principle 4).
- **No `target_res` flip** in `configs/preprocess.toml` (principle 7).
- **No OCR-reader swap and no prompt engineering of PaddleOCR-VL.** The
  A/B is done (`reports/0905_paddleocr_vl16_vs_ppocrv6.md`): character
  accuracy ties, hints are noise, batching is the only lever. A manga
  fine-tune of VL is the one thing that could move it — that is
  [`plan_ocr.md`](plan_ocr.md)'s O2 second arm (crop `OCR:` LoRA vs
  manga-ocr on COO / Manga109-s), not D2's work; D2 stays on PP-OCRv6.
- **No edit / conditioning line on the pairs.** JA→EN twins look like
  DirectEdit or EasyControl training data (translate the lettering in
  place); that is a different deliverable (a conditioned re-letterer, not a
  LoRA that knows CJK) and the in-place edit line is archived. The pairs are
  a contrastive *training* substrate here, nothing else, until D5b's gate
  reopens the glyph line — which names EasyControl as its fallback.
- Sigma-demoted training (`--sigma_lowres`) is a wall-clock lever, not a
  resolution arm; it may be enabled on any arm but is not what "512" means
  here.

## Order and budget

D0 (done) → D1 (1 d; + ½ d for the quote-aware grammar in `anime_tools`)
→ D2 (3 d: align + register + gate the paired corpus in its own checkout,
then the OCR job and census here) → D3 (1 d) → D4 (5 arms × 3 seeds ≈ 15
GPU-h at 768, T is the largest — three evenings on the daemon) → D5a (2
tables × 3 seeds ≈ 3 GPU-h at 512) → D5b only on both gates → D6. About
two and a half working weeks; every GPU step is a daemon job.

## Deliverables

- `make_random_pack.py --mode iso`; pack json `route` quote block;
  `ss_ext_pack_sha` stamp + loader warning; tests.
- `datasets/build_ocr_corpus.py` (records, captions, census, crop variant,
  two-detector floor count) — consumes the accepted-pairs manifest of the
  paired corpus beside the 873; `reports/09xx_ocr_corpus.md` (numbers
  only, no samples from the paired corpus — principle 9).
- In the paired corpus's own checkout: all-works alignment, registration,
  the non-text region-diff gate, the manifest schema — documented there in
  its own `anima_cjk_dit.md`, not here.
- `anime_tools`: quote-aware `parse_caption` / `compose_caption` + pinned
  rev bump.
- Done: `reports/0905_paddleocr_vl16_vs_ppocrv6.md`,
  `probes/ocr_vl16_ab.py`, `probes/ocr_vl16_prompt_batch.py`.
- `assets/unmask_eval_prompts_v3.txt` (32 rows, `"` delimiter),
  `assets/ja_tag_adherence_prompts.json`, `probes/ja_tag_judge.py`,
  `probes/text_cer_judge.py`, `probes/script_swap_judge.py`.
- `reports/09xx_ga_corpus_scale.md` (D4), `reports/09xx_gb_tags.md` (D5a),
  `reports/09xx_gb_text.md` (D5b if run), blind set reports per set.
- `findings.md` here, started at D0's verdict; the old line's `findings.md`
  is read-only from now on.
