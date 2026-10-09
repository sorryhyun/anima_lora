Reseed has reached diminishing returns from changing bands and embedding geometry. The next useful experiments should change how glyphs receive supervision and how whole strings are represented. The evidence does not yet prove that a static embedding pack has reached its ceiling.

These conclusions concern Japanese text rendering only, not KO/ZH.

The dialogue ruler (10-05, sensitive prompts):

 Table                Exact / 96 ↑    CER ↓
━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━  ━━━━━━━
 seed_retrain_0930              13    0.560
───────────────────  ──────────────  ───────
 retrain_kana                   12    0.595
───────────────────  ──────────────  ───────
 gs_rkstick                     11    0.598
───────────────────  ──────────────  ───────
 ball_rk_bubble                  8    0.610
───────────────────  ──────────────  ───────
 kana_up                         7    0.646

Every arm gets zero long strings within two edits. The banner-grid improvements therefore haven’t solved the intended task. Also, the two floor tables share their kana rows: the stronger floor’s advantage comes from its kanji changes. Latest dialogue findings (project/cjk_anima_reseed/reports/ruler_2026_10_05.md:178)

Constraints from the experiments:

Band choice matters, but further tuning has weak prospects. Gradient-derived bands tied the existing bands; larger glyphs didn’t close the gap; raising upper edges helped banners but not dialogue.

Mean/residual decomposition is useful diagnostically, but doesn’t cleanly separate layout from identity. Both parts affect scene appearance, and changing the mean’s effect depends on the residuals underneath it.

Lower training loss and more positional variance are insufficient. The tag-drop experiment loses words at essentially unchanged loss; shrinking Δ restores positional information while destroying rendering. Neither result establishes that sequence control itself is solved.

Directions, in priority order:

1. Give each glyph occurrence its own training credit.

Your gradient probe found that changing glyph k in a word affects other rows almost as much as its own row. Meanwhile, the actual loss weights the whole text region, or the union of grid cells; it doesn’t assign individual glyph regions to their matching tokens. Gradient evidence (project/cjk_anima_reseed/reports/grad_identity_2026_10_02.md:123), loss implementation (project/cjk_anima_scale/cjk_scale/loss.py:74)

I would test this training variant:

Render a natural word with per-glyph boxes.

Select one token occurrence and its glyph region.

Keep the complete caption and all embedding values in the forward pass.

Backpropagate the local glyph loss only through that occurrence’s embedding lookup.

Retain ordinary whole-word FM batches to teach composition and count.

Repeated characters need occurrence-level gradient gating, rather than merely unfreezing their shared table row.

This changes which row receives which error, and still exports an ordinary static table.

First diagnostic: repeat the own-versus-neighbor gradient probe on trained rows. The existing result is at a cold start. If trained rows already exhibit strong specificity, this hypothesis becomes less compelling.

2. Train actual dialogue lengths and column layouts, with controlled context variation.

There is a substantial remaining data mismatch. I checked the generated records: kana_up’s 5,603 window items and ball_rk_bubble’s 4,050 window items are all 2–6 glyphs. The renderer explicitly uses max_lines=1. The target includes 10–20 glyph dialogue across columns. Renderer (project/cjk_anima_reseed/reseed/recipes.py:167)

I would introduce natural 5–9 and 10–20 glyph phrases, one/two-column versions, and per-glyph boxes. Keep glyph size and its established band controlled; don’t move tiny text into high σ simply because the string is longer.

Balance the combinations:

The same phrase across different scenes and bubble shapes.
Different phrases in the same scene and bubble.
Glyphs appearing at different positions and with different neighbors.

The aim is to make glyph identity consistent while its surroundings vary. The tilde “green leaf” is a particularly concrete example of what happens when one row’s visual context is too consistent.

I’d run a small 2×2 experiment: current versus dialogue-length data, crossed with ordinary versus occurrence-local updates. Match initialization and effective glyph exposure. This distinguishes a data limitation from a credit-assignment limitation without another broad reseed sweep.

There is relevant precedent: Glyph-ByT5 explicitly used additional dense paragraph data to improve paragraph layout and small-text rendering. That supports testing the data axis, though its architecture differs substantially from Anima’s pack. Glyph-ByT5 paper

3. Add a spelling-sensitive objective, after checking its gradients.

The current FM objective can improve while words worsen. A frozen Japanese OCR recognizer could supply an auxiliary loss on decoded predicted-clean text crops: sequence loss where supported, or recognition-feature matching.

I would first verify that this loss:

Distinguishes a correct phrase from substitutions, insertions and missing characters.

Produces useful gradients at the relevant glyph size and σ.

Works on vertical text after appropriate crop handling.

Keep an independent reader for evaluation. AnyText provides a concrete precedent for OCR-feature supervision through predicted-clean images. Its results don’t establish that the loss will have enough leverage through Anima’s frozen model and rows alone. AnyText, text perceptual loss

4. If static rows still plateau, add a small quote-scoped sequence adapter.

My preferred architectural escalation would freeze the successful glyph rows and add a small residual module over the quoted string’s adapter outputs. Give it character identity, position within the quote, string length, and neighboring characters.

Anima already has positional processing in its LLM adapter. The hypothesis is therefore insufficient usable sequence control, rather than missing position information altogether. Adapter code (library/anima/models.py:2496)

Test whether this correction improves omissions, repetitions and word placement while preserving glyph identity. Keep it limited to quoted Japanese spans and preserve the padding contract.

This would require a runtime component; it cannot generally be baked into one static vector per glyph. I would pursue it only after the cheaper pack-compatible tests.

Two evaluation fixes before comparing new arms:

“Unseen” currently mislabels every two-glyph string. cov3() returns None, which scoring converts to zero. Three of the four such ruler strings occur inside training text. Use bigram/exact-string coverage for these, and reserve source dialogue before future builds. Coverage code (project/cjk_anima_reseed/ruler.py:193)

Add correct text inside the intended bubble as a metric. Current scoring can reward a correct banner or text assembled across unrelated boxes. Keep insertion, deletion and substitution counts separate; fewer duplicates can simply mean less text was drawn.

I would also run the cheap two-row tilde rollback as a diagnostic, since the latest report already identifies those rows and their training imbalance.

Order: the trained-row credit probe, then the 2×2 data/gradient experiment. Further stick scaling, band sweeps and unconstrained warm polishing wait until one of those finds a new source of improvement.