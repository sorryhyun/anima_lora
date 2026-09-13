# plan_wake — wake the DiT's own JA glyph units (forward plan, 2026-09-13)

> **Status (2026-09-13 evening).** W0 + W1 passed, two W2 scale arms ran,
> W2a (balanced batches) missed; a classifier diagnostic located identity at
> σ ≈ 0.8, and the W2c σ-band arm passed singles at 24 kana (**24/36**).
> Everything completed — the hypothesis, Probe 0/1 tables, the address
> geometry probe, the 256² and 24-kana arms, the cost of record — is in
> [`reports/wake_w0_w2_2026_09_13.md`](reports/wake_w0_w2_2026_09_13.md).
> This file is only what comes next. Nothing is shipped.

## Where the line stands (three sentences)

A frozen 2B DiT with a frozen adapter draws the requested hiragana from a
rows-only ext-row delta: 8 kana → 8/16 singles exact on both readers, EN
untouched. Scaling the inventory at matched per-row exposure loses
discrimination (24 kana → 10/36; か き く render at 8 and not at 24) and
multi-char strings render the *strongest* single of the string — rows
compete, and the plain FM loss never asks them to differ. Identity is decided
at σ ≈ 0.8; training only there (W2c σ band) lifts 24 kana to 24/36 singles, but
combos still render one glyph. Resolution below
512² is dead (the base cannot spell there), so cost comes from compile +
batch, not pixels.

## W2 — discrimination and composition (one arm at a time, 24-kana set)

Every arm below runs on the `w24` data (あ…ね, 40 renders/kana) unless it
needs a data rebuild, at the cost of record (512², batch 4, `--compile 1
--grad_ckpt 0 --aggressive_recompute 0`, ~23 min / 3200 steps; eval
`--no_floor`, 7.5 min). Baseline was **`rows_w24` 10/36 singles, combo CER
0.78**; the new baseline is **`rows_w24_band` 24/36, combo CER 0.67** (W2c).
Score with the largest-box script (both readers).

### W2a — balanced batches — MISS (6/36)

Every batch = 4 distinct strings rendered in one layout (font, canvas,
bubble/plain, glyph size/position), so layout would cancel across the batch.
`--stage data --data_tag w24bal --balanced 4` (same item mix and byte-identical
`eval.json` as `w24`); the train stage batches one layout group per step when
the data carries `layout_id`. Everything else as `rows_w24`.

**Result (`rows_w24bal`, train `-839501`, eval `-28f69f`):** singles **6/36**
both readers (sfx 7, vl 6) vs 10/36; combo CER 0.82/0.84 vs 0.78; EN 24/24.
Gate (≥ 18/36) missed. **Why it could not work:** rows are free per-row
parameters and the FM loss is a batch mean, so a row's gradient is the sum over
its own items. Batch composition cancels nothing; only Adam's step-to-step
dynamics change. Contrast has to live in a loss term that compares candidates,
and it doesn't need balanced data: a wrong caption on the *same image* already
shares the layout. The gate and the "rerun at 92 kana before any other lever"
rule carry over to W2c's σ-band winner.

### W2b — warm-start from the shared direction

The geometry probe found every trained delta shares a "one big kana in a
bubble" component (cos 0.25 to the mean, a quarter of each row's norm).
Initialise every new row with the mean delta of `rows_few` (or `rows_w24`)
instead of zero, so layout is solved at step 0 and the steps go to identity.
Cheap: a `--init_from <arm>` flag that reads `trained.pt`, averages the
delta, and broadcasts it. Run it on top of the W2c σ-band recipe.
**Gate: same singles at ≤ half the steps, or more singles at the same steps.**

### W2c — discrimination in the loss

**Diagnostic (`--stage classify`, job `-0fe267`).** Same-noise 24-way diffusion
classifier on 48 held-out single renders with the `rows_w24` delta: top-1 is
at chance for σ ≤ 0.65 and at σ 0.95, and peaks at **σ 0.8: 19/48 (chance 2/48),
top-3 33/48**, gap/spread 1.67 (gap 268, spread 160 in summed error per latent).
With the delta off it is at chance at every σ. The rows discriminate in the
loss only in that band, and the default sigmoid sampling puts **18.5 %** of
steps in σ 0.7–0.9 and 66 % at σ ≤ 0.6. The old "`--t_max 0.6`, layout is
fixed above it" lever pointed the wrong way.

#### σ band — PASS on singles (`rows_w24_band`)

`--stage train --arm rows --data_tag w24 --arm_tag band --t_min 0.7 --t_max 0.9`,
otherwise the `rows_w24` recipe (train `-003ed4`, 23.6 min, delta 1.02× row
norm; eval `-3b5c19`). `t_min/t_max` affinely remap σ into the band (no pile-up
at the edges).

| arm | singles (both) | seed 0 | seed 1 | combo CER sfx/vl | corpus CER | EN |
|---|---|---|---|---|---|---|
| `rows_w24` | 10/36 | 7/18 | 3/18 | 0.78 / 0.78 | 0.93 | 24/24 |
| **`rows_w24_band`** | **24/36** | 12/18 | 12/18 | **0.67 / 0.69** | **0.76** | 24/24 |

- Both-reader hits cover 18 of 24 kana (あ う え お か き く け こ さ す せ そ た
  ち). Every seed-0 single on the sheet shows the requested glyph; the misses there
  are the reader (vl returns unprintables for き こ し; つ drawn large reads as っ).
- EN untouched: leaving low σ unsupervised did not break the control.
- Renders are clean glyphs on a mostly plain canvas; the prompted bubble rarely
  appears. `rows_w24` already did this, so it is not new, but W3's bubble task
  will meet it.
- **Combos still render one glyph**, now often the *first* of the string
  (くねさ→く, さとな→さ, せいけ→せ, ちけい→ち, てきち→て) instead of the strongest;
  a few draw several (つた→「った」, つえし→「えし」, すこた→「ずすた」).
  0 exact; combo gate (≤ 0.6) still open.
- Classifier rerun on these rows: job `-8fc5de`, pending. It shows whether σ 0.8
  top-1 rose or the band widened.

**Next, per the carried-over rule: the σ-band recipe at 92 kana** before any
other lever. `data_w2` is reusable as-is (renders are 512²; the train stage
caches `latents_512.pt` next to the 256² file). Matching `rows_w2`'s exposure
(35.2k samples) at batch 4 is 8800 steps ≈ 65 min, eval 7.5 min. Gate: singles
≥ 50 % at 92 kana (`rows_w2` at 256² was 1/36).

#### Remaining W2c levers (composition — run on the σ-band recipe)

- **Same-noise classifier CE.** One image × 4 captions (right + 3 wrong, same
  template) on one noise draw, `FM_right + λ·CE(softmax(−err_k/τ))`, all inside
  σ 0.7–0.9, τ ≈ 160 summed-error units (retune from the `-8fc5de` spread).
  Same cost as a batch of 4; alternate with plain FM steps to keep positive
  exposure. For combos, add negatives that are the *same characters in another
  order* (うね vs ねう) — the only candidates that ask for order.
- **Swap-contrastive hinge** (`λ · max(0, m − (FM_wrong − FM_right))`, one extra
  forward): superseded by the CE above (one negative, 2× cost); keep only as a
  fallback if CE is unstable.

**Gate: combo CER ≤ 0.6 and no loss in singles.** Combos exact ≥ 50 % in
order stays the W2 exit gate.

### W2d — amortized glyph encoder (the way out of grinding)

Replace free per-row parameters with `g(glyph render) → row delta`, a small
CNN over the character's own font image (or a frozen vision feature of it),
trained with the same frozen-DiT loss over the whole inventory at once. The
manifold is whatever `g` constructs; a shared encoder is under pressure to
keep codes discriminable, which free rows are not. **Test: train on 60
kana, evaluate held-out singles on the other 32.** If held-out characters
render, kanji become a generalisation question rather than 2,136
inversions (W4). Cost per step equals the rows arm; one run covers the
inventory.

### W2e — several glyphs per image

A 2×2 / 3×3 grid of kana per canvas with the caption listing them
supervises many rows per DiT step and is, incidentally, the composition
training signal the 200 combos failed to give. Riskier: the DiT must bind
codes to positions, which is the open W2 problem itself. Run only after
W2a–c have moved singles; measure combos in order.

### Per-character rows (optional data lever, any time)

The shipped pack is Qwen-piece keyed (「いい」 is one row). Route each kana to
its own row (the isoq-style char partition) — a json edit + `data` rebuild —
so the training signal never lands on a multi-char piece. Cheap; combine
with W2a rather than test alone.

### EN safety (carry through every arm)

The rows delta is bit-exact on prompts without an ext id. Any arm that
re-introduces the adapter LoRA carries (1) an ext gate (LoRA scale 0 for a
sequence with no ext id), (2) a position mask (delta at ext positions only),
(3) an EN replay term for the mixed-prompt case. The gate's "EN control
unchanged" is measured on the mixed clause, not EN-only prompts.

## W3 — the product condition

Back onto `plan_render`'s task with the woken text side: EasyControl bubble
fill (S1 caches, JA edition), cond LoRA as JA-SHIP, plus the W2 rows (and
encoder, if W2d wins) loaded frozen — no target-stream LoRA. Render judge
unchanged (R1 / R2 rulers). Once a text clause is an address the DiT can
read, the OCR-quoted captions become supervision for every training run,
not just a spam suppressor. **Gate = plan_render G2 (R1 ≤ 0.5, R2 ≥ 0.7) on
JA-BODY's judge set.**

## W4 — kanji

Capacity question, not address: 2,136 jōyō against a frozen DiT whose agreed
bubble units already include kanji (Probe 0). If W2d generalises, kanji is
its held-out test at scale; otherwise the rows probe with `--only_chars` on
the top-200 corpus kanji first. Not before W3.

## Kill criteria

- W2a–c all leave singles ≤ 10/36 at 24 kana → free rows do not scale; go
  straight to W2d, and if the encoder's held-out singles are 0, order and
  identity are not learnable from the text side at this scale — the
  glyph-image path (AnyText / GlyphControl shape) is the fallback for both.
- W2 passes and W3 sits at plan_render's floor → the bubble-fill task itself
  (holed cond, whole-crop loss) is the blocker; move the woken text side to
  T2I manga-panel generation and re-judge there before killing.

## Instruments and gotchas

Full list in the report; the ones that bite while running the arms above:

- Every GPU stage via `make daemon-run ARGS="--label … --queue
  project/cjk_aware_anima_dit/probes/wake_probe.py --stage train|eval …"` then
  `make daemon-wait JOB=<id>`; the data stage is CPU-only and safe inline.
- **Never below 512²**; block compile before grad-ckpt; batch 8 OOMs at 512²
  without ckpt even compiled; activation budget stays 0.99.
- Rows lr 1e-3 in row-norm units (3e-3 walks off-manifold).
- Read the **largest detector box**, both readers; `report.md` is the
  min-over-boxes score. `--no_floor` for every eval; floor reference =
  W0/W1 (singles 0/16, JA CER 1.000, EN 24/24).
- Ext rows are Qwen-piece keyed; clear `conds_cache` on delta-scale
  switches; the VAE takes [-1, 1].
