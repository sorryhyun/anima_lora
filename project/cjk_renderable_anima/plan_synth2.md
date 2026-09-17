# plan_synth2 — the ΔFM line: paired-difference supervision for new rows, and the lr question it opened

> **Status 2026-09-17 night: Δ0 and Δ0b are read.** Record:
> [`reports/synth_pair_2026_09_17.md`](reports/synth_pair_2026_09_17.md).
> What the line has: (1) **ΔFM is a scene-holding loss** — the row learns
> the glyph and the native render keeps its scene (en cos 0.93 vs 0.855 at
> equal row norm; images under 0.85: 6 vs 20 of 64), the first recipe of
> the S line that does not wipe; (2) **lr is a lever** — ΔFM at `--lr_rows
> 2e-3` × 1 500 steps ≡ 1e-3 × 3 000 (19 vs 18/24; plain FM 20 / 21), the
> budget is the lr integral, plain FM unmeasured; (3) **ΔFM saves no
> exposure**; (4) **its cost is the base's JA pseudo-text beside the
> glyph**, which halves the reader hits (joint hit ∧ en cos ≥ 0.85: plain
> 28, ΔFM 15–20 of 64), and Δ0b showed the loss cannot remove it: the
> pseudo-text is the base free-running under `japanese text`, no training
> residual sees it, and under plain FM it is the wipe that removes it —
> wipe ↔ co-text is one axis. Next: **L0** (is the S line's exposure budget
> ∫lr — plain FM, decides B1's cost), then **Δ2** (does the co-text survive
> when the caption carries a whole sentence instead of one glyph). One GPU
> line at a time; the `sent-a0p3-s05` report is still owed before Δ2.
> Origin: the user's three-candidate note of 2026-09-17 (paired-difference
> FM / cached local Jacobian / OCR-reward rows); this plan takes the first,
> the other two are in *Not this plan* with the reason.

The S line pays for identity in exposure: ≈ 1 000 draws per row
(`plan_synth.md`, exposure curve 1 330 / 670 / 490 → 100 / 75 / 36 %). Every
shortcut to the *address* is closed (encoder, composition, transplant, warm
start, Q — `findings.md` *Do not re-propose*). This line does not touch the
address. It changes the **supervision per draw**: the row is trained on the
difference between two renders of the same scene under the same noise, one
with the new glyph and one with a reference the base already draws. It was
proposed as an exposure saver; Δ0 says it is not one. What it measurably
does is hold the scene while the row learns the glyph, and what it
measurably costs is the base's JA pseudo-text left standing next to it —
a cost the loss itself cannot pay down (Δ0b). The exposure question moved
to the learning rate (L0); the co-text question moved to the sentence
recipe (Δ2).

## What it is

Notation as in the loop (`src/train/stage.py`, `fm_training_batch`):
`z_σ = (1−σ)·x + σ·ε`, target `v* = ε − x`, rows-only trainables.

Per composite item B (scene + new glyph, caption `c_B` with the ext row)
the data stage also holds its **sibling A**: the same scene, same erase,
same font / size / position / colour / tilt, a reference string in the
box, caption `c_A` = the same frame with only the quoted string swapped.
One noise `ε`, one `σ` for the pair. Loss

    L_Δ = ‖ (v_θ(z_σ^B, c_B) − sg v_θ(z_σ^A, c_A)) − (v_B* − v_A*) ‖²_w
        = ‖ r_B − r_A ‖²_w,                r_X = v_θ(z_σ^X, c_X) − v_X*

with `w` the existing box weight. Gradient w.r.t. the new row:
`(r_B − r_A) · ∂v_θ(B)/∂f_new` — **plain FM with the sibling's residual
subtracted as a control variate.** What that buys, region by region:

- **In the target** the noise cancels exactly: `v_B* − v_A* = x_A − x_B`,
  the clean pixel delta, zero outside the box for any reference.
- **Outside the box** the term is `‖v_θ(B) − v_θ(A)‖²`: match the
  known-text render of the same scene. That is a scene-preservation term
  against the wipe (`findings.md`: wipes are the delta norm) where today
  the loss is weight-1 plain MSE against a noisy target.
- **Inside the box** the shared part of the residual cancels: the
  posterior spread of the scene given `z_σ` (the x₀ side dominates at
  σ ≈ 0.8 where identity is decided and it depends on `z_σ`, not on the
  glyph), and — because A is a paste by the same renderer — the erase
  patch / ring-median fill / font rasterisation that both items carry
  (*Erase artefacts as a cue*, `plan_synth.md` open risks, is shared and
  drops out). What is left is the glyph delta plus the part of the model's
  B error the reference does not share.

**Measured on Δ0 (2026-09-17):** ≈ 80 % of the residual cancels
(`fm_pair` 0.013–0.02 vs `fm_plain` ≈ 0.10), the outside-box term does hold
the scene — and the shared part is not only noise. Under the JA frame the
base wants JA pseudo-text; that error sits in `r_A` and `r_B` alike, plain
FM's gradient on it is what teaches the row "this glyph and nothing else",
and ΔFM cancels it. The cancelled part is also the sign-consistent part
Adam travels on, so ΔFM needs ≈ 2× the lr for the same row norm.
Δ0b corrected one step of that reading: the JA pseudo-text is **not** part
of the shared residual (an EN-frame sibling cancels exactly as much). What
cancels is the scene residual, and the scene residual is what builds the
row's "bare canvas, one glyph" component under plain FM — the wipe, which is
also the only thing that was removing the base's free-running pseudo-text.

The address is untouched: `f_new` is random-init and its Jacobian is taken
through `c_B` (the ぬ caption) exactly as today; `c_A` never contains the
row, branch A is `no_grad`, and nothing about the reference's embedding
reaches the parameters. Deployment format (`ExtDelta` table → vocab pack),
eval / native / target stages: unchanged.

What can leak is the **mean of `r_A`** — the target becomes `v_B* + r_A`,
so a systematic error of the base on the reference inside the box is
inherited (the か-with-Latin-strokes mechanism, `synth_micro_loop`). For a
calibrated model `E[r_A | z_σ] = 0`; the bias is bounded by how badly the
base renders the reference, which is why the reference is EN.

## The reference

**EN, pasted by our renderer, under the JA frame.** For each composite the
data stage draws a second image with the same `render_into_scene` fit
(font, `fs`, lines, colour, tilt angle, the erase) and a Latin string of
the same glyph count in place of the text; caption `scene_caption(sc,
ref_text)` — the JA frame the scene was drawn under (`japanese text` tag,
`Japanese text reads as "KA"`), only the quote differs. Rules:

- **Same frame in both branches.** A JA-frame B against an EN-frame A puts
  a frame-mismatch component into `r_A` that B does not share (rows are
  JA-frame-bound, `findings.md` rulers) — it adds noise instead of
  cancelling it. Frame is held; only the quoted string moves.
- **Never the romaji.** Reference letters are drawn from A–Z minus the
  target's romaji letters (す never gets S / U). The row cannot see the
  pairing either way; the rule keeps the design from reading as a mapping.
- **Matched extent.** One Latin letter per JA glyph, same `fs`, same
  column / line positions (`_draw_vertical_glyph` already draws per glyph,
  so a vertical Latin stack is free). The weighted box is the **union** of
  the two drawn boxes.
- **A small pool per scene.** `--pair_ref_pool 4`: four reference strings
  fixed per (scene, glyph count), so the reference captions stay ≈ 1 k
  unique (text cache is ≈ 1.3 MB per caption in RAM; 10 k distinct
  reference captions would be 13 GB on the 46 GB box).

**No warm start is required by the loss.** The one condition is that the
base can already draw the reference (so `r_A` is a scene residual, not
the floor's garbage); pasted Latin meets it with no table, so the EN arm
trains new rows **from random init**. Only a *kana* reference needs a
table — for the reference row, not the new one (Δ1); Δ2 / Δ3 warm-start
because the sentence recipe does, not because ΔFM does.

Why not the alternatives:

| reference | in-box cancellation | inherited bias | verdict |
|---|---|---|---|
| a known kana (す) | most (similar posterior) | most (base ≈ 34/36 on them, style quirks per glyph); **needs a trained table** — from scratch the base has no address for す and `r_A` is the floor's garbage | control arm only, on a warm start (Δ1) |
| EN letters, pasted (this plan) | middle | least among glyph refs (EN 24/24 flat); paste artefacts shared | **the arm** |
| the scene's own un-erased anchor ("hi", the base's render) | — | paste / erase artefacts are in B only → they no longer cancel; frame differs | no |
| blank box | least glyph-dependent | none in-box | caption differs by the whole clause → frame mismatch | no |

**Diagnostic for the leak, free in the loop:** `ref_bias` = in-box
‖mean_batch r_A‖ / mean_batch ‖r_A‖ (EMA over 100 steps). ≈ 0 means the
reference residual is spread, not systematic; a value that stays above
≈ 0.3 says the base draws the pasted Latin under the JA frame with a fixed
error, and the sheets are read for Latin-styled strokes on the trained
glyphs. If both show, the fallback is `--pair_sigma_min 0.7`: the paired
term on the identity band only, plain FM below it (stroke style is
committed at lower σ than identity).

**Δ0 read:** `ref_bias` 0.26–0.28 from step 250 to the end on all three ΔFM
arms — under the gate and never decaying. No Latin-styled strokes on the
hits; the systematic part of `r_A` is the JA-frame pseudo-text (a vertical
pseudo-kanji column in the bubble on 8–10 of 12 seed-0 singles, lines of it
on the natives). The 0.7 fallback is a no-op on a `--t_min 0.7` recipe;
Δ0b moved it to 0.8.

## As built

Commit `2ffacc78`: `src/data/pair.py`, `common/render/scene.py`
(`ref_text=`, one fit, two draws), `data/synth.py` (`--pair_ref none|en`,
`--pair_ref_pool`, `ref_file` / `ref_text` / `ref_caption` / `ref_box`,
union `box`, `sheet_scene_pair.png`), `train/stage.py` (`--pair_loss`,
`--pair_sigma_min`, `pair_branch` under `no_grad` with the batch's own `ε`
and `σ`, `pair_stats` → `fm_plain` / `pres` / `ref_bias`). 1.63 it/s against
2.35 plain (0.69×); the second compiled graph cost nothing. Plain-FM
controls run on the same data dir (`--pair_loss 0` ignores the siblings).
`--pair_ref_frame ja|en` (Δ0b arm 2; `data/pair.py::en_frame`) rewrites the
sibling captions under the scene's EN frame at train time — measured flat,
kept as a flag. `--pair_ref kana` (Δ1) and paired `short` / `sentence` kinds
(Δ2) are not written.

## Arms (one at a time)

All micro arms reuse `output/wake_probe/data_synth_pair_d0` (2 000 paired
dakuten singles, 12 rows) and the Δ0 argv (report, *What ran*); only the
named flags move. Eval `single` (24 renders) + `native` (`が,ガ,ご,ゴ`,
`en,swap`, 64 per clause) on every arm that is read for the scene — Δ0's
singles alone pointed the wrong way. Rulers: `single` exact, native
`both`, en cos, and the **joint** count `both ∧ en cos ≥ 0.85` from
`native_reads.json` (plain `pair0_s3000`: 28 en / 16 swap; tail under
0.85: 20 / 16).

**Δ0 — done.** `reports/synth_pair_2026_09_17.md`.

**L0 — is the exposure budget the lr integral? (plain FM, ≈ 50 min)** Every
exposure number of the S line (≈ 1 000 draws per row, the 1 330 / 670 / 490
curve) was measured at lr 1e-3 cosine, so draws and ∫lr were never
separated; Δ0 separated them for ΔFM only, and the 12-row plain control is
ceilinged at 1 500 steps. `--pair_loss 0`:

| arm | `--lr_rows` | steps | ∫lr vs `pair0_s1500` | read |
|---|---|---|---|---|
| `pair0_s750` | 1e-3 | 750 | 0.5× | the off-ceiling control |
| `pair0_s750_lr2e-3` | 2e-3 | 750 | 1× | singles |
| `pair0_s1500_lr2e-3` | 2e-3 | 1 500 | 2× | singles + native (the wipe at `pair0_s3000`'s travel) |

- **Pass:** `pair0_s750_lr2e-3` ≥ 19/24 with `pair0_s750` ≥ 4 below it →
  the budget is ∫lr. B1 (`plan_synth.md`, the 100 k base seed) is then
  priced at 2e-3 × half the steps, after one mid-size check — 12 rows read
  83 % where the 53k curve reads 36 % at the same draws, so the micro
  regime does not carry the interference term.
- **Fail** (2e-3 × 750 ≈ 1e-3 × 750): draws are the budget for plain FM and
  the lr effect is ΔFM's own (its cancelled sign-consistent gradient);
  B1 stays as priced.
- **Guard:** `rel` and the native tail on `pair0_s1500_lr2e-3`. 3e-3 is
  closed (off-manifold at 2.4× row norm, `findings.md`); 2e-3 on a warm
  start is a separate question — the anchor sweep fixed erasure at 1e-3
  with `--lr_warmup 500`, not above it.

**Δ0b — the co-text: done, not removable by the loss.** Two arms on the
lr 2e-3 × 1 500 recipe, each with native (report, *Δ0b arm 1 / arm 2*):

| arm | `single` | joint en | tail en < 0.85 | swap joint |
|---|---|---|---|---|
| plain `pair0_s3000` | 21/24 | 28 | 20 | 16 |
| ΔFM `pairEN_s1500_lr2e-3` | 19/24 | 15 | 6 | 12 |
| `--pair_sigma_min 0.8` | 20/24 | 20 | 9 | 12 |
| `--pair_ref_frame en` | 17/24 | 18 | 6 | 13 |

The σ split lands between its two parents on one line; the EN-frame
sibling changes nothing, down to the paired loss (0.0154 vs 0.0156) — the
pseudo-text is in no teacher-forced residual. `--pair_mix λ` was not run:
it mixes the same two losses and is expected on the same line. The gate
(plain's joint number with ΔFM's tail) is not reachable by reweighting the
loss on single-glyph data.

**Δ2 — the sentence recipe with pairs (≈ 3.5 h; after L0 and the
`sent-a0p3-s05` report).** The reading that keeps the line open: on a
single-glyph native the base free-runs a sentence's worth of JA text and
the table addresses one glyph of it, so everything else comes out as
pseudo-text; plain FM hides that by wiping the scene. When the caption
carries a whole sentence of trained rows there is no unaddressed text left
to invent — *if* that holds, ΔFM's cost disappears where the target lives
and its scene-holding stays. The `sent-a0p3-s05` argv (`README.md` *How to
run*) with `--pair_ref en --pair_loss 1`; every kind paired, `short` /
`sentence` siblings as vertical Latin stacks with the item's cuts (code
owed in `data/synth.py`). Steps and lr from L0.

- **Gate:** the target stage and native both clauses against the finished
  `sent-a0p3-s05` — joint number not below it, tail under it, and the
  sheets read for text *outside* the addressed string (the co-text count
  is the number this arm exists for). Flat singles, `short` / `phrase` and
  their `_held` not below the baseline.
- **Kill:** co-text beside a correctly rendered sentence at the single-glyph
  rate → close the line; ΔFM stays in the tree as a flag, one row in
  `findings.md` *What does not move it*.
- **Guard:** the sibling stacks keep one column when the item does (read
  the `_ref` images on the composite sheet before launch); warm rows keep
  lr 1e-3 unless L0's guard arm clears 2e-3 on a warm start.
- A cheaper look first, if wanted: the Δ0 tables under a native caption
  that quotes *several* trained glyphs (`がガゴ`) — does the co-text shrink
  as the addressed share of the text grows? One native stage, ≈ 9 min, no
  training.

**Δ1 — kana-reference control.** Parked. A kana sibling was meant to buy
more in-box cancellation; Δ0 showed cancellation is not the bottleneck
(≈ 80 % already, no exposure saving). Revive only if Δ2 passes and the
in-box stroke quality is the complaint.

**Δ3 — B1 with pairs.** Only on a Δ2 pass. L0 is what makes B1 cheaper;
ΔFM is what would let its table render inside a scene.

## Open risks

- **The co-text is the base, not the loss — measured (Δ0b).** What is open
  is only whether it needs unaddressed text to appear; if a fully
  addressed sentence still draws pseudo-text beside it, ΔFM trades the
  wipe for a defect of the same size and the line closes at Δ2.
- **The micro regime flatters everything.** 12 rows from scratch reach
  83 % at 500 draws; L0 is read as a lever here and confirmed at a
  mid-size inventory before any multi-hour run.
- **lr 2e-3 on a warm start.** The anchor recipe was tuned at 1e-3; Δ1 /
  Δ2 keep 1e-3 on warm rows unless L0's guard arm says otherwise.
- **`--free_residual` is unscaled against the paired loss** (5–8× smaller
  than plain): at 1e-3 the ΔFM norm peaked at 126 and sagged to 115. At
  2e-3 the norm reaches 144, so it is not gating; if a later ΔFM arm under-travels
  again, 3e-4 is the first thing to move.
- **Reader on dakuten / on co-text.** `single` exact held up against the
  sheets on Δ0; native `both` under-counts ΔFM images where the glyph is
  visible beside pseudo-text. The sheets stay the second read.
- **RAM.** Reference latents (+ 2.6 GB at 10 k items fp32, half in bf16)
  and captions (pool-bounded) on top of the text cache; the 10 k-item
  build stays the limit for Δ2 / Δ3.
- **The bubble as a unit** (`plan_synth.md`): unchanged by this line — the
  reference sits in the same bubble, so nothing here separates glyph from
  canvas.

## Not this plan

- **Cached local Jacobian, inner updates on the row** (the note's item 2).
  The objective is an expectation over noise; a linearisation at one
  (`z_σ`, σ) is not it, and inner steps on a cached `J` overfit the draw.
  One backward already gives the exact `Jᵀr` for ≈ 2 forwards, while k
  JVPs cost k forwards for a rank-k approximation. The grad_basis probe
  (2026-09-12) is the measured version: 3× first-step capture, flat in
  the blind A/B.
- **OCR-reward row search** (item 3). Reward through a sampler plus a
  reader with a domain gap (the か/日 reversal that totals hid) is
  correction, not learning; reward hacking is the failure mode. Kept as a
  last-mile idea for rows with one consistent wrong read, after this line
  has a result, and only with per-glyph sheet reads.
- **Contrastive ΔFM** (push `v_θ(B)` away from `v_θ(A)`). Different sign,
  different aim (mode separation); this line matches the delta, it does
  not repel. Contrastive terms on text-free natives are already closed.
- **The romaji as the reference**, a shape-neighbour kana as the reference
  (the encoder / IDS verdict — addresses are not shape coordinates, and
  the reference is not an address here anyway), the un-erased scene anchor
  or a blank box as the reference (table above).
- **Warm-starting the new row at the reference's row.** That is the
  transplant / warm-start line (closed 2026-09-16: inherited trigger buys
  no steps). Random init + this loss; `--init_anchor` only for rows a
  source table already has.
- **A flat-first curriculum** (flat canvas → composite ΔFM → sentences;
  asked 2026-09-17). Twenty steps is 80 draws against ≈ 1 000 per row —
  flat is fast in per-step loss, not in identity; a flat-trained `f` does
  not carry to composites (transplant 0/64 vs composite-trained 23/64;
  rows are canvas-conditional, which is why the S line exists), so stage 2
  either erases it (no anchor) or pins it to the wrong direction (anchor);
  and ΔFM's cancellation is the *scene* residual, which flat canvases
  barely have — it makes composite training cheap directly, so composite
  from step 1 with flat at the 0.1 guard share stays the shape.
- **lr 3e-3.** Closed (`findings.md`: off-manifold at 2.4× row norm); L0
  tests 2e-3 only.
