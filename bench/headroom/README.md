# bench/headroom — is the unprompted border a rendered self-attention sink?

Phase-0 investigation for the *headroom (register) tokens* proposal
(`_archive/proposals/headroom_register_tokens.md`, PR #68). **Design-only proposal;
this bench is the falsifiable kill gate that ran before any training or new
network code.** Verdict below: **RQ1 falsified — the border is a text-controllable
data prior, not a relocatable sink.** The DSR sink itself is real, always-present,
and load-bearing, but decoupled from the border.

## The hypothesis

Unprompted white/black borders are a *self-attention sink rendered into pixels*
(the ViT-registers / DSR mechanism, arXiv:2605.05206): softmax must spend its mass
somewhere, so when nothing matches it piles into the lowest-information region — a
uniform border — and because low frequencies commit early, the sink is baked into
the layout. Registers would give the model somewhere off-canvas to sink instead.

## Scripts

| Script | Question | Signature |
|---|---|---|
| `sink_probe.py` | Is there a border-located self-attn sink? | per-token residual ‖x‖ (ViT/DSR high-norm outlier), self/cross split, swept over **layer-depth × σ**; border enrichment; pixel correlation; `cropped` knockout |
| `border_toggle.py` | Does the sink track the border, at fixed noise? | matched-seed **border toggle** (induce with `black border`, suppress with a border negative); sink enrichment ON vs OFF |
| `sink_intervention.py` | Is the sink load-bearing / a quality artifact? | clamp mid-layer outlier ‖x‖ to `ratio×median` during generation, matched seed; eyeball base-vs-clamp (no Anima quality reward exists) |

Run e.g. `uv run python bench/headroom/sink_probe.py --seeds 30 --label sweep`.
Results land in `bench/headroom/results/<ts>-<label>/` (result.json + rows.csv + images).

## What we found

### 1. The border is real but a rare stochastic event (~1/6 seeds)

On the user-confirmed reproducer (a 2-girl bunnysuit bar prompt carrying the tag
`cropped`), thick near-black
letterbox bars appear on ~1 in 6 seeds. `cropped` is a **muddy proxy** — in tag
semantics it means "subject cut off by frame," not "rendered bar." Seeds that border
on this prompt at 1024/28/cfg4: **4, 10, 23, 25, 29**.

### 2. There IS a strong self-attn sink — but it is border-decoupled

`sink_probe.py` (30 seeds): a **~26× high-norm ‖x‖ outlier** token, concentrated in
**mid (intermediate) layers**, strongest at **high σ**, **self-attn-written**
(self-contribution ≫ cross). It is a *sparse* set (~7 tokens of 4200). So the DSR/ViT
sink phenomenon is real on Anima.

But it does **not** track the border. The 30-seed border-conditioned contrast and the
matched-seed toggle both show the sink present *identically* whether or not a border
renders:

- `border_toggle.py`: the border is **fully text-controllable** — a `black border,
  letterboxed` positive induces it on 5/5 clean seeds; a border negative erases it on
  5/5 bordering seeds (verified by eye + detector). Yet mid/hi sink border-enrichment
  barely moves across the full ON→OFF toggle: ON 1.92× vs OFF 1.75×, **Δ=0.17×** (≪
  the 0.50× bar); matched border-removal is inconsistent in sign.

A softmax attention sink is not erasable by a *text* negative. Text-controllability is
the signature of a **learned data prior** (letterbox/matte baked into training art),
not a rendered attention sink.

**⇒ RQ1 FALSIFIED.** This is the proposal's own kill branch: *"the sink is not
co-located with borders ⇒ borders are a data-prior problem; the honest answer is
crop-conditioning or border augmentation, not headroom."* The border-specific register
line is closed. σ-direction footnote settled: DSR's "intermediate layers" + the doc's
"high σ" were each half-right, but moot since the sink is border-decoupled.

### 3. The sink is load-bearing and global (quality follow-up)

`sink_intervention.py` asks DSR's *actual* question — is the sink load-bearing? Anima
has no per-image quality reward and the sink is in every image, so the only honest test
is the DSR masking test: clamp the outlier ‖x‖ during generation (matched seed) and
look. Clamping just **~0.16% of tokens (~7 of 4200)** re-plans the *whole* image
(16% pixel L1 on the complex prompt, change spread **off** the sink patches at conc
0.70×; near-identical 5% on the simple prompt). Quality is **roughly preserved** — one
coherent image traded for another comparable one, with minor new artifacts (a malformed
hand, faint speckles); no free quality win, no collapse. This **replicates DSR Tab. 1**
(masking the outlier doesn't help — it's a symptom/carrier) on Anima: registers would
have to *absorb* the sink, not remove it.

## Where this leaves RQ2/RQ3 (thread, not built)

The border angle is dead, but two facts reopen a *general-quality* register question:

- **The sink is load-bearing** ⇒ registers must absorb it, and "does it help" can only
  be settled by training + a quality metric Anima lacks.
- **Soft-tokens already work on this model** (frozen DiT adopts trained non-decoded
  tokens spliced into its attention stream on a small budget) ⇒ the RQ3 adoption risk
  (the DINOv2 "needs scale" fear) is largely retired. Registers ≈ "soft-tokens on the
  self-attn sequence."

So the sensible (unbuilt) redesign flips the order to **RQ3-first**, and leans on the
probes above to dodge the missing-quality-reward wall:

- **RQ3 (adoption, cheap, metric-free):** train a soft-tokens-style register adapter on
  the self-attn stream at the mid blocks. Metric from `sink_probe.py`: does the
  **image-patch** sink norm fall while **register** norm rises (the relocation
  crossover)? Soft-tokens says it should.
- **RQ2 (benefit, via proxies):** (a) do the **ex-sink patches recover local detail**
  after relocation (DSR's core prediction, per-patch Laplacian, no global reward);
  (b) does the **load-bearing sensitivity drop** — `sink_intervention.py` on the
  trained model should perturb the image *less* per unit clamp, because the load moved
  off-canvas; (c) CMMD + matched-seed eyeball as the aggregate tiebreak.
- **Clean negative worth publishing:** relocation happens but nothing recovers and CMMD
  is flat ⇒ "the DSR sink exists and is adoptable on Anima, but is load-bearing without
  being quality-limiting here" — the thing that distinguishes Anima from the ImageNet
  DiTs DSR studied.

## Phase-0.5 (RQ3-first): BUILT AND RAN — no relocation at the LoRA budget (2026-07-01)

The RQ3-first redesign above was built and run (`register_adapter.py`,
`train_registers.py`, `register_eval.py`). A true residual-carrying **register**
on the self-attn stream (concat once at `_run_blocks` entry, carried through all
28 blocks, stripped before unpatchify; forced eager native-flatten, no compile;
rope-exempt via identity cos/sin rows). Two arms adjudicate Theory 1 vs 2:
**arm A** = K fixed-zero registers + self-attn QKV-LoRA (routing only), **arm B**
= K *learnable* registers + the same LoRA (routing + content). Config: K=16,
QKV-LoRA rank 8 on all 28 blocks, 400 steps, lr 1e-3, ~500 cached latents ≤1024
tier, grad-ckpt (`use_reentrant=False`). Probe = mid block 14.

**Result — RQ3 negative at this budget; no relocation crossover.**

| cond | patch_sink_ratio (top-0.2% / median) | reg_ratio (maxReg / median) |
|---|---|---|
| base | 23.85 | — |
| arm A (fixed-zero) | 23.53 (−0.3) | 1.015 |
| arm B (learnable)  | 21.36 (−2.5) | 1.115 |

- **Metric cross-validated:** base 23.85 independently reproduces the ~26× sink
  from `sink_probe.py`.
- **The sink does NOT relocate.** Registers never become high-norm sinks
  (reg_ratio ~1.0–1.1× median, not ~24×); the image-patch sink barely moves
  (arm B −10%, arm A −1%). The load-bearing self-attn sink stays rendered.
- **Learnable > fixed, small but real:** arm B drops the patch sink 8× more than
  arm A and carries higher register norm — content matters in the *right
  direction* (Theory 2), but nowhere near absorption.
- **No quality collapse** (matched-seed eyeball + border guard): arm B re-plans
  the image into another coherent one — the `sink_intervention` "no free win"
  pattern, not a bland/borderless failure.

**Interpretation.** A frozen base + few-hundred-step LoRA-budget adapter does not
relocate the baked-in sink onto off-canvas registers — the proposal's own named
fear, and the DINOv2-needs-scale caveat, realized. This is the publishable
negative: *registers help only above the LoRA budget ⇒ headroom belongs in the
base, not a cheap adapter.*

**Provisional — one ablation point.** K=16 / all-blocks / 400-step / probe-block-14.
The DSR sweet spot (~36 registers **at block ~8**, non-monotonic; unfrozen QKV;
longer schedule) is UNtested — the "different, more expensive experiment" the
proposal says must be reported separately. A negative here says "not at this
budget," not "never on Anima."

**Recommendation:** the border line is closed (RQ1) and the cheap-adapter
general-quality line is negative-at-budget (RQ3, above). Reopen only via the
fuller experiment (block-8 insertion + K≈36 + QKV unfreeze) or if an Anima
per-image quality reward appears (`sink_intervention.py` remains the reopener).

## Follow-up (2026-07-02): RQ2 proxies on the trained arms — the "different images" are LoRA drift

The trained arms re-plan matched-seed images heavily even untuned, which looked
like register promise. `register_rq2_proxies.py` ran the two RQ2 proxy tests on
the frozen RQ3 checkpoints (base / armA / armB, matched seed, repro+control
prompts × 3 seeds, 1024/28/cfg4), with arm A (fixed-zero registers + same
QKV-LoRA) as the **LoRA-drift control**, plus direct base-vs-arm image L1:

| cond | patch_sink_ratio | clamp pix L1 (med) | re-plan vs base (L1) |
|---|---|---|---|
| base | 14.5 | 0.158 | — |
| armA | 14.2 | **0.057** | 0.13–0.24 |
| armB | 12.9 | 0.074 | 0.13–0.20 |

(sink_ratio ~14.5 here vs 23.85 in `register_eval` is prompt-dependence —
repro/control vs plain/scene/portrait — the outlier structure is the same.)

- **The re-plan is NOT register content.** Arm A — effectively LoRA-only —
  re-plans matched-seed images as much as (slightly more than) arm B. The
  visual difference between trained arms and base is generic QKV-LoRA drift.
- **Proxy (b) negative.** Clamp sensitivity (`sink_intervention` clamp on the
  trained model) drops in BOTH arms — and *more* in arm A. The drop is what
  400 steps of FM training on a QKV-LoRA does, not off-canvas load relocation.
  `clamp_frac` is ~equal across conditions, so it's not "fewer tokens clamped".
- **Proxy (a) noise-dominated.** Per-patch Laplacian detail at (ex-)sink
  locations spans 4 orders of magnitude across rows; at re-plan L1 ~0.15 the
  composition churn swamps any local recovery. No arm-B-specific recovery.
- The only surviving register-specific signal remains the small, consistent
  sink-ratio drop (armB −1.3 to −2 vs base; armA −0.3) — which translates into
  nothing measurable downstream.

⇒ **Strengthens the RQ3 negative:** at the LoRA budget nothing register-specific
reaches pixels. Reopen conditions unchanged (K≈36 @ block 8 + unfrozen QKV, or a
per-image quality reward). Run: `results/20260702-0659-rq2_proxies/`.

### Arm L addendum (same day): arm A is a LESION, not a control — eyeball tally

Eyeballing the run above found something the pixel metrics missed: **arm A
systematically collapses framing** (3/3 broken headless extreme-crops on repro,
1/3 face-crop on the "full body visible" control), while arm B stays coherent.
Since arm A ≠ neutral LoRA-only (its 16 zero-content registers all project to
the identical bias-derived key = a uniform attention-mass drain the LoRA trains
around), we trained the missing **arm L** = true LoRA-only, K=0 (`--arm A
--num_registers 0`; seed-identical LoRA init/data order to arm A) and evaluated
it alone (`--only armL`; matched seeds make cross-run image comparison valid).
Runs: `results/20260702-0715-rq3_armL/` + `results/20260702-0725-rq2_armL_only/`.

Coherent-framing tally on repro (which carries the muddy `cropped` tag):
**base 2/3 · armA 0/3 · armB 3/3 · armL 3/3.** Arm L re-plans vs base just as
much as the other arms (L1 0.14–0.23) and matches armB's clamp sensitivity
(0.076 vs 0.074; armA's lower 0.057 reflects its degenerate flat close-ups, not
extra relocation). Arm L's sink ratio 14.2 ≈ base 14.5 ≫ armB 12.9.

Adjudication:
- **"arm B ≫ arm A" = the zero-register LESION hurts, not "registers help".**
  Arm B ≈ arm L ≈ base in coherence; registers-with-content are merely harmless
  at this budget, with no benefit over plain QKV-LoRA.
- The lesion is itself a mechanistic finding: injecting a content-free
  uniform-key mass drain specifically breaks **crop/framing (global layout)** —
  causal support for the sink-owns-layout-bookkeeping reading, and for DSR's
  registers-must-carry-content (Theory 2 over Theory 1). Never ship fixed-zero
  registers.
- Arm B's small sink-ratio drop (12.9 vs armL 14.2) is now the ONLY cleanly
  register-specific effect — LoRA-controlled — and still downstream-inert.
- Meta-lesson: matched-seed "quality" eyeballs need a framing/coherence check,
  not just pixel L1 / Laplacian medians — a coherent-looking wrong crop is
  invisible to both.
