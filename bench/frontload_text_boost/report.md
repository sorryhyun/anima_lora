# Front-loaded text-drive boost — Phase-0 + Phase-1 report

Proposal: `_archive/proposals/frontload_text_boost.md`. Run:
`results/20260706-2135-phase0/` (2026-07-06, base DiT, single seed 0).
Eye-adjudicated on the contact sheet + per-prompt strips — no quality reward
exists for Anima; the montage is the gate (same discipline as
`_archive/bench/cross_attn_drive/report.md`).

**Headline: arm (b) — cross-attn residual gain, cond-pass-only, σ ≥ 0.85 —
passes G1+G2. Arm (a) — σ-gated CFG — fails G2 at every strength that does
anything, and is CLOSED.** The self-attn-dominance prior predicted arm (b)
inert; it is decisively not. This is the first *causal, exploitable*
confirmation of the front-loading finding
(`docs/findings/crossattn_self_attn_dominance.md`): amplify the text voice
only in the plan-writing window and the plan actually changes; self/MLP then
render the corrected plan in the normal style.

## Setup

- 17 prompts (`prompts.py`): 12 complex (split *base-weak concepts* vs
  *rare combos/relations*), border reproducer (with `cropped` kept in,
  worst-case for arm (a)), 4 ordinary regressions. Hand-picked;
  `tag_influence.py` deliberately not used for selection (its own Phase-0
  gate never ran).
- Arms, matched seed + NFE (28-step er_sde, flow_shift 3.0, production
  worded negative, guidance 3.5): baseline / cfg_hi 7 / cfg_hi 10 /
  xattn 1.5 / xattn 2.0.
- Band σ ≥ 0.85 = **10 of 28** shifted-schedule steps — NOT the "2–3 steps"
  the proposal text implied. Nobody has run a tighter band yet (σ ≥ 0.95
  ≈ 4 steps).
- Arm (b) mechanism: per-block non-persistent `_xattn_gain` buffer scaling
  the cross-attn residual inside `Block._forward` (compile-safe `fill_`,
  mod-guidance buffer pattern; invariant tests `tests/test_xattn_gain.py`).
  Applied on the **conditional forward only** — the boost lands entirely in
  the guidance delta, none of it spent on the negative prompt.

## Per-prompt reads (single seed — trust direction, not magnitude)

Baseline-FAILED watch tags, by arm (✓ fixed / ✗ still failed / — n/a):

| prompt | watch (failed at baseline) | cfg_hi 7 | cfg_hi 10 | xattn 1.5 | xattn 2.0 |
|---|---|---|---|---|---|
| band_trio | red=drums/blue=bass/blonde=keys binding | ✓ (style collapse) | ✗ | **✓ style kept** | **✓ style kept** |
| shared_scarf | one scarf around BOTH girls | ✓ (style collapse) | ✓ (style collapse) | **✓ style kept** | **✓ style kept** |
| shoulder_carry | carry ON SHOULDERS | ✗ (in arms) | ~ | **✓** | **✓** |
| kyudo | yugake draw-glove | ✗ | ✗ | ✗ | **✓ (glove appears)** |
| armor_apron | helmet OFF, sitting on log | ✗ | ✗ | ✗ | **✓ (uniquely)** |
| theremin | theremin (played no-touch) | ✗ | ✗ | ✗ | ✗ |
| umbrella_goldfish | umbrella upside down, fish inside | ✗ | ✗ | ✗ | ✗ |

Score on baseline-failures: **xattn fixes 5/7** (2.0 strictly ≥ 1.5);
cfg_hi fixes at most 2/7 and only by trading the whole rendering style.
Prompts where baseline already passed (flambé, mechanic, crystal_cat,
jellyfish_rain): xattn ≈ baseline or mildly better (more chalkboard text,
cleaner wrench-in-mouth); cfg_hi degrades style on ALL of them.

The two universal fails are exactly the predicted capability-limit class:
concepts with ~no signal to amplify (multiplicative lever × ~0 = ~0). The
lever moves **relations/bindings between things the model knows**; it does
not conjure unknown concepts. Same law as
LoRA training (cross-attn only discriminates labeled tags), now seen at inference.

## G2 (no-harm)

- **xattn: PASS.** Regressions ≈ baseline in style and quality; border
  reproducer shows no border/burn. Two honest caveats: mild global
  desaturation at 2.0 (sat −8.1%, contrast −3.0% — direction opposite to
  burn, but real), and small identity drift on regressions (a cap appears in
  reg_autumn; bg swaps in reg_sakura). Mechanistically consistent side
  effect: on border_repro the boost amplified the `cropped` tag — heads
  framed out. The lever amplifies **all** tags, wanted or not.
- **cfg_hi: FAIL, decisively.** Style collapse to flat lineart / poster /
  pixel-art on complex AND ordinary prompts (reg_sakura → pink monoline;
  reg_autumn → flat vector **with a white frame border**; armor_apron @10 →
  pixel art). cfg_hi 10 is burn (sat +21.6%). This is the
  mode-simplification the limited-interval-guidance literature predicts for
  high-σ guidance, on schedule.

Aggregate tone (burn detector, Δ vs baseline over all 17):

| arm | Δsat% | Δcontrast% |
|---|---|---|
| cfg_hi 7 | −1.0 | **+14.4** |
| cfg_hi 10 | **+21.6** | +6.8 |
| xattn 1.5 | −3.0 | +0.8 |
| xattn 2.0 | −8.1 | −3.0 |

## Verdicts

- **Arm (a) CLOSED** per the proposal's kill criterion (no strength passes
  G2 while doing anything). Do not re-propose plain σ-gated CFG-up; any
  revival must solve the style-collapse mode first (per-block scheduling à
  la mod-guidance, or a tighter σ ≥ 0.95 band, neither tested).
- **Arm (b) G1+G2 PASS → promote to Phase 1** (compose-flag plumbing,
  SMC/spectrum interaction, turbo grid — turbo runs CFG 1.0 so only this arm
  applies there anyway). Phase 1 also gains **arm (c), token-selective boost**
  (σ-gated embedding-span scaling of weak-tag tokens only) — motivated by the
  amplifies-everything side effects and the loudness-vs-allocation mechanism
  question; see the proposal's Phase-1 section for the design + guard note vs
  the killed late-window `tag_boost_scale`.
- **Prior updated:** cross-attn being 6–15% of the block residual does NOT
  make it un-leverable at high σ. "Small voice" ≠ "no vote" in the window
  where the plan is written.

## Caveats / owed

- **Plain grid is n=1 seed** (artist prompts are 3-seed confirmed, and every
  seed-0 artist read held or sharpened at 3 seeds — direction trustworthy).
  A multi-seed pass on the plain binding prompts (band_trio, shared_scarf,
  shoulder_carry) would fully retire this caveat.
- λ landscape unmapped: 2.0 strictly dominates 1.5 on adherence here, at the
  cost of −8% saturation. Where adherence saturates vs where desaturation
  becomes objectionable is unknown (λ ∈ {2.5, 3} untested).
- Band unmapped: 0.85 (10 steps) is the only band tested.
- Secondary mechanism probe (tag_influence rerun with boost active) skipped
  by choice — selection machinery unvalidated; revisit only if a mechanism
  dispute actually arises.

## Artist-tag addendum (run `results/20260706-2157-phase0-artist/`)

Real-caption-format prompts (`rating, count, @artist, tags…, sentence`),
reusing content the plain grid already discriminated on, so the delta
isolates (1) what the artist tag changes and (2) whether the boost preserves
artist style while fixing the same relation/binding. First-pass eye read
(artist-style *fidelity* needs the user's adjudication — the reader below
only scores content/composition):

| prompt | artist | content | xattn read | cfg_hi read |
|---|---|---|---|---|
| artist_kyudo | @yaegashi nan | kyudo + yugake | scene adherence ↑ again — baseline ignores `dojo` (white void + tatami strip), xattn renders the actual dojo interior; muneate clear; composition shifts full-body → close-up | **no style collapse** (unlike every plain prompt); composition ≈ baseline |
| artist_scarf | @jumonji | shared scarf + gift | relation already solved at baseline; xattn keeps it but **crops heads out of frame** (composition degradation) | no collapse; fine at both strengths |
| artist_band | @chen bin | red/blue/blonde role binding | muddier than plain band_trio — partial binding at 2.0 (sticks in red's hand, blonde at keys) but no arm gets a clean 3-way | partial binding, mild flattening only |

Addendum findings — **confirmed/revised on the 3-seed rerun**
(`results/20260706-2206-phase0-artist-3seed/`, seeds 0/1/2, seed 0
reproduces the single-seed run):

1. **REVISED — artist tags only *attenuate* arm (a)'s collapse, they don't
   prevent it.** The n=1 read said cfg_hi stops collapsing under @artist;
   at 3 seeds the collapse is back and seed-dependent: artist_kyudo seed 1
   goes B/W lineart at BOTH strengths, artist_band collapses at cfg_hi 10 on
   2/3 seeds (flat poster; one seed even turns the band to silhouettes).
   Arm (a) stays closed, now with artist-tag evidence too.
2. **CONFIRMED — the artist tag alone improves content adherence at
   baseline** — artist_scarf's baseline gets the shared-scarf relation on
   3/3 seeds (plain baseline failed it). Richer caption context = better
   plan, before any boost.
3. **CONFIRMED — xattn's scene-adherence gain is robust**: dojo interior on
   6/6 artist_kyudo renders (2 λ × 3 seeds) vs 0/3 baselines (white void);
   yugake gloves visible in most xattn renders. **The framing drift is
   equally robust**: close-up crop on 6/6 kyudo renders, faces
   hidden/cropped on ~2/3 scarf seeds. Same law as the `cropped`-tag
   amplification on border_repro — the uniform gain boosts *everything* in
   the caption, including the artist's framing habits.
4. **NEW — style tokens dilute the boost's content wins.** Plain band_trio:
   xattn fixed the 3-way role binding cleanly. artist_band (@chen bin, same
   content): no arm achieves a clean binding on any seed — xattn at λ2 gets
   partial credit (drumsticks in the red girl's hand, 2/3 seeds) but never
   the full assignment. With style tokens in the caption, the uniform gain
   splits its amplification across them and the binding win attenuates.
   Together with (3), this is the strongest motivation for Phase-1
   **arm (c) (token-selective boost)**: aim the gain at content tokens only.
5. Aggregate tone: the single-seed xattn 2.0 +37.9% sat was scene-change
   noise, as suspected — at 3 seeds it's +6.1% sat / −2.3% contrast. No
   burn signal for xattn at either strength.

The WHERE-probe prediction ("artist trigger is a σ-stable switch, style
delivered downstream, so the boost should leave style alone") survives on
style *identity* — no obvious style amplification/degradation at strip
resolution across any seed — but fails on *composition*: framing priors
ride the boost. User eyeball on the full-res sheets is the final word on
per-artist style fidelity.

---

# Phase-1 report (2026-07-06, run `results/20260706-2229-phase1-3seed/`)

All 20 prompts × 3 seeds × arms {baseline, xattn 2.0, token 1.5, token 2.0},
matched seed/NFE (28-step er_sde, guidance 3.5, band σ ≥ 0.85, base DiT).
Arm (c) = token-selective boost: α on the annotated weak-content T5 token
rows of the cond embedding only (spans per prompt in `prompts.py::boost`,
audit trail in `result.json::boost_spans`; artist/style/framing tags never
boosted). NB the crossattn embedding rows are **T5 target positions** (LLM
adapter output), not Qwen3 positions — spans located with the T5 fast
tokenizer's offset mapping. The 3-seed pass also retires Phase-0's
n=1-seed caveat on the plain grid.

## Headline

**Both arms stand, and they split the mechanism question by category.**
Arm (c) matches-or-beats arm (b) on *relations/bindings between known
things* while eliminating essentially every side effect of the uniform gain
(style drift, framing-prior amplification, regression identity drift,
desaturation). Arm (b) keeps a real edge on *conjuring weak/rare items* and
whole-caption corrections. Read: **allocation is the win for bindings;
loudness still buys weak-concept assembly** — the two levers are
complementary, not redundant. The "(c) ≈ (b) ⇒ ship the simpler lever"
branch of the decision rule did NOT fire cleanly in either direction.

## (b) vs (c) per category (3 seeds each)

| prompt / watch | xattn 2.0 | token 1.5 / 2.0 |
|---|---|---|
| band_trio 3-way binding | 1/3 clean | **2/3 clean** (token 1.5 gets seed 2 where xattn fails) |
| shoulder_carry ON-shoulders | ~2/3 (one held-in-air) | **3/3 clean** |
| shared_scarf relation | 3/3 | 3/3 (tie) |
| wheelchair mid-dribble lean | 2/3, one over-cooked crash pose | **3/3, lean visibly amplified** |
| artist_band binding + style | binds s0 but style drifts sepia; no clean 3-way any seed | **binds s0 with @chen bin style ≈ baseline**; no clean 3-way any seed |
| kyudo yugake glove | **glove 2/3** (+ corrects pink→white dogi s2) | glove ~1/3 (token 1.5 s2 clear brown yugake; dogi stays pink) |
| armor_apron helmet-off-on-log | **helmet-on-log 2/3** (but also still on head — duplicated) | 0/3 (helmet stays on) |
| artist_kyudo dojo interior | **3/3** (+ persistent close-up framing drift) | ~2/3 partial, full-body framing kept |
| theremin (control) | fails 3/3 (vague antenna-ish props) | fails 3/3 ✓ predicted |
| umbrella_goldfish (control) | **1/3 actually assembles it** (s2: inverted umbrella, fish inside) | 0/3 |

The two Phase-0 "universal fails" are therefore not equally dead: at 3
seeds the whole-caption gain occasionally assembles a rare combo (1/3);
the span-selective gain never does. Consistent with the mechanism split —
(c) reallocates attention *among tokens already in play*; (b) raises the
whole text pathway's vote including global assembly.

## G2 (no-harm) — arm (c) is dramatically cleaner

- **Regressions**: token arms ≈ baseline in identity, composition, and
  style on all 4 regressions × 3 seeds, with only the annotated span
  mildly amplified (a somewhat bigger straw hat). xattn 2.0 re-confirms its
  Phase-0 drift (reg_sakura: pink hair → brown + sailor → blazer on 3/3
  seeds).
- **border_repro (the designed test)**: xattn amplifies `cropped` — heads
  framed out on 2/3 seeds. **Token arms keep baseline framing on 3/3** —
  the crop side effect is dodged exactly as designed (the `cropped`/`from
  below` spans are simply not boosted).
- **Tone** (Δ vs baseline over all 60 renders): xattn 2.0 −7.7% sat /
  −1.4% contrast (Phase-0's desaturation, reproduced); token 1.5 −1.8% /
  −0.6%; token 2.0 −1.6% / +1.0% — **no meaningful tone cost**.
- **Arm (c)'s own failure mode (new)**: boosting a *garment/object* span
  hard can make it dominate the composition — artist_scarf seed 1 turns
  into a scarf close-up with heads cropped out (token 1.5 and 2.0); the
  boosted "jellyfish hair ornament" span recolors the (unspecified) hair
  jellyfish-white. Span-anchored over-literalism instead of
  caption-prior amplification. Mitigation is span curation / lower α, not
  a different mechanism.

## Verdicts

- **Arm (b) ships** — production flag `--xattn_boost` / `--xattn_boost_band`
  landed with this phase (inline + tiled loops, spectrum/SPD/foveated
  runners via `SamplerSideChannels`, `XATTN_BOOST=` env lever,
  `docs/inference/xattn_boost.md`, invariant tests). It remains the
  zero-annotation lever and the only one that helps weak-item assembly.
- **Arm (c) validated at bench level** — strictly better aimed for
  binding/relation work and for prompts carrying artist/style/framing tags.
  Productionization needs a span-annotation surface (e.g.
  `--xattn_boost_spans "substr|substr"`) — deferred until a real use case
  asks for it; the bench path (`--token_values`) is the reference
  implementation.
- **Guard update**: the Phase-0c "rescale trap" kill for embedding-level
  tag boosts stays scoped to the late/low-σ window; the pre-commitment
  window (σ ≥ 0.85) is hereby positively confirmed editable at the
  embedding level too.

## Production-path verify + compose smoke (`--xattn_boost`)

band_trio through the real `inference.py` path (28-step er_sde, seed 42,
base DiT): baseline renders no drummer; `--xattn_boost 2.0` adds a
red-haired drummer with the binding largely fixed — the bench effect
reproduces through production plumbing. `--xattn_boost 2.0 --smc_cfg` and
`--xattn_boost 2.0 --spectrum` both complete cleanly with sane, distinct
outputs (no burn, no artifacts) — the compose matrix in
`docs/inference/xattn_boost.md` is smoke-validated. One observed side
effect: the boosted render drew 4 girls on a `3girls` prompt (count drift —
consistent with the amplify-everything law; the count tag rides too).

## Turbo addendum (run `results/20260706-2325-phase1-turbo/`)

4-step DP-DMD student (`anima_turbo_S`), euler, **CFG 1.0**, band σ ≥ 0.85
= steps at σ ∈ {1.0, 0.9} (2 of 4). 8 prompts × 2 seeds × {baseline,
xattn 1.5, xattn 2.0, token 2.0}. NB at CFG 1.0 there is no guidance
delta — the "cond-only" framing is moot and the single forward is boosted;
any effect here is the boost changing the plan directly, not sharpening a
CFG direction.

- **The lever survives distillation, and arm (c) is the star**: band_trio
  binds cleanly under token 2.0 on **2/2 seeds** (red at drums with sticks,
  blue bass, blonde keys) where baseline fails and xattn stays partial
  (xattn tends to add a 4th brown-haired drummer instead of rebinding the
  red one). Given the turbo students' known caption-discriminability loss,
  recovering a 3-way binding at 4 steps is a real result.
- armor_apron: helmet-on-log appears under xattn 2.0 and token 2.0 (s0)
  — same duplicated-helmet partial as the base model.
- **G2 mirrors the base model**: token 2.0 ≈ baseline on both regressions;
  xattn 2.0 drifts reg_sakura's uniform (white sailor → navy) — the same
  identity-drift signature. border_repro: no borders/burn anywhere; token
  2.0 framing is actually the safest (frontal, both heads in frame).
  Tone: xattn ≈ 0%, token 2.0 +6.8% sat (dominated by one bright pink
  bunny suit, not burn).
- theremin control fails under all arms, as predicted (token 2.0 adds a
  second performer on one seed — the over-literal span drift again, mild).

Verdict: `--xattn_boost` is usable on turbo checkpoints as-is; if a span
surface ever ships for arm (c), turbo is a first-class beneficiary (it
needs the binding help more than the base model does).

## Phase 1'' — allocation probe, in-D arms, norm matching (2026-07-11/12)

Three sessions of arms beyond (a)-(c); runs `20260711-2307-phase1p-alloc`
(killed partial, 8 arms), `20260711-2327-phase1pp-ind` (killed at 41/60 rows,
{baseline, xattn 2, renorm 2, pdg 2, pdg 4}), `20260712-0013-phase1pp-renormI`
(complete, {baseline, renormI 2, renormI 2@0.5}). Per-prompt+seed sheets
(`sheet_<id>_s<seed>.png`) replaced the mega contact sheet mid-phase.

**Geometry finding that reframed the phase**: `k_norm` (RMSNorm per
token×head) is scale-invariant ⇒ arm (c)'s embedding-row scaling never
touched attention allocation — (c) is architecturally a pure token-selective
**V/loudness** gain. Allocation needs a QK logit bias.

- **kbias β (arm d)** — sink-preserving logit bias (−β on non-span content
  rows; span/pad/EOS untouched; `Attention._ctx_k_bias` → SDPA additive
  mask). Allocation IS a distinct axis: binds theremin heterochromia 5/6
  cells where baseline/token miss; large layout moves. But β=1.4
  over-quiets (style flattening, unprompted text posters) and even β=0.7
  trades tone. **CUT by eyeball verdict; probe stays in the bench.**
- **token 1.5 / combo** — flat vs Phase-1; cut.
- **fade (arm f)** — cosine σ-window; deferred, parameterization polish.
- **pdg α (arm h)** — prompt-delta guidance `v* = v_cfg + α(v(c) − v(c\S))`,
  fully in-distribution (+1 cond fwd/in-band step). Solid, vivid, no tone
  damage; did not beat renorm on the binding/content read. Stays a bench
  upper-bound baseline.
- **renorm λ (arm g)** — norm-matched xattn boost. Per-TOKEN matching wins
  content but **greys the tone** (the tokens whose norm spikes under the
  boost ARE the neon/highlight peaks; clamping each token flattens the
  norm distribution; sat −31%). **Per-IMAGE mean matching (`renormI`) fixes
  it**: one shared scale per image, peaks survive (jellyfish neon glows,
  reg_city stays vivid), OOD energy budget still bounded. `renormI 2@0.5`
  (ρ=0.5 partial, `scale**ρ`) = the tone sweet spot.

**SHIPPED**: `--xattn_boost_renorm {off,tok,img}` (default `img`) +
`--xattn_boost_renorm_frac` (default 0.5) — i.e. `xattn_boost 2` now runs
renormI 2@0.5. Runner entry point `adapters.set_xattn_boost_state`; wired in
generation inline+tiled, spectrum, spd, foveated. Docs:
`docs/inference/xattn_boost.md`.

**Method caveats**: (1) the mean-HSV-sat burn detector misleads in the desat
direction — boosted arms replace flat saturated washes with detailed scenes,
so the mean drops while peaks brighten; judge tone by eyeball. (2) compiled
renders are NOT bit-stable across processes (inductor autotune + er_sde
amplification) — same-seed cross-run comparisons are family-level, not
pixel-matched; within-run columns are exact.

---

# Glyph × CFG-delta measurement probe (2026-07-16, run `results/20260716-2207-glyph-delta/`)

Script: `glyph_delta_probe.py`. Not a boost arm — a **measurement** feeding this
bench's one remaining open gate. Phase-1 left the *localized feature-commit-band*
boost gated on finding "a regime that demonstrably DROPS small features"; rendered
glyphs are that regime, so the prior question is whether the model puts any extra
text drive into the band when a glyph is present.

**It does — and the effect is glyph-specific, not a length artifact.** Base DiT,
14 real captions × 2 seeds × 5 arms, paired, production `ANIMA_CFG_DELTA_LOG`
probe (no hooks). Band σ∈[0.5,0.8], rel Δ‖v_cond − v_uncond‖:

| pair | isolates | rel Δ | n_pos |
|---|---|---|---|
| `E_nonglyph − B_category` | +10 tok, no glyph | −1.18% | 13/28 — null |
| **`D_glyph_id − E_nonglyph`** | **glyph vs not, 15 vs 16 tok** | **+6.77%** | **25/28** |
| `C_glyph − B_category` | OOD glyph | +8.55% | 25/28 |
| `C_glyph − D_glyph_id` | OOD token axis | +3.14% | 22/28 |

The high-σ peak is untouched (all arms ratio 0.275–0.280 at σ=1); the bump is new,
σ≈0.8→0.45, peaking σ≈0.66. Band cos stays 0.9991–0.9992 — **harder, not rotated**,
so CFG already spans it.

**Consequence for the gate:** text does *not* go quiet in the band on glyph prompts —
the model is already spending extra (collinear) drive there. That argues the glyph
deficit is upstream (the early low-freq plan never encodes it), not something a
band-localized gain multiplies into existence. Weakens, not strengthens, the case for
building the localized arm.

Full read + the caveats (probe is direction-blind *between* arms; `E` is a
conservative control): `docs/findings/crossattn_self_attn_dominance.md` Result 5.
