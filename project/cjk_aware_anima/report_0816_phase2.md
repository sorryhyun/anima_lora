# CJK-aware Anima — Phase 2 report

Measured verdicts from the distillation loop's unit gates (2026-08-15, G2
re-run 2026-08-16) **and the Phase-2c first pass** (2026-08-16, from
[G5](#g5--the-flat-gate-is-measured-blind-2026-08-16) on). Run envelopes:
`bench/cjk_distill/results/` and (rendered eval)
`bench/cjk_adapter/results/`. Code: `scripts/distill_cjk/`.

*Line home: [`motivation.md`](motivation.md) (why) · [`done.md`](done.md)
(what exists) · [`plan.md`](plan.md) (design, phases, gates).*

## What ran

| Gate | Question | Verdict |
|---|---|---|
| **G0b** `--mode oracle` | student ids := teacher ids ⇒ loss ≡ 0? | **PASS** — worst `1-cos` 2.9e-4 (bf16 floor). Also certifies the trimming invariant the cache rests on: a non-pad adapter output does not depend on how many pads follow it. |
| **G1** pytest | EN bit-identical? | **PASS** — pure-EN text tokenizes to stock spiece ids exactly; the split embedding returns stock rows bitwise, before *and* after the ext parameters move. |
| **G0** `--mode capacity` | can ext rows *express* the teacher at all? | **PASS** — 32 pairs, loss 0.574 → **0.0244**, monotone. No escalation to 2-ii (adapter LoRA) needed. |
| **G2** `g2.py` | which loss × which parameterization? | **PASS** (2026-08-16) — `span` at `param=global`. The 2026-08-15 attempt was withdrawn; three instrumentation defects had to be fixed first, below. |
| **G3** `g34.py` | is 0.6 even the right 2c number? | **PASS** (2026-08-16) — the `tags` teacher ceiling is **0.823**, so the gate asks for 73% of it. It also found that the readout-space floor is register-dependent by 100×, which makes `recovery_attn` a mix statistic. |
| **G4** `g34.py` | corpus health; does translation noise hurt? | see below |
| **2c first pass** | train the packs, render the grid | **RAN** (2026-08-16) — packs trained to span-loss saturation; renders split cleanly by per-row supervision density. See [Phase 2c](#phase-2c--first-pass-2026-08-16). |
| **G5** `g5.py` | is the flat 0.6 gate reachable *under the settled objective*? | **NO — the gate is demoted, not the student.** The exact argmin of `L_span` scores **0.13** flat on `tags` (generous variant 0.31) against the 0.6 gate; the same oracles sit at 0.71–0.97 in the readout space. |

## The three instrumentation defects

The first G2 answered its design questions but its headline metrics were not
trustworthy, and the diagnosis recorded at the time was itself wrong. All three
faults are fixed; the numbers in the next section are from the corrected loop.

**1. The probe queries were random.** `attn_bank` synthesized them with
`torch.randn`. Queries are `q_proj(image tokens)` — they only exist during a
forward and cannot be read out of a checkpoint. `build_query_bank.py` now taps
the real thing: 32 DiT forwards on cached latents and cached post-adapter
contexts at σ ∈ {0.3, 0.6, 0.9}, hooking each sampled block's
`cross_attn.q_norm`, banking 768 real token queries per block into
`bench/cjk_distill/assets/query_bank.safetensors`. `build_bank` refuses to run
without it (`--allow_random_queries` reproduces the withdrawn run). Two traps
worth knowing: `load_anima_model`'s `device=` does not reach the runtime
buffers, so a directly-loaded DiT keeps a CPU mod-guidance schedule and dies in
`_run_blocks` (the shared harness's `.to(device)` is load-bearing); and a suffix
match on `blocks.0.cross_attn.q_norm` hits **`LLMAdapter`'s own block 0** first,
which is a text→text attention that does not even run on this path.

**2. The recorded root cause was wrong, and real queries alone made things
worse.** The withdrawn report blamed near-uniform attention ("a random query
attends almost uniformly, so the readout degenerates to a near-mean over the
sequence"). Measured: attention is **sharp** under both banks — 3–9 effective
tokens of ~99 real ones, with 80–87% of the softmax mass sitting on the zero-pad
sink. The actual defect is that every readout carries a large **common offset**
(`‖mean‖/‖vec‖` = 0.73 with random queries, **1.02** with real ones), and an
*uncentered* cosine over vectors sharing an offset that big saturates at 1 for
everything:

| step-0 metric | random q | real q, raw | real q, centered |
|---|---|---|---|
| `cos_native_vs_en_attn` (the floor) | 0.287 | 0.997 | **−0.031** |
| `cos_teacher_vs_en_attn` (the ceiling) | 0.916 | 0.999 | 0.841 |
| far discrimination, readout space | 0.374 | 0.999 | 0.269 |

The offset is real conditioning, but it is *shared* — teacher and student both
have it, so matching it is free and carries no information. `fit_centers` fits it
once on the frozen teacher outputs (arm-independent, batch-independent) and
`readout()` projects it out. This repaired `L_attn` as an **objective** as well
as a metric: uncentered, two unrelated prompts read 0.997 alike, so the loss had
almost no dynamic range about wording.

**3. The holdout was split by pair, not by image.** Every image contributes
several pairs that share tag content and differ only in wording (`tags` /
`tags_alt`, plus D6's two quote registers). A per-pair shuffle therefore
(a) **leaked** each held-out pair's sibling register into training ~91% of the
time — "held-out" was measuring generalization to new *wording of trained
content*, not to new content — and (b) left **exactly one** near pair in the
256-record eval slice. `discrimination_near` was a single-sample statistic; that
is the provenance of the **0.71 zero-shot near figure** previously quoted in
`plan.md`. Split by image it is **0.411 over 72 pairs**. `load_pairs` now groups
by image and logs the near-pair count so this cannot silently regress.

## G2 — the measured cross-tab

6 arms, staged (loss chosen at `param=global`, then parameterization at the
winning loss), 1500 steps × batch 32, 5,728 training pairs (`tags` +
`tags_alt`; D6 excluded — its teacher is degraded by construction).
Envelope: `bench/cjk_distill/results/20260816-1152-g2/`.

| param | loss | recovery_attn | disc far | disc near | held span | held attn | held flat | held pool |
|---|---|---|---|---|---|---|---|---|
| _(zero-shot)_ | — | 0.516 | 0.111 | 0.411 | 0.654 | 0.623 | 0.899 | 0.445 |
| global | `flat` | 0.758 | **0.304** | **0.910** | 0.590 | 0.392 | 0.753 | 0.279 |
| global | `span` | 0.974 | 0.085 | 0.394 | **0.120** | 0.123 | 0.859 | 0.163 |
| global | `attn` | 0.967 | 0.088 | 0.380 | 0.334 | **0.082** | 0.876 | 0.172 |
| global | `attn+span` | 0.962 | 0.094 | 0.392 | 0.173 | 0.076 | 0.869 | 0.165 |
| global_row | `span` | **0.975** | 0.089 | 0.392 | 0.107 | 0.120 | 0.858 | 0.162 |
| row | `span` | 0.923 | 0.101 | 0.408 | 0.338 | 0.318 | 0.879 | 0.280 |

- **`flat` is disqualified**, and now on a near metric that can actually see it:
  far 0.111 → **0.304** and near 0.411 → **0.910**. It buys recovery by pushing
  every prompt's conditioning toward one direction — risk 6 in `plan.md`. Both
  halves of the stratified discrimination catch it. (In the readout space its
  near reads exactly 1.000, so the *flat*-space discrimination is the collapse
  guard that matters.)
- **`span` wins, and the cross-tab is what shows it.** Each objective wins on
  its own held-out term, as expected — the question is what it costs on the
  others. `span` scores 0.123 on the attn term against `attn`'s own best 0.082;
  `attn` scores 0.334 on the span term against `span`'s own best 0.120. Span
  transfers, attn does not. `span` also wins `recovery_attn` (0.974 vs 0.967) —
  on attn's home turf. Adding attn to span (`attn+span`) does not help
  (0.962). **Ship `span`.**
- **The global correction does the work; per-row residuals do not.** `global`
  0.974, `global_row` 0.975, **`row`-only 0.923** — per-row freedom adds 0.001,
  removing the shared map costs 0.051. This reconfirms the 2-i-a hypothesis with
  trustworthy metrics: the zero-shot table's error is systematic (the anchor map
  was fit on non-CJK anchors), not 58,968 independent per-row errors — so the
  95% of rows the corpus never visits still move. `global_row` is nominally the
  winner but is inside the noise of `global` at 1,887 extra tunable rows;
  **prefer `global`** unless a later corpus makes the residuals earn their keep.
- **Discrimination stays healthy on every honest arm**: far ends at 0.085–0.101,
  *below* the 0.111 zero-shot baseline and far below the 0.2 gate. Near ends at
  0.380–0.408, i.e. wording still reaches conditioning.

**Do not read `recovery_attn` ≈ 0.97 as "97% done."** The readout is a heavy
compression (64 queries × 3 blocks) and is permutation-invariant by design, so a
student that carries the right content under a different segmentation scores
high there and low in flat space — which is exactly what happens: flat recovery
is **0.066** and `cos_student_vs_en` is 0.096 against the teacher's 0.777. The
Phase-2c gate is the existing bench's `cos_vs_en ≥ 0.6` on rendered prompts, and
nothing here says that gate is close. What G2 settled is the *design* — which
loss, which parameterization — not the distance to 2c.

## Next

*Phase 2b closed 2026-08-16 (G0b, G0, G1, G2, G3, G4 green); the 2c first pass
ran the same day and G5 re-based its gate. This list is the post-2c state —
the 2b-era version of it (superseded 2026-08-16) ordered the objective decision
first; the render grid re-ordered the levers because the tag-register failures
turned out to be coverage, which is unblocked now.*

1. **Widen D1 with a coverage floor — the highest-value work, and it is
   unblocked.** The render failures are measurably thin-row failures
   (`騎`/`鎧` = 0 visits, `博麗` = 2 — see the
   [coverage table](#the-render-failures-are-coverage-token-by-token)), and
   G4b already showed the corpus is supervision-starved. The source exists:
   `~/gelcrawl/retrieved/` holds **16,053** EN tag captions against the 3,008
   `image_dataset/` captions D1 is built on (5.3×), text-only so curation
   state is irrelevant, and the crawler can fetch more caption-only. Target it
   with `gates/coverage.py`: compose until no user-facing content token
   (caption_index.json + the D5 lexicon) sits under a visit floor —
   the working head suggests O(100+) visits; identity-carrying tokens need it
   most.
2. **Mint a name register from the D5 lexicon.** Character identity is the
   hardest render target and gets the thinnest supervision (risk 4 measured:
   MT transliterates names, so 博麗霊夢's rows saw 2–37 visits and the render
   lost Reimu while the teacher nailed her). The lexicon already pairs
   thousands of names — compose name-bearing captions directly rather than
   waiting for them to surface in D1.
3. **The objective decision still gates D2/D3/D4** (unchanged from the 2b
   handoff): prose registers carry no spans, contribute zero gradient under
   `loss=span`, and only pay through a sequence-level term (`attn+span` lifts
   commentary 0.096 → 0.109 in its own register, tags unharmed). Decide that
   before sizing any prose corpus. The flat probes rule out `flat` as the
   extra term (see [the probes](#the-two-flat-probes--the-gate-cannot-be-bought)).
4. **Score 2c on the re-based surface, not flat `cos_vs_en`** — G5 demoted the
   flat gate to a control (its oracle argmin scores 0.13 against the 0.6 bar).
   Acceptance = rendered-grid parity with the teacher + per-register
   `cos_student_vs_en_attn` (fixed holdout mix, per G3) + the coverage floor.
5. The **owed D6 instrument** (same template, different quoted strings) is
   still owed; whatever measures glyph contrast will not be a cosine in this
   space (G3: ~0.02 of readout headroom).
6. **`tag_glossary_review.md` sign-off is still open** and G4b does not close
   it — a per-row correctness question, not an aggregate-loss question. Still
   a ship blocker.

## D2 — what the commentary corpus buys (2026-08-16)

First measured slice of D2: **9,068 pairs** (5,721 JA→EN by Hy-MT2-7B greedy +
3,347 free human translations), from a partial MT pass stopped at ~6k cached
rows out of 69,668 candidates. Envelopes:
`bench/cjk_distill/results/20260816-1400-g2` (control) and
`…-1409-g2` (+D2). Both share one cache (18,090 train / 900 holdout) and one
seed; the only free variable is `--train_registers`.

**Coverage — D2 more than doubles the reachable table.** Ext rows visited by
the corpus: **3,002 → 6,394** of 58,968 (5.09% → 10.84%) for +9,068 pairs. Most
of the new rows are thin (1–4 visits: 933 → 2,463), but the 5–49 band also
doubles (1,106 → 2,411), so this is not only a tail.

| arm | train regs | pairs | recovery_attn | far | near | **cos(s,en) commentary** | cos(s,en) tags | held span | held attn |
|---|---|---|---|---|---|---|---|---|---|
| _(zero-shot)_ | — | — | 0.511 | 0.087 | 0.480 | 0.081 | 0.058 | 0.642 | 0.417 |
| `span` | tags,tags_alt | 5,730 | 0.886 | 0.068 | 0.449 | 0.097 | **0.100** | **0.128** | 0.171 |
| `attn+span` | tags,tags_alt | 5,730 | 0.868 | 0.065 | 0.442 | 0.096 | 0.096 | 0.166 | 0.146 |
| `span` | +commentary | 14,356 | 0.910 | 0.069 | 0.450 | 0.098 | 0.100 | 0.134 | 0.176 |
| `attn+span` | +commentary | 14,356 | **0.953** | 0.069 | 0.450 | **0.109** | 0.097 | 0.190 | **0.115** |

- **D2 is structurally inert under the settled `loss=span`.** Prose has no
  tag-by-tag alignment, so D2 pairs carry no `spans` and contribute *zero*
  gradient to the span term — their only effect there is to dilute the batch
  (~39% span-carrying rows instead of 100%). The `span` +D2 row moves
  commentary 0.097 → 0.098, i.e. nothing. **A corpus addition is not a mix
  question until the objective can consume it.**
- **Under a sequence-level term it works, in its own register.** `attn+span`
  +D2 lifts commentary **0.096 → 0.109 (+13%)** and leaves tags flat
  (0.096 → 0.097). It buys prose conditioning; it does **not** transfer to the
  tag register, which is what the 2c gate is scored on.
- **Read `recovery_attn` 0.953 with the holdout in mind.** The 900-pair holdout
  is ~48% commentary, so the D2-trained arm is partly being graded on its own
  domain. The per-register decomposition above is the honest column, and it is
  why the headline flip (`attn+span` 0.953 > `span` 0.910, reversing G2's
  verdict) is **not** grounds to re-open G2: on `tags`, `span` still wins.
- **Discrimination is unharmed**: far 0.065–0.069 across every arm, below the
  0.111 zero-shot baseline and far under the 0.2 gate. Near stays ~0.45.
- The 2c gate is untouched by this: `cos_vs_en` is 0.10 against a 0.6 target.

**What this settles.** D2 is worth finishing (the coverage doubling is real and
the register gain is real), but shipping it requires an objective change, not
just more data — either keep a sequence-level term in the mix for span-less
registers, or find an alignment for prose. Both are Phase-2c decisions.

## G3 — the teacher ceiling, per register (2026-08-16)

The 2c gate is `cos_vs_en ≥ 0.6`, a number taken from the Phase-0 probe where
the *teacher* measured 0.69/0.77 — on two hand-written prompts, in one
register. Distillation cannot pass its own teacher, so the gate is only
meaningful as a fraction of a ceiling, and the ceiling had never been measured
on the corpus. Measured now on all **900** held-out pairs (no training; the
teacher/reference/native arms are all cached, so this is a readout, not a run).
Envelope: `bench/cjk_distill/results/20260816-1442-g34-g3-g4a/`.

| register | n | teacher (ceiling) | native (floor) | addressable | zero-shot student | ceiling attn | floor attn | addressable attn |
|---|---|---|---|---|---|---|---|---|
| tags | 143 | **0.823** | 0.017 | 0.806 | 0.057 | 0.864 | **−0.669** | **1.533** |
| tags_alt | 143 | **0.824** | 0.016 | 0.807 | 0.057 | 0.864 | −0.672 | 1.535 |
| commentary | 442 | 0.786 | 0.056 | 0.731 | 0.091 | 0.845 | 0.737 | 0.108 |
| quote_translated | 86 | 0.725 | 0.080 | 0.646 | 0.058 | 0.983 | 0.962 | **0.021** |
| quote_preserved | 86 | 0.706 | 0.097 | 0.609 | 0.063 | 0.963 | 0.948 | **0.015** |
| _(pooled)_ | 900 | 0.785 | 0.049 | 0.735 | 0.074 | 0.875 | 0.331 | 0.544 |

**The 0.6 gate stands, and it is a real gate, not a formality.** The `tags`
ceiling is 0.823, so 0.6 asks for **73% of the teacher** — above the Phase-0
probe's 0.77 reading of the same register, i.e. if anything the gate is
slightly conservative against the corpus teacher rather than unreachable.
Matching the teacher would be 0.82. Current student: **0.10** (previous
section). The gate is far away, and it is not the gate's fault.

**`recovery_attn` is a mix statistic and must not be compared across runs.**
This is the load-bearing finding. The readout-space *floor* is register-
dependent by two orders of magnitude — `tags` sits at **−0.669** (the unk-wall
is anti-correlated with the EN reference once you look through the DiT's own
cross-attention), while `quote_preserved` sits at **+0.948**. So the
addressable denominator runs from 0.015 to 1.535 depending on register, and the
pooled `recovery_attn` divides by whatever mix the holdout happened to have
(0.544 here). Consequences, in order of importance:

- The D2 section's headline flip — `attn+span` 0.953 vs `span` 0.910 on a
  holdout that is 49% commentary — is **not** an arm difference of that size;
  it is partly a denominator that the commentary share moved. The verdict
  recorded there (read the per-register `cos(s,en)` column, not the headline)
  was right for a reason that is now measured rather than suspected.
- G2's cross-tab is unaffected: all six arms shared one holdout and one mix, so
  the ranking is internally valid. Only *cross-run* readings were ever at risk.
- Going forward, rank arms on the per-register `cos_student_vs_en_by_register`
  and quote `recovery_attn` only inside a fixed mix.

**D6's demotion is confirmed quantitatively, and the flat number is the liar.**
In flat space the quote registers look distillable (addressable 0.61/0.65); in
the space the DiT actually consumes there is **nothing there** — 0.015 and
0.021 of headroom between floor and ceiling. The flat gap is almost entirely
token-count difference (a raw JA string collapsing to an `<unk>` run changes
lengths, which a position-wise cosine sees and a permutation-invariant
consumer largely does not). D6 was demoted to eval-only on the argument that
its teacher is degraded by construction; this is the independent measurement
of that argument. Glyph identity remains Phase 4's job.

**Commentary is a legitimate register at the flat level** (ceiling 0.786, only
0.04 below `tags`) but carries just 0.108 of readout-space headroom. It is
worth distilling; it is not worth grading a mixed-mix `recovery_attn` on.

## G4a — corpus health (2026-08-16)

Same envelope, CPU-only, over the cache training actually reads (18,090 train /
900 holdout).

| register | train | holdout | mean JA tok | mean EN tok | JA/EN ratio (mean/median) | pairs w/ spans | span tokens |
|---|---|---|---|---|---|---|---|
| tags | 2,865 | 143 | 179.3 | 153.9 | 1.17 / 1.17 | 3,008 | 417,621 |
| tags_alt | 2,865 | 143 | 183.9 | 153.9 | 1.20 / 1.20 | 3,008 | 431,538 |
| commentary | 8,626 | 442 | 33.5 | 35.1 | 1.06 / 0.94 | 0 | 0 |
| quote_preserved | 1,867 | 86 | 13.5 | 11.7 | 1.17 / 1.13 | 0 | 0 |
| quote_translated | 1,867 | 86 | 13.5 | 14.2 | 0.96 / 0.93 | 0 | 0 |

**No register is length-pathological.** The JA-student / EN-teacher token ratio
is 0.96–1.20 everywhere; the position misalignment that demoted `L_flat` from
objective to control is a ~17–20% count difference on the tag registers, not
the 2× blowup a naive character-level ext vocab would have produced. This is a
*bound* on the misalignment, not a licence to re-open `L_flat` — a matched
count says nothing about matched content at position *i*.

Occurrence-weighted span provenance, recomputed off the cache (`plan.md` quotes
34% from the corpus build; against what training reads it is **36.6%**):

| via | span tokens | share | spans | weight under `provenance` |
|---|---|---|---|---|
| mt_unverified | 310,944 | **36.6%** | 81,163 | 0.3 |
| mt_verified | 274,419 | 32.3% | 80,114 | 0.8 |
| wiki_verified | 93,674 | 11.0% | 32,901 | 1.0 |
| override | 54,902 | 6.5% | 19,104 | 1.0 |
| wiki | 39,894 | 4.7% | 8,994 | 0.7 |
| passthrough | 38,422 | 4.5% | 6,076 | 1.0 |
| rating | 21,850 | 2.6% | 6,016 | 1.0 |
| wikidata | 12,576 | 1.5% | 2,606 | 1.0 |
| wiki_han | 2,478 | 0.3% | 808 | 1.0 |

Ext-row coverage over the `tags,tags_alt` training pool: **2,656 / 58,968
(4.50%)** visited — bands 1–4: 764, 5–49: 983, 50+: 909. The 909-row head is
what the span loss is really training; everything else rides the global map.

## G4b — the trust ablation (2026-08-16)

Does the 36.6% of span tokens that come from unverified MT actually hurt? Three
policies at the settled `param=global, loss=span`, 1500 steps × batch 32, one
seed, one cache. Held-out scoring uses the **cached (`provenance`) weighting for
every arm**, so no arm is graded on its own weighting. `CachedPairs.apply_trust`
re-derives weights from each span's `via` on load, so an arm is not a cache
rebuild. Envelope: `bench/cjk_distill/results/20260816-1428-g34-g4b/`.

| trust | spans kept | recovery_attn | cos(s,en) tags | disc far | disc near | held span | held attn |
|---|---|---|---|---|---|---|---|
| _(zero-shot)_ | — | 0.581 | 0.057 | 0.087 | 0.442 | 0.6465 | 0.4270 |
| `all` | 226,546 | **0.904** | **0.098** | 0.072 | 0.410 | 0.1258 | 0.1756 |
| `provenance` | 226,546 | 0.889 | **0.098** | 0.072 | 0.412 | **0.1230** | **0.1693** |
| `verified_only` | 149,295 | 0.864 | 0.096 | 0.072 | 0.412 | 0.1429 | 0.1746 |

**The trust policy is not a lever at this corpus size.** `all` and `provenance`
keep the *same* 226,546 spans (the build-time policy already dropped its own
zeros — `unresolved`, `unmapped`), so they differ only in weighting, and they
land on the same `cos(s,en)` to three decimals. `provenance` wins `held span` by
0.003 while being the arm whose objective *is* the eval weighting; that is not a
result. **Dropping the unverified spans is actively worse**: `verified_only`
gives up 34.1% of the supervision and pays for it everywhere — 0.098 → 0.096 on
tags, 0.904 → 0.864 recovery, and `held span` moves the *wrong* way,
0.1230 → 0.1429. Discrimination is 0.072 on all three. At ~10⁴ pairs the corpus
is supervision-starved, and noisy supervision beats none.

**This does not close the 2a sign-off blocker, and must not be read as
closing it.** What G4b measures is an *aggregate* objective over 226k spans; a
bad wording is a *local* vocabulary error. `colored inner hair` → 色付きの陰毛
trains the rows for 陰毛 toward the EN embedding of "inner hair" — a specific
row bound to the wrong meaning, invisible in a mean over 226k spans and fully
visible to a user who types 陰毛. `--trust provenance` was never a fix for that;
it was a hedge, and G4b says the hedge costs nothing and buys nothing.
`tag_glossary_review.md` still needs eyes. Keep `provenance` as the default
(free, and it is the right prior if the corpus ever gets big enough for the
noise to bite).

## Phase 2c — first pass (2026-08-16)

Two packs trained at the settled design (`loss=span`, `--trust provenance`,
`--train_registers tags,tags_alt`, 8,000 steps × batch 32, lr 1e-3):
`output/ckpt/cjk_vocab_pack_global{,_row}.{safetensors,json}`. Envelopes:
`bench/cjk_distill/results/20260816-1450-2c-global/` and `…-1511-2c-global-row/`;
rendered eval under `bench/cjk_adapter/results/20260816-1618-2c-gate-global/`,
`…-1619-2c-grid-global/`, `…-1634-2c-gate-globalrow/`, `…-1634-2c-grid-globalrow/`.

| pack | final span loss | cos(s,en) flat | tags flat | cos(s,en) attn | recovery_attn | disc far | disc near |
|---|---|---|---|---|---|---|---|
| `global` | 0.095 | 0.094 | 0.096 | **0.804** | 0.869 | 0.069 | 0.406 |
| `global_row` | 0.064 | 0.096 | 0.096 | **0.809** | 0.878 | 0.074 | 0.400 |

(Holdout teacher: 0.785 flat / 0.875 attn; native floor 0.049 / 0.331.) The
span loss saturates — held-out span 0.646 → 0.105/0.073 — and the two packs are
indistinguishable end-to-end (render-gate `cos_vs_en` 0.070–0.077 both, one
render grid apiece telling the same story), so G2's "prefer `global`" stands.

The two headline numbers **disagree by design**: flat 0.09 against the 0.6
gate, readout-space 0.80 against the 0.875 teacher ceiling (≈87% of the
addressable interval on the corpus holdout). G5 resolves which one is lying.

### G5 — the flat gate is measured-blind (2026-08-16)

The oracle the plan demanded before treating a flat-gate failure as a verdict.
`gates/g5.py` builds synthetic students from the *teacher's own rows* and
scores them with 2c's own metric. Envelope:
`bench/cjk_distill/results/20260816-1532-g5-flat-ceiling/`. On `tags`
(143 held-out pairs; teacher ceiling 0.823 flat / 0.864 attn):

| oracle | flat vs en | attn vs en |
|---|---|---|
| `span_perfect` (exact argmin of `L_span`) | **0.130** | 0.711 |
| `span_plus` (argmin + free unsupervised positions) | **0.315** | 0.756 |
| `ref_remap` (perfect content, student's segmentation) | 0.123 | **0.975** |
| `prefix_bound` (any tensor of that length) | 0.9999 | — |

**A perfectly-distilled span student cannot pass the 0.6 flat gate — the gate
is blind to the objective, not the student to the gate.** `prefix_bound` ≈ 1.0
says nothing is arithmetically impossible; `ref_remap` (0.123 flat, 0.975 attn)
isolates the cause — the flat cosine charges position-by-position for
segmenting the same content into a different token count, which the teacher is
exempt from because its T5 ids *are* the reference's. The same oracles sitting
at 0.71–0.97 in the readout space is the contrast that certifies the
attn-space metric as the honest one. Verdict: **flat `cos_vs_en` is demoted
from gate to control**; the student's 0.09 was never the size of the gap.

### The two flat probes — the gate cannot be bought

Could the flat number be trained up anyway? Two arms at `param=global_row`,
same corpus/steps. Envelopes: `…-1533-2c-probe-flatmax/`, `…-1557-2c-probe-spanflat/`.

| arm | loss | tags flat | cos(s,en) attn | recovery_attn | disc far | disc near |
|---|---|---|---|---|---|---|
| 2c ship | `span` | 0.096 | 0.809 | 0.878 | 0.074 | 0.400 |
| probe | `span + 0.5·flat` | 0.156 | 0.686 | 0.653 | 0.106 | 0.544 |
| probe | `flat` only | 0.245 | **−0.072** | −0.742 | 0.235 | **0.914** |

Every flat point is paid for in the space the DiT consumes: the mixed arm
gives back 0.12 of attn-space alignment for +0.06 flat, and pure `flat`
reproduces G2's disqualification exactly — near-discrimination 0.914 is the
mode-collapse signature, and the readout alignment goes *negative*. **No flat
term ships.** This is the same verdict G2 reached as an objective question,
now confirmed at the 2c scale as a gate question.

### The render failures are coverage, token by token

The 20-prompt grid (`assets/ja_eval_prompts.json`, seed 42, arms
`en / ja_t5en / ja_ext`): the **teacher is at EN parity everywhere sampled**,
including the prompts the student fails — knight + castle + red cape on t3,
canonical Reimu on n1 — so the remaining gap is entirely student-side. The
student splits cleanly:

- **Transfers**: t1 school (every content token ≥ 300 span visits), t2 maid
  (maid/blonde/twintails/blue-eyes land; the background is lost — and `緻密`
  "detailed [background]" has **0** visits), t6, s5.
- **Collapses**: t3 armor (`騎`:0, `鎧`:0, `照明`:1 — no knight, no armor),
  n1/n2 names (identity tokens `博`:2 `麗`:2 `巫`:18 `霊`:22 `夢`:37 — girl
  present, character gone), prose s1–s4 (function words `っていて`/`路面`/
  `を作る` etc. have 0 visits by construction — kept "rain + girl", lost
  umbrella/crosswalk/neon).

`gates/coverage.py` is the diagnostic (CPU-only): it tokenizes each eval
prompt through the pack and prints span-visit counts per ext row —
2,672 of 58,968 rows visited by `tags,tags_alt`, and the failure set of the
grid is exactly the prompts whose *content* tokens sit in the 0–40 band. The
working head suggests identity-carrying tokens want O(100+) visits (`教室`:39
renders a classroom; `霊夢`:37 does not render Reimu — identity is a harder
target than a generic concept).

### What the first pass settles

1. The design shipped as-is is **on-distribution correct**: ~87% addressable
   recovery in the consumed space, discrimination healthy (far 0.07, native
   render-gate control 0.91), EN bit-exactness untouched.
2. The gap to "usable from a JA prompt" is **not the objective and not
   capacity — it is supervision coverage** of the content vocabulary, plus the
   already-known span-less-register hole. Both have named levers (Next 1–3).
3. The 2c acceptance surface is re-based (Next 4); the 0.6 flat gate is
   retired as a gate and kept as a control alongside flat discrimination.

## D1-wide — the gelcrawl widening, measured (2026-08-16)

The Corpus table named D1-wide "the unblocked lever": `~/gelcrawl/retrieved/`
holds 16,053 EN tag captions against `image_dataset`'s 3,008, and the corpus
is text-only so curation state is irrelevant. Built and measured. It delivers
the visit multiplication it promised, and it **does not fix the render grid's
zero-visit failures** — those turn out to have a different cause.

### What was built

`build_pairs.py` / `tag_glossary.py` now take multiple caption roots, each
flagged curated or raw (`--captions` / `--raw-captions` / `--tag-rules`). The
two roots are not in the same format: `image_dataset` is post-processed and
`retrieved/` is raw crawler output (`&#039;` entities, booru rating words
instead of Anima's band, `highres`/`absurdres` meta, undeduped clothing
bases). Raw roots go through gelcrawl's own `tag_rules.yaml` via
`library.captioning.tag_rules` — **the same normalization that produced
`image_dataset`**, so the two roots agree on `questionable`→`nsfw` instead of
splitting the rating rows. Verified segment-for-segment against a curated
caption. Roots dedup on the artist-relative path (the bare stem is not unique:
`dan_` prefix = danbooru id space, [[project_booru_id_space_collision]]),
first root winning, so the curated copy beats its own crawl source on the
2,933 of 3,008 that overlap.

Result: **3,008 → 16,128 captions** (+13,120 gelcrawl-only), D1 6,016 →
32,256 pairs, corpus 18,990 → 45,230 pairs.

### What it bought — visits, exactly as predicted

Span-visit bands over `tags,tags_alt` (`coverage.json`), same glossary:

| band | before (3,008 caps) | after (16,128 caps) |
|---|---|---|
| 1–4 | 2,463 | 2,339 |
| 5–49 | 2,411 | 2,119 |
| 50–499 | 1,139 | 1,210 |
| **500+** | **381** | **756** |
| rows visited | 6,394 | 6,424 |
| visits total | 1,314,156 | 5,649,001 (4.3×) |

The 500+ band **doubles** and the low bands drain into it — rows migrating
up, which is the stated goal (identity-carrying tokens want O(100+) visits).
Rows *visited* is flat (6,394 → 6,424): the widening buys **visits, not
vocabulary**, because the JA side is composed from a glossary that did not
grow. That was measurable in advance and is the correct reading of it — the
current glossary already covers **97.3% of the widened corpus's occurrence
mass**; the 7,242 tags gelcrawl adds carry only **3.2%** of its occurrences,
and the top of that tail is `@artist` handles, which pass through latin by
design.

### What it did not buy — the v=0 tokens are a *wording* defect

`gates/coverage.py` before vs after: the `v<5` column improves on six prompts
(`照明` 1→4, `気が` 2→4, `女子` 3→ok, `二人`/`カフェ`/`店` clear, `博` 2→ok),
but the **`v=0` column is essentially unchanged** — `騎`:0 `鎧`:0 `京都`:0
`俯`:0 `瞰`:0 `畑`:0 `肖`:0 `接`:0 survive a 5.4× corpus. Diagnosed, and it
splits in two:

1. **Wording mismatch — the tag is present and now well-visited, but the
   glossary bound it to a different JA surface than the one users type.**
   `armor` occurs 39× in the widened corpus and its candidate list is
   `アーマー` (f1 1.0, kana) and `鎧` (f1 1.0, Han-only) — **both back-translate
   perfectly**; the kana-proves-Japanese rule picked the katakana loanword, so
   `鎧` sits at 0. Same shape for `from above`→`上から` (not `俯瞰`, which is in
   `alts`), `close-up`→`クローズアップ` (not `接写`), `portrait`→`ポートレート`
   (not `肖像`). And the rule filters twice: `alt_pool` also requires kana for
   general-axis tags, so the kanji wording is excluded from `tags_alt` as well
   — it is reachable from **neither** register.
2. **Genuinely absent vocabulary** — `kyoto`, `noren`, `knight`, `field` are
   not booru tags at any pool size, so `京都`/`暖簾`/`騎士`/`畑` cannot be
   reached from a tag caption at all. This is the span-less-register hole
   (D2/D3) and the name register, not a D1 lever.

Sizing (1): **119 general tags / 1,735 occurrences** have a pure-katakana
primary tied-or-beaten by a Han-only candidate. It is **not** automatically
fixable, and the kana guard earns its keep — the list is roughly half native
Japanese wrongly rejected (`上着 翼 靴下 腕輪 眼鏡 砂浜 刺青 逆光 直毛 漫画
扉 果物 指輪 化粧 提灯 水筒`) and half Chinese correctly rejected (`指甲油
智能手机 杯子 牛仔布 毛巾 背包 手提包 特写 影子 睡衣`), with at least one
false friend that would be a real bug: **`bed` → `床`**, which is *bed* in
Chinese and *floor* in Japanese. Han-only-plus-JA-valid-kanji does not
separate them; only Japanese knowledge does. So this is a **human review
axis**, and a new one — the existing `tag_glossary_review.md` is ordered by
`mt_unverified` disagreement and does not surface this class at all.

### The glossary rebuild is GPU work, not a free CPU pass

Re-running `tag_glossary.py` over the union **without `--mt`** was tried and
**reverted**: the back-translation scoring is what selects among candidates,
so a CPU-only rebuild drops every `mt_verified`/`wiki_verified` verdict and
the `candidates` field with it (5,920 → 0 entries), re-picking straight from
the wiki head. It reproduces exactly the failures `datasets/README.md` warns
about — `underwear`→**下着コート**, `1girl`→女の子, `black hair`→黒髪ボブ,
`large breasts`→デカ乳, `censored`→**遮盖** (Chinese) — regressing 1,991
wordings and unresolving 2,948 previously-resolved tags. **Do not rebuild the
glossary without `--mt`.** The widened glossary is a daemon job whose MT cache
makes the 7,514 existing tags nearly free; only the ~6,600-tag residue and the
new candidates' back-translations are paid.

### Verdict

- The widened corpus is **built and is the current
  `post_image_dataset/cjk_distill/`** (45,230 pairs). It is strictly better on
  the stated lever and costs nothing at train time.
- D1-wide is **not** sufficient for the 2c grid. The plan's "the tag register
  is coverage-bound" is half right: it is coverage-bound in the 1–49 bands,
  which this fixes, and **wording-bound** at v=0, which it cannot.
- Unmapped segments rise 586 → 42,530 (0.10% → 3.2% of segments, matching the
  new-tag occurrence share). These degrade to latin passthrough with
  `via: unmapped`, f1 0.0 — the trust weighting already handles them, so this
  is a coverage number to close, not a corruption.

## D1-pairs — the tail fill, measured (2026-08-16)

`p1atdev/danbooru-ja-tag-pair-20241015` (CC0; 151,431 rows — 93,393 character /
22,330 copyright / 35,708 general) is the danbooru wiki's own `other_names` at
an Oct-2024 snapshot with `calm3-22b-chat` filling tags that had no Japanese
name. `datasets/tag_pairs.py` uses it as a **fill-only** source: tags the
glossary leaves unresolved compose as latin passthrough at span weight 0, so
filling them cannot regress a wording (unlike the CPU rebuild this file already
rejects). Its own Chinese/latin noise is re-guarded here, not trusted — across
the whole table the guards drop 8,372 non-JA, 5,033 Han-only-outside-inventory
(`裙`, `百褶裙`) and 28,234 latin-bearing (`ウィンクX東方`) names.

Envelopes: `bench/cjk_distill/results/20260816-2022-2c-tagpair/`,
`bench/cjk_adapter/results/20260816-2043-2c-tagpair-{gate,grid}/`. Pack:
`output/ckpt/cjk_vocab_pack_tagpair.{safetensors,json}`.

### What it bought — vocabulary, which is what D1-wide could not buy

| | D1-wide | + tail fill |
|---|---|---|
| glossary entries with a wording | 7,348 | **12,596** (5,248 filled) |
| unmapped segments | 42,530 | **13,714** (−68%) |
| supervised span tokens (train) | 4,337,704 | **4,470,497** |
| ext rows visited (`tags,tags_alt`) | 2,759 | **3,309** (+20%) |
| rows visited (whole corpus) | 6,424 | 6,538 |

The row count moves far less than the tag count because 5,248 new wordings are
largely built from kanji the corpus already visits — the gain is that those
kanji now arrive *in the right words*. This is the axis D1-wide explicitly could
not move: widening multiplied the same glossary and left every `v=0` token at
zero, while the fill clears them.

### The coverage diagnostic clears — and the renders still do not

`gates/coverage.py` on the 20 eval prompts, against the same prompts' pre-fill
readings: `t3_tags_armor` (`騎:0 鎧:0` — the report's canonical collapse) and
`n1_name_hakurei` (`麗:2`) now carry **no** zero-visit content token;
`s3_scene_train` (`誰:0 学生:0 寝:0`) and `s2_scene_kitchen` (7 zero-visit → 2)
likewise. What remains at zero is prose function words (`っていて`, `」と`,
`、「`) and non-tag vocabulary (`京都`, `俯瞰`, `緻`) — the untrained registers,
exactly as scoped.

**The render grid does not follow.** Same seed, same prompts, same arms: `t1`
and `s3` are modestly closer to the teacher, `t3` and `n2` are not better, and
nothing shows a step change. Raising the visit floor explains why — at
`--floor 300` the newly-supervised tokens read `騎:46 士:72 鎧:24 照明:6`
against `教室:253`, the token the first pass found *does* render its concept.
Clearing zero is necessary and is not sufficient: these rows moved from
unsupervised to *thinly* supervised, an order of magnitude below the working
head's O(100+).

### Numbers, and why two of them are not an A/B

Per-register readout (the new `attn_by_register`, holdout n as shown):

| register | student | teacher | native | R |
|---|---|---|---|---|
| `tags` | 0.493 | 0.553 | −0.340 | **0.933** |
| `tags_alt` | 0.513 | 0.557 | −0.351 | **0.951** |
| `commentary` | 0.944 | 0.962 | 0.947 | −0.19 |
| `quote_preserved` | 0.988 | 0.993 | 0.990 | −0.68 |

This is G3's "the readout floor is register-dependent" at register granularity,
and it settles how to read the aggregate: on the prose registers the teacher
sits 0.015 above the native floor, so their R is noise around an empty interval
and the *only* meaningful recovery number is the trained registers' ~0.93–0.95.

Final span loss 0.114; render gate `ja_ext` flat 0.085/0.070 with
discrimination far **0.194** (under the 0.2 pathology guard, unchanged).

Two comparisons that look like an A/B and are not: aggregate `recovery_attn`
reads 0.954 pre-fill against **0.936** post-fill, and `cos_student_vs_en_attn`
0.650 → 0.632. The fill changes the *holdout text itself* — a filled tag that
used to sit in the reference as latin now sits there as Japanese — so the task
moved with the student (the teacher moved too, 0.680 → 0.674). Read these as
directional at best. The comparisons that do hold are the fixed-prompt ones:
the coverage diagnostic and the rendered grid.

### Verdict

- The fill is **kept**: it is free (CPU), it regresses no wording by
  construction, and it buys the vocabulary axis the corpus work had exhausted.
- It is **not** the render unlock. The grid is a wash, and the reason is
  measured: the rows it lit up sit at 24–46 visits.
- It changes what the *next* widening is worth. D1-wide bought no vocabulary
  because the glossary was the binding constraint; with 5,248 more tags mapped,
  more captions now multiply *these* rows too — the two levers compose in the
  order they were run, not the reverse.
- The larger prize in this source is untouched: re-selecting existing wordings
  through the back-translation arbiter (`plan.md`, D1-pairs item 2), where it
  disagrees with 65% of our MT-derived choices and is usually right on the
  high-traffic ones.

## D1-pairs item 2 — the arbiter re-selection, measured (2026-08-17)

The pass above, run. `tag_glossary.py --mt` now feeds the tag-pair names into
the arbitration as candidates (same guards as the fill; per-candidate `src`
provenance persisted; winner `via: tagpair_verified`, declared in
`TRUST_POLICIES`), with one ranking change: at equal back-translation F1 and
equal kana-tier, a community-attested candidate outranks the MT rendering. The
kana-over-kanji rule is untouched — `bed` → 床 (*floor* in JA, *bed* in ZH)
is why it must not be relaxed automatically. Because the previous glossary's
arbitration had only ever run over the narrow 3,008-caption tag set, this job
also cleared the **owed widened `--mt` rebuild** in the same pass (~50 min
GPU; the MT cache covered the old tags). Old glossary preserved at
`assets/tag_glossary_ja.pre_item2.json`.

Envelopes: glossary daemon job `20260816-212444-1f89f8`; retrain
`bench/cjk_distill/results/20260817-0111-train` (defaults probe — see the
gotcha) and `…-0115-train` (the real arm); renders
`bench/cjk_adapter/results/20260817-0138/` (gate + grid — same-minute dir
collision, [[project_bench_run_dir_collision]]: the grid's `result.json`
overwrote the gate's, gate numbers survive in the job log).

### What it bought

| | pre (tail-fill state) | post (item 2) |
|---|---|---|
| glossary, tags with a wording | 12,596 (stale narrow counts) | **14,678 / 14,753** (widened counts, coverage 99.86%) |
| wordings changed vs backup | — | **4,438** (48,909 occ, 7.6% of tag mass) |
| of which `tagpair_verified` | — | 1,538 types / 34,764 occ (5.4%) |
| pinned-source regressions | — | **0** |
| corpus untranslated segments | 13,714 | **878 (−94%)** |
| 500+ visit band | 778 | 800 |

High-traffic re-selections look exactly like the predicted class: `breasts`
巨乳→おっぱい, `cowgirl position` カウガールポーズ→**騎乗位**,
`collared shirt` カラーシャツ→襟付きシャツ, `solo focus`
一人フォーカス→ソロフォーカス.

**What the arbiter structurally cannot win: the polysemy class.** `bow` →
蝶結び back-translates to *"bow tie knot"* (F1 0.5) while MT's お辞儀
round-trips its own sense at 1.0 — so the known-right community wordings for
`bow` / `on back` / `clothes lift` / `multiple girls` stay MT-worded. F1
verifies *recovered string*, not *booru sense*; no threshold fixes that. These
are now surfaced instead of silent: the review filter includes `mt_verified`
rows carrying a community rival at F1 ≥ 0.3 (previously invisible — the top
candidate equalled the MT rendering, which the old filter read as agreement),
and `tag_glossary_review.md` (400 rows) gains the D1-words katakana-vs-kanji
section. **The tag-register wording ceiling now runs through the human
sign-off**, which was already the 2a ship blocker — the arbiter turned it from
an eyeball pass into the sourced diff the plan asked for.

### Retrain + renders (`2c-item2`, settled design)

Span loss 0.650 → 0.113; readout recovery `tags` **0.915** / `tags_alt`
**0.934** (per-register, the honest columns); flat discrimination far 0.078 /
near 0.353. Render-gate control: flat `cos_vs_en` 0.077–0.080 (retired
control, unchanged as G5 predicts), p1-vs-p2 discrimination **0.202** against
the 0.2 guard (was 0.194; n=2 prompts — watch it, don't panic over it).

The fixed-prompt comparisons — the only valid A/B — move for the first time:

- **`t3_armor`**: was a bare caped figure on a cliff; now a **rider with a
  horse** and red garment under a storm sky. 騎/士 landed; the armor itself is
  still absent — `鎧` sits at **23 visits** because the arbiter *correctly*
  kept `armor`→アーマー (kana guard), i.e. this token now waits on the
  D1-words override, not on more corpus.
- **`t2_maid`**: the lost background is back (real garden scenery vs the flat
  texture wall).
- **`t1_school`** stable; **`n1_hakurei`** unchanged — 博:10 麗:31 巫:64,
  the name register (D5) remains the un-run lever, exactly as scoped.
- `gates/coverage.py`: `t3` and `n1` carry **zero** zero-visit content tokens;
  remaining v=0 is prose function words (sequence-term decision) and non-tag
  vocabulary (`京都`, `俯瞰` — the latter now a review-file row).

### Gotcha recorded

`scripts/distill_cjk/distill.py`'s argparse defaults are **not** the settled
design (`loss=attn:1.0`, 2000 steps, batch 8). A bare `--mode train` runs a
different experiment that *looks* plausible in the logs (loss converges,
disc healthy). `…-0111-train` is such an accident — 20 wasted GPU-minutes;
rank arms only after checking `result.json`'s `args`. Pass
`--loss span --steps 8000 --batch_size 32` explicitly.

### Verdict

The dataset axis delivered what it had left to deliver: the corpus is no
longer wording-bound (−94% unmapped) and the grid moved where coverage moved.
The remaining tag-register gap is **override-bound** (human review: polysemy +
katakana/kanji rows, both now sourced in the review file) and **visit-bound**
on rare tags (`鎧` 23, `照明` 3 — the targeted-crawl lever). Names remain
register-bound (D5). Next levers in order: review sign-off → name register →
targeted caption widening; the sequence-term decision still gates D2/D3/D4.
