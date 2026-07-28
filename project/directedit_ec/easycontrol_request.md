# easycontrol_request — aligned-pair instruction-edit adapter (data + objective spec)

Request for the next EasyControl training arm: a **feed-forward in-place
instruction editor**. Contract: cond = source image, prompt = tag-delta
instruction (`additions, -removals`), output = **the same image with only the
instructed changes applied**, engaged at the trained operating point
(b_offset 0), 1× NFE, no inversion/anchor/mask.

## Why this arm (evidence, 2026-07-26)

The in-place probe line (`project/directedit_ec/bench/run_inplace_probe.py`, runs
`results/20260726-{1129,1145,1213}-inplace-*`) established:

- The shipped `subject_edit` + DirectEdit inversion + `--anchor_scale` gets
  composition-preserving instruction edits zero-training (src="" recipe), BUT
- **the aligned-copy lock**: on trivially-copyable sources the inverted init
  keeps the trajectory cond-aligned and the architectural copy path clamps the
  output to the source at every anchor scale — the edit never lands.
  `subject_edit` cannot fix this: its pairs are all non-aligned (cond = image
  A, target = a *different* image B), so "aligned cond + instruction → change
  it in place" is a training cell it never saw. It learned re-render, not edit.
- **sanitize precedent** (near-twins text-removal arm, trained Jun 12–Jul 2,
  ckpt pruned): post-hoc verdict = learned *copy-with-exception* on aligned
  pairs — selective non-copying IS learnable by this architecture. Its gap
  (exception baked in as a task prior, not prompt-driven) is exactly what the
  delta-caption objective fixed on `subject_edit` (`caption_dropout_rate=1.0`
  there; must be 0 here).

So this arm = **sanitize's data/loss recipe × subject_edit's objective**. Both
halves are individually proven; the cross product is the request.

## Data pairs needed

All pairs are (cond image, target image, delta caption). Order matters: the
model edits *cond toward target*.

### 1. Aligned change pairs (the core; near_twins miner)

Near-pixel-aligned in-artist twins that differ by a small, caption-visible
change. Mining gates as in `configs/easycontrol/sanitize.toml [staging]`:
`sim_min 0.85`, `match_frac_min 0.3`, `cell_match_min 0.9`,
`max_extra_diff 25` — the geometric checks are what make the pair *aligned*,
which is the entire point of this arm.

Discriminator slices, by priority:

| Slice | Discriminator | Notes |
|---|---|---|
| text/bubbles | `tag_any` = speech bubble, spoken text, … (sanitize's list) | Known to exist in volume; deltas are **pure removals** (`-speech bubble, …`) → directly attacks the Q9 removal weakness |
| clothing / state | `tag_any` = costume & clothing-state tags (jacket removed, shirt lift, undressing, alternate costume, …) | Booru variant sets; the highest-value edit family for users |
| expression / pose detail | `tag_any` = expression tags (smile, open mouth, blush, closed eyes, wink, …) | Small localized diffs, plentiful in variant sets |
| object add/remove | `tag_any` = holdable/accessory tags | Gives symmetric add/remove supervision |
| untried miner modes | `tag` / `region` / `signal` | Explore only if `tag_any` volume is short |

**Direction doubling**: every mined twin yields two examples (A→B and B→A)
with mirrored deltas — additions in one direction are removals in the other.
Free 2× volume and balanced add/remove supervision; sanitize used one
direction only.

**Volume target**: ≥ 662 pairs post-gates (subject_edit's count) before
doubling. **RESOLVED 2026-07-26** by the pair census
(`project/directedit_ec/bench/run_pair_census.py`, `project/directedit_ec/bench/results/20260726-1337-pair-census-full`):
**2,349 usable twins pre-doubling** (expression 1,828 / clothing_state 650 /
text_bubbles 229 / object 169; per-tag pivot semantics — the set-level
`tag_any` count is 969). Volume is not the constraint; slice priority
inverts vs the table above (expression ≫ clothing ≫ text ≫ object).

### 2. Delta captions (per pair)

Symmetric caption diff, `subject_edit_pairs.py` contract: additions in the
target's tag order + `-`-prefixed removals; shared tags (incl. character/
artist names) cancel. Constraints learned the hard way:

- Cap instruction length (`max_delta` band) — twins should sit far below the
  corpus median 31 anyway; if a "twin" has a 20+ tag delta, distrust the pair.
- Known label noise: caption inconsistency between twins shows up as spurious
  delta tags (eye-color flips etc.). `max_extra_diff` bounds it; accept the
  residual as in subject_edit.
- NO tag dropout inside the instruction at TE-encode time (dropping part of an
  instruction leaves the target unexplained); shuffle variants are fine.

### 3. Identity no-op pairs (~25% of final tree)

cond == target, **delta caption = empty**. Teaches "no instruction → no
change" — the anti-global-transform regularizer sanitize introduced
(`add_identity_pairs 0.25`) after the wash-out failure, now doing double duty
as the prompt-conditioned no-op anchor. Keep `identity_saturation_min 0.4`
(the miner's targets skew 94% pale; the floor injects the vivid cleans the
distribution lacks — unverified fix, carry it anyway).

### 4. NOT requested (explicitly out of scope for arm 1)

- Non-aligned same-character pairs — already covered by shipped
  `subject_edit`; mixing them back in dilutes the aligned signal this arm
  exists to learn. A joint/mixed arm is a follow-up A/B, not the first train.
- DirectEdit-teacher synthetic pairs (Phase-3 path) — only if mined volume
  falls short; inherits the Q3 teacher ceiling.

## Objective

Base: standard flow-matching on the target, prompt = delta caption. On top:

| Knob | Value | Why |
|---|---|---|
| `caption_dropout_rate` | **0** | sanitize's 1.0 is what made its behavior a baked-in task prior instead of prompt-driven. This single knob is the difference between "sanitize v2" and "instruction editor". |
| `cond_diff_loss` | true (floor 0.2, blur 1.5, quantile 0.9) | Twins are near-identical outside the edit; diff-weighted FM concentrates gradient on the change instead of the copy-through the extended attention gives for free. The anti-copy-lock loss shaping. |
| `b_cond_init` | −4.0 | Open-gate recipe (arm-2). `b_cond` never trains — the init IS the operating point; the engaged band must sit at b0 like subject_edit's. |
| `cond_res_scale` | 1.0 | Same reason (mass = (S_c/S_t)·e^b · scale; don't re-create arm-1's shut-gate trap). |
| `apply_ffn_lora` | 1 | arm-2 recipe. |
| `easycontrol_drop_p` | 0.05 | Cond dropout as in subject_edit (the no-cond branch must stay sane). |
| `easycontrol_cond_noise_max` | 0.0 first arm | A >0 A/B is the fallback if verbatim-copy pressure still dominates. |
| REPA-on-cond (`use_repa`, layer 8, DoG) | optional A/B | sanitize shipped it unverified; not load-bearing for this request. |

Scale: subject_edit-comparable (12 epochs over ~1k pairs ≈ 8k optimizer
steps, 16 GiB-friendly).

## Gates (all reuse existing probes)

1. **Aligned-lock flip** — feed-forward at b0 (cond = source, prompt = delta,
   random init): the yanami-class case (flat-bg, trivially copyable source,
   large structural instruction) must land the instruction while keeping
   composition. This is the single decisive gate — it is the cell that every
   current checkpoint fails. `run_inplace_probe.py --arms_spec kind=ff,...`
   plus `run_edit_probe.py` for the instruction axes.
2. **No-op stability** — cond = clean source, empty prompt → output ≈ source
   (no wash-out, no drift). Guards the identity-pair regularizer.
3. **Removal probe** — `-tag` removals of cond-present objects (Q9): the
   text/bubble slice + direction doubling is the first credible lever; judge
   removals separately from additions.
4. **Held-out ring** — all of the above on pairs outside the mining manifest
   (train-pair probes are upper bounds; Q10 discipline).

## Phase 0 before any training

Status 2026-07-26 — run via `project/directedit_ec/bench/run_pair_census.py`
(`project/directedit_ec/bench/results/20260726-1337-pair-census-full`, write-up in
`project/directedit_ec/bench/report.md`):

1. ~~Run the near_twins miner per slice; record **pair counts** + a
   saturation histogram.~~ **DONE** — 2,349 usable twins → 4,698
   post-doubling (floor 600: **PASS**); saturation skew replicates sanitize
   (96.5% of members < 0.4 → keep `identity_saturation_min 0.4`).
2. Spot-check 20 mined pairs per slice by eye: alignment quality + delta
   caption sanity — **artifacts generated**
   (`…/spotcheck/<slice>/index.html`), **eyeball pass pending**.
3. ~~Generate delta captions; report length distribution vs the `max_delta`
   cap.~~ **DONE** — median 13, p90 24, 92.2% ≤ 24 (captions per-pair in
   `pairs_manifest.json`).

If total usable pairs (post-doubling, pre-identity-fill) < ~600, stop and
reassess (widen slices vs teacher synthesis) before training. *(Not
triggered.)*

## Pointers

Probe line + evidence: `project/directedit_ec/bench/results/20260726-*inplace*`,
memory `project_directedit_ec_inplace_line`. Recipe donors:
`configs/easycontrol/sanitize.toml` (miner gates, identity pairs,
cond_diff_loss), `configs/easycontrol/subject_edit.toml` +
`easycontrol_adapters/tools/subject_edit_pairs.py` (delta captions, open-gate
training config). sanitize post-mortem: session 352e7711 (2026-07-24).
