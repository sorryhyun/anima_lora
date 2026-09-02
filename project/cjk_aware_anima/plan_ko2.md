# CJK-aware Anima — KO corpus-enhanced iteration (ko2)

*Continues [`plan_ko.md`](plan_ko.md), whose K0–K3 shipped the joint
`synthjako` pack (v2, trained 2026-08-31 on the r3/r4 wording; gates G1–G4
green, G5 owed). Everything here is a **corpus** change — the encoder, the
distill recipe (`span/global/provenance`, 12k steps, warm start), and the
JA side stay fixed. Evidence base:
[`reports/0901_ko_phase_k3.md`](reports/0901_ko_phase_k3.md).*

Premise, in one line: the remaining KO gap is not missing data — it is value
already inside the KR KB (`models/danbooru_tags_classified.csv`) that the K1
pipeline discarded or never extracted. Two extractions drive this plan:

| Extraction | What it fixes | Status |
|---|---|---|
| `키워드` field, re-arbitrated (**r5**) | the `mt_unverified` tail — 97.3 % of tail occurrences have a KB keyword that lost a rigged tiebreak (`choose()` back-translation F1 cannot verify booru jargon: 백합→"lily"≠`yuri`) | rule proposed, review file ready |
| description translations (**`desc_ko`**) | the s\* prose floor + vocabulary inside real KO sentence syntax (particles/spacing — the G5 register axis) | 11,631 pairs built (`datasets/desc_pairs.py`) |

What this plan does **not** reopen (verdicts unchanged): mT5, adapter-LoRA /
rank / attn arms, MT-prompt tuning, synthetic-pair volume, generic parallel
prose (JESC/STAIR — `desc_ko` is *domain* prose, aligned per tag, in
community terminology; that distinction is the whole argument, and the R3
gate below is where it gets falsified or kept). The shared-with-JA binding
misses (`t3` armor class) are an ext-pack ceiling — **out of scope here**;
they belong to plan.md §5b if anywhere.

## Phase R1 — r5 glossary re-arbitration (CPU + user review)

1. **User review** of
   [`datasets/assets/tag_glossary_review_ko_r5_kb.md`](datasets/assets/tag_glossary_review_ko_r5_kb.md):
   section A (117 above-floor MT-vs-KB disagreements / 93.8k occ) is a
   pick-one pass; section B (2,556 sub-floor auto-flips / 27.8k occ) is
   spot-check-and-veto. Vetoes land in `tag_overrides_ko.json` as usual.
2. **`choose()` fallback change** (`tag_glossary.py`): when no candidate
   reaches `--accept-f1`, a KB candidate outranks the unarbitrated MT
   rendering **only below the review floor (occ < 100)** → `via:
   kb_unverified`. Above the floor nothing auto-flips (r1–r4 already audited
   that band; blanket KB-first regresses it — `swimsuit`→비키니,
   `grey hair`→은발).
3. **Re-arbitration is CPU-only**: every candidate's back-translation is
   cached in the glossary JSON / `.mtcache`, so the rebuild replays
   `choose()` without the MT engine. Follow with `--reselect` → pairs
   rebuild, exactly the r3/r4 loop.
4. Known KB failure class to guard in review: rare character tags whose
   keyword is the *series* name (`elsa granhilte`→리제로) — occ≈1 noise,
   veto on sight.

Gate R1: coverage gate green (no eval-prompt token under floor); collision
audit (`audit_glossary.py`) shows no new collision groups above the accepted
list.

## Phase R2 — `desc_ko` staging decisions

Built and on disk: `post_image_dataset/cjk_distill/pairs_desc_ko.jsonl` —
11,631 pairs, EN wiki first sentence ↔ KB KO description, each carrying
**one full-width span** (`via: kb_desc`, trust 0.8). *Correction 2026-09-01:
the original span-less design was inert — under the span loss a row without
spans contributes zero gradient (the D2 commentary rows are dead weight for
exactly this reason, which is plausibly the mechanism behind the
JESC/STAIR "NO under span loss" verdict), and a pure span-less eval batch
raises (`losses.py:125` caught it on the first chain run). The full-width
span is what makes the register supervision at all.* Two knobs fixed before
staging:

- **Looseness**: the KO side keeps its full description and is sometimes one
  sentence longer than the EN (`1girl`: KO adds the dolls/posters
  exclusion). Precedent: commentary rows are loose in the same direction.
  **Fallback knob** if R3 shows the loose rows hurt: truncate KO to its
  first sentence and re-stage (extractor flag, cheap).
- **Sampling**: `desc_ko` enters `--register_sampling` at **≈0.10** — enough
  visits to move the s\* readout, small enough that tags_ko/JA sampling mass
  is essentially unchanged.

## Phase R3 — single retrain, replace-not-compare (GPU, ~55 min via daemon)

**Revised 2026-09-01 (user decision: "remove the old one")** — the two-arm
A/B design is collapsed into one retrain carrying **both** extractions
(r5 + `desc_ko`), on the argument that attribution on success is not worth a
second run: if the combined pack passes R4, both changes ship together; only
on a *failure* does attribution matter, and then an r5-only arm can be run
lazily to isolate the cause (Arm A demoted to failure-contingency).

Safety rails for "remove the old one":

- The pre-r5 corpus state is recoverable: docs and plan committed (`2ca6bc91`),
  glossary backed up on disk as `assets/tag_glossary_ko.pre_r5.json`
  (assets/ is gitignored — the on-disk copy is the real snapshot, same
  convention as r3's `pre_round3`).
- The shipped `cjk_vocab_pack_synthjako` (v2) file is **not** overwritten —
  the new pack trains to `cjk_vocab_pack_synthjako2` and takes the shipped
  slot only after R4 (the v1-overwrite lesson).
- Re-stage ≈ 44 GB in place + desc cache ≈ 8 GB; fits the ~54 GB free.

Executed 2026-09-01 morning: r5 reselect applied (`kb_unverified` 3,015
types / 32.6k occ; `mt_unverified` shrank 4,264→1,251 types), pairs + synth
names rebuilt, coverage gate green (only the expected-open s\* prose
morphemes and quote particles — which are exactly the tokens `desc_ko`
exists to visit). Daemon chain `20260901-0922*`: cache_ko re-stage →
cache_desc_ko stage → `2c-synthjako2` train (`--eval_limit 1200` so the
holdout slice spans JA+KO+desc) → the three K3-style grids.

## Phase R4 — gates

Instrument discipline learned in K3 (see the 0901 report): the ±0.013
flat-cos band does **not** imply visual sameness; renders decide, and any
single-prompt render regression claim needs a ≥3-seed probe before it counts
(the seed-42 n2/q2 "regressions" were chaos flips present in v4 at other
seeds).

1. **JA non-regression** (kill): holdout registers in the v2 band; same-seed
   grids vs the v4 twins metric-clean; render spot-eyeball. Fail → ship
   Arm-none (v2 stays) and record which extraction moved JA.
2. **tags_ko band**: holdout attn — target is closing toward the JA tags
   band (0.52–0.56) from v2's 0.370 (reworded-holdout caveat applies; the
   grid + binding list is the tiebreaker, not the scalar alone).
3. **Binding list on renders** (the K3 eyeball's named misses): 흑백/
   그레이스케일 must render monochrome (JA already does — KO-specific gap),
   쌍둥이 must produce same-face pairs. Multi-seed if marginal.
4. **`desc_ko` earns its place** (Arm B only): s\* prompts move off the K0
   floor on the grid readout with tags_ko and JA unchanged vs Arm A.
   No movement, or movement paid for elsewhere → Arm A ships and `desc_ko`
   is recorded as tried-and-neutral (the JESC/STAIR verdict then extends to
   domain prose under this recipe, and the register retires).
5. **G5′ register drift — re-scoped** (the plan_ko G5 was unsatisfiable: 300
   hand-typed prompts cannot come from the glossary's own author —
   contamination — and cannot be commissioned from strangers):
   - *Pre-ship smoke* (~50 prompts): decontaminated sourcing — describe
     random gallery images in natural Korean without thinking in tags, plus
     a handful from people who never saw the glossary. Catches only a
     *catastrophic* register gap; that is all it is for.
   - *Ship as 베타* — gate 1 bounds the JA downside, and KO zero-shot was
     inert anyway; a labeled beta cannot regress a user experience that
     does not exist yet.
   - *Post-ship collection is the real G5*: the release-thread prompts
     (Arca register, by construction) become the drift eval + the next
     review round's target list. Fold into a ko3 round only if the gap is
     structural, not vocabulary.

## Phase R5 — ship

Rides plan_ko K4 / plan.md Phase 3 unchanged: winner pack renamed to the
shipped `cjk_vocab_pack_synthjako` slot (with the losing arm and v2 kept
on disk under their labels), sidecar + auto-discovery untouched,
the `cjk_vocab_pack.md` under `docs/methods/` gains the KO-베타 note + KB attribution
(Localsmile danbooru_KR_wiki), guidebook line via the `translator` agent.

## Risks

1. **r5 wording churn re-triggers the full loop** — by design; budget one
   re-stage + two retrains, nothing more. If review A produces mass vetoes
   of KB wordings, stop and re-examine the band split before staging.
2. **`desc_ko` loose alignment** under whole-sequence loss could smear
   rather than teach — that is what the R4-4 gate and the truncation
   fallback are for.
3. **KB provenance/staleness**: our copy is the `make download-danbooru-tags`
   fetch of 2026-06-19 (upstream "based on 2026.03.23"). Pin it for ko2 —
   a refresh mid-plan would silently change both extractions. Refresh, if
   wanted, happens *before* R1 or not at all.
4. **No license is stated upstream** for the Localsmile KB. Attribution in
   the docs regardless; if the pack ships to HF, note the source the way
   the JA tag-pair set is noted.

## Open question O1 — what does the KO corpus achieve *alone*? (user, 2026-09-01)

Every pack to date is joint: warm-started from `synthja_v4`, KO ≈ 28–31 % of
a batch beside the JA mass. So the measured KO numbers are KO-corpus quality
**times** JA scaffolding, and the two have never been separated. Three
things a KO-only arm would tell us that nothing else can:

1. **Does the JA mass help or tax KO?** If KO-only lands *above* the joint
   pack's tags_ko band, joint training is costing KO capacity (the shared
   `param=global` low-rank is being spent on JA); if *below*, the
   cross-script sharing (`global` does share across scripts — gate 2, v1)
   is doing real transfer work and the joint recipe is vindicated.
2. **Corpus attribution**: KO-only-from-scratch is the cleanest possible
   readout of the r5+desc corpus itself — no warm-start inheritance, no JA
   gradient interference.
3. **The fallback path's cost**: plan_ko gate 2's fail-path ("KO ships as a
   separate pack behind a language switch") has never been priced. If KO
   ever needs to escape the joint pack, this arm is its feasibility study.

Design (diagnostic arm, ~45 GPU-min, caches already staged):

- `--cache_dir cache_ko,cache_desc_ko`, `--train_registers
  tags_ko,tags_alt_ko,names_ko,names_synth_ko,desc_ko`, **no `--init_pack`**
  (from-scratch is the point; a `synthja_v4`-warm-started KO-only variant is
  a second, cheaper sub-arm if the first result is interesting), same
  recipe otherwise, `--out output/ckpt/cjk_vocab_pack_ko_only`,
  label `2c-ko-only`.
- Readout: holdout tags_ko/names_ko attn vs the joint pack's 0.524/0.936;
  KO recovery grid same-seed vs `20260901-1111`. **G1 EN bit-exact and the
  JA grids are NOT gates here** — this pack is expected to move JA rows and
  must never ship into the joint slot; it is a measurement, not a ship
  candidate.
- Interpretation guard: `ko_only > joint` on tags_ko does NOT by itself
  justify shipping a separate pack — the language-switch UX cost and the
  two-packs-can't-coexist constraint stand; it would instead argue for
  re-balancing the joint sampling (KO weight sweep upward) at ko3.

Status: **RUN 2026-09-01 midday (`2c-ko-only`) — the JA mass helps, joint
vindicated.** Holdout tags_ko 0.418 / tags_alt_ko 0.410 vs the joint pack's
0.524/0.532 — KO-only-from-scratch lands well below the JA band, *despite*
receiving ~3.3× more KO gradient (5.2 epochs vs the joint's ~1.6 at 30%
batch share), so the gap is transfer, not data volume. names_ko is
scaffolding-indifferent (0.945 ≈ 0.936); names_synth_ko 0.683 > joint's
0.622 (the joint dip reads as JA-competition pressure — noted, no action).
Renders agree: more broken rows than joint (q1/q3/s4/t4), 흑백 still binds
(r5's doing, not JA's). Separate-pack fallback is now priced: unattractive.
Per the interpretation guard, no ko3 rebalance case either — that needed
`ko_only > joint`. Envelope `bench/cjk_distill/results/20260901-1219-2c-ko-only`,
grid `bench/cjk_adapter/results/20260901-1258-ko-only-recovery-grid`
(+`contact_joint_vs_ko_only.png`). The warm-started sub-arm (`2c-ko-warm`)
was queued then withdrawn (user call: judge on the grid first) — command
shape is in the session log if the sequential-vs-cotraining decomposition
is ever wanted.

## Not planned

- Encoder/vocab/tokenizer changes, jamo, new adapter arms — unchanged from
  plan_ko.
- zh — still third, after ko2 settles.
- In-image hangul rendering (glyph line) — still deferred behind JA.
