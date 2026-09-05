# Hypothesis — ext rows as isotropic addresses, semantics on the DiT side (2026-09-05)

**Status: hypothesis, not a proposal.** Written after blind sets s11/s12 so the
next session can review the ko/ja/zh plans (`plan.md`, `plan_ko3.md`,
`plan_zh*.md`) against it. Nothing here is adopted; the two closing
experiments at the end are unrun. Blind protocol and per-set tables:
`blind_s*.md`, digest `0905_blind_g0g1_readout.md`.

## The hypothesis (user's wording, 2026-09-05)

1. The T5 ext-vocab rows should be **isotropic**: mutually distinct, and
   distinct from the native T5 vocabulary.
2. Whether to OCR, and how CJK semantics get learned, is then the **DiT /
   LoRA's** business, not the table's.
3. **In-T5 translation** — making a CJK row take the representation of its
   EN counterpart — is not the direction.

## Mechanism this rests on

A T5 row is an address; the content lives in the DiT, keyed by that address.
An ext row has no DiT-side entry, so it can carry content only by (a) landing
near an existing EN address and borrowing, or (b) a LoRA learning content for
it. Distillation (`scripts/distill_cjk`, `param=global`, span loss against
the EN caption path) is exactly (a): it places 「少女」 near the girl address.
That is what makes the vocab-pack tag tier work zero-shot — and what makes
the same rows wrong for the unmask line, where the CJK span is OCR'd
**diegetic** text: 「爆発」 in a speech bubble does not mean an explosion is in
the image. The DiT reads the borrowed content, the image contradicts it, and
the LoRA learns to decouple or misbind — the spam/leakage seen in the unmask
grids. Rows that sit at no address (isotropic) or at one shared address
(collapse) give the DiT nothing to read, and the LoRA learns only "text
region here" from the 351 images. The "structured low-rank spread" noted in
`blind_s12_ISO1_vs_HOT.md` is the geometric shadow of the same thing: a pack
whose rows cluster near EN addresses has a low-PR spread.

## Evidence per claim

Pack geometry (`make_random_pack.py` stats; ratio = mean_norm/row_norm):

| arm | rows | ratio | PR | what it is |
|---|---|---:|---:|---|
| INIT | v2 zero-shot map (Qwen→T5, procrustes-mix, contextual chars) | 0.29 | 236 | no distill; C9 vs INIT 14–10 (s08, noise) |
| C9 | INIT + shared rank-256/diag/gain map, span loss | 0.23 | 18 | the deployed recipe (`synthjakozh1sym_r256`) |
| R | Gaussian, spectrum/norm/mean matched to C9 | 0.24 | ~18 | random content, same geometry |
| ROTATE | C9 under one random orthogonal rotation | 0.24 | 18 | structure kept, alignment broken |
| COLLIDE | random native T5 rows | 0.49 | native | real addresses, wrong content |
| HOT | isotropic Gaussian, norm ×5 | 0.004 | 1009 | |
| ISO1 | the same table at ×1 | 0.004 | 1009 | near-orthogonal rows at native scale |
| COLLAPSE | every row = trained mean direction | 1.0 | 1 | one shared address |
| P | no rows; `japanese text` presence tag only | — | — | |

Blind results (user-graded, sides and pair order shuffled; v1 = 8-row prompt
set at seeds 42/7/1234 or 1/2/3, v2 = 16 harder rows at seeds 1/2 or 3/4/5;
noise floor s02 = 15–9 on 24 pairs):

| set | pair | wins (tie) | read |
|---|---|---|---|
| s01 | C9 vs P | 19–5 | rows must exist; the presence tag alone loses |
| s03 / s10 | C9 vs R; ROTATE vs R | 7–17; 11–13 | content and alignment inert |
| s05 / s11 | C9 vs HOT (v1; v2) | 9–15; **6–19** (7) | HOT > C9, replicated on fresh prompts + seeds |
| s06 / s09 | C9 vs COLLIDE; HOT vs COLLIDE | 15–9; 15–9 | COLLIDE loses both (noise-level each), in the losing group |
| s08 | C9 vs INIT | 14–10 | distill itself changes nothing the grader sees |
| s11 | HOT vs COLLAPSE; C9 vs COLLAPSE | 11–12 (9); 9–15 (8) | COLLAPSE ≈ HOT, ≥ C9 |
| s12 | ISO1 vs HOT | 18–22 (8) | ×5 scale is not the lever; orthogonal-at-native-norm does the same |

Winners {HOT, ISO1, COLLAPSE} vs losers {C9, R, ROTATE, INIT, COLLIDE}: not
scale, not the common-direction ratio (0.004 and 1.0 both win, 0.23–0.49
lose), not content. What the losers share is that their rows sit on or near
EN addresses.

**Claim 1 — supported in half.** "Distinct from native T5" is what the
winner/loser split says. "Mutually distinct" is **not** shown: COLLAPSE (all
rows identical) ties ISO1/HOT. As of today it is a design choice — costs
nothing, and keeps rows addressable if the DiT side is to learn per-row
content later — not a finding. ISO1 > C9 is transitive only (via HOT);
transitivity already failed once (s03/s04).

**Claim 2 — the untested premise.** The unmask sets show a LoRA learns
"text region present" for isotropic addresses. Nobody has shown a LoRA
learns **tag semantics** (「猫耳」 → cat ears) for isotropic addresses from an
ordinary dataset with CJK captions. This is the load-bearing experiment.

**Claim 3 — supported, with one live counter-example.** The name tier never
rendered (plan_ko3 closed: metrics restorable, rare kanji names never
compose); U5 symbol register failed; the 2026-09-01 goal reframe already says
MT/substitution is not the direction. But the **tag tier works** and is
deployed (`sorryhyun/anima-vocab-pack-ja` synthja_v4 + AnimaVocabPackLoader).
Adopting claim 3 gives up that zero-shot behaviour: a Japanese-tag user would
need a LoRA trained with CJK captions instead of a loader.

## What the hypothesis gives up (state it so it is not re-proposed by accident)

- **Zero-shot tag tier.** Isotropic rows carry no EN content; the vocab pack
  as shipped stops doing anything until a LoRA supplies content.
- **Long-tail generalisation.** A shared map moved unseen rows +0.18 cos
  (plan_zh2 U4). Isotropic rows generalise to unseen tags by exactly zero:
  only rows visited in training acquire meaning.
- **One pack for two roles.** Today the same `synthjakozh1sym_r256` serves
  user-typed tags and OCR'd training spans, and the two want opposite things.
  Under the hypothesis the OCR span route is a separate, content-free block
  regardless of what the tag route does.

## What it would buy

- A pack that is a seed + id mapping, regenerated at load (no 285 MB of rows
  to ship or version).
- Unmask training that is clean by construction rather than by choosing a
  lucky pack (C9 vs P is the only pair the presence tag lost; every
  content-free pack beat C9).

## Closing experiments (unrun)

1. **ISO1 vs C9 direct** — render-only, C9 at seeds 3/4/5 on the v2 prompts
   (fresh for the grader), 48 pairs. Removes the transitivity dependence.
   `regrid_set.py --set s13_ISO1_vs_C9 --arms ISO1 C9 --seeds 3 4 5
   --eval_dir output/tests/cjk_unmask_eval3 --prompts …_v2.txt --push`.
2. **Semantics on the DiT side** — the ordinary LoRA dataset with
   glossary-translated JA-tag caption variants (`datasets/tag_glossary.py`,
   caption-variants machinery), TE cached once with ISO1 rows and once with
   the C9 pack, one LoRA each, JA-tag prompts rendered and scored by tagger
   adherence recall (`probes/unmask_grid_judge.py`) plus a blind set. Under
   the hypothesis ISO1+LoRA learns the visited tags; C9+LoRA additionally
   keeps zero-shot on unvisited ones. The gap on unvisited tags is the price
   of claim 1 measured, not argued.
3. Only if 2 favours ISO1: decide COLLAPSE vs ISO1 for the OCR route (s11
   cannot separate them; the simpler shared-row form is fine unless per-row
   content is wanted later).

## Pointers

- Arms and packs: `probes/g0_chain.py` (ARMS), `probes/make_random_pack.py`
  (modes matched/hot/cold/collapse/collide/rotate), configs under
  `configs/gui-methods/custom/cjk_unmask_*.toml`.
- Init recipe: `bench/cjk_adapter/build_ext.py` (v2: `--map procrustes-mix
  --char-init contextual`); student: `scripts/distill_cjk/ext_table.py`.
- Prior verdicts this leans on: `0904_text_bind_probe.md` (presence tag is
  the address, OCR rows inert), `0904_u5_symbol_register.md`, `findings.md`
  §8 (plan_ko3 name tier), plan_zh2 U4.
