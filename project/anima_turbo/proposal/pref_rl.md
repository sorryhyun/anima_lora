# Preference RL on the turbo student — train on what the user picks

Status: PROPOSAL (2026-11-13). Nothing built. One CPU probe is in § 1 (tags against the
gelcrawl selection). Open decision: the feature basis of the preference head (§ 3.2), which
P0 and P1 measure.

## TL;DR

Training the merged14 student with DM or FM losses makes Hub v3 worse (`../reports/warm_v3.md`).
What is left to optimise is whether the user likes the render, so the signal is the user's
own choice (user, 11-13):

1. render 16 images of one caption with merged14 + Hub v3;
2. the user picks 0–8 of them;
3. update the student;
4. take next the caption the user is expected to pick least from, and repeat.

The update is not a retraining on the picked images and not a policy gradient (§ 3.4). The
picks train a small **preference head**, and the student follows the head's gradient through
its own deterministic rollout. The head also ranks captions for step 4. Before any click,
gelcrawl's `retrieved/` against `selected/` gives 15 k labelled images of the same user's
taste (user, 11-13), which pretrain the head and seed the caption ranking.

## 1. Evidence

- **DM / FM training has no slack left on this student.** Hub v3 untrained on merged14 draws
  seeds as varied as the teacher (sameness 0.326 against 0.328); 250 iterations of the v1
  recipe raise it to 0.465, and neither `dm_first_step` nor the fake's seed is the cause
  (`../reports/warm_v3.md`). PSO (arXiv 2410.03190, Fig. 1) reports the same from the other
  side: a distilled SDXL tuned with the plain diffusion loss turns blurry.
- **Label budgets in the literature.** D3PO (arXiv 2311.13231), a policy gradient on human
  choices with no reward model, labels 1 000 images an epoch for 5 epochs (hands) or 10
  (safety). At 16 images a round that is 300–600 rounds. PSO's offline arm uses Pick-a-Pic's
  851 k pairs, and its online arm lets a reward model label.
- **gelcrawl as labels** (`~/gelcrawl`, read-only here): 88 artists have both folders,
  3 242 selected against 12 024 retrieved and not selected. The per-artist selection rate has
  a median of 0.17 (quartiles 0.09 / 0.38, range 0.02–0.96), so the label has to be read
  within an artist.
- **Tags carry part of that choice.** A logistic regression on the booru tags of the
  retrieved captions (2 359 tags, artist tags removed, 5 folds of held-out artists) ranks
  selected above rejected inside an artist with AUROC 0.631 (median 0.626; 71 of 76 artists
  above 0.5). Pooled across artists it is 0.668.
- **A selection classifier exists** in gelcrawl (`classify.py`: DINOv3 ViT-L CLS + MLP,
  per-artist CV). Its numbers are not on disk.

## 2. Scope

- **Policy**: merged14 + `anima_turbo_4step_v3`, 4 steps, cfg 1, Euler, shift 3. The student
  LoRA trains; the DiT stays frozen.
- **What a pick means** is left to the user: layout, anatomy, finish. The head learns
  whatever separates picked from not picked.
- Out of scope: retraining the 28-step teacher on survivors (the archived GRAFT loop,
  `_archive/graft/`), since distilling a moved teacher again is the step that fails.

## 3. Design

### 3.1 What the preference head does

A head maps a render (and, for the DiT basis, its caption) to a score. It has three uses:

| use | needs |
|---|---|
| **rank** the renders of a round (the read of whether it learned anything) | within-round discrimination |
| **reward gradient** for the student (§ 3.4) | differentiable from the latent, hard to fool |
| **caption ranking** for the next round (§ 3.5) | a score from the caption alone, or from a few cheap renders |

### 3.2 Feature basis (open)

| basis | for | against |
|---|---|---|
| **Tagger** (Anima Tagger probabilities or its backbone embedding) | readable (which tags lower the pick rate); shares a space with captions, so it scores a caption with no render; § 1 shows tags hold part of the gelcrawl choice | a tagger is trained to ignore what separates two renders of one caption: both get the same tags. Weak within a round by construction |
| **Embedding** (PE-Spatial 32 × 32 grid, already in `AtSim` and the REPA cache; DINOv3 as in `classify.py`) | general perceptual features with a spatial grid; code and caches exist | the gradient path needs the VAE decode and the tower (memory, as in `../../cjk_rl/proposal.md` § 3.7); pixel-space gradients are the easiest to fool |
| **DiT head** (frozen merged14 block features of the renoised latent, `TeacherFeatureDiscriminator`'s taps) | stays in latent space, no decode; caption-conditioned; the turbo loop already sends a generator gradient through this module for the GAN | gelcrawl images need VAE latents first (only the selected ones are cached); a discriminator is fooled unless its labels keep up, and human labels arrive slowly |

Expected split, to be measured rather than argued: tags for caption ranking, an embedding or
the DiT head for ranking renders and for the gradient. P0 ranks the bases on gelcrawl, P1 on
picks of renders, and the two can disagree (§ 3.3).

### 3.3 Two label sources and the gap between them

- **gelcrawl** labels real artwork. It records taste (content, composition, an artist's
  better pieces) and curation motives that are not quality (near-duplicates, too many of one
  character). It holds no generation defect: every image is drawn correctly.
- **Round picks** label 4-step renders of one caption. Content is nearly fixed, so the choice
  is mostly defects, layout and finish.

So a head fitted on gelcrawl may rank renders at chance. P1 reads that directly (zero-shot),
and gelcrawl is kept as pretraining only if it helps there. Its caption-level use (§ 3.5)
does not depend on that.

Selection rate differs 50-fold between artists, so every gelcrawl fit is within-artist:
held-out-artist folds, and a pairwise loss between a selected and a rejected image of the
same artist.

### 3.4 The update

**Reward gradient through the head.** Roll the student out with its Euler steps, score x̂₀
with the head, and backpropagate the score into the student LoRA. The rollout is the one the
student was distilled on; no SDE and no likelihood.

- **Steps trained**: step 0 only at first, steps 1–3 left as Hub v3. This is P-GRAFT's
  (arXiv 2510.02692) idea in a 4-step form: assign the final image's reward to an
  intermediate state and tune only the part of the sampler before it. Layout sits in step 0
  (`../reports/handoff.md`). A second arm opens steps 1–3 if the picks turn out to follow
  finish, not layout.
- **Anchor**: the step-0 anchor MSE to Hub v3's own step 0 (the loop's anchor term with v3
  as the target), and sameness across seeds read at every checkpoint.
- **Head refresh**: every N rounds of picks on the current student retrain the head, so a
  direction that fools it gets labelled.

**Not taken.**

- **GRAFT / P-GRAFT as written** (retrain on accepted samples). The student is a
  deterministic map from noise to image. Regressing it on its own accepted outputs has zero
  gradient, and a denoising loss on them is the blur of § 1. Two things carry over: the
  step restriction above, and that accepting k of 16 bounds a round's tilt at a density
  ratio of 16 / k, which is why picking several beats picking one.
- **Policy gradient** (D3PO, Flow-GRPO): 300–600 rounds by § 1, and the SDE sampler it needs
  is off the student's rollout.
- **PSO**: the pairwise objective fits a distilled student, but its evidence is 851 k
  offline pairs or a reward model's labels. It becomes an arm only if the reward gradient
  is fooled faster than the head can be refreshed.

### 3.5 A round

- 16 renders of one caption, seeds fresh each round, shown as one grid.
- The user picks **0–8**. Zero is allowed: a forced pick labels a bad render as good. The
  count itself is logged as the caption's pick rate.
- **Next caption**: the head's predicted pick count, lowest first, but not zero: a caption
  with nothing to pick has no contrast inside the round. Target 2–4 of 16. Half of the
  rounds take a random caption, which keeps the easy captions watched and corrects the
  predictor where it is wrong.
- Cold start for the predictor: the tag model of § 1, applied to the caption's tags.
- Log per round: caption, seeds, picks, the student's checkpoint, the head's scores.

### 3.6 Evaluation

The user is trainer and judge, so the read is blind. On held-out captions a grid mixes 8
renders of Hub v3 and 8 of the trained student at matched seeds, shuffled; the metric is the
share of picks that come from the trained student (0.5 = no gain). The round UI serves it
unchanged. Beside it: sameness and the `judge_hard` grids, as for every turbo arm.

## 4. Phases and gates

Criteria are fixed here, before any result. A gate passes when its evidence does (CLAUDE.md
§ Papers and gates).

**P0 — bases on gelcrawl (no clicks, no generator training)**

Held-out-artist folds as in § 1, within-artist AUROC, for each basis: booru tags (done,
0.631), tagger probabilities, tagger embedding, PE-Spatial pooled, DINOv3 CLS, and the DiT
head on a 3 000-image subsample (the VAE encode of the rejected images is the cost). Not a
gate: it ranks the bases for taste and yields the pretrained heads P1 tests.

**P1 — 24 rounds of picks on the untrained policy** (384 labelled renders, random training
captions)

- **Zero-shot**: each P0 head's within-round AUROC on the picks.
- **From picks**: each basis fitted on the picks alone, leave-rounds-out.
- **Pretrained then fitted**: gelcrawl head, then the picks.
- Rounds with zero picks are logged and left out of the AUROC.
- Gate for P2: at least one head ranks picked above unpicked on held-out rounds (sign test
  over rounds, p < 0.05). The basis is the one with the best leave-rounds-out AUROC; inside
  the fold spread the DiT head wins, since its gradient path has no decode.
- If no head passes, the picks are not predictable from these features at this count; the
  line stops or collects more rounds, and no student is trained.

**P2 — the reward-gradient arm**, step 0 only, anchor to v3, head refreshed on new rounds.

- Pass: in 20 blind mixed grids on held-out captions, the trained student's share of picks
  is above 0.5 (sign test over grids), sameness is not up against v3 untrained (paired over
  prompts), and the user reads the `judge_hard` grids as not worse.

**P3 — steps 1–3 opened**, if P2 passes and the user's picks are judged to follow finish.

## 5. Risks

- **Diversity.** Every preference method narrows the output, and sameness is already this
  line's open gap. The step restriction, the v3 anchor and the 16 / k bound work against it;
  sameness is in the P2 gate.
- **A fooled head.** The gradient finds what the head scores high, not what the user likes.
  The refresh and the blind grid are the checks; the DiT head inherits the GAN's failure
  modes.
- **gelcrawl is another task.** Its label mixes taste with curation motives and has no
  defects in it (§ 3.3). P1's zero-shot read decides how much of it is used.
- **Few labels.** 24 rounds may not fit any head. Then the cost is the rounds, and the log
  stays usable for any later estimator.
- **Self-judging.** The picks train the head and the user reads the result; only the blind
  grid separates a gain from familiarity.
